#!/usr/bin/env python3
"""Rent GPU machines on Vast.ai, check each one in its first minutes, and keep a ledger of how every machine did.

Every rental goes in one ledger, data/ledger.jsonl beside this script, that all sessions on this workstation share:
what was offered, what the check measured, how a machine failed, when it went back, and what the work noted of it.
When .env names an R2 bucket, shared_ledger.py shares the ledger with the sessions on every other machine too.
A search keeps to machines that Vast.ai has verified, unless --unverified also lets in those it has not verified yet,
leaves out machines that failed once, those that the instances --avoid names ran on, and offers whose power limit is
below the default that checked machines with the same GPU reported, and puts hosts that failed before after the others. A machine that has not begun to load its image
after five minutes goes back. A rented machine runs machine_check.py at once and goes back if its GPUs hold memory, if
its power is capped below its default, if its GPUs multiply a fixed pair of matrices wrongly or differently twice, if
it is slower than the median of the checked machines with the same GPU, if it downloads slower than --min-download, or
if its traffic leaves from another country than its listing names.
Machines with different GPUs are never compared. An image without python3 gets a check by bash alone, of the GPUs as
nvidia-smi reports them and of the network, but not of the GPUs' speeds.

A rented machine goes back when the work returns it, or else at its deadline. Each workstation runs one watcher,
`job-runner vast watch`, which job-runner's install.sh starts as a systemd user service that starts with the
workstation and comes back if it stops. Round after round the watcher sends back each machine rented on its
workstation whose deadline has passed, or whose return the work asked for but Vast.ai did not complete, whatever has
become of the work. rent rents nothing while the watcher does not run with the key that rents, since a watcher sees
the machines of its own key's account alone. return records that the machine is to go back before it destroys it, so
that a return that fails is the watcher's to finish.

rent, list, ssh-config, return and watch need the Vast.ai API key in VAST_API_KEY, which job-runner takes from its
.env, one NAME=VALUE a line, unless the environment already gives it. rent and ssh-config need the caller's SSH key as
the private key file that VAST_SSH_KEY_FILE names. rent gives its public half to the machine it rents, and only to that
machine, checks the machine over ssh with that key, the way the work will reach it, and writes how to reach the
machine it keeps to data/ssh/INSTANCE.config, printing that path with the machine's deadline: the work reaches the
machine as `ssh -F FILE -i KEY vast-INSTANCE`, and no address shows. ssh-config writes it again. Only --data, for tests
and trials, moves the ledger, so that every session keeps to the same one; a ledger that --data names has its own
watcher, started with the same --data. git ignores .env and data/. Nothing here prints an address or a key.
job-runner's install.sh starts the watcher; `job-runner vast` runs this script.

Usage:
  job-runner vast offers GPU [--count N] [--max-price DOLLARS] [--countries CC,CC] [--disk GB] [--query JSON]
              [--unverified] [--avoid INSTANCE,INSTANCE] [--show N]
  job-runner vast rent GPU --image IMAGE --label LABEL --hours HOURS [the options of offers] [--tries N]
              [--failure-budget DOLLARS] [--min-download MBPS] [--env JSON] [--onstart COMMAND]
  job-runner vast ssh-config INSTANCE
  job-runner vast extend INSTANCE HOURS
  job-runner vast note INSTANCE TEXT [--values JSON]
  job-runner vast reject INSTANCE DETAIL
  job-runner vast return (INSTANCE | --label LABEL)
  job-runner vast list [--label LABEL]
  job-runner vast ledger [GPU]
  job-runner vast watch
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import machine_check
import shared_ledger

HERE = Path(__file__).resolve().parent
# Vast.ai's API, whose list of instances has moved to its second version.
API = os.environ.get("VAST_URL", "https://console.vast.ai").rstrip("/") + "/api"
PROVIDER = "vast"
# A GPU that holds this much memory before the work starts is running someone else's.
BUSY_MIB = 1024
# A machine slower than this share of the median of the checked machines with its GPU goes back.
SLOW_SHARE = 0.9
# The fewest checked machines with the same GPU whose median a new machine is held to.
FEWEST_TO_COMPARE = 2
SPEEDS = ("matmul_tflops", "memory_gbps")
# How far a GPU's products of the check's fixed matrices may stray from the CPU's, relative to the largest entry, and
# how far two runs may differ. A healthy GPU strays far less, rounding included; one that strays more, as a 5090 that
# gave 1.2e12 and then 288 for 10.38 did, gives any work wrong numbers.
MOST_STRAY = {"float32": 1e-2, "bfloat16": 5e-2}
MOST_REPEAT_GAP = 1e-3
# A machine that downloads slower than this many MB/s from both of the check's sources goes back, unless rent is told
# another floor: such a machine is broken for any work, whatever its listing says of its line.
MIN_DOWNLOAD_MBPS = 10
START_SECONDS = 20 * 60
# A machine whose status is still empty this long after it was rented has not begun to load the image, and will not.
LOAD_SECONDS = 5 * 60
SSH_SECONDS = 4 * 60
CHECK_SECONDS = 5 * 60
# The watcher looks at the ledger this often, and rent takes a watcher that has not looked for three rounds as stopped.
WATCH_SECONDS = 60
WATCH_STALE_SECONDS = 3 * WATCH_SECONDS
RETRYABLE = {429, 502, 503, 504}
# Words of a container's status that blame the image, not the machine.
IMAGE_FAULTS = ("manifest unknown", "pull access denied", "repository does not exist", "invalid reference format")
# The commands that call Vast.ai with the key.
KEYED = {"rent", "list", "ssh-config", "return", "watch"}


class VastError(Exception):
    def __init__(self, code, message):
        super().__init__(f"Vast.ai answered {code}: {message}")
        self.code = code


class ImageFault(Exception):
    pass


def now():
    return datetime.now(timezone.utc)


def stamp(moment=None):
    return (moment or now()).isoformat(timespec="seconds")


# The folder of the ledger, unless --data names another.
folder = HERE / "data"


def home():
    return folder


def key_file():
    """The private key file that VAST_SSH_KEY_FILE names, or None when it names none."""
    name = os.environ.get("VAST_SSH_KEY_FILE")
    return Path(name) if name else None


def public_key():
    """The public half of the key in the file that VAST_SSH_KEY_FILE names, which rented machines are given and ssh
    uses alone."""
    file = key_file()
    if file is None:
        raise SystemExit("rent and ssh-config need VAST_SSH_KEY_FILE to name the private key file of the caller's key")
    try:
        derived = subprocess.run(["ssh-keygen", "-y", "-f", str(file)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        derived = None
    words = derived.stdout.split() if derived is not None and derived.returncode == 0 else []
    if len(words) < 2:
        raise SystemExit("VAST_SSH_KEY_FILE must name a private key file without a passphrase")
    return " ".join(words[:2])


def hide_addresses(text):
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", text)
    text = re.sub(r"\b(?:[0-9a-fA-F]{1,4}:){4,7}[0-9a-fA-F]{1,4}\b", "<ip>", text)
    return re.sub(r"\b[\w-]+\.vast\.ai\b", "<host>", text)


# The ledger


def record(event, **fields):
    entry = {"at": stamp(), "event": event, "id": uuid.uuid4().hex, **fields}
    home().mkdir(parents=True, exist_ok=True)
    shared_ledger.append(home(), entry)
    # The ledger beside this script is shared through R2 when .env names a bucket; one that --data names, for tests and
    # trials, never is. An event that cannot be uploaded waits for the next try, and the work goes on.
    if home() == HERE / "data" and (config := shared_ledger.settings()):
        shared_ledger.send(home(), entry, config)
    return entry


def entries():
    """The ledger's events, each once, with those that other machines shared added first when it is shared."""
    if home() == HERE / "data" and (config := shared_ledger.settings()):
        shared_ledger.pull(home(), config)
    try:
        lines = (home() / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    found, seen = [], set()
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:  # a line that a writer left unfinished
            continue
        if entry.get("id") is not None:
            if entry["id"] in seen:  # pulled twice, by two processes at once
                continue
            seen.add(entry["id"])
        found.append(entry)
    return found


def avoided(ledger):
    """The machines that failed once: they did not start, could not be reached or failed their check."""
    return {(entry["provider"], entry["machine"]) for entry in ledger if entry["event"] == "unusable"}


def failing_hosts(ledger):
    return Counter((entry["provider"], entry["host"]) for entry in ledger
                   if entry["event"] == "unusable" and entry.get("host") is not None)


def typical(ledger, gpu, version):
    """The median of each speed over the machines with this GPU that passed a check of this version, and their number."""
    checks = [entry["results"] for entry in ledger if entry["event"] == "checked" and entry.get("kept")
              and entry.get("gpu") == gpu and entry["results"].get("version") == version]
    medians = {}
    for name in SPEEDS:
        values = [check[name] for check in checks if check.get(name)]
        if values:
            medians[name] = statistics.median(values)
    return medians, len(checks)


def lowest_default_power(ledger, gpu):
    """The lowest default power limit that a checked machine with this GPU reported, or None before the first check.
    Makers give one GPU different defaults, an RTX 5090 575 W or 600 W, and an offer names only its limit, so a
    limit is below its machine's default for sure only when it is below the lowest."""
    values = [row["power_default_w"] for entry in ledger if entry["event"] == "checked" and entry.get("gpu") == gpu
              for row in entry["results"].get("gpus") or [] if row.get("power_default_w")]
    return min(values) if values else None


def choose(offers, ledger, avoid=()):
    """The offers to try, in order: without the machines that failed once or that the instances in avoid ran on, nor
    offers whose power limit is below the lowest default of checked machines with the same GPU, which the check then
    holds to its own default; hosts that failed before after the rest; each group cheapest first."""
    left_out, failures, defaults, kept = avoided(ledger), failing_hosts(ledger), {}, []
    held = rentals(ledger)
    left_out |= {(PROVIDER, held[instance]["machine"]) for instance in avoid if instance in held}
    for offer in offers:
        if (PROVIDER, offer["machine_id"]) in left_out:
            continue
        gpu = offer["gpu_name"]
        if gpu not in defaults:
            defaults[gpu] = lowest_default_power(ledger, gpu)
        limit = offer.get("gpu_max_power")
        if defaults[gpu] and limit and limit < defaults[gpu] - 1:
            continue
        kept.append(offer)
    return sorted(kept, key=lambda offer: (failures[(PROVIDER, offer["host_id"])] > 0, offer["dph_total"]))


def verdict(results, ledger, gpu, min_download_mbps=0, listed_country=None):
    """Why a checked machine should go back, if at all: GPUs that hold memory, power capped below its default, products
    that stray from the CPU's or differ between two runs, speeds below SLOW_SHARE of the median of checked machines with
    the same GPU, a download slower than the work needs, or traffic that leaves from another country than the listing
    names. What a check could not measure blames nothing."""
    reasons = []
    rows = results.get("gpus") or []
    if not rows:
        reasons.append("no_gpu: nvidia-smi reported no GPU")
    for index, row in enumerate(rows):
        if (row.get("memory_used_mib") or 0) >= BUSY_MIB:
            reasons.append(f"gpu_busy: GPU {index} holds {row['memory_used_mib']:.0f} MiB")
        limit, default = row.get("power_limit_w"), row.get("power_default_w")
        if limit and default and limit < default - 1:
            reasons.append(f"power_capped: GPU {index} is limited to {limit:.0f} of its default {default:.0f} W")
    for dtype, most in MOST_STRAY.items():
        error, gap = results.get(f"{dtype}_error"), results.get(f"{dtype}_repeat_gap")
        if error is not None and error > most:
            reasons.append(f"wrong_results: {dtype} products stray {error:.2g} from the CPU's")
        if gap is not None and gap > MOST_REPEAT_GAP:
            reasons.append(f"unrepeatable: two runs of the same {dtype} product differ by {gap:.2g}")
    medians, count = typical(ledger, gpu, results.get("version"))
    if count >= FEWEST_TO_COMPARE:
        for name, median in medians.items():
            value = results.get(name)
            if value is not None and value < SLOW_SHARE * median:
                reasons.append(f"slow: {name} {value:.0f} against a median of {median:.0f} over {count} machines")
    download = results.get("download_mbps")
    if min_download_mbps and download is not None and download < min_download_mbps:
        reasons.append(f"slow_download: {download:.1f} MB/s against the {min_download_mbps:.0f} needed")
    egress = results.get("egress_country")
    if listed_country and egress and egress != listed_country:
        reasons.append(f"egress_elsewhere: listed in {listed_country}, but its traffic leaves from {egress}")
    return reasons


def rentals(ledger):
    """Each rental in the ledger by instance: the entry that rented it, with its latest deadline, whether the work
    asked for it to go back, and whether it went back."""
    found = {}
    for entry in ledger:
        instance = entry.get("instance")
        if entry.get("provider") != PROVIDER or instance is None:
            continue
        if entry["event"] == "rented":
            found[instance] = {**entry, "released": False, "returned": False}
        elif instance in found and entry["event"] == "extended":
            found[instance]["deadline"] = entry["deadline"]
        elif instance in found and entry["event"] == "released":
            found[instance]["released"] = True
        elif instance in found and entry["event"] == "returned":
            found[instance]["returned"] = True
    return found


def rental(ledger, instance):
    """The ledger's record of one rental, as rentals gives it; None if it has none."""
    return rentals(ledger).get(instance)


# Vast.ai


def call(method, path, body=None, query=None, version="v0"):
    url = f"{API}/{version}{path}"
    if query:
        url += "?" + urllib.parse.urlencode({key: value if isinstance(value, str) else json.dumps(value)
                                             for key, value in query.items()})
    headers = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "job-runner"}
    if os.environ.get("VAST_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VAST_API_KEY"]
    data = None if body is None else json.dumps(body).encode()
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, data=data, method=method, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE or attempt == 3:
                raise VastError(error.code, hide_addresses(error.read().decode(errors="replace"))[:300]) from None
        except urllib.error.URLError:
            if attempt == 3:
                raise
        time.sleep(5 * 2**attempt)


def search(gpu, count=1, disk=60, query=None, limit=200, unverified=False):
    """The offers, cheapest first: of machines that Vast.ai has verified, as its own command line searches by default,
    or with `unverified` also of those it has not verified yet. Machines that lost their verification never."""
    filters = {"gpu_name": {"eq": gpu}, "num_gpus": {"eq": count}, "rentable": {"eq": True}, "rented": {"eq": False},
               "external": {"eq": False}, "disk_space": {"gte": disk}}
    if not unverified:
        filters["verified"] = {"eq": True}
    filters.update(query or {})
    found = call("GET", "/bundles/", query={"q": {**filters, "order": [["dph_total", "asc"]], "type": "on-demand",
                                                  "limit": limit, "allocated_storage": disk}})
    # Hosts that lost their verification have not started a container for us, and unverified ones include machines
    # whose listing misstates where their traffic leaves from.
    allowed = ("verified", "unverified") if unverified else ("verified",)
    return [offer for offer in found["offers"] if offer.get("verification") in allowed]


def create(offer, image, disk, label, env=None, onstart=None):
    body = {"client_id": "me", "image": image, "env": env or {}, "price": None, "disk": disk, "label": label,
            "extra": None, "onstart": onstart, "image_login": None, "python_utf8": False, "lang_utf8": False,
            "use_jupyter_lab": False, "jupyter_dir": None, "force": False, "cancel_unavail": True,
            "template_hash_id": None, "user": None, "runtype": "ssh_direc ssh_proxy"}
    return call("PUT", f"/asks/{offer['id']}/", body)["new_contract"]


def show(instance):
    """The instance as Vast.ai lists it, or None once it is gone."""
    try:
        return call("GET", f"/instances/{instance}/", query={"owner": "me"})["instances"]
    except VastError as error:
        if error.code == 404:
            return None
        raise


def destroy(instance):
    return call("DELETE", f"/instances/{instance}/", {})


def attach(instance, public_key):
    return call("POST", f"/instances/{instance}/ssh/", {"ssh_key": public_key})


# SSH


def routes(row):
    """Where the instance's SSH answers: the host's own port first, as `vastai ssh-url` prefers, then Vast.ai's proxy."""
    mapped = (row.get("ports") or {}).get("22/tcp") or []
    found = [(row["public_ipaddr"], mapped[0]["HostPort"])] if mapped and row.get("public_ipaddr") else []
    if row.get("ssh_host") and row.get("ssh_port"):
        found.append((row["ssh_host"], row["ssh_port"]))
    return found


def ssh_base(instance, host, port):
    """ssh to the instance by one route with the caller's key file alone, without this workstation's settings, other
    keys or agent."""
    known = home() / "known_hosts"
    known.mkdir(parents=True, exist_ok=True)
    return ["ssh", "-F", "/dev/null", "-o", f"IdentityFile={key_file()}", "-o", "IdentitiesOnly=yes",
            "-o", "IdentityAgent=none", "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={known / str(instance)}", f"root@{host}"]


def reach(instance):
    """The first route, as host and port, by which the instance accepts ssh; None if none does."""
    for host, port in routes(show(instance) or {}):
        try:
            if subprocess.run([*ssh_base(instance, host, port), "true"], capture_output=True, timeout=90).returncode == 0:
                return host, port
        except subprocess.TimeoutExpired:
            continue
    return None


def connection_file(instance):
    """Where the instance's ssh configuration goes, for the host vast-INSTANCE."""
    return home() / "ssh" / f"{instance}.config"


def write_connection(instance, host, port):
    """Write how to reach the instance as an ssh configuration, so that the work reaches it with the caller's key as
    `ssh -F FILE vast-INSTANCE` and its address shows in no command line or log; return the file."""
    path = connection_file(instance)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = home() / "known_hosts"
    known.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Host vast-{instance}\n    HostName {host}\n    Port {port}\n    User root\n"
                    f"    StrictHostKeyChecking accept-new\n    UserKnownHostsFile {known / str(instance)}\n"
                    "    ServerAliveInterval 30\n")
    return path


def wait_until_running(instance):
    """None once the container runs, or why it did not."""
    began, row = time.monotonic(), {}
    while time.monotonic() - began < START_SECONDS:
        time.sleep(30)
        try:
            row = show(instance) or {}
        except (VastError, OSError):
            continue
        if row.get("actual_status") == "running":
            return None
        message = (row.get("status_msg") or "").lower()
        if any(fault in message for fault in IMAGE_FAULTS):
            raise ImageFault(hide_addresses(row.get("status_msg") or ""))
        if row.get("intended_status") == "stopped" or "error" in message:
            break
        if not row.get("actual_status") and time.monotonic() - began >= LOAD_SECONDS:
            return f"the machine had not begun to load the image after {LOAD_SECONDS // 60} minutes"
    return f"{row.get('actual_status')}: {hide_addresses(row.get('status_msg') or '')}"


def wait_for_ssh(instance):
    """The first route, as host and port, by which the instance accepts ssh within SSH_SECONDS; None if none does."""
    deadline = time.monotonic() + SSH_SECONDS
    while time.monotonic() < deadline:
        try:
            route = reach(instance)
        except (VastError, OSError):
            route = None
        if route:
            return route
        time.sleep(20)
    return None


# The check of an image without python3, which may lack curl and wget too: nvidia-smi, then, with bash alone over plain
# HTTP, a download from Cloudflare and the country that Cloudflare sees the machine's traffic leave from.
SHELL_CHECK = r"""
nvidia-smi --query-gpu=QUERY --format=csv,noheader,nounits | sed 's/^/gpu: /'
get() {
  exec 3<>"/dev/tcp/$1/80" &&
    printf 'GET %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: machine-check\r\nConnection: close\r\n\r\n' "$2" "$1" >&3 &&
    timeout "$3" cat <&3
  exec 3<&-
}
began=$(date +%s.%N); bytes=$(get speed.cloudflare.com '/__down?bytes=99999999' 15 | wc -c); ended=$(date +%s.%N)
echo "download: $bytes $began $ended"
get www.cloudflare.com /cdn-cgi/trace 15 | grep '^loc='
exit 0
"""


def read_shell_check(text):
    """SHELL_CHECK's output as a check's results, under a version of their own that no full check is held to."""
    found = {"version": "shell", "limited": "the image has no python3, so the GPUs' speeds were not measured"}
    rows = [line[len("gpu: "):] for line in text.splitlines() if line.startswith("gpu: ")]
    found["gpus"] = machine_check.parse_gpus("\n".join(rows))
    for line in text.splitlines():
        if line.startswith("download: "):
            try:
                size, began, ended = line.split()[1:4]
                found["download_mbps"] = round(int(size) / max(float(ended) - float(began), 1e-3) / 1e6, 1)
                found["downloads"] = {"cloudflare": found["download_mbps"]}
            except ValueError:
                pass
        elif line.startswith("loc="):
            found["egress_country"] = line[len("loc="):].strip() or None
    return found


def check(base):
    """machine_check.py's results on the machine, or None and why it did not run. An image without python3 gets
    SHELL_CHECK instead, which checks the GPUs as nvidia-smi reports them and the network, but not the GPUs' speeds."""
    try:
        probe = subprocess.run([*base, "command -v python3"], capture_output=True, text=True, timeout=90)
        if probe.returncode == 1:
            done = subprocess.run([*base, "bash -s"], input=SHELL_CHECK.replace("QUERY", ",".join(machine_check.QUERY)),
                                  capture_output=True, text=True, timeout=CHECK_SECONDS)
            if done.returncode:
                return None, f"the shell check failed with {done.returncode}: {hide_addresses(done.stderr)[-300:]}"
            return read_shell_check(done.stdout), None
        done = subprocess.run([*base, "python3 -"], input=(HERE / "machine_check.py").read_text(), capture_output=True,
                              text=True, timeout=CHECK_SECONDS)
    except subprocess.TimeoutExpired:
        return None, f"the check ran longer than {CHECK_SECONDS} seconds"
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode or not lines:
        return None, f"the check failed with {done.returncode}: {hide_addresses(done.stderr)[-300:]}"
    return json.loads(lines[-1]), None


# Rentals


def identity(offer):
    return {"provider": PROVIDER, "machine": offer["machine_id"], "host": offer["host_id"], "gpu": offer["gpu_name"],
            "count": offer["num_gpus"], "where": offer.get("geolocation")}


@contextlib.contextmanager
def return_lock():
    """Hold this workstation's lock on sending machines back, so that the watcher and a command never send back the
    same machine at once."""
    home().mkdir(parents=True, exist_ok=True)
    with open(home() / "return.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def send_back(instance, by):
    """Send the machine back: record that it is to go back, so that the watcher finishes the job if Vast.ai does not
    destroy it now, destroy it, and record that it went back, with its hours and their cost at its price. None if it
    went back before; raise if it is still there."""
    with return_lock():
        held = rental(entries(), instance)
        if held["returned"]:
            return None
        if not held["released"]:
            record("released", provider=PROVIDER, instance=instance, by=by)
        try:
            answer = destroy(instance)
            destroyed = not isinstance(answer, dict) or answer.get("success", True) is not False
        except VastError:
            destroyed = False
        if not destroyed and show(instance) is not None:
            raise VastError(None, f"instance {instance} could not be destroyed")
        hours = (now() - datetime.fromisoformat(held["at"])).total_seconds() / 3600
        connection_file(instance).unlink(missing_ok=True)
        return record("returned", provider=PROVIDER, instance=instance, by=by, hours=round(hours, 3),
                      dollars=round(hours * (held.get("price") or 0), 3))


def workstation():
    """This workstation's name, which each rental records so that the watcher here looks after it."""
    return socket.gethostname()


def key_mark():
    """What the watcher marks its rounds with for the Vast.ai key it watches with: a digest, which tells keys apart and
    gives none away."""
    return hashlib.sha256(os.environ.get("VAST_API_KEY", "").encode()).hexdigest()[:16]


def watching():
    """Whether this workstation's watcher watches the machines that the key rents: it is still there and finished a
    round with the key within WATCH_STALE_SECONDS."""
    try:
        beat = json.loads((home() / "watch.json").read_text())
        os.kill(beat["pid"], 0)
        age = (now() - datetime.fromisoformat(beat["at"])).total_seconds()
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return beat.get("workstation") == workstation() and beat.get("key") == key_mark() and age <= WATCH_STALE_SECONDS


def watch_round():
    """Send back each machine rented on this workstation whose deadline has passed or whose return the work asked for,
    then write the heartbeat that tells rent the watcher runs."""
    here = workstation()
    for instance, held in rentals(entries()).items():
        if held.get("workstation") != here or held["returned"]:
            continue
        if held["released"] or now() >= datetime.fromisoformat(held["deadline"]):
            try:
                send_back(instance, by="watch")
            except (VastError, OSError) as error:
                print(stamp(), f"instance {instance} could not go back yet:", hide_addresses(str(error)), flush=True)
    path = home() / "watch.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"workstation": here, "pid": os.getpid(), "at": stamp(), "key": key_mark()}) + "\n")
    temporary.replace(path)


def watch(args):
    """Run watch_round every WATCH_SECONDS for as long as this runs; a round that fails is tried again the next time."""
    while True:
        try:
            watch_round()
        except Exception as error:  # noqa: BLE001 - the watcher outlives any one round
            print(stamp(), "the round failed:", type(error).__name__, hide_addresses(str(error)), flush=True)
        time.sleep(WATCH_SECONDS)


def country(offer):
    """The country code that ends the offer's listed place, such as US of "California, US"; None without a place."""
    return (offer.get("geolocation") or "").rsplit(", ", 1)[-1] or None


def matches(offer, max_price=None, countries=None):
    return (max_price is None or offer["dph_total"] <= max_price) and (not countries or country(offer) in countries)


def summary(results):
    names = ("matmul_tflops", "memory_gbps", "sm_clock_under_load_mhz", "power_under_load_w", "host_to_gpu_gbps",
             "download_mbps", "disk_write_mbps")
    found = {name: round(results[name], 1) for name in names if isinstance(results.get(name), (int, float))}
    found["power_limit_w"] = [row.get("power_limit_w") for row in results.get("gpus") or []]
    for name in ("egress_country", "limited"):
        if results.get(name):
            found[name] = results[name]
    return found


def rent_one(offer, args, key):
    """Rent the offer and check its machine, giving it `key`: "kept" with the instance, the check's results, the file
    that says how to reach it and its deadline if it passed, "failed" with what it cost if it went back, or "taken" if
    someone rented the offer first."""
    try:
        instance = create(offer, args.image, args.disk, args.label, args.env, args.onstart)
    except VastError as error:
        if error.code in (401, 403):
            raise
        return "taken", None
    deadline = stamp(now() + timedelta(hours=args.hours))
    record("rented", **identity(offer), instance=instance, offer=offer["id"], price=offer["dph_total"],
           power_limit_w=offer.get("gpu_max_power"), reliability=offer.get("reliability2"),
           verification=offer.get("verification"), image=args.image, label=args.label, deadline=deadline,
           workstation=workstation())
    try:
        stage, problem = "start", wait_until_running(instance)
        if problem is None:
            attach(instance, key)
            route = wait_for_ssh(instance)
            if route is None:
                # A key file that went away since rent began is the caller's doing, not the machine's fault.
                public_key()
            stage, problem = ("ssh", "no route accepted ssh") if route is None else ("check", None)
        results = None
        if problem is None:
            results, failure = check(ssh_base(instance, *route))
            if results is None:
                # A check that does not run says more of the image than of the machine.
                raise RuntimeError(failure)
            reasons = verdict(results, entries(), offer["gpu_name"], args.min_download, country(offer))
            record("checked", **identity(offer), instance=instance, results=results, kept=not reasons, reasons=reasons)
            problem = "; ".join(reasons) or None
    except BaseException:
        send_back(instance, by="error")
        raise
    if problem is not None:
        record("unusable", **identity(offer), instance=instance, stage=stage, detail=problem[-500:])
        returned = send_back(instance, by=stage)
        print(json.dumps({"unusable": instance, "machine": offer["machine_id"], "stage": stage, "detail": problem}),
              flush=True)
        return "failed", returned["dollars"]
    return "kept", (instance, results, write_connection(instance, *route), deadline)


# Commands


def list_offers(args):
    ledger = entries()
    offers = [offer for offer in choose(search(args.gpu, args.count, args.disk, args.query,
                                               unverified=args.unverified), ledger, args.avoid)
              if matches(offer, args.max_price, args.countries)]
    failures = failing_hosts(ledger)
    for offer in offers[:args.show]:
        passed = sum(1 for entry in ledger if entry["event"] == "checked" and entry.get("kept")
                     and entry.get("machine") == offer["machine_id"])
        print(json.dumps({"offer": offer["id"], "price": round(offer["dph_total"], 3), "where": offer.get("geolocation"),
                          "host": offer["host_id"], "machine": offer["machine_id"],
                          "power_limit_w": offer.get("gpu_max_power"), "reliability": offer.get("reliability2"),
                          "verification": offer.get("verification"), "passed_checks": passed,
                          "host_failures": failures[(PROVIDER, offer["host_id"])]}, ensure_ascii=False))
    return 0


def rent(args):
    """Rent and check offers in rent's order until a machine passes, as many machines may fail as --tries allows, or
    the failed ones have cost more than --failure-budget dollars in all."""
    # Before renting: a machine nothing would send back at its deadline must not be rented at all.
    if not watching():
        raise SystemExit("rent needs this workstation's watcher, `job-runner vast watch`, to run with the key that "
                         "rents; job-runner's install.sh starts it with the key in .env")
    # Without the caller's key every machine would fail at SSH, and the ledger would blame them.
    key, failed, spent = public_key(), 0, 0.0
    offers = [offer for offer in choose(search(args.gpu, args.count, args.disk, args.query,
                                               unverified=args.unverified), entries(), args.avoid)
              if matches(offer, args.max_price, args.countries)]
    for offer in offers:
        if failed == args.tries or (args.failure_budget is not None and spent > args.failure_budget):
            break
        outcome, rented = rent_one(offer, args, key)
        if outcome == "kept":
            instance, results, connection, deadline = rented
            print(json.dumps({"rented": instance, "price": round(offer["dph_total"], 3),
                              "where": offer.get("geolocation"), "machine": offer["machine_id"],
                              "host": offer["host_id"], "check": summary(results), "ssh_config": str(connection),
                              "ssh_host": f"vast-{instance}", "deadline": deadline, "failed": failed,
                              "failed_dollars": round(spent, 3)}, ensure_ascii=False))
            return 0
        if outcome == "failed":
            failed, spent = failed + 1, spent + rented
    print(json.dumps({"rented": None, "offers": len(offers), "failed": failed, "failed_dollars": round(spent, 3)}))
    return 1


def ssh_config(args):
    """Write the held instance's ssh configuration again, by the route that answers now, and print where it is."""
    held_or_exit(args.instance)
    public_key()
    route = reach(args.instance)
    if route is None:
        print("no route accepted ssh", file=sys.stderr)
        return 255
    print(json.dumps({"ssh_config": str(write_connection(args.instance, *route)), "ssh_host": f"vast-{args.instance}"}))
    return 0


def held_or_exit(instance):
    held = rental(entries(), instance)
    if held is None:
        raise SystemExit(f"the ledger has no rental {instance}")
    return held


def extend(args):
    held = held_or_exit(args.instance)
    if held["returned"] or held["released"]:
        raise SystemExit(f"{args.instance} went back already or is going back")
    entry = record("extended", provider=PROVIDER, instance=args.instance,
                   deadline=stamp(now() + timedelta(hours=args.hours)))
    print(json.dumps({"instance": args.instance, "deadline": entry["deadline"]}))
    return 0


def note(args):
    held = held_or_exit(args.instance)
    record("noted", provider=PROVIDER, instance=args.instance, machine=held["machine"], host=held["host"],
           gpu=held["gpu"], note=args.text, values=args.values or {})
    return 0


def reject(args):
    """Record that the machine failed the work, as a failed check would have, so that rent never rents it again."""
    held = held_or_exit(args.instance)
    record("unusable", provider=PROVIDER, instance=args.instance, machine=held["machine"], host=held["host"],
           gpu=held["gpu"], stage="work", detail=args.detail[-500:])
    print(json.dumps({"rejected": args.instance, "machine": held["machine"]}))
    return 0


def give_back(args):
    """Send back the machine, or every machine with the label that has not gone back, and print each one; a machine
    that Vast.ai did not destroy is the watcher's to send back, and makes this fail."""
    if args.label is None:
        held_or_exit(args.instance)
        instances = [args.instance]
    else:
        instances = [instance for instance, held in rentals(entries()).items()
                     if held.get("label") == args.label and not held["returned"]]
    failed = 0
    for instance in instances:
        try:
            entry = send_back(instance, by="command")
        except (VastError, OSError) as error:
            failed += 1
            print(f"instance {instance} is going back, but not yet: {hide_addresses(str(error))}", file=sys.stderr)
            continue
        print(json.dumps({"instance": instance, "returned": "before"} if entry is None else
                         {"returned": instance, "hours": entry["hours"], "dollars": entry["dollars"]}))
    return 1 if failed else 0


def list_machines(args):
    """Print each machine that Vast.ai holds for this account, or each with the label: its deadline, and the hours and
    dollars it has run up at its price since it was rented, as the ledger knows them, or as Vast.ai does when the ledger
    has no rental of it."""
    held = rentals(entries())
    listed = call("GET", "/instances/", query={"owner": "me"}, version="v1")
    if listed.get("next_token"):
        raise VastError(None, "the instances fill more than one page of the list, which list does not follow")
    for row in listed["instances"]:
        if args.label is not None and row.get("label") != args.label:
            continue
        rented = held.get(row["id"])
        if rented is None:
            since = datetime.fromtimestamp(row.get("start_date") or now().timestamp(), timezone.utc)
            price, deadline = row.get("dph_total") or 0.0, None
        else:
            since, price, deadline = datetime.fromisoformat(rented["at"]), rented.get("price") or 0.0, rented["deadline"]
        hours = max(0.0, (now() - since).total_seconds() / 3600)
        print(json.dumps({"instance": row["id"], "label": row.get("label"), "status": row.get("actual_status"),
                          "deadline": deadline, "hours": round(hours, 3), "dollars": round(hours * price, 4)}))
    return 0


def show_ledger(args):
    ledger, machines = entries(), {}
    for entry in ledger:
        if entry.get("machine") is None or (args.gpu and entry.get("gpu") != args.gpu):
            continue
        row = machines.setdefault((entry["provider"], entry["machine"]), {
            "machine": entry["machine"], "host": entry.get("host"), "gpu": entry.get("gpu"), "where": entry.get("where"),
            "rented": 0, "passed": 0, "failed": [], "notes": []})
        if entry["event"] == "rented":
            row["rented"] += 1
            row["last"] = entry["at"][:10]
        elif entry["event"] == "checked" and entry.get("kept"):
            row["passed"] += 1
            row["check"] = summary(entry["results"])
        elif entry["event"] == "unusable":
            row["failed"].append(f"{entry['stage']}: {entry.get('detail', '')[:120]}")
        elif entry["event"] == "noted":
            row["notes"].append({"note": entry["note"], **entry.get("values", {})})
    for gpu in sorted({row["gpu"] for row in machines.values() if row["gpu"]}):
        medians, count = typical(ledger, gpu, machine_check.VERSION)
        print(json.dumps({"gpu": gpu, "passed_machines": count,
                          "medians": {name: round(value, 1) for name, value in medians.items()}}))
    for row in machines.values():
        print(json.dumps(row, ensure_ascii=False))
    return 0


def parser():
    found = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Another folder for the ledger, for tests and trials, with a watcher of its own started with the same --data.
    found.add_argument("--data", help=argparse.SUPPRESS)
    commands = found.add_subparsers(dest="command", required=True)

    def searching(command):
        command.add_argument("gpu", help='the GPU as Vast.ai names it, such as "RTX 5090" or "H100 SXM"')
        command.add_argument("--count", type=int, default=1, help="GPUs a machine")
        command.add_argument("--max-price", type=float, help="dollars an hour, with the disk")
        command.add_argument("--countries", type=lambda text: [code for code in text.split(",") if code],
                             help="country codes, comma-separated")
        command.add_argument("--disk", type=float, default=60, help="GB")
        command.add_argument("--query", type=json.loads,
                             help='more of Vast.ai\'s search filters, as JSON, such as {"cuda_max_good": {"gte": 12.8}} '
                                  "for machines whose driver runs CUDA 12.8")
        command.add_argument("--unverified", action="store_true",
                             help="also machines that Vast.ai has not verified yet")
        command.add_argument("--avoid", type=lambda text: [int(item) for item in text.split(",") if item],
                             default=[], help="instances, comma-separated, whose machines to leave out")

    offers = commands.add_parser("offers", help="show the offers that rent would try, in its order")
    searching(offers)
    offers.add_argument("--show", type=int, default=10)
    renting = commands.add_parser("rent", help="rent and check machines until one passes")
    searching(renting)
    renting.add_argument("--image", required=True)
    renting.add_argument("--label", required=True)
    renting.add_argument("--hours", type=float, required=True, help="until the watcher sends the machine back")
    renting.add_argument("--tries", type=int, default=3, help="machines that may fail before rent gives up")
    renting.add_argument("--failure-budget", type=float,
                         help="dollars that machines failing their check may cost in all before rent gives up")
    renting.add_argument("--min-download", type=float, default=MIN_DOWNLOAD_MBPS,
                         help="MB/s that a machine must download at, from the faster of the check's two sources; "
                              "0 skips it")
    renting.add_argument("--env", type=json.loads, help="the container's environment, as JSON")
    renting.add_argument("--onstart", help="a command the container runs when it starts")
    extending = commands.add_parser("extend", help="move the machine's deadline to this many hours from now")
    extending.add_argument("instance", type=int)
    extending.add_argument("hours", type=float)
    noting = commands.add_parser("note", help="record how the work ran on the machine")
    noting.add_argument("instance", type=int)
    noting.add_argument("text")
    noting.add_argument("--values", type=json.loads, help="measurements, as a JSON object")
    rejecting = commands.add_parser("reject", help="record that the machine failed the work, so that rent never "
                                                   "rents it again")
    rejecting.add_argument("instance", type=int)
    rejecting.add_argument("detail")
    giving_back = commands.add_parser("return", help="send back the machine, or every machine with the label")
    which = giving_back.add_mutually_exclusive_group(required=True)
    which.add_argument("instance", type=int, nargs="?")
    which.add_argument("--label")
    holding = commands.add_parser("list", help="show the machines that Vast.ai holds, with each one's deadline and cost")
    holding.add_argument("--label")
    connecting = commands.add_parser("ssh-config", help="write how to reach the machine over ssh again, and print where")
    connecting.add_argument("instance", type=int)
    listing = commands.add_parser("ledger", help="show how every machine did, and each GPU's medians")
    listing.add_argument("gpu", nargs="?")
    commands.add_parser("watch", help="send back this workstation's machines at their deadlines, round after round")
    return found


COMMANDS = {"offers": list_offers, "rent": rent, "ssh-config": ssh_config, "extend": extend, "note": note,
            "reject": reject, "return": give_back, "list": list_machines, "ledger": show_ledger, "watch": watch}


def main(argv=None):
    global folder
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    if args.data:
        folder = Path(args.data)
    if args.command in KEYED and not os.environ.get("VAST_API_KEY"):
        raise SystemExit(f"{args.command} needs the Vast.ai API key in VAST_API_KEY, in .env or the environment")
    try:
        return COMMANDS[args.command](args)
    except ImageFault as fault:
        print(f"the image did not start, and the machine went back: {fault}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(f"{error}, and the machine went back", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
