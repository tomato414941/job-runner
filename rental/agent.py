#!/usr/bin/env python3
"""The agent on a rented machine: one daemon that installs a job, runs its steps and sends its files.

The machine holds no credential. The controller runs `agent.py COMMAND` over
ssh, through bootstrap.sh, with any payload as JSON on standard input; every
URL arrives that way, serves one transfer, and stays in the daemon's memory.
The first command starts the daemon, which serves every command on a Unix
socket and owns the job's processes: the job ends when the daemon does.

  status   the phase, the step and its rate, the outbox, the GPUs and the free disk
  install  {"code": {"url", "sha256", "bytes"}, "setup": [[argument, ...], ...], "check": [[argument, ...], ...]}
           fetch and unpack the job's code, then run its setup commands there, then its check
  start    {"files": [{"path", "url", "sha256", "bytes", "if_missing"}],
            "steps": [{"name", "command", "environment", "outputs"}], "check": [[argument, ...], ...]}
           fetch the files into the code, then run the steps in order; when one fails, run the check
  send     {"uploads": [{"id", "url"} | {"id", "upload", "parts": [{"number", "start", "length", "url"}]}]}
           upload outbox files where told
  forget   {"ids": [...]}  drop outbox files that storage now keeps
  stop     stop the step, and let the job save what it can
A command answers with one line of JSON: the report for status, {} for the
others, or {"error", "detail"} when the agent refuses, as it refuses an install
or a start while it installs or runs steps.

The phase says what the agent does, or how its last work ended: empty, without
the job; installing, then ready or install_failed; fetching what the steps
read, then working through them, then finished, stopped (told to), fetch_failed
or crashed; or, when a step exits with another code than 0, checking the
machine, then failed. A failure tells the step, its exit code, whether the
machine still passes its check, and what failed of the check if not. Each
command runs in a process group of its own, and what it leaves running ends with
it, so that nothing of a failed step stays to fail the check or the next step.

What a step's command does for the run:
- run again, it finishes what is left and redoes nothing done;
- it may write {"done": N, "total": M} to the file RENTAL_PROGRESS names;
- a file it wants kept, such as a checkpoint once saved, it links into the
  directory RENTAL_KEEP names, under its path relative to the code with each
  "/" written "%2F", replacing that name atomically; the newest link of a path wins;
- on SIGTERM it saves what it can, keeps it as above, and exits.
When a step's command exits 0, its outputs and its log go to the outbox, and its
checkpoints not yet on their way are dropped: nothing needs them any more.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import queue
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote
from urllib.request import Request, urlopen

HOME = Path(os.environ.get("RENTAL_HOME") or Path.home() / ".rental")
CODE, KEEP, OUTBOX, LOGS = HOME / "code", HOME / "keep", HOME / "outbox", HOME / "logs"
SOCKET, LOCK, STATE = HOME / "agent.sock", HOME / "agent.lock", HOME / "state.json"
PROGRESS, INSTALLED = HOME / "progress.json", HOME / "installed"
# A download arrives in parts over several connections; a connection that brings
# less than SLOW_BYTES a second over SLOW_SECONDS asks again, as a new one usually
# runs at full speed.
PART_BYTES, CONNECTIONS = 64 * 1024 * 1024, 8
SLOW_SECONDS, SLOW_BYTES, SLOW_RESTARTS = 60, 1024 * 1024, 5
ATTEMPTS = 3
# The job is started through this, so that it is sent SIGTERM when the thread that
# started it ends, as when the daemon does; Linux keeps that across exec.
WITH_PARENT = ("import ctypes, os, signal, sys; ctypes.CDLL(None).prctl(1, signal.SIGTERM); "
               "os.getppid() == int(sys.argv[1]) or os._exit(1); os.execvp(sys.argv[2], sys.argv[2:])")


# Transfers


def digests(path, start=0, length=None):
    """The SHA-256 and MD5 of a file, or of length bytes of it from start, in one read."""
    checksum, content = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as source:
        source.seek(start)
        left = float("inf") if length is None else length
        while left > 0 and (chunk := source.read(int(min(1 << 20, left)))):
            checksum.update(chunk)
            content.update(chunk)
            left -= len(chunk)
    return checksum.hexdigest(), content.hexdigest()


def fetch_part(url, descriptor, start, end, size):
    """Write bytes start to end - 1 of the object at the same offsets; a short or slow answer asks again."""
    position, stalled, restarts, error = start, 0, 0, None
    while position < end:
        before = position
        try:
            with urlopen(Request(url, headers={"Range": f"bytes={position}-{end - 1}"}), timeout=120) as response:
                total = (response.headers.get("Content-Range", "").rpartition("/")[2] if response.status == 206
                         else response.headers.get("Content-Length") if (position, end) == (0, size) else None)
                if total != str(size):
                    raise ValueError("the store answered with another object")
                since, reached = time.monotonic(), position
                while position < end and (chunk := response.read1(min(1 << 20, end - position))):
                    os.pwrite(descriptor, chunk, position)
                    position += len(chunk)
                    if time.monotonic() - since >= SLOW_SECONDS:
                        if restarts < SLOW_RESTARTS and position - reached < SLOW_BYTES * (time.monotonic() - since):
                            restarts += 1
                            break
                        since, reached = time.monotonic(), position
        except (OSError, HTTPException) as failure:
            error = failure
        if position == before:
            stalled += 1
            if stalled == ATTEMPTS:
                raise error or ValueError("the download ended early")
            time.sleep(2**stalled)


def download(url, path, sha256, size):
    """Fetch an object into path, checking its digest; a file already there with that digest stays."""
    path = Path(path)
    if path.is_file() and path.stat().st_size == size and digests(path)[0] == sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.download")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(descriptor, size)
        with ThreadPoolExecutor(CONNECTIONS) as pool:
            for part in [pool.submit(fetch_part, url, descriptor, start, min(start + PART_BYTES, size), size)
                         for start in range(0, size, PART_BYTES)]:
                part.result()
    finally:
        os.close(descriptor)
    if digests(temporary)[0] != sha256:
        temporary.unlink()
        raise ValueError("the download does not match its digest")
    os.replace(temporary, path)


class Slice:
    """Length bytes of a file from start, read as a request's body."""

    def __init__(self, source, start, length):
        self.source, self.left = source, length
        source.seek(start)

    def read(self, size=-1):
        data = self.source.read(self.left if size is None or size < 0 else min(size, self.left))
        self.left -= len(data)
        return data


def put(path, url, start, length):
    """Send length bytes of the file from start and return the ETag; a refusal is not asked again."""
    for attempt in range(ATTEMPTS):
        try:
            with open(path, "rb") as source:
                request = Request(url, data=Slice(source, start, length), method="PUT",
                                  headers={"Content-Length": str(length)})
                with urlopen(request, timeout=120) as response:
                    response.read()
                    return (response.headers.get("ETag") or "").strip('"')
        except HTTPError as error:
            if error.code < 500 or attempt + 1 == ATTEMPTS:
                raise
        except (OSError, HTTPException):
            if attempt + 1 == ATTEMPTS:
                raise
        time.sleep(2**attempt)


# The daemon


def within(root, relative):
    """The path relative to root, which must not lead out of it."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("a path leads out of the code")
    return path


def ended(command, code, log):
    """What a command that failed said last, for the controller to tell."""
    try:
        with open(log, "rb") as source:
            source.seek(max(0, source.seek(0, os.SEEK_END) - 400))
            said = " ".join(source.read().decode(errors="replace").split())
    except OSError:
        said = ""
    return f"{' '.join(command)[:120]} ended with {code}: {said}"


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value) + "\n")
    os.replace(temporary, path)


def gpus():
    fields = ["name", "memory.used", "memory.total", "utilization.gpu", "power.draw", "power.limit",
              "clocks.sm", "clocks.max.sm"]
    try:
        listed = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listed.returncode:
        return None
    return [dict(zip(fields, (value.strip() for value in line.split(",")))) for line in listed.stdout.splitlines()]


class Busy(Exception):
    """The agent installs or runs steps, and takes no other install or start meanwhile."""


class Daemon:
    def __init__(self):
        self.lock = threading.RLock()
        saved = json.loads(STATE.read_text()) if STATE.exists() else {}
        # An upload the daemon was making when it stopped did not finish.
        self.outbox = {entry["id"]: {**entry, "state": "failed" if entry["state"] == "sending" else entry["state"]}
                       for entry in saved.get("outbox", [])}
        # Each link in the keep directory taken into the outbox, as [name, inode, modified nanoseconds].
        self.seen = {tuple(link) for link in saved.get("seen", [])}
        self.phase = "ready" if INSTALLED.exists() else "empty"
        self.step, self.failure, self.progress, self.started = None, None, None, None
        self.worker, self.process, self.stopping = None, None, False
        # How often the daemon looks at the job, and over how long it reckons its rate.
        self.poll, self.window = 2.0, 600.0
        self.uploads = queue.Queue()
        for directory in (KEEP, OUTBOX, LOGS):
            directory.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self.send_loop, daemon=True).start()

    def save(self):
        with self.lock:
            write_json(STATE, {"outbox": list(self.outbox.values()), "seen": sorted(self.seen)})

    # Commands

    def handle(self, command, payload):
        if command == "status":
            return self.status()
        if command in ("install", "start"):
            with self.lock:
                if self.worker and self.worker.is_alive():
                    raise Busy(f"the agent is {self.phase}")
                self.worker = threading.Thread(target=self.work, args=(command, payload), daemon=True)
                self.worker.start()
        elif command == "send":
            with self.lock:
                for upload in payload["uploads"]:
                    entry = self.outbox.get(upload["id"])
                    if entry and entry["state"] in ("ready", "failed", "sending"):
                        entry.update(state="sending", error=None)
                        self.uploads.put(upload)
            self.save()
        elif command == "forget":
            self.forget(set(payload["ids"]))
        elif command == "stop":
            with self.lock:
                self.stopping = True
                if self.process and self.process.poll() is None:
                    self.process.send_signal(signal.SIGTERM)
        else:
            raise ValueError(f"no command {command}")
        return {}

    def status(self):
        with self.lock:
            # How long the step, or the check after it failed, has run.
            seconds = time.time() - self.started if self.phase in ("working", "checking") else None
            report = {
                "phase": self.phase, "step": self.step, "failure": self.failure, "seconds": seconds,
                "progress": self.progress, "outbox": [dict(entry) for entry in self.outbox.values()],
            }
        report["gpus"] = gpus()
        report["disk_free"] = shutil.disk_usage(HOME).free
        return report

    def run(self, command, log, environment=None, step=None):
        """Run a command in the code, its output appended to log, unless stopping, and return its exit code.

        A step's command learns where to report its progress and keep its files, which are taken in while it runs.
        The command runs in a process group of its own, and what it leaves running is ended with it.
        """
        variables = {**os.environ, **(environment or {})}
        if step:
            variables.update(RENTAL_PROGRESS=str(PROGRESS), RENTAL_KEEP=str(KEEP))
        with self.lock:
            if self.stopping:
                return None
            with open(log, "a") as output:
                self.process = subprocess.Popen(
                    [sys.executable, "-c", WITH_PARENT, str(os.getpid()), *command], cwd=CODE, stdout=output,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=variables, process_group=0,
                )
        process, observations = self.process, deque()
        while process.poll() is None:
            time.sleep(self.poll)
            if step:
                self.observe(observations)
                self.keep_scan(step)
        if step:
            self.observe(observations)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        return process.returncode

    def check(self, commands):
        """Whether the machine passes the job's check, and what failed of it if not."""
        for command in commands:
            code = self.run(command, LOGS / "check.log")
            if code != 0:
                return False, ended(command, code, LOGS / "check.log")
        return True, None

    def observe(self, observations):
        """Read the job's progress and work out its rate over the last window."""
        try:
            reported = json.loads(PROGRESS.read_text())
        except (FileNotFoundError, ValueError):
            return
        now, done = time.time(), reported.get("done")
        if done is None:
            return
        if not observations or done != observations[-1][1]:
            observations.append((now, done))
        while len(observations) > 1 and observations[1][0] < now - self.window:
            observations.popleft()
        if observations[-1][0] < now - self.window:
            rate = 0.0
        elif len(observations) > 1 and observations[-1][0] - observations[0][0] >= self.window / 2:
            (first, start), (last, end) = observations[0], observations[-1]
            rate = (end - start) / (last - first)
        else:
            rate = None
        with self.lock:
            self.progress = {"done": done, "total": reported.get("total"), "rate": rate,
                             "changed_at": observations[-1][0]}

    def work(self, command, payload):
        """Install or start; whatever breaks it unforeseen ends it in a phase the controller acts on."""
        try:
            getattr(self, command)(payload)
        except Exception as error:  # A worker has no caller; the controller learns of it from the phase.
            traceback.print_exc()
            with self.lock:
                self.phase = "install_failed" if command == "install" else "crashed"
                self.failure = {"error": type(error).__name__}

    def install(self, payload):
        source = payload["code"]
        with self.lock:
            self.phase, self.failure, self.stopping = "installing", None, False
        try:
            if not (INSTALLED.exists() and INSTALLED.read_text().strip() == source["sha256"] and CODE.exists()):
                INSTALLED.unlink(missing_ok=True)
                archive = HOME / "code.tar.gz"
                download(source["url"], archive, source["sha256"], source["bytes"])
                unpacked = HOME / "code.new"
                shutil.rmtree(unpacked, ignore_errors=True)
                with tarfile.open(archive) as bundle:
                    bundle.extractall(unpacked, filter="data")
                shutil.rmtree(CODE, ignore_errors=True)
                unpacked.rename(CODE)
                archive.unlink()
            for command in payload.get("setup", []):
                code = self.run(command, LOGS / "setup.log")
                if code != 0:
                    raise RuntimeError(f"the setup failed: {ended(command, code, LOGS / 'setup.log')}")
            sound, detail = self.check(payload.get("check", []))
            if not sound:
                raise RuntimeError(f"the check failed: {detail}")
            INSTALLED.write_text(source["sha256"] + "\n")
            with self.lock:
                self.phase = "ready"
        except (OSError, ValueError, RuntimeError, HTTPException, tarfile.TarError) as error:
            with self.lock:
                self.phase, self.failure = "install_failed", {"error": type(error).__name__, "detail": str(error)}

    def start(self, payload):
        with self.lock:
            self.phase, self.failure, self.stopping = "fetching", None, False
            self.window = payload.get("rate_seconds", self.window)
            self.poll = min(2.0, self.window / 10)
        try:
            for item in payload["files"]:
                if self.stopping:
                    break
                path = within(CODE, item["path"])
                if not (item.get("if_missing") and path.exists()):
                    download(item["url"], path, item["sha256"], item["bytes"])
        except (OSError, ValueError, HTTPException) as error:
            with self.lock:
                self.phase, self.failure = "fetch_failed", {"error": type(error).__name__}
            return
        for step in payload["steps"]:
            with self.lock:
                self.phase, self.step, self.progress, self.started = "working", step["name"], None, time.time()
            PROGRESS.unlink(missing_ok=True)
            log = LOGS / f"{step['name']}.log"
            code = self.run(step["command"], log, step.get("environment"), step["name"])
            self.keep_scan(step["name"])
            if self.stopping or code != 0:
                if log.exists():
                    # A step told to stop before it began has no log.
                    self.enqueue("log", step["name"], log, f".rental/logs/{step['name']}.log")
                if not self.stopping:
                    # Whose fault it is: a machine that passed its check and fails it now is what broke.
                    with self.lock:
                        self.phase, self.started = "checking", time.time()
                    sound, detail = self.check(payload.get("check", []))
                with self.lock:
                    if self.stopping:
                        self.phase = "stopped"
                    else:
                        self.phase, self.failure = "failed", {"step": step["name"], "exit_code": code,
                                                              "sound": sound, "check": detail, "at": time.time()}
                return
            outputs = []
            for output in step.get("outputs", []):
                found = CODE / output
                outputs += sorted(item for item in found.rglob("*") if item.is_file()) if found.is_dir() else [found]
            listed = [str(item.relative_to(CODE)) for item in outputs]
            for item in outputs:
                self.enqueue("output", step["name"], item, str(item.relative_to(CODE)), listed)
            self.enqueue("log", step["name"], log, f".rental/logs/{step['name']}.log", listed)
            self.drop_checkpoints(step["name"])
        with self.lock:
            self.phase, self.step = "finished", None

    # The outbox

    def enqueue(self, kind, step, source, path, outputs=None, link=None):
        """Link a file into the outbox, so that the job may change it meanwhile, and list it there."""
        ident = uuid.uuid4().hex[:16]
        held = OUTBOX / ident
        os.link(source, held)
        sha256, md5 = digests(held)
        entry = {"id": ident, "kind": kind, "step": step, "path": path, "sha256": sha256, "md5": md5,
                 "bytes": held.stat().st_size, "state": "ready", "at": time.time()}
        if outputs is not None:
            entry["outputs"] = outputs
        if link is not None:
            entry["link"] = list(link)
        with self.lock:
            if kind == "checkpoint":
                # A newer checkpoint of a path takes the place of one not yet on its way.
                for older in [item for item in self.outbox.values()
                              if item["kind"] == "checkpoint" and item["path"] == path and item["state"] == "ready"]:
                    del self.outbox[older["id"]]
                    (OUTBOX / older["id"]).unlink(missing_ok=True)
                    self.seen.discard(tuple(older["link"]))
            self.outbox[ident] = entry
        self.save()

    def keep_scan(self, step):
        """Put each file the job newly linked into the keep directory in the outbox as a checkpoint."""
        for item in sorted(KEEP.iterdir()):
            if item.name.startswith("."):
                continue
            try:
                stat = item.stat()
            except FileNotFoundError:
                continue
            link = (item.name, stat.st_ino, stat.st_mtime_ns)
            if link in self.seen:
                continue
            with self.lock:
                self.seen.add(link)
            self.enqueue("checkpoint", step, item, unquote(item.name), link=link)

    def drop_checkpoints(self, step):
        """Forget the step's checkpoints not yet on their way: once it has ended well, nothing needs them."""
        with self.lock:
            ids = {ident for ident, entry in self.outbox.items()
                   if entry["kind"] == "checkpoint" and entry["step"] == step and entry["state"] == "ready"}
        self.forget(ids)

    def forget(self, ids):
        with self.lock:
            for ident in ids:
                entry = self.outbox.pop(ident, None)
                if not entry:
                    continue
                (OUTBOX / ident).unlink(missing_ok=True)
                if entry.get("link"):
                    # The job's link goes too, unless the job has linked a newer file there since.
                    name, inode, modified = entry["link"]
                    with contextlib.suppress(FileNotFoundError):
                        stat = (KEEP / name).stat()
                        if (stat.st_ino, stat.st_mtime_ns) == (inode, modified):
                            (KEEP / name).unlink()
                    self.seen.discard((name, inode, modified))
        self.save()

    def send_loop(self):
        while True:
            upload = self.uploads.get()
            with self.lock:
                entry = self.outbox.get(upload["id"])
            if entry and entry["state"] == "sending":
                self.send(upload, entry)

    def send(self, upload, entry):
        """Upload one file, whole or in parts, checking each ETag; whatever goes wrong fails this upload alone."""
        link = OUTBOX / entry["id"]
        try:
            if "url" in upload:
                etags = [[0, put(link, upload["url"], 0, entry["bytes"])]]
                if etags[0][1] != entry["md5"]:
                    raise ValueError("the store received other bytes")
            else:
                etags = []
                for part in upload["parts"]:
                    expected = digests(link, part["start"], part["length"])[1]
                    etag = put(link, part["url"], part["start"], part["length"])
                    if etag != expected:
                        raise ValueError("the store received other bytes")
                    etags.append([part["number"], etag])
            changes = {"state": "sent", "upload": upload.get("upload"), "etags": etags}
        except Exception as error:  # The sender serves on; the controller sends the file again.
            if not isinstance(error, (OSError, ValueError, HTTPException)):
                traceback.print_exc()
            changes = {"state": "failed", "error": type(error).__name__}
        with self.lock:
            if self.outbox.get(entry["id"], {}).get("state") == "sending":
                self.outbox[entry["id"]].update(changes)
        self.save()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        request = json.loads(self.rfile.readline())
        try:
            answer = self.server.daemon_.handle(request["command"], request.get("payload"))
        except Exception as error:  # The client reports it; the daemon serves on.
            answer = {"error": type(error).__name__, "detail": str(error)}
        self.wfile.write((json.dumps(answer) + "\n").encode())


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve():
    HOME.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    (HOME / "agent.pid").write_text(f"{os.getpid()}\n")
    SOCKET.unlink(missing_ok=True)
    server = Server(str(SOCKET), Handler)
    server.daemon_ = Daemon()
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    server.serve_forever()


# The client


def ask(command, payload):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(str(SOCKET))
        connection.sendall((json.dumps({"command": command, "payload": payload}) + "\n").encode())
        with connection.makefile() as answer:
            return json.loads(answer.readline())


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["serve"]:
        return serve()
    if len(arguments) != 1 or arguments[0] not in ("status", "install", "start", "send", "forget", "stop"):
        raise SystemExit(__doc__)
    command = arguments[0]
    payload = None if command in ("status", "stop") else json.load(sys.stdin)
    HOME.mkdir(parents=True, exist_ok=True)
    for attempt in range(50):
        try:
            answer = ask(command, payload)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if attempt == 0:
                with open(HOME / "daemon.log", "a") as log:
                    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve"], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
            time.sleep(0.2)
    else:
        raise SystemExit("the daemon did not start")
    print(json.dumps(answer))
    return 1 if "error" in answer else 0


if __name__ == "__main__":
    sys.exit(main())
