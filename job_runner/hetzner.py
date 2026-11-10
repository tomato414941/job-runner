"""Servers from Hetzner Cloud, rented by the hour for a run and deleted when it is done.

The spec's provider says what to rent:
  {"kind": "hetzner", "server_type": "cpx51", "location": "hel1", "image": "ubuntu-24.04"}
HCLOUD_TOKEN is the API token of the Hetzner Cloud project to rent in, and HETZNER_SSH_KEY_FILE
the private key whose public half each server is given; rent adds that half to the project's
keys when it is not among them. Each server carries the run's label and its deadline as labels,
so that this workstation's watcher, `job-runner hetzner watch`, deletes the servers past their
deadline whatever has become of their run; rent rents nothing while the watcher does not run
with the same token. A server is reached as root at its address, which never shows here. Costs
are counted in dollars, so rent rents nothing either from an account that Hetzner bills in
another currency.
"""

import base64
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

from .provider import RentError, Unreachable, state
from .watcher import needed, watch, watching

API = "https://api.hetzner.cloud/v1"
LABEL, DEADLINE = "job-runner", "job-runner-deadline"
# Hetzner's answers when the server type has no room in the location, which is a want of an offer.
UNAVAILABLE = {"resource_unavailable", "placement_error"}
# A new server answers ssh within a minute or two; one that has not by then goes back.
START_SECONDS = 600
SSH_FAILED = 255


class HetznerProvider:
    def __init__(self, environment=None):
        self.environment = os.environ if environment is None else environment
        self.api = self.environment.get("HCLOUD_URL", API)
        self.addresses = {}

    def request(self, method, path, body=None):
        request = urllib.request.Request(
            self.api + path, method=method, data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.environment["HCLOUD_TOKEN"], "Content-Type": "application/json", "User-Agent": "job-runner"})
        with urllib.request.urlopen(request, timeout=60) as answer:
            text = answer.read()
        return json.loads(text) if text else {}

    def servers(self, selector):
        query = urllib.parse.urlencode({"label_selector": selector, "per_page": 50})
        return self.request("GET", f"/servers?{query}")["servers"]

    def answers(self, address):
        """Whether the server takes connections to ssh's port."""
        try:
            socket.create_connection((address, 22), timeout=5).close()
            return True
        except OSError:
            return False

    def rent(self, label, needs, hours, avoid=()):
        if not watching("hetzner", self.environment["HCLOUD_TOKEN"]):
            raise RentError(needed("hetzner"))
        public = subprocess.run(["ssh-keygen", "-y", "-f", self.environment["HETZNER_SSH_KEY_FILE"]],
                                capture_output=True, text=True, check=True, env=self.environment).stdout.strip()
        deadline = int(time.time() + hours * 3600)
        body = {"name": f"{label}-{deadline}"[-63:].lstrip("-"), "server_type": needs["server_type"],
                "image": needs["image"], "location": needs["location"],
                "labels": {LABEL: label, DEADLINE: str(deadline)}, "start_after_create": True}
        try:
            if self.request("GET", "/pricing")["pricing"]["currency"] != "USD":
                raise RentError("job-runner counts costs in dollars, and Hetzner bills this account in another currency")
            server = self.request("POST", "/servers", {**body, "ssh_keys": [self.key(public)]})["server"]
        except urllib.error.HTTPError as error:
            code = json.loads(error.read() or b"{}").get("error", {}).get("code")
            if code in UNAVAILABLE:
                return None, 0.0
            raise RentError(f"Hetzner refused the server: {code or error.code}") from None
        except (urllib.error.URLError, OSError) as error:
            raise RentError(f"Hetzner did not answer: {type(error).__name__}") from None
        ident = str(server["id"])
        started = time.monotonic()
        while time.monotonic() - started < START_SECONDS:
            server = self.request("GET", f"/servers/{ident}")["server"]
            address = server["public_net"]["ipv4"]["ip"]
            if server["status"] == "running" and self.answers(address):
                self.addresses[ident] = address
                return ident, 0.0
            time.sleep(5)
        self.release(ident)
        raise RentError(f"the server did not answer ssh within {START_SECONDS // 60} minutes, and it was deleted")

    def key(self, public):
        """The name of the project's copy of a public key, which is added to the project when it holds none."""
        digest = hashlib.md5(base64.b64decode(public.split()[1]), usedforsecurity=False).hexdigest()
        # Hetzner finds a key by this fingerprint of it.
        query = urllib.parse.urlencode({"fingerprint": ":".join(digest[at:at + 2] for at in range(0, 32, 2))})
        held = self.request("GET", f"/ssh_keys?{query}")["ssh_keys"]
        if not held:
            try:
                held = [self.request("POST", "/ssh_keys",
                                     {"name": f"job-runner-{digest[:12]}", "public_key": public})["ssh_key"]]
            except urllib.error.HTTPError as error:
                # Another run may have added it meanwhile.
                held = self.request("GET", f"/ssh_keys?{query}")["ssh_keys"] if error.code == 409 else []
                if not held:
                    raise
        return held[0]["name"]

    @staticmethod
    def cost(server):
        """Dollars so far: each hour begun at the server type's hourly price in its location."""
        location = server["location"]["name"]
        hourly = next(float(price["price_hourly"]["gross"]) for price in server["server_type"]["prices"]
                      if price["location"] == location)
        hours = (time.time() - datetime.fromisoformat(server["created"]).timestamp()) / 3600
        return round(math.ceil(max(hours, 1e-9)) * hourly, 4)

    def machines(self, label):
        try:
            servers = self.servers(f"{LABEL}=={label}")
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            return None
        return [{"id": str(server["id"]), "cost": self.cost(server),
                 "deadline": int(server["labels"][DEADLINE]) if DEADLINE in server["labels"] else None}
                for server in servers]

    def run(self, ident, command, data, timeout):
        if ident not in self.addresses:
            try:
                self.addresses[ident] = self.request("GET", f"/servers/{ident}")["server"]["public_net"]["ipv4"]["ip"]
            except (urllib.error.URLError, OSError, KeyError, ValueError):
                raise Unreachable(ident) from None
        # Each server keeps its own host key, so that one that takes over an address is not taken for another.
        known = state("hetzner") / "known_hosts"
        known.mkdir(parents=True, exist_ok=True, mode=0o700)
        ssh = ["ssh", "-i", self.environment["HETZNER_SSH_KEY_FILE"], "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
               "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known / ident}",
               f"root@{self.addresses[ident]}", command]
        try:
            ran = subprocess.run(ssh, input=data, capture_output=True, timeout=timeout, env=self.environment)
        except subprocess.TimeoutExpired:
            raise Unreachable(ident) from None
        if ran.returncode == SSH_FAILED:
            raise Unreachable(ident)
        return ran.returncode, ran.stdout.decode(errors="replace")

    def extend(self, ident, hours):
        deadline = int(time.time() + hours * 3600)
        try:
            labels = self.request("GET", f"/servers/{ident}")["server"]["labels"]
            self.request("PUT", f"/servers/{ident}", {"labels": {**labels, DEADLINE: str(deadline)}})
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            return None
        return deadline

    def reject(self, ident, reason):
        """A new server is a new machine, so there is none to avoid."""

    def release(self, ident):
        try:
            server = self.request("GET", f"/servers/{ident}")["server"]
            self.request("DELETE", f"/servers/{ident}")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        self.addresses.pop(ident, None)
        return self.cost(server)

    def sweep(self):
        """Delete every run's server whose deadline has passed; print each one deleted as a JSON line."""
        now = time.time()
        for server in self.servers(LABEL):
            if int(server["labels"].get(DEADLINE, "0")) < now:
                print(json.dumps({"deleted": server["id"], "label": server["labels"].get(LABEL),
                                  "dollars": self.release(str(server["id"]))}), flush=True)


def main(argv):
    if argv != ["watch"]:
        print("usage: job-runner hetzner watch", file=sys.stderr)
        return 2
    provider = HetznerProvider()
    watch("hetzner", provider.environment["HCLOUD_TOKEN"], provider.sweep)
