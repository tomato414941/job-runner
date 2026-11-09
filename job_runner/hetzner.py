"""Servers from Hetzner Cloud, rented by the hour for a run and deleted when it is done.

The spec's provider says what to rent:
  {"kind": "hetzner", "server_type": "cpx51", "location": "hel1", "image": "ubuntu-24.04", "ssh_key": NAME}
ssh_key names a key already in the Hetzner Cloud project, whose private half is the file
HETZNER_SSH_KEY_FILE names; HCLOUD_TOKEN is the project's API token. Each server carries the
run's label and its deadline as labels, so that `job-runner hetzner sweep` deletes the servers
past their deadline whatever has become of their run. A server is reached as root at its
address, which never shows here.
"""

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
from pathlib import Path

from .provider import RentError, Unreachable

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
        deadline = int(time.time() + hours * 3600)
        body = {"name": f"{label}-{deadline}"[-63:].lstrip("-"), "server_type": needs["server_type"],
                "image": needs["image"], "location": needs["location"], "ssh_keys": [needs["ssh_key"]],
                "labels": {LABEL: label, DEADLINE: str(deadline)}, "start_after_create": True}
        try:
            server = self.request("POST", "/servers", body)["server"]
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

    @staticmethod
    def cost(server):
        """Euros so far: each hour begun at the server type's hourly price in its location."""
        location = server["datacenter"]["location"]["name"]
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
        known = Path.home() / ".local/state/job-runner/hetzner"
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
                                  "euros": self.release(str(server["id"]))}), flush=True)


def main(argv):
    if argv != ["sweep"]:
        print("usage: job-runner hetzner sweep", file=sys.stderr)
        return 2
    HetznerProvider().sweep()
    return 0
