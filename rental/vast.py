"""Machines from Vast.ai, rented through vast-rental, which checks each, keeps the shared ledger and returns them.

The spec's provider says what to rent:
  {"kind": "vast", "gpu": "RTX 5090", "image": ..., "disk": GB, "max_price": dollars an hour,
   "query": {...}, "tries": N, "min_download": Mbps}
vast-rental needs the Vast.ai API key in VAST_API_KEY and the SSH key in the file
VAST_SSH_KEY_FILE names; a machine is reached as `ssh -F CONFIG -i KEY vast-ID`,
with the configuration vast-rental writes for it, so no address shows here.
"""

import json
import math
import os
import subprocess
import time
from urllib.request import Request, urlopen

from .controller import Unreachable

API = os.environ.get("VAST_URL", "https://console.vast.ai").rstrip("/") + "/api/v0"
# ssh exits with this when it cannot reach the machine; anything else is the command's own.
SSH_FAILED = 255


class VastProvider:
    def __init__(self, command="vast-rental", environment=None):
        self.command = command
        self.environment = os.environ if environment is None else environment
        self.configs = {}

    def vast_rental(self, *arguments, timeout=600):
        ran = subprocess.run([self.command, *arguments], capture_output=True, text=True, timeout=timeout)
        answers = [json.loads(line) for line in ran.stdout.splitlines() if line.startswith("{")]
        return ran.returncode, answers

    def rent(self, label, needs, hours):
        """Rent a machine that passes vast-rental's checks: its id, or None, and the dollars spent on failed ones."""
        arguments = ["rent", needs["gpu"], "--image", needs["image"], "--label", label,
                     "--hours", str(math.ceil(hours)), "--disk", str(needs["disk"]),
                     "--max-price", str(needs["max_price"]), "--tries", str(needs.get("tries", 6))]
        if needs.get("query"):
            arguments += ["--query", json.dumps(needs["query"])]
        if needs.get("min_download"):
            arguments += ["--min-download", str(needs["min_download"])]
        _, answers = self.vast_rental(*arguments, timeout=3 * 3600)
        result = next((answer for answer in answers if "rented" in answer), {})
        ident = result.get("rented")
        if ident is not None:
            self.configs[str(ident)] = result["ssh_config"]
        return (None if ident is None else str(ident)), result.get("failed_dollars", 0.0)

    def machines(self, label):
        request = Request(f"{API}/instances/?owner=me", headers={
            "Accept": "application/json", "User-Agent": "rental",
            "Authorization": "Bearer " + self.environment["VAST_API_KEY"]})
        with urlopen(request, timeout=60) as response:
            listed = json.load(response)["instances"]
        now = time.time()
        return [{"id": str(row["id"]),
                 "cost": (row.get("dph_total") or 0.0) * max(0.0, now - (row.get("start_date") or now)) / 3600}
                for row in listed if row.get("label") == label]

    def config(self, ident):
        if ident not in self.configs:
            code, answers = self.vast_rental("ssh-config", ident)
            if code or not answers:
                raise Unreachable(ident)
            self.configs[ident] = answers[-1]["ssh_config"]
        return self.configs[ident]

    def run(self, ident, command, data, timeout):
        ssh = ["ssh", "-F", self.config(ident), "-i", self.environment["VAST_SSH_KEY_FILE"],
               "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", f"vast-{ident}", command]
        try:
            ran = subprocess.run(ssh, input=data, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise Unreachable(ident) from None
        if ran.returncode == SSH_FAILED:
            raise Unreachable(ident)
        return ran.returncode, ran.stdout.decode(errors="replace")

    def extend(self, ident, hours):
        self.vast_rental("extend", ident, str(math.ceil(hours)))

    def reject(self, ident, reason):
        self.vast_rental("reject", ident, reason)

    def release(self, ident):
        _, answers = self.vast_rental("return", ident)
        self.configs.pop(ident, None)
        return next((answer["dollars"] for answer in answers if "dollars" in answer), None)
