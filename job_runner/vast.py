"""Machines from Vast.ai through vast/machines.py, which checks each before it is used, keeps the shared ledger of them, and
sends each back at its deadline.

The spec's provider says what to rent:
  {"kind": "vast", "gpu": "RTX 5090", "image": ..., "disk": GB, "max_price": dollars an hour,
   "query": {...}, "tries": N, "min_download": MB/s}
machines.py keeps the Vast.ai API key itself, and takes the SSH key from the file VAST_SSH_KEY_FILE names, as this
does: a machine is reached as `ssh -F CONFIG -i KEY vast-ID`, with the configuration machines.py writes for it, so
that no address shows here.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .provider import RentError, Unreachable

# The Vast.ai part of job-runner, a script beside the package.
MACHINES = Path(__file__).resolve().parents[1] / "vast" / "machines.py"
# ssh exits with this when it cannot reach the machine; anything else is the command's own.
SSH_FAILED = 255


def moment(stamp):
    return None if stamp is None else datetime.fromisoformat(stamp).timestamp()


class VastProvider:
    def __init__(self, command=(sys.executable, str(MACHINES)), environment=None):
        self.command = tuple(command)
        self.environment = os.environ if environment is None else environment
        self.configs = {}

    def script(self, *arguments, timeout=600):
        """machines.py's exit code, the JSON lines it printed, and the last line it wrote to standard error."""
        ran = subprocess.run([*self.command, *arguments], capture_output=True, text=True, timeout=timeout,
                             env=self.environment)
        answers = [json.loads(line) for line in ran.stdout.splitlines() if line.startswith("{")]
        return ran.returncode, answers, (ran.stderr.strip().splitlines() or [""])[-1]

    def rent(self, label, needs, hours, avoid=()):
        arguments = ["rent", needs["gpu"], "--image", needs["image"], "--label", label, "--hours", f"{hours:g}",
                     "--disk", str(needs["disk"]), "--max-price", str(needs["max_price"]),
                     "--tries", str(needs.get("tries", 6))]
        if avoid:
            arguments += ["--avoid", ",".join(avoid)]
        if needs.get("query"):
            arguments += ["--query", json.dumps(needs["query"])]
        if needs.get("min_download") is not None:
            arguments += ["--min-download", str(needs["min_download"])]
        code, answers, complaint = self.script(*arguments, timeout=3 * 3600)
        # rent ends with a line that names the machine it kept, or None when no offer passed.
        result = next((answer for answer in answers if "rented" in answer), None)
        if result is None:
            raise RentError(complaint or f"rent ended with {code}")
        if result["rented"] is None:
            return None, result.get("failed_dollars", 0.0)
        ident = str(result["rented"])
        self.configs[ident] = result["ssh_config"]
        return ident, result.get("failed_dollars", 0.0)

    def machines(self, label):
        try:
            code, answers, _ = self.script("list", "--label", label, timeout=300)
        except subprocess.TimeoutExpired:
            return None
        if code:
            return None
        return [{"id": str(row["instance"]), "cost": row["dollars"], "deadline": moment(row["deadline"])}
                for row in answers]

    def config(self, ident):
        if ident not in self.configs:
            code, answers, _ = self.script("ssh-config", ident)
            if code or not answers:
                raise Unreachable(ident)
            self.configs[ident] = answers[-1]["ssh_config"]
        return self.configs[ident]

    def run(self, ident, command, data, timeout):
        ssh = ["ssh", "-F", self.config(ident), "-i", self.environment["VAST_SSH_KEY_FILE"],
               "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", f"vast-{ident}", command]
        try:
            ran = subprocess.run(ssh, input=data, capture_output=True, timeout=timeout, env=self.environment)
        except subprocess.TimeoutExpired:
            raise Unreachable(ident) from None
        if ran.returncode == SSH_FAILED:
            raise Unreachable(ident)
        return ran.returncode, ran.stdout.decode(errors="replace")

    def extend(self, ident, hours):
        code, answers, _ = self.script("extend", ident, f"{hours:g}")
        return moment(answers[-1]["deadline"]) if not code and answers else None

    def reject(self, ident, reason):
        self.script("reject", ident, reason)

    def release(self, ident):
        _, answers, _ = self.script("return", ident)
        self.configs.pop(ident, None)
        return next((answer["dollars"] for answer in answers if "dollars" in answer), None)
