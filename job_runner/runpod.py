"""Pods from RunPod, rented for a run and deleted when it is done.

The spec's provider says what to rent:
  {"kind": "runpod", "compute": "GPU", "gpus": ["NVIDIA GeForce RTX 5090"], "gpu_count": 1,
   "image": IMAGE, "disk": GB, "cloud": "COMMUNITY", "min_vcpus_per_gpu": N, "allowed_cuda_versions": ["13.0"]}
or, for a CPU pod, {"kind": "runpod", "compute": "CPU", "vcpus": N, "image": IMAGE, "disk": GB, "cloud": ...}.
RunPod takes the first of the GPU types with room, on a machine with at least min_vcpus_per_gpu
CPUs a GPU and a driver for one of allowed_cuda_versions when they are given. The image must run sshd and take the public
key from PUBLIC_KEY, as RunPod's own images do. RUNPOD_API_KEY is the account's key, and
RUNPOD_SSH_KEY_FILE the private key whose public half the pod is given. RunPod keeps no deadline,
and renaming a pod may restart it, so each pod's run and deadline are kept here, in
~/.local/state/job-runner/runpod/pods.json, where this workstation's watcher, `job-runner runpod
watch`, finds those past their deadline and deletes them whatever has become of their run; rent
rents nothing while the watcher does not run with the same key. A pod is reached as root at its public address,
which never shows here.
"""

import json
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

API = "https://rest.runpod.io/v1"
# RunPod answers a want of room with a message rather than a code.
NO_ROOM = ("available", "capacity", "instances")
# A new pod answers ssh within minutes; one that has not by then goes back.
START_SECONDS = 900
SSH_FAILED = 255


class RunPodProvider:
    def __init__(self, environment=None):
        self.environment = os.environ if environment is None else environment
        self.api = self.environment.get("RUNPOD_URL", API)

    def request(self, method, path, body=None):
        request = urllib.request.Request(
            self.api + path, method=method, data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.environment["RUNPOD_API_KEY"],
                     "Content-Type": "application/json", "User-Agent": "job-runner"})
        with urllib.request.urlopen(request, timeout=60) as answer:
            text = answer.read()
        return json.loads(text) if text else {}

    def kept(self):
        path = state("runpod") / "pods.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def keep(self, pods):
        state("runpod").mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = state("runpod") / "pods.json.tmp"
        temporary.write_text(json.dumps(pods, indent=1) + "\n")
        temporary.replace(state("runpod") / "pods.json")

    def answers(self, address, port):
        """Whether the pod takes connections to ssh's port."""
        try:
            socket.create_connection((address, port), timeout=5).close()
            return True
        except OSError:
            return False

    @staticmethod
    def route(pod):
        """The pod's public address and ssh's port there, or None while it has none."""
        port = (pod.get("portMappings") or {}).get("22")
        return (pod["publicIp"], int(port)) if pod.get("publicIp") and port else None

    def rent(self, label, needs, hours, avoid=()):
        if not watching("runpod", self.environment["RUNPOD_API_KEY"]):
            raise RentError(needed("runpod"))
        public = subprocess.run(["ssh-keygen", "-y", "-f", self.environment["RUNPOD_SSH_KEY_FILE"]],
                                capture_output=True, text=True, check=True).stdout.strip()
        body = {"name": label, "computeType": needs.get("compute", "GPU"), "imageName": needs["image"],
                "containerDiskInGb": needs["disk"], "ports": ["22/tcp"], "env": {"PUBLIC_KEY": public},
                "cloudType": needs.get("cloud", "COMMUNITY"), "supportPublicIp": True}
        if body["computeType"] == "GPU":
            body.update(gpuTypeIds=needs["gpus"], gpuCount=needs.get("gpu_count", 1))
            if "min_vcpus_per_gpu" in needs:
                body["minVCPUPerGPU"] = needs["min_vcpus_per_gpu"]
            if "allowed_cuda_versions" in needs:
                body["allowedCudaVersions"] = needs["allowed_cuda_versions"]
        else:
            body["vcpuCount"] = needs["vcpus"]
        try:
            pod = self.request("POST", "/pods", body)
        except urllib.error.HTTPError as error:
            message = (error.read() or b"").decode(errors="replace")
            if any(word in message.lower() for word in NO_ROOM):
                return None, 0.0
            raise RentError(f"RunPod refused the pod: {message[:200] or error.code}") from None
        except (urllib.error.URLError, OSError) as error:
            raise RentError(f"RunPod did not answer: {type(error).__name__}") from None
        ident = pod["id"]
        pods = self.kept()
        pods[ident] = {"label": label, "deadline": int(time.time() + hours * 3600)}
        self.keep(pods)
        started = time.monotonic()
        while time.monotonic() - started < START_SECONDS:
            route = self.route(self.request("GET", f"/pods/{ident}"))
            if route and self.answers(*route):
                return ident, 0.0
            time.sleep(10)
        self.release(ident)
        raise RentError(f"the pod did not answer ssh within {START_SECONDS // 60} minutes, and it was deleted")

    @staticmethod
    def cost(pod):
        """Dollars since the pod last started, at its hourly price."""
        started = datetime.fromisoformat(pod["lastStartedAt"].replace("Z", "+00:00")).timestamp()
        return round(max(time.time() - started, 0) / 3600 * float(pod["costPerHr"]), 4)

    def machines(self, label):
        try:
            pods = self.request("GET", f"/pods?name={urllib.parse.quote(label)}")
        except (urllib.error.URLError, OSError, ValueError):
            return None
        kept = self.kept()
        return [{"id": pod["id"], "cost": self.cost(pod), "deadline": kept.get(pod["id"], {}).get("deadline")}
                for pod in pods if pod.get("desiredStatus") != "TERMINATED"]

    def run(self, ident, command, data, timeout):
        try:
            route = self.route(self.request("GET", f"/pods/{ident}"))
        except (urllib.error.URLError, OSError, ValueError):
            route = None
        if route is None:
            raise Unreachable(ident)
        # Each pod keeps its own host key, so that one that takes over an address is not taken for another.
        known = state("runpod") / "known_hosts"
        known.mkdir(parents=True, exist_ok=True, mode=0o700)
        ssh = ["ssh", "-i", self.environment["RUNPOD_SSH_KEY_FILE"], "-p", str(route[1]), "-o", "IdentitiesOnly=yes",
               "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=4", "-o", "StrictHostKeyChecking=accept-new",
               "-o", f"UserKnownHostsFile={known / ident}", f"root@{route[0]}", command]
        try:
            ran = subprocess.run(ssh, input=data, capture_output=True, timeout=timeout, env=self.environment)
        except subprocess.TimeoutExpired:
            raise Unreachable(ident) from None
        if ran.returncode == SSH_FAILED:
            raise Unreachable(ident)
        return ran.returncode, ran.stdout.decode(errors="replace")

    def extend(self, ident, hours):
        pods = self.kept()
        if ident not in pods:
            return None
        pods[ident]["deadline"] = int(time.time() + hours * 3600)
        self.keep(pods)
        return pods[ident]["deadline"]

    def reject(self, ident, reason):
        """RunPod picks the machine, so there is none to avoid."""

    def release(self, ident):
        try:
            pod = self.request("GET", f"/pods/{ident}")
            self.request("DELETE", f"/pods/{ident}")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                pod = None
            else:
                raise
        pods = self.kept()
        pods.pop(ident, None)
        self.keep(pods)
        return None if pod is None else self.cost(pod)

    def sweep(self):
        """Delete every run's pod whose deadline has passed; print each one deleted as a JSON line."""
        now = time.time()
        for ident, kept in list(self.kept().items()):
            if kept["deadline"] < now:
                print(json.dumps({"deleted": ident, "label": kept["label"], "dollars": self.release(ident)}),
                      flush=True)


def main(argv):
    if argv != ["watch"]:
        print("usage: job-runner runpod watch", file=sys.stderr)
        return 2
    provider = RunPodProvider()
    watch("runpod", provider.environment["RUNPOD_API_KEY"], provider.sweep)
