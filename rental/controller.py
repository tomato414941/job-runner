"""Carry a run to its end: observe, decide, act, until it is complete, failed or over its budget.

The controller is the run's only authority. It alone holds the store's key: a
machine gets a presigned URL for each transfer when the transfer is due, on
standard input, and only the controller writes the record, the run's truth, to
the store. Its memory and a log of what it did stay in the run's directory,
beside spec.json. Whatever it does not expect stops it; started again, it takes
the run up from the record and its memory, and the machines it no longer watches
go back at their deadlines.
"""

import hashlib
import io
import json
import math
import tarfile
import time
from pathlib import Path

from .decide import End, KeepFailed, Memory, NoOffer, Released, Seen, Spent, decide
from .record import Record
from .spec import Spec

AGENT = '"$HOME/.rental/bootstrap.sh" {command}'
SHIP = 'mkdir -p "$HOME/.rental" && tar -x -C "$HOME/.rental"'
MACHINE_FILES = ("agent.py", "bootstrap.sh")


class Unreachable(Exception):
    """The machine did not answer; a command that ran and failed is not this."""


def digest(path):
    checksum = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1 << 20):
            checksum.update(chunk)
    return checksum.hexdigest(), Path(path).stat().st_size


def etag(parts):
    """The store's ETag for an object sent whole, [[0, MD5]], or in parts, [[number, MD5], ...]."""
    if len(parts) == 1 and parts[0][0] == 0:
        return parts[0][1]
    return f"{hashlib.md5(b''.join(bytes.fromhex(md5) for _, md5 in parts)).hexdigest()}-{len(parts)}"


def machine_files():
    """The agent and its entry, as the tarball the first command unpacks on a machine."""
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        for name in MACHINE_FILES:
            bundle.add(Path(__file__).with_name(name), arcname=name)
    return archive.getvalue()


class Controller:
    def __init__(self, directory, provider, store, clock=time.time, sleep=time.sleep):
        self.directory = Path(directory)
        self.spec = Spec.load(self.directory / "spec.json")
        self.provider, self.store, self.clock, self.sleep = provider, store, clock, sleep
        path = self.directory / "memory.json"
        self.memory = Memory.read(json.loads(path.read_text()) if path.exists() else None)
        saved = store.read(self.key("record.json"))
        self.record = Record.read(json.loads(saved) if saved else None)
        self.facts, self.reports, self.code = [], {}, None

    def key(self, relative):
        return f"{self.spec.prefix}/{relative}"

    def event(self, name, **details):
        with (self.directory / "events.jsonl").open("a") as log:
            log.write(json.dumps({"at": round(self.clock(), 3), "event": name, **details}) + "\n")

    def save(self):
        temporary = self.directory / ".memory.json"
        temporary.write_text(json.dumps(self.memory.write(), indent=1) + "\n")
        temporary.replace(self.directory / "memory.json")

    def prepare(self):
        """Put the code in the store, and check it holds every file the steps read, before renting anything."""
        sha256, size = digest(self.directory / self.spec.code)
        self.code = {"key": self.key(f"code/{sha256}.tar.gz"), "sha256": sha256, "bytes": size}
        if self.store.size(self.code["key"]) != size:
            self.store.upload(self.code["key"], self.directory / self.spec.code)
        missing = [item["path"] for item in self.spec.files if self.store.size(item["key"]) != item["bytes"]]
        if missing:
            raise SystemExit(f"the store lacks what the steps read: {', '.join(missing)}")

    def run(self):
        self.prepare()
        self.event("controlling", finished=self.record.finished(self.spec))
        while not (outcome := self.tick()):
            self.sleep(self.spec.policy.tick)
        return outcome

    def tick(self):
        world = {}
        for row in self.provider.machines(self.spec.name):
            ident = row["id"]
            world[ident] = Seen(row["cost"], None if ident in self.memory.leaving else self.status(ident))
        facts, self.facts = self.facts, []
        actions, self.memory = decide(self.spec, self.record, self.memory, world, facts, self.clock())
        for action in actions:
            getattr(self, type(action).__name__.lower())(action)
        self.save()
        return self.memory.outcome

    # Commands on a machine

    def remote(self, ident, command, data):
        """Run a command on a machine: its exit code and output, or None when the machine does not answer."""
        try:
            return self.provider.run(ident, command, data, 600)
        except Unreachable:
            return None

    def ask(self, ident, command, payload=None):
        """Run an agent command: its answer, or None when the machine did not answer or the agent refused."""
        ran = self.remote(ident, AGENT.format(command=command), None if payload is None else json.dumps(payload).encode())
        if ran is None:
            self.event("unanswered", machine=ident, command=command)
            return None
        code, output = ran
        if command == "status" and code == 127:
            # No agent there yet.
            return {"phase": "new"}
        lines = output.strip().splitlines()
        answer = json.loads(lines[-1]) if lines and lines[-1].startswith("{") else None
        if code or answer is None or "error" in answer:
            self.event("refused", machine=ident, command=command, exit_code=code,
                       error=(answer or {}).get("error"))
            return None
        return answer

    def status(self, ident):
        report = self.ask(ident, "status")
        if report is not None:
            self.reports[ident] = report
        return report

    def entry(self, ident, id_):
        return next(item for item in self.reports[ident]["outbox"] if item["id"] == id_)

    # Actions

    def rent(self, _action):
        ident, dollars = self.provider.rent(self.spec.name, self.spec.provider, self.spec.policy.lease / 3600)
        if dollars:
            self.facts.append(Spent(dollars))
        if ident is None:
            if self.memory.waiting_since is None:
                self.event("waiting", reason="no offer fits")
            self.facts.append(NoOffer())
        else:
            self.event("rented", machine=ident)

    def install(self, action):
        shipped = self.remote(action.machine, SHIP, machine_files())
        if shipped is None or shipped[0]:
            self.event("unanswered" if shipped is None else "refused", machine=action.machine, command="ship")
            return
        url = self.store.presign("GET", self.code["key"], self.spec.policy.url)
        code = {"url": url, "sha256": self.code["sha256"], "bytes": self.code["bytes"]}
        if self.ask(action.machine, "install", {"code": code, "setup": [list(item) for item in self.spec.setup]}):
            self.event("installing", machine=action.machine)

    def start(self, action):
        seconds = self.spec.policy.url
        files = [{**item, "url": self.store.presign("GET", item["key"], seconds)}
                 for item in self.record.fetch(self.spec)]
        steps = [{"name": step.name, "command": list(step.command), "outputs": list(step.outputs)}
                 for step in self.record.remaining(self.spec)]
        if self.ask(action.machine, "start", {"files": files, "steps": steps,
                                              "rate_seconds": self.spec.policy.rate_window}):
            self.event("started", machine=action.machine, steps=[step["name"] for step in steps],
                       files=len(files))

    def send(self, action):
        seconds, uploads = self.spec.policy.url, []
        for id_ in action.ids:
            entry = self.entry(action.machine, id_)
            key = self.key(Record.key(entry))
            if entry["bytes"] <= self.store.max_put:
                uploads.append({"id": id_, "url": self.store.presign("PUT", key, seconds)})
                continue
            size = max(min(self.spec.policy.part_bytes, self.store.max_put), math.ceil(entry["bytes"] / 10_000))
            upload = self.store.start_upload(key)
            uploads.append({"id": id_, "upload": upload, "parts": [
                {"number": number, "start": start, "length": min(size, entry["bytes"] - start),
                 "url": self.store.presign("PUT", key, seconds, {"partNumber": number, "uploadId": upload})}
                for number, start in enumerate(range(0, entry["bytes"], size), 1)
            ]})
        if self.ask(action.machine, "send", {"uploads": uploads}):
            self.event("sending", machine=action.machine, files=len(uploads))

    def keep(self, action):
        kept = []
        for id_ in action.ids:
            entry = self.entry(action.machine, id_)
            key = self.key(Record.key(entry))
            # The ETag names the bytes: an upload already finished, as before a restart, is not finished twice,
            # and an older file at the key is not taken for this one.
            expected = etag(entry["etags"])
            if entry.get("upload") and self.store.etag(key) != expected:
                self.store.finish_upload(key, entry["upload"], entry["etags"])
            if self.store.etag(key) != expected:
                self.event("keep_failed", machine=action.machine, kind=entry["kind"], step=entry["step"])
                self.facts.append(KeepFailed(action.machine, id_))
                continue
            self.record = self.record.kept(entry, key, self.clock())
            kept.append(entry)
        if kept:
            written = json.dumps(self.record.write()).encode()
            self.store.write(self.key("record.json"), written)
            # A copy beside the memory, for `rental status`; the store's is the truth.
            (self.directory / "record.json").write_bytes(written)
            for entry in kept:
                self.event("kept", machine=action.machine, kind=entry["kind"], step=entry["step"],
                           path=entry["path"], bytes=entry["bytes"])
            self.ask(action.machine, "forget", {"ids": [entry["id"] for entry in kept]})

    def stop(self, action):
        self.ask(action.machine, "stop")
        self.event("stopping", machine=action.machine)

    def extend(self, action):
        self.provider.extend(action.machine, self.spec.policy.lease / 3600)
        self.event("extended", machine=action.machine)

    def release(self, action):
        if action.reject:
            self.provider.reject(action.machine, action.reason)
        dollars = self.provider.release(action.machine)
        self.facts.append(Released(action.machine, dollars))
        self.event("released", machine=action.machine, reason=action.reason, rejected=action.reject,
                   dollars=dollars)

    def end(self, action: End):
        self.event("ended", outcome=action.outcome, detail=action.detail,
                   dollars=round(self.memory.spending(), 4))
