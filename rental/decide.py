"""What the controller does next, from what it sees, what it remembers and the record, and nothing else.

decide is a pure function of the run's spec and record, the controller's memory,
the world as seen at `now` and the facts its last actions left. It returns the
actions to take and the memory to keep; the controller observes and acts, and a
test gives decide whatever world it likes.

A machine passes through these states:
  new         rented, with nothing of the job on it    -> install the job
  installing  fetching the code and running its setup   -> ready; out if the setup fails or takes too long
  ready       the job installed and idle                -> start the steps left
  working     fetching what they read, running them     -> handing when slow; out when a step fails
  handing     told to stop; its checkpoint goes to store -> out once kept, or after policy.handover
A machine that does not answer for policy.unreachable goes out from any state.
Out, it is released, and rejected when the fault is the machine's; until the
provider lists it no more, it is leaving.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field

# What the controller does.


@dataclass(frozen=True)
class Rent:
    pass


@dataclass(frozen=True)
class Install:
    machine: str


@dataclass(frozen=True)
class Start:
    machine: str


@dataclass(frozen=True)
class Send:
    """Authorize these outbox files' uploads."""
    machine: str
    ids: tuple


@dataclass(frozen=True)
class Keep:
    """Complete and check these sent files, record them, and let the machine forget them."""
    machine: str
    ids: tuple


@dataclass(frozen=True)
class Stop:
    machine: str


@dataclass(frozen=True)
class Extend:
    machine: str


@dataclass(frozen=True)
class Release:
    machine: str
    reason: str
    reject: bool


@dataclass(frozen=True)
class End:
    outcome: str
    detail: str


# What the last actions left, which only the controller saw.


@dataclass(frozen=True)
class NoOffer:
    pass


@dataclass(frozen=True)
class Spent:
    """Dollars paid outside any machine the controller saw, such as for machines that failed the provider's checks."""
    dollars: float


@dataclass(frozen=True)
class KeepFailed:
    machine: str
    id: str


@dataclass(frozen=True)
class Released:
    machine: str
    dollars: float | None


@dataclass(frozen=True)
class Seen:
    """A machine as the provider lists it, and its agent's report, or None when it did not answer."""
    cost: float
    report: dict | None


@dataclass
class Machine:
    state: str
    since: float
    renewed: float
    cost: float = 0.0
    restarts: int = 0
    unreachable_since: float | None = None
    slow_since: float | None = None
    reason: str | None = None
    reject: bool = False


@dataclass
class Memory:
    machines: dict = field(default_factory=dict)
    # Released machines the provider still lists: id -> {"since", "cost"}.
    leaving: dict = field(default_factory=dict)
    # Uploads authorized: outbox id -> {"machine", "at", "attempts", "failed"}.
    uploads: dict = field(default_factory=dict)
    spent: float = 0.0
    # step -> how many machines it failed on.
    failures: dict = field(default_factory=dict)
    install_failures: int = 0
    waiting_since: float | None = None
    outcome: str | None = None
    detail: str | None = None

    def write(self):
        return asdict(self)

    @classmethod
    def read(cls, value):
        if not value:
            return cls()
        return cls(**{**value, "machines": {key: Machine(**item) for key, item in value["machines"].items()}})

    def spending(self):
        return (self.spent + sum(machine.cost for machine in self.machines.values())
                + sum(item["cost"] for item in self.leaving.values()))


ACTIVE = ("installing", "ready", "working", "handing")
RUNNING = ("fetching", "working")


def decide(spec, record, memory, world, facts, now):
    return Decision(spec, record, deepcopy(memory), now).make(world, facts)


class Decision:
    def __init__(self, spec, record, memory, now):
        self.spec, self.record, self.memory, self.now = spec, record, memory, now
        self.policy, self.actions = spec.policy, []

    def make(self, world, facts):
        memory = self.memory
        if memory.outcome:
            return [], memory
        for fact in facts:
            if isinstance(fact, NoOffer):
                memory.waiting_since = memory.waiting_since or self.now
            elif isinstance(fact, Spent):
                memory.spent += fact.dollars
            elif isinstance(fact, Released) and fact.dollars is not None and fact.machine in memory.leaving:
                memory.leaving[fact.machine]["cost"] = fact.dollars
            elif isinstance(fact, KeepFailed) and fact.id in memory.uploads:
                memory.uploads[fact.id]["failed"] = True
        for ident, item in list(memory.leaving.items()):
            if ident not in world:
                memory.spent += memory.leaving.pop(ident)["cost"]
            elif self.now - item["since"] > self.policy.unreachable:
                # The provider did not take it back; ask again.
                self.actions.append(Release(ident, "still listed after its release", False))
                item["since"] = self.now
        for ident in [ident for ident in memory.machines if ident not in world]:
            # Returned at its deadline, or by hand.
            memory.spent += memory.machines.pop(ident).cost
        for ident, seen in world.items():
            if ident in memory.leaving:
                memory.leaving[ident]["cost"] = max(memory.leaving[ident]["cost"], seen.cost)
                continue
            if ident not in memory.machines:
                memory.machines[ident] = Machine("new", self.now, self.now)
                memory.waiting_since = None
            memory.machines[ident].cost = seen.cost
            self.tend(ident, memory.machines[ident], seen.report)
            if memory.outcome:
                return self.actions, memory
        if not self.record.remaining(self.spec):
            self.end("complete", "every step's outputs are in the store")
        elif self.policy.budget is not None and memory.spending() >= self.policy.budget:
            self.end("over budget", f"{memory.spending():.2f} dollars spent")
        elif not memory.machines:
            self.actions.append(Rent())
        return self.actions, memory

    # Machines

    def to(self, machine, state):
        machine.state, machine.since = state, self.now

    def release(self, ident, reason, reject):
        machine = self.memory.machines.pop(ident)
        self.memory.leaving[ident] = {"since": self.now, "cost": machine.cost}
        self.actions.append(Release(ident, reason, reject))

    def end(self, outcome, detail):
        for ident in list(self.memory.machines):
            self.release(ident, outcome, False)
        self.memory.outcome, self.memory.detail = outcome, detail
        self.actions.append(End(outcome, detail))

    def start(self, ident, machine):
        self.actions.append(Start(ident))
        self.to(machine, "working")

    def restart(self, ident, machine, reason):
        if machine.restarts >= self.policy.restarts:
            self.release(ident, f"{reason}, again", True)
            return
        machine.restarts += 1
        self.start(ident, machine)

    def tend(self, ident, machine, report):
        if report is None:
            machine.unreachable_since = machine.unreachable_since or self.now
            if self.now - machine.unreachable_since >= self.policy.unreachable:
                self.release(ident, "it stopped answering", True)
            return
        machine.unreachable_since = None
        phase = report["phase"]
        outbox = report.get("outbox", [])
        self.tend_outbox(ident, machine, report)
        if self.memory.outcome or ident not in self.memory.machines:
            return
        if machine.state == "new":
            if phase in ("new", "empty"):
                self.actions.append(Install(ident))
                self.to(machine, "installing")
                return
            else:
                # A machine the controller forgot, with the job on it: take it up where it is.
                self.to(machine, {"installing": "installing", "install_failed": "installing",
                                  "ready": "ready"}.get(phase, "working"))
        if machine.state == "installing":
            if phase == "ready":
                self.to(machine, "ready")
            elif phase == "install_failed":
                self.memory.install_failures += 1
                if self.memory.install_failures >= self.policy.install_failures:
                    self.end("failed", f"the setup failed on {self.memory.install_failures} machines")
                else:
                    self.release(ident, "its setup failed", True)
                return
            elif phase in ("new", "empty"):
                # The agent started over before it was done.
                self.actions.append(Install(ident))
            elif self.now - machine.since > self.policy.install:
                self.release(ident, "its setup took too long", True)
                return
        if machine.state == "ready":
            if phase == "ready":
                self.start(ident, machine)
            elif phase in ("new", "empty"):
                self.to(machine, "new")
            else:
                self.to(machine, "working")
        elif machine.state == "working":
            self.work(ident, machine, report, outbox)
        elif machine.state == "handing":
            if (phase not in RUNNING and not outbox) or self.now - machine.since > self.policy.handover:
                self.release(ident, machine.reason, machine.reject)
                return
        if ident in self.memory.machines and machine.state in ACTIVE:
            if self.now - machine.renewed >= self.policy.lease - self.policy.renew:
                self.actions.append(Extend(ident))
                machine.renewed = self.now

    def work(self, ident, machine, report, outbox):
        phase = report["phase"]
        if phase in RUNNING:
            self.watch_rate(ident, machine, report)
        elif phase == "finished":
            # Every step it was given ended, yet the store lacks some of what they left.
            if not outbox and self.record.remaining(self.spec):
                self.restart(ident, machine, "the store lacks what its steps left")
        elif phase == "failed":
            if not outbox:
                step = report["failure"]["step"]
                failures = self.memory.failures[step] = self.memory.failures.get(step, 0) + 1
                if failures >= self.policy.step_failures:
                    self.end("failed", f"{step} failed on {failures} machines")
                else:
                    self.release(ident, f"{step} failed on it", True)
        elif phase in ("new", "empty"):
            self.to(machine, "new")
        else:
            self.restart(ident, machine, f"its job stopped by itself ({phase})")

    def watch_rate(self, ident, machine, report):
        """Hand a machine over when its step goes well below its expected rate, if a new one saves time."""
        step = self.spec.step(report.get("step"))
        progress = report.get("progress") or {}
        rate = progress.get("rate")
        if (not step or not step.rate or rate is None or report.get("step_seconds", 0) < self.policy.warmup
                or rate >= self.policy.slow_ratio * step.rate):
            machine.slow_since = None
            return
        machine.slow_since = machine.slow_since or self.now
        if self.now - machine.slow_since < self.policy.slow:
            return
        # What the steps left would take at their expected rates, and what this machine adds to it.
        remaining = self.record.remaining(self.spec)
        expected = 0.0
        for item in remaining[[other.name for other in remaining].index(step.name):]:
            if item is step and progress.get("total"):
                expected += (progress["total"] - progress["done"]) / step.rate
            elif item.rate and item.units:
                expected += item.units / item.rate
        if expected * (step.rate / max(rate, 1e-9) - 1) <= self.policy.replace:
            return
        self.actions.append(Stop(ident))
        self.to(machine, "handing")
        machine.reason, machine.reject = f"slow: {rate:.4g} a second on {step.name}, {step.rate:.4g} expected", True

    # Uploads

    def due(self, machine, report, entry, entries):
        """Whether an outbox file should go now: a checkpoint at most every policy.sync, one per path at a time."""
        if entry["kind"] != "checkpoint":
            return True
        if any(other["path"] == entry["path"] and other["kind"] == "checkpoint" and other["state"] in ("sending", "sent")
               for other in entries):
            return False
        if machine.state == "handing" or report["phase"] in ("failed", "stopped"):
            return True
        kept = self.record.checkpoints.get(entry["path"])
        return not kept or kept["step"] != entry["step"] or self.now - kept["at"] >= self.policy.sync

    def tend_outbox(self, ident, machine, report):
        entries = sorted(report.get("outbox", []), key=lambda entry: entry["at"])
        listed = {entry["id"] for entry in entries}
        for upload in [key for key, item in self.memory.uploads.items() if item["machine"] == ident and key not in listed]:
            del self.memory.uploads[upload]
        send, keep = [], []
        for entry in entries:
            flight = self.memory.uploads.get(entry["id"])
            state = entry["state"]
            if state == "sent" and not (flight and flight.get("failed")):
                keep.append(entry["id"])
                continue
            if state == "ready" and not flight:
                if not self.due(machine, report, entry, entries):
                    continue
            elif not (state == "failed" or (flight and flight.get("failed"))
                      or (state == "sending" and (not flight or self.now - flight["at"] > self.policy.send))):
                continue
            attempts = (flight or {}).get("attempts", 0)
            if attempts >= self.policy.send_attempts:
                self.end("failed", f"the upload of {entry['path']} failed {attempts} times")
                return
            self.memory.uploads[entry["id"]] = {"machine": ident, "at": self.now, "attempts": attempts + 1,
                                                "failed": False}
            send.append(entry["id"])
        if send:
            self.actions.append(Send(ident, tuple(send)))
        if keep:
            self.actions.append(Keep(ident, tuple(keep)))
