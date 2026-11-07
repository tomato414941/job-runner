"""What the controller does next, from what it sees, what it remembers and the record, and nothing else.

decide is a pure function of the run's spec and record, the controller's memory,
the world as seen at `now` and the facts its last actions left. It returns the
actions to take and the memory to keep; the controller observes and acts, and a
test gives decide whatever world it likes. An action that did not take effect,
as when its machine did not answer, shows in what decide sees next, and decide
acts on that.

What a machine's agent reports puts the machine in a state:
  installing  nothing of the job there, or its setup running -> out if the setup fails or takes too long
  working     the job installed, its steps fetching, running or ended
                                                             -> handing when slow; out when a step fails
  handing     told to stop, its checkpoint going to store    -> out once kept, or after policy.handover
An installed agent that is idle is started. Only handing is the controller's
own: a machine it hands over stays handing whatever its agent does next. A
machine that does not answer for policy.unreachable goes out from any state.
Out, it is released, and rejected when the fault is the machine's; until the
provider lists it no more, it is leaving. A machine's deadline moves on while
less than policy.renew of it is left.

The run ends complete once the store keeps every step's outputs; failed when a
step or the setup fails on too many machines, renting keeps failing or an
upload keeps failing; over budget once it has cost policy.budget. Ending
releases every machine, and the run is over once the provider lists none.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import dropwhile

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
    """Move the machine's deadline to policy.lease from now."""
    machine: str


@dataclass(frozen=True)
class Release:
    machine: str
    reason: str
    reject: bool


@dataclass(frozen=True)
class End:
    """The run is over: it has its outcome, and the provider lists none of its machines."""
    outcome: str
    detail: str


# What the last actions left, which only the controller saw.


@dataclass(frozen=True)
class NoOffer:
    pass


@dataclass(frozen=True)
class RentFailed:
    """Renting failed otherwise than for want of an offer."""
    detail: str


@dataclass(frozen=True)
class Spent:
    """Dollars paid outside any machine the provider lists, such as for machines that failed the provider's checks."""
    dollars: float


@dataclass(frozen=True)
class KeepFailed:
    """The store does not hold what the machine sent."""
    machine: str
    id: str


@dataclass(frozen=True)
class Released:
    """What the provider says a released machine cost, or None when it did not say."""
    machine: str
    dollars: float | None


@dataclass(frozen=True)
class Seen:
    """A machine as the provider lists it: what it has cost, its deadline, and its agent's report, None unanswered."""
    cost: float
    deadline: float | None
    report: dict | None


# What the controller remembers.


@dataclass
class Machine:
    state: str
    since: float
    cost: float = 0.0
    restarts: int = 0
    unreachable_since: float | None = None
    slow_since: float | None = None
    # Why it is handed over, and whether that is its fault.
    reason: str | None = None
    reject: bool = False


@dataclass
class Leaving:
    """A released machine that the provider still lists."""
    since: float
    cost: float


@dataclass
class Upload:
    """An outbox file's upload, last authorized at `at`; failed when the store did not keep what the machine sent."""
    machine: str
    at: float
    attempts: int
    failed: bool = False


KINDS = {"machines": Machine, "leaving": Leaving, "uploads": Upload}


@dataclass
class Memory:
    # id -> Machine, id -> Leaving, outbox id -> Upload
    machines: dict = field(default_factory=dict)
    leaving: dict = field(default_factory=dict)
    uploads: dict = field(default_factory=dict)
    spent: float = 0.0
    # step -> how many machines it failed on
    failures: dict = field(default_factory=dict)
    install_failures: int = 0
    rent_failures: int = 0
    waiting_since: float | None = None
    outcome: str | None = None
    detail: str | None = None
    ended: bool = False

    def write(self):
        return asdict(self)

    @classmethod
    def read(cls, value):
        if not value:
            return cls()
        return cls(**{**value, **{name: {key: kind(**item) for key, item in value[name].items()}
                                  for name, kind in KINDS.items()}})

    def spending(self):
        return self.spent + sum(item.cost for item in (*self.machines.values(), *self.leaving.values()))


RUNNING = ("fetching", "working")


def decide(spec, record, memory, world, facts, now):
    return Decision(spec, record, deepcopy(memory), now).make(world, facts)


class Decision:
    def __init__(self, spec, record, memory, now):
        self.spec, self.record, self.memory, self.now = spec, record, memory, now
        self.policy, self.actions = spec.policy, []

    def make(self, world, facts):
        memory = self.memory
        if memory.ended:
            return [], memory
        self.learn(facts)
        for ident in [ident for ident in memory.leaving if ident not in world]:
            memory.spent += memory.leaving.pop(ident).cost
        for ident in [ident for ident in memory.machines if ident not in world]:
            # Returned at its deadline, or by hand.
            self.forget(ident)
            memory.spent += memory.machines.pop(ident).cost
        for ident, seen in world.items():
            if ident in memory.leaving:
                self.linger(ident, seen)
            elif memory.outcome:
                # Rented before the run ended, and unknown to the controller until now.
                memory.machines[ident] = Machine("installing", self.now, seen.cost)
                self.release(ident, memory.outcome, False)
            else:
                self.tend(ident, seen)
        if not memory.outcome:
            self.judge()
        if memory.outcome and not memory.machines and not memory.leaving:
            memory.ended = True
            self.actions.append(End(memory.outcome, memory.detail))
        return self.actions, memory

    def learn(self, facts):
        memory = self.memory
        for fact in facts:
            if isinstance(fact, NoOffer):
                memory.waiting_since = memory.waiting_since or self.now
            elif isinstance(fact, RentFailed):
                memory.rent_failures += 1
                if memory.rent_failures >= self.policy.rent_failures:
                    self.end("failed", f"renting failed {memory.rent_failures} times: {fact.detail}")
            elif isinstance(fact, Spent):
                memory.spent += fact.dollars
            elif isinstance(fact, Released) and fact.dollars is not None and fact.machine in memory.leaving:
                memory.leaving[fact.machine].cost = fact.dollars
            elif isinstance(fact, KeepFailed) and fact.id in memory.uploads:
                memory.uploads[fact.id].failed = True

    def judge(self):
        """End the run once it is complete or over its budget; rent a machine when none works on it."""
        memory = self.memory
        if not self.record.remaining(self.spec):
            self.end("complete", "every step's outputs are in the store")
        elif self.policy.budget is not None and memory.spending() >= self.policy.budget:
            self.end("over budget", f"{memory.spending():.2f} dollars spent")
        elif not memory.machines:
            self.actions.append(Rent())

    def end(self, outcome, detail):
        """Give the run its outcome and release its machines; it is over once they are gone."""
        self.memory.outcome, self.memory.detail = outcome, detail
        for ident in list(self.memory.machines):
            self.release(ident, outcome, False)

    # Machines

    def to(self, machine, state):
        machine.state, machine.since = state, self.now

    def forget(self, ident):
        for key in [key for key, upload in self.memory.uploads.items() if upload.machine == ident]:
            del self.memory.uploads[key]

    def release(self, ident, reason, reject):
        self.forget(ident)
        self.memory.leaving[ident] = Leaving(self.now, self.memory.machines.pop(ident).cost)
        self.actions.append(Release(ident, reason, reject))

    def linger(self, ident, seen):
        """Count what a released machine costs while the provider lists it, and ask again when it stays."""
        leaving = self.memory.leaving[ident]
        leaving.cost = max(leaving.cost, seen.cost)
        if self.now - leaving.since > self.policy.unreachable:
            self.actions.append(Release(ident, "still listed after its release", False))
            leaving.since = self.now

    def tend(self, ident, seen):
        memory = self.memory
        if ident not in memory.machines:
            memory.machines[ident] = Machine("installing", self.now)
            memory.waiting_since, memory.rent_failures = None, 0
        machine = memory.machines[ident]
        machine.cost = seen.cost
        report = seen.report
        if report is None:
            machine.unreachable_since = machine.unreachable_since or self.now
            if self.now - machine.unreachable_since >= self.policy.unreachable:
                self.release(ident, "it stopped answering", True)
            return
        machine.unreachable_since = None
        self.tend_outbox(ident, machine, report)
        if memory.outcome or ident not in memory.machines:
            return
        if machine.state == "handing":
            self.hand_over(ident, machine, report)
        else:
            self.follow(ident, machine, report)
        if ident in memory.machines and seen.deadline is not None and seen.deadline - self.now < self.policy.renew:
            self.actions.append(Extend(ident))

    def follow(self, ident, machine, report):
        """Act on what the agent of a machine not handed over says it is doing."""
        phase = report["phase"]
        if phase in ("empty", "installing", "install_failed"):
            if machine.state != "installing":
                self.to(machine, "installing")
            elif self.now - machine.since > self.policy.install:
                self.release(ident, "its setup took too long", True)
                return
            if phase == "empty":
                # A machine without the job, or an agent that started over before its install was done.
                self.actions.append(Install(ident))
            elif phase == "install_failed":
                self.setup_failed(ident)
        elif phase == "ready":
            if machine.state == "working":
                # The start did not arrive, or the agent started over.
                self.restart(ident, machine, "its job is not running")
            else:
                self.actions.append(Start(ident))
                self.to(machine, "working")
        else:
            if machine.state != "working":
                # A machine the controller forgot, with its job under way: take it up where it is.
                self.to(machine, "working")
            self.work(ident, machine, report)

    def work(self, ident, machine, report):
        """A machine whose agent is running the steps, or ended them."""
        phase = report["phase"]
        if phase in RUNNING:
            self.watch_rate(ident, machine, report)
        elif report.get("outbox"):
            # What the job left goes to the store before anything else happens to the machine.
            return
        elif phase == "finished":
            if self.record.remaining(self.spec):
                # Every step it was given ended well, yet the store lacks some of what they left.
                self.restart(ident, machine, "the store lacks what its steps left")
        elif phase == "failed":
            self.step_failed(ident, report["failure"]["step"])
        else:
            self.restart(ident, machine, f"its job stopped by itself ({phase})")

    def restart(self, ident, machine, reason):
        if machine.restarts >= self.policy.restarts:
            self.release(ident, f"{reason}, again", True)
        else:
            machine.restarts += 1
            self.actions.append(Start(ident))

    def setup_failed(self, ident):
        self.memory.install_failures += 1
        if self.memory.install_failures >= self.policy.install_failures:
            self.end("failed", f"the setup failed on {self.memory.install_failures} machines")
        else:
            self.release(ident, "its setup failed", True)

    def step_failed(self, ident, step):
        failures = self.memory.failures[step] = self.memory.failures.get(step, 0) + 1
        if failures >= self.policy.step_failures:
            self.end("failed", f"{step} failed on {failures} machines")
        else:
            self.release(ident, f"{step} failed on it", True)

    def hand_over(self, ident, machine, report):
        """Release a machine handed over once its job has stopped and its files are kept, or when that takes too long."""
        stopped = report["phase"] not in RUNNING and not report.get("outbox")
        if stopped or self.now - machine.since > self.policy.handover:
            self.release(ident, machine.reason, machine.reject)

    def watch_rate(self, ident, machine, report):
        """Hand a machine over when its step goes well below its expected rate, if a new one saves time."""
        step = self.spec.step(report.get("step"))
        progress = report.get("progress") or {}
        rate = progress.get("rate")
        if (not step or not step.rate or rate is None or (report.get("step_seconds") or 0) < self.policy.warmup
                or rate >= self.policy.slow_ratio * step.rate):
            machine.slow_since = None
            return
        machine.slow_since = machine.slow_since or self.now
        if self.now - machine.slow_since < self.policy.slow:
            return
        if self.seconds_left(step, progress) * (step.rate / max(rate, 1e-9) - 1) <= self.policy.replace:
            return
        self.actions.append(Stop(ident))
        self.to(machine, "handing")
        machine.reason, machine.reject = f"slow: {rate:.4g} a second on {step.name}, {step.rate:.4g} expected", True

    def seconds_left(self, step, progress):
        """What the steps left, from this one on, would take at their expected rates."""
        seconds = 0.0
        for item in dropwhile(lambda item: item is not step, self.record.remaining(self.spec)):
            if item is step and progress.get("total"):
                seconds += (progress["total"] - progress["done"]) / step.rate
            elif item.rate and item.units:
                seconds += item.units / item.rate
        return seconds

    # Uploads

    def tend_outbox(self, ident, machine, report):
        """Keep what the machine sent, and authorize what should go now."""
        entries = sorted(report.get("outbox", []), key=lambda entry: entry["at"])
        listed = {entry["id"] for entry in entries}
        for key in [key for key, upload in self.memory.uploads.items() if upload.machine == ident and key not in listed]:
            # The machine forgot it: kept, or replaced by a newer checkpoint.
            del self.memory.uploads[key]
        send, keep = [], []
        for entry in entries:
            upload = self.memory.uploads.get(entry["id"])
            if entry["state"] == "sent" and not (upload and upload.failed):
                keep.append(entry["id"])
            elif self.goes_now(machine, report, entry, entries, upload):
                attempts = upload.attempts if upload else 0
                if attempts >= self.policy.send_attempts:
                    self.end("failed", f"the upload of {entry['path']} failed {attempts} times")
                    return
                self.memory.uploads[entry["id"]] = Upload(ident, self.now, attempts + 1)
                send.append(entry["id"])
        if send:
            self.actions.append(Send(ident, tuple(send)))
        if keep:
            self.actions.append(Keep(ident, tuple(keep)))

    def goes_now(self, machine, report, entry, entries, upload):
        """Whether to authorize the upload of an outbox file that the store does not hold yet."""
        state = entry["state"]
        if upload is None:
            # Never authorized, or authorized by a controller that lost its memory since.
            return state != "ready" or self.due(machine, report, entry, entries)
        if upload.failed or state in ("ready", "failed"):
            # The store lacks what was sent, the authorization did not arrive, or the upload failed.
            return True
        return self.now - upload.at > self.policy.send

    def due(self, machine, report, entry, entries):
        """Whether a file should go: outputs and logs at once; a step's checkpoint one per path at a time,
        at most every policy.sync while its job runs, and at once when it does not."""
        if entry["kind"] != "checkpoint":
            return True
        if any(other["path"] == entry["path"] and other["kind"] == "checkpoint" and other["state"] in ("sending", "sent")
               for other in entries):
            return False
        if machine.state == "handing" or report["phase"] not in RUNNING:
            return True
        kept = self.record.checkpoints.get(entry["path"])
        return not kept or kept["step"] != entry["step"] or self.now - kept["at"] >= self.policy.sync
