"""What a run is: the job's code and setup, the files it starts from, its steps, where it runs, and its limits.

A run lives in a directory whose spec.json holds:

  name      the machines' label, and the run's name in the store
  provider  what to rent, as the provider takes it: {"kind": "vast", "gpu": ..., ...}
  store     where the files go: {"kind": "r2", "bucket": ..., "prefix": ...}
  code      the job's code, a tarball in the run's directory: {"path": "code.tar.gz"}
  setup     commands run in the code on each machine before any step: [[argument, ...], ...]
  files     files already in the store that steps read: [{"path", "key", "sha256", "bytes"}]
  steps     [{"name", "command", "inputs", "outputs", "rate", "units"}], in order: a step reads
            its inputs, files or earlier steps' outputs, and leaves its outputs; rate is
            the units a second its progress is expected to go at, units how many it does
  policy    any of Policy's fields, to change its default

Every path is relative to the code. A spec is checked when it is read: every
input must be a file or an earlier step's output, so a run never starts
without what it reads.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import PurePosixPath

HOUR = 3600.0


@dataclass(frozen=True)
class Policy:
    tick: float = 60.0
    # Machines are rented for `lease` and extended by it once less than `renew` is left.
    lease: float = 6 * HOUR
    renew: float = 3 * HOUR
    # How long a URL for one transfer stays good.
    url: float = 6 * HOUR
    unreachable: float = 15 * 60.0
    install: float = 60 * 60.0
    # A step goes slow below slow_ratio of its rate for `slow`, from `warmup` after it
    # started; its machine is replaced when that saves more than `replace`.
    slow_ratio: float = 0.85
    slow: float = 30 * 60.0
    warmup: float = 30 * 60.0
    replace: float = HOUR
    rate_window: float = 600.0
    # A step's newest checkpoint goes to the store at most every `sync`.
    sync: float = 4 * HOUR
    # An upload the machine has not reported sent by then is authorized again.
    send: float = 3 * HOUR
    send_attempts: int = 5
    handover: float = 30 * 60.0
    part_bytes: int = 1 << 30
    # A job that stops by itself starts again this many times on one machine.
    restarts: int = 2
    # The run fails when a step fails on this many machines, or the setup on this many.
    step_failures: int = 2
    install_failures: int = 2
    budget: float | None = None


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple
    inputs: tuple = ()
    outputs: tuple = ()
    rate: float | None = None
    units: float | None = None


@dataclass(frozen=True)
class Spec:
    name: str
    provider: dict
    store: dict
    code: str
    setup: tuple
    files: tuple
    steps: tuple
    policy: Policy = field(default_factory=Policy)

    @property
    def prefix(self):
        return self.store["prefix"]

    def step(self, name):
        return next((step for step in self.steps if step.name == name), None)

    @classmethod
    def read(cls, value):
        known = {item.name for item in fields(Policy)}
        policy = value.get("policy", {})
        if set(policy) - known:
            raise ValueError(f"the policy has no {sorted(set(policy) - known)}")
        spec = cls(
            name=value["name"], provider=value["provider"], store=value["store"], code=value["code"]["path"],
            setup=tuple(tuple(command) for command in value.get("setup", [])),
            files=tuple(dict(item) for item in value.get("files", [])),
            steps=tuple(
                Step(item["name"], tuple(item["command"]), tuple(item.get("inputs", [])),
                     tuple(item.get("outputs", [])), item.get("rate"), item.get("units"))
                for item in value["steps"]
            ),
            policy=Policy(**policy),
        )
        spec.check()
        return spec

    @classmethod
    def load(cls, path):
        return cls.read(json.loads(path.read_text()))

    def check(self):
        def relative(path):
            parts = PurePosixPath(path).parts
            if not parts or PurePosixPath(path).is_absolute() or ".." in parts:
                raise ValueError(f"{path!r} is not a path within the code")

        names = [step.name for step in self.steps]
        if len(set(names)) != len(names) or any(not name or "/" in name for name in names):
            raise ValueError("steps need names of their own, without /")
        paths = [item["path"] for item in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("two files share a path")
        made = set()
        for step in self.steps:
            for path in (*step.inputs, *step.outputs):
                relative(path)
            for path in step.inputs:
                if path not in paths and not any(path == output or path.startswith(f"{output}/") for output in made):
                    raise ValueError(f"{step.name} reads {path}, which is neither a file nor an earlier output")
            made.update(step.outputs)
        if not self.steps:
            raise ValueError("a run needs a step")
