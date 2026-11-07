"""The run's truth, kept in the store beside its files: which of them the store holds.

Nothing else is true about a run's progress. A step is finished when the store
keeps every output its machine listed for it; the next step is the first not
finished; a new machine fetches what the steps left to do read, and the next
step's newest checkpoints.

Each file is under the run's prefix: an output at outputs/PATH, a checkpoint at
checkpoints/STEP/PATH, a log at logs/STEP.log.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Record:
    # path -> {"key", "sha256", "bytes", "step", "at"}
    outputs: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    # step -> {"key", "sha256", "bytes", "at"}
    logs: dict = field(default_factory=dict)
    # step -> the output paths its machine listed when it ended
    steps: dict = field(default_factory=dict)

    @classmethod
    def read(cls, value):
        return cls(**value) if value else cls()

    def write(self):
        return {"outputs": self.outputs, "checkpoints": self.checkpoints, "logs": self.logs, "steps": self.steps}

    @staticmethod
    def key(entry):
        """Where an outbox entry goes, under the run's prefix."""
        if entry["kind"] == "output":
            return f"outputs/{entry['path']}"
        if entry["kind"] == "checkpoint":
            return f"checkpoints/{entry['step']}/{entry['path']}"
        return f"logs/{entry['step']}.log"

    def kept(self, entry, key, at):
        """The record once the store keeps the entry's file at key."""
        item = {"key": key, "sha256": entry["sha256"], "bytes": entry["bytes"], "step": entry["step"], "at": at}
        record = self
        if "outputs" in entry:
            # Sent when the step ended well: all the outputs it left.
            record = replace(record, steps={**record.steps, entry["step"]: sorted(entry["outputs"])})
        if entry["kind"] == "output":
            return replace(record, outputs={**record.outputs, entry["path"]: item})
        if entry["kind"] == "checkpoint":
            # A path's checkpoints go to the store one at a time, newest last.
            return replace(record, checkpoints={**record.checkpoints, entry["path"]: item})
        return replace(record, logs={**record.logs, entry["step"]: item})

    def finished(self, spec):
        return [step.name for step in spec.steps
                if step.name in self.steps and all(path in self.outputs for path in self.steps[step.name])]

    def remaining(self, spec):
        finished = set(self.finished(spec))
        return [step for step in spec.steps if step.name not in finished]

    def fetch(self, spec):
        """What a machine fetches before the steps left: the files and outputs they read, and the next one's checkpoints.

        Each is {"path", "key", "sha256", "bytes"}; a checkpoint also has "if_missing", as one the
        machine saved itself is newer.
        """
        remaining = self.remaining(spec)
        files = {item["path"]: item for item in spec.files}
        wanted = {}
        for step in remaining:
            for path in step.inputs:
                if path in files:
                    wanted[path] = {key: files[path][key] for key in ("path", "key", "sha256", "bytes")}
                for kept, item in self.outputs.items():
                    if kept == path or kept.startswith(f"{path}/"):
                        wanted[kept] = {"path": kept, "key": item["key"], "sha256": item["sha256"],
                                        "bytes": item["bytes"]}
        if remaining:
            for path, item in self.checkpoints.items():
                if item["step"] == remaining[0].name:
                    wanted[path] = {"path": path, "key": item["key"], "sha256": item["sha256"],
                                    "bytes": item["bytes"], "if_missing": True}
        return list(wanted.values())
