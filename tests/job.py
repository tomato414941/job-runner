"""A job that keeps the run's side of the contract, standing in for training in the rehearsal.

train steps at the speed its machine sets in JOB_RUNNER_SPEED, reports its progress,
saves its checkpoint every few steps and keeps it, takes up from its own
checkpoint when run again, saves and keeps on SIGTERM, and can fail at a step.
evaluate writes the step of the checkpoint it reads, unless it already has.
check passes on a machine whose GPU works; train fails on one whose GPU is lost,
which a rehearsal marks with the file `broken` in the machine's HOME.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import quote

UNITS_PER_STEP = 1000
SECONDS_PER_STEP = 0.02


def keep(path):
    directory = os.environ.get("JOB_RUNNER_KEEP")
    if directory:
        temporary = Path(directory) / f".{quote(str(path), safe='')}.{os.getpid()}"
        os.link(path, temporary)
        os.replace(temporary, Path(directory) / quote(str(path), safe=""))


def gpu_lost():
    return (Path.home() / "broken").exists()


def progress(done, total):
    path = os.environ.get("JOB_RUNNER_PROGRESS")
    if path:
        temporary = Path(f"{path}.{os.getpid()}")
        temporary.write_text(json.dumps({"done": done, "total": total}))
        os.replace(temporary, path)


def save(path, step, size):
    header = (json.dumps({"step": step}) + "\n").encode()
    temporary = path.with_name(f".{path.name}")
    temporary.write_bytes(header + os.urandom(max(0, size - len(header))))
    os.replace(temporary, path)


def saved_step(path):
    return json.loads(Path(path).read_bytes().split(b"\n", 1)[0])["step"]


def train(args):
    if not Path("environment").exists():
        raise SystemExit("the setup did not run")
    checkpoint = Path(args.output) / "last.pt"
    for given in (args.data, args.initialize_from):
        if given and not checkpoint.exists() and not Path(given).exists():
            raise SystemExit(f"{given} is missing")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start = saved_step(checkpoint) if checkpoint.exists() else 0
    speed = float(os.environ.get("JOB_RUNNER_SPEED", "1"))
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    print(json.dumps({"event": "start", "resumed_from": start}), flush=True)
    for step in range(start + 1, args.steps + 1):
        time.sleep(SECONDS_PER_STEP / speed)
        if gpu_lost():
            sys.exit("no CUDA GPUs are available")
        progress(step * UNITS_PER_STEP, args.steps * UNITS_PER_STEP)
        if step == args.fail_at:
            sys.exit(3)
        if stopping:
            save(checkpoint, step, args.checkpoint_bytes)
            keep(checkpoint)
            sys.exit(143)
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(checkpoint, step, args.checkpoint_bytes)
            if step < args.steps:
                keep(checkpoint)


def setup(_args):
    Path("environment").write_text("ready\n")


def check(_args):
    if gpu_lost():
        sys.exit("no CUDA GPUs are available")


def evaluate(args):
    output = Path(args.output)
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"checkpoint_step": saved_step(args.checkpoint),
                                      "by": os.environ.get("EVALUATED_BY")}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    training = commands.add_parser("train")
    training.add_argument("--output", required=True)
    training.add_argument("--steps", type=int, required=True)
    training.add_argument("--checkpoint-every", type=int, default=10)
    training.add_argument("--checkpoint-bytes", type=int, default=100 * 1024)
    training.add_argument("--data")
    training.add_argument("--initialize-from")
    training.add_argument("--fail-at", type=int)
    training.set_defaults(function=train)
    commands.add_parser("setup").set_defaults(function=setup)
    commands.add_parser("check").set_defaults(function=check)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
