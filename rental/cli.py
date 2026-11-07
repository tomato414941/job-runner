"""Carry a run on rented machines to its end, and see how it goes.

  rental run DIRECTORY      carry the run whose spec.json is in DIRECTORY to its end
  rental status DIRECTORY   what the run has finished, what its machines do, and what it has cost
  rental wait DIRECTORY     wait until the run ends or a machine leaves it; exit 0 once it is complete

run keeps its memory, a copy of the record and a log of what it did in the
directory. It needs the provider's and the store's keys in its environment, and
is best run as a service that starts it again if it stops, such as
  systemd-run --user --unit rental-NAME -p Restart=on-failure -p RestartSec=60 -- rental run DIRECTORY
behind whatever gives it its keys. status and wait read the directory alone.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from .controller import Controller
from .record import Record
from .spec import Spec


def outside(spec):
    """The provider and the store the spec names."""
    if spec.provider.get("kind") != "vast" or spec.store.get("kind") != "r2":
        raise SystemExit("run knows the provider vast and the store r2")
    from .r2 import R2Store
    from .vast import VastProvider

    return VastProvider(), R2Store(spec.store["bucket"])


def read(directory, name, default):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else default


def events(directory):
    path = directory / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run(args):
    spec = Spec.load(args.directory / "spec.json")
    provider, store = outside(spec)
    outcome = Controller(args.directory, provider, store).run()
    print(json.dumps({"outcome": outcome}))
    return 0 if outcome == "complete" else 1


def status(args):
    spec = Spec.load(args.directory / "spec.json")
    memory = read(args.directory, "memory.json", {})
    finished = Record.read(read(args.directory, "record.json", None)).finished(spec)
    print(json.dumps({
        "outcome": memory.get("outcome"), "detail": memory.get("detail"),
        "finished": finished, "left": [step.name for step in spec.steps if step.name not in finished],
        "machines": {ident: {"state": machine["state"], "dollars": round(machine["cost"], 3)}
                     for ident, machine in memory.get("machines", {}).items()},
        "dollars": round(memory.get("spent", 0.0) + sum(machine["cost"] for machine in memory.get("machines", {}).values())
                         + sum(item["cost"] for item in memory.get("leaving", {}).values()), 3),
        "waiting_since": memory.get("waiting_since"),
    }, ensure_ascii=False))
    for event in events(args.directory)[-args.events:]:
        print(json.dumps(event, ensure_ascii=False))
    return 0


def wait(args):
    """Exit 0 once the run is complete; 1 when it ends otherwise; 2 when a machine leaves it, with the event."""
    seen = len(events(args.directory))
    while True:
        memory = read(args.directory, "memory.json", {})
        if memory.get("outcome"):
            print(json.dumps({"outcome": memory["outcome"], "detail": memory.get("detail")}))
            return 0 if memory["outcome"] == "complete" else 1
        happened = events(args.directory)
        for event in happened[seen:]:
            if event["event"] == "released":
                print(json.dumps(event, ensure_ascii=False))
                return 2
        seen = len(happened)
        time.sleep(args.seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rental", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    for name, function in (("run", run), ("status", status), ("wait", wait)):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
        command.set_defaults(function=function)
        if name == "status":
            command.add_argument("--events", type=int, default=5, help="how many of the last events to show")
        if name == "wait":
            command.add_argument("--seconds", type=float, default=30, help="how often to look")
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
