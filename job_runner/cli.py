"""Carry a run on rented machines to its end, and see how it goes.

  job-runner check DIRECTORY    what would keep the run whose spec.json is in DIRECTORY from starting now
  job-runner run DIRECTORY      carry the run to its end
  job-runner status DIRECTORY   what the run has finished, what its machines do, and what it has cost
  job-runner wait DIRECTORY     wait until the run is over; exit 0 when it is complete
  job-runner stop DIRECTORY     ask the run to end now: its machines keep what they hold in the store and go back
  job-runner vast COMMAND ...   Vast.ai's machines, as vast/machines.py says: their offers, renting and
                                checking one, the ledger, and the watcher that sends them back
  job-runner hetzner sweep      delete the Hetzner Cloud servers of runs whose deadline has passed

run keeps its memory, a copy of the record and a log of what it did,
events.jsonl, in the directory, and exits 0 once the run is over. It takes the
store's key from AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_ENDPOINT_URL,
with AWS_SESSION_TOKEN and AWS_CREDENTIAL_EXPIRATION for a temporary key, and
reaches the provider's machines with the SSH key in the file VAST_SSH_KEY_FILE
names. It exits 75 when the store's key would expire before a URL it gives, so
it runs best as a service that starts it again with new keys:
  systemd-run --user --unit job-runner-NAME -p Restart=on-failure -p RestartSec=60 -- KEYS job-runner run DIRECTORY
where KEYS is the command that gives it its keys. check needs the same keys and
changes nothing; status and wait read the directory alone. stop writes the
request, which the controller carrying the run takes up at its next look; a new
run with the same spec carries on from what the store keeps.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .controller import STOP, Controller
from .decide import Memory
from .record import Record
from .spec import Spec

# The exit code that asks for the controller to be started again, with new keys.
AGAIN = 75


def outside(spec):
    """The provider and the store the spec names."""
    if spec.provider.get("kind") not in ("vast", "hetzner") or spec.store.get("kind") != "r2":
        raise SystemExit("run knows the providers vast and hetzner and the store r2")
    from .r2 import R2Store

    if spec.provider["kind"] == "hetzner":
        from .hetzner import HetznerProvider

        return HetznerProvider(), R2Store(spec.store["bucket"])
    from .vast import VastProvider

    return VastProvider(), R2Store(spec.store["bucket"])


def controller(directory):
    return Controller(directory, *outside(Spec.load(directory / "spec.json")))


def memory(directory):
    path = directory / "memory.json"
    return Memory.read(json.loads(path.read_text()) if path.exists() else None)


def check(args):
    problems = controller(args.directory).check()
    for problem in problems:
        print(problem)
    return 1 if problems else 0


def run(args):
    if not controller(args.directory).run():
        print("the store's key runs out; start the run again with a new one", file=sys.stderr)
        return AGAIN
    return 0


def status(args):
    spec, remembered = Spec.load(args.directory / "spec.json"), memory(args.directory)
    path = args.directory / "record.json"
    finished = Record.read(json.loads(path.read_text()) if path.exists() else None).finished(spec)
    print(json.dumps({
        "outcome": remembered.outcome, "detail": remembered.detail, "over": remembered.ended,
        "finished": finished, "left": [step.name for step in spec.steps if step.name not in finished],
        "machines": {ident: {"state": machine.state, "dollars": round(machine.cost, 3)}
                     for ident, machine in remembered.machines.items()},
        "leaving": sorted(remembered.leaving), "dollars": round(remembered.spending(), 3),
        "waiting_since": remembered.waiting_since, "stopping": remembered.stopping,
    }, ensure_ascii=False))
    path = args.directory / "events.jsonl"
    for line in (path.read_text().splitlines() if path.exists() else [])[-args.events:]:
        print(line)
    return 0


def stop(args):
    if not (args.directory / "spec.json").exists():
        raise SystemExit(f"{args.directory} holds no run")
    request = args.directory / f".{STOP}"
    request.write_text(json.dumps({"reason": args.reason, "at": round(time.time(), 3)}) + "\n")
    request.replace(args.directory / STOP)
    return 0


def wait(args):
    while not (remembered := memory(args.directory)).ended:
        time.sleep(args.seconds)
    print(json.dumps({"outcome": remembered.outcome, "detail": remembered.detail}, ensure_ascii=False))
    return 0 if remembered.outcome == "complete" else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["hetzner"]:
        from .hetzner import main as hetzner

        return hetzner(argv[1:])
    if argv[:1] == ["vast"]:
        from .vast import MACHINES

        return subprocess.run([sys.executable, str(MACHINES), *argv[1:]]).returncode
    parser = argparse.ArgumentParser(prog="job-runner", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    for name, function in (("check", check), ("run", run), ("status", status), ("wait", wait), ("stop", stop)):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
        command.set_defaults(function=function)
        if name == "status":
            command.add_argument("--events", type=int, default=5, help="how many of the last events to show")
        if name == "wait":
            command.add_argument("--seconds", type=float, default=30, help="how often to look")
        if name == "stop":
            command.add_argument("--reason", default="asked to stop", help="why, as the run's outcome tells it")
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
