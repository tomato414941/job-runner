"""A run's whole life on this machine: the real controller and agent, a local provider and store, a stand-in job."""

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path

from rental.cli import main
from rental.controller import Controller
from rental.local import SIGNATURE, LocalProvider, LocalStore

HERE = Path(__file__).resolve().parent
PREFIX = "runs/rehearsal"
POLICY = {
    "tick": 0.05, "lease": 60.0, "renew": 30.0, "url": 2.0, "unreachable": 0.6, "install": 30.0, "slow": 0.3,
    "warmup": 0.2, "replace": 0.05, "rate_window": 0.4, "sync": 0.3, "send": 10.0, "handover": 5.0,
    "part_bytes": 32 * 1024,
}


def train(name, steps, *extra, inputs=()):
    return {"name": name, "command": ["python3", "job.py", "train", "--output", f"out/{name}", "--steps", str(steps),
                                      *extra],
            "inputs": list(inputs), "outputs": [f"out/{name}/last.pt"], "rate": 1000 / 0.02, "units": steps * 1000}


class RehearsalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = LocalStore(max_put=64 * 1024)
        self.offers = [{"price": 0.5, "speed": 1.0, "count": 5}]
        self.provider = LocalProvider(self.root / "machines", self.offers, shutil.which("uv"))
        corpus = os.urandom(50 * 1024)
        self.store.objects["data/corpus.bin"] = corpus
        self.run_directory = self.root / "run"
        self.run_directory.mkdir()
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            bundle.add(HERE / "job.py", arcname="job.py")
        (self.run_directory / "code.tar.gz").write_bytes(archive.getvalue())
        self.spec = {
            "name": "rehearsal", "provider": {"kind": "local"}, "store": {"kind": "local", "prefix": PREFIX},
            "code": {"path": "code.tar.gz"},
            "setup": [["python3", "job.py", "setup"]],
            "check": [["python3", "job.py", "check"]],
            "files": [{"path": "data/corpus.bin", "key": "data/corpus.bin",
                       "sha256": hashlib.sha256(corpus).hexdigest(), "bytes": len(corpus)}],
            "steps": [
                train("first", 60, "--data", "data/corpus.bin", inputs=["data/corpus.bin"]),
                train("second", 40, "--initialize-from", "out/first/last.pt", inputs=["out/first/last.pt"]),
                {"name": "evaluate", "command": ["python3", "job.py", "evaluate", "--checkpoint", "out/second/last.pt",
                                                 "--output", "out/evaluation.json"],
                 "environment": {"EVALUATED_BY": "rehearsal"},
                 "inputs": ["out/second/last.pt"], "outputs": ["out/evaluation.json"]},
            ],
            "policy": dict(POLICY),
        }

    def tearDown(self):
        self.provider.close()
        self.store.close()
        self.directory.cleanup()

    def controller(self):
        (self.run_directory / "spec.json").write_text(json.dumps(self.spec))
        return Controller(self.run_directory, self.provider, self.store)

    def carry(self, controller, until=None, seconds=60):
        """Tick until the run is over, or until(controller) holds, and give its outcome; fail if neither comes in time."""
        controller.prepare()
        ended = time.monotonic() + seconds
        while time.monotonic() < ended:
            controller.tick()
            if controller.memory.ended or (until and until(controller)):
                return controller.memory.outcome
            time.sleep(controller.spec.policy.tick)
        self.fail("the run did not end in time")

    def events(self):
        return [json.loads(line) for line in (self.run_directory / "events.jsonl").read_text().splitlines()]

    def lengthen_first(self):
        """Give the first step 300 steps, so that it still runs when a test acts on it after its first kept checkpoint,
        however long the commands of the ticks before take."""
        self.spec["steps"][0] = train("first", 300, "--data", "data/corpus.bin", inputs=["data/corpus.bin"])

    def job_log(self, machine, step):
        return self.root / "machines" / machine / ".rental/logs" / f"{step}.log"

    def started(self, machine, step):
        """Whether the step's job on the machine is under way, past setting up what it does on SIGTERM."""
        log = self.job_log(machine, step)
        return log.exists() and '"event": "start"' in log.read_text()

    def assert_run_complete(self):
        self.assertEqual(json.loads(self.store.objects[f"{PREFIX}/outputs/out/evaluation.json"]),
                         {"checkpoint_step": 40, "by": "rehearsal"})
        record = json.loads(self.store.objects[f"{PREFIX}/record.json"])
        self.assertEqual(sorted(record["steps"]), ["evaluate", "first", "second"])
        last = self.store.objects[f"{PREFIX}/outputs/out/first/last.pt"]
        self.assertEqual(hashlib.sha256(last).hexdigest(), record["outputs"]["out/first/last.pt"]["sha256"])
        self.assertEqual(self.provider.machines("rehearsal"), [])

    def assert_no_url_written(self):
        for path in self.root.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".jsonl", ".log", ""):
                self.assertNotIn(SIGNATURE.encode(), path.read_bytes(), path)

    def test_rents_a_machine_runs_every_step_keeps_their_files_and_returns_the_machine(self):
        began = time.monotonic()
        outcome = self.carry(self.controller())

        self.assertEqual(outcome, "complete")
        self.assert_run_complete()
        self.assertEqual(self.provider.rented, ["m1"])
        # The checkpoints are larger than a PUT takes, and went up in parts; the run outlasted any URL.
        self.assertGreater(self.store.joined, 0)
        self.assertGreater(time.monotonic() - began, POLICY["url"])
        self.assertEqual(self.store.refused, [])
        self.assert_no_url_written()

    def test_replaces_a_machine_that_dies_and_takes_the_step_up_from_its_last_kept_checkpoint(self):
        def kill_once_a_checkpoint_is_kept(controller):
            if self.provider.rented == ["m1"] and controller.record.checkpoints and \
                    self.provider.machines_["m1"]["answers"]:
                self.provider.kill("m1")
            return False

        self.lengthen_first()
        outcome = self.carry(self.controller(), until=kill_once_a_checkpoint_is_kept)

        self.assertEqual(outcome, "complete")
        self.assert_run_complete()
        self.assertEqual(self.provider.rented, ["m1", "m2"])
        self.assertEqual(self.provider.rejected, [("m1", "it stopped answering")])
        start = json.loads(self.job_log("m2", "first").read_text().splitlines()[0])
        self.assertGreaterEqual(start["resumed_from"], 10)

    def test_hands_over_from_a_slow_machine_once_its_checkpoint_is_kept(self):
        self.offers[:] = [{"price": 0.4, "speed": 0.25, "count": 1}, {"price": 0.5, "speed": 1.0, "count": 1}]

        outcome = self.carry(self.controller())

        self.assertEqual(outcome, "complete")
        self.assert_run_complete()
        self.assertEqual(self.provider.rented, ["m1", "m2"])
        self.assertTrue(self.provider.rejected[0][1].startswith("slow:"))
        start = json.loads(self.job_log("m2", "first").read_text().splitlines()[0])
        self.assertGreater(start["resumed_from"], 0)

    def test_runs_a_failed_step_again_on_its_machine_then_ends_the_run_failed_with_its_log_and_checkpoint_kept(self):
        self.spec["steps"][0]["command"] += ["--fail-at", "25"]

        outcome = self.carry(self.controller())

        self.assertEqual(outcome, "failed")
        self.assertEqual(self.provider.rented, ["m1"])
        self.assertEqual(self.provider.rejected, [])
        runs = [json.loads(line) for line in self.job_log("m1", "first").read_text().splitlines() if line.startswith("{")]
        self.assertEqual([run["resumed_from"] for run in runs], [0, 20])
        self.assertIn(f"{PREFIX}/logs/first.log", self.store.objects)
        self.assertIn(f"{PREFIX}/checkpoints/first/out/first/last.pt", self.store.objects)
        self.assertEqual(self.events()[-1]["outcome"], "failed")
        self.assertEqual(self.provider.machines("rehearsal"), [])
        self.assert_no_url_written()

    def test_rejects_a_machine_that_loses_its_gpu_and_carries_the_step_on_elsewhere_from_its_last_checkpoint(self):
        def lose_the_gpu_once_a_checkpoint_is_kept(controller):
            if controller.record.checkpoints and not (self.root / "machines/m1/broken").exists():
                self.provider.lose_gpu("m1")
            return False

        self.lengthen_first()
        outcome = self.carry(self.controller(), until=lose_the_gpu_once_a_checkpoint_is_kept)

        self.assertEqual(outcome, "complete")
        self.assert_run_complete()
        [(machine, reason)] = self.provider.rejected
        self.assertEqual(machine, "m1")
        self.assertRegex(reason, r"^it failed its check after first failed: python3 job\.py check ended with 1")
        start = json.loads(self.job_log("m2", "first").read_text().splitlines()[0])
        self.assertGreaterEqual(start["resumed_from"], 10)

    def test_rents_another_machine_when_the_install_fails_and_rejects_the_first_once_the_other_passes_it(self):
        self.offers[:] = [{"price": 0.4, "speed": 1.0, "count": 1, "broken": True},
                          {"price": 0.5, "speed": 1.0, "count": 1}]

        outcome = self.carry(self.controller())

        self.assertEqual(outcome, "complete")
        self.assert_run_complete()
        self.assertEqual(self.provider.rented, ["m1", "m2"])
        self.assertEqual(self.provider.avoided, [(), ("m1",)])
        [(machine, reason)] = self.provider.rejected
        self.assertEqual(machine, "m1")
        self.assertRegex(reason, r"^its install failed: the check failed: python3 job\.py check ended with 1.*"
                                 r", which another machine of the run passed$")

    def test_ends_the_run_stopped_when_asked_once_its_machine_kept_what_it_held(self):
        self.lengthen_first()
        controller = self.controller()
        self.carry(controller, until=lambda controller: self.started("m1", "first"))
        main(["stop", str(self.run_directory), "--reason", "the spec is wrong"])

        outcome = self.carry(controller)

        self.assertEqual((outcome, controller.memory.detail), ("stopped", "the spec is wrong"))
        self.assertEqual((self.provider.rented, self.provider.rejected), (["m1"], []))
        # The job saved its checkpoint when told to stop, and the store keeps it for a run that carries on.
        self.assertIn(f"{PREFIX}/checkpoints/first/out/first/last.pt", self.store.objects)
        self.assertIn(f"{PREFIX}/logs/first.log", self.store.objects)
        self.assertEqual(self.provider.machines("rehearsal"), [])

    def test_carries_the_run_on_when_the_controller_stops_and_starts_again(self):
        first = self.controller()
        self.carry(first, until=lambda controller: "first" in controller.record.finished(controller.spec))

        outcome = self.carry(self.controller())

        self.assertEqual(outcome, "complete")
        self.assert_run_complete()
        self.assertEqual(self.provider.rented, ["m1"])

    def test_stops_before_its_store_key_runs_out_and_carries_on_once_started_again_with_a_new_one(self):
        self.store.expires = time.time() + POLICY["url"] + 1.0

        self.assertFalse(self.controller().run())

        self.store.expires = None
        outcome = self.carry(self.controller())
        self.assertEqual(outcome, "complete")
        self.assert_run_complete()
        self.assertEqual(self.provider.rented, ["m1"])
        self.assertIn("key_running_out", [event["event"] for event in self.events()])

    def test_waits_while_no_offer_fits_and_rents_once_one_does(self):
        self.offers.clear()
        later = threading.Timer(1.0, lambda: self.offers.append({"price": 0.5, "speed": 1.0, "count": 1}))
        later.start()

        outcome = self.carry(self.controller())

        later.cancel()
        self.assertEqual(outcome, "complete")
        names = [event["event"] for event in self.events()]
        self.assertLess(names.index("waiting"), names.index("rented"))
        self.assertEqual(names.count("waiting"), 1)

    def test_the_provider_returns_the_machine_at_its_deadline_with_no_controller_left(self):
        self.spec["policy"].update(lease=1.5, renew=0.5)
        self.carry(self.controller(), until=lambda controller: any(
            machine.state == "working" for machine in controller.memory.machines.values()))

        time.sleep(2.5)

        self.assertEqual(self.provider.machines("rehearsal"), [])
        self.assertEqual(self.provider.machines_["m1"]["returned"], "deadline")


if __name__ == "__main__":
    unittest.main()
