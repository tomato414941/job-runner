import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from rental.cli import main


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "spec.json").write_text(json.dumps({
            "name": "run", "provider": {"kind": "vast"}, "store": {"kind": "r2", "prefix": "p"},
            "code": {"path": "code.tar.gz"},
            "steps": [{"name": "first", "command": ["x"], "outputs": ["a"]},
                      {"name": "second", "command": ["x"], "outputs": ["b"]}],
        }))

    def tearDown(self):
        self.directory.cleanup()

    def command(self, *arguments):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = main([*arguments, str(self.root)])
        return code, [json.loads(line) for line in printed.getvalue().splitlines()]

    def test_tells_what_the_run_finished_what_its_machines_do_and_what_it_cost(self):
        (self.root / "record.json").write_text(json.dumps({
            "outputs": {"a": {"key": "k", "sha256": "s", "bytes": 1, "step": "first", "at": 1.0}},
            "steps": {"first": ["a"]}}))
        (self.root / "memory.json").write_text(json.dumps({
            "machines": {"m2": {"state": "working", "cost": 0.5}}, "leaving": {"m1": {"since": 1.0, "cost": 0.25}},
            "spent": 1.0, "outcome": None}))
        (self.root / "events.jsonl").write_text(json.dumps({"at": 1.0, "event": "rented", "machine": "m2"}) + "\n")

        code, (summary, event) = self.command("status")

        self.assertEqual(code, 0)
        self.assertEqual((summary["finished"], summary["left"], summary["dollars"]), (["first"], ["second"], 1.75))
        self.assertEqual(summary["machines"], {"m2": {"state": "working", "dollars": 0.5}})
        self.assertEqual(event["event"], "rented")

    def test_waits_until_a_machine_leaves_the_run_after_the_wait_began(self):
        (self.root / "memory.json").write_text(json.dumps({"outcome": None}))
        (self.root / "events.jsonl").write_text(
            json.dumps({"at": 1.0, "event": "released", "machine": "m0", "reason": "earlier"}) + "\n")

        def release():
            with (self.root / "events.jsonl").open("a") as log:
                log.write(json.dumps({"at": 2.0, "event": "released", "machine": "m1", "reason": "slow"}) + "\n")

        later = threading.Timer(0.2, release)
        later.start()
        code, printed = self.command("wait", "--seconds", "0.05")
        later.join()
        self.assertEqual((code, printed[0]["reason"]), (2, "slow"))

    def test_waits_until_the_run_ends_and_says_whether_it_completed(self):
        (self.root / "memory.json").write_text(json.dumps({"outcome": "complete", "detail": "done"}))
        self.assertEqual(self.command("wait")[0], 0)
        (self.root / "memory.json").write_text(json.dumps({"outcome": "failed", "detail": "first failed"}))
        self.assertEqual(self.command("wait")[0], 1)


if __name__ == "__main__":
    unittest.main()
