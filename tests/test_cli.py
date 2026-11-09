import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from job_runner.cli import main
from job_runner.decide import Leaving, Machine, Memory


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

    def remember(self, memory):
        (self.root / "memory.json").write_text(json.dumps(memory.write()))

    def command(self, *arguments):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = main([*arguments, str(self.root)])
        return code, [json.loads(line) for line in printed.getvalue().splitlines()]

    def test_tells_what_the_run_finished_what_its_machines_do_and_what_it_cost(self):
        (self.root / "record.json").write_text(json.dumps({
            "outputs": {"a": {"key": "k", "sha256": "s", "bytes": 1, "step": "first", "at": 1.0}},
            "steps": {"first": ["a"]}}))
        self.remember(Memory(machines={"m2": Machine("working", 1.0, cost=0.5)}, leaving={"m1": Leaving(1.0, 0.25)},
                             spent=1.0))
        (self.root / "events.jsonl").write_text(json.dumps({"at": 1.0, "event": "rented", "machine": "m2"}) + "\n")

        code, (summary, event) = self.command("status")

        self.assertEqual(code, 0)
        self.assertEqual((summary["finished"], summary["left"], summary["dollars"]), (["first"], ["second"], 1.75))
        self.assertEqual((summary["machines"], summary["leaving"]), ({"m2": {"state": "working", "dollars": 0.5}},
                                                                     ["m1"]))
        self.assertEqual(event["event"], "rented")

    def test_waits_until_the_run_is_over_with_its_machines_gone(self):
        self.remember(Memory(outcome="complete", detail="done", leaving={"m1": Leaving(1.0, 0.25)}))
        later = threading.Timer(0.2, self.remember, [Memory(outcome="complete", detail="done", ended=True)])
        later.start()
        code, printed = self.command("wait", "--seconds", "0.05")
        later.join()
        self.assertEqual((code, printed), (0, [{"outcome": "complete", "detail": "done"}]))

    def test_asks_the_run_to_stop_with_the_reason_given(self):
        self.assertEqual(main(["stop", str(self.root), "--reason", "the spec is wrong"]), 0)
        self.assertEqual(json.loads((self.root / "stop").read_text())["reason"], "the spec is wrong")
        with self.assertRaisesRegex(SystemExit, "holds no run"):
            main(["stop", str(self.root / "elsewhere")])

    def test_says_by_its_exit_code_whether_the_run_completed(self):
        self.remember(Memory(outcome="failed", detail="first failed on 2 machines", ended=True))
        self.assertEqual(self.command("wait")[0], 1)


if __name__ == "__main__":
    unittest.main()
