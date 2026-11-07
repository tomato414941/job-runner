import unittest

from rental.record import Record
from rental.spec import Spec

SPEC = Spec.read({
    "name": "run", "provider": {}, "store": {"prefix": "runs/run"}, "code": {"path": "code.tar.gz"},
    "files": [{"path": "data/a.bin", "key": "data/a.bin", "sha256": "a", "bytes": 1},
              {"path": "data/b.bin", "key": "data/b.bin", "sha256": "b", "bytes": 2}],
    "steps": [
        {"name": "first", "command": ["x"], "inputs": ["data/a.bin"], "outputs": ["out/first"]},
        {"name": "check", "command": ["x"]},
        {"name": "second", "command": ["x"], "inputs": ["data/b.bin", "out/first"], "outputs": ["out/second.pt"]},
    ],
})


def entry(kind, step, path, outputs=None, at=1.0):
    item = {"id": path, "kind": kind, "step": step, "path": path, "sha256": "s", "bytes": 3, "at": at}
    if outputs is not None:
        item["outputs"] = outputs
    return item


class RecordTests(unittest.TestCase):
    def test_finishes_a_step_once_the_store_keeps_every_output_its_machine_listed(self):
        listed = ["out/first/a.pt", "out/first/b.pt"]
        record = Record().kept(entry("output", "first", "out/first/a.pt", listed), "k", 1.0)
        self.assertEqual(record.finished(SPEC), [])
        record = record.kept(entry("output", "first", "out/first/b.pt", listed), "k", 2.0)
        self.assertEqual(record.finished(SPEC), ["first"])

    def test_finishes_a_step_without_outputs_once_its_log_says_it_ended_well(self):
        failed = Record().kept(entry("log", "check", ".rental/logs/check.log"), "k", 1.0)
        self.assertEqual(failed.finished(SPEC), [])
        ended = Record().kept(entry("log", "check", ".rental/logs/check.log", []), "k", 1.0)
        self.assertEqual(ended.finished(SPEC), ["check"])

    def test_fetches_what_the_steps_left_read_and_the_next_step_checkpoints(self):
        record = Record().kept(entry("checkpoint", "first", "out/first/a.pt"), "runs/run/checkpoints/first/x", 1.0)
        self.assertEqual(
            [(item["path"], item.get("if_missing", False)) for item in record.fetch(SPEC)],
            [("data/a.bin", False), ("data/b.bin", False), ("out/first/a.pt", True)],
        )
        listed = ["out/first/a.pt"]
        record = record.kept(entry("output", "first", "out/first/a.pt", listed), "runs/run/outputs/out/first/a.pt", 2)
        record = record.kept(entry("log", "check", ".rental/logs/check.log", []), "k", 3.0)
        self.assertEqual([(item["path"], item["key"]) for item in record.fetch(SPEC)],
                         [("data/b.bin", "data/b.bin"), ("out/first/a.pt", "runs/run/outputs/out/first/a.pt")])

    def test_puts_outputs_checkpoints_and_logs_each_under_its_own_key(self):
        self.assertEqual(Record.key(entry("output", "first", "out/first/a.pt")), "outputs/out/first/a.pt")
        self.assertEqual(Record.key(entry("checkpoint", "first", "out/first/a.pt")),
                         "checkpoints/first/out/first/a.pt")
        self.assertEqual(Record.key(entry("log", "first", ".rental/logs/first.log")), "logs/first.log")

    def test_reads_back_what_it_wrote(self):
        record = Record().kept(entry("output", "first", "out/first/a.pt", ["out/first/a.pt"]), "k", 1.0)
        self.assertEqual(Record.read(record.write()), record)


class SpecTests(unittest.TestCase):
    def spec(self, steps, policy=None):
        return {"name": "run", "provider": {}, "store": {"prefix": "p"}, "code": {"path": "code.tar.gz"},
                "files": [{"path": "data/a.bin", "key": "data/a.bin", "sha256": "a", "bytes": 1}],
                "steps": steps, "policy": policy or {}}

    def test_refuses_a_step_that_reads_neither_a_file_nor_an_earlier_output(self):
        with self.assertRaisesRegex(ValueError, "late reads data/late.bin"):
            Spec.read(self.spec([{"name": "late", "command": ["x"], "inputs": ["data/late.bin"]}]))

    def test_lets_a_step_read_within_an_earlier_step_output_directory(self):
        spec = Spec.read(self.spec([
            {"name": "decay", "command": ["x"], "outputs": ["out/snapshots"]},
            {"name": "average", "command": ["x"], "inputs": ["out/snapshots/0001.pt"]},
        ]))
        self.assertEqual([step.name for step in spec.steps], ["decay", "average"])

    def test_refuses_a_path_that_leads_out_of_the_code(self):
        with self.assertRaisesRegex(ValueError, "within the code"):
            Spec.read(self.spec([{"name": "a", "command": ["x"], "outputs": ["../escape"]}]))

    def test_refuses_a_policy_field_it_does_not_know(self):
        with self.assertRaisesRegex(ValueError, "no \\['ticks'\\]"):
            Spec.read(self.spec([{"name": "a", "command": ["x"]}], {"ticks": 1}))


if __name__ == "__main__":
    unittest.main()
