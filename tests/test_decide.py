import unittest

from rental.decide import (
    End, Extend, Install, Keep, KeepFailed, Machine, Memory, Release, Released, Rent, Seen, Send, Start, Stop, decide,
)
from rental.record import Record
from rental.spec import Spec

NOW = 100_000.0
SPEC = Spec.read({
    "name": "run", "provider": {}, "store": {"prefix": "runs/run"}, "code": {"path": "code.tar.gz"},
    "files": [{"path": "data/corpus.bin", "key": "data/corpus.bin", "sha256": "d", "bytes": 1}],
    "steps": [
        {"name": "first", "command": ["train"], "inputs": ["data/corpus.bin"], "outputs": ["out/first/last.pt"],
         "rate": 100.0, "units": 1_000_000},
        {"name": "second", "command": ["evaluate"], "inputs": ["out/first/last.pt"], "outputs": ["out/score.json"]},
    ],
    "policy": {"budget": 10.0},
})
POLICY = SPEC.policy


def report(phase, step=None, outbox=(), rate=None, done=0, total=None, step_seconds=None, failure=None):
    return {"phase": phase, "step": step, "outbox": list(outbox), "failure": failure, "step_seconds": step_seconds,
            "progress": None if rate is None else {"done": done, "total": total, "rate": rate}}


def entry(ident, kind="output", state="ready", path="out/first/last.pt", step="first", at=1.0, outputs=None):
    item = {"id": ident, "kind": kind, "state": state, "path": path, "step": step, "at": at, "bytes": 10,
            "sha256": "s", "md5": "m"}
    if outputs is not None:
        item["outputs"] = outputs
    return item


def finished(*steps):
    record = Record()
    for name in steps:
        for output in SPEC.step(name).outputs:
            record = record.kept(entry(name + output, step=name, path=output, outputs=[output]), output, NOW)
    return record


def working(**changes):
    return Memory(machines={"m1": Machine(**{"state": "working", "since": NOW - 60, "renewed": NOW - 60, **changes})})


class DecideTests(unittest.TestCase):
    def decide(self, world, memory=None, record=None, facts=(), spec=SPEC):
        return decide(spec, record or Record(), memory or Memory(), world, list(facts), NOW)

    def test_rents_a_machine_when_none_works_on_steps_left(self):
        actions, _ = self.decide({})
        self.assertEqual(actions, [Rent()])

    def test_rents_nothing_more_while_a_machine_works(self):
        actions, _ = self.decide({"m1": Seen(0.1, report("working", "first"))}, working())
        self.assertEqual(actions, [])

    def test_installs_the_job_on_a_new_machine(self):
        actions, memory = self.decide({"m1": Seen(0.0, {"phase": "new"})})
        self.assertEqual(actions, [Install("m1")])
        self.assertEqual(memory.machines["m1"].state, "installing")

    def test_takes_up_a_forgotten_machine_where_its_job_is_without_installing_it_again(self):
        actions, memory = self.decide({"m1": Seen(0.0, report("working", "first"))})
        self.assertEqual(actions, [])
        self.assertEqual(memory.machines["m1"].state, "working")

    def test_starts_the_steps_left_once_the_job_is_installed(self):
        actions, memory = self.decide({"m1": Seen(0.0, report("ready"))}, working(state="installing"))
        self.assertEqual(actions, [Start("m1")])
        self.assertEqual(memory.machines["m1"].state, "working")

    def test_waits_a_while_for_a_machine_that_does_not_answer(self):
        actions, memory = self.decide({"m1": Seen(0.1, None)}, working(unreachable_since=NOW - 1))
        self.assertEqual(actions, [])
        self.assertIn("m1", memory.machines)

    def test_releases_and_rejects_a_machine_that_stopped_answering_and_rents_another(self):
        actions, memory = self.decide({"m1": Seen(0.1, None)}, working(unreachable_since=NOW - POLICY.unreachable))
        self.assertEqual(actions, [Release("m1", "it stopped answering", True), Rent()])
        self.assertIn("m1", memory.leaving)

    def test_extends_a_machine_before_its_lease_runs_short(self):
        memory = working(renewed=NOW - (POLICY.lease - POLICY.renew))
        actions, memory = self.decide({"m1": Seen(0.1, report("working", "first"))}, memory)
        self.assertEqual(actions, [Extend("m1")])
        self.assertEqual(memory.machines["m1"].renewed, NOW)

    def test_sends_outputs_and_logs_at_once_and_keeps_what_the_machine_sent(self):
        outbox = [entry("o1"), entry("l1", kind="log", state="sent")]
        actions, memory = self.decide({"m1": Seen(0.1, report("working", "first", outbox))}, working())
        self.assertEqual(actions, [Send("m1", ("o1",)), Keep("m1", ("l1",))])
        self.assertEqual(memory.uploads["o1"]["attempts"], 1)

    def test_sends_a_step_checkpoint_at_most_every_sync_interval(self):
        kept = Record(checkpoints={"out/first/last.pt": {"key": "k", "sha256": "s", "bytes": 10, "step": "first",
                                                         "at": NOW - 10}})
        outbox = [entry("c1", kind="checkpoint")]
        actions, _ = self.decide({"m1": Seen(0.1, report("working", "first", outbox))}, working(), kept)
        self.assertEqual(actions, [])
        kept.checkpoints["out/first/last.pt"]["at"] = NOW - POLICY.sync
        actions, _ = self.decide({"m1": Seen(0.1, report("working", "first", outbox))}, working(), kept)
        self.assertEqual(actions, [Send("m1", ("c1",))])

    def test_sends_one_checkpoint_of_a_path_at_a_time(self):
        outbox = [entry("c1", kind="checkpoint", state="sending"), entry("c2", kind="checkpoint", at=2.0)]
        memory = working()
        memory.uploads["c1"] = {"machine": "m1", "at": NOW - 5, "attempts": 1, "failed": False}
        actions, _ = self.decide({"m1": Seen(0.1, report("working", "first", outbox))}, memory)
        self.assertEqual(actions, [])

    def test_sends_a_checkpoint_at_once_from_a_machine_handing_over(self):
        kept = Record(checkpoints={"out/first/last.pt": {"key": "k", "sha256": "s", "bytes": 10, "step": "first",
                                                         "at": NOW - 10}})
        outbox = [entry("c1", kind="checkpoint")]
        actions, _ = self.decide({"m1": Seen(0.1, report("stopped", "first", outbox))}, working(state="handing"), kept)
        self.assertEqual(actions, [Send("m1", ("c1",))])

    def test_sends_again_an_upload_that_failed_and_ends_the_run_when_it_keeps_failing(self):
        outbox = [entry("o1", state="failed")]
        memory = working()
        memory.uploads["o1"] = {"machine": "m1", "at": NOW - 5, "attempts": 1, "failed": False}
        actions, memory = self.decide({"m1": Seen(0.1, report("working", "first", outbox))}, memory)
        self.assertEqual(actions, [Send("m1", ("o1",))])
        memory.uploads["o1"]["attempts"] = POLICY.send_attempts
        actions, memory = self.decide({"m1": Seen(0.1, report("working", "first", outbox))}, memory)
        self.assertEqual(actions[-1], End("failed", f"the upload of out/first/last.pt failed {POLICY.send_attempts} times"))
        self.assertEqual(memory.outcome, "failed")

    def test_sends_again_an_upload_the_store_did_not_keep_although_the_machine_sent_it(self):
        outbox = [entry("o1", state="sent")]
        memory = working()
        memory.uploads["o1"] = {"machine": "m1", "at": NOW - 5, "attempts": 1, "failed": False}
        actions, _ = self.decide({"m1": Seen(0.1, report("working", "first", outbox))}, memory,
                                 facts=[KeepFailed("m1", "o1")])
        self.assertEqual(actions, [Send("m1", ("o1",))])

    def test_sends_again_an_upload_unheard_of_for_too_long(self):
        outbox = [entry("o1", state="sending")]
        memory = working()
        memory.uploads["o1"] = {"machine": "m1", "at": NOW - POLICY.send - 1, "attempts": 1, "failed": False}
        actions, _ = self.decide({"m1": Seen(0.1, report("working", "first", outbox))}, memory)
        self.assertEqual(actions, [Send("m1", ("o1",))])

    def slow(self, done=0, total=1_000_000, step_seconds=POLICY.warmup, slow_since=NOW - POLICY.slow):
        seen = Seen(0.1, report("working", "first", rate=30.0, done=done, total=total, step_seconds=step_seconds))
        return self.decide({"m1": seen}, working(slow_since=slow_since))

    def test_hands_over_a_machine_well_below_its_step_rate_when_a_new_one_saves_time(self):
        actions, memory = self.slow()
        self.assertEqual(actions, [Stop("m1")])
        self.assertEqual((memory.machines["m1"].state, memory.machines["m1"].reject), ("handing", True))

    def test_bears_with_a_slow_machine_near_the_end_of_the_steps(self):
        actions, _ = self.slow(done=999_990)
        self.assertEqual(actions, [])

    def test_bears_with_a_slow_start_during_the_warmup(self):
        actions, memory = self.slow(step_seconds=POLICY.warmup - 1)
        self.assertEqual(actions, [])
        self.assertIsNone(memory.machines["m1"].slow_since)

    def test_notes_when_a_machine_first_goes_slow(self):
        actions, memory = self.slow(slow_since=None)
        self.assertEqual(actions, [])
        self.assertEqual(memory.machines["m1"].slow_since, NOW)

    def test_releases_a_machine_handing_over_once_its_job_stopped_and_its_files_are_kept(self):
        memory = working(state="handing", reason="slow", reject=True)
        actions, _ = self.decide({"m1": Seen(0.1, report("stopped", "first"))}, memory)
        self.assertEqual(actions, [Release("m1", "slow", True), Rent()])

    def test_releases_a_machine_handing_over_when_the_handover_takes_too_long(self):
        memory = working(state="handing", since=NOW - POLICY.handover - 1, reason="slow", reject=True)
        actions, _ = self.decide({"m1": Seen(0.1, report("working", "first"))}, memory)
        self.assertEqual(actions, [Release("m1", "slow", True), Rent()])

    def test_rejects_a_machine_a_step_failed_on_once_its_files_are_kept(self):
        failed = report("failed", "first", failure={"step": "first", "exit_code": 3})
        actions, memory = self.decide({"m1": Seen(0.1, failed)}, working())
        self.assertEqual(actions, [Release("m1", "first failed on it", True), Rent()])
        self.assertEqual(memory.failures, {"first": 1})

    def test_keeps_a_failed_step_files_before_releasing_its_machine(self):
        failed = report("failed", "first", [entry("l1", kind="log")], failure={"step": "first", "exit_code": 3})
        actions, _ = self.decide({"m1": Seen(0.1, failed)}, working())
        self.assertEqual(actions, [Send("m1", ("l1",))])

    def test_ends_the_run_when_a_step_fails_on_a_second_machine(self):
        memory = working()
        memory.failures["first"] = 1
        failed = report("failed", "first", failure={"step": "first", "exit_code": 3})
        actions, memory = self.decide({"m1": Seen(0.1, failed)}, memory)
        self.assertEqual(actions, [Release("m1", "failed", False), End("failed", "first failed on 2 machines")])

    def test_starts_again_a_job_that_stopped_by_itself_and_gives_up_its_machine_after_too_many_times(self):
        actions, memory = self.decide({"m1": Seen(0.1, report("stopped", "first"))}, working())
        self.assertEqual(actions, [Start("m1")])
        self.assertEqual(memory.machines["m1"].restarts, 1)
        actions, _ = self.decide({"m1": Seen(0.1, report("stopped", "first"))}, working(restarts=POLICY.restarts))
        self.assertEqual(actions, [Release("m1", "its job stopped by itself (stopped), again", True), Rent()])

    def test_starts_again_when_the_store_lacks_what_finished_steps_left(self):
        actions, _ = self.decide({"m1": Seen(0.1, report("finished"))}, working())
        self.assertEqual(actions, [Start("m1")])

    def test_rejects_a_machine_whose_setup_failed_and_ends_the_run_when_it_fails_again(self):
        actions, memory = self.decide({"m1": Seen(0.1, report("install_failed"))}, working(state="installing"))
        self.assertEqual(actions, [Release("m1", "its setup failed", True), Rent()])
        memory.machines["m2"] = Machine("installing", NOW, NOW)
        actions, memory = self.decide({"m2": Seen(0.1, report("install_failed"))}, memory)
        self.assertEqual(actions[-1], End("failed", "the setup failed on 2 machines"))

    def test_releases_a_machine_whose_setup_takes_too_long(self):
        memory = working(state="installing", since=NOW - POLICY.install - 1)
        actions, _ = self.decide({"m1": Seen(0.1, report("installing"))}, memory)
        self.assertEqual(actions, [Release("m1", "its setup took too long", True), Rent()])

    def test_ends_the_run_complete_once_the_store_keeps_every_step_outputs(self):
        actions, memory = self.decide({"m1": Seen(0.1, report("finished"))}, working(),
                                      finished("first", "second"))
        self.assertEqual(actions, [Release("m1", "complete", False),
                                   End("complete", "every step's outputs are in the store")])
        self.assertEqual(memory.outcome, "complete")

    def test_ends_the_run_over_its_budget(self):
        memory = working()
        memory.spent = 9.5
        actions, _ = self.decide({"m1": Seen(0.6, report("working", "first"))}, memory)
        self.assertEqual(actions, [Release("m1", "over budget", False), End("over budget", "10.10 dollars spent")])

    def test_counts_what_a_machine_cost_once_the_provider_no_longer_lists_it(self):
        actions, memory = self.decide({}, working(cost=1.5))
        self.assertEqual(actions, [Rent()])
        self.assertEqual(memory.spent, 1.5)

    def test_counts_the_dollars_the_provider_reports_for_a_released_machine(self):
        memory = Memory(leaving={"m1": {"since": NOW - 1, "cost": 1.0}})
        _, memory = self.decide({}, memory, facts=[Released("m1", 1.2)])
        self.assertEqual(memory.spent, 1.2)

    def test_asks_again_to_release_a_machine_the_provider_still_lists(self):
        memory = Memory(leaving={"m1": {"since": NOW - POLICY.unreachable - 1, "cost": 1.0}})
        actions, _ = self.decide({"m1": Seen(1.0, None)}, memory)
        self.assertEqual(actions, [Release("m1", "still listed after its release", False), Rent()])

    def test_does_nothing_once_the_run_has_ended(self):
        actions, _ = self.decide({}, Memory(outcome="complete"))
        self.assertEqual(actions, [])

    def test_remembers_what_it_decided_across_a_restart(self):
        _, memory = self.decide({"m1": Seen(0.0, {"phase": "new"})})
        self.assertEqual(Memory.read(memory.write()), memory)


if __name__ == "__main__":
    unittest.main()
