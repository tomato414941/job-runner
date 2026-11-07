import unittest

from rental.decide import (
    End, Extend, Install, Keep, KeepFailed, Leaving, Machine, Memory, NoOffer, Release, Released, Rent, RentFailed,
    Seen, Send, Start, Stop, Upload, decide,
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
FAR = NOW + POLICY.lease


def report(phase, step=None, outbox=(), rate=None, done=0, total=None, step_seconds=None, failure=None):
    return {"phase": phase, "step": step, "outbox": list(outbox), "failure": failure, "step_seconds": step_seconds,
            "progress": None if rate is None else {"done": done, "total": total, "rate": rate}}


def seen(answer, cost=0.1, deadline=FAR):
    """A machine the provider lists, whose agent answers with a report, or with the phase alone."""
    return Seen(cost, deadline, report(answer) if isinstance(answer, str) else answer)


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
    return Memory(machines={"m1": Machine(**{"state": "working", "since": NOW - 60, **changes})})


def checkpoint_kept(at):
    return Record(checkpoints={"out/first/last.pt": {"key": "k", "sha256": "s", "bytes": 10, "step": "first",
                                                     "at": at}})


class DecideTests(unittest.TestCase):
    def decide(self, world, memory=None, record=None, facts=(), spec=SPEC):
        return decide(spec, record or Record(), memory or Memory(), world, list(facts), NOW)

    # Machines

    def test_rents_a_machine_when_none_works_on_the_steps_left(self):
        actions, _ = self.decide({})
        self.assertEqual(actions, [Rent()])

    def test_rents_nothing_more_while_a_machine_works(self):
        actions, _ = self.decide({"m1": seen(report("working", "first"))}, working())
        self.assertEqual(actions, [])

    def test_installs_the_job_on_a_machine_without_it(self):
        actions, memory = self.decide({"m1": seen("empty")})
        self.assertEqual(actions, [Install("m1")])
        self.assertEqual(memory.machines["m1"].state, "installing")

    def test_installs_again_when_the_agent_started_over_and_the_setup_time_runs_on(self):
        memory = working(state="installing", since=NOW - 600)
        actions, memory = self.decide({"m1": seen("empty")}, memory)
        self.assertEqual(actions, [Install("m1")])
        self.assertEqual(memory.machines["m1"].since, NOW - 600)

    def test_takes_up_a_forgotten_machine_where_its_job_is_without_installing_it_again(self):
        actions, memory = self.decide({"m1": seen(report("working", "first"))})
        self.assertEqual(actions, [])
        self.assertEqual(memory.machines["m1"].state, "working")

    def test_starts_the_steps_left_once_the_job_is_installed(self):
        actions, memory = self.decide({"m1": seen("ready")}, working(state="installing"))
        self.assertEqual(actions, [Start("m1")])
        self.assertEqual(memory.machines["m1"].state, "working")

    def test_starts_again_a_job_whose_start_did_not_arrive(self):
        actions, memory = self.decide({"m1": seen("ready")}, working())
        self.assertEqual(actions, [Start("m1")])
        self.assertEqual(memory.machines["m1"].restarts, 1)

    def test_waits_a_while_for_a_machine_that_does_not_answer(self):
        actions, memory = self.decide({"m1": seen(None)}, working(unreachable_since=NOW - 1))
        self.assertEqual(actions, [])
        self.assertIn("m1", memory.machines)

    def test_releases_and_rejects_a_machine_that_stopped_answering_and_rents_another(self):
        actions, memory = self.decide({"m1": seen(None)}, working(unreachable_since=NOW - POLICY.unreachable))
        self.assertEqual(actions, [Release("m1", "it stopped answering", True), Rent()])
        self.assertIn("m1", memory.leaving)

    def test_moves_a_deadline_on_while_little_of_it_is_left_until_it_has_moved(self):
        near = NOW + POLICY.renew - 1
        for _ in range(2):
            actions, _ = self.decide({"m1": seen(report("working", "first"), deadline=near)}, working())
            self.assertEqual(actions, [Extend("m1")])
        actions, _ = self.decide({"m1": seen(report("working", "first"), deadline=NOW + POLICY.lease)}, working())
        self.assertEqual(actions, [])

    # Uploads

    def test_sends_outputs_and_logs_at_once_and_keeps_what_the_machine_sent(self):
        outbox = [entry("o1"), entry("l1", kind="log", state="sent")]
        actions, memory = self.decide({"m1": seen(report("working", "first", outbox))}, working())
        self.assertEqual(actions, [Send("m1", ("o1",)), Keep("m1", ("l1",))])
        self.assertEqual(memory.uploads["o1"], Upload("m1", NOW, 1))

    def test_sends_a_step_checkpoint_at_most_every_sync_interval_while_its_job_runs(self):
        outbox = [entry("c1", kind="checkpoint")]
        actions, _ = self.decide({"m1": seen(report("working", "first", outbox))}, working(), checkpoint_kept(NOW - 10))
        self.assertEqual(actions, [])
        actions, _ = self.decide({"m1": seen(report("working", "first", outbox))}, working(),
                                 checkpoint_kept(NOW - POLICY.sync))
        self.assertEqual(actions, [Send("m1", ("c1",))])

    def test_sends_one_checkpoint_of_a_path_at_a_time(self):
        outbox = [entry("c1", kind="checkpoint", state="sending"), entry("c2", kind="checkpoint", at=2.0)]
        memory = working()
        memory.uploads["c1"] = Upload("m1", NOW - 5, 1)
        actions, _ = self.decide({"m1": seen(report("working", "first", outbox))}, memory)
        self.assertEqual(actions, [])

    def test_sends_a_checkpoint_at_once_from_a_machine_handing_over_or_a_job_that_stopped(self):
        outbox = [entry("c1", kind="checkpoint")]
        for phase, state in (("working", "handing"), ("crashed", "working")):
            actions, _ = self.decide({"m1": seen(report(phase, "first", outbox))}, working(state=state),
                                     checkpoint_kept(NOW - 10))
            self.assertEqual(actions[0], Send("m1", ("c1",)))

    def test_sends_again_an_upload_that_failed_and_ends_the_run_when_it_keeps_failing(self):
        outbox = [entry("o1", state="failed")]
        memory = working()
        memory.uploads["o1"] = Upload("m1", NOW - 5, 1)
        actions, memory = self.decide({"m1": seen(report("working", "first", outbox))}, memory)
        self.assertEqual(actions, [Send("m1", ("o1",))])
        memory.uploads["o1"].attempts = POLICY.send_attempts
        actions, memory = self.decide({"m1": seen(report("working", "first", outbox))}, memory)
        self.assertEqual(actions, [Release("m1", "failed", False)])
        self.assertEqual((memory.outcome, memory.detail),
                         ("failed", f"the upload of out/first/last.pt failed {POLICY.send_attempts} times"))

    def test_sends_again_an_upload_the_store_did_not_keep_although_the_machine_sent_it(self):
        outbox = [entry("o1", state="sent")]
        memory = working()
        memory.uploads["o1"] = Upload("m1", NOW - 5, 1)
        actions, _ = self.decide({"m1": seen(report("working", "first", outbox))}, memory,
                                 facts=[KeepFailed("m1", "o1")])
        self.assertEqual(actions, [Send("m1", ("o1",))])

    def test_sends_again_an_upload_whose_authorization_did_not_arrive(self):
        outbox = [entry("o1", state="ready")]
        memory = working()
        memory.uploads["o1"] = Upload("m1", NOW - 60, 1)
        actions, memory = self.decide({"m1": seen(report("working", "first", outbox))}, memory)
        self.assertEqual(actions, [Send("m1", ("o1",))])
        self.assertEqual(memory.uploads["o1"].attempts, 2)

    def test_sends_again_an_upload_unheard_of_for_too_long(self):
        outbox = [entry("o1", state="sending")]
        memory = working()
        memory.uploads["o1"] = Upload("m1", NOW - POLICY.send - 1, 1)
        actions, _ = self.decide({"m1": seen(report("working", "first", outbox))}, memory)
        self.assertEqual(actions, [Send("m1", ("o1",))])

    # Slow machines

    def slow(self, done=0, total=1_000_000, step_seconds=POLICY.warmup, slow_since=NOW - POLICY.slow):
        answer = report("working", "first", rate=30.0, done=done, total=total, step_seconds=step_seconds)
        return self.decide({"m1": seen(answer)}, working(slow_since=slow_since))

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
        actions, _ = self.decide({"m1": seen(report("stopped", "first"))}, memory)
        self.assertEqual(actions, [Release("m1", "slow", True), Rent()])

    def test_releases_a_machine_handing_over_when_the_handover_takes_too_long(self):
        memory = working(state="handing", since=NOW - POLICY.handover - 1, reason="slow", reject=True)
        actions, _ = self.decide({"m1": seen(report("working", "first"))}, memory)
        self.assertEqual(actions, [Release("m1", "slow", True), Rent()])

    # Failures

    def test_rejects_a_machine_a_step_failed_on_once_its_files_are_kept(self):
        failed = report("failed", "first", failure={"step": "first", "exit_code": 3})
        actions, memory = self.decide({"m1": seen(failed)}, working())
        self.assertEqual(actions, [Release("m1", "first failed on it", True), Rent()])
        self.assertEqual(memory.failures, {"first": 1})

    def test_keeps_a_failed_step_files_before_releasing_its_machine(self):
        failed = report("failed", "first", [entry("l1", kind="log")], failure={"step": "first", "exit_code": 3})
        actions, _ = self.decide({"m1": seen(failed)}, working())
        self.assertEqual(actions, [Send("m1", ("l1",))])

    def test_ends_the_run_failed_when_a_step_fails_on_a_second_machine(self):
        memory = working()
        memory.failures["first"] = 1
        failed = report("failed", "first", failure={"step": "first", "exit_code": 3})
        actions, memory = self.decide({"m1": seen(failed)}, memory)
        self.assertEqual(actions, [Release("m1", "failed", False)])
        self.assertEqual((memory.outcome, memory.detail), ("failed", "first failed on 2 machines"))

    def test_starts_again_a_job_that_stopped_by_itself_and_gives_up_its_machine_after_too_many_times(self):
        actions, memory = self.decide({"m1": seen(report("crashed", "first"))}, working())
        self.assertEqual(actions, [Start("m1")])
        self.assertEqual(memory.machines["m1"].restarts, 1)
        actions, _ = self.decide({"m1": seen(report("crashed", "first"))}, working(restarts=POLICY.restarts))
        self.assertEqual(actions, [Release("m1", "its job stopped by itself (crashed), again", True), Rent()])

    def test_keeps_what_a_job_left_before_starting_it_again(self):
        actions, memory = self.decide({"m1": seen(report("crashed", "first", [entry("c1", kind="checkpoint")]))},
                                      working())
        self.assertEqual(actions, [Send("m1", ("c1",))])
        self.assertEqual(memory.machines["m1"].restarts, 0)

    def test_starts_again_when_the_store_lacks_what_finished_steps_left(self):
        actions, _ = self.decide({"m1": seen("finished")}, working())
        self.assertEqual(actions, [Start("m1")])

    def test_rejects_a_machine_whose_setup_failed_and_ends_the_run_when_it_fails_again(self):
        actions, memory = self.decide({"m1": seen("install_failed")}, working(state="installing"))
        self.assertEqual(actions, [Release("m1", "its setup failed", True), Rent()])
        memory.machines["m2"] = Machine("installing", NOW)
        actions, memory = self.decide({"m2": seen("install_failed")}, memory)
        self.assertEqual(actions, [Release("m2", "failed", False)])
        self.assertEqual(memory.detail, "the setup failed on 2 machines")

    def test_releases_a_machine_whose_setup_takes_too_long(self):
        memory = working(state="installing", since=NOW - POLICY.install - 1)
        actions, _ = self.decide({"m1": seen("installing")}, memory)
        self.assertEqual(actions, [Release("m1", "its setup took too long", True), Rent()])

    def test_waits_while_no_offer_fits(self):
        actions, memory = self.decide({}, facts=[NoOffer()])
        self.assertEqual((actions, memory.waiting_since), ([Rent()], NOW))

    def test_ends_the_run_failed_when_renting_keeps_failing_otherwise_than_for_want_of_an_offer(self):
        actions, memory = self.decide({}, facts=[RentFailed("the image did not start")])
        self.assertEqual((actions, memory.rent_failures), ([Rent()], 1))
        _, memory = self.decide({"m1": seen("empty")}, memory)
        self.assertEqual(memory.rent_failures, 0)
        memory = Memory(rent_failures=POLICY.rent_failures - 1)
        actions, memory = self.decide({}, memory, facts=[RentFailed("the image did not start")])
        self.assertEqual(actions, [End("failed", "renting failed 2 times: the image did not start")])

    # The end

    def test_ends_the_run_complete_once_the_store_keeps_every_step_outputs_and_its_machines_are_gone(self):
        actions, memory = self.decide({"m1": seen("finished")}, working(), finished("first", "second"))
        self.assertEqual(actions, [Release("m1", "complete", False)])
        self.assertEqual((memory.outcome, memory.ended), ("complete", False))
        actions, memory = self.decide({}, memory, finished("first", "second"), facts=[Released("m1", 0.2)])
        self.assertEqual(actions, [End("complete", "every step's outputs are in the store")])
        self.assertEqual((memory.ended, memory.spent), (True, 0.2))

    def test_ends_the_run_over_its_budget(self):
        memory = working()
        memory.spent = 9.5
        actions, memory = self.decide({"m1": seen(report("working", "first"), cost=0.6)}, memory)
        self.assertEqual(actions, [Release("m1", "over budget", False)])
        self.assertEqual(memory.detail, "10.10 dollars spent")

    def test_releases_a_machine_it_did_not_know_of_once_the_run_has_its_outcome(self):
        memory = Memory(outcome="complete", detail="every step's outputs are in the store")
        actions, memory = self.decide({"m9": seen("ready")}, memory)
        self.assertEqual(actions, [Release("m9", "complete", False)])
        self.assertIn("m9", memory.leaving)

    def test_counts_what_a_machine_cost_once_the_provider_no_longer_lists_it(self):
        actions, memory = self.decide({}, working(cost=1.5))
        self.assertEqual(actions, [Rent()])
        self.assertEqual(memory.spent, 1.5)

    def test_counts_the_dollars_the_provider_reports_for_a_released_machine(self):
        memory = Memory(leaving={"m1": Leaving(NOW - 1, 1.0)})
        _, memory = self.decide({}, memory, facts=[Released("m1", 1.2)])
        self.assertEqual(memory.spent, 1.2)

    def test_asks_again_to_release_a_machine_the_provider_still_lists(self):
        memory = Memory(leaving={"m1": Leaving(NOW - POLICY.unreachable - 1, 1.0)})
        actions, _ = self.decide({"m1": seen(None, cost=1.0)}, memory)
        self.assertEqual(actions, [Release("m1", "still listed after its release", False), Rent()])

    def test_does_nothing_once_the_run_is_over(self):
        actions, _ = self.decide({}, Memory(outcome="complete", ended=True))
        self.assertEqual(actions, [])

    def test_remembers_what_it_decided_across_a_restart(self):
        memory = working()
        memory.leaving["m0"] = Leaving(NOW - 5, 0.3)
        memory.uploads["o1"] = Upload("m1", NOW - 5, 1, True)
        self.assertEqual(Memory.read(memory.write()), memory)


if __name__ == "__main__":
    unittest.main()
