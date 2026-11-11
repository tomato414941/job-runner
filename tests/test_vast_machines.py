"""Checks of machines.py and machine_check.py without Vast.ai, SSH or a GPU.

Run in this folder: python3 -m unittest
"""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# The Vast.ai part runs as scripts beside each other in vast/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vast"))

import machines
import machine_check


def offer(machine, host, price, gpu="RTX 5090", power=575.0):
    return {"id": machine * 10, "machine_id": machine, "host_id": host, "dph_total": price, "gpu_name": gpu,
            "num_gpus": 1, "gpu_max_power": power, "geolocation": "Taiwan, TW", "reliability2": 0.99,
            "verification": "verified"}


def results(matmul=100.0, memory=1500.0, used=0.0, limit=575.0, default=575.0, download=500.0,
            version=machine_check.VERSION):
    return {"version": version, "matmul_tflops": matmul, "memory_gbps": memory, "download_mbps": download,
            "gpus": [{"name": "NVIDIA GeForce RTX 5090", "memory_used_mib": used, "power_limit_w": limit,
                      "power_default_w": default}]}


def renting_arguments(**changes):
    return SimpleNamespace(**{"gpu": "RTX 5090", "count": 1, "disk": 60, "query": None, "max_price": None,
                              "countries": None, "unverified": False, "image": "image", "label": "test", "hours": 1.0,
                              "tries": 3, "failure_budget": None, "min_download": 0, "env": None, "onstart": None,
                              "avoid": [], **changes})


class LedgerTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        moved = patch.object(machines, "folder", self.folder)
        moved.start()
        self.addCleanup(moved.stop)
        # A ledger in a test's own folder is never shared; a test that reached R2 all the same fails.
        offline = patch.object(machines.shared_ledger, "request", side_effect=AssertionError("a test reached R2"))
        offline.start()
        self.addCleanup(offline.stop)
        # The workstation's own key file, if a session hands one over, stays out of the tests.
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("VAST_SSH_KEY_FILE", None)

    def passed(self, machine, gpu="RTX 5090", **measured):
        machines.record("checked", provider="vast", instance=machine, machine=machine, host=machine, gpu=gpu,
                          results=results(**measured), kept=True, reasons=[])

    def failed(self, machine, host, stage="start"):
        machines.record("unusable", provider="vast", instance=machine, machine=machine, host=host, gpu="RTX 5090",
                          stage=stage, detail="")

    def rented(self, instance, deadline, price=0.5, label="test", workstation=None):
        machines.record("rented", provider="vast", instance=instance, machine=instance, host=instance,
                          gpu="RTX 5090", price=price, deadline=machines.stamp(deadline), label=label,
                          workstation=machines.workstation() if workstation is None else workstation)

    def events(self):
        return [(entry["event"], entry.get("instance")) for entry in machines.entries()]


class LedgerTest(LedgerTestCase):
    def test_reads_back_the_entries_in_the_order_they_were_recorded(self):
        machines.record("first", value=1)
        machines.record("second", value=2)
        self.assertEqual([(entry["event"], entry["value"]) for entry in machines.entries()],
                         [("first", 1), ("second", 2)])

    def test_skips_a_line_that_a_writer_left_unfinished(self):
        machines.record("whole")
        with open(machines.home() / "ledger.jsonl", "a") as stream:
            stream.write('{"at": "2026-11-06T00:00:00+00:00", "ev')
        self.assertEqual([entry["event"] for entry in machines.entries()], ["whole"])

    def test_follows_a_rentals_latest_deadline_until_it_goes_back(self):
        moment = machines.now()
        self.rented(7, moment)
        machines.record("extended", provider="vast", instance=7, deadline=machines.stamp(moment + timedelta(hours=2)))
        held = machines.rental(machines.entries(), 7)
        self.assertEqual(held["deadline"], machines.stamp(moment + timedelta(hours=2)))
        self.assertEqual((held["released"], held["returned"]), (False, False))
        machines.record("released", provider="vast", instance=7, by="command")
        held = machines.rental(machines.entries(), 7)
        self.assertEqual((held["released"], held["returned"]), (True, False))
        machines.record("returned", provider="vast", instance=7, by="command")
        self.assertTrue(machines.rental(machines.entries(), 7)["returned"])

    def test_reads_and_writes_the_ledger_in_the_folder_that_data_names(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        rented = {"at": machines.stamp(), "event": "rented", "provider": "vast", "instance": 7, "machine": 7,
                  "host": 7, "gpu": "RTX 5090", "price": 0.5, "deadline": machines.stamp()}
        (Path(other.name) / "ledger.jsonl").write_text(json.dumps(rented) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(machines.main(["--data", other.name, "extend", "7", "2"]), 0)
        lines = (Path(other.name) / "ledger.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["event"] for line in lines], ["rented", "extended"])
        self.assertFalse((self.folder / "ledger.jsonl").exists())


class ChoiceTest(LedgerTestCase):
    def test_leaves_out_machines_that_failed_once(self):
        self.failed(1, host=10)
        chosen = machines.choose([offer(1, 10, 0.4), offer(2, 20, 0.5)], machines.entries())
        self.assertEqual([each["machine_id"] for each in chosen], [2])

    def test_leaves_out_the_machines_that_the_instances_to_avoid_ran_on(self):
        self.rented(7, machines.now())
        chosen = machines.choose([offer(7, 70, 0.4), offer(8, 80, 0.5)], machines.entries(), avoid=[7, 99])
        self.assertEqual([each["machine_id"] for each in chosen], [8])

    def test_puts_hosts_that_failed_before_after_the_others_each_group_cheapest_first(self):
        self.failed(1, host=10)
        offers = [offer(3, 10, 0.4), offer(4, 20, 0.7), offer(5, 30, 0.6), offer(6, 10, 0.3)]
        chosen = machines.choose(offers, machines.entries())
        self.assertEqual([each["machine_id"] for each in chosen], [5, 4, 6, 3])

    def test_leaves_out_offers_whose_power_limit_is_below_the_lowest_default_of_checked_machines_with_the_same_gpu(self):
        # Two of three checked machines default to 600 W; a machine whose default is 575 W still offers its own.
        self.passed(1, default=575.0)
        self.passed(6, default=600.0)
        self.passed(7, default=600.0)
        offers = [offer(2, 20, 0.4, power=450.0), offer(3, 30, 0.5, power=575.0), offer(4, 40, 0.6, power=600.0),
                  offer(5, 50, 0.3, gpu="RTX 4090", power=450.0)]
        chosen = machines.choose(offers, machines.entries())
        self.assertEqual([each["machine_id"] for each in chosen], [5, 3, 4])

    def test_keeps_offers_of_a_gpu_whose_default_power_no_check_has_reported(self):
        chosen = machines.choose([offer(2, 20, 0.4, power=450.0)], machines.entries())
        self.assertEqual([each["machine_id"] for each in chosen], [2])


class VerdictTest(LedgerTestCase):
    def test_keeps_a_machine_that_matches_its_gpus_checked_machines(self):
        self.passed(1)
        self.passed(2)
        self.assertEqual(machines.verdict(results(matmul=95.0), machines.entries(), "RTX 5090"), [])

    def test_sends_back_a_machine_whose_gpu_holds_memory(self):
        reasons = machines.verdict(results(used=14000.0), machines.entries(), "RTX 5090")
        self.assertEqual([reason.split(":")[0] for reason in reasons], ["gpu_busy"])

    def test_sends_back_a_machine_whose_power_is_capped_below_its_default(self):
        reasons = machines.verdict(results(limit=450.0), machines.entries(), "RTX 5090")
        self.assertEqual([reason.split(":")[0] for reason in reasons], ["power_capped"])

    def test_sends_back_a_machine_whose_gpu_multiplies_wrongly(self):
        answers = {**results(), "bfloat16_error": machine_check.NOT_FINITE, "bfloat16_repeat_gap": 0.0}
        reasons = machines.verdict(answers, machines.entries(), "RTX 5090")
        self.assertEqual([reason.split(":")[0] for reason in reasons], ["wrong_results"])

    def test_sends_back_a_machine_whose_gpu_answers_the_same_product_differently(self):
        answers = {**results(), "float32_error": 1e-6, "float32_repeat_gap": 0.4}
        reasons = machines.verdict(answers, machines.entries(), "RTX 5090")
        self.assertEqual([reason.split(":")[0] for reason in reasons], ["unrepeatable"])

    def test_keeps_a_machine_whose_products_stray_no_more_than_rounding(self):
        answers = {**results(), "float32_error": 3e-7, "float32_repeat_gap": 0.0, "bfloat16_error": 4e-3,
                   "bfloat16_repeat_gap": 0.0}
        self.assertEqual(machines.verdict(answers, machines.entries(), "RTX 5090"), [])

    def test_sends_back_a_machine_slower_than_the_median_of_checked_machines_with_its_gpu(self):
        self.passed(1, matmul=100.0)
        self.passed(2, matmul=104.0)
        reasons = machines.verdict(results(matmul=80.0), machines.entries(), "RTX 5090")
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("slow: matmul_tflops 80 against a median of 102 over 2 machines"))

    def test_compares_a_machine_only_with_checked_machines_with_its_gpu_and_its_checks_version(self):
        self.passed(1, gpu="H100 SXM", matmul=600.0)
        self.passed(2, gpu="H100 SXM", matmul=600.0)
        self.passed(3, matmul=100.0)
        self.passed(4, matmul=100.0, version=machine_check.VERSION + 1)
        self.assertEqual(machines.verdict(results(matmul=50.0), machines.entries(), "RTX 5090"), [])

    def test_sends_back_a_machine_whose_download_is_slower_than_the_work_needs(self):
        reasons = machines.verdict(results(download=40.0), machines.entries(), "RTX 5090", min_download_mbps=100)
        self.assertEqual([reason.split(":")[0] for reason in reasons], ["slow_download"])

    def test_sends_back_a_machine_that_could_not_download_at_all(self):
        reasons = machines.verdict(results(download=0.0), machines.entries(), "RTX 5090", min_download_mbps=10)
        self.assertEqual([reason.split(":")[0] for reason in reasons], ["slow_download"])

    def test_sends_back_a_machine_whose_traffic_leaves_from_another_country_than_its_listing(self):
        measured = {**results(), "egress_country": "CN"}
        reasons = machines.verdict(measured, machines.entries(), "RTX 5090", listed_country="US")
        self.assertEqual(reasons, ["egress_elsewhere: listed in US, but its traffic leaves from CN"])

    def test_keeps_a_machine_whose_traffic_leaves_from_its_listed_country(self):
        measured = {**results(), "egress_country": "US"}
        self.assertEqual(machines.verdict(measured, machines.entries(), "RTX 5090", listed_country="US"), [])

    def test_holds_a_check_by_bash_alone_to_no_median_of_the_gpus_speeds(self):
        self.passed(1, matmul=100.0)
        self.passed(2, matmul=100.0)
        shell = {"version": "shell", "gpus": results()["gpus"], "download_mbps": 80.0, "egress_country": "US"}
        self.assertEqual(machines.verdict(shell, machines.entries(), "RTX 5090", min_download_mbps=10,
                                             listed_country="US"), [])

    def test_blames_no_download_that_the_check_could_not_measure(self):
        unmeasured = {key: value for key, value in results().items() if key != "download_mbps"}
        self.assertEqual(machines.verdict(unmeasured, machines.entries(), "RTX 5090", min_download_mbps=10), [])


class SendBackTest(LedgerTestCase):
    def test_records_that_the_machine_is_to_go_back_then_that_it_went_back(self):
        self.rented(7, machines.now())
        destroyed = []
        with patch.object(machines, "destroy", destroyed.append):
            entry = machines.send_back(7, by="command")
        self.assertEqual(destroyed, [7])
        self.assertEqual(self.events(), [("rented", 7), ("released", 7), ("returned", 7)])
        self.assertEqual((entry["event"], entry["by"]), ("returned", "command"))

    def test_leaves_a_machine_that_could_not_be_destroyed_released_for_the_watcher(self):
        self.rented(7, machines.now())
        with patch.object(machines, "destroy", side_effect=machines.VastError(500, "busy")), \
                patch.object(machines, "show", lambda instance: {"id": instance}):
            with self.assertRaises(machines.VastError):
                machines.send_back(7, by="command")
        held = machines.rental(machines.entries(), 7)
        self.assertEqual((held["released"], held["returned"]), (True, False))

    def test_leaves_a_machine_released_when_vast_answers_that_the_destruction_failed_and_it_is_still_there(self):
        self.rented(7, machines.now())
        with patch.object(machines, "destroy", lambda instance: {"success": False, "msg": "try later"}), \
                patch.object(machines, "show", lambda instance: {"id": instance}):
            with self.assertRaises(machines.VastError):
                machines.send_back(7, by="command")
        held = machines.rental(machines.entries(), 7)
        self.assertEqual((held["released"], held["returned"]), (True, False))

    def test_records_the_return_of_a_machine_that_is_gone_already(self):
        self.rented(7, machines.now(), price=0.6)
        with patch.object(machines, "destroy", side_effect=machines.VastError(404, "gone")), \
                patch.object(machines, "show", lambda instance: None):
            entry = machines.send_back(7, by="command")
        self.assertEqual((entry["event"], entry["by"]), ("returned", "command"))
        self.assertTrue(machines.rental(machines.entries(), 7)["returned"])

    def test_sends_a_machine_back_once(self):
        self.rented(7, machines.now())
        destroyed = []
        with patch.object(machines, "destroy", destroyed.append):
            machines.send_back(7, by="command")
            self.assertIsNone(machines.send_back(7, by="watch"))
        self.assertEqual(destroyed, [7])


class WatchTest(LedgerTestCase):
    def test_sends_back_a_machine_of_this_workstation_once_its_deadline_has_passed(self):
        self.rented(7, machines.now() - timedelta(minutes=1))
        self.rented(8, machines.now() + timedelta(hours=1))
        destroyed = []
        with patch.object(machines, "destroy", destroyed.append):
            machines.watch_round()
        self.assertEqual(destroyed, [7])
        self.assertEqual(machines.entries()[-1]["by"], "watch")

    def test_sends_back_at_once_a_machine_whose_return_the_work_asked_for(self):
        self.rented(7, machines.now() + timedelta(hours=1))
        machines.record("released", provider="vast", instance=7, by="command")
        destroyed = []
        with patch.object(machines, "destroy", destroyed.append):
            machines.watch_round()
        self.assertEqual(destroyed, [7])

    def test_tries_again_the_next_round_to_send_back_a_machine_that_vast_did_not_destroy(self):
        self.rented(7, machines.now() - timedelta(minutes=1))
        answers = [machines.VastError(500, "busy"), None]

        def destroy(instance):
            answer = answers.pop(0)
            if answer is not None:
                raise answer

        with patch.object(machines, "destroy", destroy), \
                patch.object(machines, "show", lambda instance: {"id": instance}), \
                contextlib.redirect_stdout(io.StringIO()):
            machines.watch_round()
            self.assertFalse(machines.rental(machines.entries(), 7)["returned"])
            machines.watch_round()
        self.assertTrue(machines.rental(machines.entries(), 7)["returned"])

    def test_leaves_the_machines_of_other_workstations_to_their_own_watchers(self):
        self.rented(7, machines.now() - timedelta(minutes=1), workstation="elsewhere")
        with patch.object(machines, "destroy", side_effect=AssertionError("sent back another's machine")):
            machines.watch_round()

    def test_tells_rent_that_it_runs_until_its_heartbeat_grows_stale(self):
        self.assertFalse(machines.watching())
        machines.watch_round()
        self.assertTrue(machines.watching())
        later = datetime.now(timezone.utc) + timedelta(seconds=machines.WATCH_STALE_SECONDS + 1)
        with patch.object(machines, "now", lambda: later):
            self.assertFalse(machines.watching())

    def test_takes_a_watcher_that_watches_with_another_key_as_stopped(self):
        os.environ["VAST_API_KEY"] = "one"
        machines.watch_round()
        self.assertTrue(machines.watching())
        os.environ["VAST_API_KEY"] = "another"
        self.assertFalse(machines.watching())

    def test_takes_a_watcher_that_is_gone_as_stopped(self):
        machines.watch_round()
        ended = subprocess.Popen(["true"])
        ended.wait()
        beat = json.loads((self.folder / "watch.json").read_text())
        (self.folder / "watch.json").write_text(json.dumps({**beat, "pid": ended.pid}))
        self.assertFalse(machines.watching())


class GiveBackTest(LedgerTestCase):
    def setUp(self):
        super().setUp()
        os.environ["VAST_API_KEY"] = "test"

    def test_sends_back_every_machine_with_the_label_that_has_not_gone_back(self):
        for instance, label in ((7, "job-a"), (8, "job-a"), (9, "job-b")):
            self.rented(instance, machines.now(), label=label)
        machines.record("returned", provider="vast", instance=8, by="command")
        destroyed = []
        with patch.object(machines, "destroy", destroyed.append), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(machines.main(["return", "--label", "job-a"]), 0)
        self.assertEqual(destroyed, [7])

    def test_fails_and_leaves_to_the_watcher_a_machine_that_vast_did_not_destroy(self):
        self.rented(7, machines.now(), label="job-a")
        with patch.object(machines, "destroy", side_effect=machines.VastError(500, "busy")), \
                patch.object(machines, "show", lambda instance: {"id": instance}), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(machines.main(["return", "--label", "job-a"]), 1)
        self.assertTrue(machines.rental(machines.entries(), 7)["released"])

    def test_rents_no_machine_again_that_the_work_found_unusable(self):
        self.rented(7, machines.now())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(machines.main(["reject", "7", "unreachable during the work"]), 0)
        self.assertEqual(machines.avoided(machines.entries()), {("vast", 7)})
        self.assertEqual(machines.choose([offer(7, 7, 0.4), offer(8, 8, 0.5)], machines.entries()),
                         [offer(8, 8, 0.5)])

    def test_moves_no_deadline_of_a_machine_that_is_going_back(self):
        self.rented(7, machines.now())
        machines.record("released", provider="vast", instance=7, by="command")
        with self.assertRaises(SystemExit):
            machines.main(["extend", "7", "2"])


class ListTest(LedgerTestCase):
    def setUp(self):
        super().setUp()
        os.environ["VAST_API_KEY"] = "test"

    def test_lists_the_machines_with_the_label_with_their_deadlines_and_what_they_have_cost(self):
        self.rented(7, machines.now() + timedelta(hours=2), price=0.5, label="job")
        started = (machines.now() - timedelta(hours=2)).timestamp()
        listed = {"instances": [{"id": 7, "label": "job", "actual_status": "running"},
                                {"id": 8, "label": "other", "actual_status": "running"},
                                {"id": 9, "label": "job", "actual_status": "loading", "dph_total": 0.25,
                                 "start_date": started}]}
        output = io.StringIO()
        with patch.object(machines, "call", lambda *arguments, **options: {**listed, "next_token": None}), \
                contextlib.redirect_stdout(output):
            self.assertEqual(machines.main(["list", "--label", "job"]), 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["instance"] for row in rows], [7, 9])
        self.assertEqual(rows[0]["deadline"], machines.rental(machines.entries(), 7)["deadline"])
        self.assertIsNone(rows[1]["deadline"])
        self.assertAlmostEqual(rows[1]["dollars"], 0.5, places=2)


class RentTest(LedgerTestCase):
    def renting(self, offers, measured, taken=()):
        """Fakes Vast.ai, a running watcher and SSH: someone else rents the offers of the machines in `taken` first, and the
        machine of any other offer reports measured[machine], or does not run the check if that is None."""
        self.destroyed = []

        def create(chosen, *rest):
            if chosen["machine_id"] in taken:
                raise machines.VastError(400, "the offer is no longer available")
            return 1000 + chosen["machine_id"]

        def check(instance):
            found = measured[instance - 1000]
            return (found, None) if found is not None else (None, "the check failed with 127: python3: not found")

        stack = contextlib.ExitStack()
        for name, fake in {"search": lambda *arguments, **options: offers, "create": create,
                           "watching": lambda: True,
                           "wait_until_running": lambda instance: None, "attach": lambda instance, key: {},
                           "wait_for_ssh": lambda instance: ("203.0.113.9", instance),
                           "ssh_base": lambda instance, host, port: instance, "check": check,
                           "destroy": self.destroyed.append, "show": lambda instance: None,
                           "public_key": lambda: "ssh-ed25519 AAAA test"}.items():
            stack.enter_context(patch.object(machines, name, fake))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_sends_back_a_machine_that_fails_its_check_and_keeps_the_next_one_that_passes(self):
        with self.renting([offer(1, 10, 0.5), offer(2, 20, 0.6)], {1: results(limit=450.0), 2: results()}):
            self.assertEqual(machines.rent(renting_arguments()), 0)
        self.assertEqual(self.destroyed, [1001])
        self.assertEqual(self.events(), [("rented", 1001), ("checked", 1001), ("unusable", 1001), ("released", 1001),
                                         ("returned", 1001), ("rented", 1002), ("checked", 1002)])
        self.assertEqual(machines.avoided(machines.entries()), {("vast", 1)})

    def test_sends_back_a_machine_whose_traffic_leaves_from_another_country_and_keeps_the_next(self):
        measured = {1: {**results(), "egress_country": "CN"}, 2: {**results(), "egress_country": "TW"}}
        with self.renting([offer(1, 10, 0.5), offer(2, 20, 0.6)], measured):
            self.assertEqual(machines.rent(renting_arguments()), 0)
        self.assertEqual(self.destroyed, [1001])
        self.assertEqual(machines.avoided(machines.entries()), {("vast", 1)})

    def test_gives_up_after_as_many_failed_machines_as_it_may_try(self):
        offers = [offer(machine, machine * 10, 0.5) for machine in (1, 2, 3)]
        with self.renting(offers, {machine: results(used=9000.0) for machine in (1, 2, 3)}):
            self.assertEqual(machines.rent(renting_arguments(tries=2)), 1)
        self.assertEqual(self.destroyed, [1001, 1002])

    def test_writes_how_to_reach_a_kept_machine_and_forgets_it_once_the_machine_goes_back(self):
        with self.renting([offer(1, 10, 0.5)], {1: results()}):
            self.assertEqual(machines.rent(renting_arguments()), 0)
            connection = machines.connection_file(1001)
            text = connection.read_text()
            machines.send_back(1001, by="test")
        self.assertIn("Host vast-1001\n", text)
        self.assertIn("HostName 203.0.113.9\n", text)
        self.assertIn("Port 1001\n", text)
        self.assertFalse(connection.exists())

    def test_moves_on_from_an_offer_that_someone_rented_first(self):
        with self.renting([offer(1, 10, 0.5), offer(2, 20, 0.6)], {2: results()}, taken={1}):
            self.assertEqual(machines.rent(renting_arguments(tries=1)), 0)
        self.assertEqual(self.events(), [("rented", 1002), ("checked", 1002)])

    def test_sends_back_the_machine_without_blaming_it_when_the_agent_has_dropped_its_key(self):
        answers = iter(["ssh-ed25519 AAAA test"])

        def public_key():
            try:
                return next(answers)
            except StopIteration:
                raise SystemExit("no key") from None

        with self.renting([offer(1, 10, 0.5)], {1: results()}), \
                patch.object(machines, "wait_for_ssh", lambda instance: None), \
                patch.object(machines, "public_key", public_key):
            with self.assertRaises(SystemExit):
                machines.rent(renting_arguments())
        self.assertEqual(self.destroyed, [1001])
        self.assertEqual(self.events(), [("rented", 1001), ("released", 1001), ("returned", 1001)])
        self.assertEqual(machines.avoided(machines.entries()), set())

    def test_sends_back_the_machine_without_blaming_it_when_the_check_does_not_run(self):
        with self.renting([offer(1, 10, 0.5)], {1: None}):
            with self.assertRaises(RuntimeError):
                machines.rent(renting_arguments())
        self.assertEqual(self.destroyed, [1001])
        self.assertEqual(self.events(), [("rented", 1001), ("released", 1001), ("returned", 1001)])
        self.assertEqual(machines.avoided(machines.entries()), set())

    def test_rents_nothing_while_the_watcher_does_not_run(self):
        with self.renting([offer(1, 10, 0.5)], {1: results()}), \
                patch.object(machines, "watching", lambda: False), \
                patch.object(machines, "create", side_effect=AssertionError("rented")):
            with self.assertRaises(SystemExit):
                machines.rent(renting_arguments())

    def test_stops_trying_once_the_failed_machines_have_cost_more_than_the_failure_budget(self):
        offers = [offer(machine, machine * 10, 0.5) for machine in (1, 2, 3)]
        real = machines.send_back
        output = io.StringIO()
        with self.renting(offers, {machine: results(used=9000.0) for machine in (1, 2, 3)}), \
                patch.object(machines, "send_back", lambda instance, by: {**real(instance, by), "dollars": 0.3}), \
                contextlib.redirect_stdout(output):
            self.assertEqual(machines.rent(renting_arguments(tries=3, failure_budget=0.5)), 1)
        self.assertEqual(self.destroyed, [1001, 1002])
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["failed_dollars"], 0.6)

    def test_prints_the_kept_machines_deadline_and_records_the_workstation_that_rented_it(self):
        output = io.StringIO()
        with self.renting([offer(1, 10, 0.5), offer(2, 20, 0.6)], {1: results(limit=450.0), 2: results()}), \
                contextlib.redirect_stdout(output):
            machines.rent(renting_arguments())
        kept = json.loads(output.getvalue().splitlines()[-1])
        held = machines.rental(machines.entries(), 1002)
        self.assertEqual((kept["rented"], kept["deadline"], kept["failed"]), (1002, held["deadline"], 1))
        self.assertEqual(held["workstation"], machines.workstation())


class SearchTest(unittest.TestCase):
    def searched(self, unverified):
        """The machines that search keeps of one verified, one unverified and one deverified offer, and its filters."""
        asked = []
        listed = {"offers": [{**offer(1, 10, 0.4), "verification": "verified"},
                             {**offer(2, 20, 0.3), "verification": "unverified"},
                             {**offer(3, 30, 0.2), "verification": "deverified"}]}

        def call(method, path, body=None, query=None):
            asked.append(query["q"])
            return listed

        with patch.object(machines, "call", call):
            found = machines.search("RTX 5090", unverified=unverified)
        return [each["machine_id"] for each in found], asked[0]

    def test_searches_only_machines_that_vast_has_verified_by_default(self):
        machines, filters = self.searched(unverified=False)
        self.assertEqual(machines, [1])
        self.assertEqual(filters["verified"], {"eq": True})

    def test_lets_in_unverified_machines_when_asked_but_never_those_that_lost_their_verification(self):
        machines, filters = self.searched(unverified=True)
        self.assertEqual(machines, [1, 2])
        self.assertNotIn("verified", filters)


class StartTest(unittest.TestCase):
    def waited(self, rows):
        """wait_until_running against a machine whose status at each look is the next of `rows`, on a clock that moves
        only while it sleeps; what it returned and how long it waited."""
        moment, looks = [0.0], iter(rows)

        def sleep(seconds):
            moment[0] += seconds

        with patch.object(machines.time, "monotonic", lambda: moment[0]), \
                patch.object(machines.time, "sleep", sleep), \
                patch.object(machines, "show", lambda instance: next(looks)):
            return machines.wait_until_running(7), moment[0]

    def test_gives_up_on_a_machine_that_has_not_begun_to_load_the_image_after_five_minutes(self):
        problem, waited = self.waited([{"actual_status": None, "status_msg": None}] * 100)
        self.assertEqual(problem, "the machine had not begun to load the image after 5 minutes")
        self.assertEqual(waited, machines.LOAD_SECONDS)

    def test_waits_for_a_machine_that_is_still_loading_the_image(self):
        rows = [{"actual_status": None}] * 2 + [{"actual_status": "loading", "status_msg": "pulling"}] * 18
        problem, waited = self.waited(rows + [{"actual_status": "running"}])
        self.assertIsNone(problem)
        self.assertEqual(waited, 21 * 30)

    def test_gives_up_at_once_on_a_machine_whose_container_reports_an_error(self):
        row = {"actual_status": "created", "status_msg": "Error: GPU error, unable to start instance."}
        problem, waited = self.waited([row] * 100)
        self.assertEqual(problem, "created: Error: GPU error, unable to start instance.")
        self.assertEqual(waited, 30)


class CheckTest(unittest.TestCase):
    SHELL_OUTPUT = ("gpu: NVIDIA GeForce RTX 5090, 0, 32607, 450.00, 575.00, 600.00, 3090, 5, 16, 30, 580.95.05\n"
                    "download: 1600000 100.0 120.0\nloc=CN\n")

    def test_checks_an_image_without_python3_with_bash_alone(self):
        ran = []

        def run(command, **options):
            ran.append((command[-1], options.get("input")))
            if command[-1] == "command -v python3":
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout=self.SHELL_OUTPUT, stderr="")

        with patch.object(machines.subprocess, "run", run):
            found, failure = machines.check(["ssh", "machine"])
        self.assertIsNone(failure)
        self.assertEqual([command for command, _ in ran], ["command -v python3", "bash -s"])
        self.assertIn("--query-gpu=name,memory.used,", ran[1][1])
        self.assertEqual((found["version"], found["gpus"][0]["power_limit_w"], found["download_mbps"],
                          found["egress_country"]), ("shell", 450.0, 0.1, "CN"))
        reasons = machines.verdict(found, [], "RTX 5090", min_download_mbps=10, listed_country="US")
        self.assertEqual([reason.split(":")[0] for reason in reasons], ["power_capped", "slow_download",
                                                                         "egress_elsewhere"])

    def test_reads_a_shell_check_that_reached_nothing_as_a_download_of_nothing_and_no_country(self):
        found = machines.read_shell_check("gpu: NVIDIA GeForce RTX 5090, 0, 32607, 575, 575, 600, 3090, 5, 16, 30, "
                                             "580.95.05\ndownload: 0 100.0 100.5\n")
        self.assertEqual((found["download_mbps"], found.get("egress_country")), (0.0, None))

    def test_runs_machine_check_with_the_images_python3(self):
        def run(command, **options):
            if command[-1] == "command -v python3":
                return SimpleNamespace(returncode=0, stdout="/usr/bin/python3\n", stderr="")
            self.assertEqual(command[-1], "python3 -")
            return SimpleNamespace(returncode=0, stdout='{"version": 2, "gpus": []}\n', stderr="")

        with patch.object(machines.subprocess, "run", run):
            self.assertEqual(machines.check(["ssh", "machine"]), ({"version": 2, "gpus": []}, None))


class AccessTest(LedgerTestCase):
    def test_stops_a_command_that_calls_vast_when_vast_api_key_is_missing(self):
        with patch.dict(os.environ):
            os.environ.pop("VAST_API_KEY", None)
            with patch.object(machines, "call", side_effect=AssertionError("called Vast.ai")):
                with self.assertRaises(SystemExit) as stopped:
                    machines.main(["return", "7"])
        self.assertIn("VAST_API_KEY", str(stopped.exception))

    def test_takes_the_public_key_from_the_file_that_vast_ssh_key_file_names(self):
        key = self.folder / "key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)], check=True)
        os.environ["VAST_SSH_KEY_FILE"] = str(key)
        self.assertEqual(machines.public_key(), " ".join(key.with_suffix(".pub").read_text().split()[:2]))

    def test_offers_the_key_file_alone_when_vast_ssh_key_file_names_one(self):
        os.environ["VAST_SSH_KEY_FILE"] = "/run/key"
        command = machines.ssh_base(7, "203.0.113.9", 41022)
        self.assertIn("IdentityFile=/run/key", command)
        self.assertIn("IdentitiesOnly=yes", command)
        self.assertIn("IdentityAgent=none", command)

    def test_stops_when_no_key_file_is_given(self):
        with self.assertRaises(SystemExit):
            machines.public_key()

    def test_stops_when_vast_ssh_key_file_names_no_key(self):
        os.environ["VAST_SSH_KEY_FILE"] = str(self.folder / "missing")
        with self.assertRaises(SystemExit):
            machines.public_key()


class AddressTest(unittest.TestCase):
    def test_hides_addresses_urls_and_vast_host_names_but_not_times(self):
        text = "00:03:38 ssh: connect to host 203.0.113.9 port 4022 via ssh5.vast.ai, see https://example.com/x?a=1"
        self.assertEqual(machines.hide_addresses(text),
                         "00:03:38 ssh: connect to host <ip> port 4022 via <host>, see <url>")

    def test_tries_the_hosts_own_port_before_vasts_proxy(self):
        row = {"public_ipaddr": "203.0.113.9", "ports": {"22/tcp": [{"HostPort": "41022"}]},
               "ssh_host": "ssh5.vast.ai", "ssh_port": 12345}
        self.assertEqual(machines.routes(row), [("203.0.113.9", "41022"), ("ssh5.vast.ai", 12345)])
        self.assertEqual(machines.routes({"ssh_host": "ssh5.vast.ai", "ssh_port": 12345}), [("ssh5.vast.ai", 12345)])


class MachineCheckTest(unittest.TestCase):
    def test_measures_how_far_a_product_strays_relative_to_the_largest_entry(self):
        self.assertEqual(machine_check.stray(1.0, 10.0), 0.1)

    def test_counts_a_product_that_is_not_finite_as_straying_without_bound(self):
        for value in (float("nan"), float("inf")):
            self.assertEqual(machine_check.stray(value, 10.0), machine_check.NOT_FINITE)

    def test_takes_the_worst_gpus_answers_from_the_nvcc_programs_output(self):
        found = machine_check.read_answers(
            "built\n[{\"float32_error\": 1e-07, \"float32_repeat_gap\": 0, \"bfloat16_error\": 1e-06, "
            "\"bfloat16_repeat_gap\": 0}, {\"float32_error\": 0.5, \"float32_repeat_gap\": 0, \"bfloat16_error\": "
            "1e+30, \"bfloat16_repeat_gap\": 2}]\n")
        self.assertEqual((found["answers_by"], found["float32_error"], found["bfloat16_error"],
                          found["bfloat16_repeat_gap"]), ("nvcc", 0.5, 1e30, 2))

    def test_reports_the_error_of_an_nvcc_program_that_could_not_answer(self):
        self.assertEqual(machine_check.read_answers("{\"answers_error\": \"cublasCreate(&handle) failed\"}\n"),
                         {"answers_error": "cublasCreate(&handle) failed"})

    def test_checks_no_products_on_an_image_without_nvcc(self):
        with patch.object(machine_check.shutil, "which", lambda name: None), \
                patch.object(machine_check.os.path, "exists", lambda path: False):
            self.assertIsNone(machine_check.answers_by_nvcc())

    def test_reads_nvidia_smis_rows_with_missing_values_as_none(self):
        rows = machine_check.parse_gpus("NVIDIA GeForce RTX 5090, 3, 32607, 450.00, 575.00, 600.00, 3090, [N/A], 16, "
                                        "35, 580.95.05\n")
        self.assertEqual(rows, [{"name": "NVIDIA GeForce RTX 5090", "memory_used_mib": 3.0, "memory_total_mib": 32607.0,
                                 "power_limit_w": 450.0, "power_default_w": 575.0, "power_max_w": 600.0,
                                 "sm_clock_max_mhz": 3090.0, "pcie_gen": None, "pcie_width": 16.0,
                                 "temperature_c": 35.0, "driver": "580.95.05"}])

    def test_takes_the_medians_of_the_sampled_clock_and_power(self):
        self.assertEqual(machine_check.parse_samples("2160, 449.12\n2145, 450.01\n2175, 448.90\n"), (2160.0, 449.12))

    def test_takes_the_faster_of_the_two_sources_as_the_machines_download_speed(self):
        speeds = {machine_check.DOWNLOADS["huggingface"]: 3.0, machine_check.DOWNLOADS["cloudflare"]: 80.0}
        self.assertEqual(machine_check.downloads(lambda url: speeds[url]),
                         ({"huggingface": 3.0, "cloudflare": 80.0}, 80.0))

    def test_reads_the_country_that_traffic_leaves_from_in_cloudflares_trace(self):
        self.assertEqual(machine_check.parse_trace("fl=1\ncolo=LAX\nloc=CN\ntls=TLSv1.3\n"), "CN")
        self.assertIsNone(machine_check.parse_trace("fl=1\ncolo=LAX\n"))


if __name__ == "__main__":
    unittest.main()
