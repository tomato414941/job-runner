import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from job_runner.controller import Controller, etag
from job_runner.decide import Install, Keep, KeepFailed, Reject, RentFailed, Send, Start
from job_runner.local import LocalStore
from job_runner.provider import RentError

PREFIX = "runs/run"


class StubProvider:
    """Lists what the test put in `listed`, answers each command with what it put in `answers`, keeps what was run,
    and rents nothing."""

    def __init__(self):
        self.ran, self.answers, self.listed, self.rented, self.rejected = [], {}, [], [], []

    def machines(self, label):
        return self.listed

    def rent(self, label, needs, hours, avoid=()):
        self.rented.append((label, avoid))
        raise RentError("the image did not start")

    def reject(self, ident, reason):
        self.rejected.append((ident, reason))

    def run(self, ident, command, data, timeout):
        self.ran.append((ident, command, data))
        for word, answer in self.answers.items():
            if command.endswith(word):
                return answer
        return 0, "{}"


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "code.tar.gz").write_bytes(b"code")
        (self.root / "spec.json").write_text(json.dumps({
            "name": "run", "provider": {}, "store": {"prefix": PREFIX}, "code": {"path": "code.tar.gz"},
            "files": [{"path": "data/a.bin", "key": "data/a.bin", "sha256": "a", "bytes": 4}],
            "setup": [["sh", "setup.sh"]], "check": [["python3", "check.py"]],
            "steps": [{"name": "first", "command": ["x"], "inputs": ["data/a.bin"], "outputs": ["out/a.pt"]}],
            "policy": {"part_bytes": 4},
        }))
        self.store = LocalStore(max_put=8)
        self.provider = StubProvider()
        self.controller = Controller(self.root, self.provider, self.store)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def events(self):
        return [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]

    def test_puts_the_code_in_the_store_and_refuses_to_start_while_a_file_the_steps_read_is_missing(self):
        with self.assertRaisesRegex(SystemExit, "data/a.bin"):
            self.controller.prepare()
        self.assertEqual(self.store.read(f"{PREFIX}/code/{hashlib.sha256(b'code').hexdigest()}.tar.gz"), b"code")
        self.store.write("data/a.bin", b"data")
        self.controller.prepare()

    def test_finds_what_would_keep_the_run_from_starting_without_writing_anything(self):
        self.store.expires = time.time() + self.controller.spec.policy.url - 60
        self.provider.listed = None
        self.assertEqual(self.controller.check(), [
            "the store lacks data/a.bin", "the store's key expires within 21600 seconds",
            "the provider does not list the machines"])
        self.assertEqual(self.store.objects, {})

    def test_stops_before_the_store_key_would_expire_within_a_url_it_gives(self):
        self.store.write("data/a.bin", b"data")
        self.store.expires = time.time() + self.controller.spec.policy.url - 60
        self.assertFalse(self.controller.run())
        self.assertEqual(self.provider.rented, [])
        self.assertEqual(self.events()[-1]["event"], "key_running_out")

    def test_looks_again_later_when_the_provider_does_not_list_the_machines(self):
        self.provider.listed = None
        self.controller.tick()
        self.controller.tick()
        self.assertEqual(self.provider.rented, [])
        self.assertEqual([event["event"] for event in self.events()], ["unlisted"])

    def test_tells_decide_when_renting_failed_otherwise_than_for_want_of_an_offer(self):
        self.controller.tick()
        self.assertEqual(self.controller.facts, [RentFailed("the image did not start")])

    def test_gives_the_machine_its_check_with_the_install_and_with_each_start(self):
        self.controller.code = {"key": "code", "sha256": "c", "bytes": 4}
        self.controller.install(Install("m1"))
        self.controller.start(Start("m1"))
        # The agent's own files go first, as a tarball; the commands' payloads are JSON.
        install, start = (json.loads(data) for _, _, data in self.provider.ran if data and data.startswith(b"{"))
        self.assertEqual((install["setup"], install["check"]), ([["sh", "setup.sh"]], [["python3", "check.py"]]))
        self.assertEqual(start["check"], [["python3", "check.py"]])

    def test_takes_up_a_request_to_stop_written_beside_the_run(self):
        (self.root / "stop").write_text(json.dumps({"reason": "the spec is wrong"}))
        self.controller.tick()
        self.assertEqual((self.controller.memory.outcome, self.controller.memory.detail, self.controller.memory.ended),
                         ("stopped", "the spec is wrong", True))
        self.assertEqual(self.provider.rented, [])

    def test_has_the_provider_reject_a_machine_already_released(self):
        self.controller.reject(Reject("m1", "its install failed, which another machine of the run passed"))
        self.assertEqual(self.provider.rejected, [("m1", "its install failed, which another machine of the run passed")])
        self.assertEqual(self.events()[-1]["event"], "rejected")

    def test_takes_a_machine_without_the_agent_for_one_without_the_job(self):
        self.provider.answers["status"] = (127, "")
        self.assertEqual(self.controller.status("m1"), {"phase": "empty"})

    def test_takes_a_refused_command_for_one_not_done(self):
        self.controller.code = {"key": "code", "sha256": "c", "bytes": 4}
        self.provider.answers["install"] = (1, json.dumps({"error": "Busy", "detail": "the agent is installing"}))
        self.controller.install(Install("m1"))
        self.assertEqual([(event["event"], event.get("error")) for event in self.events()], [("refused", "Busy")])

    def test_sends_a_file_larger_than_one_put_in_parts(self):
        self.controller.reports["m1"] = {"outbox": [{"id": "o1", "kind": "output", "step": "first",
                                                     "path": "out/a.pt", "bytes": 10}]}
        self.controller.send(Send("m1", ("o1",)))
        upload = json.loads(self.provider.ran[-1][2])["uploads"][0]
        self.assertEqual([(part["number"], part["start"], part["length"]) for part in upload["parts"]],
                         [(1, 0, 4), (2, 4, 4), (3, 8, 2)])
        self.assertIn(upload["upload"], self.store.uploads)

    def test_finishes_an_upload_in_parts_over_an_older_file_of_the_same_size_at_its_key(self):
        key = f"{PREFIX}/outputs/out/a.pt"
        self.store.write(key, b"old bytes!")
        upload = self.store.start_upload(key)
        parts = [b"new ", b"byte", b"s!"]
        self.store.uploads[upload]["parts"] = dict(enumerate(parts, 1))
        md5s = [[number, hashlib.md5(part).hexdigest()] for number, part in enumerate(parts, 1)]
        self.controller.reports["m1"] = {"outbox": [{
            "id": "o1", "kind": "output", "step": "first", "path": "out/a.pt", "bytes": 10, "sha256": "s",
            "state": "sent", "upload": upload, "etags": md5s, "outputs": ["out/a.pt"], "at": 1.0}]}

        self.controller.keep(Keep("m1", ("o1",)))

        self.assertEqual(self.store.read(key), b"new bytes!")
        self.assertEqual(self.store.etag(key), etag(md5s))
        self.assertEqual(self.controller.record.finished(self.controller.spec), ["first"])
        self.assertEqual(self.controller.facts, [])

    def test_does_not_record_a_file_the_store_does_not_hold(self):
        self.controller.reports["m1"] = {"outbox": [{
            "id": "o1", "kind": "output", "step": "first", "path": "out/a.pt", "bytes": 10, "sha256": "s",
            "state": "sent", "upload": None, "etags": [[0, hashlib.md5(b"never sent").hexdigest()]],
            "outputs": ["out/a.pt"], "at": 1.0}]}

        self.controller.keep(Keep("m1", ("o1",)))

        self.assertEqual(self.controller.facts, [KeepFailed("m1", "o1")])
        self.assertEqual(self.controller.record.finished(self.controller.spec), [])


if __name__ == "__main__":
    unittest.main()
