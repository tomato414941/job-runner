import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from rental.controller import Controller, etag
from rental.decide import Install, Keep, KeepFailed, RentFailed, Send
from rental.local import LocalStore
from rental.provider import RentError

PREFIX = "runs/run"


class StubProvider:
    """Lists what the test put in `listed`, answers each command with what it put in `answers`, keeps what was run,
    and rents nothing."""

    def __init__(self):
        self.ran, self.answers, self.listed, self.rented = [], {}, [], []

    def machines(self, label):
        return self.listed

    def rent(self, label, needs, hours):
        self.rented.append(label)
        raise RentError("the image did not start")

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
