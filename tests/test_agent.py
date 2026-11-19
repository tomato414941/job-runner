"""The agent's daemon sending its outbox to a stand-in for the store."""

import hashlib
import importlib
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from job_runner import agent

PART = 1024


def md5(data):
    return hashlib.md5(data).hexdigest()


class SendTests(unittest.TestCase):
    """A daemon in a home of its own, and a store that keeps which parts it was sent, in the order they came, and
    answers each as `answer(name, number, body)` says: with the ETag it gives, or refusing with the status it gives."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with mock.patch.dict(os.environ, {"JOB_RUNNER_HOME": str(Path(self.directory.name) / "home")}):
            importlib.reload(agent)
        self.daemon = agent.Daemon()
        self.asked, self.answer = [], lambda name, number, body: md5(body)
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_PUT(self):
                name, number = self.path.lstrip("/").split("/")
                body = self.rfile.read(int(self.headers["Content-Length"]))
                test.asked.append((name, int(number), body))
                answer = test.answer(name, int(number), body)
                self.send_response(answer if isinstance(answer, int) else 200)
                if not isinstance(answer, int):
                    self.send_header("ETag", f'"{answer}"')
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def held(self, name, parts):
        """Put a file of so many parts in the outbox: its bytes, and its upload as a controller authorizes it."""
        data = os.urandom(parts * PART)
        source = Path(self.directory.name) / name
        source.write_bytes(data)
        self.daemon.enqueue("output", "step", source, name)
        ident = next(entry["id"] for entry in self.daemon.status()["outbox"] if entry["path"] == name)
        base = f"http://127.0.0.1:{self.server.server_port}/{name}"
        return data, {"id": ident, "upload": f"upload of {name}", "parts": [
            {"number": number, "start": (number - 1) * PART, "length": PART, "url": f"{base}/{number}"}
            for number in range(1, parts + 1)]}

    def send(self, *uploads):
        """Have the daemon send the uploads, and give its outbox by file once none is on its way any more."""
        self.daemon.handle("send", {"uploads": list(uploads)})
        ended = time.monotonic() + 30
        while time.monotonic() < ended:
            outbox = {entry["path"]: entry for entry in self.daemon.status()["outbox"]}
            if all(entry["state"] != "sending" for entry in outbox.values()):
                return outbox
            time.sleep(0.01)
        self.fail("the daemon did not end its uploads in time")

    def test_sends_the_parts_of_a_file_over_several_connections_at_once_and_tells_their_etags_in_order(self):
        data, upload = self.held("a.bin", agent.CONNECTIONS)
        together = threading.Barrier(agent.CONNECTIONS, timeout=10)

        def answer(name, number, body):
            # No part is answered until as many are under way as the agent has connections.
            try:
                together.wait()
            except threading.BrokenBarrierError:
                return 403
            return md5(body)

        self.answer = answer

        entry = self.send(upload)["a.bin"]

        self.assertEqual((entry["state"], entry["upload"]), ("sent", "upload of a.bin"))
        self.assertEqual(entry["etags"], [[number, md5(data[(number - 1) * PART:number * PART])]
                                          for number in range(1, agent.CONNECTIONS + 1)])
        self.assertEqual(b"".join(body for _, _, body in sorted(self.asked)), data)

    def test_fails_a_file_when_the_store_received_other_bytes_for_one_of_its_parts(self):
        _, upload = self.held("a.bin", 3)
        self.answer = lambda name, number, body: md5(b"other bytes" if number == 2 else body)

        entry = self.send(upload)["a.bin"]

        self.assertEqual((entry["state"], entry["error"]), ("failed", "ValueError"))

    def test_sends_one_file_at_a_time_and_fails_alone_the_one_whose_part_the_store_refuses(self):
        _, first = self.held("a.bin", 3)
        data, second = self.held("b.bin", 3)
        self.answer = lambda name, number, body: 403 if (name, number) == ("a.bin", 2) else md5(body)

        outbox = self.send(first, second)

        self.assertEqual((outbox["a.bin"]["state"], outbox["a.bin"]["error"]), ("failed", "HTTPError"))
        self.assertEqual(outbox["b.bin"]["state"], "sent")
        self.assertEqual(outbox["b.bin"]["etags"], [[number, md5(data[(number - 1) * PART:number * PART])]
                                                    for number in (1, 2, 3)])
        # Every part of the first file came to the store before any of the second.
        self.assertEqual([name for name, _, _ in self.asked], ["a.bin"] * 3 + ["b.bin"] * 3)

    def test_leaves_the_parts_not_yet_begun_unsent_once_the_store_refuses_one(self):
        parts = 10 * agent.CONNECTIONS
        _, upload = self.held("a.bin", parts)
        refused = threading.Event()

        def answer(name, number, body):
            if number == 1:
                refused.set()
                return 403
            # The parts under way with the refused one end only after the refusal has had time to stop the rest.
            refused.wait(10)
            time.sleep(0.2)
            return md5(body)

        self.answer = answer

        entry = self.send(upload)["a.bin"]

        self.assertEqual((entry["state"], entry["error"]), ("failed", "HTTPError"))
        self.assertLess(len(self.asked), parts)


if __name__ == "__main__":
    unittest.main()
