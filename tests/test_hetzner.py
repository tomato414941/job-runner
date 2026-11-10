"""HetznerProvider against a stand-in for Hetzner Cloud's API kept in memory, and a stand-in for ssh on its PATH."""

import contextlib
import io
import json
import os
import stat
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from job_runner import hetzner
from job_runner.hetzner import DEADLINE, LABEL, HetznerProvider
from job_runner.provider import RentError, Unreachable, state


class Cloud:
    """The servers a stand-in API holds, and the requests it took."""

    def __init__(self):
        self.servers, self.requests, self.next = {}, [], 1

    def server(self, body):
        location = body["location"]
        server = {"id": self.next, "name": body["name"], "status": "running", "labels": body["labels"],
                  "created": datetime.now(UTC).isoformat(), "public_net": {"ipv4": {"ip": "127.0.0.1"}},
                  "location": {"name": location},
                  "server_type": {"name": body["server_type"],
                                  "prices": [{"location": location, "price_hourly": {"gross": "0.0500"}}]}}
        self.servers[self.next] = server
        self.next += 1
        return server


def handler(cloud):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, code, body=None):
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length)) if length else None

        def handle_one(self):
            parts = urlsplit(self.path)
            body = self.body()
            cloud.requests.append((self.command, parts.path, body))
            ident = int(parts.path.rsplit("/", 1)[1]) if parts.path.count("/") == 2 else None
            if self.command == "POST":
                if body["server_type"] == "scarce":
                    return self.answer(412, {"error": {"code": "resource_unavailable"}})
                if body["server_type"] == "forbidden":
                    return self.answer(403, {"error": {"code": "forbidden"}})
                return self.answer(201, {"server": cloud.server(body)})
            if self.command == "GET" and ident is None:
                selector = parse_qs(parts.query)["label_selector"][0]
                key, _, value = selector.partition("==")
                return self.answer(200, {"servers": [server for server in cloud.servers.values()
                                                     if key in server["labels"]
                                                     and (not value or server["labels"][key] == value)]})
            if ident not in cloud.servers:
                return self.answer(404, {"error": {"code": "not_found"}})
            if self.command == "GET":
                return self.answer(200, {"server": cloud.servers[ident]})
            if self.command == "PUT":
                cloud.servers[ident]["labels"] = body["labels"]
                return self.answer(200, {"server": cloud.servers[ident]})
            if self.command == "DELETE":
                del cloud.servers[ident]
                return self.answer(200, {"action": {"status": "running"}})

        do_GET = do_POST = do_PUT = do_DELETE = handle_one

    return Handler


class HetznerProviderTests(unittest.TestCase):
    def setUp(self):
        self.cloud = Cloud()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.cloud))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.environment = {"HCLOUD_TOKEN": "token", "HCLOUD_URL": f"http://127.0.0.1:{self.server.server_port}",
                            "HETZNER_SSH_KEY_FILE": "/keys/hetzner", "PATH": f"{self.root}:{os.environ['PATH']}"}
        self.home = patch.dict(os.environ, {"HOME": str(self.root)})
        self.home.start()
        # The watcher has just finished a round.
        state("hetzner").mkdir(parents=True)
        (state("hetzner") / "watched").touch()
        self.provider = HetznerProvider(self.environment)
        self.needs = {"server_type": "cpx51", "location": "hel1", "image": "ubuntu-24.04", "ssh_key": "jobs"}

    def tearDown(self):
        self.home.stop()
        self.server.shutdown()
        self.server.server_close()
        self.directory.cleanup()

    def rent(self, needs=None, hours=2.0):
        with patch.object(HetznerProvider, "answers", return_value=True):
            return self.provider.rent("run", needs or self.needs, hours)

    def test_rents_a_server_labelled_with_its_run_and_deadline_or_none_when_there_is_no_room(self):
        self.assertEqual(self.rent(), ("1", 0.0))
        method, path, body = self.cloud.requests[0]
        self.assertEqual((method, path, body["ssh_keys"], body["labels"][LABEL]), ("POST", "/servers", ["jobs"], "run"))
        self.assertAlmostEqual(int(body["labels"][DEADLINE]), time.time() + 7200, delta=60)
        self.assertEqual(self.rent({**self.needs, "server_type": "scarce"}), (None, 0.0))
        with self.assertRaisesRegex(RentError, "forbidden"):
            self.rent({**self.needs, "server_type": "forbidden"})

    def test_deletes_a_server_that_never_answers_ssh(self):
        with patch.object(HetznerProvider, "answers", return_value=False), patch.object(hetzner, "START_SECONDS", 0.05), \
                patch.object(hetzner.time, "sleep"):
            with self.assertRaisesRegex(RentError, "did not answer ssh"):
                self.provider.rent("run", self.needs, 2.0)
        self.assertEqual(self.cloud.servers, {})

    def test_lists_its_runs_servers_with_the_hours_begun_and_their_deadline(self):
        ident, _ = self.rent()
        deadline = int(self.cloud.servers[1]["labels"][DEADLINE])
        self.assertEqual(self.provider.machines("run"), [{"id": ident, "cost": 0.05, "deadline": deadline}])
        self.cloud.servers[1]["created"] = (datetime.now(UTC) - timedelta(hours=2, minutes=10)).isoformat()
        self.assertEqual(self.provider.machines("run")[0]["cost"], 0.15)
        self.assertEqual(self.provider.machines("other"), [])
        self.assertIsNone(HetznerProvider({**self.environment, "HCLOUD_URL": "http://127.0.0.1:9"}).machines("run"))

    def test_moves_the_deadline_and_deletes_the_server_with_what_it_cost(self):
        ident, _ = self.rent()
        deadline = self.provider.extend(ident, 5.5)
        self.assertAlmostEqual(deadline, time.time() + 5.5 * 3600, delta=60)
        self.assertEqual(self.cloud.servers[1]["labels"], {LABEL: "run", DEADLINE: str(deadline)})
        self.assertEqual(self.provider.release(ident), 0.05)
        self.assertEqual(self.cloud.servers, {})
        self.assertIsNone(self.provider.release(ident))

    def test_rents_nothing_while_its_watcher_has_not_finished_a_round_lately(self):
        long_ago = time.time() - 600
        os.utime(state("hetzner") / "watched", (long_ago, long_ago))
        with self.assertRaisesRegex(RentError, "job-runner hetzner watch"):
            self.rent()
        (state("hetzner") / "watched").unlink()
        with self.assertRaisesRegex(RentError, "job-runner hetzner watch"):
            self.rent()
        self.assertEqual((self.cloud.servers, self.cloud.requests), ({}, []))

    def test_sweep_deletes_only_the_servers_past_their_deadline(self):
        self.rent()
        self.rent()
        self.cloud.servers[1]["labels"][DEADLINE] = str(int(time.time()) - 60)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.provider.sweep()
        self.assertEqual(list(self.cloud.servers), [2])
        self.assertEqual(json.loads(printed.getvalue()), {"deleted": 1, "label": "run", "euros": 0.05})

    def test_runs_a_command_as_root_over_ssh_and_takes_ssh_failure_for_no_answer(self):
        calls = self.root / "calls.jsonl"
        script = self.root / "ssh"
        script.write_text(f"#!/usr/bin/env python3\nimport json, sys\n"
                          f"open({str(calls)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                          "sys.exit(255 if sys.argv[-1] == 'unreachable' else 3)\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        ident, _ = self.rent()
        self.assertEqual(self.provider.run(ident, "fails", b"payload", 30), (3, ""))
        with self.assertRaises(Unreachable):
            self.provider.run(ident, "unreachable", None, 30)
        ssh = json.loads(calls.read_text().splitlines()[0])
        self.assertEqual(ssh[:2], ["-i", "/keys/hetzner"])
        self.assertEqual(ssh[-2:], ["root@127.0.0.1", "fails"])
        self.assertIn(f"UserKnownHostsFile={self.root / '.local/state/job-runner/hetzner/known_hosts' / ident}", ssh)


if __name__ == "__main__":
    unittest.main()
