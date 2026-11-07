"""The adapters to the world outside: R2's signatures and requests, and Vast.ai through vast-rental and ssh."""

import calendar
import json
import os
import stat
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from rental.provider import RentError, Unreachable
from rental.r2 import EMPTY, R2Store, authorization, presign
from rental.vast import VastProvider

# AWS's own examples of Signature Version 4 for S3, with their published signatures.
EXAMPLE = {"id": "AKIAIOSFODNN7EXAMPLE", "secret": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}
MOMENT = time.gmtime(calendar.timegm((2013, 5, 24, 0, 0, 0)))
HOST = "examplebucket.s3.amazonaws.com"


class SignatureTests(unittest.TestCase):
    def test_presigns_a_url_as_aws_example_does(self):
        url = presign(EXAMPLE, "us-east-1", MOMENT, "GET", HOST, "/test.txt", 86400)
        self.assertTrue(url.endswith(
            "&X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"))

    def test_signs_a_ranged_get_as_aws_example_does(self):
        headers = authorization(EXAMPLE, "us-east-1", MOMENT, "GET", HOST, "/test.txt", None,
                                {"Range": "bytes=0-9"}, EMPTY)
        self.assertTrue(headers["authorization"].endswith(
            "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
            "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"))

    def test_signs_a_put_with_an_encoded_key_as_aws_example_does(self):
        headers = authorization(
            EXAMPLE, "us-east-1", MOMENT, "PUT", HOST, "/test%24file.text", None,
            {"Date": "Fri, 24 May 2013 00:00:00 GMT", "x-amz-storage-class": "REDUCED_REDUNDANCY"},
            "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072")
        self.assertTrue(headers["authorization"].endswith(
            "Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"))


class R2StoreTests(unittest.TestCase):
    """R2Store's requests against a stand-in that answers as S3 does and keeps what it was asked."""

    def setUp(self):
        self.asked, objects = [], {}
        asked = self.asked

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def respond(self, code, body=b"", headers=()):
                self.send_response(code)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                asked.append(("POST", urlsplit(self.path).path, sorted(query), body,
                              self.headers["Authorization"].startswith("AWS4-HMAC-SHA256 ")))
                if "uploads" in query:
                    self.respond(200, b'<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/'
                                      b'2006-03-01/"><UploadId>up1</UploadId></InitiateMultipartUploadResult>')
                else:
                    self.respond(200, b"<CompleteMultipartUploadResult/>")

            def do_PUT(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                objects[urlsplit(self.path).path] = body
                self.respond(200, headers=[("ETag", '"e1"')])

            def do_HEAD(self):
                body = objects.get(urlsplit(self.path).path)
                if body is None:
                    return self.respond(404)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("ETag", '"e1"')
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.store = R2Store("bucket", {"AWS_ENDPOINT_URL": f"http://127.0.0.1:{self.server.server_port}",
                                        "AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "secret",
                                        "AWS_SESSION_TOKEN": "token"})

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_starts_and_finishes_an_upload_in_parts(self):
        upload = self.store.start_upload("runs/a b.pt")
        self.store.finish_upload("runs/a b.pt", upload, [[1, "m1"], [2, "m2"]])
        self.assertEqual(upload, "up1")
        (method, path, query, _, signed), (_, _, finish_query, body, _) = self.asked
        self.assertEqual((method, path, query, signed), ("POST", "/bucket/runs/a%20b.pt", ["uploads"], True))
        self.assertEqual(finish_query, ["uploadId"])
        self.assertIn(b'<PartNumber>2</PartNumber><ETag>"m2"</ETag>', body)

    def test_tells_the_size_and_etag_of_what_it_holds_and_none_of_what_it_does_not(self):
        self.assertIsNone(self.store.size("runs/record.json"))
        self.store.write("runs/record.json", b"{}")
        self.assertEqual((self.store.size("runs/record.json"), self.store.etag("runs/record.json")), (2, "e1"))

    def test_knows_when_a_temporary_key_expires_and_that_another_does_not(self):
        self.assertIsNone(self.store.expires)
        store = R2Store("bucket", {"AWS_ENDPOINT_URL": "http://127.0.0.1:1", "AWS_ACCESS_KEY_ID": "id",
                                   "AWS_SECRET_ACCESS_KEY": "secret", "AWS_SESSION_TOKEN": "token",
                                   "AWS_CREDENTIAL_EXPIRATION": "2026-11-08T03:00:00Z"})
        self.assertEqual(store.expires, calendar.timegm((2026, 11, 8, 3, 0, 0)))

    def test_presigns_part_urls_with_the_upload_and_the_temporary_key(self):
        url = self.store.presign("PUT", "runs/a.pt", 60, {"partNumber": 3, "uploadId": "up1"})
        query = parse_qs(urlsplit(url).query)
        self.assertEqual((query["partNumber"], query["uploadId"], query["X-Amz-Security-Token"]),
                         (["3"], ["up1"], ["token"]))


class VastProviderTests(unittest.TestCase):
    """VastProvider through stand-ins for vast-rental and ssh on its PATH, which answer as the real ones do."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.calls = self.root / "calls.jsonl"
        self.script("vast-rental", """
command, label = sys.argv[1], sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else None
if command == "rent" and label == "broken":
    sys.exit(print("the image did not start, and the machine went back", file=sys.stderr) or 2)
if command == "rent" and label == "scarce":
    sys.exit(print(json.dumps({"rented": None, "offers": 0, "failed": 0, "failed_dollars": 0.0})) or 1)
if command == "list" and label == "unlisted":
    sys.exit(1)
answers = {
    "rent": [{"rented": 42, "ssh_config": "/tmp/42.config", "failed": 1, "failed_dollars": 0.07}],
    "list": [{"instance": 42, "label": label, "status": "running", "deadline": "2026-11-07T06:00:00+00:00",
              "hours": 1.0, "dollars": 0.5}],
    "extend": [{"instance": 42, "deadline": "2026-11-07T12:00:00+00:00"}],
    "return": [{"returned": 42, "hours": 1.5, "dollars": 0.9}],
    "ssh-config": [{"ssh_config": "/tmp/42.config", "ssh_host": "vast-42"}],
}
for answer in answers.get(command, []):
    print(json.dumps(answer))
""")
        self.script("ssh", """
command = sys.argv[-1]
if command == "unreachable":
    sys.exit(255)
print(json.dumps({"stdin": sys.stdin.read()}))
sys.exit(3 if command == "fails" else 0)
""")
        self.provider = VastProvider(environment={"VAST_SSH_KEY_FILE": "/keys/vast",
                                                  "PATH": f"{self.root}:{os.environ['PATH']}"})

    def tearDown(self):
        self.directory.cleanup()

    def script(self, name, body):
        path = self.root / name
        path.write_text(f"#!/usr/bin/env python3\nimport json, sys\n"
                        f"open({str(self.calls)!r}, 'a').write(json.dumps([{name!r}, *sys.argv[1:]]) + '\\n')\n"
                        + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def called(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_rents_through_vast_rental_and_reports_the_dollars_its_checks_spent(self):
        needs = {"gpu": "RTX 5090", "image": "image", "disk": 200, "max_price": 0.64, "query": {"a": 1}}
        self.assertEqual(self.provider.rent("run", needs, 6.5), ("42", 0.07))
        self.assertEqual(self.called()[0], ["vast-rental", "rent", "RTX 5090", "--image", "image", "--label", "run",
                                            "--hours", "6.5", "--disk", "200", "--max-price", "0.64", "--tries", "6",
                                            "--query", '{"a": 1}'])
        self.assertEqual(self.provider.rent("scarce", needs, 6), (None, 0.0))
        self.provider.rent("run", needs, 6, avoid=("7", "9"))
        rent = self.called()[-1]
        self.assertEqual(rent[rent.index("--avoid") + 1], "7,9")

    def test_tells_a_rent_that_failed_from_one_without_an_offer(self):
        needs = {"gpu": "RTX 5090", "image": "image", "disk": 200, "max_price": 0.64}
        with self.assertRaisesRegex(RentError, "the image did not start"):
            self.provider.rent("broken", needs, 6)

    def test_lists_the_machines_with_their_cost_and_deadline_or_none_when_vast_rental_cannot(self):
        self.assertEqual(self.provider.machines("run"), [
            {"id": "42", "cost": 0.5, "deadline": calendar.timegm((2026, 11, 7, 6, 0, 0))}])
        self.assertIsNone(self.provider.machines("unlisted"))

    def test_runs_a_command_over_ssh_with_its_input_and_takes_ssh_failure_for_no_answer(self):
        self.assertEqual(self.provider.run("42", "fails", b"payload", 30), (3, '{"stdin": "payload"}\n'))
        with self.assertRaises(Unreachable):
            self.provider.run("42", "unreachable", None, 30)
        ssh = self.called()[1]
        self.assertEqual(ssh[1:3], ["-F", "/tmp/42.config"])
        self.assertIn("IdentitiesOnly=yes", ssh)

    def test_moves_a_deadline_and_returns_a_machine_with_its_dollars(self):
        self.assertEqual(self.provider.extend("42", 5.5), calendar.timegm((2026, 11, 7, 12, 0, 0)))
        self.provider.reject("42", "it stopped answering")
        self.assertEqual(self.provider.release("42"), 0.9)
        self.assertEqual(self.called(), [["vast-rental", "extend", "42", "5.5"],
                                         ["vast-rental", "reject", "42", "it stopped answering"],
                                         ["vast-rental", "return", "42"]])


if __name__ == "__main__":
    unittest.main()
