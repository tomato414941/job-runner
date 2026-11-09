"""Checks of shared_ledger.py, and of how machines.py shares its ledger, against a bucket kept in memory.

Run in this folder: python3 -m unittest
"""

import contextlib
import io
import json
import os
import tempfile
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

# The Vast.ai part runs as scripts beside each other in vast/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vast"))

import shared_ledger
import machines

SETTINGS = {"LEDGER_R2_ENDPOINT": "https://account.r2.example", "LEDGER_R2_BUCKET": "ledger",
            "LEDGER_R2_ACCESS_KEY_ID": "key", "LEDGER_R2_SECRET_ACCESS_KEY": "secret"}
EMPTY = shared_ledger.hashlib.sha256(b"").hexdigest()


class Bucket:
    """An R2 bucket in memory behind shared_ledger.request; while `down`, every request fails as an unreachable one."""

    def __init__(self):
        self.objects, self.page, self.down, self.requests = {}, 1000, False, 0

    def request(self, config, method, key="", query=None, body=b""):
        self.requests += 1
        if self.down:
            raise OSError("unreachable")
        if method == "PUT":
            self.objects[key] = body
            return b""
        if key:
            return self.objects[key]
        query = query or {}
        keys = sorted(name for name in self.objects
                      if name.startswith(query.get("prefix", "")) and name > query.get("start-after", ""))
        start = int(query.get("continuation-token", 0))
        more = start + self.page < len(keys)
        contents = "".join(f"<Contents><Key>{name}</Key></Contents>" for name in keys[start:start + self.page])
        token = f"<NextContinuationToken>{start + self.page}</NextContinuationToken>" if more else ""
        return ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>{token}{contents}</ListBucketResult>").encode()

    def put(self, entry, moment):
        self.objects[shared_ledger.object_key(entry, moment)] = json.dumps(entry).encode()


def event(identity, **fields):
    return {"at": machines.stamp(), "event": "noted", "id": identity, "provider": "vast", **fields}


class SharingTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bucket = Bucket()
        self.stderr = io.StringIO()
        for change in (patch.object(shared_ledger, "request", self.bucket.request),
                       patch.object(shared_ledger, "PULL_SECONDS", 0), patch.object(shared_ledger, "pulled_at", {}),
                       patch.object(shared_ledger, "warned", set()), patch.dict(os.environ, SETTINGS),
                       contextlib.redirect_stderr(self.stderr)):
            change.__enter__()
            self.addCleanup(change.__exit__, None, None, None)

    @contextlib.contextmanager
    def machine(self, name):
        """machines as it runs on one machine, with the ledger beside its script."""
        here = self.root / name
        with patch.object(machines, "HERE", here), patch.object(machines, "folder", here / "data"):
            yield here / "data"


class SharingTest(SharingTestCase):
    def test_shares_an_event_recorded_on_one_machine_with_the_ledger_of_another(self):
        with self.machine("a"):
            machines.record("unusable", provider="vast", instance=1, machine=54650, host=10, stage="start",
                               detail="the machine had not begun to load the image after 5 minutes")
        self.assertEqual(len(self.bucket.objects), 1)
        with self.machine("b"):
            self.assertEqual(machines.avoided(machines.entries()), {("vast", 54650)})

    def test_goes_on_when_the_bucket_cannot_be_reached_and_uploads_the_event_later(self):
        self.bucket.down = True
        with self.machine("a") as data:
            rented = machines.record("rented", provider="vast", instance=7, machine=7, host=7, gpu="RTX 5090",
                                        price=0.5, deadline=machines.stamp())
            machines.record("noted", provider="vast", instance=7, note="first try")
            self.assertEqual([entry["id"] for entry in machines.entries()][:1], [rented["id"]])
            self.assertEqual(len(list((data / "outbox").glob("*.json"))), 2)
            self.bucket.down = False
            machines.entries()
            self.assertEqual(list((data / "outbox").glob("*.json")), [])
        self.assertEqual(len(self.bucket.objects), 2)
        self.assertEqual(self.stderr.getvalue().count("could not upload"), 1)
        self.assertNotIn("account.r2.example", self.stderr.getvalue())

    def test_sends_back_a_machine_at_its_deadline_while_the_bucket_cannot_be_reached(self):
        self.bucket.down = True
        with self.machine("a"):
            machines.record("rented", provider="vast", instance=7, machine=7, host=7, gpu="RTX 5090", price=0.5,
                               deadline=machines.stamp(machines.now() - timedelta(minutes=1)),
                               workstation=machines.workstation())
            destroyed = []
            with patch.object(machines, "destroy", destroyed.append):
                machines.watch_round()
            self.assertTrue(machines.rental(machines.entries(), 7)["returned"])
        self.assertEqual(destroyed, [7])

    def test_never_touches_the_bucket_for_a_ledger_that_data_names(self):
        with patch.object(machines, "folder", self.root / "elsewhere"):
            machines.record("noted", provider="vast", instance=7, note="trial")
            machines.entries()
        self.assertEqual(self.bucket.requests, 0)

    def test_reads_each_event_once_when_two_processes_pulled_it(self):
        self.bucket.put(event("a" * 32, instance=1), datetime.now(timezone.utc))
        with self.machine("b") as data:
            machines.entries()
            with open(data / "ledger.jsonl", "a") as stream:
                stream.write((data / "ledger.jsonl").read_text())
            self.assertEqual([entry["id"] for entry in machines.entries()], ["a" * 32])

    def test_finds_an_event_that_a_machine_whose_clock_is_behind_uploaded_later(self):
        moment = datetime.now(timezone.utc)
        self.bucket.put(event("a" * 32, instance=1), moment)
        with self.machine("b"):
            self.assertEqual([entry["id"] for entry in machines.entries()], ["a" * 32])
            self.bucket.put(event("b" * 32, instance=2), moment - timedelta(minutes=10))
            self.assertEqual([entry["id"] for entry in machines.entries()], ["a" * 32, "b" * 32])


class BucketTest(SharingTestCase):
    def test_lists_every_page_of_a_long_listing(self):
        self.bucket.page = 2
        moment = datetime.now(timezone.utc)
        for index in range(5):
            self.bucket.put(event(f"{index:032x}"), moment + timedelta(seconds=index))
        keys = shared_ledger.listing(shared_ledger.settings())
        self.assertEqual([shared_ledger.entry_id(key) for key in keys], [f"{index:032x}" for index in range(5)])

    def test_shares_only_when_every_setting_is_there(self):
        self.assertEqual(shared_ledger.settings()["host"], "account.r2.example")
        with patch.dict(os.environ, {"LEDGER_R2_BUCKET": ""}):
            self.assertIsNone(shared_ledger.settings())
        with patch.dict(os.environ, {name: "" for name in shared_ledger.NAMES}):
            self.assertIsNone(shared_ledger.settings())
        self.assertIn("missing LEDGER_R2_BUCKET", self.stderr.getvalue())

    def test_signs_like_the_aws_reference_example_of_getting_an_object(self):
        signature = shared_ledger.signature(
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "20130524T000000Z", "GET", "/test.txt", {},
            {"host": "examplebucket.s3.amazonaws.com", "range": "bytes=0-9", "x-amz-content-sha256": EMPTY,
             "x-amz-date": "20130524T000000Z"}, EMPTY, region="us-east-1")
        self.assertEqual(signature, "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")

    def test_signs_like_the_aws_reference_example_of_listing_a_bucket(self):
        signature = shared_ledger.signature(
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "20130524T000000Z", "GET", "/", {"max-keys": "2", "prefix": "J"},
            {"host": "examplebucket.s3.amazonaws.com", "x-amz-content-sha256": EMPTY,
             "x-amz-date": "20130524T000000Z"}, EMPTY, region="us-east-1")
        self.assertEqual(signature, "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7")


if __name__ == "__main__":
    unittest.main()
