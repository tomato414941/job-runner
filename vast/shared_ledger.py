"""Share the ledger's events through a Cloudflare R2 bucket, one small object an event, so that the sessions on every
machine see the same ledger.

The settings are LEDGER_R2_ENDPOINT (https://ACCOUNT.r2.cloudflarestorage.com), LEDGER_R2_BUCKET,
LEDGER_R2_ACCESS_KEY_ID and LEDGER_R2_SECRET_ACCESS_KEY, which machines.py takes from job-runner's .env. Without them
the ledger stays on the machine. Objects are named by the time they were uploaded, so that a listing that starts a
little before the newest object a machine has seen finds whatever has come since, from any machine.

Sharing never stops the work: an event that cannot be uploaded waits in the outbox folder beside the ledger for the
next try, a listing that fails leaves the ledger as it was, and either says so once on standard error.
"""

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timedelta, timezone

NAMES = ("LEDGER_R2_ENDPOINT", "LEDGER_R2_BUCKET", "LEDGER_R2_ACCESS_KEY_ID", "LEDGER_R2_SECRET_ACCESS_KEY")
PREFIX = "events/"
TIME_FORMAT = "%Y%m%dT%H%M%S%fZ"
# A listing starts this long before the newest object seen, so that an event uploaded late, or by a machine whose clock
# is behind, is still found; events seen twice are read once.
OVERLAP = timedelta(minutes=30)
# A process lists the bucket at most this often; the ledger is read many times while one machine is rented.
PULL_SECONDS = 30
TIMEOUT = 10
pulled_at = {}
warned = set()


def warn(what, error, detail=None):
    """Say once a process that sharing failed and the work goes on: the error's kind and code, never its message,
    which can name the bucket's address."""
    if what not in warned:
        warned.add(what)
        code = getattr(error, "code", None)
        why = detail or f"{type(error).__name__}{f' {code}' if code else ''}"
        print(f"the shared ledger could not {what} ({why}); the ledger on this machine goes on", file=sys.stderr,
              flush=True)


def settings():
    """The bucket to share through, or None when the settings are missing or incomplete."""
    values = {name: os.environ.get(name, "").strip() for name in NAMES}
    if not all(values.values()):
        if any(values.values()):
            warn("start", None, "missing " + ", ".join(name for name, value in values.items() if not value))
        return None
    host = urllib.parse.urlsplit(values["LEDGER_R2_ENDPOINT"]).netloc or values["LEDGER_R2_ENDPOINT"]
    return {"host": host, "bucket": values["LEDGER_R2_BUCKET"], "key_id": values["LEDGER_R2_ACCESS_KEY_ID"],
            "secret": values["LEDGER_R2_SECRET_ACCESS_KEY"]}


# R2's S3 interface


def quote(text, safe="-_.~"):
    return urllib.parse.quote(text, safe=safe)


def signature(secret, stamp, method, path, query, headers, payload_hash, region="auto"):
    """AWS Signature Version 4 of one request, as R2's S3 interface checks it."""
    date = stamp[:8]
    canonical_query = "&".join(f"{quote(name)}={quote(value)}" for name, value in sorted(query.items()))
    names = sorted(headers)
    canonical = "\n".join([method, quote(path, safe="/-_.~"), canonical_query,
                           "".join(f"{name}:{headers[name].strip()}\n" for name in names), ";".join(names),
                           payload_hash])
    scope = f"{date}/{region}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = ("AWS4" + secret).encode()
    for part in (date, region, "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()


def request(config, method, key="", query=None, body=b""):
    """One signed request to the bucket, or to an object in it; the response's body."""
    query = query or {}
    path = f"/{config['bucket']}" + (f"/{key}" if key else "")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {"host": config["host"], "x-amz-content-sha256": payload_hash, "x-amz-date": stamp}
    signed = signature(config["secret"], stamp, method, path, query, headers, payload_hash)
    headers["authorization"] = (f"AWS4-HMAC-SHA256 Credential={config['key_id']}/{stamp[:8]}/auto/s3/aws4_request, "
                                "SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=" + signed)
    url = f"https://{config['host']}{quote(path, safe='/-_.~')}"
    if query:
        url += "?" + "&".join(f"{quote(name)}={quote(value)}" for name, value in sorted(query.items()))
    message = urllib.request.Request(url, data=body if method == "PUT" else None, method=method,
                                     headers={name: value for name, value in headers.items() if name != "host"})
    with urllib.request.urlopen(message, timeout=TIMEOUT) as response:
        return response.read()


def object_key(entry, moment=None):
    return f"{PREFIX}{(moment or datetime.now(timezone.utc)).strftime(TIME_FORMAT)}-{entry['id']}.json"


def entry_id(key):
    """The id of the event in an object's name."""
    return key[len(PREFIX):].split("-", 1)[1].removesuffix(".json")


def start_after(newest):
    """Where a listing starts, OVERLAP before the newest object seen; None to list from the first."""
    if not newest:
        return None
    moment = datetime.strptime(newest[len(PREFIX):].split("-", 1)[0], TIME_FORMAT).replace(tzinfo=timezone.utc)
    return PREFIX + (moment - OVERLAP).strftime(TIME_FORMAT)


def listing(config, after=None):
    """The names of the bucket's events after `after`, oldest first, over every page."""
    keys, token = [], None
    while True:
        query = {"list-type": "2", "prefix": PREFIX}
        if after:
            query["start-after"] = after
        if token:
            query["continuation-token"] = token
        root = ElementTree.fromstring(request(config, "GET", query=query))
        named = {element.tag.rsplit("}", 1)[-1]: element for element in root}
        keys += [element.text for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "Key"]
        if named.get("IsTruncated") is None or named["IsTruncated"].text != "true":
            return keys
        token = named["NextContinuationToken"].text


# The ledger's folder: ledger.jsonl, the outbox of events not uploaded yet, and the newest object seen


def append(folder, entry):
    # One write of one line to a file opened for appending, so that the lines of concurrent writers do not mix.
    with open(folder / "ledger.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def known_ids(folder):
    try:
        lines = (folder / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return set()
    found = set()
    for line in lines:
        try:
            found.add(json.loads(line).get("id"))
        except json.JSONDecodeError:
            continue
    return found


def push(folder, config):
    """Upload the events waiting in the outbox, oldest first, until one fails."""
    outbox = folder / "outbox"
    if not outbox.is_dir():
        return
    for path in sorted(outbox.glob("*.json"), key=lambda path: path.stat().st_mtime):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            request(config, "PUT", object_key(entry), body=json.dumps(entry, ensure_ascii=False).encode())
        except Exception as error:  # noqa: BLE001 - an event that cannot be uploaded now waits for the next try
            warn("upload", error)
            return
        path.unlink(missing_ok=True)


def send(folder, entry, config):
    """Put a recorded event in the outbox and upload what waits there."""
    try:
        outbox = folder / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        (outbox / f"{entry['id']}.json").write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    except OSError as error:
        warn("keep an event to upload", error)
        return
    push(folder, config)


def pull(folder, config):
    """Upload what waits in the outbox, then add the bucket's events that the ledger lacks, at most once every
    PULL_SECONDS in a process."""
    moment = time.monotonic()
    if moment - pulled_at.get(folder, -PULL_SECONDS) < PULL_SECONDS:
        return
    pulled_at[folder] = moment
    push(folder, config)
    marker = folder / "newest-shared"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        newest = marker.read_text().strip() if marker.exists() else ""
        keys = listing(config, start_after(newest))
        known = known_ids(folder)
        for key in keys:
            if entry_id(key) not in known:
                entry = json.loads(request(config, "GET", key))
                append(folder, entry)
                known.add(entry.get("id"))
        if keys and max(keys) > newest:
            marker.write_text(max(keys) + "\n")
    except Exception as error:  # noqa: BLE001 - a ledger that cannot be brought up to date is still read
        warn("be read", error)
