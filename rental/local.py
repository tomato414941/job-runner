"""A provider and a store on this machine, for rehearsing a run's whole life without renting or storing anything.

LocalStore answers presigned URLs as R2 does in what a run depends on: a URL is
good for one object, one operation and its parts until it expires; one PUT
takes at most max_put bytes; a range comes back as asked; parts are joined when
the upload is finished. LocalProvider's machines are directories here, each its
own HOME, whose processes it can kill and whose speed it sets; it returns a
machine at its deadline, as a provider's watcher does.
"""

import hashlib
import hmac
import os
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .controller import Unreachable

# Every URL the local store makes carries this, so that a rehearsal can look for one written down.
SIGNATURE = "LocalSignature"


class LocalStore:
    def __init__(self, max_put=64 * 1024):
        self.max_put, self.objects, self.etags, self.uploads, self.refused, self.joined = max_put, {}, {}, {}, [], 0
        self.secret, self.lock = os.urandom(16), threading.Lock()
        store = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                store.answer(self)

            def do_PUT(self):
                store.answer(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def sign(self, *values):
        return hmac.new(self.secret, "\n".join(map(str, values)).encode(), hashlib.sha256).hexdigest()

    def presign(self, method, key, seconds, query=None):
        query = query or {}
        expires = round(time.time() + seconds, 3)
        part, upload = query.get("partNumber", ""), query.get("uploadId", "")
        signature = self.sign(method, key, part, upload, expires)
        return (f"{self.base}/{quote(key)}?method={method}&partNumber={part}&uploadId={upload}"
                f"&expires={expires}&{SIGNATURE}={signature}")

    def answer(self, handler):
        address = urlsplit(handler.path)
        key = unquote(address.path.lstrip("/"))
        query = {name: values[0] for name, values in parse_qs(address.query, keep_blank_values=True).items()}
        part, upload = query.get("partNumber", ""), query.get("uploadId", "")
        signature = self.sign(handler.command, key, part, upload, query.get("expires", ""))
        if query.get("method") != handler.command or not hmac.compare_digest(signature, query.get(SIGNATURE, "")):
            return self.refuse(handler, 403, key, "signature")
        if time.time() > float(query["expires"]):
            return self.refuse(handler, 403, key, "expired")
        if handler.command == "GET":
            return self.serve(handler, key)
        body = handler.rfile.read(int(handler.headers["Content-Length"]))
        if len(body) > self.max_put:
            return self.refuse(handler, 400, key, "too large")
        with self.lock:
            if not upload:
                self.objects[key], self.etags[key] = body, hashlib.md5(body).hexdigest()
            elif upload in self.uploads and self.uploads[upload]["key"] == key:
                self.uploads[upload]["parts"][int(part)] = body
            else:
                return self.refuse(handler, 404, key, "no such upload")
        handler.send_response(200)
        handler.send_header("ETag", f'"{hashlib.md5(body).hexdigest()}"')
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def refuse(self, handler, code, key, reason):
        self.refused.append((key, reason))
        handler.send_response(code)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def serve(self, handler, key):
        data = self.objects.get(key)
        if data is None:
            return self.refuse(handler, 404, key, "missing")
        asked = handler.headers.get("Range")
        if asked:
            start, end = (int(value) for value in asked.removeprefix("bytes=").split("-"))
            data, whole = data[start:end + 1], len(data)
            handler.send_response(206)
            handler.send_header("Content-Range", f"bytes {start}-{start + len(data) - 1}/{whole}")
        else:
            handler.send_response(200)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    # What the controller does with the store's key

    def start_upload(self, key):
        upload = uuid.uuid4().hex
        with self.lock:
            self.uploads[upload] = {"key": key, "parts": {}}
        return upload

    def finish_upload(self, key, upload, etags):
        with self.lock:
            parts = self.uploads.pop(upload)["parts"]
            if sorted(parts) != [number for number, _ in etags] or any(
                    hashlib.md5(parts[number]).hexdigest() != etag for number, etag in etags):
                raise ValueError("the parts are not the ones sent")
            self.objects[key] = b"".join(parts[number] for number, _ in etags)
            digest = hashlib.md5(b"".join(bytes.fromhex(md5) for _, md5 in etags)).hexdigest()
            self.etags[key] = f"{digest}-{len(etags)}"
            self.joined += 1

    def size(self, key):
        data = self.objects.get(key)
        return None if data is None else len(data)

    def etag(self, key):
        return self.etags.get(key) if key in self.objects else None

    def read(self, key):
        return self.objects.get(key)

    def write(self, key, data):
        self.objects[key], self.etags[key] = data, hashlib.md5(data).hexdigest()

    def upload(self, key, path):
        self.write(key, Path(path).read_bytes())


class LocalProvider:
    """Machines that are directories here. An offer is {"price", "speed", "count"}: dollars an hour, a speed the
    machine's processes see as RENTAL_SPEED, and how many machines it has."""

    def __init__(self, directory, offers, uv):
        self.directory, self.offers, self.uv = Path(directory), offers, uv
        self.machines_, self.rejected, self.rented = {}, [], []
        self.lock = threading.Lock()

    def rent(self, label, _needs, hours):
        with self.lock:
            offers = sorted((offer for offer in self.offers if offer.get("count", 1) > 0), key=lambda o: o["price"])
            if not offers:
                return None, 0.0
            offer = offers[0]
            offer["count"] = offer.get("count", 1) - 1
            ident = f"m{len(self.rented) + 1}"
            home = self.directory / ident
            (home / ".rental/uv").mkdir(parents=True)
            # The machine's uv is this one's, so that a rehearsal downloads nothing.
            (home / ".rental/uv/uv").symlink_to(self.uv)
            self.machines_[ident] = {"home": home, "label": label, "price": offer["price"], "speed": offer["speed"],
                                     "since": time.time(), "deadline": time.time() + hours * 3600, "answers": True}
            self.rented.append(ident)
            return ident, 0.0

    def machines(self, label):
        with self.lock:
            for machine in self.machines_.values():
                if not machine.get("returned") and time.time() > machine["deadline"]:
                    self.end(machine, "deadline")
            return [{"id": ident, "cost": self.cost(machine)} for ident, machine in self.machines_.items()
                    if machine["label"] == label and not machine.get("returned")]

    @staticmethod
    def cost(machine):
        return machine["price"] * ((machine.get("returned_at") or time.time()) - machine["since"]) / 3600

    def run(self, ident, command, data, timeout):
        machine = self.machines_[ident]
        if not machine["answers"] or machine.get("returned"):
            raise Unreachable(ident)
        environment = {"PATH": os.environ["PATH"], "HOME": str(machine["home"]), "RENTAL_SPEED": str(machine["speed"])}
        ran = subprocess.run(["sh", "-c", command], input=data, capture_output=True, timeout=timeout,
                             cwd=machine["home"], env=environment)
        return ran.returncode, ran.stdout.decode(errors="replace")

    def extend(self, ident, hours):
        self.machines_[ident]["deadline"] = time.time() + hours * 3600

    def reject(self, ident, reason):
        self.rejected.append((ident, reason))

    def release(self, ident):
        machine = self.machines_[ident]
        self.end(machine, "released")
        return self.cost(machine)

    def end(self, machine, how):
        self.kill_processes(machine)
        if not machine.get("returned"):
            machine.update(returned=how, returned_at=time.time())

    # What only a rehearsal does

    def kill(self, ident):
        """The machine dies: its processes end and it answers no more."""
        self.kill_processes(self.machines_[ident])
        self.machines_[ident]["answers"] = False

    @staticmethod
    def kill_processes(machine):
        """End the machine's agent and everything it started, which share its process group."""
        try:
            pid = int((machine["home"] / ".rental/agent.pid").read_text())
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, ValueError):
            return
        if str(machine["home"]).encode() in command:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        for machine in self.machines_.values():
            self.end(machine, "released")
