"""RunPodProvider against a stand-in for RunPod's API kept in memory, and stand-ins for ssh and ssh-keygen on its PATH."""

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

from job_runner import runpod
from job_runner.provider import RentError, Unreachable
from job_runner.runpod import RunPodProvider


def handler(cloud):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, code, body=None):
            data = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def handle_one(self):
            parts = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            cloud["requests"].append((self.command, parts.path, body))
            ident = parts.path.rsplit("/", 1)[1] if parts.path.count("/") == 2 else None
            if self.command == "POST":
                if body.get("gpuTypeIds") == ["scarce"]:
                    return self.answer(400, b'{"error": "There are no longer any instances available"}')
                if body.get("gpuTypeIds") == ["wrong"]:
                    return self.answer(400, b'{"error": "imageName is required"}')
                pod = {"id": f"pod{len(cloud['pods']) + 1}", "name": body["name"], "desiredStatus": "RUNNING",
                       "costPerHr": 0.6, "lastStartedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                       "publicIp": "127.0.0.1", "portMappings": {"22": 10341}}
                cloud["pods"][pod["id"]] = pod
                return self.answer(201, pod)
            if self.command == "GET" and ident is None:
                name = parse_qs(parts.query)["name"][0]
                return self.answer(200, [pod for pod in cloud["pods"].values() if pod["name"] == name])
            if ident not in cloud["pods"]:
                return self.answer(404, {"error": "pod not found"})
            if self.command == "GET":
                return self.answer(200, cloud["pods"][ident])
            if self.command == "DELETE":
                del cloud["pods"][ident]
                return self.answer(204)

        do_GET = do_POST = do_DELETE = handle_one

    return Handler


class RunPodProviderTests(unittest.TestCase):
    def setUp(self):
        self.cloud = {"pods": {}, "requests": []}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.cloud))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.calls = self.root / "calls.jsonl"
        self.script("ssh-keygen", "print('ssh-ed25519 AAAA public')")
        self.script("ssh", "sys.exit(255 if sys.argv[-1] == 'unreachable' else 3)")
        self.environment = {"RUNPOD_API_KEY": "key", "RUNPOD_URL": f"http://127.0.0.1:{self.server.server_port}",
                            "RUNPOD_SSH_KEY_FILE": "/keys/runpod", "PATH": f"{self.root}:{os.environ['PATH']}"}
        self.home = patch.dict(os.environ, {"HOME": str(self.root), "PATH": self.environment["PATH"]})
        self.home.start()
        self.provider = RunPodProvider(self.environment)
        self.needs = {"gpus": ["NVIDIA GeForce RTX 5090"], "image": "runpod/base", "disk": 40}

    def tearDown(self):
        self.home.stop()
        self.server.shutdown()
        self.server.server_close()
        self.directory.cleanup()

    def script(self, name, body):
        path = self.root / name
        path.write_text(f"#!/usr/bin/env python3\nimport json, sys\n"
                        f"open({str(self.calls)!r}, 'a').write(json.dumps([{name!r}, *sys.argv[1:]]) + '\\n')\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def rent(self, needs=None, hours=2.0):
        with patch.object(RunPodProvider, "answers", return_value=True):
            return self.provider.rent("run", needs or self.needs, hours)

    def test_rents_a_pod_with_its_public_key_and_keeps_its_deadline_or_none_when_there_is_no_room(self):
        self.assertEqual(self.rent(), ("pod1", 0.0))
        method, path, body = self.cloud["requests"][0]
        self.assertEqual((method, path, body["name"], body["env"], body["ports"], body["gpuTypeIds"]),
                         ("POST", "/pods", "run", {"PUBLIC_KEY": "ssh-ed25519 AAAA public"}, ["22/tcp"],
                          ["NVIDIA GeForce RTX 5090"]))
        kept = json.loads((self.root / ".local/state/job-runner/runpod/pods.json").read_text())
        self.assertAlmostEqual(kept["pod1"]["deadline"], time.time() + 7200, delta=60)
        self.assertEqual(self.rent({**self.needs, "gpus": ["scarce"]}), (None, 0.0))
        with self.assertRaisesRegex(RentError, "imageName is required"):
            self.rent({**self.needs, "gpus": ["wrong"]})

    def test_asks_for_cpus_a_gpu_and_cuda_versions_only_when_given(self):
        self.rent({**self.needs, "min_vcpus_per_gpu": 8, "allowed_cuda_versions": ["13.0"]})
        self.rent()
        given, plain = [body for method, _, body in self.cloud["requests"] if method == "POST"]
        self.assertEqual((given["minVCPUPerGPU"], given["allowedCudaVersions"]), (8, ["13.0"]))
        self.assertNotIn("minVCPUPerGPU", plain)
        self.assertNotIn("allowedCudaVersions", plain)

    def test_asks_for_cpus_for_a_cpu_pod(self):
        self.rent({"compute": "CPU", "vcpus": 16, "image": "runpod/base", "disk": 40})
        body = self.cloud["requests"][0][2]
        self.assertEqual((body["computeType"], body["vcpuCount"]), ("CPU", 16))
        self.assertNotIn("gpuTypeIds", body)

    def test_deletes_a_pod_that_never_answers_ssh(self):
        with patch.object(RunPodProvider, "answers", return_value=False), patch.object(runpod, "START_SECONDS", 0.05), \
                patch.object(runpod.time, "sleep"):
            with self.assertRaisesRegex(RentError, "did not answer ssh"):
                self.provider.rent("run", self.needs, 2.0)
        self.assertEqual(self.cloud["pods"], {})

    def test_lists_its_runs_pods_with_their_cost_and_moves_and_ends_them(self):
        ident, _ = self.rent()
        self.cloud["pods"][ident]["lastStartedAt"] = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        [listed] = self.provider.machines("run")
        self.assertEqual((listed["id"], round(listed["cost"], 2)), (ident, 0.3))
        self.assertEqual(self.provider.machines("other"), [])
        deadline = self.provider.extend(ident, 5.5)
        self.assertAlmostEqual(deadline, time.time() + 5.5 * 3600, delta=60)
        self.assertEqual(self.provider.machines("run")[0]["deadline"], deadline)
        self.assertEqual(round(self.provider.release(ident), 2), 0.3)
        self.assertEqual(self.cloud["pods"], {})
        self.assertIsNone(self.provider.extend(ident, 1))
        self.assertIsNone(RunPodProvider({**self.environment, "RUNPOD_URL": "http://127.0.0.1:9"}).machines("run"))

    def test_sweep_deletes_only_the_pods_past_their_deadline(self):
        self.rent()
        self.rent()
        pods = self.provider.kept()
        pods["pod1"]["deadline"] = int(time.time()) - 60
        self.provider.keep(pods)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.provider.sweep()
        self.assertEqual(list(self.cloud["pods"]), ["pod2"])
        self.assertEqual(json.loads(printed.getvalue())["deleted"], "pod1")
        self.assertEqual(list(self.provider.kept()), ["pod2"])

    def test_runs_a_command_as_root_at_the_pods_ssh_port_and_takes_ssh_failure_for_no_answer(self):
        ident, _ = self.rent()
        self.assertEqual(self.provider.run(ident, "fails", b"payload", 30), (3, ""))
        with self.assertRaises(Unreachable):
            self.provider.run(ident, "unreachable", None, 30)
        with self.assertRaises(Unreachable):
            self.provider.run("gone", "anything", None, 30)
        ssh = [json.loads(line) for line in self.calls.read_text().splitlines() if line.startswith('["ssh"')][0]
        self.assertEqual(ssh[1:5], ["-i", "/keys/runpod", "-p", "10341"])
        self.assertEqual(ssh[-2:], ["root@127.0.0.1", "fails"])


if __name__ == "__main__":
    unittest.main()
