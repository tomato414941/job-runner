import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from job_runner import watcher
from job_runner.watcher import watch, watching


class Stop(Exception):
    """Ends a test's watcher, which otherwise runs for good."""


class WatcherTests(unittest.TestCase):
    def test_marks_each_round_that_finishes_and_outlives_one_that_fails(self):
        rounds, marked = [], []

        def sweep():
            rounds.append(len(rounds))
            if len(rounds) == 1:
                raise OSError("no answer from 203.0.113.9")

        def sleep(seconds):
            marked.append(watching("hetzner"))
            if len(rounds) == 2:
                raise Stop

        said = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HOME": directory}), \
                patch.object(watcher.time, "sleep", sleep), contextlib.redirect_stderr(said):
            with self.assertRaises(Stop):
                watch("hetzner", sweep)
            self.assertFalse(watching("runpod"))
        self.assertEqual((rounds, marked), ([0, 1], [False, True]))
        # A failed round tells the error's kind, never its message, which can name an address.
        self.assertIn("OSError", said.getvalue())
        self.assertNotIn("203.0.113.9", said.getvalue())


if __name__ == "__main__":
    unittest.main()
