"""The watchers of Hetzner Cloud and RunPod, which send a provider's machines back at their deadlines.

Round after round a watcher deletes the machines of runs whose deadline has passed, whatever has become of their
runs, and marks each round it finishes. Renting needs it: a provider rents nothing while its watcher has not finished
a round lately. job-runner's install.sh starts `job-runner PROVIDER watch` for each provider whose key .env gives.
Vast.ai's watcher is vast/machines.py's own.
"""

import sys
import time

from .provider import state

# A watcher looks this often, and renting takes one that has not finished a round for three rounds as stopped.
SECONDS = 60
STALE_SECONDS = 3 * SECONDS


def watching(provider):
    """Whether this workstation's watcher of the provider's machines runs: it finished a round within STALE_SECONDS."""
    try:
        return time.time() - (state(provider) / "watched").stat().st_mtime <= STALE_SECONDS
    except OSError:
        return False


def needed(provider):
    """What renting says while the provider's watcher does not run."""
    return (f"renting needs this workstation's watcher, `job-runner {provider} watch`, "
            "which job-runner's install.sh starts")


def watch(provider, sweep):
    """Run sweep every SECONDS for as long as this runs, marking each round that finishes; a round that fails is
    tried again the next time."""
    while True:
        try:
            sweep()
            state(provider).mkdir(parents=True, exist_ok=True, mode=0o700)
            (state(provider) / "watched").touch()
        except Exception as error:  # noqa: BLE001 - the watcher outlives any one round
            # The error's kind and code alone: its message can name an address.
            print("the round failed:", type(error).__name__, getattr(error, "code", ""), file=sys.stderr, flush=True)
        time.sleep(SECONDS)
