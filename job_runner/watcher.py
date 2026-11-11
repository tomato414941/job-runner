"""The watchers of Hetzner Cloud and RunPod, which send a provider's machines back at their deadlines.

Round after round a watcher deletes the machines of runs whose deadline has passed, whatever has become of their
runs, and marks each round it finishes with the key it watches with. Renting needs it: a provider rents nothing while
its watcher has not lately finished a round with the key that rents, since a watcher sees the machines of its own
key's account alone. job-runner's install.sh starts `job-runner PROVIDER watch` for each provider whose key .env
gives. Vast.ai's watcher is vast/machines.py's own, which marks its rounds with its key the same way.
"""

import hashlib
import sys
import time

from .provider import state

# A watcher looks this often, and renting takes one that has not finished a round for three rounds as stopped.
SECONDS = 60
STALE_SECONDS = 3 * SECONDS


def mark(key):
    """What a round is marked with for the provider's key: a digest, which tells keys apart and gives none away."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def watching(provider, key):
    """Whether this workstation's watcher watches the machines that the key rents: it finished a round with the key
    within STALE_SECONDS."""
    watched = state(provider) / "watched"
    try:
        return time.time() - watched.stat().st_mtime <= STALE_SECONDS and watched.read_text() == mark(key)
    except OSError:
        return False


def needed(provider):
    """What renting says while the provider's watcher does not watch the machines that the key rents."""
    return (f"renting needs this workstation's watcher, `job-runner {provider} watch`, to run with the key that "
            "rents; job-runner's install.sh starts it with the key in .env")


def watch(provider, key, sweep):
    """Run sweep every SECONDS for as long as this runs, marking each round that finishes with the key; a round that
    fails is tried again the next time."""
    while True:
        try:
            sweep()
            state(provider).mkdir(parents=True, exist_ok=True, mode=0o700)
            # The mark goes in place whole, so that renting never reads half of one.
            (state(provider) / "watched.new").write_text(mark(key))
            (state(provider) / "watched.new").replace(state(provider) / "watched")
        except Exception as error:  # noqa: BLE001 - the watcher outlives any one round
            # The error's kind and code alone: its message can name an address.
            print("the round failed:", type(error).__name__, getattr(error, "code", ""), file=sys.stderr, flush=True)
        time.sleep(SECONDS)
