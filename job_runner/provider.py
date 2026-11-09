"""What the controller asks of a provider and of a store: vast.py and r2.py give it, and local.py stands in for both.

A provider:
  rent(label, needs, hours, avoid)  a machine that passed the provider's checks, with the label and its deadline
                                    hours from now, and none that the released machines in avoid ran on: (its id,
                                    dollars spent on machines that failed the checks), the id None when no offer
                                    fits; RentError when renting failed otherwise
  machines(label)                   [{"id", "cost", "deadline"}] for the machines with the label: dollars so far,
                                    and seconds since the epoch or None; None when the provider cannot tell
  run(id, command, data, timeout)   (exit code, output) of a shell command on the machine, with data on its standard
                                    input; Unreachable when the machine does not answer
  extend(id, hours)                 move its deadline to hours from now: the new deadline, or None when it could not
  reject(id, reason)                rent the machine never again, released or not
  release(id)                       send the machine back: what it cost, or None when the provider does not say

A store:
  max_put                           the most bytes one PUT takes
  expires                           when its key expires, in seconds since the epoch, or None
  presign(method, key, seconds, query=None)
                                    a URL for one request
  size(key), etag(key), read(key)   None when it holds no object at key
  write(key, data), upload(key, path)
  start_upload(key), finish_upload(key, upload, etags)
                                    an upload in parts, which machines send through presigned part URLs
"""

from pathlib import Path


def state(provider):
    """Where this workstation keeps what it knows of the provider's machines."""
    return Path.home() / ".local/state/job-runner" / provider


class Unreachable(Exception):
    """The machine did not answer; a command that ran and failed is not this."""


class RentError(Exception):
    """Renting failed otherwise than for want of an offer."""
