"""The rate a lane is achieving right now, from the counter in its log.

A figure derived from completed clips cannot be the live one: state.jsonl gains
a record only when a clip finishes, and the longest clip here is tens of thousands of frames.
So the live rate is read from a frame counter instead.

Two producers write that counter and both emit the same shape:

  local lane    vspipe -p            -> <stem>_vspipe.log
  remote lane   netstream recv       -> <stem>_netstream.log

A remote lane is counted at the receiving socket on purpose. Its denoiser runs
on another host and writes its vspipe log on that host's disk, while the ssh
channel captured locally carries only the remote dispatch's own output.
"""
import os
import re
from collections import deque

_PROGRESS = re.compile(r"Frame:\s*(\d+)")

# Both logs for one clip, most specific first. Order only breaks ties that
# mtime cannot, which in practice does not happen: a clip is local or remote.
_SUFFIXES = ("_vspipe.log", "_netstream.log")

# Enough to hold the last progress lines without reading a log that has grown
# to a quarter of a megabyte over three hours.
_TAIL_BYTES = 8192


def frames_from_log(temp_dir, stem):
    """Frames produced so far, or None when nothing has been counted yet.

    None rather than 0: for the first seconds of every clip the log exists and
    holds no counter, and a lane that has produced nothing yet must not render
    the same as one that has stalled.
    """
    best, best_mtime = None, None
    for suffix in _SUFFIXES:
        path = os.path.join(temp_dir, f"{stem}{suffix}")
        try:
            mtime = os.path.getmtime(path)
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - _TAIL_BYTES))
                # vspipe separates progress with \r, so split on both. Decoding
                # before the replace keeps a tail that starts mid-character
                # harmless: \r is a single byte that cannot occur inside a
                # multi-byte sequence, and "replace" turns the orphan bytes into
                # one U+FFFD in the first line, which no longer holds a count.
                text = fh.read().decode("utf-8", "replace").replace("\r", "\n")
        except OSError:
            continue
        found = _PROGRESS.findall(text)
        if not found:
            continue
        if best_mtime is None or mtime > best_mtime:
            best, best_mtime = int(found[-1]), mtime
    return best


class RateTracker:
    """Frames per second per lane, smoothed over at least `smooth_s`.

    A windowed lane publishes a whole window when its sweep completes, so a
    short sample reads zero for most of a sweep and then spikes. Smoothing over
    several sweeps is what makes the number readable: with a step input any
    finite window sees N or N+1 bursts depending on phase, so one sweep's worth
    of smoothing still swings by a factor of two.

    This is the same burst artefact that had window 750 recorded as 30% faster
    than window 500 until 2026-08-15; see docs/window-sizing.md.
    """

    def __init__(self, smooth_s=30.0):
        self.smooth_s = smooth_s
        self._series = {}

    def sample(self, lane, frames, now):
        """Record a count and return the current rate, or None if unknown."""
        series = self._series.setdefault(lane, deque())
        if series and frames < series[-1][1]:
            # The count went backwards: a new clip, or the same clip restarted
            # after a yield. The old series describes different work.
            series.clear()
        series.append((now, frames))
        # Keep one sample older than the smoothing window, and drop the rest,
        # so the slope always spans at least smooth_s once it can. This is what
        # bounds the memory: everything younger than smooth_s is kept and
        # nothing else, so a lane holds smooth_s/poll_interval entries however
        # long the run lasts.
        while len(series) > 2 and now - series[1][0] >= self.smooth_s:
            series.popleft()
        if len(series) < 2:
            return None
        t0, f0 = series[0]
        t1, f1 = series[-1]
        span = t1 - t0
        if span <= 0:
            return None
        return (f1 - f0) / span

    def forget(self, lane):
        """Drop a lane's history when it stops working, so the next clip does
        not inherit a slope from the last one."""
        self._series.pop(lane, None)
