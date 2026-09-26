"""The rate a lane is achieving right now, from the counter in its log.

A figure derived from completed clips cannot be the live one: state.jsonl gains
a record only when a clip finishes, and the longest clip here is tens of thousands of frames.
So the live rate is read from a frame counter instead.

Three producers write that counter, in two shapes:

  local encode    vspipe -p          -> <stem>_vspipe.log      "Frame: N"
  local encode    netstream recv     -> <stem>_netstream.log   "Frame: N"
  remote encode   SvtAv1EncApp       -> <stem>_encode.log      "Encoding: N Frames"

A lane is counted at the receiving end of the y4m on purpose. Its denoiser runs
on another host and writes its vspipe log on that host's disk, while the ssh
channel captured locally carries only the remote dispatch's own output.

With a remote encoder the receiving end moves off this host too: netstream recv
now runs beside SvtAv1EncApp on the encode host and writes its log there, so
neither of the first two files exists here. What does reach this host is the
encode ssh's stderr, and the encoder's own counter is in it. Counting that
keeps the same meaning -- frames the receiver has taken delivery of -- which is
why it is read rather than left blank. Without it every remote-encode lane
renders as unknown, which is what the pool shipped as.
"""
import os
import re
import threading
from collections import deque

# Anchored to a line start, matching archive-batch.py:78, which filters these
# same two logs. Both producers emit the counter at the start of a line, so the
# anchor costs nothing and stops a diagnostic that happens to mention a frame
# mid-sentence from being read as a count. Two readers of one file format
# disagreeing about what that format is invites exactly one silent bug.
_PROGRESS = re.compile(r"^Frame:\s*(\d+)", re.MULTILINE)

# SvtAv1EncApp's counter, which colours its output unconditionally: the line is
# "Encoding: <ESC>[33m   1 Frames" early on and "<ESC>[33m3258/3289 Frames" once
# it knows the total. Both give the done count first, so one group covers them.
# The escapes are skipped rather than stripped from the whole tail, because a
# strip would have to run over every byte read on every poll.
_ENCODE_PROGRESS = re.compile(r"^Encoding:(?:\s|\x1b\[[0-9;]*m)*(\d+)",
                              re.MULTILINE)

# Every log that can hold a counter, each with the shape it writes, in the
# order that decides between producers writing at the same moment: the one
# nearest the front of this lane's pipeline wins.
#
# A clip does write more than one of these at once. A lane that denoises here
# and encodes on another host has vspipe's log and the encode ssh's stderr
# side by side, both live, their counters a pipe's depth apart. Picking the
# newer alternated between them once per poll -- see the test named for it.
# The denoiser's count is the lane's own production, so it comes first, and
# the encoder's is what remains when the denoise happens elsewhere.
_SUFFIXES = (("_vspipe.log", _PROGRESS),
             ("_netstream.log", _PROGRESS),
             ("_encode.log", _ENCODE_PROGRESS))

# How far behind the newest log a log may be and still count as a producer of
# this clip. Order alone cannot decide, because a retry reuses the per-clip
# directory and an abandoned attempt's log keeps its last count for ever;
# first place would then pin the lane at zero. Generous on purpose: the gap
# between two live producers is milliseconds, and the gap to a leftover is the
# length of the attempt that wrote it.
_STALE_S = 300.0

# Enough to hold the last progress lines without reading a log that has grown
# to a quarter of a megabyte over three hours.
#
# Known and accepted: if more than 8 KB of non-progress output follows the last
# counter, this returns None and the lane renders as unknown. Reaching that
# needs the producer to have stopped emitting progress and then printed 8 KB,
# which means the clip has already crashed or finished -- on a live lane the
# tail always holds the last few hundred seconds of counters. Unknown is the
# right answer for a dead clip anyway; a frozen count that never moves is worse.
_TAIL_BYTES = 8192


def frames_from_log(temp_dir, stem):
    """Frames produced so far, or None when nothing has been counted yet.

    None rather than 0: for the first seconds of every clip the log exists and
    holds no counter, and a lane that has produced nothing yet must not render
    the same as one that has stalled.
    """
    found_at = []
    for suffix, pattern in _SUFFIXES:
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
        found = pattern.findall(text)
        if not found:
            continue
        found_at.append((int(found[-1]), mtime))
    if not found_at:
        return None
    newest = max(mtime for _, mtime in found_at)
    # found_at is in _SUFFIXES order, and the newest log is always fresh
    # against itself, so this can never be empty.
    fresh = [count for count, mtime in found_at if newest - mtime <= _STALE_S]
    return fresh[0]


def _burst_edges(pts):
    """The samples at which a burst of frames starts, oldest first.

    A windowed lane's counter is flat while the GPU works a sweep and then
    climbs while the encoder drains the window it just got, so a burst starts
    at the first sample whose count moves after one that did not. Returning the
    START of each burst rather than its end keeps the interval between two
    edges equal to a whole number of sweeps however long a drain takes.

    Fewer than two edges means the counter never paused -- a lane draining
    continuously, or a series shorter than one sweep. The caller falls back to
    the raw endpoints, which is what it always did.
    """
    edges, rising = [], True
    for i in range(1, len(pts)):
        moved = pts[i][1] > pts[i - 1][1]
        if moved and not rising:
            edges.append(pts[i - 1])
        rising = moved
    return edges


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
        # One tracker is shared by every request thread of a ThreadingHTTPServer,
        # and sample() is a check-then-index across bytecode boundaries: a
        # clear() from a concurrent poll landing between `len(series) < 2` and
        # `series[0]` empties the deque and raises IndexError, which surfaces as
        # a 500 on /metrics and breaks the scrape.
        #
        # The GIL makes each deque operation atomic; it does not make the
        # sequence atomic. A stress run finds nothing because the window is a
        # few bytecodes wide and needs a clip restart to coincide with it -- that
        # is evidence of low probability, not of safety. Demonstrated with a
        # deterministic two-thread interleave. state.py uses the same pattern
        # for the same reason.
        self._lock = threading.Lock()

    def sample(self, lane, frames, now, smooth_s=None, min_span_s=None):
        """Record a count and return the current rate, or None if unknown.

        `smooth_s` overrides the default for this lane only. One global value
        cannot serve both lane kinds: a full-frame lane streams frames evenly
        and wants a short window so its figure is current, while a windowed lane
        steps by a whole window and needs several sweeps. The caller knows which
        it is, because the roster says so.

        `min_span_s` refuses to answer until the series covers that long. The
        series is cleared at every change of clip, so early in each clip a
        windowed lane's slope spans less than one sweep -- and a sweep is the
        smallest interval over which it produces anything at all. Measured
        2026-08-26: the 2070s lane at window 750 read 23.6 fps against a true
        4.6, because its counter had gone from 1 to 669 in the 28 s the encoder
        spent draining the first delivered window. Smoothing cannot fix that;
        there is nothing yet to smooth. None is the honest answer, and the
        caller shows the completed-clip average instead.
        """
        smooth_s = smooth_s or self.smooth_s
        with self._lock:
            series = self._series.setdefault(lane, deque())
            if series and frames < series[-1][1]:
                # The count went backwards: a new clip, or the same clip
                # restarted after a yield. The old series describes other work.
                series.clear()
            series.append((now, frames))
            # Keep one sample older than the smoothing window, and drop the
            # rest, so the slope always spans at least smooth_s once it can.
            # This is what bounds the memory: everything younger than smooth_s
            # is kept and nothing else, so a lane holds smooth_s/poll_interval
            # entries however long the run lasts.
            while len(series) > 2 and now - series[1][0] >= smooth_s:
                series.popleft()
            if len(series) < 2:
                return None
            pts = list(series)
        # Snap a windowed lane's slope to whole bursts. Between the raw
        # endpoints the span holds N or N+1 bursts depending on where the
        # series happens to start, and at SWEEPS_SMOOTHED = 2.5 that is 2 or 3
        # bursts of `window` frames over the same seconds -- a +-20% swing by
        # phase alone, with no change in the lane. Measuring edge to edge holds
        # a whole number of bursts by construction, so the phase cancels
        # exactly instead of being averaged down.
        #
        # Only for a lane the caller called windowed. A full-frame lane streams
        # evenly, has no edges to find, and wants every sample it has.
        edges = _burst_edges(pts) if min_span_s else []
        if len(edges) >= 2:
            t0, f0 = edges[0]
            t1, f1 = edges[-1]
        else:
            t0, f0 = pts[0]
            t1, f1 = pts[-1]
        span = t1 - t0
        if span <= 0:
            return None
        # Against the RAW span, not the snapped one. min_span_s asks whether
        # this lane has produced anything measurable yet, and snapping shortens
        # the span by up to one burst -- judging the snapped span would reject
        # series that have genuinely covered a sweep.
        if min_span_s and (pts[-1][0] - pts[0][0]) < min_span_s:
            return None
        return (f1 - f0) / span

    def forget(self, lane):
        """Drop a lane's history when it stops working, so the next clip does
        not inherit a slope from the last one."""
        with self._lock:
            self._series.pop(lane, None)
