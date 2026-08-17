"""The sustained rate must not reward a bigger window for bursting harder.

A windowed denoiser publishes a whole window at once when its sweep finishes,
so every frame in that window shares one arrival time. If the point where the
measurement starts falls inside that first window, those frames enter the
average at no time cost and the figure climbs with window size instead of with
the lane's real rate. That is how window 750 came to be recorded as 30% faster
than window 500 on a lane that was in fact 2% slower.
"""

import importlib.util
from pathlib import Path

RATE_PY = Path(__file__).resolve().parent.parent / "tools" / "denoise-rate.py"
_spec = importlib.util.spec_from_file_location("denoise_rate", RATE_PY)
denoise_rate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(denoise_rate)
rates = denoise_rate.rates


def _marks(window, total, sweep_s, flush_per_frame=0.002):
    """Frame/time marks for a lane that emits `window` frames every `sweep_s`.

    The sink stamps each frame as its bytes arrive, so a window's frames are
    spread across the short time it takes to flush them down the socket, NOT
    given one shared timestamp. That is why the anchor cannot be found by
    comparing timestamps. A final short window costs time in proportion to the
    frames it carries, sweep and flush alike.
    """
    out = []
    t = 0.0
    frame = 0
    while frame < total:
        end = min(frame + window, total)
        n = end - frame
        t += sweep_s * n / window
        for i, f in enumerate(range(frame + 1, end + 1), start=1):
            out.append((f, t + flush_per_frame * i))
        t += flush_per_frame * n
        frame = end
    return out


def test_a_burst_inside_a_window_is_not_counted_as_free_frames():
    """Two lanes at the same real rate must report the same sustained rate.

    Both lanes below take one second per output frame: 500 frames per 500 s and
    750 per 750 s, each plus the same per-frame flush. The old code read them as
    1.20 and 1.27 fps, rising with the window instead of matching the lane.
    """
    slow = rates(_marks(500, 6726, 500.0), t0=0.0, window=500)
    fast = rates(_marks(750, 6726, 750.0), t0=0.0, window=750)
    assert slow["sustained_fps"] == fast["sustained_fps"]
    assert slow["sustained_fps"] == round(1 / 1.002, 3)


def test_the_anchor_lands_on_a_sweep_boundary():
    """The skip point is rounded up to a multiple of the window."""
    out = rates(_marks(500, 6726, 500.0), t0=0.0, window=500)
    # 20% of 6726 is 1345, which falls inside the sweep covering 1001-1500, so
    # the count starts at frame 1500 and not at frame 1345.
    assert out["sustained_over_frames"] == 6726 - 1500


def test_an_unwindowed_lane_uses_the_plain_fraction():
    """A full-frame lane has no bursts, so nothing is rounded."""
    out = rates([(f, f * 0.5) for f in range(1, 1001)], t0=0.0, window=0)
    assert out["sustained_over_frames"] == 1000 - 200
    assert out["sustained_fps"] == 2.0


def test_overhead_is_the_wait_for_the_first_frame():
    out = rates(_marks(500, 1700, 500.0), t0=-10.0, window=500)
    assert out["overhead_s"] == 510.0
    assert out["frames"] == 1700
