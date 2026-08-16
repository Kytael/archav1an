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


def _marks(window, total, sweep_s):
    """Frame/time marks for a lane that emits `window` frames every `sweep_s`.

    Every frame of a window carries the timestamp of the sweep that produced
    it, which is what the real sink records. A final short window costs time in
    proportion to the frames it carries.
    """
    out = []
    t = 0.0
    frame = 0
    while frame < total:
        end = min(frame + window, total)
        t += sweep_s * (end - frame) / window
        for f in range(frame + 1, end + 1):
            out.append((f, t))
        frame = end
    return out


def test_a_burst_inside_the_first_window_is_not_counted_as_free_frames():
    """Two lanes at the same real rate must report the same sustained rate."""
    # 1 fps in both cases: 500 frames per 500 s, 750 frames per 750 s.
    slow = rates(_marks(500, 1700, 500.0), t0=0.0, window=500)
    fast = rates(_marks(750, 1700, 750.0), t0=0.0, window=750)
    assert slow["sustained_fps"] == fast["sustained_fps"] == 1.0


def test_the_skip_fraction_still_applies_when_it_exceeds_the_window():
    """On a long clip the 20% fraction is the binding term, not the window."""
    out = rates(_marks(500, 6726, 500.0), t0=0.0, window=500)
    # 20% of 6726 is 1345. That falls inside the burst covering 1001-1500, so
    # the count starts at the end of that burst and not at frame 1345.
    assert out["sustained_over_frames"] == 6726 - 1500
    assert out["sustained_fps"] == 1.0


def test_overhead_is_the_wait_for_the_first_frame():
    out = rates(_marks(500, 1700, 500.0), t0=-10.0, window=500)
    assert out["overhead_s"] == 510.0
    assert out["frames"] == 1700
