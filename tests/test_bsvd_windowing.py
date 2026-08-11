"""Geometry for the 2070S windowed tile-sequential path (spec 5.5).

These are the numbers that decide whether windowed output is bit-identical to a
whole-clip run. They are plain arithmetic on purpose, so they can be checked
without a GPU.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from bsvd_windowed import plan_window, reflect_idx, tile_origins

SHIFT = 16          # future frames the model reads
PAST = 16           # SKIP_LENS [8, 8, 4] over two cascaded DenBlocks


def test_tiles_cover_every_pixel_of_1080p():
    tile, overlap = 576, 16
    ys = tile_origins(1080, tile, overlap)
    xs = tile_origins(1920, tile, overlap)
    out = tile - overlap
    assert max(ys) + out >= 1080
    assert max(xs) + out >= 1920


def test_tile_origins_never_run_past_the_frame():
    for size in (1080, 1920, 720, 577):
        for start in tile_origins(size, 576, 16):
            assert 0 <= start <= size


def test_reflect_has_no_edge_repeat():
    # ... f2 f1 | f0 f1 f2 ... f8 | f7 f6 ...
    assert [reflect_idx(i, 9) for i in (-3, -2, -1)] == [3, 2, 1]
    assert [reflect_idx(i, 9) for i in (9, 10, 11)] == [7, 6, 5]
    assert reflect_idx(0, 9) == 0 and reflect_idx(8, 9) == 8


def test_reflect_handles_a_single_frame_clip():
    assert reflect_idx(-5, 1) == 0 and reflect_idx(5, 1) == 0


def test_window_feeds_a_full_margin_on_both_sides():
    b, fs, fe = plan_window(3000, num_frames=10000, window=1500, margin=32)
    assert (b, fs, fe) == (4500, 2968, 4532)


def test_margin_exceeds_the_models_reach_on_both_sides():
    """M must cover past AND future context, or a window is not exact."""
    _, fs, fe = plan_window(3000, num_frames=10000, window=1500, margin=32)
    assert 3000 - fs >= PAST
    assert fe - 4500 >= SHIFT


def test_the_last_window_is_clipped_not_padded_out():
    b, fs, fe = plan_window(9000, num_frames=9700, window=1500, margin=32)
    assert b == 9700, "must not claim frames the clip does not have"
    assert fe == 9732, "the tail margin is reflected by the caller, not truncated"


def test_windows_tile_the_clip_with_no_gap_or_overlap():
    num_frames, window, margin = 9700, 1500, 32
    covered, a = [], 0
    while a < num_frames:
        b, _, _ = plan_window(a, num_frames, window, margin)
        covered.append((a, b))
        a = b
    assert covered[0][0] == 0 and covered[-1][1] == num_frames
    for (_, prev_end), (next_start, _) in zip(covered, covered[1:]):
        assert prev_end == next_start


def test_a_clip_shorter_than_one_window_is_a_single_window():
    b, fs, fe = plan_window(0, num_frames=800, window=1500, margin=32)
    assert (b, fs, fe) == (800, -32, 832)


def test_the_builder_refuses_a_margin_below_the_models_reach():
    """M < shift_num cannot be exact, so it must fail loudly, not silently blur.

    Checked before any import or clip access, so it holds on a host with no GPU
    stack and fails before an engine build rather than after one.
    """
    from bsvd_windowed import build_bsvd_windowed_tiled
    with pytest.raises(ValueError, match="below shift_num"):
        build_bsvd_windowed_tiled(object(), onnx_path="x.onnx", sigma=0.05,
                                  margin=8, shift_num=16)


def test_the_builder_refuses_an_odd_overlap_and_an_empty_window():
    from bsvd_windowed import build_bsvd_windowed_tiled
    with pytest.raises(ValueError, match="overlap must be even"):
        build_bsvd_windowed_tiled(object(), onnx_path="x.onnx", sigma=0.05, overlap=15)
    with pytest.raises(ValueError, match="window must be at least"):
        build_bsvd_windowed_tiled(object(), onnx_path="x.onnx", sigma=0.05, window=0)
