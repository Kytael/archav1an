"""Geometry for the 2070S windowed tile-sequential path (spec 5.5).

These are the numbers that decide whether windowed output is bit-identical to a
whole-clip run. They are plain arithmetic on purpose, so they can be checked
without a GPU.
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from bsvd_windowed import (DEFAULT_OVERLAP, TILE_BUDGET_MPX, check_tile_fits,
                           plan_tiling,
                           plan_window, reflect_idx, tile_origins,
                           window_read_plan)


def sweep_mpx(H, W, tile_h, tile_w, overlap=16):
    """Model pixels per output frame for this tiling -- what costs time."""
    n = len(tile_origins(H, tile_h, overlap)) * len(tile_origins(W, tile_w, overlap))
    return n * tile_h * tile_w / 1e6

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


def test_auto_tiling_covers_1080p_in_two_tiles_of_one_row():
    """1096x976 yields exactly 1080x960, so 1x2 covers the frame."""
    th, tw = plan_tiling(1080, 1920, overlap=16)
    assert (th, tw) == (1096, 976)
    ys = tile_origins(1080, th, 16)
    xs = tile_origins(1920, tw, 16)
    assert (len(ys), len(xs)) == (1, 2), "one row, two columns"


def test_the_default_overlap_is_the_one_that_closes_the_seam():
    """16 leaves a 5.5-code seam at the tile join; 32 cuts it 79x.

    Measured at 1080p against an overlap-128 render over 120 frames: the share
    of pixels differing by a whole 8-bit code goes 0.0079% -> 0.0001%, and it
    rendered in the same time, so the 3.1% extra tile area costs nothing
    measurable. A revert to 16 would reintroduce a visible-in-numbers seam.
    """
    assert DEFAULT_OVERLAP == 32
    th, tw = plan_tiling(1080, 1920)
    assert (th, tw) == (1112, 992), "1x2 grid yielding exactly 1080x960"
    assert (len(tile_origins(1080, th, DEFAULT_OVERLAP)),
            len(tile_origins(1920, tw, DEFAULT_OVERLAP))) == (1, 2)
    assert th * tw / 1e6 <= TILE_BUDGET_MPX, "must still fit the 8 GB card"


def test_the_default_overlap_still_beats_the_old_square_tile():
    """The seam fix must not cost back the tiling win."""
    th, tw = plan_tiling(1080, 1920)
    auto = sweep_mpx(1080, 1920, th, tw, DEFAULT_OVERLAP)
    square = sweep_mpx(1080, 1920, 576, 576, 16)
    assert auto < square / 1.15, f"auto {auto:.3f} vs old 576 {square:.3f}"


def test_auto_tiling_beats_the_square_tile_it_replaces():
    """The whole point: less redundant area than square 576."""
    th, tw = plan_tiling(1080, 1920, overlap=16)
    auto = sweep_mpx(1080, 1920, th, tw)
    square = sweep_mpx(1080, 1920, 576, 576)
    frame = 1080 * 1920 / 1e6
    assert auto / frame < 1.05, f"auto wastes {auto / frame:.2f}x"
    assert square / frame > 1.25, "square 576 was the 1.28x baseline"
    assert auto < square / 1.2, "must be at least a 1.2x cut in pixel work"


def test_auto_tiling_never_exceeds_the_measured_vram_budget():
    for H, W in ((1080, 1920), (720, 1280), (1080, 1440), (2160, 3840)):
        th, tw = plan_tiling(H, W, overlap=16)
        assert th * tw / 1e6 <= TILE_BUDGET_MPX, f"{H}x{W} -> {th}x{tw}"
        check_tile_fits(H, W, th, tw, 16)


def test_auto_tiling_covers_every_pixel():
    for H, W in ((1080, 1920), (720, 1280), (2160, 3840)):
        th, tw = plan_tiling(H, W, overlap=16)
        ys, xs = tile_origins(H, th, 16), tile_origins(W, tw, 16)
        assert max(ys) + th - 16 >= H, f"{H}x{W} leaves rows uncovered"
        assert max(xs) + tw - 16 >= W, f"{H}x{W} leaves columns uncovered"


def test_the_largest_tile_1080p_allows_is_the_one_that_fits_in_one_row():
    """1096 is both the ceiling and the ideal: 1096 - 16 = 1080 exactly."""
    check_tile_fits(1080, 1920, 1096, 1096, 16)
    with pytest.raises(ValueError, match="height 1216 exceeds"):
        check_tile_fits(1080, 1920, 1216, 1216, 16)
    # Wide is allowed where tall is not: the padded frame is 1096 x 1936.
    check_tile_fits(1080, 1920, 1096, 1936, 16)
    with pytest.raises(ValueError, match="width 1940 exceeds"):
        check_tile_fits(1080, 1920, 1096, 1940, 16)


def test_a_tile_must_be_a_multiple_of_four_on_both_axes():
    check_tile_fits(1080, 1920, 576, 576, 16)
    with pytest.raises(ValueError, match="height must be a multiple of 4"):
        check_tile_fits(1080, 1920, 1094, 976, 16)
    with pytest.raises(ValueError, match="width must be a multiple of 4"):
        check_tile_fits(1080, 1920, 1096, 974, 16)


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


def test_the_builder_refuses_to_read_its_source_in_process():
    """No source script means the old in-graph read, which deadlocks."""
    from bsvd_windowed import build_bsvd_windowed_tiled
    with pytest.raises(ValueError, match="source_script is required"):
        build_bsvd_windowed_tiled(object(), onnx_path="x.onnx", sigma=0.05)


# --- the read plan: one forward pass over the source, reflected at the ends ---

def test_read_plan_is_one_contiguous_forward_range():
    real_start, real_end, slots = window_read_plan(2968, 4532, 10000)
    assert (real_start, real_end) == (2968, 4532)
    assert len(slots) == 4532 - 2968
    assert all(len(s) == 1 for s in slots), "no reflection away from the clip ends"
    assert [s[0] for s in slots] == list(range(len(slots)))


def test_read_plan_never_asks_for_a_frame_outside_the_clip():
    for a, n in ((0, 800), (0, 10000), (9000, 9700)):
        b, fs, fe = plan_window(a, n, window=1500, margin=32)
        real_start, real_end, _ = window_read_plan(fs, fe, n)
        assert 0 <= real_start < real_end <= n


def test_read_plan_sends_a_reflected_frame_to_both_of_its_slots():
    # Window 0 of a 10000-frame clip feeds [-32, 1532): frame 32 is read once
    # and used both as the reflection of -32 and as itself.
    _, fs, fe = plan_window(0, 10000, window=1500, margin=32)
    real_start, _, slots = window_read_plan(fs, fe, 10000)
    assert real_start == 0
    assert slots[32] == [0, 64], "slot 0 is the reflection of frame -32"
    assert slots[0] == [32], "frame 0 appears once; reflect has no edge repeat"


def test_read_plan_covers_every_window_slot_exactly_once():
    for a, n in ((0, 800), (0, 10000), (9000, 9700), (1500, 10000)):
        b, fs, fe = plan_window(a, n, window=1500, margin=32)
        _, _, slots = window_read_plan(fs, fe, n)
        filled = sorted(k for s in slots for k in s)
        assert filled == list(range(fe - fs))


# --- the reader itself, against the frames VapourSynth hands back -----------

VSPIPE = os.environ.get("VSPIPE") or "/opt/archav1an/bin/vspipe"
if not os.path.exists(VSPIPE):
    VSPIPE = shutil.which("vspipe") or ""

SYNTH = '''\
import vapoursynth as vs
core = vs.core
clip = core.std.Splice([
    core.std.BlankClip(width=8, height=4, format=vs.{fmt}, length=1,
                       color=[i / 100, i / 100 + 0.01, i / 100 + 0.02])
    for i in range({n})])
clip.set_output(0)
'''


def synth(n, fmt="RGBS"):
    return SYNTH.format(n=n, fmt=fmt)


@pytest.mark.skipif(not VSPIPE, reason="vspipe not installed")
@pytest.mark.parametrize("fmt", ["RGBS", "RGBH"])
def test_the_subprocess_reader_matches_get_frame(tmp_path, fmt):
    """The whole point of the reader: identical pixels, decoded out of process.

    This is what catches the plane order. vspipe writes RGB planes as GBR
    while frame[i] is RGB, so a reader that copies plane p to plane p swaps
    two of the three channels and denoises a colour-rotated clip.
    """
    vs = pytest.importorskip("vapoursynth")
    np = pytest.importorskip("numpy")
    from bsvd_windowed import VspipeWindowSource

    n = 40
    script = tmp_path / "synth.vpy"
    script.write_text(synth(n, fmt))
    ns = {}
    exec(compile(script.read_text(), str(script), "exec"), ns)
    clip = ns["clip"]

    reader = VspipeWindowSource(str(script), clip.width, clip.height,
                                clip.num_frames, vspipe=VSPIPE)
    assert reader.info() == (clip.width, clip.height, clip.num_frames)
    # The pipe's dtype is read from the script, never assumed: guessing wrong
    # reinterprets the bytes into a garbled frame instead of raising.
    assert reader.raw_dtype == (np.float16 if fmt == "RGBH" else np.float32)

    # Spans both ends of the clip, so the reflected slots are checked too.
    feed_start, feed_end = -5, n + 5
    got = np.empty((feed_end - feed_start, 3, clip.height, clip.width),
                   dtype=np.float32)
    reader.read_into(got, feed_start, feed_end)

    for k in range(feed_end - feed_start):
        frame = clip.get_frame(reflect_idx(feed_start + k, n))
        for c in range(3):
            assert np.array_equal(got[k, c], np.asarray(frame[c])), \
                f"slot {k} plane {c} differs from get_frame"


@pytest.mark.skipif(not VSPIPE, reason="vspipe not installed")
def test_the_reader_refuses_a_source_that_is_not_float_rgb(tmp_path):
    """An integer source would be read as float and come out as noise."""
    from bsvd_windowed import VspipeWindowSource
    script = tmp_path / "synth.vpy"
    script.write_text(synth(4, "YUV420P8"))
    reader = VspipeWindowSource(str(script), 8, 4, 4, vspipe=VSPIPE)
    with pytest.raises(RuntimeError, match="must be float RGB"):
        reader.info()


def test_the_reader_resolves_vspipe_the_way_dispatch_does(tmp_path, monkeypatch):
    """Parent and child must be the same VapourSynth build.

    The MIGraphX lane pins $VSPIPE to its own interpreter's binary
    (dispatch_cmd.py), so a reader that only looked at PATH would decode the
    window with a different build than the graph it feeds.
    """
    from bsvd_windowed import VspipeWindowSource
    script = tmp_path / "synth.vpy"
    script.write_text(synth(1))

    monkeypatch.setenv("VSPIPE", "/pinned/vspipe")
    assert VspipeWindowSource(str(script), 8, 4, 1).vspipe == "/pinned/vspipe"
    # An explicit argument still wins over the environment.
    assert VspipeWindowSource(str(script), 8, 4, 1,
                              vspipe="/explicit/vspipe").vspipe == "/explicit/vspipe"

    monkeypatch.delenv("VSPIPE")
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match=r"vspipe not found"):
        VspipeWindowSource(str(script), 8, 4, 1)


@pytest.mark.skipif(not VSPIPE, reason="vspipe not installed")
def test_the_reader_reports_a_geometry_mismatch(tmp_path):
    """A source script that drifted from the clip must fail, not blur."""
    np = pytest.importorskip("numpy")
    from bsvd_windowed import VspipeWindowSource

    script = tmp_path / "synth.vpy"
    script.write_text(synth(10))
    reader = VspipeWindowSource(str(script), 8, 4, 10, vspipe=VSPIPE)
    assert reader.info() == (8, 4, 10)

    short = VspipeWindowSource(str(script), 8, 4, 99, vspipe=VSPIPE)
    got = np.empty((5, 3, 4, 8), dtype=np.float32)
    with pytest.raises(RuntimeError, match="stopped early|exited"):
        short.read_into(got, 90, 95)
