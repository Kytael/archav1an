from tools.archive_ui.liverate import RateTracker, frames_from_log


def _write(tmp_path, name, counts):
    (tmp_path / name).write_text("\n".join(f"Frame: {c}/6726" for c in counts))


def test_reads_the_last_count_from_a_vspipe_log(tmp_path):
    _write(tmp_path, "MVI_1_vspipe.log", [1, 2, 3, 4, 250])
    assert frames_from_log(str(tmp_path), "MVI_1") == 250


def test_reads_a_netstream_log_with_the_same_parser(tmp_path):
    """A remote lane is counted at the socket, and netstream deliberately
    emits the same shape vspipe does."""
    (tmp_path / "MVI_2_netstream.log").write_text(
        "[netstream] listening on port 5300\nFrame: 12\nFrame: 96\n")
    assert frames_from_log(str(tmp_path), "MVI_2") == 96


def test_a_log_with_no_counter_yet_is_none_not_zero(tmp_path):
    """Zero would render as a lane doing nothing. None renders as unknown, and
    the difference matters in the first seconds of every clip."""
    (tmp_path / "MVI_3_vspipe.log").write_text("Script evaluation done\n")
    assert frames_from_log(str(tmp_path), "MVI_3") is None


def test_no_log_at_all_is_none(tmp_path):
    assert frames_from_log(str(tmp_path), "nothing") is None


def test_the_newer_log_wins_when_both_exist(tmp_path):
    import os
    (tmp_path / "MVI_4_vspipe.log").write_text("Frame: 5/10\n")
    (tmp_path / "MVI_4_netstream.log").write_text("Frame: 900\n")
    os.utime(tmp_path / "MVI_4_vspipe.log", (1000, 1000))
    os.utime(tmp_path / "MVI_4_netstream.log", (2000, 2000))
    assert frames_from_log(str(tmp_path), "MVI_4") == 900


def test_the_rate_is_the_slope_between_two_samples():
    t = RateTracker(smooth_s=10.0)
    assert t.sample("a", 0, now=0.0) is None, "one point is not a rate"
    assert t.sample("a", 100, now=10.0) == 10.0


def test_a_bursty_window_smooths_to_one_steady_rate():
    """A windowed lane emits 750 frames at once every 138 s. Sampled every 2 s,
    the raw slope alternates between 0 and 375 fps; the true rate is 5.43.

    The smoothing window has to be several sweeps wide, not one. With a step
    input, any finite window sees either N or N+1 bursts depending on phase, so
    the reported rate swings by 1/N. At one sweep that is a factor of two -- the
    very artefact this is supposed to remove. Five sweeps holds it inside 20%.
    """
    t = RateTracker(smooth_s=700.0)         # ~5 sweeps
    frames, now = 0, 0.0
    rates = []
    for _ in range(12):
        for _ in range(68):            # 68 polls of 2 s, nothing published
            now += 2.0
            rates.append(t.sample("w", frames, now=now))
        frames += 750                  # the window lands
        now += 2.0
        rates.append(t.sample("w", frames, now=now))
    settled = [r for r in rates[-40:] if r is not None]
    assert settled, "no rate produced at all"
    # True rate is 750/138 = 5.43. Bounds are deliberately wide enough to
    # survive the N-vs-N+1 phase effect and tight enough that a leaked burst
    # (375 fps) or a dead window (0) fails loudly.
    assert max(settled) < 7.0, f"burst leaked through: {max(settled)}"
    assert min(settled) > 4.0, f"went dead between bursts: {min(settled)}"


def test_a_restarted_clip_does_not_report_a_negative_rate():
    """Frame counts reset to zero when a clip is retried on the same lane.
    A naive slope would go hugely negative."""
    t = RateTracker(smooth_s=5.0)
    t.sample("a", 5000, now=0.0)
    t.sample("a", 5200, now=5.0)
    assert t.sample("a", 3, now=10.0) is None, "should drop history and restart"


def test_lanes_do_not_share_history():
    t = RateTracker(smooth_s=5.0)
    t.sample("a", 0, now=0.0)
    t.sample("b", 1000, now=0.0)
    t.sample("a", 50, now=5.0)
    assert t.sample("b", 1050, now=5.0) == 10.0
