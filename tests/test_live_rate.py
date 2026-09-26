from tools.encode_dash.liverate import RateTracker, frames_from_log


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


def test_a_diagnostic_mentioning_a_frame_is_not_read_as_a_count(tmp_path):
    """Only a line that starts with the counter is a counter. A denoiser log
    ends in diagnostics, and one naming a frame must not be mistaken for the
    lane's progress -- that would show a rate jumping backwards to the frame
    the error happened to name."""
    (tmp_path / "MVI_5_vspipe.log").write_text(
        "Frame: 4210/6726\n"
        "Error Code 1: Cuda Runtime at Frame: 17 in deallocate\n")
    assert frames_from_log(str(tmp_path), "MVI_5") == 4210


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


def test_a_per_lane_window_overrides_the_default():
    """One global value cannot serve both lane kinds: a full-frame lane wants a
    short window so its figure is current, a windowed lane needs several
    sweeps."""
    t = RateTracker(smooth_s=10.0)
    t.sample("w", 0, now=0.0, smooth_s=100.0)
    t.sample("w", 100, now=20.0, smooth_s=100.0)
    # Still inside the 100 s override, so the anchor is the first sample.
    assert t.sample("w", 200, now=40.0, smooth_s=100.0) == 5.0


def test_reads_the_encoder_counter_when_the_encode_is_remote(tmp_path):
    """With a remote encoder the vspipe and netstream logs are both on other
    hosts, and the only counter that reaches this one is the encoder's own,
    inside the encode ssh's stderr. SvtAv1EncApp colours it unconditionally."""
    (tmp_path / "MVI_6_encode.log").write_text(
        "Svt[info]: SVT [version]: v2.3.0-C\n"
        "Encoding: \x1b[33m   1 Frames\x1b[0m @ \x1b[32m7.36\x1b[0m fpm\n"
        "Encoding: \x1b[33m3258/3289 Frames\x1b[0m @ \x1b[32m10.27\x1b[0m fps\n")
    assert frames_from_log(str(tmp_path), "MVI_6") == 3258


def test_the_encoder_counter_is_not_read_from_a_summary_line(tmp_path):
    """The encoder prints a summary table after the last counter. Only a line
    that starts with the counter is a counter, the same rule the vspipe shape
    already follows."""
    (tmp_path / "MVI_7_encode.log").write_text(
        "Encoding: \x1b[33m900/900 Frames\x1b[0m @ \x1b[32m9.1\x1b[0m fps\n"
        "Total Frames    Frame Rate    Byte Count    Bitrate\n"
        "         900        29.97          123456      1234\n")
    assert frames_from_log(str(tmp_path), "MVI_7") == 900


def test_a_windowed_lane_says_nothing_until_it_has_seen_a_whole_sweep():
    """A windowed lane produces nothing for a sweep and then delivers the whole
    window at once, so a slope drawn across less than one sweep measures the
    encoder draining a buffer, not the lane.

    Measured 2026-08-26: the 2070s lane at window 750 read 23.6 fps against a
    true 4.6, its counter having gone 1 -> 669 in the 28 s the encoder spent on
    the first delivered window. Smoothing cannot help -- there is nothing yet
    to smooth."""
    t = RateTracker(smooth_s=400.0)
    assert t.sample("2070s", 1, 0.0, 400.0, 163.0) is None
    # The burst: 669 frames land 28 s later. Still inside one sweep, so silent.
    assert t.sample("2070s", 669, 28.0, 400.0, 163.0) is None
    # Past a full sweep the slope is real and gets reported.
    rate = t.sample("2070s", 760, 165.0, 400.0, 163.0)
    assert rate is not None
    assert abs(rate - 760 / 165.0) < 0.01, rate


def test_a_full_frame_lane_is_unaffected_by_the_sweep_floor():
    """Only a windowed lane passes a min_span_s. Everything else must answer as
    soon as it has two samples, or the fast lanes go blank."""
    t = RateTracker(smooth_s=30.0)
    assert t.sample("gpu1_4090", 0, 0.0) is None      # one sample only
    rate = t.sample("gpu1_4090", 20, 2.0)
    assert rate == 10.0, rate


def test_the_denoiser_counter_wins_over_a_live_encoder_log(tmp_path):
    # A lane that denoises here and encodes on another host writes both files
    # on this host, milliseconds apart, and the two counters sit a pipe's
    # depth apart. Picking the newer alternates between them: every other poll
    # reads a count lower than the last, which the tracker reads as a new clip
    # and answers nothing, and the poll after it reads the whole gap as one
    # step -- 82 frames over one interval showed as 22 fps against a true 3.0.
    import os
    (tmp_path / "MVI_5_vspipe.log").write_text("Frame: 8071/21336\n")
    (tmp_path / "MVI_5_encode.log").write_text(
        "Encoding: \x1b[33m7989 Frames\x1b[0m @ 3.00 fps\n")
    os.utime(tmp_path / "MVI_5_vspipe.log", (2000, 2000))
    os.utime(tmp_path / "MVI_5_encode.log", (2000.004, 2000.004))
    assert frames_from_log(str(tmp_path), "MVI_5") == 8071


def test_a_stale_producer_log_does_not_win_on_priority(tmp_path):
    # Priority decides between producers that are both writing now. A log left
    # by an earlier attempt in the same per-clip directory is not one of them,
    # and its frozen count would pin the lane's rate at zero for ever.
    import os
    (tmp_path / "MVI_6_vspipe.log").write_text("Frame: 400/21336\n")
    (tmp_path / "MVI_6_encode.log").write_text("Encoding: 120 Frames\n")
    os.utime(tmp_path / "MVI_6_vspipe.log", (1000, 1000))
    os.utime(tmp_path / "MVI_6_encode.log", (9000, 9000))
    assert frames_from_log(str(tmp_path), "MVI_6") == 120


def _windowed_series(tracker, phase, window=1000, true_fps=4.2,
                     poll=5.0, drain=30.0, sweeps=4):
    """Feed a windowed lane's step-shaped counter and return its last rate.

    `phase` shifts where the series starts relative to a sweep boundary. The
    lane itself is identical in every case, so every phase must report the
    same rate.
    """
    sweep = window / true_fps
    smooth = 2.5 * sweep
    last, t = None, 0.0
    while t < smooth * sweeps:
        bursts, into = divmod(t + phase, sweep)
        # Whole windows delivered, plus the one the encoder is draining now.
        frames = bursts * window + (min(into, drain) / drain) * window
        rate = tracker.sample("L", int(frames), t, smooth, sweep)
        if rate is not None:
            last = rate
        t += poll
    return last


def test_a_windowed_lane_reads_the_same_rate_at_every_phase():
    # The burst artefact this guards: between raw endpoints a 2.5-sweep window
    # holds 2 or 3 whole bursts depending only on where the series started, so
    # the same lane read 3.33 or 5.00 fps against a true 4.20 -- a 50% spread
    # with nothing about the lane changing. Snapping to burst edges holds a
    # whole number of sweeps by construction, so the phase cancels.
    rates = [_windowed_series(RateTracker(), p)
             for p in (0, 40, 80, 120, 160, 200)]
    assert all(r is not None for r in rates)
    assert max(rates) - min(rates) < 0.01, rates
    # And it is the true rate, not merely a consistent wrong one.
    assert all(abs(r - 4.2) / 4.2 < 0.02 for r in rates), rates


def test_a_full_frame_lane_still_uses_every_sample():
    # No min_span_s means the caller did not call this lane windowed, so there
    # are no bursts to snap to and the slope must span the whole series. A
    # steady 10 fps must read 10, not something shortened by a false edge.
    tr = RateTracker()
    for i in range(41):
        rate = tr.sample("F", i * 10, float(i), 30.0)
    assert abs(rate - 10.0) < 1e-9
