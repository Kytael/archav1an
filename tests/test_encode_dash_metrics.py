from tools.encode_dash.metrics import render

SNAP = {
    "batch": {"running": True, "pid": 42},
    "roster_error": None,
    "manifest_error": None,
    "encode": {"slots": 6, "lp_level": 4},
    "totals": {"clips": 3389, "done": 1204, "failed": 7, "queued": 2178,
               "frames": 17380000, "frames_done": 6104882,
               "fps_live": 18.9, "eta_finish": 1787680000.0},
    "lanes": [
        {"name": "gpu1_4090", "enabled": True, "state": "working",
         "fps_live": 15.6, "fps_recent": 15.2, "clips_done": 402,
         "phase_split": {"stage": 0.09, "work": 0.88, "publish": 0.03}},
        {"name": "2070s", "enabled": False, "state": "off", "fps_live": None,
         "fps_recent": None, "clips_done": 0, "phase_split": None},
    ],
    "queue": [], "failures": [],
}


def _parse(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        out[name] = float(value)
    return out


def test_every_sample_parses_as_prometheus_text():
    got = _parse(render(SNAP))
    assert got["encode_batch_up"] == 1.0
    assert got['encode_clips{status="done"}'] == 1204.0
    assert got['encode_clips{status="queued"}'] == 2178.0
    assert got["encode_frames_done_total"] == 6104882.0


def test_a_lane_gets_one_label_set_per_metric():
    got = _parse(render(SNAP))
    assert got['encode_lane_enabled{lane="gpu1_4090"}'] == 1.0
    assert got['encode_lane_enabled{lane="2070s"}'] == 0.0
    assert got['encode_lane_busy{lane="gpu1_4090"}'] == 1.0
    assert got['encode_lane_busy{lane="2070s"}'] == 0.0
    assert got['encode_lane_fps_live{lane="gpu1_4090"}'] == 15.6


def test_an_unknown_rate_is_omitted_rather_than_reported_as_zero():
    """Zero fps and 'no measurement yet' are different states, and a graph that
    conflates them shows a lane flatlining when it has simply not started."""
    text = render(SNAP)
    assert 'encode_lane_fps_live{lane="2070s"}' not in text
    assert 'encode_lane_fps{lane="2070s"}' not in text


def test_a_measured_zero_is_reported_not_omitted():
    """The mirror of the test above, and the one that matters for alerting. A
    windowed lane reads 0 fps between sweeps and a wedged run reads 0 on every
    lane. If a real zero were dropped as 'unknown', an alert on a stalled lane
    could never fire."""
    snap = dict(SNAP, lanes=[dict(SNAP["lanes"][0], fps_live=0.0)])
    got = _parse(render(snap))
    assert got['encode_lane_fps_live{lane="gpu1_4090"}'] == 0.0


def test_a_broken_roster_or_manifest_raises_the_flag():
    got = _parse(render(dict(SNAP, roster_error="duplicate port", lanes=[])))
    assert got["encode_roster_error"] == 1.0
    got = _parse(render(dict(SNAP, manifest_error="will not parse")))
    assert got["encode_manifest_error"] == 1.0
    clean = _parse(render(SNAP))
    assert clean["encode_roster_error"] == 0.0
    assert clean["encode_manifest_error"] == 0.0


def test_every_metric_carries_help_and_type():
    text = render(SNAP)
    for name in ("encode_batch_up", "encode_clips",
                 "encode_frames", "encode_frames_done_total",
                 "encode_lane_enabled", "encode_lane_busy",
                 "encode_lane_fps_live", "encode_lane_fps",
                 "encode_lane_clips_done_total", "encode_lane_phase_ratio",
                 "encode_roster_error", "encode_manifest_error"):
        assert f"# HELP {name} " in text, f"{name} has no HELP"
        assert f"# TYPE {name} " in text, f"{name} has no TYPE"


def test_a_lane_name_with_a_quote_is_escaped():
    snap = dict(SNAP, lanes=[dict(SNAP["lanes"][0], name='we"ird')])
    text = render(snap)
    assert 'lane="we\\"ird"' in text


def test_no_clips_omits_the_frames_counter_rather_than_zeroing_it():
    """model._clips yields nothing when the manifest is missing or unparseable.
    Emitting frames_done as 0 there would look to Prometheus like a counter
    reset, and the repair would look like millions of frames of fresh progress
    -- so increase() over the next four hours is nonsense and the no-progress
    alert cannot fire during exactly the window something is wrong."""
    snap = dict(SNAP, manifest_error="will not parse",
                totals=dict(SNAP["totals"], clips=0, frames=0, frames_done=0))
    text = render(snap)
    assert "encode_frames_done_total" not in text
    assert "encode_frames" not in text
    # The error flag is still raised, so the absence is explained.
    assert _parse(text)["encode_manifest_error"] == 1.0


def test_an_empty_run_still_renders():
    """The daemon starts before any run exists. Prometheus scrapes it anyway,
    and a scrape that errors is indistinguishable from the host being down."""
    empty = {"batch": {"running": False, "pid": None},
             "roster_error": None, "manifest_error": None, "encode": None,
             "totals": {"clips": 0, "done": 0, "failed": 0, "queued": 0,
                        "frames": 0, "frames_done": 0, "fps_live": None,
                        "eta_finish": None},
             "lanes": [], "queue": [], "failures": []}
    got = _parse(render(empty))
    assert got["encode_batch_up"] == 0.0
