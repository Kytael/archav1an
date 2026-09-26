"""Encode jobs: a second job type that skips the denoiser."""
import threading
import time

import pytest

from tools.archive_batch.dispatch_cmd import (ENCODER_PARAMS, build_command,
                                             build_encode_command)
from tools.archive_batch.presets import PresetError, catalogue, load_preset
from tools.archive_batch.manifest import Clip, parse_encode_manifest, parse_manifest
from tools.archive_batch.roster import load_roster
from tools.archive_batch.scheduler import Scheduler
from tools.archive_batch.transfer import (TransferError, safe_dest, stage_cmd,
                                          stage_job_cmd)

ROSTER = '''\
[[denoiser]]
name    = "gpu1_4090"
host    = "gpu1"
backend = "trt"
device  = 0
tiling  = "none"
enabled = true

[[encoder]]
name      = "encoder-host"
host      = "local"
port_base = 5300
slots     = 2
lp_level  = 4
enabled   = true
'''


@pytest.fixture
def roster(tmp_path):
    p = tmp_path / "denoisers.toml"
    p.write_text(ROSTER)
    return lambda: load_roster(str(p))


def archive_clip(name="x"):
    return Clip(f"SetA/2001/a/{name}.MOV", "SetA/2001/a", name, 1, 100)


def encode_job(name="j", host="gpu1", dest="SetA/2026/New"):
    return Clip(f"/mnt/media/fresh/{name}.MOV", dest, name, 1, 100,
                denoise=False, src_host=host)


# --- The manifest -----------------------------------------------------------

def test_encode_manifest_rel_dir_is_the_dest_not_the_source_parent():
    """The two strings were always the same for an archive clip and code
    derived either from either. For an encode job they are unrelated."""
    rows = parse_encode_manifest(
        "gpu1\tSetA/2026/Comp\t\t/mnt/media/fresh/a.MOV\t123"
        "\t30000/1001,900\t30.0")
    assert len(rows) == 1
    c = rows[0]
    assert c.src == "/mnt/media/fresh/a.MOV"
    assert c.rel_dir == "SetA/2026/Comp"     # not "/mnt/media/fresh"
    assert c.stem == "a" and c.size == 123 and c.frames == 900
    assert c.denoise is False and c.src_host == "gpu1"
    assert c.preset == ""


def test_encode_manifest_reads_the_preset_from_the_third_column():
    """Third, among the columns this program writes. The ffprobe ones trail
    off -- a source with an embedded thumbnail emits a second rate column --
    so a preset read from the end would be a duration for some files."""
    rows = parse_encode_manifest(
        "gpu1\tSetA/2026/Comp\trun_linux_anime_crf18.sh"
        "\t/mnt/media/fresh/a.MOV\t123\t30000/1001,900\t30000/1001,900\t30.0")
    assert [c.preset for c in rows] == ["run_linux_anime_crf18.sh"]
    assert rows[0].frames == 900, "the second rate column shifted the read"


def test_encode_manifest_skips_a_short_row_as_parse_manifest_does():
    rows = parse_encode_manifest("gpu1\tdest\t\t/a.MOV\n"
                                 "gpu1\tdest\t\t/b.MOV\t1\t30,2\t1.0")
    assert [c.stem for c in rows] == ["b"]


def test_an_archive_clip_still_denoises_by_default():
    """The defaulted fields are what keep every existing manifest, record and
    test meaning what they meant."""
    c = parse_manifest("SetA/2001/a/x.MOV\t1\t30,100\t3.3")[0]
    assert c.denoise is True and c.src_host == "" and not c.is_encode_job


# --- Selective take ---------------------------------------------------------

def test_take_returns_the_oldest_match_and_preserves_the_rest(roster, tmp_path):
    a, b, c = archive_clip("a"), encode_job("b"), archive_clip("c")
    s = Scheduler([a, b, c], roster, lambda *_: None,
                  state_path=tmp_path / "state.jsonl")
    assert s._take(False).stem == "b"
    assert [x.stem for x in list(s.queue.queue)] == ["a", "c"]
    assert s._take(True).stem == "a"
    assert [x.stem for x in list(s.queue.queue)] == ["c"]


def test_take_returns_none_when_no_job_matches(roster, tmp_path):
    s = Scheduler([archive_clip()], roster, lambda *_: None,
                  state_path=tmp_path / "state.jsonl")
    assert s._take(False) is None
    assert s.queue.qsize() == 1, "a non-matching take must not consume"


def test_a_lane_worker_never_takes_an_encode_job(roster, tmp_path):
    took = []
    s = Scheduler([encode_job()], roster,
                  lambda c, d, e, sl: took.append(c) or (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.01
    t = threading.Thread(target=s._worker, args=("gpu1_4090",), daemon=True)
    t.start()
    time.sleep(0.2)
    s.stop()
    t.join(5)
    assert took == [], "a lane worker ran an encode job"
    assert s.queue.qsize() == 1


def test_a_slot_worker_never_takes_a_denoise_job(roster, tmp_path):
    took = []
    s = Scheduler([archive_clip()], roster,
                  lambda c, d, e, sl: took.append(c) or (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.01
    t = threading.Thread(target=s._slot_worker, args=("encoder-host",), daemon=True)
    t.start()
    time.sleep(0.2)
    s.stop()
    t.join(5)
    assert took == [], "a slot worker ran a denoise job"


# --- Priority ---------------------------------------------------------------

def test_the_last_free_slot_is_left_for_the_lanes(roster, tmp_path):
    """An encoder with one free slot and a denoise job still queued must not
    let a slot worker take it: a lane can only reach that host through a slot.
    """
    s = Scheduler([archive_clip(), encode_job()], roster, lambda *_: None,
                  state_path=tmp_path / "state.jsonl")
    encoder = s._read_roster().encoders[0]
    # Occupy one of the two slots, as a lane would.
    held = s._try_take(encoder)
    assert held is not None
    took = []
    s.runner = lambda c, d, e, sl: took.append(c) or (True, 1.0, 1.0, 1, "")
    s.POLL_SECONDS = 0.01
    t = threading.Thread(target=s._slot_worker, args=("encoder-host",), daemon=True)
    t.start()
    time.sleep(0.2)
    s.stop()
    t.join(5)
    assert took == [], "the slot worker took the lanes' last free slot"


def test_the_last_slot_is_taken_once_no_denoise_work_remains(roster, tmp_path):
    s = Scheduler([encode_job()], roster, lambda *_: None,
                  state_path=tmp_path / "state.jsonl")
    encoder = s._read_roster().encoders[0]
    assert s._try_take(encoder) is not None
    took = []
    s.runner = lambda c, d, e, sl: took.append(c) or (True, 1.0, 1.0, 1, "")
    s.POLL_SECONDS = 0.01
    t = threading.Thread(target=s._slot_worker, args=("encoder-host",), daemon=True)
    t.start()
    for _ in range(200):
        if took:
            break
        time.sleep(0.02)
    s.stop()
    t.join(5)
    assert [c.stem for c in took] == ["j"]


def test_the_last_slot_is_taken_when_every_lane_is_disabled(tmp_path):
    """The reserve is FOR the lanes. With none enabled it holds a slot open
    for nobody, and an encode-only run loses one slot per encoder for as long
    as an archive clip sits queued -- which, with no lane, is for ever."""
    p = tmp_path / "denoisers.toml"
    p.write_text(ROSTER.replace("""tiling  = "none"
enabled = true""", """tiling  = "none"
enabled = false"""))
    s = Scheduler([archive_clip(), encode_job()], lambda: load_roster(str(p)),
                  lambda *_: None, state_path=tmp_path / "state.jsonl")
    encoder = s._read_roster().encoders[0]
    assert s._try_take(encoder) is not None
    took = []
    s.runner = lambda c, d, e, sl: took.append(c) or (True, 1.0, 1.0, 1, "")
    s.POLL_SECONDS = 0.01
    t = threading.Thread(target=s._slot_worker, args=("encoder-host",), daemon=True)
    t.start()
    for _ in range(200):
        if took:
            break
        time.sleep(0.02)
    s.stop()
    t.join(5)
    assert [c.stem for c in took] == ["j"]


def test_the_last_slot_is_taken_when_no_lane_allows_this_encoder(tmp_path):
    """Same rule one step finer: the lane is on, but its allowlist names
    another host, so it can never reach this one through that slot."""
    p = tmp_path / "denoisers.toml"
    p.write_text(ROSTER.replace("""tiling  = "none"
enabled = true""", """tiling   = "none"
encoders = ["elsewhere"]
enabled  = true""") + """
[[encoder]]
name      = "elsewhere"
host      = "gpu2"
stream_ip = "10.0.0.12"
port_base = 5400
slots     = 2
lp_level  = 4
enabled   = true
""")
    s = Scheduler([archive_clip(), encode_job()], lambda: load_roster(str(p)),
                  lambda *_: None, state_path=tmp_path / "state.jsonl")
    encoder = s._read_roster().encoders[0]
    assert s._try_take(encoder) is not None
    took = []
    s.runner = lambda c, d, e, sl: took.append(c) or (True, 1.0, 1.0, 1, "")
    s.POLL_SECONDS = 0.01
    t = threading.Thread(target=s._slot_worker, args=("encoder-host",), daemon=True)
    t.start()
    for _ in range(200):
        if took:
            break
        time.sleep(0.02)
    s.stop()
    t.join(5)
    assert [c.stem for c in took] == ["j"]


def test_a_slot_worker_on_a_disabled_encoder_parks(tmp_path):
    p = tmp_path / "denoisers.toml"
    p.write_text(ROSTER.replace("""slots     = 2
lp_level  = 4
enabled   = true""", """slots     = 2
lp_level  = 4
enabled   = false"""))
    took = []
    s = Scheduler([encode_job()], lambda: load_roster(str(p)),
                  lambda c, d, e, sl: took.append(c) or (True, 1.0, 1.0, 1, ""),
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.01
    t = threading.Thread(target=s._slot_worker, args=("encoder-host",), daemon=True)
    t.start()
    time.sleep(0.2)
    s.stop()
    t.join(5)
    assert took == []


def test_a_slot_worker_does_not_hold_a_slot_with_no_job(roster, tmp_path):
    """Slot first, then job -- but the slot goes back the moment there is no
    job, or the worker idles the host for the rest of the run."""
    s = Scheduler([archive_clip()], roster, lambda *_: None,
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.01
    t = threading.Thread(target=s._slot_worker, args=("encoder-host",), daemon=True)
    t.start()
    time.sleep(0.2)
    assert s._slots_in_use("encoder-host") == 0
    s.stop()
    t.join(5)


# --- The command ------------------------------------------------------------

def _encoder(roster_fn):
    return roster_fn().encoders[0]


def test_an_encode_job_command_carries_no_bsvd_flags(roster):
    argv, env = build_encode_command(_encoder(roster), 0, "/s/in.MOV",
                                     "/s/out.mkv", "encoder-host-0")
    joined = " ".join(argv)
    assert "--denoise-bsvd" not in joined
    assert "--bsvd" not in joined
    assert "--remote-denoise" not in joined
    assert "--remote-source" not in joined
    assert "--remote-callback" not in joined
    # The fleet-fixed settings are kept verbatim.
    assert "--quality" in joined and "--photon-noise" in joined
    assert "--speed" in joined and "--encoder-params" in joined
    assert "--lp" in argv and str(_encoder(roster).lp_level) in argv
    assert env == {}


def test_a_local_encoder_gets_no_remote_encode_flags(roster):
    argv, _ = build_encode_command(_encoder(roster), 0, "/s/in.MOV",
                                   "/s/out.mkv", "encoder-host-0")
    assert "--remote-encode" not in argv
    assert "--remote-port" not in argv


def test_a_remote_encoder_gets_the_remote_encode_flags(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ROSTER.replace('''name      = "encoder-host"
host      = "local"
port_base = 5300''', '''name      = "gpu4"
host      = "gpu4"
root      = "/opt/x"
stream_ip = "10.0.0.14"
port_base = 5300'''))
    enc = load_roster(str(p)).encoders[0]
    argv, _ = build_encode_command(enc, 1, "/s/in.MOV", "/s/out.mkv", "gpu4-1")
    assert "--remote-encode" in argv and "gpu4" in argv
    assert "10.0.0.14" in argv
    assert "--remote-encode-root" in argv and "/opt/x" in argv
    # Slot 1 of a block based at 5300.
    assert str(enc.port_for(1)) in argv


def test_two_slot_workers_on_one_host_differ_in_tag_and_port(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text(ROSTER.replace('''name      = "encoder-host"
host      = "local"
port_base = 5300''', '''name      = "gpu4"
host      = "gpu4"
stream_ip = "10.0.0.14"
port_base = 5300'''))
    enc = load_roster(str(p)).encoders[0]
    a, _ = build_encode_command(enc, 0, "/s/a.MOV", "/s/a.mkv", "gpu4-0")
    b, _ = build_encode_command(enc, 1, "/s/b.MOV", "/s/b.mkv", "gpu4-1")
    assert a[a.index("--temp-tag") + 1] != b[b.index("--temp-tag") + 1]
    assert a[a.index("--remote-port") + 1] != b[b.index("--remote-port") + 1]


def test_a_preset_replaces_the_four_fleet_fixed_settings(roster, tmp_path):
    script = tmp_path / "run_linux_test_crf18.sh"
    script.write_text('pipeline.py --quality 18 --photon-noise 2 '
                      '--final-speed 6 --final-params "--tune 0 --keyint 240"\n')
    preset = load_preset("run_linux_test_crf18.sh", repo=str(tmp_path))
    argv, _ = build_encode_command(_encoder(roster), 0, "/s/in.MOV",
                                   "/s/out.mkv", "encoder-host-0", preset=preset)
    assert argv[argv.index("--quality") + 1] == "18"
    assert argv[argv.index("--photon-noise") + 1] == "2"
    assert argv[argv.index("--speed") + 1] == "6"
    assert argv[argv.index("--encoder-params") + 1] == "--tune 0 --keyint 240"


def test_no_preset_leaves_the_fleet_fixed_settings(roster):
    """What every encode job got before presets existed, unchanged."""
    argv, _ = build_encode_command(_encoder(roster), 0, "/s/in.MOV",
                                   "/s/out.mkv", "encoder-host-0")
    assert argv[argv.index("--quality") + 1] == "27"
    assert argv[argv.index("--photon-noise") + 1] == "6"
    assert argv[argv.index("--speed") + 1] == "4"
    assert argv[argv.index("--encoder-params") + 1] == ENCODER_PARAMS


def test_a_preset_never_hands_dispatch_a_second_lp(roster, tmp_path):
    """--lp belongs to the encode HOST, and build_encode_command already
    passes the roster's. A second one loses the argument-order lottery."""
    script = tmp_path / "run_linux_test_crf18.sh"
    script.write_text('pipeline.py --quality 18 --final-speed 4 '
                      '--final-params "--tune 0 --lp 3 --keyint 240"\n')
    preset = load_preset("run_linux_test_crf18.sh", repo=str(tmp_path))
    argv, _ = build_encode_command(_encoder(roster), 0, "/s/in.MOV",
                                   "/s/out.mkv", "encoder-host-0", preset=preset)
    assert argv.count("--lp") == 1
    assert "--lp" not in argv[argv.index("--encoder-params") + 1]


# --- The preset catalogue ---------------------------------------------------

def test_the_catalogue_reads_both_script_shapes(tmp_path):
    """A pipeline.py script names --final-speed/--final-params; a direct
    dispatch script names --speed/--encoder-params. Neither may read the
    other's flag."""
    (tmp_path / "run_linux_pipe_crf15.sh").write_text(
        'pipeline.py --quality 15 --speed 8 --final-speed 4 '
        '--final-params "--tune 0"\n')
    (tmp_path / "run_linux_direct_crf27.sh").write_text(
        'svtav1-dispatch.py --quality 27 --speed 5 --encoder-params "--tune 3"\n')
    got = {p["id"]: p for p in catalogue(repo=str(tmp_path))}
    # --final-speed wins where both appear: 8 is the fast pass, which an
    # encode job does not run.
    assert got["run_linux_pipe_crf15.sh"]["speed"] == "4"
    assert got["run_linux_pipe_crf15.sh"]["params"] == "--tune 0"
    assert got["run_linux_direct_crf27.sh"]["speed"] == "5"
    assert got["run_linux_direct_crf27.sh"]["params"] == "--tune 3"


def test_the_catalogue_skips_a_script_it_can_use_nothing_from(tmp_path):
    """av1an-batch-*.sh drive av1an directly and share no flag with dispatch.
    An empty entry in the picker is one an operator would try."""
    (tmp_path / "run_linux_useless.sh").write_text("av1an -i in.mkv\n")
    assert catalogue(repo=str(tmp_path)) == []


def test_an_unknown_preset_is_refused_by_name(tmp_path):
    with pytest.raises(PresetError, match="nope.sh"):
        load_preset("nope.sh", repo=str(tmp_path))


def test_every_preset_in_the_repo_parses():
    """The scripts ARE the catalogue, so a script the parser cannot read is a
    preset silently missing from the picker."""
    got = catalogue()
    ids = {p["id"] for p in got}
    assert "run_linux_dance_HQ_crf27.sh" in ids
    assert len(ids) >= 11
    for preset in got:
        assert preset["quality"].isdigit() and preset["speed"].isdigit()
        assert preset["label"] and "_" not in preset["label"]


# --- Staging and the destination -------------------------------------------

def test_stage_job_cmd_takes_an_absolute_source_on_a_named_host():
    cmd = stage_job_cmd("gpu1", "/mnt/media/fresh/a.MOV", "/stage/encoder-host-0")
    assert "gpu1:/mnt/media/fresh/a.MOV" in cmd


def test_stage_job_cmd_treats_local_as_no_host():
    """'local' is a roster host NAME, not a machine: as an rsync prefix it
    becomes the ssh target local:/path, which resolves to nothing."""
    for host in ("local", ""):
        cmd = stage_job_cmd(host, "/mnt/media/fresh/a.MOV", "/stage/x")
        assert "/mnt/media/fresh/a.MOV" in cmd
        assert not any(c.startswith("local:") for c in cmd)


def test_stage_cmd_still_refuses_an_absolute_archive_source():
    """That refusal is load-bearing: for an archive clip src is relative to
    ARCHIVE_ROOT and an absolute one would silently escape the root."""
    with pytest.raises(TransferError):
        stage_cmd("gpu1", "/etc/passwd", "/stage/x")


def test_stage_job_cmd_refuses_a_relative_source():
    with pytest.raises(TransferError):
        stage_job_cmd("gpu1", "relative/a.MOV", "/stage/x")


def test_safe_dest_refuses_an_absolute_destination():
    """Stripping it instead would turn /mnt/media/dance into
    encoded/mnt/media/dance and publish there without a word."""
    with pytest.raises(TransferError):
        safe_dest("/mnt/media/dance")


def test_safe_dest_refuses_an_escape():
    for bad in ("../outside", "a/../../b", ".."):
        with pytest.raises(TransferError):
            safe_dest(bad)


def test_safe_dest_normalises_a_trailing_slash():
    assert safe_dest("SetA/2026/New/") == "SetA/2026/New"


def test_safe_dest_requires_something():
    for bad in ("", "   ", None):
        with pytest.raises(TransferError):
            safe_dest(bad)


# --- Submission -------------------------------------------------------------

def test_submit_queues_new_jobs(roster, tmp_path):
    s = Scheduler([], roster, lambda *_: None, state_path=tmp_path / "state.jsonl")
    assert s.submit([encode_job("a"), encode_job("b")]) == 2
    assert s.queue.qsize() == 2


def test_submit_refuses_a_src_already_queued(roster, tmp_path):
    """retry()'s rule: two copies of one job would be encoded twice and
    published to one destination path at the same time."""
    job = encode_job("a")
    s = Scheduler([job], roster, lambda *_: None,
                  state_path=tmp_path / "state.jsonl")
    assert s.submit([job]) == 0
    assert s.queue.qsize() == 1


def test_submit_refuses_a_src_already_in_flight(roster, tmp_path):
    job = encode_job("a")
    s = Scheduler([], roster, lambda *_: None,
                  state_path=tmp_path / "state.jsonl")
    with s._lock:
        s._in_flight.add(job.src)
    assert s.submit([job]) == 0
    assert s.queue.empty()


# --- The probe --------------------------------------------------------------

def test_probe_reads_a_real_folder(tmp_path):
    """Against ffprobe itself, not a stub: the row shape is the contract
    between this walk and parse_encode_manifest."""
    from tools.archive_batch.probe import probe_folder
    import shutil as _shutil
    src = None
    for candidate in ("Input", "tests/fixtures"):
        import glob
        hits = glob.glob(f"{candidate}/*.MOV") + glob.glob(f"{candidate}/*.mp4")
        if hits:
            src = hits[0]
            break
    if src is None:
        pytest.skip("no sample video in this checkout")
    _shutil.copy(src, tmp_path / "sample.MOV")
    rows = probe_folder("local", str(tmp_path))
    assert len(rows) == 1
    name, size, rate = rows[0]
    assert name == "sample.MOV" and size > 0
    assert "," in rate, "the rate column must carry rate,frames"


def test_probe_refuses_a_relative_path():
    from tools.archive_batch.probe import probe_folder
    with pytest.raises(TransferError):
        probe_folder("local", "relative/dir")


def test_probe_refuses_a_missing_folder(tmp_path):
    from tools.archive_batch.probe import probe_folder
    with pytest.raises(TransferError, match="not a directory"):
        probe_folder("local", str(tmp_path / "nope"))


# --- The dashboard's two populations ---------------------------------------

def test_a_slot_worker_heartbeat_is_not_read_as_a_vanished_lane():
    """model splits by NAME. A slot worker's file is not a lane that has been
    removed from the roster, and reading it as one drops the running job from
    the page."""
    from tools.encode_dash.model import _encoder_of
    assert _encoder_of("encoder-host-0") == "encoder-host"
    assert _encoder_of("gpu4-2") == "gpu4"
    # A lane name has no numeric slot tail.
    assert _encoder_of("gpu1_4090") == ""


def test_an_encoder_name_with_a_dash_survives_the_split():
    """This fleet has an encoder called encoder-host-lp6. An unconditional
    rsplit("-", 1) reads its slot-1 worker as encoder "encoder-host" slot "lp6",
    which matches the wrong host -- the same class of bug that made yielding
    encode host gpu4 kill denoise lane gpu4."""
    from tools.encode_dash.model import _encoder_of
    assert _encoder_of("encoder-host-lp6-1") == "encoder-host-lp6"
    assert _encoder_of("encoder-host-lp6") == ""


def test_the_submit_form_does_not_offer_the_archive_run():
    """The archive run is driven by manifest-raw.tsv, it denoises, and it is
    started by its own button. It was briefly listed here as a folder, which
    offered a choice and then refused half of it -- the entry blanked the
    fields and disabled the button. Not listing it is the honest shape."""
    import os as _os
    base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    js = open(_os.path.join(base, "tools", "encode_dash", "static", "app.js"),
              encoding="utf-8").read()
    body = js[js.index("const JOB_FIELDS"):js.index("function buildHostForm")]
    assert "/api/jobs/submit" in body, "the form must post a submission"
    for key in ("host", "path", "dest"):
        assert f'["{key}"' in body, f"the form cannot set {key}"
    # No dead entry, and no disabled-submit machinery left behind with it.
    assert "JOB_PLACES" not in js
    assert "archive" not in body.lower(), \
        "the archive run must not be offered as a submittable folder"


def test_the_folder_picker_is_not_hidden_with_the_jobs_table():
    """The section holds "Encode a folder" as well as the table, so hiding the
    SECTION when nothing is running made the only control that creates a job
    appear only once a job already existed."""
    import os as _os
    base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    js = open(_os.path.join(base, "tools", "encode_dash", "static", "app.js"),
              encoding="utf-8").read()
    body = js[js.index("function renderJobs"):js.index("function buildJobForm")]
    assert 'closest("section")' not in body, \
        "renderJobs hides the section that holds the submit control"
    assert '#jobs").hidden' in body, "the table itself must be what hides"


def test_the_long_panels_are_scroll_windows():
    """Failed carries up to 50 rows, each with a reason block. Unbounded it
    pushes the rest of the page off the screen, and the two panels sit side by
    side where unequal heights read as a layout bug."""
    import os as _os
    base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    html = open(_os.path.join(base, "tools", "encode_dash", "static", "app.html"),
                encoding="utf-8").read()
    css = open(_os.path.join(base, "tools", "encode_dash", "static", "app.css"),
               encoding="utf-8").read()
    assert html.count('class="scroll"') == 2, "queue and failures both need one"
    assert 'class="scroll"><table id="failures"' in html
    assert ".scroll" in css and "overflow-y: auto" in css
    assert "max-height" in css


# --- Rates on encode rows ---------------------------------------------------

def _beat(src, temp_dir, frames=100, started=1000.0):
    return {"src": src, "frames": frames, "state": "working",
            "started_at": started, "temp_dir": temp_dir}


def test_a_job_row_carries_the_rate_from_the_same_two_sources_a_lane_uses():
    """A slot worker writes the same heartbeat and dispatch writes the same
    log, so the live figure needs no new machinery -- only the call."""
    from tools.encode_dash.liverate import RateTracker
    from tools.encode_dash.model import _job
    tracker = RateTracker()
    history = {"enc-0": [{"fps": 10.0}, {"fps": 20.0}]}
    row = _job("enc-0", _beat("", ""), history, tracker, 1000.0)
    assert row["encoder"] == "enc" and row["state"] == "working"
    assert row["fps_recent"] == 15.0 and row["clips_done"] == 2
    assert row["fps_live"] is None, "idle, so nothing to measure"


def test_a_job_row_reports_progress_from_the_dispatch_log(tmp_path):
    from tools.encode_dash.liverate import RateTracker
    from tools.encode_dash.model import _job
    temp = tmp_path / "enc-0" / "a"
    temp.mkdir(parents=True)
    (temp / "a_encode.log").write_text("Encoding: 40/100 Frames\n")
    tracker = RateTracker()
    beat = _beat("/f/a.MOV", str(temp), frames=100, started=900.0)
    row = _job("enc-0", beat, {}, tracker, 1000.0)
    assert row["frames_done"] == 40
    assert row["progress"] == 0.4
    # One sample cannot make a slope; the second one can.
    assert row["fps_live"] is None
    (temp / "a_encode.log").write_text("Encoding: 90/100 Frames\n")
    row = _job("enc-0", beat, {}, tracker, 1050.0)
    assert row["fps_live"] == 1.0
    assert row["eta_s"] == 10.0


def test_the_host_rate_sums_the_slots_but_averages_the_clips():
    """Two slots encoding at once really do produce both rates, so live sums.
    fps_recent is per clip, and summing it would report a two-slot host as
    twice as fast as it is."""
    from tools.encode_dash.model import _encoder_rates
    jobs = [{"name": "enc-0", "encoder": "enc", "fps_live": 4.0},
            {"name": "enc-1", "encoder": "enc", "fps_live": 5.0}]
    history = {"enc-0": [{"fps": 10.0}], "enc-1": [{"fps": 20.0}]}
    got = _encoder_rates(jobs, history)
    assert got["enc"]["fps_live"] == 9.0
    assert got["enc"]["fps_recent"] == 15.0
    assert got["enc"]["clips_done"] == 2


def test_the_host_rate_is_none_while_every_slot_is_idle():
    """None, not 0.0: an idle host has no rate, and a zero reads as a stall."""
    from tools.encode_dash.model import _encoder_rates
    jobs = [{"name": "enc-0", "encoder": "enc", "fps_live": None}]
    got = _encoder_rates(jobs, {})
    assert got["enc"]["fps_live"] is None
    assert got["enc"]["fps_recent"] is None


def test_the_no_denoise_path_writes_a_progress_log():
    """The dashboard reads frames from <stem>_vspipe.log. This branch was the
    one run_piped call with no log and no -p, so an encode job produced no
    counter anywhere and rendered with no rate and no progress bar for its
    whole length -- until encode jobs existed, nothing in the batch reached
    it."""
    import os as _os
    base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    src = open(_os.path.join(base, "tools", "svtav1-dispatch.py"),
               encoding="utf-8").read()
    # The plain-source branch, from where it builds the vspipe command to the
    # frame-count check that follows both of its arms.
    at = src.index("_src_cmd = [vspipe_exe")
    branch = src[at:src.index("# --- Frame-count verification", at)]
    assert "src_vpy_path" in branch, "found the wrong branch"
    assert '"-p"' in branch, "vspipe prints no progress without -p"
    # Both arms need one: local encodes read the vspipe log, and
    # run_remote_encode writes the same file for the remote arm.
    assert "_vspipe.log" in branch or "run_remote_encode(" in branch, \
        "no stderr log, so no counter to read"
