from tools.archive_batch.dispatch_cmd import build_command
from tools.archive_batch.roster import Denoiser, EncodePool

ENCODE = EncodePool(host="local", slots=2, threads_per_slot=16)
REMOTE = Denoiser(name="gpu1_4090", host="gpu1", backend="trt", device=0,
                  tiling="none", enabled=True, port=5300)
LOCAL = Denoiser(name="igpu", host="local", backend="migraphx", device=0,
                 tiling="none", enabled=True)


def test_remote_denoiser_gets_remote_flags():
    argv, env = build_command(REMOTE, ENCODE, staged="/t/x.MOV",
                              out="/t/x-av1.mkv", remote_src="/mnt/media/a/x.MOV",
                              callback="10.0.0.10")
    assert "--remote-denoise" in argv and argv[argv.index("--remote-denoise") + 1] == "gpu1"
    assert argv[argv.index("--remote-source") + 1] == "/mnt/media/a/x.MOV"
    assert argv[argv.index("--remote-port") + 1] == "5300"
    assert argv[argv.index("--remote-callback") + 1] == "10.0.0.10"


def test_local_denoiser_has_no_remote_flags():
    argv, env = build_command(LOCAL, ENCODE, staged="/t/x.MOV",
                              out="/t/x-av1.mkv", remote_src=None, callback=None)
    assert "--remote-denoise" not in argv
    assert "--remote-source" not in argv


def test_sigma_is_always_the_fleet_default():
    argv, _ = build_command(REMOTE, ENCODE, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src="/mnt/media/x.MOV", callback="1.2.3.4")
    assert argv[argv.index("--bsvd-sigma") + 1] == "0.05"


def test_lp_comes_from_the_encode_pool():
    argv, _ = build_command(LOCAL, ENCODE, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src=None, callback=None)
    assert argv[argv.index("--lp") + 1] == "16"


def test_migraphx_denoiser_pins_vspipe_and_interpreter():
    argv, env = build_command(LOCAL, ENCODE, staged="/t/x.MOV", out="/t/o.mkv",
                              remote_src=None, callback=None)
    assert env["VSPIPE"].endswith("migraphx-venv/bin/vspipe")
    assert argv[0].endswith("migraphx-venv/bin/python")


def test_trt_denoiser_uses_the_managed_interpreter():
    argv, env = build_command(REMOTE, ENCODE, staged="/t/x.MOV", out="/t/o.mkv",
                              remote_src="/mnt/media/x.MOV", callback="1.2.3.4")
    assert argv[0] == "/opt/archav1an/venv/bin/python"
    assert "VSPIPE" not in env


def test_temp_tag_isolates_denoisers_from_each_other():
    """185 stems repeat across the archive; a shared temp dir would let one
    worker delete another's working files mid-encode."""
    remote, _ = build_command(REMOTE, ENCODE, staged="/t/MVI_0090.MOV",
                              out="/t/o.mkv", remote_src="/mnt/media/a/MVI_0090.MOV",
                              callback="1.2.3.4")
    local, _ = build_command(LOCAL, ENCODE, staged="/t/MVI_0090.MOV",
                             out="/t/o.mkv", remote_src=None, callback=None)
    assert remote[remote.index("--temp-tag") + 1] == "gpu1_4090"
    assert local[local.index("--temp-tag") + 1] == "igpu"


def test_preset_matches_the_dance_hq_script():
    argv, _ = build_command(REMOTE, ENCODE, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src="/mnt/media/x.MOV", callback="1.2.3.4")
    assert argv[argv.index("--quality") + 1] == "27"
    assert argv[argv.index("--photon-noise") + 1] == "6"
    assert argv[argv.index("--speed") + 1] == "4"
    params = argv[argv.index("--encoder-params") + 1]
    assert "--tune 3" in params and "--variance-octile 7" in params


TILED = Denoiser(name="2070s", host="local", backend="trt", device=0,
                 tiling="auto", enabled=True, window=1500, margin=32)


def test_a_tiled_denoiser_gets_the_windowed_flags():
    argv, _ = build_command(TILED, ENCODE, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src=None, callback=None)
    # "auto" rather than a fixed size: the filter sizes each axis to the
    # frame, which a square tile cannot do on a 16:9 source.
    assert argv[argv.index("--bsvd-tile") + 1] == "auto"
    assert argv[argv.index("--bsvd-window") + 1] == "1500"
    assert argv[argv.index("--bsvd-margin") + 1] == "32"


def test_an_untiled_denoiser_gets_no_windowing_flags():
    argv, _ = build_command(LOCAL, ENCODE, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src=None, callback=None)
    assert "--bsvd-tile" not in argv and "--bsvd-window" not in argv


def test_a_remote_without_the_archive_omits_remote_source():
    """remote_src=None must drop the flag, not pass None as its value.

    Appending None put it straight into argv and subprocess raised
    TypeError('expected str, bytes or os.PathLike object, not NoneType'),
    which the scheduler recorded as a per-clip failure on every remote lane.
    """
    d = Denoiser(name="gpu2_5070", host="gpu2", backend="trt", device=0,
                 tiling="auto", window=500, margin=32, port=5301,
                 stage_source=True, enabled=True)
    argv, _ = build_command(d, ENCODE, staged="/t/x.MOV", out="/t/x-av1.mkv",
                            remote_src=None, callback="10.0.0.10")
    assert "--remote-source" not in argv
    assert all(a is not None for a in argv)
    assert "--remote-denoise" in argv and "gpu2" in argv


def test_a_remote_root_is_forwarded_when_the_checkout_moves():
    d = Denoiser(name="gpu2_5070", host="gpu2", backend="trt", device=0,
                 tiling="auto", window=500, margin=32, port=5301,
                 stage_source=True, root="~/reposetc/archav1an", enabled=True)
    argv, _ = build_command(d, ENCODE, staged="/t/x.MOV", out="/t/x-av1.mkv",
                            remote_src=None, callback="10.0.0.10")
    assert argv[argv.index("--remote-root") + 1] == "~/reposetc/archav1an"


def test_no_remote_root_flag_when_the_default_layout_applies():
    argv, _ = build_command(REMOTE, ENCODE, staged="/t/x.MOV",
                            out="/t/x-av1.mkv", remote_src="/mnt/media/a/x.MOV",
                            callback="1.2.3.4")
    assert "--remote-root" not in argv
