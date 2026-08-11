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


def test_preset_matches_the_dance_hq_script():
    argv, _ = build_command(REMOTE, ENCODE, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src="/mnt/media/x.MOV", callback="1.2.3.4")
    assert argv[argv.index("--quality") + 1] == "27"
    assert argv[argv.index("--photon-noise") + 1] == "6"
    assert argv[argv.index("--speed") + 1] == "4"
    params = argv[argv.index("--encoder-params") + 1]
    assert "--tune 3" in params and "--variance-octile 7" in params
