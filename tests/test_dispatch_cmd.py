from tools.archive_batch.dispatch_cmd import build_command
from tools.archive_batch.roster import Denoiser, Encoder

ENCODE = Encoder(name="local", host="local", slots=2, lp_level=6,
                 port_base=5300)
REMOTE = Denoiser(name="gpu1_4090", host="gpu1", backend="trt", device=0,
                  tiling="none", enabled=True)
LOCAL = Denoiser(name="igpu", host="local", backend="migraphx", device=0,
                 tiling="none", enabled=True)


def test_remote_denoiser_gets_remote_flags():
    argv, env = build_command(REMOTE, ENCODE, 0, staged="/t/x.MOV",
                              out="/t/x-av1.mkv", remote_src="/mnt/media/a/x.MOV",
                              callback="10.0.0.10")
    assert "--remote-denoise" in argv and argv[argv.index("--remote-denoise") + 1] == "gpu1"
    assert argv[argv.index("--remote-source") + 1] == "/mnt/media/a/x.MOV"
    assert argv[argv.index("--remote-port") + 1] == "5300"
    assert argv[argv.index("--remote-callback") + 1] == "10.0.0.10"


def test_local_denoiser_has_no_remote_flags():
    argv, env = build_command(LOCAL, ENCODE, 0, staged="/t/x.MOV",
                              out="/t/x-av1.mkv", remote_src=None, callback=None)
    assert "--remote-denoise" not in argv
    assert "--remote-source" not in argv


def test_sigma_is_always_the_fleet_default():
    argv, _ = build_command(REMOTE, ENCODE, 0, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src="/mnt/media/x.MOV", callback="1.2.3.4")
    assert argv[argv.index("--bsvd-sigma") + 1] == "0.05"


def test_lp_comes_from_the_encode_pool():
    argv, _ = build_command(LOCAL, ENCODE, 0, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src=None, callback=None)
    assert argv[argv.index("--lp") + 1] == "6"


def test_migraphx_denoiser_pins_vspipe_and_interpreter():
    argv, env = build_command(LOCAL, ENCODE, 0, staged="/t/x.MOV", out="/t/o.mkv",
                              remote_src=None, callback=None)
    assert env["VSPIPE"].endswith("migraphx-venv/bin/vspipe")
    assert argv[0].endswith("migraphx-venv/bin/python")


def test_trt_denoiser_uses_the_managed_interpreter():
    argv, env = build_command(REMOTE, ENCODE, 0, staged="/t/x.MOV", out="/t/o.mkv",
                              remote_src="/mnt/media/x.MOV", callback="1.2.3.4")
    assert argv[0] == "/opt/archav1an/venv/bin/python"
    assert "VSPIPE" not in env


def test_temp_tag_isolates_denoisers_from_each_other():
    """185 stems repeat across the archive; a shared temp dir would let one
    worker delete another's working files mid-encode."""
    remote, _ = build_command(REMOTE, ENCODE, 0, staged="/t/MVI_0090.MOV",
                              out="/t/o.mkv", remote_src="/mnt/media/a/MVI_0090.MOV",
                              callback="1.2.3.4")
    local, _ = build_command(LOCAL, ENCODE, 0, staged="/t/MVI_0090.MOV",
                             out="/t/o.mkv", remote_src=None, callback=None)
    assert remote[remote.index("--temp-tag") + 1] == "gpu1_4090"
    assert local[local.index("--temp-tag") + 1] == "igpu"


def test_preset_matches_the_dance_hq_script():
    argv, _ = build_command(REMOTE, ENCODE, 0, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src="/mnt/media/x.MOV", callback="1.2.3.4")
    assert argv[argv.index("--quality") + 1] == "27"
    assert argv[argv.index("--photon-noise") + 1] == "6"
    assert argv[argv.index("--speed") + 1] == "4"
    params = argv[argv.index("--encoder-params") + 1]
    assert "--tune 3" in params and "--variance-octile 7" in params


TILED = Denoiser(name="2070s", host="local", backend="trt", device=0,
                 tiling="auto", enabled=True, window=1500, margin=32)


def test_a_tiled_denoiser_gets_the_windowed_flags():
    argv, _ = build_command(TILED, ENCODE, 0, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src=None, callback=None)
    # "auto" rather than a fixed size: the filter sizes each axis to the
    # frame, which a square tile cannot do on a 16:9 source.
    assert argv[argv.index("--bsvd-tile") + 1] == "auto"
    assert argv[argv.index("--bsvd-window") + 1] == "1500"
    assert argv[argv.index("--bsvd-margin") + 1] == "32"


def test_an_untiled_denoiser_gets_no_windowing_flags():
    argv, _ = build_command(LOCAL, ENCODE, 0, staged="/t/x.MOV", out="/t/o.mkv",
                            remote_src=None, callback=None)
    assert "--bsvd-tile" not in argv and "--bsvd-window" not in argv


def test_a_remote_without_the_archive_omits_remote_source():
    """remote_src=None must drop the flag, not pass None as its value.

    Appending None put it straight into argv and subprocess raised
    TypeError('expected str, bytes or os.PathLike object, not NoneType'),
    which the scheduler recorded as a per-clip failure on every remote lane.
    """
    d = Denoiser(name="gpu2_5070", host="gpu2", backend="trt", device=0,
                 tiling="auto", window=500, margin=32,
                 stage_source=True, enabled=True)
    argv, _ = build_command(d, ENCODE, 0, staged="/t/x.MOV", out="/t/x-av1.mkv",
                            remote_src=None, callback="10.0.0.10")
    assert "--remote-source" not in argv
    assert all(a is not None for a in argv)
    assert "--remote-denoise" in argv and "gpu2" in argv


def test_a_remote_root_is_forwarded_when_the_checkout_moves():
    d = Denoiser(name="gpu2_5070", host="gpu2", backend="trt", device=0,
                 tiling="auto", window=500, margin=32,
                 stage_source=True, root="~/reposetc/archav1an", enabled=True)
    argv, _ = build_command(d, ENCODE, 0, staged="/t/x.MOV", out="/t/x-av1.mkv",
                            remote_src=None, callback="10.0.0.10")
    assert argv[argv.index("--remote-root") + 1] == "~/reposetc/archav1an"


def test_no_remote_root_flag_when_the_default_layout_applies():
    argv, _ = build_command(REMOTE, ENCODE, 0, staged="/t/x.MOV",
                            out="/t/x-av1.mkv", remote_src="/mnt/media/a/x.MOV",
                            callback="1.2.3.4")
    assert "--remote-root" not in argv


LOCAL_ENC = Encoder(name="encoder-host", host="local", slots=6, lp_level=6)
GPU2_ENC = Encoder(name="gpu2", host="gpu2", slots=2, lp_level=6,
                    stream_ip="10.0.0.17", port_base=5310,
                    root="/home/user/reposetc/archav1an")
IGPU = Denoiser(name="igpu", host="local", backend="migraphx", device=0,
                tiling="none", enabled=True)
GPU1 = Denoiser(name="gpu1_4090", host="gpu1", backend="trt", device=0,
                 tiling="none", enabled=True)


def test_local_encoder_adds_no_remote_encode_flag():
    argv, _ = build_command(IGPU, LOCAL_ENC, 0, staged="/s.MOV", out="/o.mkv",
                            remote_src=None, callback=None)
    assert "--remote-encode" not in argv
    assert "--lp" in argv and argv[argv.index("--lp") + 1] == "6"


def test_remote_encoder_carries_host_ip_port_and_root():
    argv, _ = build_command(IGPU, GPU2_ENC, 1, staged="/s.MOV", out="/o.mkv",
                            remote_src=None, callback=None)
    assert argv[argv.index("--remote-encode") + 1] == "gpu2"
    assert argv[argv.index("--remote-encode-ip") + 1] == "10.0.0.17"
    # slot 1 of the block that starts at 5310
    assert argv[argv.index("--remote-port") + 1] == "5311"
    assert (argv[argv.index("--remote-encode-root") + 1]
            == "/home/user/reposetc/archav1an")


def test_remote_denoise_to_remote_encode_leaves_the_callback_out():
    """One source of truth for the encoder's address.

    --remote-callback names THIS host, and this path never streams here, so
    dispatch does not read it: `_callback = remote_encode_ip if remote_encode`.
    Emitting it as well passed the same address twice, agreeing only because
    both sides happened to read encoder.stream_ip.
    """
    argv, _ = build_command(GPU1, GPU2_ENC, 0, staged="/s.MOV", out="/o.mkv",
                            remote_src="/mnt/media/x.MOV", callback="10.0.0.10")
    assert argv[argv.index("--remote-denoise") + 1] == "gpu1"
    # The denoise half streams to the ENCODER, and --remote-encode-ip says so.
    assert argv[argv.index("--remote-encode-ip") + 1] == "10.0.0.17"
    assert "--remote-callback" not in argv


def test_remote_denoise_to_local_encode_keeps_the_local_callback():
    argv, _ = build_command(GPU1, LOCAL_ENC, 0, staged="/s.MOV", out="/o.mkv",
                            remote_src="/mnt/media/x.MOV", callback="10.0.0.10")
    assert argv[argv.index("--remote-callback") + 1] == "10.0.0.10"


def test_lp_comes_from_the_chosen_encoder():
    slow = Encoder(name="a", host="local", slots=1, lp_level=2)
    argv, _ = build_command(IGPU, slow, 0, staged="/s.MOV", out="/o.mkv",
                            remote_src=None, callback=None)
    assert argv[argv.index("--lp") + 1] == "2"
