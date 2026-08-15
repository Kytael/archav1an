"""Turn one clip plus one denoiser into an svtav1-dispatch invocation.

Encoder settings are copied verbatim from run_linux_dance_HQ_crf27.sh:51-58.
Sigma and Opus are fixed fleet-wide so that output never depends on which
device took the clip (spec 5.1, 5.5(d)).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISPATCH = os.path.join(REPO, "tools", "svtav1-dispatch.py")
MANAGED_PYTHON = "/opt/archav1an/venv/bin/python"
MIGX_VENV = os.path.expanduser("~/reposetc/bsvd/migraphx-venv")
MIGX_LIBS = os.path.expanduser("~/reposetc/bsvd/migraphx-libs/lib")

BSVD_SIGMA = "0.05"
# "auto" lets the filter size each axis to the frame. A square tile cannot
# cover 16:9 evenly -- 576 does 1.28x the frame's pixels at 1080p, and a
# bigger square is worse, not better -- so auto picks 1096x976 there, 1.03x,
# and never returns a tile the card cannot hold.
#
# An explicit "HxW" or square size is passed through unchanged. That was
# unusable until engines were named for the shape that built them, because two
# tile sizes shared one cache file and each switch silently rebuilt it.
ENCODER_PARAMS = ("--tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --sharp-tx 1 "
                  "--sharpness 1 --tf-strength 2 --variance-boost-strength 1 "
                  "--variance-octile 7 --enable-dlf 2")


def build_command(denoiser, encode, staged, out, remote_src, callback):
    """Return (argv, env_overlay) for one clip on one denoiser."""
    env = {}
    if denoiser.backend == "migraphx":
        # BSVD's MIGraphX wheels stop at cp312, so this lane needs its own
        # interpreter and its own vspipe (see setup/denoiser.sh:562-570).
        python = os.path.join(MIGX_VENV, "bin", "python")
        env["VSPIPE"] = os.path.join(MIGX_VENV, "bin", "vspipe")
        env["LD_LIBRARY_PATH"] = MIGX_LIBS
    else:
        python = MANAGED_PYTHON

    argv = [python, DISPATCH,
            "-i", staged, "-o", out,
            "--quality", "27",
            "--photon-noise", "6",
            "--lp", str(encode.lp_level),
            "--speed", "4",
            "--denoise-bsvd",
            "--bsvd-sigma", BSVD_SIGMA,
            "--bsvd-device", str(denoiser.device),
            "--temp-tag", denoiser.name,
            "--encoder-params", ENCODER_PARAMS]

    if denoiser.tiling != "none":
        # Cards that cannot hold the full-frame BSVD state run tile-sequential
        # and windowed, so memory is one window rather than one clip (spec 5.5).
        argv += ["--bsvd-tile", str(denoiser.tiling),
                 "--bsvd-window", str(denoiser.window),
                 "--bsvd-margin", str(denoiser.margin)]

    if denoiser.is_remote:
        argv += ["--remote-denoise", denoiser.host]
        # dispatch defaults to ~/archav1an. Any host that keeps its checkout
        # elsewhere must say so, or the remote half cds into the wrong tree.
        if denoiser.root:
            argv += ["--remote-root", denoiser.root]
        # Omit the flag entirely when the remote has no copy of the archive:
        # dispatch then stages the clip itself. Passing None here put a None
        # into argv and subprocess rejected the whole command.
        if remote_src:
            argv += ["--remote-source", remote_src]
        argv += ["--remote-port", str(denoiser.port),
                 "--remote-callback", callback]
    return argv, env
