#!/usr/bin/env python3
"""
svtav1-dispatch.py — Single-pass SvtAv1EncApp encode with Opus mux and SSIMU2 measurement.

Pipes ffmpeg → SvtAv1EncApp, muxes Opus audio, then measures SSIMU2 scores
(mean + 15th percentile) for comparison against av1an pipeline output.
"""

import errno
import os
import re
import socket
import sys
import shlex
import subprocess
import shutil
import sysconfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tag as _tag
# Tile parsing lives with the tiling code so the split-host path and the gates
# read the flag the same way. Module-level imports there are stdlib only.
from bsvd_windowed import parse_tile_arg, tile_arg
from prefix_env import prefer_prefix_bin


# ---------------------------------------------------------------------------
# Denoise helpers
# ---------------------------------------------------------------------------

def vs_plugin_dirs():
    """Legacy plugin dirs to load explicitly, most authoritative first.

    VS R77 autoloads only site-packages/vapoursynth/plugins, and LoadPlugin
    raises on a namespace that is already claimed, so the first directory to
    supply a plugin wins.

    The prefix leads because setup.sh manages it and keeps the fleet in step:
    gpu1's pacman vszip is still 13.0, which lacks Dither and breaks vstools
    3.1.0. /usr/local is deliberately absent -- it was the default prefix
    before VS_PREFIX (f5b1614) and is now an unowned four-month-old copy that
    would shadow current pacman builds. That is the failure 6fad5c6 fixed.
    """
    prefix = os.environ.get("VS_PREFIX", "/opt/archav1an")
    return (f"{prefix}/lib/vapoursynth", "/usr/lib/vapoursynth")


def plugin_preamble(namespaces=("ffms2",)):
    """The LoadPlugin block every generated VPY needs, as source text.

    VS R77+ autoloads only site-packages/vapoursynth/plugins, so the legacy
    dirs that hold ffms2, vszip and vship must be loaded by hand. A plugin we
    do not use may fail to load -- encoder-host carries a libvsncnn.so with an
    undefined onnx symbol -- so a load error is fatal only when the namespace
    we need is the missing one, and then it says so HERE with the loader's
    message, rather than surfacing later as a bare AttributeError (or, on a
    split-host encode, as a 0 MB read blamed on the encoder host).
    """
    return (
        'import glob as _glob, os as _os\n'
        '_failed = []\n'
        f'for _d in {vs_plugin_dirs()!r}:\n'
        '    for _p in sorted(_glob.glob(_os.path.join(_d, "*.so"))):\n'
        '        try:\n'
        '            core.std.LoadPlugin(_p)\n'
        '        except vs.Error as _e:\n'
        '            if "already loaded" not in str(_e):\n'
        '                _failed.append(_p + ": " + str(_e))\n'
        f'for _ns in {tuple(namespaces)!r}:\n'
        '    if not hasattr(core, _ns):\n'
        '        raise RuntimeError("VapourSynth plugin " + _ns + " is missing. '
        'Plugin load errors: " + ("; ".join(_failed) or "none"))\n'
    )


def find_mlrt_plugin():
    """Return path to libvstrt.so (NVIDIA) or libvsmigx.so (AMD), or empty string."""
    _vs_prefix = os.environ.get("VS_PREFIX", "/opt/archav1an")
    for p in [
        f"{_vs_prefix}/lib/vapoursynth/libvstrt.so",
        "/usr/local/lib/vapoursynth/libvstrt.so",
        "/usr/lib/vapoursynth/libvstrt.so",
        f"{_vs_prefix}/lib/vapoursynth/libvsmigx.so",
        "/usr/local/lib/vapoursynth/libvsmigx.so",
        "/usr/lib/vapoursynth/libvsmigx.so",
    ]:
        if os.path.exists(p):
            return p
    return ""

def _mlrt_backend_lines(streams):
    """Return multi-line Python code that sets _backend to the best available vs-mlrt backend."""
    _vs_prefix = os.environ.get("VS_PREFIX", "/opt/archav1an")
    for p in [f"{_vs_prefix}/lib/vapoursynth/libvstrt.so",
              "/usr/local/lib/vapoursynth/libvstrt.so",
              "/usr/lib/vapoursynth/libvstrt.so"]:
        if os.path.exists(p):
            # vsmlrt.py calls trtexec with a minimal env dict (no inheritance).
            # On WSL2, /usr/lib/libcuda.so is a stub; the real driver is in /usr/lib/wsl/lib/.
            # Pass os.environ + WSL2 lib path so the subprocess can find the CUDA device.
            return (
                f'import os as _os\n'
                f'_trt_env = _os.environ.copy()\n'
                f'if _os.path.isdir("/usr/lib/wsl/lib"):\n'
                f'    _trt_env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib" + (":" + _trt_env.get("LD_LIBRARY_PATH", "") if _trt_env.get("LD_LIBRARY_PATH") else "")\n'
                f'_backend = _Backend.TRT(device_id=0, fp16=True, num_streams={streams}, use_cuda_graph=True, custom_env=_trt_env)'
            )
    return f'_backend = _Backend.MIGX(device_id=0, fp16=True, exhaustive_tune=False, num_streams={streams}, custom_env={{"MIGRAPHX_GPU_COMPILE_PARALLEL": "8"}})'

def write_denoise_vpy(vpy_path, source, cachefile, model_name, tile, streams,
                      use_smdegrain=False, tr=3, thsad=350,
                      use_rvrt=False, rvrt_sigma=12.0,
                      use_stasunet=False, stasunet_engine="", stasunet_pre_darken_ev=0.0,
                      use_bsvd=False, use_bsvd_smdegrain=False,
                      bsvd_onnx="", bsvd_sigma=0.08, bsvd_ep="TRT", bsvd_device=0,
                      bsvd_tile=0, bsvd_overlap=32, bsvd_window=0, bsvd_margin=32):
    source = os.path.abspath(source)
    backend_lines = _mlrt_backend_lines(streams)
    model_line = f'_model_enum = _SCUNetModel["scunet_{model_name}"]'
    # Set only by the windowed tile-sequential path, which reads its window
    # source from a second vspipe process instead of from its own graph.
    source_vpy_path = os.path.splitext(vpy_path)[0] + ".source.vpy"
    source_vpy_body = ""

    if use_bsvd or use_bsvd_smdegrain:
        bsvd_onnx = os.path.abspath(bsvd_onnx)
        _tools_dir = os.path.dirname(os.path.abspath(__file__))
        _head = (
            f'import sys as _sys; _sys.path.insert(0, r{_tools_dir!r})\n'
            f'_src_fmt = src.format\n'
            f'_rgb = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709", range_in_s="limited")\n'
        )
        if bsvd_tile:
            # Tile-sequential windowed path for cards that cannot hold the
            # full-frame state (spec 5.5). Memory is one window, not one clip.
            # The window's source frames come from a subprocess running the
            # companion script below: reading them from the selector would
            # deadlock the thread pool this graph is running on.
            denoise_lines = _head + (
                f'from bsvd_windowed import build_bsvd_windowed_tiled\n'
                f'_bsvd_rgb = build_bsvd_windowed_tiled(_rgb, onnx_path=r{bsvd_onnx!r}, '
                f'sigma={bsvd_sigma}, ep={bsvd_ep!r}, device_id={bsvd_device}, fp16=True, '
                f'tile={bsvd_tile!r}, overlap={bsvd_overlap}, window={bsvd_window}, '
                f'margin={bsvd_margin}, source_script=r{source_vpy_path!r})\n'
            )
            # Same _head, so the subprocess decodes the same clip by
            # construction. It only ever streams forward, so it needs a small
            # cache, not the warmup-sized one the denoise graph needs.
            #
            # Emitted as RGBH, not RGBS: the filter holds the window in the
            # engine's fp16 anyway, so sending fp32 doubles the pipe traffic
            # (24.9 MB a frame against 12.4) and costs a cast on arrival. The
            # RGBS line above is unchanged, so the pixels are the same ones
            # the denoise graph computes; this only rounds them once, where
            # the reader used to.
            source_vpy_body = _head + (
                'core.max_cache_size = 512\n'
                '_rgb16 = core.resize.Point(_rgb, format=vs.RGBH)\n'
                '_rgb16.set_output(0)\n'
            )
        else:
            denoise_lines = _head + (
                f'from bsvd_vs_filter import build_bsvd_streaming\n'
                f'_bsvd_rgb = build_bsvd_streaming(_rgb, onnx_path=r{bsvd_onnx!r}, '
                f'sigma={bsvd_sigma}, ep={bsvd_ep!r}, device_id={bsvd_device}, fp16=True)\n'
            )
        if use_bsvd_smdegrain:
            denoise_lines += (
                f'import havsfunc_legacy as _haf\n'
                f'_bsvd_yuv = core.resize.Bicubic(_bsvd_rgb, format=vs.YUV444P16, '
                f'matrix_s="709", range_s="limited")\n'
                f'_src444 = core.resize.Bicubic(src, format=vs.YUV444P16, range_s="limited")\n'
                f'src = _haf.SMDegrain(_src444, tr={tr}, thSAD={thsad}, plane=0, '
                f'prefilter=_bsvd_yuv, contrasharp=True, RefineMotion=True)\n'
                f'src = core.std.ShufflePlanes([src, _bsvd_yuv, _bsvd_yuv], '
                f'planes=[0, 1, 2], colorfamily=vs.YUV)\n'
                f'src = core.resize.Bicubic(src, format=_src_fmt)'
            )
        else:
            denoise_lines += (
                f'src = core.resize.Bicubic(_bsvd_rgb, format=_src_fmt, matrix_s="709", range_s="limited")'
            )
    elif use_stasunet:
        stasunet_engine = os.path.abspath(stasunet_engine)
        # STA-SUNet training normalizes BOTH input and GT to [-1,1] via (x-0.5)/0.5
        # (datasets/data_augment.py:49). This applies to custom and BVI-RLV engines alike.
        _norm_in = '# STA-SUNet: normalize [0,1] -> [-1,1] to match training.\n_rgb = core.std.Expr(_rgb, "x 2 * 1 -")\n'
        _norm_out = '# Denormalize [-1,1] -> [0,1] and clip.\n_den = core.std.Expr(_den, "x 1 + 2 / 0 max 1 min")\n'
        # STA-SUNet: 5-frame temporal denoiser via vs-mlrt vstrt plugin.
        # Engine expects rank-4 input [1, 15, 512, 512] (5 frames * 3 RGB chans, channel-concat).
        # Build 5 clips offset by [-2,-1,0,+1,+2] with edge padding, pass as list to trt.Model.
        # Requires initLibNvInferPlugins() for ModulatedDeformConv2d v2 plugin (not auto-loaded by vstrt).
        denoise_lines = (
            f'import ctypes as _ct\n'
            f'_plug = _ct.CDLL("/usr/lib/libnvinfer_plugin.so.10", mode=_ct.RTLD_GLOBAL)\n'
            f'_plug.initLibNvInferPlugins.argtypes = [_ct.c_void_p, _ct.c_char_p]\n'
            f'_plug.initLibNvInferPlugins.restype = _ct.c_bool\n'
            f'assert _plug.initLibNvInferPlugins(None, b""), "initLibNvInferPlugins failed"\n'
            f'_src_fmt = src.format\n'
            f'_rgb = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")\n'
            f'# Pre-darken to match training distribution: sRGB→linear, scale by alpha (2^stops), linear→sRGB.\n'
            f'# Mirrors prepare_dataset.py: alpha=0.10 (~-3.32 EV, _10 variant), alpha=0.05 (~-4.32 EV, _20).\n'
            f'_alpha = {2.0 ** stasunet_pre_darken_ev}\n'
            f'if _alpha != 1.0:\n'
            f'    _rgb = core.std.Expr(_rgb, "x 0.04045 <= x 12.92 / x 0.055 + 1.055 / 2.4 pow ?")\n'
            f'    _rgb = core.std.Expr(_rgb, f"x {{_alpha}} *")\n'
            f'    _rgb = core.std.Expr(_rgb, "x 0.0031308 <= x 12.92 * 1.055 x 0.4166666667 pow * 0.055 - ?")\n'
            f'{_norm_in}'
            f'_nf = _rgb.num_frames\n'
            f'_m2 = _rgb.std.DuplicateFrames([0, 0]).std.Trim(first=0, last=_nf - 1)\n'
            f'_m1 = _rgb.std.DuplicateFrames([0]).std.Trim(first=0, last=_nf - 1)\n'
            f'_p0 = _rgb\n'
            f'_p1 = _rgb.std.Trim(first=1).std.DuplicateFrames([_nf - 2])\n'
            f'_p2 = _rgb.std.Trim(first=2).std.DuplicateFrames([_nf - 3, _nf - 3])\n'
            f'_den = core.trt.Model([_m2, _m1, _p0, _p1, _p2], engine_path=r{stasunet_engine!r}, '
            f'overlap=[64, 64], tilesize=[{tile}, {tile}], '
            f'num_streams={streams}, use_cuda_graph=True, device_id=0)\n'
            f'{_norm_out}'
            f'src = core.resize.Bicubic(_den, format=_src_fmt, matrix_s="709")'
        )
    elif use_rvrt:
        denoise_lines = (
            f'import vsrvrt as _vsrvrt\n'
            f'_src_fmt = src.format\n'
            f'_rgb = core.resize.Bicubic(src, format=vs.RGB24, matrix_in_s="709")\n'
            f'_rgb = _vsrvrt.Denoise(_rgb, sigma={rvrt_sigma}, tile_size=(16, {tile}, {tile}), '
            f'tile_overlap=(2, 20, 20), use_fp16=True)\n'
            f'src = core.resize.Bicubic(_rgb, format=_src_fmt, matrix_s="709")'
        )
    elif use_smdegrain:
        denoise_lines = (
            f'{model_line}\n'
            f'{backend_lines}\n'
            f'import havsfunc_legacy as _haf\n'
            f'_src_fmt = src.format\n'
            f'_scunet_pre = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")\n'
            f'_scunet_pre = _SCUNet(_scunet_pre, model=_model_enum, tilesize={tile}, overlap=8, backend=_backend)\n'
            f'_scunet_pre = core.resize.Bicubic(_scunet_pre, format=vs.YUV444P16, matrix_s="709", range_s="limited")\n'
            f'_src444 = core.resize.Bicubic(src, format=vs.YUV444P16, range_s="limited")\n'
            f'src = _haf.SMDegrain(_src444, tr={tr}, thSAD={thsad}, plane=0, prefilter=_scunet_pre, contrasharp=True, RefineMotion=True)\n'
            f'src = core.std.ShufflePlanes([src, _scunet_pre, _scunet_pre], planes=[0, 1, 2], colorfamily=vs.YUV)\n'
            f'src = core.resize.Bicubic(src, format=_src_fmt)'
        )
    elif model_name.startswith("gray_"):
        denoise_lines = (
            f'{model_line}\n'
            f'{backend_lines}\n'
            f'_luma = core.std.ShufflePlanes(src, planes=0, colorfamily=vs.GRAY)\n'
            f'_luma_f = core.resize.Bicubic(_luma, format=vs.GRAYS)\n'
            f'_luma_d = _SCUNet(_luma_f, model=_model_enum, tilesize={tile}, overlap=8, backend=_backend)\n'
            f'_luma_out = core.resize.Bicubic(_luma_d, format=_luma.format)\n'
            f'src = core.std.ShufflePlanes([_luma_out, src, src], planes=[0, 1, 2], colorfamily=vs.YUV)'
        )
    else:
        denoise_lines = (
            f'{model_line}\n'
            f'{backend_lines}\n'
            f'_src_fmt = src.format\n'
            f'_rgb = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")\n'
            f'_rgb = _SCUNet(_rgb, model=_model_enum, tilesize={tile}, overlap=8, backend=_backend)\n'
            f'src = core.resize.Bicubic(_rgb, format=_src_fmt, matrix_s="709")'
        )
    venv_site_pkgs = sysconfig.get_path('purelib')
    vsmlrt_import = '' if (use_rvrt or use_stasunet or use_bsvd or use_bsvd_smdegrain) else 'from vsmlrt import SCUNet as _SCUNet, SCUNetModel as _SCUNetModel, Backend as _Backend\n'
    # BSVD's mirror-pad warmup materializes ~2*shift_num frames to emit frame 0;
    # with a 10-bit source the 16-bit decode + YUV444P16 intermediates exceed a
    # 1024MB cache and VS deadlocks in its "flushing pipeline" throttle (0 frames
    # out, GPU idle). 4096 fits the warmup for 8/10-bit — confirmed on encoder-host
    # MIGraphX: 1024 hangs, 4096 encodes at full speed. (SMDegrain also needs 4096.)
    _cache_mb = 4096
    prologue = (
        f'import sys as _sys; _sys.path.insert(0, {venv_site_pkgs!r})\n'
        f'from vstools import vs, core, initialize_clip, finalize_clip\n'
        f'core.max_cache_size = {_cache_mb}\n'
        f'# VS R77+ autoloads only site-packages/vapoursynth/plugins; the legacy dirs\n# (ffms2/vszip/vship live there) must be loaded explicitly.\nimport glob as _glob, os as _os\n_failed = []\nfor _d in {vs_plugin_dirs()!r}:\n    for _p in sorted(_glob.glob(_os.path.join(_d, "*.so"))):\n        try:\n            core.std.LoadPlugin(_p)\n        except vs.Error as _e:\n            if "already loaded" not in str(_e):\n                _failed.append(_p + ": " + str(_e))\n# A plugin we do not use may fail to load -- encoder-host carries a libvsncnn.so\n# with an undefined onnx symbol -- so a load error is only fatal when the\n# namespace we need is the missing one. Then say so HERE, with the loader\n# message included, rather than leaving an AttributeError to surface further down\n# (or, on a split-host encode, as a 0 MB read blamed on the encoder host).\nfor _ns in ("ffms2",):\n    if not hasattr(core, _ns):\n        raise RuntimeError("VapourSynth plugin " + _ns + " is missing. Plugin load errors: " + ("; ".join(_failed) or "none"))\n\n'
        f'\n'
        f'src = core.ffms2.Source(source=r{source!r}, cachefile=r{cachefile!r})\n'
        f'if src.format.color_family == vs.RGB:\n'
        f'    # RGB sources (Lagarith/FFV1 eval intermediates): normalize once to\n'
        f'    # YUV420P10 so every denoise builder below can assume YUV input and\n'
        f'    # the y4m pipe to SvtAv1EncApp stays 4:2:0.\n'
        f'    src = core.resize.Bicubic(src, format=vs.YUV420P10, matrix_s="709", chromaloc_s="left")\n'
        f'src = initialize_clip(src)\n'
    )
    vpy = (
        prologue +
        f'\n'
        f'{vsmlrt_import}'
        f'{denoise_lines}\n'
        f'\n'
        f'# SVT-AV1 requires 4:2:0; denoise builders round-trip back to source chroma\n'
        f'# (may be 4:2:2/4:4:4), so force 4:2:0 for the encoder like the base path does.\n'
        f'if (src.format.subsampling_w, src.format.subsampling_h) != (1, 1):\n'
        f'    src = core.resize.Bicubic(src, format=src.format.replace(subsampling_w=1, subsampling_h=1), chromaloc_s="left")\n'
        f'\n'
        f'final = finalize_clip(src)\n'
        f'final.set_output(0)\n'
    )
    with open(vpy_path, "w") as f:
        f.write(vpy)
    if source_vpy_body:
        with open(source_vpy_path, "w") as f:
            f.write(prologue + "\n" + source_vpy_body)


# ---------------------------------------------------------------------------
# Encode helpers
# ---------------------------------------------------------------------------

# The frame counter vspipe -p and netstream --progress emit. Kept identical to
# archive-batch.py:78 and encode_dash/liverate.py: three readers of one log
# format disagreeing about what a counter looks like is a silent bug waiting.
_PROGRESS = re.compile(r"^Frame:\s*\d+")


def _print_log_tail(log_path, label, max_lines=40):
    """Print the last max_lines of a captured stderr log, for post-mortem.

    Progress lines are dropped first. vspipe -p and netstream --progress emit
    one about once a second, and text-mode readlines() splits on the carriage
    return they use, so an unfiltered tail of a netstream log is 39 counters
    and one summary -- the listening and accepted lines that say whether the
    remote ever connected get pushed out of view.
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            tail = [ln for ln in f.readlines()
                    if not _PROGRESS.match(ln.strip())][-max_lines:]
    except OSError:
        return
    if not tail:
        return
    print(f"[svtav1-dispatch] --- last {len(tail)} line(s) of {label} ---")
    for line in tail:
        print("    " + line.rstrip("\n"))
    print(f"[svtav1-dispatch] --- end {label} ---")


def resolve_bsvd_sigma(sigma_arg, input_file, optsig_model):
    """--bsvd-sigma as a float, running the auto pre-pass when asked."""
    if str(sigma_arg).lower() != "auto":
        return float(sigma_arg)
    if not os.path.exists(optsig_model):
        print(f"[svtav1-dispatch] Error: --bsvd-sigma=auto needs {optsig_model}; "
              "pass --bsvd-sigma <float>.")
        sys.exit(2)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bsvd_optsig import compute_sigma_for_video
        return compute_sigma_for_video(input_file, model_json=optsig_model)
    except Exception as e:
        print(f"[svtav1-dispatch] Error: --bsvd-sigma=auto pre-pass failed ({e}); "
              "pass --bsvd-sigma <float>.")
        sys.exit(2)


def callback_address(ssh_target):
    """The local IP the remote denoiser should stream back to.

    Resolves the ssh target the way ssh itself would (it is usually a
    ~/.ssh/config alias, not a DNS name) and asks the routing table which
    source address reaches it.
    """
    resolved = subprocess.run(["ssh", "-G", ssh_target], capture_output=True,
                              text=True).stdout
    host = next((l.split()[1] for l in resolved.splitlines()
                 if l.startswith("hostname ")), ssh_target)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 9))
        return s.getsockname()[0]
    finally:
        s.close()


# A leading ~ or ~user, and nothing else. Anything outside this is quoted
# whole: a root that is not a plain tilde path has no business reaching the
# remote shell unquoted.
_TILDE_HEAD = re.compile(r"^~[\w.-]*$")


def quote_remote_path(path):
    """shlex.quote, except a leading ~ or ~user stays OUTSIDE the quotes.

    Every remote path here is expanded by the remote shell, and a shell only
    expands a tilde that is unquoted at the start of a word. shlex.quote wraps
    the whole thing, so `cd '~/archav1an'` looks for a directory literally
    named "~" -- which is what dispatch's own default remote root did, on
    every host that did not set an absolute --remote-root. The remote half
    then ran in the wrong tree, or not at all.

    The rest of the path is still quoted, so a space or a metacharacter in a
    checkout path cannot reach the remote shell.
    """
    head, sep, tail = path.partition("/")
    if not _TILDE_HEAD.match(head):
        return shlex.quote(path)
    if not sep:
        return head
    return f"{head}/{shlex.quote(tail)}" if tail else f"{head}/"


def build_remote_shell_cmd(remote_root, inner):
    """The one-line bash a remote half runs: cd into the checkout, then work.

    Deliberately does NOT set PATH. main() calls prefer_prefix_bin() as its
    first statement, which prepends $VS_PREFIX/bin AND appends $VS_PREFIX/lib
    together -- and those two must move together or not at all. Setting PATH
    here as well would hardcode /opt/archav1an while prefer_prefix_bin honours
    $VS_PREFIX, so a host with a different prefix would run our binaries
    against the other prefix's libraries. That is the mismatch that made
    ffmpeg write a complete output file and then exit 127 on 2026-08-12, and
    it read as a bad-clip problem for two runs.

    Verified on gpu4, whose non-interactive ssh PATH omits the prefix:
    shutil.which finds nothing before prefer_prefix_bin() and both binaries
    after it, in this exact command shape.

    quote_remote_path leaves a leading tilde outside the quotes, because a
    shell only expands one that is unquoted at the start of a word.

    PYTHONUNBUFFERED so the remote's diagnostics reach the log even when it is
    killed: over a pipe its stdout would otherwise be block-buffered.
    """
    return (f"cd {quote_remote_path(remote_root)} && "
            f"PYTHONUNBUFFERED=1 exec {inner}")


def remote_encode_ivf(stem, temp_tag):
    """Where the encode half writes its IVF, relative to its own checkout.

    Not a path this side gets to choose. main() derives it from the -o
    basename and ignores the directory it was given, so an IVF asked for at
    Temp/_encode/<stem>.ivf lands at Temp/<stem>/<stem>.ivf and the rsync then
    fetches a path nothing ever wrote. One definition, used by both
    orchestrators and by the -o they send, so the two cannot drift.

    --temp-tag carries over for the reason it exists locally: 185 stems repeat
    across the dance archive, and an encode host runs several slots at once,
    so without it two clips of one name share a temp dir and the loser has its
    IVF overwritten mid-encode.
    """
    return (f"Temp/{temp_tag}/{stem}/{stem}.ivf" if temp_tag
            else f"Temp/{stem}/{stem}.ivf")


def remove_remote_ivf(ssh_target, remote_root, remote_ivf):
    """Delete the encode host's IVF once it is no longer wanted.

    Best effort. A leftover IVF is about 77 MB and harmless; a failure to
    delete it must not fail a clip that already encoded correctly.

    A failed clip leaves a short one behind too, and archive-batch retries it,
    so the failure paths have to call this as well: bounded by lanes x stems x
    attempts, the dead IVFs otherwise fill an encode host's disk.
    """
    subprocess.call(["ssh", ssh_target,
                     f"rm -f {quote_remote_path(remote_root)}/{remote_ivf}"])


def remove_staged_source(ssh_target, remote_root, remote_src, was_staged):
    """Delete a source this host copied to the denoise machine.

    `was_staged` is the whole safety of this function and is never optional.
    A --remote-source path is the operator's archive, read in place, and
    deleting one destroys an original. Only the copy under Temp/_remote that
    resolve_remote_src invented may go, so the caller passes the same flag it
    got from there rather than this deciding for itself.

    Best effort, like remove_remote_ivf: a clip that encoded correctly must not
    fail because a cleanup did. A retry re-stages, which costs one transfer and
    is the intended trade -- nothing cleaned these up before, and the leftovers
    reached 3.3 GB on gpu2 and 775 MB on gpu4 over 69 gate clips.
    """
    if not was_staged:
        return
    subprocess.call(["ssh", ssh_target,
                     f"rm -f {quote_remote_path(remote_root)}/{remote_src}"])


def build_encode_serve_cmd(remote_python, remote_root, bind, ivf, forward_args):
    """The one-line bash the encode host runs.

    Split out from run_remote_encode so the argument shape is testable without
    an ssh. The cd, the tilde handling and PYTHONUNBUFFERED come from
    build_remote_shell_cmd, which the denoise half uses too -- one definition,
    so the two halves cannot drift. Note it deliberately sets no PATH; see its
    docstring.
    """
    inner = " ".join(shlex.quote(a) for a in [
        remote_python, "tools/svtav1-dispatch.py",
        "--encode-serve", bind, "-o", ivf, *forward_args])
    return build_remote_shell_cmd(remote_root, inner)


def build_encode_forward(quality, speed, lp, photon_noise, encoder_params,
                         temp_tag):
    """What the remote half needs to rebuild an identical encoder command.

    Rebuilt from the parsed flags rather than from svt_cmd, which names this
    host's IVF path and this host's binary.

    --lp travels with them because it costs memory, not quality: every level
    produced a bit-identical bitstream, but a slot is 4.8 GB at level 6
    against 2.1 GB at level 4 (docs/encode-capacity.md). Left out, the remote
    would fall back to dispatch's own default of 16, which SVT clamps to 6, so
    a roster asking for 4 would quietly cost that host 2.3x the memory.
    """
    # Guarded the same way svt_params is built below. Unguarded, an unset
    # flag travelled as the literal "None", which the encode half finds
    # truthy and turns into `--crf None`.
    out = []
    if quality:
        out += ["--quality", str(quality)]
    if speed:
        out += ["--speed", str(speed)]
    out += ["--lp", str(lp)]
    if photon_noise and photon_noise != "0":
        out += ["--photon-noise", photon_noise]
    if encoder_params:
        out += ["--encoder-params", encoder_params]
    if temp_tag:
        out += ["--temp-tag", temp_tag]
    return out


def resolve_remote_src(remote_source, input_file):
    """Where the remote half reads the source, and whether we must stage it.

    With --remote-source the file already lives on the denoise host, so nothing
    is copied: this is what keeps 2.5 TB of archive out of the remote's root.
    """
    if remote_source:
        return remote_source, False
    # Relative to remote_root: the remote command runs after `cd`, and quoting
    # a leading ~ would stop the remote shell expanding it.
    return f"Temp/_remote/{os.path.basename(input_file)}", True


def run_remote_denoise(ssh_target, remote_root, remote_python, callback, port,
                       input_file, forward_args, sink_cmd, temp_dir, stem,
                       remote_source=None):
    """Denoise on ssh_target, stream the y4m back over TCP, encode into sink_cmd.

    ssh carries only control (launch, stderr, exit status): a single ssh stream
    caps at ~1.3 Gbps between encoder-host and gpu1 even on the 10G LAN, while a
    plain socket sustains 5+ Gbps. The encoder side listens so that the listener
    dies with the process that owns the encoder.

    A remote that dies mid-stream closes the socket cleanly, so the local
    encoder finalizes a short IVF and exits 0 -- run_piped cannot see it. The
    ssh exit-status check below is advisory only (a nonzero status also
    results from the remote's teardown outlasting its wait); the definitive
    guard is the frame-count verification before the mux.
    """
    remote_src, needs_staging = resolve_remote_src(remote_source, input_file)
    if needs_staging:
        remote_dir = f"{remote_root}/Temp/_remote"
        # Both of these are read by a shell on the remote: --rsync-path is run
        # there verbatim, and rsync hands the destination to the remote shell
        # too. So both need the tilde left expandable.
        quoted_dir = quote_remote_path(remote_dir)
        print(f"[svtav1-dispatch] staging source -> {ssh_target}:{remote_dir}/")
        subprocess.check_call(["rsync", "-a", "--rsync-path",
                               f"mkdir -p {quoted_dir} && rsync",
                               os.path.abspath(input_file),
                               f"{ssh_target}:{quoted_dir}/"])
    else:
        print(f"[svtav1-dispatch] remote source in place: {ssh_target}:{remote_src}")
    # An absolute --remote-source path is unaffected by the `cd {remote_root}` below.

    remote_cmd = " ".join(shlex.quote(a) for a in [
        remote_python, "tools/svtav1-dispatch.py",
        "--denoise-serve", f"{callback}:{port}", "-i", remote_src, *forward_args])
    remote_log = os.path.join(temp_dir, f"{stem}_remote.log")
    print(f"[svtav1-dispatch] {ssh_target}: vspipe (BSVD) | netstream -> "
          f"{callback}:{port} | SvtAv1EncApp (local)")
    sys.stdout.flush()
    with open(remote_log, "w", encoding="utf-8") as log_fh:
        # bash -s over stdin: the remote login shell is fish, which mangles
        # quoting in `ssh host bash -c '...'`.
        ssh_proc = subprocess.Popen(["ssh", ssh_target, "bash", "-s"],
                                    stdin=subprocess.PIPE,
                                    stdout=log_fh, stderr=log_fh)
        ssh_proc.stdin.write(
            (build_remote_shell_cmd(remote_root, remote_cmd) + "\n").encode())
        ssh_proc.stdin.close()
        local_failed = False
        try:
            run_piped([sys.executable,
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "netstream.py"),
                       "recv", "--port", str(port),
                       # The remote's vspipe log stays on the remote host, so
                       # this socket is where a remote lane gets counted.
                       "--progress"],
                      sink_cmd, source_label="netstream recv",
                      sink_label="SvtAv1EncApp",
                      source_stderr_log=os.path.join(temp_dir, f"{stem}_netstream.log"))
        except SystemExit:
            # A local failure is usually the remote's fault (it never connected,
            # or died mid-stream), so surface its log too -- but only once the
            # remote has exited and finished writing it.
            local_failed = True
            raise
        finally:
            # The remote is still tearing down its VS core and TRT session when
            # the last frame lands, so wait it out; only kill one that hangs.
            try:
                ssh_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                ssh_proc.terminate()
                ssh_proc.wait()
            if local_failed:
                _print_log_tail(remote_log, f"{ssh_target} stderr")
            # In the finally, so a failed clip cleans up too: archive-batch
            # retries it, and a retry that lands on another host would leave
            # this copy behind for the rest of the run.
            remove_staged_source(ssh_target, remote_root, remote_src,
                                 needs_staging)
    if ssh_proc.returncode != 0:
        # The remote tears down its VS core and TRT session after the last
        # frame lands, and that teardown can outlast the 30 s wait above -- a
        # kill here exits nonzero even though every frame arrived and the
        # local encoder finalized a complete IVF. This point is only reachable
        # on the success path (a local failure re-raises inside the block), so
        # a nonzero status is not proof of truncation. Warn instead of
        # aborting: the frame-count verification before the mux stays the
        # definitive guard and still stops a genuinely short encode.
        print(f"[svtav1-dispatch] Warning: remote denoise on {ssh_target} "
              f"exited with {ssh_proc.returncode} during teardown; relying "
              f"on the frame-count check before the mux.")
        _print_log_tail(remote_log, f"{ssh_target} stderr")


def run_remote_encode(ssh_target, remote_root, remote_python, bind, ivf_path,
                      forward_args, source_cmd, temp_dir, stem, temp_tag=None):
    """Launch the encoder on ssh_target, feed it, and bring the IVF back.

    The encoder listens and the source connects out, which is the same rule as
    the denoise split (docs/split-host-denoise.md): the listener is owned by
    the process that owns the encoder, and a failed run leaves nothing behind
    on the other host. The only thing that changed is which host that is.

    `source_cmd` is whatever produces y4m here: a local vspipe in case 3, or
    None in case 4, where the denoise host streams straight to the encoder and
    this process only waits on the two ssh sessions.
    """
    remote_ivf = remote_encode_ivf(stem, temp_tag)
    cmd = build_encode_serve_cmd(remote_python, remote_root, bind, remote_ivf,
                                 forward_args)
    remote_log = os.path.join(temp_dir, f"{stem}_encode.log")
    print(f"[svtav1-dispatch] {ssh_target}: netstream recv on {bind} | "
          f"SvtAv1EncApp")
    sys.stdout.flush()
    with open(remote_log, "w", encoding="utf-8") as log_fh:
        # bash -s over stdin, as run_remote_denoise does: a remote login shell
        # may be fish, which mangles `ssh host bash -c '...'` silently.
        ssh_proc = subprocess.Popen(["ssh", ssh_target, "bash", "-s"],
                                    stdin=subprocess.PIPE,
                                    stdout=log_fh, stderr=log_fh)
        # cmd carries its own cd via build_remote_shell_cmd, and needs no
        # mkdir before it: the remote's main() creates its own temp dir.
        ssh_proc.stdin.write((cmd + "\n").encode())
        ssh_proc.stdin.close()
        failed = False
        rc = None
        try:
            if source_cmd is not None:
                host, _, port = bind.rpartition(":")
                run_piped(source_cmd,
                          [sys.executable,
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "netstream.py"),
                           "send", "--host", host, "--port", port],
                          source_label="vspipe",
                          sink_label=f"netstream -> {bind}",
                          source_stderr_log=os.path.join(
                              temp_dir, f"{stem}_vspipe.log"))
            rc = ssh_proc.wait(timeout=1800)
        except SystemExit:
            failed = True
            raise
        except subprocess.TimeoutExpired:
            ssh_proc.terminate()
            ssh_proc.wait()
            failed = True
            print(f"[svtav1-dispatch] Error: encode on {ssh_target} did not "
                  f"finish within 1800s of the last frame; killed.")
            sys.exit(1)
        finally:
            if failed:
                # The remote is still writing this log through the fd the
                # `with` is about to close, so the tail is only readable once
                # it has exited -- and an ssh nobody waits on outlives this
                # process with the remote encoder still attached to it.
                try:
                    ssh_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    ssh_proc.terminate()
                    ssh_proc.wait()
                _print_log_tail(remote_log, f"{ssh_target} encode stderr")
                remove_remote_ivf(ssh_target, remote_root, remote_ivf)
    if rc != 0:
        # Unlike the denoise side, a nonzero status here is definitive. There
        # the remote tears down a VS core and a TRT session after the last
        # frame and can outlast the wait, so a kill exits nonzero on a healthy
        # run. This process owns the encoder outright, so there is no teardown
        # race to forgive.
        print(f"[svtav1-dispatch] Error: encode on {ssh_target} exited {rc}.")
        _print_log_tail(remote_log, f"{ssh_target} encode stderr")
        remove_remote_ivf(ssh_target, remote_root, remote_ivf)
        sys.exit(1)
    print(f"[svtav1-dispatch] fetching IVF from {ssh_target}")
    sys.stdout.flush()
    subprocess.check_call(["rsync", "-a",
                           f"{ssh_target}:{quote_remote_path(remote_root)}/"
                           f"{remote_ivf}", ivf_path])
    remove_remote_ivf(ssh_target, remote_root, remote_ivf)


# The denoise half of case 4 streams for the whole clip, so this cannot be a
# short grace like run_remote_denoise's 30 s teardown wait. It is a backstop
# against a half that never makes progress at all -- a TRT session that hangs
# on init, or an ssh whose peer died with no keepalive to notice. Without it
# the orchestrator blocks until archive-batch's own dispatch_timeout expires
# (an hour plus a second per frame, and 24 h when the frame count is unknown),
# holding the encoder slot for that whole budget. BSVD runs at 4-21 fps across
# the fleet, so six hours is far longer than any clip in the archive needs.
SPLIT_DENOISE_TIMEOUT = 21600
SPLIT_ENCODE_TIMEOUT = 1800


def run_remote_encode_split(denoise_target, denoise_root, encode_target,
                            encode_root, remote_python, bind, ivf_path,
                            input_file, denoise_forward, encode_forward,
                            temp_dir, stem, remote_source=None, temp_tag=None):
    """Case 4: denoise on one host, encode on a second, orchestrate from a third.

    Both halves block, so neither can be driven by run_remote_denoise or
    run_remote_encode alone -- this function owns both ssh sessions and waits
    on them together. The encoder is launched first because it listens.
    """
    remote_ivf = remote_encode_ivf(stem, temp_tag)
    enc_cmd = build_encode_serve_cmd(remote_python, encode_root, bind,
                                     remote_ivf, encode_forward)
    enc_log = os.path.join(temp_dir, f"{stem}_encode.log")
    dn_log = os.path.join(temp_dir, f"{stem}_remote.log")
    remote_src, needs_staging = resolve_remote_src(remote_source, input_file)
    if needs_staging:
        quoted = quote_remote_path(f"{denoise_root}/Temp/_remote")
        print(f"[svtav1-dispatch] staging source -> {denoise_target}:"
              f"{denoise_root}/Temp/_remote/")
        subprocess.check_call(["rsync", "-a", "--rsync-path",
                               f"mkdir -p {quoted} && rsync",
                               os.path.abspath(input_file),
                               f"{denoise_target}:{quoted}/"])
    dn_cmd = " ".join(shlex.quote(a) for a in [
        remote_python, "tools/svtav1-dispatch.py",
        "--denoise-serve", bind, "-i", remote_src, *denoise_forward])
    print(f"[svtav1-dispatch] {denoise_target}: vspipe (BSVD) --{bind}--> "
          f"{encode_target}: SvtAv1EncApp")
    sys.stdout.flush()
    with open(enc_log, "w", encoding="utf-8") as efh, \
            open(dn_log, "w", encoding="utf-8") as dfh:
        enc = subprocess.Popen(["ssh", encode_target, "bash", "-s"],
                               stdin=subprocess.PIPE, stdout=efh, stderr=efh)
        enc.stdin.write((enc_cmd + "\n").encode())
        enc.stdin.close()
        dn = subprocess.Popen(["ssh", denoise_target, "bash", "-s"],
                              stdin=subprocess.PIPE, stdout=dfh, stderr=dfh)
        dn.stdin.write(
            (build_remote_shell_cmd(denoise_root, dn_cmd) + "\n").encode())
        dn.stdin.close()
        dn_timed_out = False
        enc_timed_out = False
        try:
            dn_rc = dn.wait(timeout=SPLIT_DENOISE_TIMEOUT)
        except subprocess.TimeoutExpired:
            dn.terminate()
            dn_rc = dn.wait()
            dn_timed_out = True
        # enc is waited on second either way: it only sees end-of-stream once
        # dn has closed the socket, whether dn finished or was killed above.
        try:
            enc_rc = enc.wait(timeout=SPLIT_ENCODE_TIMEOUT)
        except subprocess.TimeoutExpired:
            enc.terminate()
            enc_rc = enc.wait()
            enc_timed_out = True
    # Both halves have exited, so nothing is reading the copy any more. Placed
    # above the failure branch because that branch exits, and a failed clip has
    # to clean up too -- archive-batch retries it, possibly on another host.
    remove_staged_source(denoise_target, denoise_root, remote_src,
                         needs_staging)
    failed = dn_timed_out or enc_timed_out or enc_rc != 0
    if dn_timed_out:
        print(f"[svtav1-dispatch] Error: denoise on {denoise_target} did not "
              f"finish within {SPLIT_DENOISE_TIMEOUT}s; killed.")
    elif dn_rc != 0:
        # Advisory, exactly as in run_remote_denoise: the denoise host tears
        # down a VS core and a TRT session after the last frame and can exit
        # nonzero on a run where every frame arrived. The frame-count check
        # before the mux stays the definitive guard.
        print(f"[svtav1-dispatch] Warning: denoise on {denoise_target} exited "
              f"{dn_rc} during teardown.")
        if not failed:
            _print_log_tail(dn_log, f"{denoise_target} stderr")
    if enc_timed_out:
        # Not "exited -1": that reads as a crash and sends the reader looking
        # for one. run_remote_encode names the same condition the same way.
        print(f"[svtav1-dispatch] Error: encode on {encode_target} did not "
              f"finish within {SPLIT_ENCODE_TIMEOUT}s of the last frame; killed.")
    elif enc_rc != 0:
        print(f"[svtav1-dispatch] Error: encode on {encode_target} exited {enc_rc}.")
    if failed:
        _print_log_tail(enc_log, f"{encode_target} encode stderr")
        _print_log_tail(dn_log, f"{denoise_target} stderr")
        remove_remote_ivf(encode_target, encode_root, remote_ivf)
        sys.exit(1)
    print(f"[svtav1-dispatch] fetching IVF from {encode_target}")
    sys.stdout.flush()
    subprocess.check_call(["rsync", "-a",
                           f"{encode_target}:{quote_remote_path(encode_root)}/"
                           f"{remote_ivf}", ivf_path])
    remove_remote_ivf(encode_target, encode_root, remote_ivf)


def run_encode_serve(bind, svt_cmd, ivf_path, temp_dir, stem):
    """Receive y4m on `bind` and encode it. The other end of --remote-encode.

    The mirror of --denoise-serve: the job ends with the last frame, and the
    frame-count check, the mux and the tag all live on the host that holds the
    source. That host already stages every clip, so nothing is copied here.

    The address is bound explicitly and never 0.0.0.0. gpu2 is a laptop: off
    the home LAN it does not hold its roster address, the bind fails, and the
    lane refuses to start instead of listening on an untrusted network. That
    is the half of the access control that does not depend on a firewall rule
    being right.
    """
    host, _, port = bind.rpartition(":")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, int(port)))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            # The likeliest failure here, and nothing to do with the LAN: a
            # killed or timed-out clip leaves a netstream recv holding the
            # port, the scheduler releases the slot, and the next clip draws
            # the same slot index and so the same port.
            print(f"[svtav1-dispatch] Error: cannot bind {host}:{port} ({exc}). "
                  f"That port is already in use on this host, most likely by a "
                  f"netstream recv left behind by an earlier clip on this slot.")
        else:
            print(f"[svtav1-dispatch] Error: cannot bind {host}:{port} ({exc}). "
                  f"This host does not hold that address, so it will not listen "
                  f"on any other one.")
        sys.exit(1)
    finally:
        probe.close()
    print(f"[svtav1-dispatch] encode-serve: netstream recv on {bind} | "
          f"SvtAv1EncApp -> {ivf_path}")
    sys.stdout.flush()
    run_piped([sys.executable,
               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "netstream.py"),
               "recv", "--port", str(port), "--bind", host, "--progress"],
              svt_cmd, source_label="netstream recv",
              sink_label="SvtAv1EncApp",
              source_stderr_log=os.path.join(temp_dir, f"{stem}_netstream.log"))


def run_piped(source_cmd, sink_cmd, source_label="source",
              sink_label="SvtAv1EncApp",
              source_stderr_log=None, suppress_sink_stderr=False):
    """Run source_cmd | sink_cmd, forwarding Ctrl+C to both processes.

    A nonzero exit from the *source* is fatal, not a warning: when vspipe dies
    mid-stream (denoiser OOM, TRT/import failure) the sink sees a clean EOF at a
    frame boundary, finalizes a truncated output and exits 0 -- so without this
    check a short encode ships as success. When source_stderr_log is given,
    vspipe's stderr is captured there (keeping the console clean) and its tail is
    printed on failure so the root cause is diagnosable.
    """
    log_fh = open(source_stderr_log, "w", encoding="utf-8") if source_stderr_log else None
    try:
        source_proc = subprocess.Popen(source_cmd, stdout=subprocess.PIPE,
                                       stderr=log_fh)
        sink_proc   = subprocess.Popen(sink_cmd,   stdin=source_proc.stdout,
                                       stderr=subprocess.DEVNULL if suppress_sink_stderr else None)
        source_proc.stdout.close()
        try:
            sink_proc.wait()
            source_proc.wait()
        except KeyboardInterrupt:
            source_proc.terminate(); sink_proc.terminate()
            source_proc.wait();      sink_proc.wait()
            sys.exit(130)
    finally:
        if log_fh:
            log_fh.close()
    if sink_proc.returncode != 0:
        print(f"[svtav1-dispatch] Error: {sink_label} exited with {sink_proc.returncode}")
        if source_stderr_log:
            _print_log_tail(source_stderr_log, f"{source_label} stderr")
        sys.exit(sink_proc.returncode)
    if source_proc.returncode not in (0, None):
        print(f"[svtav1-dispatch] Error: {source_label} exited with {source_proc.returncode}; "
              f"output is truncated -- aborting before mux.")
        if source_stderr_log:
            _print_log_tail(source_stderr_log, f"{source_label} stderr")
        sys.exit(source_proc.returncode or 1)


# ---------------------------------------------------------------------------
# Audio helpers (shared with av1an-dispatch.py)
# ---------------------------------------------------------------------------

def get_audio_channels(input_file):
    """Detect audio channel count via ffprobe.
    Returns 0 when the file verifiably has no audio stream, the channel count
    when it does, and 2 when the probe itself is unavailable/unreadable."""
    ffprobe_exe = shutil.which("ffprobe")
    if not ffprobe_exe:
        return 2
    try:
        result = subprocess.run(
            [ffprobe_exe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels", "-of", "csv=p=0", input_file],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and not result.stdout.strip():
            return 0
        return int(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 2


def opus_bitrate_for_channels(channels):
    if channels > 6:
        return "320k"
    elif channels >= 6:
        return "256k"
    elif channels >= 3:
        return "192k"
    return "128k"


# ---------------------------------------------------------------------------
# Color space detection (same logic as av1an-dispatch.py)
# ---------------------------------------------------------------------------

def detect_color_flags(input_file):
    """Returns extra SvtAv1EncApp color flags string, or empty string."""
    mediainfo_exe = shutil.which("mediainfo")
    if not mediainfo_exe or not os.path.exists(input_file):
        return ""

    f_prim_709 = f_trans_709 = f_mat_709 = False
    f_prim_601 = f_trans_601 = f_mat_601 = False

    try:
        result = subprocess.run(
            [mediainfo_exe, input_file],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        if result.returncode != 0:
            return ""
        for line in result.stdout.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "Color primaries":
                if value == "BT.709":
                    f_prim_709 = True
                elif "BT.601" in value:
                    f_prim_601 = True
            elif key == "Transfer characteristics":
                if value == "BT.709":
                    f_trans_709 = True
                elif "BT.601" in value:
                    f_trans_601 = True
            elif key == "Matrix coefficients":
                if value == "BT.709":
                    f_mat_709 = True
                elif "BT.601" in value:
                    f_mat_601 = True
    except Exception as e:
        print(f"[svtav1-dispatch] Warning: MediaInfo failed: {e}")
        return ""

    if f_prim_709 and f_trans_709 and f_mat_709:
        print("[svtav1-dispatch] MediaInfo confirmed full BT.709 source.")
        return " --color-primaries 1 --transfer-characteristics 1 --matrix-coefficients 1"
    elif f_prim_601 and f_trans_601 and f_mat_601:
        print("[svtav1-dispatch] MediaInfo confirmed full BT.601 source.")
        return " --color-primaries 6 --transfer-characteristics 6 --matrix-coefficients 6"
    else:
        print(
            f"[svtav1-dispatch] MediaInfo — 709: ({f_prim_709},{f_trans_709},{f_mat_709}) | "
            f"601: ({f_prim_601},{f_trans_601},{f_mat_601}). No standard color match."
        )
        return ""


# ---------------------------------------------------------------------------
# SSIMU2 measurement
# ---------------------------------------------------------------------------

def read_ssimu2_config():
    """Read tool from tools/workercount-ssimu2.txt."""
    config_path = os.path.join(os.path.dirname(__file__), "workercount-ssimu2.txt")
    tool = "vs-hip"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("tool="):
                        tool = line.split("=", 1)[1].strip()
        except Exception:
            pass
    return tool


def measure_ssimu2(source_file, encoded_file, tool, temp_dir=None):
    """
    Runs SSIMU2 comparison in a subprocess VapourSynth script.
    Returns (mean, p15) as floats, or (None, None) on failure.
    """
    # numStream=4 controls internal GPU parallelism within one measurement.
    # This is separate from workercount (concurrent processes in the pipeline).
    import os as _os
    _src_stem = _os.path.splitext(_os.path.basename(str(source_file)))[0]
    _enc_stem = _os.path.splitext(_os.path.basename(str(encoded_file)))[0]
    if temp_dir:
        _src_idx = _os.path.join(str(temp_dir), f"{_src_stem}.ffindex").replace("\\", "/")
        _enc_idx = _os.path.join(str(temp_dir), f"{_enc_stem}.ffindex").replace("\\", "/")
    else:
        _src_idx = (_os.path.splitext(_os.path.abspath(str(source_file)))[0] + ".ffindex").replace("\\", "/")
        _enc_idx = (_os.path.splitext(_os.path.abspath(str(encoded_file)))[0] + ".ffindex").replace("\\", "/")

    if tool == "vs-hip":
        vs_script = f"""
import vapoursynth as vs
from vstools import clip_async_render
core = vs.core
# VS R77+ autoloads only site-packages/vapoursynth/plugins; the legacy dirs
# (ffms2/vszip/vship live there) must be loaded explicitly.
import glob as _glob, os as _os
_failed = []
for _d in {vs_plugin_dirs()!r}:
    for _p in sorted(_glob.glob(_os.path.join(_d, "*.so"))):
        try:
            core.std.LoadPlugin(_p)
        except vs.Error as _e:
            if "already loaded" not in str(_e):
                _failed.append(_p + ": " + str(_e))
# A plugin we do not use may fail to load -- encoder-host carries a libvsncnn.so
# with an undefined onnx symbol -- so a load error is only fatal when the
# namespace we need is the missing one. Then say so HERE, with the loader
# message included, rather than leaving an AttributeError to surface further down
# (or, on a split-host encode, as a 0 MB read blamed on the encoder host).
for _ns in ("ffms2", "vship"):
    if not hasattr(core, _ns):
        raise RuntimeError("VapourSynth plugin " + _ns + " is missing. Plugin load errors: " + ("; ".join(_failed) or "none"))
src = core.ffms2.Source(source=r"{source_file}", cachefile=r"{_src_idx}").resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
enc = core.ffms2.Source(source=r"{encoded_file}", cachefile=r"{_enc_idx}").resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
res = core.vship.SSIMULACRA2(src, enc, numStream=4)
scores = clip_async_render(res, outfile=None, callback=lambda n, f: f.props["_SSIMULACRA2"])
for s in scores:
    print(s, flush=True)
"""
    elif tool == "vs-zip":
        vs_script = f"""
import vapoursynth as vs
from vstools import clip_async_render
core = vs.core
# VS R77+ autoloads only site-packages/vapoursynth/plugins; the legacy dirs
# (ffms2/vszip/vship live there) must be loaded explicitly.
import glob as _glob, os as _os
_failed = []
for _d in {vs_plugin_dirs()!r}:
    for _p in sorted(_glob.glob(_os.path.join(_d, "*.so"))):
        try:
            core.std.LoadPlugin(_p)
        except vs.Error as _e:
            if "already loaded" not in str(_e):
                _failed.append(_p + ": " + str(_e))
# A plugin we do not use may fail to load -- encoder-host carries a libvsncnn.so
# with an undefined onnx symbol -- so a load error is only fatal when the
# namespace we need is the missing one. Then say so HERE, with the loader
# message included, rather than leaving an AttributeError to surface further down
# (or, on a split-host encode, as a 0 MB read blamed on the encoder host).
for _ns in ("ffms2", "vszip"):
    if not hasattr(core, _ns):
        raise RuntimeError("VapourSynth plugin " + _ns + " is missing. Plugin load errors: " + ("; ".join(_failed) or "none"))
src = core.ffms2.Source(source=r"{source_file}", cachefile=r"{_src_idx}").resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
enc = core.ffms2.Source(source=r"{encoded_file}", cachefile=r"{_enc_idx}").resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
res = core.vszip.SSIMULACRA2(src, enc)
scores = clip_async_render(res, outfile=None, callback=lambda n, f: f.props["_SSIMULACRA2"])
for s in scores:
    print(s, flush=True)
"""
    else:
        print(f"[svtav1-dispatch] SSIMU2: unsupported tool '{tool}', skipping.")
        return None, None

    try:
        result = subprocess.run(
            [sys.executable, "-c", vs_script],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        scores = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                try:
                    scores.append(float(line))
                except ValueError:
                    pass
        if not scores:
            print(f"[svtav1-dispatch] SSIMU2: no scores returned.")
            if result.stderr:
                print(f"[svtav1-dispatch] SSIMU2 stderr: {result.stderr[:400]}")
            return None, None

        mean = sum(scores) / len(scores)
        scores_sorted = sorted(scores)
        p15_idx = max(0, int(len(scores_sorted) * 0.15) - 1)
        p15 = scores_sorted[p15_idx]
        return mean, p15

    except Exception as e:
        print(f"[svtav1-dispatch] SSIMU2 measurement failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def count_video_packets(path):
    """Video packet count of the first video stream (demux-only, no decode).

    Packets == frames for the codecs this pipeline handles; used to verify the
    encode is complete before muxing.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=1800)
        return int(out.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


USAGE = """\
svtav1-dispatch.py — single-pass vspipe|SvtAv1EncApp encode with Opus mux.

Required:
  -i/--input FILE, -o/--output FILE

Encode:      --quality CRF, --speed PRESET, --lp N, --photon-noise N,
             --encoder-params "...", --no-opus, --ssimu2
Denoisers (mutually exclusive):
  --denoise-scunet        [--denoise-model NAME --denoise-tile N --denoise-streams N]
  --denoise-smdegrain     [--denoise-tr N --denoise-thsad N]
  --denoise-rvrt          [--denoise-rvrt-sigma F]
  --denoise-stasunet      [--denoise-stasunet-engine PATH --denoise-stasunet-pre-darken-ev F]
  --denoise-bsvd          [--bsvd-onnx PATH --bsvd-sigma F|auto (default 0.05; auto = brightness-threshold pre-pass)
                           --bsvd-device N --bsvd-warmup N (legacy, no effect — auto window comes from the optsig model)
                           --bsvd-tile N (tile-sequential; for cards too small for the full-frame state)
                           --bsvd-overlap N (default 16) --bsvd-window N --bsvd-margin N (default 32)]
  --denoise-bsvd-smdegrain  (BSVD as SMDegrain prefilter; same --bsvd-* options)

Split-host denoise (BSVD only) — denoise on a remote GPU, encode here:
  --remote-denoise SSH_TARGET   [--remote-root PATH (default ~/archav1an)
                                 --remote-source PATH (read in place, skip staging)
                                 --remote-python PATH (default /opt/archav1an/venv/bin/python)
                                 --remote-port N (default 5300)
                                 --remote-callback IP (default: this host's IP toward the remote)]
  --denoise-serve HOST:PORT     internal: run the denoise half and stream y4m back

Split-host encode -- denoise here, encode on a remote CPU:
  --encode-serve IP:PORT        internal: receive y4m on IP:PORT and encode it
  --remote-encode HOST          encode on HOST instead of here
  --remote-encode-ip IP         the address HOST binds its y4m listener to, and
                                the address a --remote-denoise half connects to;
                                --remote-callback is ignored with --remote-encode
  --remote-encode-root PATH     the checkout on HOST (default ~/archav1an)

Concurrency:
  --temp-tag NAME               put working files in Temp/NAME/<stem> instead of
                                Temp/<stem>, so parallel runs over same-named
                                sources cannot delete each other's files
"""


def main():
    prefer_prefix_bin()
    args = sys.argv[1:]

    input_file = None
    output_file = None
    quality = None
    speed = None
    lp = "16"
    photon_noise = None
    encoder_params = ""
    no_opus = False
    measure_ssimu2_flag = False
    denoise_scunet = False
    denoise_model = "color_real_psnr"
    denoise_tile = 256
    denoise_streams = 2
    denoise_smdegrain = False
    denoise_rvrt = False
    denoise_rvrt_sigma = 12.0
    denoise_stasunet = False
    _default_stasunet_engine = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "stasunet_denoise_ep16_512_r4_fp16.engine")
    denoise_stasunet_engine = _default_stasunet_engine
    denoise_stasunet_pre_darken_ev = 0.0
    denoise_tr = 3
    denoise_thsad = 350
    denoise_bsvd = False
    bsvd_tile = 0
    bsvd_overlap = 32
    bsvd_window = 0
    bsvd_margin = 32
    denoise_bsvd_smdegrain = False
    _default_bsvd_onnx = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "bsvd_realpair_ep14_stateful_v2_dyn_fp16.onnx")
    _default_bsvd_optsig_model = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "bsvd_optsig_pref_v1.json")
    denoise_bsvd_onnx = _default_bsvd_onnx
    # 0.05 per 2026-07 preference-label study (tools/optsig_pref/loco_report.md);
    # "auto" = brightness-threshold rule.
    denoise_bsvd_sigma = "0.05"
    denoise_bsvd_device = 0
    denoise_bsvd_warmup = 30
    remote_denoise = None
    remote_root = "~/archav1an"
    remote_python = "/opt/archav1an/venv/bin/python"
    remote_port = 5300
    remote_callback = None
    remote_source = None
    denoise_serve = None
    encode_serve = None
    remote_encode = None
    remote_encode_ip = None
    remote_encode_root = "~/archav1an"
    temp_tag = None

    i = 0
    while i < len(args):
        arg = args[i]
        def nextval():
            return args[i + 1] if i + 1 < len(args) else None

        if arg in ("-i", "--input"):
            input_file = nextval(); i += 2
        elif arg in ("-o", "--output"):
            output_file = nextval(); i += 2
        elif arg == "--quality":
            quality = nextval(); i += 2
        elif arg == "--speed":
            speed = nextval(); i += 2
        elif arg == "--lp":
            lp = nextval(); i += 2
        elif arg == "--photon-noise":
            photon_noise = nextval(); i += 2
        elif arg == "--encoder-params":
            encoder_params = nextval() or ""; i += 2
        elif arg == "--no-opus":
            no_opus = True; i += 1
        elif arg == "--ssimu2":
            measure_ssimu2_flag = True; i += 1
        elif arg == "--denoise-scunet":
            denoise_scunet = True; i += 1
        elif arg == "--denoise-model":
            denoise_model = nextval() or "color_real_psnr"; i += 2
        elif arg == "--denoise-tile":
            denoise_tile = int(nextval() or 256); i += 2
        elif arg == "--denoise-streams":
            denoise_streams = int(nextval() or 2); i += 2
        elif arg == "--denoise-smdegrain":
            denoise_smdegrain = True; i += 1
        elif arg == "--denoise-rvrt":
            denoise_rvrt = True; i += 1
        elif arg == "--denoise-rvrt-sigma":
            denoise_rvrt_sigma = float(nextval() or 12.0); i += 2
        elif arg == "--denoise-stasunet":
            denoise_stasunet = True; i += 1
        elif arg == "--denoise-stasunet-engine":
            denoise_stasunet_engine = nextval() or _default_stasunet_engine; i += 2
        elif arg == "--denoise-stasunet-pre-darken-ev":
            denoise_stasunet_pre_darken_ev = float(nextval() or 0.0); i += 2
        elif arg == "--denoise-tr":
            denoise_tr = int(nextval() or 4); i += 2
        elif arg == "--denoise-thsad":
            denoise_thsad = int(nextval() or 350); i += 2
        elif arg == "--denoise-bsvd":
            denoise_bsvd = True; i += 1
        elif arg == "--denoise-bsvd-smdegrain":
            denoise_bsvd_smdegrain = True; i += 1
        elif arg == "--bsvd-onnx":
            denoise_bsvd_onnx = nextval() or _default_bsvd_onnx; i += 2
        elif arg == "--bsvd-sigma":
            denoise_bsvd_sigma = nextval() or "auto"; i += 2
        elif arg == "--bsvd-device":
            denoise_bsvd_device = int(nextval() or 0); i += 2
        elif arg == "--bsvd-warmup":
            denoise_bsvd_warmup = int(nextval() or 30); i += 2
        elif arg == "--remote-denoise":
            remote_denoise = nextval(); i += 2
        elif arg == "--remote-root":
            remote_root = nextval() or "~/archav1an"; i += 2
        elif arg == "--remote-python":
            remote_python = nextval() or "/opt/archav1an/venv/bin/python"; i += 2
        elif arg == "--remote-port":
            remote_port = int(nextval() or 5300); i += 2
        elif arg == "--remote-callback":
            remote_callback = nextval(); i += 2
        elif arg == "--remote-source":
            remote_source = nextval(); i += 2
        elif arg == "--denoise-serve":
            denoise_serve = nextval(); i += 2
        elif arg == "--encode-serve":
            encode_serve = nextval(); i += 2
        elif arg == "--remote-encode":
            remote_encode = nextval(); i += 2
        elif arg == "--remote-encode-ip":
            remote_encode_ip = nextval(); i += 2
        elif arg == "--remote-encode-root":
            remote_encode_root = nextval(); i += 2
        elif arg == "--temp-tag":
            temp_tag = nextval(); i += 2
        elif arg == "--bsvd-tile":
            bsvd_tile = parse_tile_arg(nextval()); i += 2
        elif arg == "--bsvd-overlap":
            bsvd_overlap = int(nextval() or 32); i += 2
        elif arg == "--bsvd-window":
            bsvd_window = int(nextval() or 0); i += 2
        elif arg == "--bsvd-margin":
            bsvd_margin = int(nextval() or 32); i += 2
        elif arg in ("-h", "--help"):
            print(USAGE)
            sys.exit(0)
        else:
            # Unknown flags used to be silently ignored — a typo'd
            # --denoise flag meant an un-denoised encode shipped as success.
            print(f"[svtav1-dispatch] Error: unrecognized argument: {arg}")
            print(USAGE)
            sys.exit(2)

    if denoise_serve:
        # The serve half never muxes: it streams y4m and exits.
        output_file = output_file or os.devnull
    if encode_serve and not output_file:
        # Checked before the generic guard below, which would otherwise answer
        # with "-i and -o are required" and send the operator looking for an
        # input file this mode does not have.
        print("[svtav1-dispatch] Error: --encode-serve needs -o, the IVF path to write.")
        sys.exit(1)
    if encode_serve:
        # The frames arrive on a socket, so there is no input file and Task 9's
        # remote command sends none. os.devnull keeps the shared setup below
        # happy; stem and temp_dir come from the output path instead.
        input_file = input_file or os.devnull
    if not input_file or not output_file:
        print("[svtav1-dispatch] Error: -i and -o are required.")
        sys.exit(1)
    if remote_denoise and not (denoise_bsvd or denoise_bsvd_smdegrain):
        print("[svtav1-dispatch] Error: --remote-denoise supports the BSVD paths "
              "(--denoise-bsvd / --denoise-bsvd-smdegrain).")
        sys.exit(2)
    if remote_denoise and denoise_serve:
        print("[svtav1-dispatch] Error: --remote-denoise and --denoise-serve are exclusive.")
        sys.exit(2)
    if remote_source and not remote_denoise:
        print("[svtav1-dispatch] Error: --remote-source needs --remote-denoise.")
        sys.exit(2)
    if denoise_serve and not (denoise_bsvd or denoise_bsvd_smdegrain):
        print("[svtav1-dispatch] Error: --denoise-serve needs a BSVD denoise flag.")
        sys.exit(2)
    if encode_serve and remote_encode:
        print("[svtav1-dispatch] Error: --encode-serve and --remote-encode are exclusive.")
        sys.exit(1)
    if denoise_serve and remote_encode:
        # The sink chain gives --denoise-serve priority, so --remote-encode
        # would be ignored without a word and the clip would encode wherever
        # the serve target points.
        print("[svtav1-dispatch] Error: --denoise-serve and --remote-encode are exclusive.")
        sys.exit(1)
    if encode_serve and ":" not in encode_serve:
        print("[svtav1-dispatch] Error: --encode-serve takes IP:PORT, not a bare port. "
              "The listener binds that address on purpose: off-LAN the bind fails and "
              "the lane refuses to start rather than listen on an untrusted network.")
        sys.exit(1)
    if remote_encode and not remote_encode_ip:
        print("[svtav1-dispatch] Error: --remote-encode needs --remote-encode-ip, "
              "the address the encoder binds and the denoise host connects to.")
        sys.exit(1)

    svt_exe = shutil.which("SvtAv1EncApp")
    ffmpeg_exe = shutil.which("ffmpeg")
    # The serve half only denoises: a remote GPU box needs neither encoder.
    if not svt_exe and not denoise_serve:
        print("[svtav1-dispatch] Error: SvtAv1EncApp not found in PATH.")
        sys.exit(1)
    if not ffmpeg_exe and not denoise_serve:
        print("[svtav1-dispatch] Error: ffmpeg not found in PATH.")
        sys.exit(1)
    if remote_denoise and not remote_source and not shutil.which("rsync"):
        print("[svtav1-dispatch] Error: --remote-denoise needs rsync in PATH.")
        sys.exit(1)
    # The frame-count verification before the mux is the guard that stops a
    # truncated encode from being published; without ffprobe it fails open,
    # so an encoding host without it must not start.
    if not denoise_serve and not shutil.which("ffprobe"):
        print("[svtav1-dispatch] Error: ffprobe not found in PATH; it is needed "
              "to verify the frame count before the mux.")
        sys.exit(1)

    # Color detection
    # Not for the encode half: its input_file is os.devnull, so these probe
    # nothing and print a colour and audio line that describe no real clip.
    # The host that holds the source does the muxing, and does these there.
    color_flags = "" if encode_serve else detect_color_flags(input_file)
    if color_flags:
        encoder_params = encoder_params + color_flags

    # Build SvtAv1EncApp params string
    svt_params = ""
    if speed:
        svt_params += f" --preset {speed}"
    if quality:
        svt_params += f" --crf {quality}"
    svt_params += f" --lp {lp}"
    if photon_noise and photon_noise != "0":
        svt_params += f" --film-grain {photon_noise}"
    if encoder_params.strip():
        svt_params += " " + encoder_params.strip()

    # Temp ivf path. --temp-tag inserts one component so that concurrent runs
    # cannot share a temp dir: 185 stems repeat across the dance archive, and
    # the loser has its working files deleted mid-encode.
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # The encode half has no input file, so its only name is the IVF it was
    # told to write. Everything else -- the netstream log especially -- hangs
    # off this.
    _name_source = output_file if encode_serve else input_file
    stem = os.path.splitext(os.path.basename(_name_source))[0]
    if temp_tag and (os.sep in temp_tag or temp_tag in (".", "..")):
        print(f"[svtav1-dispatch] Error: --temp-tag must be one path component, "
              f"got {temp_tag!r}.")
        sys.exit(2)
    temp_dir = (os.path.join(root_dir, "Temp", temp_tag, stem) if temp_tag
                else os.path.join(root_dir, "Temp", stem))
    os.makedirs(temp_dir, exist_ok=True)
    ivf_path = os.path.join(temp_dir, f"{stem}.ivf")



    # VSPIPE env pins a specific vspipe binary. Needed on encoder-host, where BSVD's
    # MIGraphX wheels stop at cp312: the migraphx-venv ships its own py3.12 vspipe
    # (pip vapoursynth) while system vspipe stays py3.14 — see denoiser docs.
    vspipe_exe = os.environ.get("VSPIPE") or shutil.which("vspipe")
    # The encode half runs netstream recv | SvtAv1EncApp and never calls
    # vspipe, so an encode-only host must not be refused for it.
    if not encode_serve and (not vspipe_exe or not os.path.exists(vspipe_exe)):
        print("[svtav1-dispatch] Error: vspipe not found (PATH or $VSPIPE).")
        sys.exit(1)

    svt_cmd = [svt_exe, "-i", "stdin", "--progress", "2"] + shlex.split(svt_params.strip()) + ["-b", ivf_path]


    # Audio -- skipped for the encode half, see the color detection above.
    has_audio = False
    if not encode_serve:
        has_audio = get_audio_channels(input_file) != 0
        if not has_audio:
            print("[svtav1-dispatch] Audio: none in source — video-only mux")
        elif no_opus:
            print("[svtav1-dispatch] Audio: passthrough (--no-opus)")
        else:
            channels = get_audio_channels(input_file)
            opus_bitrate = opus_bitrate_for_channels(channels)
            print(f"[svtav1-dispatch] Audio: Opus {opus_bitrate} ({channels}ch)")

    # Preserve source mtime
    src_stat = os.stat(input_file) if os.path.exists(input_file) else None

    # BSVD+SMDegrain wants a much lower thSAD than SCUNet+SMDegrain, because
    # BSVD already does heavy temporal denoising — high thSAD just smears it.
    # Lossless FFV1 sweep on MVI_4378/0487/8656 showed thSAD=150 strictly
    # dominates 350 by 0.5–1.3 SSIMU2 (see memory bsvd_smdegrain_hybrid_sweep.md).
    # Only swap if the user didn't explicitly pass --denoise-thsad.
    if denoise_bsvd_smdegrain and denoise_thsad == 350:
        denoise_thsad = 150

    # Mutual exclusion: BSVD is incompatible with the other temporal denoisers.
    _denoise_flags = [denoise_scunet, denoise_smdegrain, denoise_rvrt,
                       denoise_stasunet, denoise_bsvd, denoise_bsvd_smdegrain]
    if sum(bool(f) for f in _denoise_flags) > 1:
        print("[svtav1-dispatch] Error: --denoise-{scunet,smdegrain,rvrt,stasunet,bsvd,bsvd-smdegrain} are mutually exclusive.")
        sys.exit(1)

    # --- Encode ---
    _encode_forward = build_encode_forward(quality, speed, lp, photon_noise,
                                           encoder_params, temp_tag)
    if encode_serve:
        run_encode_serve(encode_serve, svt_cmd, ivf_path, temp_dir, stem)
        # The receiver's job ends with the last frame. The frame-count check,
        # the mux and the tag all live on the host that holds the source.
        sys.exit(0)
    if remote_denoise:
        # Model, EP and VPY all belong to the remote half; this side only
        # resolves sigma (it has the source) and receives y4m.
        bsvd_sigma_val = resolve_bsvd_sigma(denoise_bsvd_sigma, input_file,
                                            _default_bsvd_optsig_model)
        _forward = ["--denoise-bsvd-smdegrain" if denoise_bsvd_smdegrain
                    else "--denoise-bsvd",
                    "--bsvd-sigma", f"{bsvd_sigma_val:.4f}",
                    "--bsvd-device", str(denoise_bsvd_device)]
        if denoise_bsvd_smdegrain:
            _forward += ["--denoise-tr", str(denoise_tr),
                         "--denoise-thsad", str(denoise_thsad)]
        if temp_tag:
            # The remote resolves this against its own root, so two remote
            # denoisers on one host stay out of each other's Temp.
            _forward += ["--temp-tag", temp_tag]
        if bsvd_tile:
            # The denoise half runs remotely, so the tiling belongs there.
            _forward += ["--bsvd-tile", tile_arg(bsvd_tile),
                         "--bsvd-overlap", str(bsvd_overlap),
                         "--bsvd-window", str(bsvd_window),
                         "--bsvd-margin", str(bsvd_margin)]
        # With a remote encoder the denoise half streams straight to it, so
        # this host never sees the y4m at all and only the IVF comes back.
        _callback = (remote_encode_ip if remote_encode
                     else (remote_callback or callback_address(remote_denoise)))
        # Only where nothing explicit chose the address. --remote-encode-ip is
        # required, so a remote encoder always has an explicit one -- and it
        # would be the wrong flag to name here anyway, because --remote-callback
        # names THIS host and this path never streams here.
        if (_callback.startswith("100.")
                and not (remote_callback or remote_encode)):
            print(f"[svtav1-dispatch] Warning: streaming back over {_callback} "
                  "(tailscale) caps at ~1.5 Gbps; pass --remote-callback with "
                  "this host's LAN IP for the direct path.")
        print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
        if remote_encode:
            # The encoder is started first because it LISTENS, and netstream's
            # send retries the connect for 120s. Starting the denoiser first
            # would work too, but only inside that retry window; this way the
            # window is never needed.
            #
            # Both calls block, so they cannot be sequential. run_remote_encode
            # owns the wait, and the denoise ssh is launched inside it.
            run_remote_encode_split(
                denoise_target=remote_denoise, denoise_root=remote_root,
                encode_target=remote_encode, encode_root=remote_encode_root,
                remote_python=remote_python,
                bind=f"{remote_encode_ip}:{remote_port}", ivf_path=ivf_path,
                input_file=input_file, denoise_forward=_forward,
                encode_forward=_encode_forward, temp_dir=temp_dir, stem=stem,
                remote_source=remote_source, temp_tag=temp_tag)
        else:
            run_remote_denoise(remote_denoise, remote_root, remote_python,
                               _callback, remote_port, input_file, _forward,
                               svt_cmd, temp_dir, stem,
                               remote_source=remote_source)
    elif any(_denoise_flags):
        if denoise_bsvd or denoise_bsvd_smdegrain:
            if not os.path.exists(denoise_bsvd_onnx):
                print(f"[svtav1-dispatch] Error: BSVD ONNX not found at {denoise_bsvd_onnx}. "
                      "Stage it via setup.sh --install denoiser or pass --bsvd-onnx.")
                sys.exit(1)
            # EP detection: BSVD runs via Python onnxruntime, so ask ORT what it
            # can actually use (presence of the unrelated vstrt VS plugin used to
            # pick MIGraphX on NVIDIA hosts and let ORT fall back to CPU silently).
            try:
                import onnxruntime as _ort
            except ImportError:
                print("[svtav1-dispatch] Error: --denoise-bsvd needs onnxruntime "
                      "(pip install onnxruntime-gpu, or setup.sh --install denoiser).")
                sys.exit(1)
            _ort_providers = _ort.get_available_providers()
            # onnxruntime-gpu always LISTS the TRT provider; session creation
            # still fails if the TensorRT runtime isn't installed. So ask the
            # question that actually decides it: can the provider library load?
            #
            # This used to be ctypes.util.find_library("nvinfer"), which asks
            # only whether SOME nvinfer sits in the ldconfig cache, and got both
            # ends of that wrong. It answers yes to the system AUR tensorrt
            # (11.x, soname .so.11) that the provider cannot use, so the session
            # then dies with "libnvinfer.so.10 => not found"; and it reads only
            # the ldconfig cache, so it answers no on a non-root install even
            # though setup.sh's own LD_LIBRARY_PATH advice has made the library
            # loadable -- selecting the slower CUDA EP with no message at all.
            # Loading the provider resolves its real DT_NEEDED chain (nvinfer
            # .so.10 AND cudnn .so.9), honours LD_LIBRARY_PATH, and needs no
            # soname hardcoded here.
            _trt_loads = False
            if "TensorrtExecutionProvider" in _ort_providers:
                import ctypes
                _prov_so = os.path.join(os.path.dirname(_ort.__file__), "capi",
                                        "libonnxruntime_providers_tensorrt.so")
                try:
                    ctypes.CDLL(_prov_so)
                    _trt_loads = True
                except OSError as _e:
                    print(f"[svtav1-dispatch] TensorRT EP unavailable ({_e}); "
                          "falling back. Run setup.sh --install denoiser as root "
                          "to wire it.")
            if _trt_loads:
                bsvd_ep = "TRT"
            elif "CUDAExecutionProvider" in _ort_providers:
                bsvd_ep = "CUDA"
            elif "MIGraphXExecutionProvider" in _ort_providers:
                bsvd_ep = "MIGRAPHX"
            else:
                print("[svtav1-dispatch] Error: no GPU execution provider in onnxruntime "
                      f"(available: {_ort_providers}). Install onnxruntime-gpu (NVIDIA) "
                      "or an ORT-ROCm build (AMD).")
                sys.exit(1)
            bsvd_sigma_val = resolve_bsvd_sigma(denoise_bsvd_sigma, input_file,
                                                _default_bsvd_optsig_model)
            _backend_name = f"BSVD-V2-ORT-{bsvd_ep}"
        elif denoise_rvrt:
            _backend_name = "RVRT"
        elif denoise_stasunet:
            if not os.path.exists(denoise_stasunet_engine):
                print(f"[svtav1-dispatch] Error: STA-SUNet engine not found at {denoise_stasunet_engine}")
                sys.exit(1)
            _engine_base = os.path.basename(denoise_stasunet_engine)
            _engine_tile = 768 if "_768" in _engine_base else 512
            if denoise_tile != _engine_tile:
                print(f"[svtav1-dispatch] STA-SUNet engine {_engine_base} is fixed-shape {_engine_tile}x{_engine_tile}; forcing --denoise-tile {_engine_tile} (was {denoise_tile}).")
                denoise_tile = _engine_tile
            _backend_name = "STA-SUNet-TRT"
        else:
            mlrt_plugin = find_mlrt_plugin()
            if not mlrt_plugin:
                print("[svtav1-dispatch] Error: no vs-mlrt plugin found (libvstrt.so or libvsmigx.so). Run setup.sh --install denoiser.")
                sys.exit(1)
            _backend_name = "TRT" if "vstrt" in mlrt_plugin else "MIGraphX"
        if denoise_serve:
            _serve_host, _, _serve_port = denoise_serve.rpartition(":")
            _sink_cmd = [sys.executable,
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "netstream.py"),
                         "send", "--host", _serve_host, "--port", _serve_port]
            _sink_label = f"netstream -> {denoise_serve}"
        elif remote_encode:
            # Case 3: denoise here, encode there. run_remote_encode owns the
            # pipe because it must also wait on the remote ssh, so the sink is
            # left unset and the run_piped call below is skipped.
            _sink_cmd = None
            _sink_label = f"netstream -> {remote_encode_ip}:{remote_port}"
        else:
            _sink_cmd = svt_cmd
            _sink_label = f"SvtAv1EncApp{svt_params}"
        vpy_path = os.path.join(temp_dir, f"{stem}_denoise.vpy")
        cachefile = os.path.join(temp_dir, f"{stem}.ffindex")
        write_denoise_vpy(vpy_path, input_file, cachefile,
                      denoise_model, denoise_tile, denoise_streams,
                      use_smdegrain=denoise_smdegrain,
                      tr=denoise_tr, thsad=denoise_thsad,
                      use_rvrt=denoise_rvrt, rvrt_sigma=denoise_rvrt_sigma,
                      use_stasunet=denoise_stasunet, stasunet_engine=denoise_stasunet_engine,
                      stasunet_pre_darken_ev=denoise_stasunet_pre_darken_ev,
                      use_bsvd=denoise_bsvd, use_bsvd_smdegrain=denoise_bsvd_smdegrain,
                      bsvd_onnx=denoise_bsvd_onnx,
                      bsvd_sigma=(bsvd_sigma_val if (denoise_bsvd or denoise_bsvd_smdegrain) else 0.08),
                      bsvd_ep=(bsvd_ep if (denoise_bsvd or denoise_bsvd_smdegrain) else "TRT"),
                      bsvd_tile=bsvd_tile, bsvd_overlap=bsvd_overlap,
                      bsvd_window=bsvd_window, bsvd_margin=bsvd_margin,
                      bsvd_device=denoise_bsvd_device)
        if denoise_bsvd_smdegrain:
            print(f"[svtav1-dispatch] vspipe ({_backend_name} + SMDegrain tr={denoise_tr} thSAD={denoise_thsad}, σ={bsvd_sigma_val:.3f}) | {_sink_label}")
        elif denoise_bsvd:
            print(f"[svtav1-dispatch] vspipe ({_backend_name} σ={bsvd_sigma_val:.3f}) | {_sink_label}")
        elif denoise_rvrt:
            print(f"[svtav1-dispatch] vspipe (RVRT sigma={denoise_rvrt_sigma}, tile={denoise_tile}) | {_sink_label}")
        elif denoise_stasunet:
            print(f"[svtav1-dispatch] vspipe (STA-SUNet engine={os.path.basename(denoise_stasunet_engine)}, tile={denoise_tile}, streams={denoise_streams}) | {_sink_label}")
        elif denoise_smdegrain:
            print(f"[svtav1-dispatch] vspipe ({_backend_name} SCUNet+SMDegrain tr={denoise_tr} thSAD={denoise_thsad}, tile={denoise_tile}, streams={denoise_streams}) | {_sink_label}")
        else:
            print(f"[svtav1-dispatch] vspipe ({_backend_name} SCUNet-{denoise_model}, tile={denoise_tile}, streams={denoise_streams}) | {_sink_label}")
        if not denoise_serve:
            print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
        sys.stdout.flush()
        # -p makes vspipe print "Frame: N/M" to stderr about once a second, and
        # source_stderr_log below already sends that stderr to a file. Without
        # it the log holds only the closing summary and there is no live rate.
        if _sink_cmd is None:
            run_remote_encode(remote_encode, remote_encode_root, remote_python,
                              f"{remote_encode_ip}:{remote_port}", ivf_path,
                              _encode_forward,
                              [vspipe_exe, "-p", "-c", "y4m", vpy_path, "-"],
                              temp_dir, stem, temp_tag=temp_tag)
        else:
            run_piped([vspipe_exe, "-p", "-c", "y4m", vpy_path, "-"], _sink_cmd,
                      source_label="vspipe",
                      sink_label=("netstream send" if denoise_serve
                                  else "SvtAv1EncApp"),
                      source_stderr_log=os.path.join(temp_dir,
                                                     f"{stem}_vspipe.log"))
        if denoise_serve:
            # The serve half's job ends with the last frame on the socket:
            # the encode, frame-count check and mux all live on the receiver.
            sys.exit(0)
    else:
        src_vpy_path = os.path.join(temp_dir, f"{stem}_src.vpy")
        src_cachefile = os.path.join(temp_dir, f"{stem}.ffindex")
        input_file_fwd = os.path.abspath(input_file).replace("\\", "/")
        with open(src_vpy_path, "w", encoding="utf-8") as vf:
            vf.write(
                f"import vapoursynth as vs\n"
                f"core = vs.core\n"
                # The same preamble the denoise VPY carries. This is the
                # no-denoise path, and until encode jobs existed nothing in the
                # batch reached it -- so it was the one generated script still
                # relying on autoload, and on a host whose ffms2 lives in a
                # legacy dir it failed with a bare AttributeError.
                + plugin_preamble() +
                f"src = core.ffms2.Source(r'{input_file_fwd}', cachefile=r'{src_cachefile}')\n"
                f"if src.format.color_family == vs.RGB:\n"
                f"    # RGB sources (e.g. Lagarith/FFV1 eval intermediates) need an explicit\n"
                f"    # matrix for the YUV conversion; no chroma siting on RGB input.\n"
                f"    src = src.resize.Bicubic(format=vs.YUV420P10, matrix_s='709', chromaloc_s='left')\n"
                f"else:\n"
                f"    src = src.resize.Bicubic(format=vs.YUV420P10, chromaloc_in_s='left', chromaloc_s='left')\n"
                f"src.set_output()\n"
            )
        # -p, like every other vspipe here. Without it this path prints no
        # counter at all, so the dashboard has nothing to read and an encode
        # job renders with no rate and no progress bar for its whole length.
        # It was the one branch nothing measured, because until encode jobs
        # existed nothing in the batch reached it. Both arms below then send
        # that stderr to <stem>_vspipe.log, which is the file the daemon reads.
        _src_cmd = [vspipe_exe, "-p", "-c", "y4m", src_vpy_path, "-"]
        if remote_encode:
            # Decode here, encode there, with no denoise pass anywhere: an
            # encode job on a remote host. run_remote_encode asks only for a
            # command producing y4m, and this branch produces the same y4m the
            # denoise branch does -- one bicubic convert earlier in the chain.
            #
            # This used to be refused outright. The check was right about the
            # code as it stood, because nothing here routed to the remote and
            # the encode would have run locally while claiming otherwise, but
            # it made every encode job a encoder-host job: --remote-encode is what
            # archive-batch passes for every host whose roster entry is not
            # "local", so the whole pool answered with one error line.
            print(f"[svtav1-dispatch] vspipe (bicubic 422→420) --> "
                  f"{remote_encode}")
            print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
            sys.stdout.flush()
            run_remote_encode(remote_encode, remote_encode_root, remote_python,
                              f"{remote_encode_ip}:{remote_port}", ivf_path,
                              _encode_forward, _src_cmd, temp_dir, stem,
                              temp_tag=temp_tag)
        else:
            print(f"[svtav1-dispatch] vspipe (bicubic 422→420) | SvtAv1EncApp{svt_params}")
            print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
            sys.stdout.flush()
            run_piped(_src_cmd, svt_cmd,
                      source_label="vspipe",
                      source_stderr_log=os.path.join(temp_dir,
                                                     f"{stem}_vspipe.log"))

    # --- Frame-count verification (closes the truncated-encode hole even when
    # every process exits 0, e.g. an upstream EOF at a frame boundary) ---
    _src_frames = count_video_packets(input_file)
    _enc_frames = count_video_packets(ivf_path)
    if _src_frames and _enc_frames:
        if _enc_frames != _src_frames:
            print(f"[svtav1-dispatch] Error: encoded frame count {_enc_frames} "
                  f"!= source {_src_frames}; aborting before mux.")
            sys.exit(1)
        print(f"[svtav1-dispatch] Frame count verified: {_enc_frames} frames.")
    elif _src_frames and not _enc_frames:
        # The source counts, so the encode side must too. Muxing on would be
        # exactly the silent truncated publish this check exists to stop.
        print(f"[svtav1-dispatch] Error: could not count frames in {ivf_path}; "
              f"aborting before mux.")
        sys.exit(1)
    else:
        # The source itself is uncountable, so there is nothing to compare
        # against; say so and continue.
        print("[svtav1-dispatch] Warning: could not verify frame count "
              "(source stream unreadable).")

    # --- Mux ---
    print("[svtav1-dispatch] Muxing...")
    if has_audio:
        audio_codec = ["-c:a", "copy"] if no_opus else ["-c:a", "libopus", "-b:a", opus_bitrate]
        audio_args = ["-map", "1:a", *audio_codec]
    else:
        audio_args = []
    mux_cmd = [
        ffmpeg_exe, "-y",
        "-i", ivf_path, "-i", input_file,
        "-map", "0:v", *audio_args,
        "-c:v", "copy",
        output_file,
    ]

    try:
        subprocess.check_call(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[svtav1-dispatch] Mux failed: {e}")
        sys.exit(e.returncode)

    # --- SSIMU2 (opt-in via --ssimu2) ---
    if measure_ssimu2_flag:
        ssimu2_tool = read_ssimu2_config()
        print(f"[svtav1-dispatch] Measuring SSIMU2 ({ssimu2_tool})...")
        mean, p15 = measure_ssimu2(input_file, output_file, ssimu2_tool, temp_dir=temp_dir)
        if mean is not None:
            print(f"[svtav1-dispatch] SSIMU2  mean: {mean:.2f} | p15: {p15:.2f}")
        else:
            print("[svtav1-dispatch] SSIMU2 measurement failed.")

    # --- Preserve mtime ---
    if src_stat and os.path.exists(output_file):
        os.utime(output_file, (src_stat.st_atime, src_stat.st_mtime))

    # --- Tag output file ---
    if os.path.exists(output_file):
        fish_version = _tag.get_5fish_version()
        general_flags = [f"--quality {quality}"]
        if photon_noise and photon_noise != "0":
            general_flags.append(f"--photon-noise {photon_noise}")
        general_flags.append(f"--speed {speed}")
        if denoise_bsvd_smdegrain:
            general_flags.append(f"--denoise-bsvd-smdegrain --bsvd-sigma {bsvd_sigma_val:.3f} --denoise-tr {denoise_tr} --denoise-thsad {denoise_thsad}")
        elif denoise_bsvd:
            general_flags.append(f"--denoise-bsvd --bsvd-sigma {bsvd_sigma_val:.3f}")
        elif denoise_rvrt:
            general_flags.append(f"--denoise-rvrt --denoise-rvrt-sigma {denoise_rvrt_sigma}")
        elif denoise_stasunet:
            general_flags.append(f"--denoise-stasunet --denoise-tile {denoise_tile}")
        elif denoise_scunet:
            general_flags.append(f"--denoise-scunet --denoise-model {denoise_model} --denoise-tile {denoise_tile}")
        encoding_settings, encoder_name = _tag.build_tag_strings(
            general_flags, encoder_params, quality, speed, fish_version
        )
        _tag.apply_tag_to_file(output_file, encoding_settings, encoder_name)

    # --- Cleanup temp files ---
    for tmp in (ivf_path,
                os.path.join(temp_dir, f"{stem}_denoise.vpy"),
                os.path.join(temp_dir, f"{stem}_src.vpy"),
                os.path.join(temp_dir, f"{stem}_vspipe.log")):
        try:
            os.remove(tmp)
        except OSError:
            pass

    # Register output in tag manifest so tag.py only tags this run's files
    manifest_path = os.path.join(root_dir, "tools", "tag-manifest.txt")
    try:
        with open(manifest_path, "a", encoding="utf-8") as mf:
            mf.write(os.path.abspath(output_file) + "\n")
    except OSError:
        pass

    print(f"[svtav1-dispatch] Done: {output_file}")


if __name__ == "__main__":
    main()
