#!/usr/bin/env python3
"""
svtav1-dispatch.py — Single-pass SvtAv1EncApp encode with Opus mux and SSIMU2 measurement.

Pipes ffmpeg → SvtAv1EncApp, muxes Opus audio, then measures SSIMU2 scores
(mean + 15th percentile) for comparison against av1an pipeline output.
"""

import os
import socket
import sys
import shlex
import subprocess
import shutil
import sysconfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tag as _tag


# ---------------------------------------------------------------------------
# Denoise helpers
# ---------------------------------------------------------------------------

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
                      bsvd_onnx="", bsvd_sigma=0.08, bsvd_ep="TRT", bsvd_device=0):
    source = os.path.abspath(source)
    backend_lines = _mlrt_backend_lines(streams)
    model_line = f'_model_enum = _SCUNetModel["scunet_{model_name}"]'

    if use_bsvd or use_bsvd_smdegrain:
        bsvd_onnx = os.path.abspath(bsvd_onnx)
        _tools_dir = os.path.dirname(os.path.abspath(__file__))
        denoise_lines = (
            f'import sys as _sys; _sys.path.insert(0, r{_tools_dir!r})\n'
            f'from bsvd_vs_filter import build_bsvd_streaming\n'
            f'_src_fmt = src.format\n'
            f'_rgb = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709", range_in_s="limited")\n'
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
    vpy = (
        f'import sys as _sys; _sys.path.insert(0, {venv_site_pkgs!r})\n'
        f'from vstools import vs, core, initialize_clip, finalize_clip\n'
        f'core.max_cache_size = {_cache_mb}\n'
        f'# VS R77+ autoloads only site-packages/vapoursynth/plugins; the legacy dirs\n# (ffms2/vszip/vship live there) must be loaded explicitly.\nimport glob as _glob, os as _os\nfor _d in ("/usr/lib/vapoursynth", "/usr/local/lib/vapoursynth"):\n    for _p in sorted(_glob.glob(_os.path.join(_d, "*.so"))):\n        try:\n            core.std.LoadPlugin(_p)\n        except vs.Error:\n            pass  # already loaded\n\n'
        f'\n'
        f'src = core.ffms2.Source(source=r{source!r}, cachefile=r{cachefile!r})\n'
        f'if src.format.color_family == vs.RGB:\n'
        f'    # RGB sources (Lagarith/FFV1 eval intermediates): normalize once to\n'
        f'    # YUV420P10 so every denoise builder below can assume YUV input and\n'
        f'    # the y4m pipe to SvtAv1EncApp stays 4:2:0.\n'
        f'    src = core.resize.Bicubic(src, format=vs.YUV420P10, matrix_s="709", chromaloc_s="left")\n'
        f'src = initialize_clip(src)\n'
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


# ---------------------------------------------------------------------------
# Encode helpers
# ---------------------------------------------------------------------------

def _print_log_tail(log_path, label, max_lines=40):
    """Print the last max_lines of a captured stderr log, for post-mortem."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            tail = f.readlines()[-max_lines:]
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


def run_remote_denoise(ssh_target, remote_root, remote_python, callback, port,
                       input_file, forward_args, sink_cmd, temp_dir, stem):
    """Denoise on ssh_target, stream the y4m back over TCP, encode into sink_cmd.

    ssh carries only control (launch, stderr, exit status): a single ssh stream
    caps at ~1.3 Gbps between encoder-host and gpu1 even on the 10G LAN, while a
    plain socket sustains 5+ Gbps. The encoder side listens so that the listener
    dies with the process that owns the encoder.

    A remote that dies mid-stream closes the socket cleanly, so the local
    encoder finalizes a short IVF and exits 0 -- run_piped cannot see it. The
    guards are this function's ssh exit-status check and, definitively, the
    frame-count verification before the mux.
    """
    remote_dir = f"{remote_root}/Temp/_remote"
    print(f"[svtav1-dispatch] staging source -> {ssh_target}:{remote_dir}/")
    subprocess.check_call(["rsync", "-a", "--rsync-path",
                           f"mkdir -p {remote_dir} && rsync",
                           os.path.abspath(input_file),
                           f"{ssh_target}:{remote_dir}/"])
    # Paths in the remote command are relative to remote_root: the command runs
    # after `cd`, and quoting a leading ~ would stop the remote shell expanding it.
    remote_src = f"Temp/_remote/{os.path.basename(input_file)}"

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
        # PYTHONUNBUFFERED so the remote's diagnostics reach the log even when it
        # is killed: over a pipe its stdout would otherwise be block-buffered.
        ssh_proc.stdin.write(
            f"cd {remote_root} && PYTHONUNBUFFERED=1 exec {remote_cmd}\n".encode())
        ssh_proc.stdin.close()
        local_failed = False
        try:
            run_piped([sys.executable,
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "netstream.py"),
                       "recv", "--port", str(port)],
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
    if ssh_proc.returncode != 0:
        print(f"[svtav1-dispatch] Error: remote denoise on {ssh_target} exited with "
              f"{ssh_proc.returncode}; output is truncated -- aborting before mux.")
        _print_log_tail(remote_log, f"{ssh_target} stderr")
        sys.exit(abs(ssh_proc.returncode) or 1)


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
for _d in ("/usr/lib/vapoursynth", "/usr/local/lib/vapoursynth"):
    for _p in sorted(_glob.glob(_os.path.join(_d, "*.so"))):
        try:
            core.std.LoadPlugin(_p)
        except vs.Error:
            pass  # already loaded
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
for _d in ("/usr/lib/vapoursynth", "/usr/local/lib/vapoursynth"):
    for _p in sorted(_glob.glob(_os.path.join(_d, "*.so"))):
        try:
            core.std.LoadPlugin(_p)
        except vs.Error:
            pass  # already loaded
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
                           --bsvd-device N --bsvd-warmup N (legacy, no effect — auto window comes from the optsig model)]
  --denoise-bsvd-smdegrain  (BSVD as SMDegrain prefilter; same --bsvd-* options)

Split-host denoise (BSVD only) — denoise on a remote GPU, encode here:
  --remote-denoise SSH_TARGET   [--remote-root PATH (default ~/archav1an)
                                 --remote-python PATH (default /opt/archav1an/venv/bin/python)
                                 --remote-port N (default 5300)
                                 --remote-callback IP (default: this host's IP toward the remote)]
  --denoise-serve HOST:PORT     internal: run the denoise half and stream y4m back
"""


def main():
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
    denoise_serve = None

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
        elif arg == "--denoise-serve":
            denoise_serve = nextval(); i += 2
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
    if denoise_serve and not (denoise_bsvd or denoise_bsvd_smdegrain):
        print("[svtav1-dispatch] Error: --denoise-serve needs a BSVD denoise flag.")
        sys.exit(2)

    svt_exe = shutil.which("SvtAv1EncApp")
    ffmpeg_exe = shutil.which("ffmpeg")
    # The serve half only denoises: a remote GPU box needs neither encoder.
    if not svt_exe and not denoise_serve:
        print("[svtav1-dispatch] Error: SvtAv1EncApp not found in PATH.")
        sys.exit(1)
    if not ffmpeg_exe and not denoise_serve:
        print("[svtav1-dispatch] Error: ffmpeg not found in PATH.")
        sys.exit(1)
    if remote_denoise and not shutil.which("rsync"):
        print("[svtav1-dispatch] Error: --remote-denoise needs rsync in PATH.")
        sys.exit(1)

    # Color detection
    color_flags = detect_color_flags(input_file)
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

    # Temp ivf path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stem = os.path.splitext(os.path.basename(input_file))[0]
    temp_dir = os.path.join(root_dir, "Temp", stem)
    os.makedirs(temp_dir, exist_ok=True)
    ivf_path = os.path.join(temp_dir, f"{stem}.ivf")



    # VSPIPE env pins a specific vspipe binary. Needed on encoder-host, where BSVD's
    # MIGraphX wheels stop at cp312: the migraphx-venv ships its own py3.12 vspipe
    # (pip vapoursynth) while system vspipe stays py3.14 — see denoiser docs.
    vspipe_exe = os.environ.get("VSPIPE") or shutil.which("vspipe")
    if not vspipe_exe or not os.path.exists(vspipe_exe):
        print("[svtav1-dispatch] Error: vspipe not found (PATH or $VSPIPE).")
        sys.exit(1)

    svt_cmd = [svt_exe, "-i", "stdin", "--progress", "2"] + shlex.split(svt_params.strip()) + ["-b", ivf_path]


    # Audio
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
        _callback = remote_callback or callback_address(remote_denoise)
        if _callback.startswith("100.") and not remote_callback:
            print(f"[svtav1-dispatch] Warning: streaming back over {_callback} "
                  "(tailscale) caps at ~1.5 Gbps; pass --remote-callback with "
                  "this host's LAN IP for the direct path.")
        print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
        run_remote_denoise(remote_denoise, remote_root, remote_python,
                           _callback, remote_port, input_file, _forward,
                           svt_cmd, temp_dir, stem)
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
            # still fails if libnvinfer isn't installed (e.g. encoder-host 2070S).
            # Only pick TRT when the library actually resolves.
            _has_nvinfer = False
            if "TensorrtExecutionProvider" in _ort_providers:
                import ctypes.util
                _has_nvinfer = ctypes.util.find_library("nvinfer") is not None
            if _has_nvinfer:
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
        run_piped([vspipe_exe, "-c", "y4m", vpy_path, "-"], _sink_cmd,
                  source_label="vspipe",
                  sink_label=("netstream send" if denoise_serve else "SvtAv1EncApp"),
                  source_stderr_log=os.path.join(temp_dir, f"{stem}_vspipe.log"))
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
                f"src = core.ffms2.Source(r'{input_file_fwd}', cachefile=r'{src_cachefile}')\n"
                f"if src.format.color_family == vs.RGB:\n"
                f"    # RGB sources (e.g. Lagarith/FFV1 eval intermediates) need an explicit\n"
                f"    # matrix for the YUV conversion; no chroma siting on RGB input.\n"
                f"    src = src.resize.Bicubic(format=vs.YUV420P10, matrix_s='709', chromaloc_s='left')\n"
                f"else:\n"
                f"    src = src.resize.Bicubic(format=vs.YUV420P10, chromaloc_in_s='left', chromaloc_s='left')\n"
                f"src.set_output()\n"
            )
        print(f"[svtav1-dispatch] vspipe (bicubic 422→420) | SvtAv1EncApp{svt_params}")
        print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
        sys.stdout.flush()
        run_piped([vspipe_exe, "-c", "y4m", src_vpy_path, "-"], svt_cmd,
                  source_label="vspipe")

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
    else:
        print("[svtav1-dispatch] Warning: could not verify frame count "
              "(ffprobe missing or stream unreadable).")

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
