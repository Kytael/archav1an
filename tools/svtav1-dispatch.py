#!/usr/bin/env python3
"""
svtav1-dispatch.py — Single-pass SvtAv1EncApp encode with Opus mux and SSIMU2 measurement.

Pipes ffmpeg → SvtAv1EncApp, muxes Opus audio, then measures SSIMU2 scores
(mean + 15th percentile) for comparison against av1an pipeline output.
"""

import os
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
    for p in ["/usr/local/lib/vapoursynth/libvstrt.so", "/usr/lib/vapoursynth/libvstrt.so"]:
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
                      use_smdegrain=False, tr=3, thsad=350, thsadc=450,
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
    # BSVD streaming + SMDegrain stacks many YUV444P16 intermediates; bump cache.
    _cache_mb = 4096 if use_bsvd_smdegrain else 1024
    vpy = (
        f'import sys as _sys; _sys.path.insert(0, {venv_site_pkgs!r})\n'
        f'from vstools import vs, core, initialize_clip, finalize_clip\n'
        f'core.max_cache_size = {_cache_mb}\n'
        f'\n'
        f'src = core.ffms2.Source(source=r{source!r}, cachefile=r{cachefile!r})\n'
        f'src = initialize_clip(src)\n'
        f'\n'
        f'{vsmlrt_import}'
        f'{denoise_lines}\n'
        f'\n'
        f'final = finalize_clip(src)\n'
        f'final.set_output(0)\n'
    )
    with open(vpy_path, "w") as f:
        f.write(vpy)


# ---------------------------------------------------------------------------
# Encode helpers
# ---------------------------------------------------------------------------

def run_piped(source_cmd, sink_cmd, source_label="source",
              source_stderr_log=None, suppress_sink_stderr=False):
    """Run source_cmd | sink_cmd, forwarding Ctrl+C to both processes.

    A nonzero exit from either process is fatal (prevents muxing a truncated
    encode as success). source_stderr_log, if given, captures the source's
    stderr to that file; its tail is printed when the source fails.
    """
    log_fh = open(source_stderr_log, "w") if source_stderr_log else None
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
    if source_proc.returncode not in (0, None):
        print(f"[svtav1-dispatch] Error: {source_label} exited with {source_proc.returncode}; aborting before mux.")
        if source_stderr_log:
            try:
                with open(source_stderr_log) as lf:
                    tail = lf.readlines()[-30:]
                if tail:
                    print(f"[svtav1-dispatch] --- {source_label} stderr tail ({source_stderr_log}) ---")
                    sys.stdout.write("".join(tail))
            except OSError:
                pass
        sys.exit(source_proc.returncode)
    if sink_proc.returncode != 0:
        print(f"[svtav1-dispatch] Error: SvtAv1EncApp exited with {sink_proc.returncode}")
        sys.exit(sink_proc.returncode)


# ---------------------------------------------------------------------------
# Audio helpers (shared with av1an-dispatch.py)
# ---------------------------------------------------------------------------

def get_audio_channels(input_file):
    """Detect audio channel count via ffprobe. Returns int (default 2)."""
    ffprobe_exe = shutil.which("ffprobe")
    if not ffprobe_exe:
        return 2
    try:
        result = subprocess.run(
            [ffprobe_exe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels", "-of", "csv=p=0", input_file],
            capture_output=True, text=True,
        )
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
    denoise_thsadc = 450
    denoise_bsvd = False
    denoise_bsvd_smdegrain = False
    _default_bsvd_onnx = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "bsvd_ft_ep5_stateful_v2_dyn_fp16.onnx")
    _default_bsvd_sigma_estimator = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "bsvd_sigma_estimator_v3.pth")
    denoise_bsvd_onnx = _default_bsvd_onnx
    denoise_bsvd_sigma = "auto"
    denoise_bsvd_device = 0
    denoise_bsvd_warmup = 30

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
        elif arg == "--denoise-thsadc":
            denoise_thsadc = int(nextval() or 450); i += 2
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
        else:
            i += 1

    if not input_file or not output_file:
        print("[svtav1-dispatch] Error: -i and -o are required.")
        sys.exit(1)

    svt_exe = shutil.which("SvtAv1EncApp")
    ffmpeg_exe = shutil.which("ffmpeg")
    if not svt_exe:
        print("[svtav1-dispatch] Error: SvtAv1EncApp not found in PATH.")
        sys.exit(1)
    if not ffmpeg_exe:
        print("[svtav1-dispatch] Error: ffmpeg not found in PATH.")
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



    vspipe_exe = shutil.which("vspipe")
    if not vspipe_exe:
        print("[svtav1-dispatch] Error: vspipe not found in PATH.")
        sys.exit(1)

    svt_cmd = [svt_exe, "-i", "stdin", "--progress", "2"] + shlex.split(svt_params.strip()) + ["-b", ivf_path]


    # Audio
    if no_opus:
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
    if any(_denoise_flags):
        if denoise_bsvd or denoise_bsvd_smdegrain:
            if not os.path.exists(denoise_bsvd_onnx):
                print(f"[svtav1-dispatch] Error: BSVD ONNX not found at {denoise_bsvd_onnx}. "
                      "Stage it via setup.sh --install denoiser or pass --bsvd-onnx.")
                sys.exit(1)
            # EP detection: prefer NVIDIA (TRT EP) if the vstrt plugin is present,
            # else AMD (MIGraphX EP). Both rely on Python onnxruntime providers.
            mlrt_plugin = find_mlrt_plugin()
            bsvd_ep = "TRT" if (mlrt_plugin and "vstrt" in mlrt_plugin) else "MIGRAPHX"
            # σ resolution
            if str(denoise_bsvd_sigma).lower() == "auto":
                if not os.path.exists(_default_bsvd_sigma_estimator):
                    print(f"[svtav1-dispatch] Error: --bsvd-sigma=auto needs {_default_bsvd_sigma_estimator}; "
                          "stage via setup.sh --install denoiser or pass --bsvd-sigma <float>.")
                    sys.exit(1)
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from bsvd_optsig import compute_sigma_for_video
                bsvd_sigma_val = compute_sigma_for_video(
                    input_file, sigma_estimator_pth=_default_bsvd_sigma_estimator,
                    warmup_frames=denoise_bsvd_warmup,
                    device=f"cuda:{denoise_bsvd_device}")
            else:
                bsvd_sigma_val = float(denoise_bsvd_sigma)
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
        vpy_path = os.path.join(temp_dir, f"{stem}_denoise.vpy")
        cachefile = os.path.join(temp_dir, f"{stem}.ffindex")
        write_denoise_vpy(vpy_path, input_file, cachefile,
                          denoise_model, denoise_tile, denoise_streams,
                          use_smdegrain=denoise_smdegrain,
                          tr=denoise_tr, thsad=denoise_thsad, thsadc=denoise_thsadc,
                          use_rvrt=denoise_rvrt, rvrt_sigma=denoise_rvrt_sigma,
                          use_stasunet=denoise_stasunet, stasunet_engine=denoise_stasunet_engine,
                          stasunet_pre_darken_ev=denoise_stasunet_pre_darken_ev,
                          use_bsvd=denoise_bsvd, use_bsvd_smdegrain=denoise_bsvd_smdegrain,
                          bsvd_onnx=denoise_bsvd_onnx,
                          bsvd_sigma=(bsvd_sigma_val if (denoise_bsvd or denoise_bsvd_smdegrain) else 0.08),
                          bsvd_ep=(bsvd_ep if (denoise_bsvd or denoise_bsvd_smdegrain) else "TRT"),
                          bsvd_device=denoise_bsvd_device)
        if denoise_bsvd_smdegrain:
            print(f"[svtav1-dispatch] vspipe ({_backend_name} + SMDegrain tr={denoise_tr} thSAD={denoise_thsad}, σ={bsvd_sigma_val:.3f}) | SvtAv1EncApp{svt_params}")
        elif denoise_bsvd:
            print(f"[svtav1-dispatch] vspipe ({_backend_name} σ={bsvd_sigma_val:.3f}) | SvtAv1EncApp{svt_params}")
        elif denoise_rvrt:
            print(f"[svtav1-dispatch] vspipe (RVRT sigma={denoise_rvrt_sigma}, tile={denoise_tile}) | SvtAv1EncApp{svt_params}")
        elif denoise_stasunet:
            print(f"[svtav1-dispatch] vspipe (STA-SUNet engine={os.path.basename(denoise_stasunet_engine)}, tile={denoise_tile}, streams={denoise_streams}) | SvtAv1EncApp{svt_params}")
        elif denoise_smdegrain:
            print(f"[svtav1-dispatch] vspipe ({_backend_name} SCUNet+SMDegrain tr={denoise_tr} thSAD={denoise_thsad}, tile={denoise_tile}, streams={denoise_streams}) | SvtAv1EncApp{svt_params}")
        else:
            print(f"[svtav1-dispatch] vspipe ({_backend_name} SCUNet-{denoise_model}, tile={denoise_tile}, streams={denoise_streams}) | SvtAv1EncApp{svt_params}")
        print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
        sys.stdout.flush()
        run_piped([vspipe_exe, "-c", "y4m", vpy_path, "-"], svt_cmd,
                  source_label="vspipe",
                  source_stderr_log=os.path.join(temp_dir, f"{stem}_vspipe.log"))
    else:
        src_vpy_path = os.path.join(temp_dir, f"{stem}_src.vpy")
        src_cachefile = os.path.join(temp_dir, f"{stem}.ffindex")
        input_file_fwd = os.path.abspath(input_file).replace("\\", "/")
        with open(src_vpy_path, "w", encoding="utf-8") as vf:
            vf.write(
                f"import vapoursynth as vs\n"
                f"core = vs.core\n"
                f"src = core.ffms2.Source(r'{input_file_fwd}', cachefile=r'{src_cachefile}')\n"
                f"src = src.resize.Bicubic(format=vs.YUV420P10, chromaloc_in_s='left', chromaloc_s='left')\n"
                f"src.set_output()\n"
            )
        print(f"[svtav1-dispatch] vspipe (bicubic 422→420) | SvtAv1EncApp{svt_params}")
        print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
        sys.stdout.flush()
        run_piped([vspipe_exe, "-c", "y4m", src_vpy_path, "-"], svt_cmd,
                  source_label="vspipe")

    # --- Mux ---
    print("[svtav1-dispatch] Muxing...")
    audio_codec = ["-c:a", "copy"] if no_opus else ["-c:a", "libopus", "-b:a", opus_bitrate]
    mux_cmd = [
        ffmpeg_exe, "-y",
        "-i", ivf_path, "-i", input_file,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", *audio_codec,
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
                os.path.join(temp_dir, f"{stem}_src.vpy")):
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
