#!/usr/bin/env python3
"""
svtav1-dispatch.py — Single-pass SvtAv1EncApp encode with Opus mux and SSIMU2 measurement.

Pipes ffmpeg → SvtAv1EncApp, muxes Opus audio, then measures SSIMU2 scores
(mean + 15th percentile) for comparison against av1an pipeline output.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import textwrap


# ---------------------------------------------------------------------------
# Denoise helpers
# ---------------------------------------------------------------------------

def find_migx_plugin():
    for p in ["/usr/local/lib/vapoursynth/libvsmigx.so", "/usr/lib/vapoursynth/libvsmigx.so"]:
        if os.path.exists(p):
            return p
    return ""

def write_denoise_vpy(vpy_path, source, cachefile, migx_plugin, strength, tile, streams):
    vpy = textwrap.dedent(f"""
        from vstools import vs, core, initialize_clip, finalize_clip
        core.max_cache_size = 1024

        import os as _os
        _migx_so = {migx_plugin!r}
        if not _migx_so or not _os.path.exists(_migx_so):
            raise RuntimeError("--denoise-scunet: libvsmigx.so not found. Run setup/denoiser.sh.")
        core.std.LoadPlugin(_migx_so)

        src = core.ffms2.Source(source=r{source!r}, cachefile=r{cachefile!r})
        src = initialize_clip(src)

        from vsmlrt import SCUNet as _SCUNet, SCUNetModel as _SCUNetModel, Backend as _Backend
        _src_fmt = src.format
        _rgb = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")
        _rgb = _SCUNet(_rgb,
                       model={{15: _SCUNetModel.scunet_color_15, 25: _SCUNetModel.scunet_color_25,
                               50: _SCUNetModel.scunet_color_50}}.get({strength}, _SCUNetModel.scunet_color_15),
                       tilesize={tile}, overlap=8,
                       backend=_Backend.MIGX(device_id=0, fp16=True, exhaustive_tune=False, num_streams={streams}))
        src = core.resize.Bicubic(_rgb, format=_src_fmt, matrix_s="709")

        final = finalize_clip(src)
        final.set_output(0)
    """).lstrip()
    with open(vpy_path, "w") as f:
        f.write(vpy)


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


def measure_ssimu2(source_file, encoded_file, tool):
    """
    Runs SSIMU2 comparison in a subprocess VapourSynth script.
    Returns (mean, p15) as floats, or (None, None) on failure.
    """
    # numStream=4 controls internal GPU parallelism within one measurement.
    # This is separate from workercount (concurrent processes in the pipeline).
    if tool == "vs-hip":
        vs_script = f"""
import vapoursynth as vs
from vstools import clip_async_render
core = vs.core
src = core.ffms2.Source(source=r"{source_file}").resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
enc = core.ffms2.Source(source=r"{encoded_file}").resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
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
src = core.ffms2.Source(source=r"{source_file}").resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
enc = core.ffms2.Source(source=r"{encoded_file}").resize.Bicubic(format=vs.RGB24, matrix_in_s="709")
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
        if result.stderr:
            for line in result.stderr.splitlines():
                if line.strip() and "Warning" not in line and "Info" not in line:
                    pass  # suppress VS noise

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
    denoise_strength = 15
    denoise_tile = 256
    denoise_streams = 2

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
        elif arg == "--denoise-strength":
            denoise_strength = int(nextval() or 15); i += 2
        elif arg == "--denoise-tile":
            denoise_tile = int(nextval() or 256); i += 2
        elif arg == "--denoise-streams":
            denoise_streams = int(nextval() or 2); i += 2
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

    import shlex
    svt_cmd = [svt_exe, "-i", "stdin"] + shlex.split(svt_params.strip()) + ["-b", ivf_path]

    # Audio
    if no_opus:
        print("[svtav1-dispatch] Audio: passthrough (--no-opus)")
    else:
        channels = get_audio_channels(input_file)
        opus_bitrate = opus_bitrate_for_channels(channels)
        print(f"[svtav1-dispatch] Audio: Opus {opus_bitrate} ({channels}ch)")

    # Preserve source mtime
    src_stat = os.stat(input_file) if os.path.exists(input_file) else None

    # --- Encode ---
    if denoise_scunet:
        migx_plugin = find_migx_plugin()
        if not migx_plugin:
            print("[svtav1-dispatch] Error: --denoise-scunet: libvsmigx.so not found. Run setup/denoiser.sh.")
            sys.exit(1)
        vspipe_exe = shutil.which("vspipe")
        if not vspipe_exe:
            print("[svtav1-dispatch] Error: vspipe not found in PATH.")
            sys.exit(1)
        vpy_path = os.path.join(temp_dir, f"{stem}_denoise.vpy")
        cachefile = os.path.join(temp_dir, f"{stem}.ffindex")
        write_denoise_vpy(vpy_path, input_file, cachefile, migx_plugin,
                          denoise_strength, denoise_tile, denoise_streams)
        print(f"[svtav1-dispatch] vspipe (MIGraphX SCUNet-{denoise_strength}, tile={denoise_tile}, streams={denoise_streams}) | SvtAv1EncApp{svt_params}")
        print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
        try:
            sys.stdout.flush()
            vspipe_proc = subprocess.Popen([vspipe_exe, vpy_path, "--"], stdout=subprocess.PIPE)
            svt_proc = subprocess.Popen(svt_cmd, stdin=vspipe_proc.stdout)
            vspipe_proc.stdout.close()
            svt_proc.wait()
            vspipe_proc.wait()
            if svt_proc.returncode != 0:
                print(f"[svtav1-dispatch] Error: SvtAv1EncApp exited with {svt_proc.returncode}")
                sys.exit(svt_proc.returncode)
            if vspipe_proc.returncode != 0:
                print(f"[svtav1-dispatch] Warning: vspipe exited with {vspipe_proc.returncode}")
        except Exception as e:
            print(f"[svtav1-dispatch] Encode failed: {e}")
            sys.exit(1)
    else:
        ffmpeg_cmd = [
            ffmpeg_exe, "-y", "-i", input_file,
            "-an", "-f", "yuv4mpegpipe", "-strict", "-1",
            "-pix_fmt", "yuv420p10le", "-",
        ]
        print(f"[svtav1-dispatch] ffmpeg | SvtAv1EncApp{svt_params}")
        print(f"[svtav1-dispatch] Output IVF: {ivf_path}")
        try:
            sys.stdout.flush()
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            svt_proc = subprocess.Popen(svt_cmd, stdin=ffmpeg_proc.stdout)
            ffmpeg_proc.stdout.close()
            svt_proc.wait()
            ffmpeg_proc.wait()
            if svt_proc.returncode != 0:
                print(f"[svtav1-dispatch] Error: SvtAv1EncApp exited with {svt_proc.returncode}")
                sys.exit(svt_proc.returncode)
            if ffmpeg_proc.returncode not in (0, None):
                print(f"[svtav1-dispatch] Warning: ffmpeg exited with {ffmpeg_proc.returncode}")
        except Exception as e:
            print(f"[svtav1-dispatch] Encode failed: {e}")
            sys.exit(1)

    # --- Mux ---
    print("[svtav1-dispatch] Muxing...")
    if no_opus:
        mux_cmd = [
            ffmpeg_exe, "-y",
            "-i", ivf_path, "-i", input_file,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "copy",
            output_file,
        ]
    else:
        mux_cmd = [
            ffmpeg_exe, "-y",
            "-i", ivf_path, "-i", input_file,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "libopus", "-b:a", opus_bitrate,
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
        mean, p15 = measure_ssimu2(input_file, output_file, ssimu2_tool)
        if mean is not None:
            print(f"[svtav1-dispatch] SSIMU2  mean: {mean:.2f} | p15: {p15:.2f}")
        else:
            print("[svtav1-dispatch] SSIMU2 measurement failed.")

    # --- Preserve mtime ---
    if src_stat and os.path.exists(output_file):
        os.utime(output_file, (src_stat.st_atime, src_stat.st_mtime))

    # --- Cleanup temp IVF ---
    try:
        os.remove(ivf_path)
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
