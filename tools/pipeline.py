#!/usr/bin/env python3
"""
Automated Encoding Pipeline — batch AV1+Opus encoder with recursive directory support.

Replaces the loop logic in run_linux_*.sh scripts. Discovers video files recursively
under Input/, encodes to AV1 via dispatch.py, muxes audio, and mirrors the folder
structure into Output/.

Usage:
    source activate-venv.sh
    python3 tools/pipeline.py --preset live-crf32
    python3 tools/pipeline.py --preset anime-crf32 --no-opus
    python3 tools/pipeline.py --preset live-crf32 --workers 6
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

INPUT_DIR = ROOT_DIR / "Input"
OUTPUT_DIR = ROOT_DIR / "Output"
TEMP_DIR = ROOT_DIR / "Temp"

SCENE_DETECT_SCRIPT = SCRIPT_DIR / "Progressive-Scene-Detection.py"
DISPATCH_SCRIPT = SCRIPT_DIR / "dispatch.py"
MUX_SCRIPT = SCRIPT_DIR / "mux.py"
TAG_SCRIPT = SCRIPT_DIR / "tag.py"
CLEANUP_SCRIPT = SCRIPT_DIR / "cleanup.py"

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m2ts", ".mov"}

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
MKVMERGE = shutil.which("mkvmerge") or "mkvmerge"

# ---------------------------------------------------------------------------
# Presets — extracted from the 9 run_linux_*.sh scripts
# ---------------------------------------------------------------------------
PRESETS = {
    "live-crf15": {
        "quality": 15,
        "photon_noise": 4,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": True,
        "aggressive": True,
        "fast_params": "--ac-bias 1.0 --complex-hvs 1 --keyint -1 --variance-boost-strength 2 --enable-dlf 2 --luminance-qp-bias 20 --tune 3",
        "final_params": "--ac-bias 1.0 --complex-hvs 1 --keyint -1 --variance-boost-strength 2 --enable-dlf 2 --luminance-qp-bias 20 --tune 3 --lp 3",
    },
    "live-crf18": {
        "quality": 18,
        "photon_noise": 4,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": True,
        "aggressive": False,
        "fast_params": "--lp 3 --tune 3 --hbd-mds 0 --keyint 305 --ac-bias 1.2 --filtering-noise-detection 1",
        "final_params": "--lp 3 --tune 3 --hbd-mds 1 --keyint 305 --ac-bias 1.2 --filtering-noise-detection 1",
    },
    "live-crf25": {
        "quality": 25,
        "photon_noise": 4,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": True,
        "aggressive": False,
        "fast_params": "--lp 3 --tune 3 --hbd-mds 0 --keyint 305 --ac-bias 1.0 --filtering-noise-detection 4",
        "final_params": "--lp 3 --tune 3 --hbd-mds 1 --keyint 305 --ac-bias 1.0 --filtering-noise-detection 4",
    },
    "live-crf32": {
        "quality": 32,
        "photon_noise": 4,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": True,
        "aggressive": False,
        "fast_params": "--lp 3 --tune 3 --hbd-mds 0 --keyint 305 --ac-bias 0.8 --filtering-noise-detection 4",
        "final_params": "--lp 3 --tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --filtering-noise-detection 4",
    },
    "anime-crf15": {
        "quality": 15,
        "photon_noise": 2,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": False,
        "aggressive": True,
        "fast_params": "--ac-bias 1.0 --complex-hvs 1 --keyint -1 --variance-boost-strength 1 --enable-dlf 2 --noise-level-thr 16000 --variance-md-bias 1 --cdef-bias 1 --luminance-qp-bias 20 --qm-min 8 --keyint -1 --tune 0 --balancing-q-bias 1 --chroma-qmc-bias 2 --filtering-noise-detection 4 --balancing-r0-based-layer-offset 0 --chroma-qm-min 10",
        "final_params": "--ac-bias 1.0 --complex-hvs 1 --keyint -1 --variance-boost-strength 1 --enable-dlf 2 --noise-level-thr 16000 --variance-md-bias 1 --cdef-bias 1 --luminance-qp-bias 20 --qm-min 8 --keyint -1 --tune 0 --balancing-q-bias 1 --chroma-qmc-bias 2 --filtering-noise-detection 4 --balancing-r0-based-layer-offset 0 --chroma-qm-min 10 --lp 3",
    },
    "anime-crf18": {
        "quality": 18,
        "photon_noise": 2,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": False,
        "aggressive": False,
        "fast_params": "--lp 3 --tune 0 --hbd-mds 0 --keyint 305 --noise-level-thr 16000 --lineart-psy-bias 5 --texture-psy-bias 4 --filtering-noise-detection 1",
        "final_params": "--lp 3 --tune 0 --hbd-mds 1 --keyint 305 --noise-level-thr 16000 --lineart-psy-bias 5 --texture-psy-bias 4 --filtering-noise-detection 1",
    },
    "anime-crf25": {
        "quality": 25,
        "photon_noise": 2,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": False,
        "aggressive": False,
        "fast_params": "--lp 3 --tune 0 --hbd-mds 0 --keyint 305 --noise-level-thr 16000 --lineart-psy-bias 4 --texture-psy-bias 2 --filtering-noise-detection 4",
        "final_params": "--lp 3 --tune 0 --hbd-mds 1 --keyint 305 --noise-level-thr 16000 --lineart-psy-bias 4 --texture-psy-bias 2 --filtering-noise-detection 4",
    },
    "anime-crf32": {
        "quality": 32,
        "photon_noise": 2,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": False,
        "aggressive": False,
        "fast_params": "--lp 3 --tune 0 --hbd-mds 0 --keyint 305 --noise-level-thr 16000 --lineart-psy-bias 4 --texture-psy-bias 2 --filtering-noise-detection 4",
        "final_params": "--lp 3 --tune 0 --hbd-mds 1 --keyint 305 --noise-level-thr 16000 --lineart-psy-bias 4 --texture-psy-bias 2 --filtering-noise-detection 4",
    },
    "dance-crf27": {
        "quality": 27,
        "photon_noise": 6,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": True,
        "aggressive": False,
        "fast_params": "--lp 3 --tune 3 --hbd-mds 0 --keyint 305 --ac-bias 0.8 --sharp-tx 1 --sharpness 1 --tf-strength 2 --variance-boost-strength 2 --variance-octile 5 --enable-dlf 2 --filtering-noise-detection 4",
        "final_params": "--lp 3 --tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --sharp-tx 1 --sharpness 1 --tf-strength 2 --variance-boost-strength 2 --variance-octile 5 --enable-dlf 2 --filtering-noise-detection 4",
    },
    "sports-crf27": {
        "quality": 27,
        "photon_noise": 6,
        "fast_speed": 8,
        "final_speed": 4,
        "autocrop": True,
        "aggressive": False,
        "fast_params": "--lp 3 --tune 3 --hbd-mds 0 --keyint 305 --ac-bias 0.6 --sharp-tx 0 --sharpness 1 --tf-strength 3 --variance-boost-strength 1 --variance-octile 7 --enable-dlf 2",
        "final_params": "--lp 3 --tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.6 --sharp-tx 0 --sharpness 1 --tf-strength 3 --variance-boost-strength 1 --variance-octile 7 --enable-dlf 2",
    },
}

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
def _handle_sigint(signum, frame):
    print("\nInterrupted.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_ssimu2_config():
    """Load SSIMU2 tool and worker count from config, generating if needed."""
    config_file = ROOT_DIR / "tools" / "workercount-ssimu2.txt"

    if not config_file.exists():
        print("First Run Detected: Calculating optimal SSIMU2 settings...")
        subprocess.run(
            [sys.executable, str(ROOT_DIR / "tools" / "ssimu2-workercount.py")],
            cwd=str(ROOT_DIR),
        )

    tool = "vs-zip"
    workers = "4"

    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("tool="):
                tool = line.split("=", 1)[1].strip()
            elif line.startswith("workercount="):
                workers = line.split("=", 1)[1].strip()

    return tool, workers


def discover_videos(input_dir):
    """Recursively find all video files under input_dir, sorted by path."""
    videos = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(Path(root) / f)
    return sorted(videos)


def relative_subpath(source, input_dir):
    """Return the subdirectory path relative to Input/ (excluding the filename)."""
    return source.parent.relative_to(input_dir)


def get_audio_channels(source):
    """Detect audio channel count via ffprobe. Returns int."""
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels", "-of", "csv=p=0",
             str(source)],
            capture_output=True, text=True, encoding="utf-8",
        )
        return int(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 2


def opus_bitrate_for_channels(channels):
    """Select Opus bitrate based on channel count (matches opus.py defaults)."""
    if channels > 6:
        return "320k"
    elif channels >= 6:
        return "256k"
    elif channels >= 3:
        return "192k"
    return "128k"


def format_duration(seconds):
    """Format seconds as Xh Ym."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def format_size(nbytes):
    """Format byte count as human-readable."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    elif nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    return f"{nbytes / (1 << 10):.1f} KB"


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------


def process_file(source, preset, ssimu2_tool, ssimu2_workers, worker_count, no_opus):
    """Run the full encode pipeline for a single source file. Returns True on success."""
    stem = source.stem
    subpath = relative_subpath(source, INPUT_DIR)

    # Paths
    temp_dir = TEMP_DIR / subpath / stem
    scene_file = temp_dir / f"{stem}_scenedetect.json"
    vid_only = temp_dir / f"{stem}-av1.mkv"
    output_parent = OUTPUT_DIR / subpath
    output_file = output_parent / f"{stem}-av1.mkv"

    # 1. Skip if output already exists
    if output_file.exists():
        print(f"[SKIP] Already encoded: {output_file.relative_to(ROOT_DIR)}")
        return True

    # 2. Create directories
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_parent.mkdir(parents=True, exist_ok=True)

    p = preset
    source_str = str(source)
    scene_str = str(scene_file)

    # 3. Scene detection (skip if JSON already exists)
    if scene_file.exists():
        print(f"\n{'='*79}")
        print(f"Scene JSON found for \"{source.name}\" — skipping scene detection.")
        print(f"{'='*79}")
    else:
        print(f"\n{'='*79}")
        print(f"Detecting scenes for \"{source.name}\"...")
        print(f"{'='*79}")
        ret = subprocess.run(
            [sys.executable, str(SCENE_DETECT_SCRIPT),
             "-i", source_str,
             "-o", scene_str,
             "--temp", str(temp_dir)],
            cwd=str(ROOT_DIR),
        )
        if ret.returncode != 0:
            print(f"[ERROR] Scene detection failed for {source.name}")
            return False

    # 4. Encode via dispatch.py → video-only AV1 in Temp/
    #    Clean stale av1an temp dirs (hidden .XXXX dirs) to avoid resume failures
    for item in temp_dir.iterdir():
        if item.is_dir() and item.name.startswith(".") and item.name != ".":
            try:
                shutil.rmtree(item)
            except OSError:
                pass

    print(f"\n{'='*79}")
    print(f"Encoding \"{source.name}\"...")
    print(f"{'='*79}")

    dispatch_cmd = [
        sys.executable, str(DISPATCH_SCRIPT),
        "-i", source_str,
        "-o", str(vid_only),
        "--scenes", scene_str,
        "--temp", str(temp_dir),
        "--quality", str(p["quality"]),
        "--ssimu2", ssimu2_tool,
        "--ssimu2-cpu-workers", ssimu2_workers,
        "--resume",
        "--verbose",
        "--photon-noise", str(p["photon_noise"]),
        "--workers", str(worker_count),
        "--fast-speed", str(p["fast_speed"]),
        "--final-speed", str(p["final_speed"]),
        "--fast-params", p["fast_params"],
        "--final-params", p["final_params"],
    ]
    if p.get("autocrop"):
        dispatch_cmd.append("--autocrop")
    if p.get("aggressive"):
        dispatch_cmd.append("--aggressive")

    ret = subprocess.run(dispatch_cmd, cwd=str(ROOT_DIR))
    if ret.returncode != 0:
        print(f"[ERROR] Encoding failed for {source.name}")
        return False

    if not vid_only.exists():
        print(f"[ERROR] Expected video-only output not found: {vid_only}")
        return False

    # 5. Combine video + audio
    if no_opus:
        # Passthrough audio via mkvmerge
        print(f"Muxing (passthrough audio): {source.name}")
        mux_cmd = [
            MKVMERGE, "-o", str(output_file),
            "--no-audio", str(vid_only),
            "--no-video", source_str,
        ]
        ret = subprocess.run(mux_cmd, cwd=str(ROOT_DIR))
        if ret.returncode not in (0, 1):  # mkvmerge returns 1 for warnings
            print(f"[ERROR] Muxing failed for {source.name}")
            return False
    else:
        # Re-encode audio to Opus via ffmpeg
        channels = get_audio_channels(source)
        bitrate = opus_bitrate_for_channels(channels)
        print(f"Muxing (Opus {bitrate} {channels}ch): {source.name}")
        mux_cmd = [
            FFMPEG, "-y",
            "-i", str(vid_only),
            "-i", source_str,
            "-map", "0:v",
            "-map", "1:a?",
            "-c:v", "copy",
            "-c:a", "libopus",
            "-b:a", bitrate,
            str(output_file),
        ]
        ret = subprocess.run(mux_cmd, cwd=str(ROOT_DIR))
        if ret.returncode != 0:
            print(f"[ERROR] Audio muxing failed for {source.name}")
            return False

    if not output_file.exists():
        print(f"[ERROR] Output file was not created: {output_file}")
        return False

    # 6. Preserve source modification time
    src_stat = source.stat()
    os.utime(output_file, (src_stat.st_atime, src_stat.st_mtime))

    # 7. Clean up temp dir for this file
    try:
        shutil.rmtree(temp_dir)
    except OSError as e:
        print(f"[WARN] Could not clean up temp dir {temp_dir}: {e}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(
        description="Automated AV1 encoding pipeline with recursive directory support"
    )
    parser.add_argument(
        "--preset", required=True, choices=sorted(PRESETS.keys()),
        help="Encoding preset name",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Number of av1an workers (default: 6)",
    )
    parser.add_argument(
        "--no-opus", action="store_true",
        help="Passthrough audio instead of re-encoding to Opus",
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]

    # Ensure directories exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load SSIMU2 config
    ssimu2_tool, ssimu2_workers = load_ssimu2_config()

    print(f"Pipeline: {args.preset} | Workers: {args.workers}")
    print(f"SSIMU2 Mode: {ssimu2_tool} | SSIMU2 Workers: {ssimu2_workers}")

    # Create tagging marker (for tag.py compatibility)
    preset_to_script = {
        "live-crf15": "run_linux_live_crf15.sh",
        "live-crf18": "run_linux_live_crf18.sh",
        "live-crf25": "run_linux_live_crf25.sh",
        "live-crf32": "run_linux_live_crf32.sh",
        "anime-crf15": "run_linux_anime_crf15.sh",
        "anime-crf18": "run_linux_anime_crf18.sh",
        "anime-crf25": "run_linux_anime_crf25.sh",
        "anime-crf32": "run_linux_anime_crf32.sh",
        "sports-crf27": "run_linux_sports_crf27.sh",
    }
    marker_name = preset_to_script.get(args.preset, f"pipeline-{args.preset}")
    (SCRIPT_DIR / f"sh-used-{marker_name}.txt").touch()

    # Discover videos
    videos = discover_videos(INPUT_DIR)
    if not videos:
        print("No video files found in Input/. Nothing to do.")
        return

    print(f"Found {len(videos)} video file(s) in Input/\n")

    # Process each file
    start_time = time.monotonic()
    success_count = 0
    total = len(videos)
    total_input_size = 0
    total_output_size = 0

    for i, source in enumerate(videos, 1):
        print(f"\n{'#'*79}")
        print(f"# [{i}/{total}] {source.relative_to(INPUT_DIR)}")
        print(f"{'#'*79}")

        total_input_size += source.stat().st_size

        ok = process_file(
            source, preset, ssimu2_tool, ssimu2_workers,
            args.workers, args.no_opus,
        )
        if ok:
            success_count += 1
            subpath = relative_subpath(source, INPUT_DIR)
            out = OUTPUT_DIR / subpath / f"{source.stem}-av1.mkv"
            if out.exists():
                total_output_size += out.stat().st_size

    # Post-processing: tagging and cleanup
    if success_count > 0:
        print(f"\n{'='*79}")
        print("Tagging output files...")
        subprocess.run([sys.executable, str(TAG_SCRIPT)], cwd=str(ROOT_DIR))

        print("Cleaning up temporary files and folders...")
        subprocess.run([sys.executable, str(CLEANUP_SCRIPT)], cwd=str(ROOT_DIR))

    # Summary report
    elapsed = time.monotonic() - start_time
    print(f"\n{'='*79}")
    print("Pipeline complete:")
    print(f"  Files processed: {success_count} / {total}")
    print(f"  Total time: {format_duration(elapsed)}")
    print(f"  Input size:  {format_size(total_input_size)}")
    if total_output_size > 0:
        reduction = (1 - total_output_size / total_input_size) * 100 if total_input_size else 0
        print(f"  Output size: {format_size(total_output_size)} ({reduction:.1f}% reduction)")
    print(f"{'='*79}")


if __name__ == "__main__":
    main()
