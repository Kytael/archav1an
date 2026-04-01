#!/usr/bin/env python3
"""
av1an-dispatch.py — Lightweight dispatch for single-pass av1an batch encoding.

Similar to dispatch.py but calls av1an directly instead of Auto-Boost-Av1an.py.
Handles color space detection (BT.709/BT.601) and injects appropriate flags.
Re-encodes audio to Opus and preserves source modification time.
"""

import sys
import subprocess
import os
import shutil


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
    """Select Opus bitrate based on channel count."""
    if channels > 6:
        return "320k"
    elif channels >= 6:
        return "256k"
    elif channels >= 3:
        return "192k"
    return "128k"


def main():
    script_path = os.path.abspath(__file__)
    tools_dir = os.path.dirname(script_path)
    root_dir = os.path.dirname(tools_dir)

    # Locate av1an and mediainfo
    av1an_exe = shutil.which("av1an")
    if not av1an_exe:
        print("[av1an-dispatch] Error: av1an not found in PATH.")
        sys.exit(1)

    mediainfo_exe = shutil.which("mediainfo")

    # --- Argument Parsing ---
    args = sys.argv[1:]
    input_file = None
    output_file = None

    for idx, arg in enumerate(args):
        if arg in ("-i", "--input") and idx + 1 < len(args):
            input_file = args[idx + 1]
        elif arg in ("-o", "--output") and idx + 1 < len(args):
            output_file = args[idx + 1]

    # --- Color Space Detection via MediaInfo ---
    is_bt709 = False
    is_bt601 = False

    f_prim_709 = f_trans_709 = f_mat_709 = False
    f_prim_601 = f_trans_601 = f_mat_601 = False

    if input_file and os.path.exists(input_file):
        if mediainfo_exe:
            try:
                result = subprocess.run(
                    [mediainfo_exe, input_file],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                if result.returncode == 0:
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

                    if f_prim_709 and f_trans_709 and f_mat_709:
                        is_bt709 = True
                        print("[av1an-dispatch] MediaInfo confirmed full BT.709 source.")
                    elif f_prim_601 and f_trans_601 and f_mat_601:
                        is_bt601 = True
                        print("[av1an-dispatch] MediaInfo confirmed full BT.601 source.")
                    else:
                        print(
                            f"[av1an-dispatch] MediaInfo results - 709: ({f_prim_709},{f_trans_709},{f_mat_709}) | "
                            f"601: ({f_prim_601},{f_trans_601},{f_mat_601}). No standard color match."
                        )
            except Exception as e:
                print(f"[av1an-dispatch] Warning: MediaInfo execution failed: {e}")
        else:
            print("[av1an-dispatch] Warning: mediainfo not found in PATH.")
            print(
                "[av1an-dispatch] Install with: sudo pacman -S mediainfo (Arch) or sudo apt install mediainfo (Ubuntu)"
            )

    # --- Color flags to inject into encoder params ---
    bt709_flags = " --color-primaries 1 --transfer-characteristics 1 --matrix-coefficients 1"
    bt601_flags = " --color-primaries 6 --transfer-characteristics 6 --matrix-coefficients 6"

    current_flags = ""
    if is_bt709:
        print("[av1an-dispatch] Injecting BT.709 parameters.")
        current_flags = bt709_flags
    elif is_bt601:
        print("[av1an-dispatch] Injecting BT.601 parameters.")
        current_flags = bt601_flags

    # --- Build av1an command ---
    final_cmd = [av1an_exe]

    # Translate our args to av1an args
    skip_next = False
    quality = None
    workers = None
    speed = None
    photon_noise = None
    encoder_params = ""
    no_opus = False

    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("-i", "--input"):
            skip_next = True
            continue
        elif arg in ("-o", "--output"):
            skip_next = True
            continue
        elif arg in ("--quality",):
            quality = args[idx + 1] if idx + 1 < len(args) else None
            skip_next = True
        elif arg in ("--workers",):
            workers = args[idx + 1] if idx + 1 < len(args) else None
            skip_next = True
        elif arg in ("--final-speed",):
            speed = args[idx + 1] if idx + 1 < len(args) else None
            skip_next = True
        elif arg in ("--photon-noise",):
            photon_noise = args[idx + 1] if idx + 1 < len(args) else None
            skip_next = True
        elif arg in ("--final-params",):
            encoder_params = args[idx + 1] if idx + 1 < len(args) else ""
            skip_next = True
        elif arg == "--no-opus":
            no_opus = True

    # Append color flags to encoder params
    if current_flags:
        encoder_params += current_flags

    # --- Audio params ---
    if no_opus:
        print("[av1an-dispatch] Audio: passthrough (--no-opus)")
    else:
        channels = get_audio_channels(input_file) if input_file else 2
        opus_bitrate = opus_bitrate_for_channels(channels)
        print(f"[av1an-dispatch] Audio: Opus {opus_bitrate} ({channels}ch)")

    # Build the av1an command
    if input_file:
        final_cmd.extend(["-i", input_file])
    if output_file:
        final_cmd.extend(["-o", output_file])

    final_cmd.extend(["--encoder", "svt-av1"])
    if not no_opus:
        final_cmd.extend(["-a", f"-c:a libopus -b:a {opus_bitrate}"])

    if workers:
        final_cmd.extend(["-w", workers])
    if photon_noise and photon_noise != "0":
        final_cmd.extend(["--photon-noise", photon_noise])

    # CRF and preset are svt-av1 encoder params, not av1an flags
    if speed:
        encoder_params = f"--preset {speed} " + encoder_params
    if quality:
        encoder_params = f"--crf {quality} " + encoder_params

    if encoder_params.strip():
        final_cmd.extend(["-v", encoder_params.strip()])

    print(f"[av1an-dispatch] Running: {' '.join(final_cmd)}")

    # --- Execute ---
    src_stat = os.stat(input_file) if input_file and os.path.exists(input_file) else None
    try:
        sys.stdout.flush()
        subprocess.check_call(final_cmd)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"[av1an-dispatch] Error: Could not execute av1an at {av1an_exe}")
        sys.exit(1)

    # Preserve source modification time
    if src_stat and output_file and os.path.exists(output_file):
        os.utime(output_file, (src_stat.st_atime, src_stat.st_mtime))


if __name__ == "__main__":
    main()
