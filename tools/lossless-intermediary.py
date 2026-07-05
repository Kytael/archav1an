import os
import glob
import subprocess
import shutil

# --- CONFIGURATION ---

import shutil
import sys

# --- CONFIGURATION ---

# Find x265 in system path
X265_EXE = shutil.which("x265")
if not X265_EXE:
    # Fallback/Check
    print(
        "[ERROR] x265 executable not found in PATH. Please install it (sudo pacman -S x265 (Arch) or sudo apt install x265 (Ubuntu))."
    )
    sys.exit(1)

# Stock x265 only reads raw YUV/Y4M, so the VPY must be piped through vspipe,
# and its raw HEVC output must be muxed into a real MKV with mkvmerge.
VSPIPE_EXE = shutil.which("vspipe")
if not VSPIPE_EXE:
    print("[ERROR] vspipe executable not found in PATH.")
    sys.exit(1)
MKVMERGE_EXE = shutil.which("mkvmerge")
if not MKVMERGE_EXE:
    print("[ERROR] mkvmerge executable not found in PATH.")
    sys.exit(1)

X265_SETTINGS = [
    "--preset",
    "superfast",
    "--output-depth",
    "10",
    "--lossless",
    "--colorprim",
    "bt709",
    "--colormatrix",
    "bt709",
    "--transfer",
    "bt709",
    "--level-idc",
    "0",
]


def run_piped_encode(vpy_file, hevc_file):
    """Pipes vspipe -c y4m into x265, writing a raw HEVC elementary stream."""
    vspipe_cmd = [VSPIPE_EXE, "-c", "y4m", vpy_file, "-"]
    x265_cmd = [X265_EXE] + X265_SETTINGS + ["-o", hevc_file, "-", "--y4m"]
    print(f"Running: {' '.join(vspipe_cmd)} | {' '.join(x265_cmd)}")
    try:
        p1 = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(x265_cmd, stdin=p1.stdout)
        p1.stdout.close()
        p2.wait()
        p1.wait()
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return False
    if p1.returncode != 0:
        print(f"[ERROR] vspipe failed with code {p1.returncode}")
        return False
    if p2.returncode != 0:
        print(f"[ERROR] x265 failed with code {p2.returncode}")
        return False
    return True


def run_shell_command(cmd_list):
    """Executes a command list via subprocess."""
    # Convert list to string for display/debugging
    cmd_str = " ".join([f'"{c}"' if " " in c else c for c in cmd_list])
    print(f"Running: {cmd_str}")

    try:
        subprocess.run(cmd_list, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        return False


def sanitize_filenames():
    """
    Renames files in the current directory to be CLI-friendly.
    Removes: () [] and braces. Replaces spaces with periods.
    """
    print("--- Checking for filenames to sanitize ---")

    # Files to ignore (outputs or scripts)
    exclusions = ("-x265lossless", ".vpy", ".py", ".bat")

    mkv_files = glob.glob("*.mkv")

    for filename in mkv_files:
        base, ext = os.path.splitext(filename)

        # Skip files that are likely already outputs
        if base.endswith("-x265lossless"):
            continue

        new_base = base
        # Remove brackets/braces
        for char in ["(", ")", "[", "]", "{", "}"]:
            new_base = new_base.replace(char, "")

        # Replace spaces with periods
        new_base = new_base.replace(" ", ".")

        # Clean up double dots potentially created by replacements
        while ".." in new_base:
            new_base = new_base.replace("..", ".")

        new_filename = new_base + ext

        if new_filename != filename:
            try:
                os.rename(filename, new_filename)
                print(f"Renamed: '{filename}' -> '{new_filename}'")
            except OSError as e:
                print(f"[ERROR] Could not rename '{filename}': {e}")

    print("------------------------------------------\n")


def create_vpy_script(source_file, vpy_filename):
    """
    Generates a simple VapourSynth script that:
    1. Loads video using ffms2.
    2. Converts to 10-bit (YUV420P10).
    """
    # Use absolute path for source to avoid ambiguity in VapourSynth
    abs_source = os.path.abspath(source_file).replace("\\", "/")

    vpy_content = f"""
import vapoursynth as vs
core = vs.core

# Load source
clip = core.ffms2.Source(source=r"{abs_source}")

# Convert to 10-bit using Point resize (clean bit-depth padding)
clip = clip.resize.Point(format=vs.YUV420P10)

clip.set_output()
"""
    try:
        with open(vpy_filename, "w", encoding="utf-8") as f:
            f.write(vpy_content)
        return True
    except IOError as e:
        print(f"[ERROR] Failed to write VPY script: {e}")
        return False


def main():
    # 1. Sanitize filenames in current folder (extras)
    sanitize_filenames()

    # 2. Find all MKV files
    source_files = glob.glob("*.mkv")

    if not source_files:
        print("No .mkv files found in the current folder.")
        return

    for source_file in source_files:
        # Skip existing output files to prevent loops
        if "-x265lossless" in source_file:
            continue

        base_name, ext = os.path.splitext(source_file)
        output_file = f"{base_name}-x265lossless.mkv"
        vpy_file = f"{base_name}.vpy"

        # Check if output already exists
        if os.path.exists(output_file):
            print(f"Skipping: {source_file}")
            print(f"Reason: Output '{output_file}' already exists.\n")
            continue

        print(f"\n=== Processing: {source_file} ===")

        # A. Create VapourSynth script
        if not create_vpy_script(source_file, vpy_file):
            continue

        # B. Encode via vspipe -> x265 (stock x265 cannot read a .vpy, and its
        # -o output is a raw HEVC stream, not a Matroska container)
        hevc_file = f"{base_name}-x265lossless.hevc"
        success = run_piped_encode(vpy_file, hevc_file)

        # C. Mux the raw HEVC with the source's audio/subs into a real MKV
        if success:
            success = run_shell_command(
                [MKVMERGE_EXE, "-o", output_file, hevc_file, "--no-video", source_file]
            )

        # D. Cleanup
        for tmp_file in (vpy_file, hevc_file):
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass

        if success:
            print(f"Success! Created: {output_file}")
        else:
            print(f"Failed to encode: {source_file}")
            # Optional: Delete partial output if failed
            if os.path.exists(output_file):
                os.remove(output_file)


if __name__ == "__main__":
    main()
