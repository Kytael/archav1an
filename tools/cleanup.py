import os
import shutil
import glob


def cleanup_workspace():
    # Directories to scan for trash (flat scan for root-level junk)
    scan_dirs = [".", "Input", "Output"]

    # Extensions to delete (Files)
    extensions = [
        ".ffindex",
        ".lwi",
        ".json",
        ".log",
        ".temp",
        ".vpy",
        ".stats",
        ".mbtree",
        ".zone",
        ".csv",
    ]

    print("Cleaning up workspace...")

    for d in scan_dirs:
        if not os.path.exists(d):
            continue

        for item in os.listdir(d):
            item_path = os.path.join(d, item)

            # 1. DELETE FILES
            if os.path.isfile(item_path):
                for ext in extensions:
                    if item.lower().endswith(ext):
                        try:
                            os.remove(item_path)
                            print(f"Deleted: {item_path}")
                        except:
                            pass
                        break

            # 2. DELETE DIRECTORIES (Temp folders)
            elif os.path.isdir(item_path):
                # Whitelist: Critical system/project folders to never touch
                if item in [
                    ".git",
                    ".vscode",
                    ".idea",
                    "Input",
                    "Output",
                    "Temp",
                    "tools",
                    "VapourSynth",
                    "__pycache__",
                    "venv",
                ]:
                    continue

                # Condition: Starts with "." (Hidden temp) OR ends with ".tmp" OR ends with "-source"
                if (
                    item.startswith(".")
                    or item.endswith(".tmp")
                    or item.endswith("-source")
                ):
                    # Safety: Only delete if there are no "real" (non-fastpass) MKV files inside.
                    # Preserves folders mid-final-encode. Mirrors Windows cleanup.py logic.
                    mkv_files = glob.glob(os.path.join(item_path, "*.mkv"))
                    files_to_preserve = [
                        f for f in mkv_files
                        if "fastpass" not in os.path.basename(f).lower()
                    ]
                    if files_to_preserve:
                        print(f"Skipped temp dir (contains {len(files_to_preserve)} essential .mkv file(s)): {item_path}")
                        continue
                    try:
                        shutil.rmtree(item_path)
                        print(f"Deleted temp dir: {item_path}")
                    except:
                        pass

    # 2b. Recursive cleanup of Temp/ directory (pipeline temp files)
    temp_root = "Temp"
    if os.path.exists(temp_root):
        # Walk Temp/ and remove empty directories bottom-up
        # Also remove stale temp files (ffindex, lwi, json, log, etc.)
        for root, dirs, files in os.walk(temp_root, topdown=False):
            for f in files:
                fpath = os.path.join(root, f)
                for ext in extensions:
                    if f.lower().endswith(ext):
                        try:
                            os.remove(fpath)
                            print(f"Deleted: {fpath}")
                        except:
                            pass
                        break
            # Remove directory if empty after file cleanup
            try:
                if not os.listdir(root):
                    os.rmdir(root)
                    print(f"Deleted empty dir: {root}")
            except:
                pass

    # 2c. Recursive cleanup of temp files inside Output/ subdirectories
    output_dir = "Output"
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                fpath = os.path.join(root, f)
                for ext in extensions:
                    if f.lower().endswith(ext):
                        try:
                            os.remove(fpath)
                            print(f"Deleted: {fpath}")
                        except:
                            pass
                        break

    # 2d. Recursive cleanup of temp files inside Input/ subdirectories
    input_dir = "Input"
    if os.path.exists(input_dir):
        for root, dirs, files in os.walk(input_dir):
            # Skip the Input/ root (already handled above)
            if root == input_dir:
                continue
            for f in files:
                fpath = os.path.join(root, f)
                for ext in extensions:
                    if f.lower().endswith(ext):
                        try:
                            os.remove(fpath)
                            print(f"Deleted: {fpath}")
                        except:
                            pass
                        break

    # 3. DELETE filter/*.ffindex files (filter dir uses FFMS2 caches too)
    filter_dir = "filter"
    if os.path.exists(filter_dir):
        for f in glob.glob(os.path.join(filter_dir, "*.ffindex")):
            try:
                os.remove(f)
                print(f"Deleted: {f}")
            except:
                pass

    # 4. DELETE Input/logs/ folder (av1an leaves log dirs in source folder)
    logs_dir = os.path.join("Input", "logs")
    if os.path.exists(logs_dir):
        try:
            shutil.rmtree(logs_dir)
            print(f"Deleted: {logs_dir}")
        except:
            pass

    # 5. DELETE Input/*.bsindex files (bestsource index files) — recursive
    for root, _dirs, files in os.walk("Input") if os.path.exists("Input") else []:
        for f in files:
            if f.lower().endswith(".bsindex"):
                fpath = os.path.join(root, f)
                try:
                    os.remove(fpath)
                    print(f"Deleted: {fpath}")
                except:
                    pass

    # 6. DELETE tools/ssimu2_bench_temp/ folder (leftover from benchmarking)
    bench_temp = os.path.join("tools", "ssimu2_bench_temp")
    if os.path.exists(bench_temp):
        try:
            shutil.rmtree(bench_temp)
            print(f"Deleted: {bench_temp}")
        except:
            pass


if __name__ == "__main__":
    cleanup_workspace()
