import os
import sys
import psutil
import subprocess
import time
import math
import shutil

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AV1AN_PATH = "av1an"
SAMPLE_FILE = os.path.join(BASE_DIR, "tools", "sample.mkv")
CONFIG_FILE = os.path.join(BASE_DIR, "tools", "workercount-config.txt")

def cleanup_temp_folders():
    """Deletes temp folders and the test output file with retry logic."""
    print("Cleaning up temporary test files...", file=sys.stderr)
    
    # Wait 2 seconds to let Windows release file locks from the killed process
    time.sleep(2)

    # 1. Clean up av1an temp folders (hex-named, e.g. .df97856)
    # IMPORTANT: only delete short all-hex dotdirs to avoid nuking .git, .claude, etc.
    import re
    av1an_tmp_pattern = re.compile(r'^\.[0-9a-f]{6,16}$')
    try:
        for item in os.listdir(BASE_DIR):
            item_path = os.path.join(BASE_DIR, item)
            if os.path.isdir(item_path) and av1an_tmp_pattern.match(item):
                deleted = False
                for attempt in range(3):
                    try:
                        shutil.rmtree(item_path)
                        print(f"   - Deleted: {item}", file=sys.stderr)
                        deleted = True
                        break 
                    except OSError:
                        time.sleep(1)
                
                if not deleted:
                    print(f"   - Warning: Could not fully delete {item} (File in use).", file=sys.stderr)
    except Exception as e:
        print(f"Error during folder cleanup: {e}", file=sys.stderr)

    # 2. Clean up the test output video file (sample_svt-av1.mkv)
    output_file = os.path.join(BASE_DIR, "sample_svt-av1.mkv")
    if os.path.exists(output_file):
        deleted = False
        for attempt in range(3):
            try:
                os.remove(output_file)
                print(f"   - Deleted: sample_svt-av1.mkv", file=sys.stderr)
                deleted = True
                break
            except OSError:
                time.sleep(1)
        
        if not deleted:
            print(f"   - Warning: Could not delete sample_svt-av1.mkv (File in use).", file=sys.stderr)

def get_optimal_workers():
    print(f"Running one-time RAM test on {os.path.basename(SAMPLE_FILE)}...", file=sys.stderr)
    print("Please wait while we measure memory usage...", file=sys.stderr)
    
    # 1. Start the test process with 1 worker
    cmd = [
        AV1AN_PATH,
        "-i", SAMPLE_FILE,
        "-y",
        "--workers", "1",
        "--verbose",
        "-e", "svt-av1",
        "-m", "bestsource",
        "--cache-mode", "temp",
        "-v", " --preset 4 --crf 30 --lp 3",
    ]

    try:
        # Start av1an
        process = subprocess.Popen(
            cmd, 
            cwd=BASE_DIR
        )
    except FileNotFoundError:
        print("Error: av1an executable not found.", file=sys.stderr)
        return 1

    max_total_rss = 0
    
    # 2. Monitor RAM usage (Parent + Children)
    try:
        # Monitor for up to 20 seconds
        for _ in range(40): 
            if process.poll() is not None:
                break
            
            try:
                current_rss = 0
                parent = psutil.Process(process.pid)
                current_rss += parent.memory_info().rss
                
                # Add up memory of all child processes (the encoders)
                for child in parent.children(recursive=True):
                    try:
                        current_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                if current_rss > max_total_rss:
                    max_total_rss = current_rss
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            time.sleep(0.5)
    finally:
        # Ensure process is killed if it's still running after timeout
        if process.poll() is None:
            process.kill()

    # 3. Perform Calculations
    if max_total_rss == 0:
        print("\nWarning: Could not measure RAM. Defaulting to 2 workers.")
        cleanup_temp_folders()
        return 2

    total_ram = psutil.virtual_memory().total
    total_ram_gb = total_ram / (1024 ** 3)
    cpu_threads = os.cpu_count()
    physical_cores = psutil.cpu_count(logical=False) or cpu_threads

    # Worker count uses three constraints:
    # 1. sqrt(cpu_threads) — models diminishing returns from cache/bandwidth contention
    #    4T→2, 16T→4, 32T→5, 64T→8
    # 2. RAM — measured peak per worker, leave 15% headroom
    # 3. Hard cap at 8 — beyond this, av1an chunk overhead and mkvmerge concat cost dominate
    max_workers_cpu = int(math.sqrt(cpu_threads))
    max_workers_ram = int((total_ram * 0.85) / max_total_rss) if max_total_rss > 0 else 8
    final_workers = min(max_workers_cpu, max_workers_ram, 8)
    final_workers = max(2, final_workers)  # always at least 2

    print("\n------------------------------------------------")
    print(f"   - Total System RAM: {total_ram // (1024**2)} MB")
    print(f"   - Peak RAM (1 Worker): {max_total_rss // (1024**2)} MB")
    print(f"   - CPU Threads: {cpu_threads} ({physical_cores} physical)")
    print(f"   - Max by CPU (sqrt): {max_workers_cpu}")
    print(f"   - Max by RAM: {max_workers_ram}")
    print(f"   - Calculated Optimal Workers: {final_workers}")
    print("------------------------------------------------")
    
    # Run cleanup before returning
    cleanup_temp_folders()

    return final_workers

if __name__ == "__main__":
    if not os.path.exists(SAMPLE_FILE):
        print(f"Error: {SAMPLE_FILE} missing. Defaulting to 1.")
        workers = 1
    else:
        workers = get_optimal_workers()

    # Save to config file
    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(f"workers={workers}\n")
        print("\nOne-time test complete. Auto worker count set, please run the script again.")
        print("You may manually edit tools/workercount-config.txt if needed")
    except Exception as e:
        print(f"Error writing config file: {e}")