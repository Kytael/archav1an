"""Stage one source in, publish one output out.

rsync writes to a temporary name in the destination directory and renames on
success, so an interrupted publish never leaves a file that resume would treat
as complete (spec 5.1 step 4).

The remote directory is created with --rsync-path rather than --mkpath, which
matches the existing pattern at svtav1-dispatch.py:285-286.
"""
import os
import shlex
import subprocess
import sys
import time

from . import ARCHIVE_ROOT, ENCODED_SUBDIR

# Both staging and publishing target the source host, so an outage there stops
# every denoiser at once. Without a retry window two workers burn the whole
# remaining queue in about half an hour, which is less than a Windows update
# takes (spec 6).
RETRY_WINDOW_S = 1800.0
BASE_DELAY_S = 5.0
MAX_DELAY_S = 300.0


class TransferError(Exception):
    """A transfer could not even be attempted."""


class TransferOutage(TransferError):
    """The retry window expired: the source host is down, not the clip bad."""


def stage_cmd(host, rel_src, dest_dir):
    if rel_src.startswith("/"):
        raise TransferError(f"source must be relative to {ARCHIVE_ROOT}: {rel_src}")
    return ["rsync", "-a", f"{host}:{ARCHIVE_ROOT}/{rel_src}", f"{dest_dir}/"]


def publish_cmd(host, local_out, rel_dir):
    if rel_dir.startswith("/"):
        raise TransferError(f"rel_dir must be relative: {rel_dir}")
    remote_dir = f"{ARCHIVE_ROOT}/{ENCODED_SUBDIR}/{rel_dir}"
    # --rsync-path is the one argument rsync hands to the remote shell verbatim,
    # so it needs quoting of its own. Every other path rides protect-args.
    return ["rsync", "-a", "--rsync-path",
            f"mkdir -p {shlex.quote(remote_dir)} && rsync",
            local_out, f"{host}:{remote_dir}/"]


def run(cmd, timeout=3600, retry_window=RETRY_WINDOW_S,
        sleep=time.sleep, clock=time.monotonic):
    """Run a transfer, retrying with backoff until the window expires.

    rsync is restartable and both callers are idempotent, so a retry is always
    safe: staging re-copies, publishing re-uploads to the same temp-and-rename.
    """
    deadline = clock() + retry_window
    delay = BASE_DELAY_S
    while True:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            reason = f"{cmd[0]} timed out after {timeout}s"
        else:
            if proc.returncode == 0:
                return proc
            reason = f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()}"

        if clock() >= deadline:
            raise TransferOutage(f"{reason} -- gave up after {retry_window:.0f}s")
        print(f"[archive-batch] {reason}; retrying in {delay:.0f}s",
              file=sys.stderr, flush=True)
        sleep(delay)
        delay = min(delay * 2, MAX_DELAY_S)


def staged_path(dest_dir, rel_src):
    return os.path.join(dest_dir, os.path.basename(rel_src))
