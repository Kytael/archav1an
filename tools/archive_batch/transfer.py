"""Stage one source in, publish one output out.

rsync writes to a temporary name in the destination directory and renames on
success, so an interrupted publish never leaves a file that resume would treat
as complete (spec 5.1 step 4).

The remote directory is created with --rsync-path rather than --mkpath, which
matches the existing pattern at svtav1-dispatch.py:285-286.
"""
import os
import subprocess

from . import ARCHIVE_ROOT, ENCODED_SUBDIR


class TransferError(Exception):
    """A transfer could not even be attempted."""


def stage_cmd(host, rel_src, dest_dir):
    if rel_src.startswith("/"):
        raise TransferError(f"source must be relative to {ARCHIVE_ROOT}: {rel_src}")
    return ["rsync", "-a", f"{host}:{ARCHIVE_ROOT}/{rel_src}", f"{dest_dir}/"]


def publish_cmd(host, local_out, rel_dir):
    if rel_dir.startswith("/"):
        raise TransferError(f"rel_dir must be relative: {rel_dir}")
    remote_dir = f"{ARCHIVE_ROOT}/{ENCODED_SUBDIR}/{rel_dir}"
    return ["rsync", "-a", "--rsync-path", f"mkdir -p '{remote_dir}' && rsync",
            local_out, f"{host}:{remote_dir}/"]


def run(cmd, timeout=3600):
    """Run a transfer, raising TransferError with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise TransferError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def staged_path(dest_dir, rel_src):
    return os.path.join(dest_dir, os.path.basename(rel_src))
