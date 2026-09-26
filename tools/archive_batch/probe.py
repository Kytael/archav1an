"""Probe a folder for encodable video, on this host or over ssh.

The same ffprobe walk tools/archive-batch-manifest.sh does, narrowed to one
directory and returning rows rather than writing a file. It is the slow part
of a submission -- a network path, one ffprobe per file -- so its caller runs
it off the control poller's thread.
"""
import subprocess

from .transfer import TransferError

# Matches the manifest walk. Deliberately not "every file ffprobe will open":
# a folder an operator points at holds stills and sidecars too, and probing
# each to find out costs a round trip apiece.
EXTENSIONS = (".mov", ".mp4")

# One ffprobe per file over a network path. Generous, because the alternative
# to waiting is a submission that silently drops the files it timed out on.
PROBE_TIMEOUT_S = 900

_WALK = r'''
set -u
cd "$DIR" 2>/dev/null || { echo "__NODIR__"; exit 0; }
find . -maxdepth 1 -type f \( -iname '*.mov' -o -iname '*.mp4' \) -print0 \
  | sort -z | while IFS= read -r -d '' f; do
    size=$(stat -c%s "$f" 2>/dev/null) || continue
    probe=$(ffprobe -v error -select_streams v \
              -show_entries stream=r_frame_rate,nb_frames:format=duration \
              -of csv=p=0 "$f" 2>/dev/null) || continue
    [ -n "$probe" ] || continue
    printf '%s\t%s\t%s\t\n' "${f#./}" "$size" \
      "$(printf '%s' "$probe" | tr '\n' '\t' | sed 's/\t$//')"
done
'''


def probe_folder(host, path, timeout=PROBE_TIMEOUT_S, runner=subprocess.run):
    """[(name, size, rate_field)] for each video directly in `path`.

    One level, not a recursive walk: a submission names a folder, and pulling
    in a whole tree under it would publish files to a destination the operator
    chose for their parent.

    Raises TransferError for a path that is not a directory, so the caller can
    refuse the submission with an ack rather than an exception.
    """
    if not path.startswith("/"):
        raise TransferError(f"the folder must be an absolute path: {path}")
    # "local" is a roster host name, not a machine.
    remote = bool(host) and host != "local"
    # DIR travels in the command string, not the environment: ssh carries no
    # environment to the remote shell, so an env= that works locally would
    # silently probe the remote's home directory instead.
    if remote:
        cmd = ["ssh", "-o", "BatchMode=yes", host,
               f"DIR={_quote(path)} bash -s"]
    else:
        cmd = ["bash", "-c", f"DIR={_quote(path)} bash -s"]
    try:
        proc = runner(cmd, input=_WALK, capture_output=True, text=True,
                      timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TransferError(f"probing {path} on {host or 'this host'} timed "
                            f"out after {timeout}s")
    if proc.returncode != 0:
        raise TransferError(
            f"cannot read {path} on {host or 'this host'}: "
            f"{(proc.stderr or '').strip()[:200]}")
    if "__NODIR__" in proc.stdout:
        raise TransferError(f"{path} is not a directory on "
                            f"{host or 'this host'}")
    rows = []
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 3 or not fields[1].isdigit():
            continue
        rows.append((fields[0], int(fields[1]), fields[2]))
    return rows


def _quote(value):
    return "'" + value.replace("'", "'\\''") + "'"
