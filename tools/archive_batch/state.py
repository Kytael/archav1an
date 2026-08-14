"""Append-only run state.

Resume reads this file and never the filesystem: an output that exists on F:
may be a partial publish, so only an explicit `done` record counts. See the
spec, 5.3.
"""
import json
import os
import threading
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

MAX_ATTEMPTS = 2

_write_lock = threading.Lock()


@dataclass(frozen=True)
class Record:
    src: str
    status: str        # "done" or "failed"
    denoiser: str
    wall_s: float
    fps: float
    out_bytes: int
    # Exit code plus the tail of the denoiser's log (spec 6). Without it a
    # failure three days into an unattended run cannot be diagnosed at all.
    reason: str = ""
    # wall_s covers the whole clip, so the fps derived from it is the rate the
    # archive actually drains at -- staging and publishing included. That is the
    # number that predicts when the run finishes, but it is not the number that
    # says how fast a denoiser is, and the two can differ by a lot on a lane
    # that copies every source across the network. Record the split so both
    # questions can be answered from the same file. Defaulted, so records
    # written before this existed still load.
    stage_s: float = 0.0
    work_s: float = 0.0
    publish_s: float = 0.0


@dataclass(frozen=True)
class State:
    """What earlier runs already did, by name rather than by position.

    frozen=True only seals the two fields; it does nothing about what they
    hold. A plain set and dict could still be written into behind that seal, so
    the contents are read-only types too and the promise is honest. Nothing
    should ever edit resume state in memory: the file on disk is the record.
    """
    done: frozenset = frozenset()
    failures: Mapping = MappingProxyType({})


def append_record(path, record):
    """Append one record and fsync, so a power cut cannot lose a finished clip."""
    with _write_lock:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        is_new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if is_new:
            # Fsyncing the file does not make its directory entry durable, so a
            # power cut right after the first record could lose the whole file.
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)


def load_state(path):
    done, failures = set(), {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src, status = row.get("src"), row.get("status")
                if not src:
                    continue
                if status == "done":
                    done.add(src)
                elif status == "failed":
                    failures[src] = failures.get(src, 0) + 1
    except FileNotFoundError:
        pass
    return State(done=frozenset(done), failures=MappingProxyType(failures))


def pending_clips(clips, state):
    """Clips still to do: not done, and not already failed MAX_ATTEMPTS times.

    A clip that failed once is retried, because some failures are specific to
    the denoiser that took it.
    """
    return tuple(c for c in clips
                 if c.src not in state.done
                 and state.failures.get(c.src, 0) < MAX_ATTEMPTS)
