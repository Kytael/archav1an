"""Process identity for the batch.json pid-file contract.

Every liveness decision in this contract used to be "os.kill(pid, 0)
succeeded". On a host that shells out to ffmpeg constantly, a dead batch's
pid is recycled by the kernel within minutes: the recycled pid makes start()
raise AlreadyRunning forever, lets adopt() claim an unrelated process, and
holds encode_batch_up at 1 so EncodeBatchDown never fires. The fix is to pin
the spawn-time /proc starttime alongside the pid and compare on every probe:
a reused pid has a different starttime, so the record reads as stale.

A record without a pid_start field -- anything written before this existed --
cannot be compared that way, so it is checked against the command line
instead. Existence alone would report a recycled pid as the batch for ever;
rejecting such a record outright would report a LIVE pre-upgrade batch as
dead, which re-enables the dashboard's Start button and puts a second batch
over the first. The command line separates the two.
"""
import fcntl
import os
from contextlib import contextmanager

# What the batch's own command line holds: the daemon spawns
# [sys.executable, <repo>/tools/archive-batch.py], and an operator running it
# from a shell passes the same path. A record with no pid_start is matched
# against this. A false positive needs a recycled pid that is itself running
# something named archive-batch, and it errs towards "the run is live", which
# is the safe direction here -- a false negative offers Start over a live run.
BATCH_MARKER = "archive-batch"


def _stat_after_comm(pid):
    """/proc/<pid>/stat from the state field on, or None if unreadable.

    comm (field 2) can hold spaces and parentheses of its own, so the fields
    are split after comm's closing paren. State is field 3, so it is index 0
    of what this returns, and starttime (field 22) is index 19.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    close = data.rfind(b")")
    if close < 0:
        return None
    return data[close + 2:].split()


def start_time(pid):
    """The process's /proc starttime (field 22), or None if unreadable."""
    fields = _stat_after_comm(pid)
    if fields is None:
        return None
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def cmdline(pid):
    """The process's /proc command line as one string, or None if unreadable.

    Decoded permissively: an argument this daemon never controls must not
    raise out of a liveness probe.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def alive(pid, pid_start=None):
    """True if pid runs and is the same process a record was written for.

    A pid that is not a positive int reads as dead, matching every reader's
    corrupt-row handling: -1 would make os.kill probe every process this
    user owns, and a string would raise TypeError. With no recorded start
    time -- a record written before pid_start existed -- identity falls back
    to the command line, for the reason in the module docstring.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                # exists, owned by someone else
    except OSError:
        return False
    fields = _stat_after_comm(pid)
    if fields and fields[0] == b"Z":
        # A zombie: the process is over, but the entry stays in the table
        # until its parent waits on it, and os.kill(pid, 0) succeeds
        # throughout. The dashboard's daemon is the parent of the batch it
        # spawned, so a batch it did not reap yet would otherwise read as a
        # live run: encode_batch_up pinned at 1 and Stop offered for a run
        # that is over. The state is checked in the one place every reader
        # goes through, rather than trusting each of them to reap.
        return False
    if pid_start is None:
        line = cmdline(pid)
        return line is not None and BATCH_MARKER in line
    now = None
    if fields is not None:
        try:
            now = int(fields[19])
        except (IndexError, ValueError):
            now = None
    return now is not None and now == pid_start


@contextmanager
def exclusive(lock_path):
    """Serialize the decide-then-write sequences around batch.json.

    The takeover paths read the file, decide the recorded batch is dead, then
    replace it. Two batches that read the same stale file could both decide
    and both write -- the exact disaster the O_EXCL fast-path claim exists to
    prevent -- because nothing made their read-decide-write atomic. An flock
    does: every writer takes it around the whole sequence, so the second one
    re-reads the winner's live claim and exits.
    """
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        # Closing the descriptor releases the lock.
        os.close(fd)
