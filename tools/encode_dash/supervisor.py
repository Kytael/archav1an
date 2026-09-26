"""Own or adopt the archive batch process.

The supervisor spawns tools/archive-batch.py and also claims a batch a
previous daemon left behind (the batch writes batch.json itself; the
supervisor's own write here just closes the window between spawn and the
batch's record).
"""

import json
import os
import subprocess
import threading
import time
from contextlib import ExitStack

from tools.archive_batch import pidfile


# How long start() watches a fresh spawn before it calls the run started.
# A batch that refuses to run does so from its own argument and manifest
# checks, measured at 40 ms, so this is an order of magnitude of headroom on a
# loaded box. It is paid once per click of Start and by nothing else.
_SPAWN_GRACE = 0.5

# The batch's stderr, beside batch.json. A file rather than a pipe to this
# daemon, and that is the whole point: KillMode=process and adopt() exist so a
# batch outlives a daemon restart, and a pipe would break that contract -- the
# read end dies with the daemon, and every write from the batch or from any
# encoder it spawned then takes EPIPE. Measured on 2026-08-26: a daemon restart
# under a live run killed SvtAv1EncApp with SIGPIPE on eleven consecutive
# clips. A file descriptor to a file depends on nothing.
_STDERR_LOG = "batch-stderr.log"


class AlreadyRunning(Exception):
    """A live batch is on file; the daemon will not spawn a second."""


class SupervisorError(Exception):
    """The run directory cannot be managed: no lock, no batch.json, no spawn.

    Carries a sentence an operator can act on. adopt() logs it and degrades to
    a read-only dashboard; start() raises it, and the POST handler answers 503
    with the message rather than a bare repr.
    """


def _log_line(*args):
    # The daemon's own log calls flush; a bare print here would sit in the
    # block buffer for the whole run, so this wrapper flushes too.
    print(*args, flush=True)


class Supervisor:
    def __init__(self, cmd, cwd, batch_file, log=_log_line):
        self._cmd = list(cmd)
        self._cwd = cwd
        self._batch_file = batch_file
        self._lock = threading.Lock()
        self._pid = None
        # The /proc starttime of _pid, captured when we spawned or adopted
        # it. A bare os.kill probe would keep trusting a recycled pid.
        self._start = None
        self._proc = None       # Popen for a pid this process spawned
        self._spawned = False
        self._log = log

    def _own_alive(self):
        """Reap our spawned child and report whether it still runs.

        A spawned child is a direct child of this daemon, so waitpid (via
        Popen.poll) reaps it. os.kill(pid, 0) would keep reporting a zombie as
        alive until the wait, which would block a second start forever, so
        poll is the only correct probe for our own spawn.
        """
        if self._proc is None:
            return False
        if self._proc.poll() is not None:
            self._pid = None
            self._start = None
            self._proc = None
            self._spawned = False
            return False
        return True

    def _read(self):
        try:
            with open(self._batch_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pid_from_row(row):
        """The batch_pid a batch.json row names, or None if unusable.

        A corrupt file must read as no run, exactly as model.py's _batch
        treats it: a file holding "[]" or "null" has no .get, and -1 would
        make os.kill(-1, 0) probe every process this user owns, so any pid
        that is not a positive int is rejected.
        """
        pid = row.get("batch_pid") if isinstance(row, dict) else None
        if not isinstance(pid, int) or pid <= 0:
            return None
        return pid

    def _forget(self):
        """Drop what this process knows about a batch. Caller holds _lock."""
        self._pid = None
        self._start = None
        self._proc = None
        self._spawned = False

    def _clear(self):
        try:
            os.remove(self._batch_file)
        except OSError:
            pass

    def _make_run_dir(self):
        """Create the run directory, or say why the daemon cannot manage it.

        Every path here needs two things in that directory: batch.json and
        batch.json.lock. makedirs is not the whole test -- it SUCCEEDS on an
        existing read-only directory, and the failure then surfaces from
        pidfile.exclusive's os.open instead. Both are wrapped by the callers,
        so the caller decides between degrading and reporting.
        """
        os.makedirs(os.path.dirname(self._batch_file), exist_ok=True)

    def _record_pid(self, pid):
        """Persist a spawn so a restarting daemon can adopt it.

        Takes batch.json.lock itself, so standalone callers are safe;
        start() uses _record_pid_locked because it already holds the lock
        (flock from a second fd in the same process would block forever).
        """
        with pidfile.exclusive(self._batch_file + ".lock"):
            self._record_pid_locked(pid)

    def _record_pid_locked(self, pid):
        """_record_pid for a caller already holding batch.json.lock."""
        self._make_run_dir()
        # The read-decide-replace sequence is serialized against the batch's
        # own takeover under the same flock. Without it, a batch claiming via
        # O_EXCL in between our read and our replace would win and then have
        # its file overwritten with our pid -- two owners over one manifest.
        row = self._read()
        if row is not None:
            other = row.get("batch_pid")
            if isinstance(other, int) and other == pid:
                # The file already names this spawn (our own write, or
                # the batch's claim). Rewriting would drop the batch's
                # "claimed" marker, so leave it alone.
                return
            other_start = row.get("pid_start")
            if isinstance(other, int) and other != pid \
                    and pidfile.alive(other, other_start):
                return
        tmp = self._batch_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"batch_pid": pid,
                       "pid_start": pidfile.start_time(pid),
                       "started_at": time.time()}, fh)
        os.replace(tmp, self._batch_file)

    def adopt(self):
        """Claim a live batch from file, or clear a stale one.

        The read-decide-clear sits under batch.json.lock: unlocked, start()
        could unlink a live claim that a shell batch had just written after
        reading the same stale row this function is looking at.

        Runs unguarded at daemon boot, so an environment it cannot manage --
        a read-only run dir, a batch owned by another user -- must degrade to
        "no run known" rather than take the whole dashboard down. The try
        covers the flock as well as the makedirs: makedirs succeeds on an
        existing read-only directory, and it is pidfile.exclusive's os.open
        that raises there.
        """
        try:
            # The lock lives beside batch.json and must be creatable on a
            # fresh checkout where no run directory exists yet.
            self._make_run_dir()
            with self._lock, pidfile.exclusive(self._batch_file + ".lock"):
                row = self._read()
                if row is None:
                    self._forget()
                    return None
                pid = self._pid_from_row(row)
                # Identity-checked: a recycled pid reads as stale and the
                # file is cleared instead of adopting an unrelated process.
                if pid is not None and pidfile.alive(pid, row.get("pid_start")):
                    self._pid = pid
                    self._start = row.get("pid_start")
                    self._proc = None
                    self._spawned = False
                    self._log(f"[encode-dash] adopting batch pid {pid}")
                    return pid
                self._clear()
                self._forget()
                return None
        except OSError as exc:
            self._log(f"[encode-dash] cannot manage {self._batch_file}: "
                      f"{exc!r}; continuing without batch supervision")
            with self._lock:
                self._forget()
            return None

    @staticmethod
    def _exited_at_once(proc):
        """The child's exit status if it is already over, else None."""
        try:
            return proc.wait(timeout=_SPAWN_GRACE)
        except subprocess.TimeoutExpired:
            return None

    def _stderr_path(self):
        return os.path.join(os.path.dirname(self._batch_file), _STDERR_LOG)

    def _death(self, status):
        """The sentence the operator gets for a spawn that died at once.

        The batch's own last words, because they name the fault: a missing
        manifest, an unreadable roster. Bounded read: a child that managed to
        print a traceback per lane must not put all of it in a status line.
        """
        try:
            with open(self._stderr_path(), "rb") as fh:
                text = fh.read(4096).decode("utf-8", "replace")
        except OSError:
            text = ""
        said = " ".join(l.strip() for l in text.splitlines() if l.strip())
        if not said:
            return (f"the batch exited at once with status {status} and "
                    f"printed nothing. Run it by hand to see why.")
        return f"the batch exited at once with status {status}: {said}"

    def _clear_if_ours(self, pid):
        """Unlink the spawn record for a batch that died before it ran.

        Only when the file still names this spawn, and under the lock: the
        batch replaces our record with its own claim, and a later batch's
        live claim must never be unlinked by a dead one's cleanup.
        """
        try:
            with pidfile.exclusive(self._batch_file + ".lock"):
                row = self._read()
                if isinstance(row, dict) and row.get("batch_pid") == pid:
                    self._clear()
        except OSError:
            # The run directory was writable moments ago, when the record was
            # written. If it is not now, the batch's death is still the news.
            pass

    def start(self):
        """Spawn the batch, or raise AlreadyRunning if one is live.

        The read-decide-clear-spawn-record sequence holds batch.json.lock
        throughout: a stale row decided dead here must not be unlinked after
        a shell batch has already taken it over. _record_pid_locked rather
        than _record_pid because flock from a second fd in this process
        would block on our own lock. The watch for an early death comes
        after the lock is released, because the batch wants it too.

        A run directory this daemon cannot write raises SupervisorError, not
        the bare OSError: adopt() has already degraded quietly by this point,
        so the operator clicking Start is owed the reason. A batch that exits
        during its own pre-flight raises the same, carrying what the batch
        said, for the same reason.
        """
        with self._lock:
            if self._own_alive():
                raise AlreadyRunning(
                    f"a batch is already running (pid {self._pid})")
            # ExitStack so only the two calls that touch the run directory
            # are inside the try: an OSError from the spawn below (a missing
            # interpreter, say) is a different fault and must not be reported
            # as an unwritable run directory.
            stack = ExitStack()
            try:
                # The lock lives beside batch.json and must be creatable on a
                # fresh checkout where no run directory exists yet.
                self._make_run_dir()
                stack.enter_context(
                    pidfile.exclusive(self._batch_file + ".lock"))
            except OSError as exc:
                stack.close()
                raise SupervisorError(
                    f"cannot write {self._batch_file}: {exc.strerror or exc}. "
                    f"The run directory is not writable by this daemon, so a "
                    f"batch cannot be claimed or recorded.") from exc
            with stack:
                row = self._read()
                if row is not None:
                    pid = self._pid_from_row(row)
                    if pid is not None \
                            and pidfile.alive(pid, row.get("pid_start")):
                        self._pid = pid
                        self._start = row.get("pid_start")
                        self._proc = None
                        self._spawned = False
                        raise AlreadyRunning(
                            f"a batch is already running (pid {pid}, from "
                            f"{os.path.basename(self._batch_file)})")
                    self._clear()
                with open(self._stderr_path(), "wb") as errors:
                    proc = subprocess.Popen(self._cmd, cwd=self._cwd,
                                            start_new_session=True,
                                            stderr=errors)
                self._pid = proc.pid
                self._start = pidfile.start_time(proc.pid)
                self._proc = proc
                self._spawned = True
                self._record_pid_locked(proc.pid)
            # A spawn is not a run. The batch validates its roster and its
            # manifest before it does anything, and a refusal there exits in
            # milliseconds -- so returning the pid unconditionally answers 200
            # with "started" for a run that is already over, and the operator
            # is left with a button that does nothing.
            #
            # Watched from outside the lock, and that is not a detail: the
            # batch's own _claim_run takes this same flock as its first act,
            # so a spawn watched from inside it blocks on us and can only ever
            # look alive.
            status = self._exited_at_once(proc)
            if status is not None:
                self._clear_if_ours(proc.pid)
                self._forget()
                raise SupervisorError(self._death(status))
            self._log(f"[encode-dash] started batch pid {proc.pid}")
            return proc.pid

    def status(self):
        """(pid, alive, spawned). The snapshot merge uses this."""
        with self._lock:
            if self._spawned:
                # Tuple elements evaluate left-to-right: read the pid AFTER
                # _own_alive() reaps/resets, so the first post-death call is
                # (None, False, False), not (dead_pid, False, False).
                alive = self._own_alive()
                if not alive:
                    self._start = None
                return self._pid, alive, self._spawned
            return self._pid, pidfile.alive(self._pid, self._start), \
                self._spawned
