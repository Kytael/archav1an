"""One worker per enabled denoiser, all pulling from one queue.

The roster is re-read before each clip, so disabling a denoiser lets it finish
its current clip and then stop taking work: nothing is killed and no progress
is lost (spec 5.2). Disabling is a pause, not an exit -- a worker whose
denoiser has gone away parks and keeps re-checking, so re-enabling the device
mid-run puts it straight back to work (spec 7.1 gate 7).
"""
import queue
import threading
import time

from .state import Record, append_record


class Scheduler:
    POLL_SECONDS = 5.0      # how often a parked worker re-reads the roster

    def __init__(self, clips, roster_fn, runner, state_path):
        """
        clips     -- ordered tuple of Clip still to do
        roster_fn -- callable returning a fresh Roster, called before each clip
        runner    -- callable(clip, denoiser) -> (ok, wall_s, fps, out_bytes)
        """
        self.queue = queue.Queue()
        for clip in clips:
            self.queue.put(clip)
        self.roster_fn = roster_fn
        self.runner = runner
        self.state_path = state_path
        self.done = 0
        self.failed = 0
        self.failures = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        roster = roster_fn()
        # Deliberate asymmetry: enablement is live and re-read per clip, but the
        # slot count is fixed for the life of the run.
        self._slots = threading.Semaphore(max(1, roster.encode.slots))

    def stop(self):
        """Wind the run down: workers finish the clip they hold and take no more."""
        self._stop.set()

    def run(self):
        roster = self.roster_fn()
        names = [d.name for d in roster.enabled()]
        threads = [threading.Thread(target=self._worker, args=(name,), daemon=True)
                   for name in names]
        for t in threads:
            t.start()

        parked = False
        pending = list(threads)
        while pending:
            pending[0].join(self.POLL_SECONDS)
            pending = [t for t in pending if t.is_alive()]
            if parked or not pending or self.queue.empty():
                continue
            if not self._any_enabled(names):
                parked = True
                print(f"archive-batch: every denoiser is disabled and "
                      f"{self.queue.qsize()} clip(s) are still queued. Workers are "
                      f"parked, not stuck -- re-enable a denoiser in the roster to "
                      f"resume.", flush=True)
        return self.failed

    def _any_enabled(self, names):
        try:
            roster = self.roster_fn()
        except Exception:
            return False
        return any(d.name in names for d in roster.enabled())

    def _worker(self, name):
        while True:
            if self._stop.is_set() or self.queue.empty():
                return
            denoiser = self._current(name)
            if denoiser is None:
                # Disabled or removed. Do not exit: the roster is live and the
                # user may re-enable this device mid-run (spec 7.1 gate 7).
                if self._stop.wait(self.POLL_SECONDS):
                    return
                continue
            try:
                clip = self.queue.get_nowait()
            except queue.Empty:
                return
            self._process(clip, denoiser)

    def _current(self, name):
        """Look this denoiser up in a freshly read roster; None if it is gone."""
        try:
            roster = self.roster_fn()
        except Exception:
            return None
        for d in roster.enabled():
            if d.name == name:
                return d
        return None

    def _process(self, clip, denoiser):
        self._slots.acquire()
        started = time.monotonic()
        raised = False
        try:
            ok, wall_s, fps, out_bytes = self.runner(clip, denoiser)
        except Exception as exc:
            ok, wall_s, fps, out_bytes = False, time.monotonic() - started, 0.0, 0
            raised = True
            with self._lock:
                self.failures.append((clip.src, denoiser.name, repr(exc)))
        finally:
            self._slots.release()

        try:
            append_record(self.state_path,
                          Record(src=clip.src, status="done" if ok else "failed",
                                 denoiser=denoiser.name, wall_s=round(wall_s, 2),
                                 fps=round(fps, 2), out_bytes=out_bytes))
        except Exception as exc:
            # Losing the state file loses resume, so stop the run out loud rather
            # than let workers die one by one with nothing written down.
            print(f"archive-batch: cannot write state to {self.state_path}: {exc!r} "
                  f"-- stopping the run", flush=True)
            self._stop.set()

        with self._lock:
            if ok:
                self.done += 1
            else:
                self.failed += 1
                if not raised:
                    self.failures.append((clip.src, denoiser.name,
                                          "runner reported failure"))
