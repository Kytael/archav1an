"""One worker per enabled denoiser, all pulling from one queue.

The roster is re-read before each clip, so disabling a denoiser lets it finish
its current clip and then stop taking work: nothing is killed and no progress
is lost (spec 5.2).
"""
import queue
import threading
import time

from .state import Record, append_record


class Scheduler:
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
        roster = roster_fn()
        self._slots = threading.Semaphore(max(1, roster.encode.slots))

    def run(self):
        roster = self.roster_fn()
        threads = [threading.Thread(target=self._worker, args=(d.name,), daemon=True)
                   for d in roster.enabled()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return self.failed

    def _worker(self, name):
        while True:
            denoiser = self._current(name)
            if denoiser is None:
                return          # disabled or removed: stop taking work
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
        try:
            ok, wall_s, fps, out_bytes = self.runner(clip, denoiser)
        except Exception as exc:
            ok, wall_s, fps, out_bytes = False, time.monotonic() - started, 0.0, 0
            with self._lock:
                self.failures.append((clip.src, denoiser.name, repr(exc)))
        finally:
            self._slots.release()

        append_record(self.state_path,
                      Record(src=clip.src, status="done" if ok else "failed",
                             denoiser=denoiser.name, wall_s=round(wall_s, 2),
                             fps=round(fps, 2), out_bytes=out_bytes))
        with self._lock:
            if ok:
                self.done += 1
            else:
                self.failed += 1
                if not self.failures or self.failures[-1][0] != clip.src:
                    self.failures.append((clip.src, denoiser.name, "runner reported failure"))
