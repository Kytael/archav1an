"""One worker per enabled denoiser, all pulling from one queue.

The roster is re-read before each clip, so disabling a denoiser lets it finish
its current clip and then stop taking work: nothing is killed and no progress
is lost (spec 5.2). Disabling is a pause, not an exit -- a worker whose
denoiser has gone away parks and keeps re-checking, so re-enabling the device
mid-run puts it straight back to work (spec 7.1 gate 7).
"""
import os
import queue
import threading
import time

from dataclasses import replace

from .heartbeat import clear as clear_heartbeat, write as write_heartbeat
from .netresolve import NoTrustedAddress
from . import pidfile
from .state import MAX_ATTEMPTS, Record, append_record
from .transfer import TransferOutage
from tools.archive_batch.control import YieldRequested

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _SlotLane:
    """What a slot worker passes _process in a Denoiser's place.

    _process reads only `.name` from it once the slot is already reserved, and
    `.name` is `<encoder>-<slot>` -- which is what Record.denoiser carries for
    an encode job, what the heartbeat file is called, and what --temp-tag
    becomes. The field already means "which worker did this", and a slot worker
    is a worker.
    """
    __slots__ = ("name", "host")

    def __init__(self, name, host):
        self.name = name
        self.host = host


class Scheduler:
    POLL_SECONDS = 5.0      # how often a parked worker re-reads the roster
    BOUNCE_PAUSE = 0.5      # give another denoiser time to claim a bounced clip
    # How long an encoder that refused a clip before any work started stays out
    # of the picker. Long enough that a dead host is not retried per clip, short
    # enough that a laptop back on the LAN rejoins the pool within a coffee
    # break rather than at the next restart.
    QUARANTINE_SECONDS = 300.0

    def __init__(self, clips, roster_fn, runner, state_path, prior_failures=None,
                 lanes_dir=None, roster_error_path=None, clock=time.monotonic):
        """
        clips          -- ordered tuple of Clip still to do
        roster_fn      -- callable returning a fresh Roster, called before each clip
        runner         -- callable(clip, denoiser, encoder, slot)
                          -> (ok, wall_s, fps, out_bytes, reason)
        prior_failures -- {src: count} from earlier runs, so the in-run retry
                          respects the same MAX_ATTEMPTS ceiling as resume does
        clock          -- monotonic seconds, injectable so a test can run the
                          quarantine timer out without sleeping five minutes
        """
        self.queue = queue.Queue()
        for clip in clips:
            self.queue.put(clip)
        self.roster_fn = roster_fn
        self.runner = runner
        self.state_path = state_path
        # Where each worker publishes the clip it holds. None disables it, which
        # is what the older tests and any caller predating the UI expect.
        self.lanes_dir = lanes_dir
        self.roster_error_path = roster_error_path
        # The message from the last failed roster read, or None. The daemon
        # reads the file rather than this field -- it is a different process --
        # but tests and the run summary read the field.
        self.roster_error = None
        self._roster_said = None    # the last message actually printed
        self.done = 0
        self.failed = 0
        self.failures = []
        self._attempts = dict(prior_failures or {})
        # Which denoisers have already failed a clip, so a retry prefers another
        # device: some failures are specific to the device that took it (spec 6).
        self._failed_on = {}
        self._bounced = set()
        # The srcs a worker is inside the runner for. Only retry() reads it,
        # and only to refuse a second retry for a clip already being encoded.
        self._in_flight = set()
        # The encode-job subset of _in_flight, so "does denoise work remain"
        # is answerable without holding a Clip for every src. Maintained
        # wherever _in_flight is, and only ever a subset of it.
        self._encode_in_flight = set()
        # This process's /proc identity, published in every heartbeat so the
        # daemon can tell a live lane from a recycled pid (tools/archive_batch/
        # pidfile.py).
        self._pid_start = pidfile.start_time(os.getpid())
        self._lock = threading.Lock()
        # Serialises the error FILE only. Separate from _lock, which must not
        # cover I/O: six workers take it on every clip.
        self._roster_file_lock = threading.Lock()
        self._stop = threading.Event()
        # Set when there is nothing left to take, so parked workers wake at once
        # instead of sitting out a whole poll interval after the queue drains.
        self._wake = threading.Event()
        roster = roster_fn()
        # One semaphore per encoder, not one for the fleet. Same deliberate
        # asymmetry as before: enablement is live and re-read per clip, but the
        # slot counts are fixed for the life of the run. A count that changed
        # mid-run would have to grow or shrink a semaphore that workers are
        # already blocked on, and there is no safe way to shrink one.
        self._pool = {e.name: threading.Semaphore(max(1, e.slots))
                      for e in roster.encoders}
        # The same startup roster the semaphores were sized from. Every later
        # read of slots or port_base comes from here, never from the live
        # entry: the two must agree, and only one of them can be a snapshot.
        self._fixed = {e.name: (max(1, e.slots), e.port_base)
                       for e in roster.encoders}
        self._ports = {}
        # encoder name -> the clock reading at which it rejoins the picker.
        self._quarantined = {}
        self._clock = clock
        # A file from a run that died with a broken roster is indistinguishable
        # from a live one to the daemon, which believes it and parks the page
        # forever. _roster_ok's early return (both fields still None) means the
        # first successful read after a restart would never reach
        # _clear_roster_error on its own, so a fresh Scheduler clears it here.
        self._clear_roster_error()

    def stop(self):
        """Wind the run down: workers finish the clip they hold and take no more."""
        self._stop.set()
        self._wake.set()

    def retry(self, clip):
        """Put a clip past its attempt ceiling back on the queue.

        Spec 5.4. The state.jsonl record makes the next run eligible; this
        makes this one eligible, which is what matters in a fifteen-day job.

        _bounced goes with the rest. It holds (src, lane) pairs meaning "this
        lane already gave that clip to someone else once", and leaving the pair
        behind would make _should_bounce hand the clip straight back to the
        lane that just failed it twice.

        Returns False if the clip is already queued or already being encoded,
        and does nothing. Two retries for one clip would put the same Clip on
        the queue twice, two lanes would encode it, and both would publish to
        the same destination path at the same time -- a corrupted output over
        the archive, for one double-click. NOT _wake.set(): see the arm in
        _process below.
        """
        with self._lock:
            if clip.src in self._in_flight or any(
                    c.src == clip.src for c in list(self.queue.queue)):
                return False
            self._attempts.pop(clip.src, None)
            self._failed_on.pop(clip.src, None)
            self._bounced = {b for b in self._bounced if b[0] != clip.src}
        self.queue.put(clip)
        return True

    def _take(self, want_denoise):
        """Take the oldest job matching the flag, or None. Order preserved.

        get_nowait cannot pass over a job, and both worker populations share
        one queue, so a lane worker must be able to leave an encode job where
        it is. Drains under the lock into a list, removes the first match and
        puts the rest back in order.

        Cost is a queue scan per JOB, not per frame, on a queue of a few
        thousand at most.
        """
        with self._lock:
            rest = []
            found = None
            while True:
                try:
                    clip = self.queue.get_nowait()
                except queue.Empty:
                    break
                if found is None and clip.denoise == want_denoise:
                    found = clip
                else:
                    rest.append(clip)
            for clip in rest:
                self.queue.put(clip)
            return found

    def _has(self, want_denoise):
        """Is any job of this kind still queued? Under the lock, see _empty."""
        with self._lock:
            return any(c.denoise == want_denoise for c in list(self.queue.queue))

    def _empty(self):
        """Queue emptiness, under the lock.

        _take empties the queue and refills it, so an UNLOCKED read can land in
        that window, see zero, and retire a worker for the rest of a run with
        thousands of clips still to do.
        """
        with self._lock:
            return self.queue.empty()

    def _lane_can_use(self, encoder_name):
        """Can any ENABLED lane reach this encoder?

        Gates the reserve below. A slot held back for a lane that is switched
        off, or whose allowlist never names this host, is not a reserve; it is
        capacity nothing can ever claim. With every lane disabled that cost the
        pool one slot on every encoder, for the whole run.
        """
        roster = self._read_roster()
        if roster is None:
            return True     # unreadable: keep the reserve, the cautious way round
        return any(d.allows(encoder_name) for d in roster.enabled())

    def _denoise_work_remains(self):
        """A denoise job queued or in flight. Drives the slot-priority rule."""
        with self._lock:
            if any(c.denoise for c in list(self.queue.queue)):
                return True
            return bool(self._in_flight - self._encode_in_flight)

    def submit(self, clips):
        """Queue new jobs mid-run. Returns how many were queued.

        retry()'s shape: a src already queued or in flight is skipped rather
        than duplicated, because two copies of one job would be encoded twice
        and published to one destination path at the same time.

        A submission into a DRAINED run queues nothing useful: slot workers end
        when the queue empties, exactly as lane workers do, so a submission has
        a worker waiting for it only while the run still has work. That matches
        the case this exists for -- a folder submitted while the archive is
        still draining -- and is the price of not inventing a run that never
        ends.
        """
        queued = 0
        with self._lock:
            known = self._in_flight | {c.src for c in list(self.queue.queue)}
            fresh = [c for c in clips if c.src not in known]
            for clip in fresh:
                self.queue.put(clip)
                queued += 1
        if queued:
            # Slot workers park on _wake between polls; without this a
            # submission waits out a full POLL_SECONDS before anything moves.
            self._wake.set()
            self._wake.clear()
        return queued

    def run(self):
        # Every rostered denoiser gets a worker, not only the enabled ones: a
        # worker for a disabled device parks immediately and costs nothing, but
        # without it, switching from roster B back to roster A mid-run would do
        # nothing at all. The supervise loop below re-reads the roster on each
        # wake, so a denoiser ADDED to the file after the run starts gets a
        # worker too, and adding a host no longer needs a restart.
        started = set()
        pending = self._spawn(started, self._names())

        if not pending and not self.queue.empty():
            # The roster is read once more here, by _names(), between
            # construction (which already read it once, for the slot count)
            # and this call -- a narrow window, but on a fortnight-long job
            # even a narrow window deserves a loud failure. _names() swallows
            # the parse error on purpose, because that is exactly what lets a
            # mid-run re-read tolerate a momentarily broken file -- but that
            # same swallowing means a broken roster right here would otherwise
            # return with zero workers started and not a line of output,
            # indistinguishable from a run that simply had nothing to do.
            print(f"archive-batch: the roster could not be read, so no worker "
                  f"was started and {self.queue.qsize()} clip(s) are still "
                  f"queued. Fix the roster file and re-run.", flush=True)
            return

        # AFTER the guard above, never before it. Spawned above, slot workers
        # ARE the pending list when the roster will not parse: no lane worker
        # starts, the guard sees live threads and stays quiet, and run() then
        # waits for ever on workers that cannot drain a queue of denoise jobs.
        #
        # One per slot of every ROSTERED encoder, disabled ones included, the
        # same rule lane workers follow -- a worker for a disabled host parks
        # and costs nothing, and without it re-enabling a host mid-run would
        # give it no worker. The slot count comes from the startup snapshot in
        # _fixed, so it agrees with the semaphore that admits them.
        for name, (slots, _port) in self._fixed.items():
            for ordinal in range(max(1, slots)):
                t = threading.Thread(target=self._slot_worker, args=(name,),
                                     daemon=True)
                t.start()
                pending.append(t)

        # The reason last printed, so a condition that persists is said once
        # and a condition that changes is said again. Quarantine can clear on
        # its own, so this is no longer a latch.
        parked = None
        while pending:
            pending[0].join(self.POLL_SECONDS)
            # Before pruning: a name that appeared while we slept must get its
            # thread even on the wake that retires the last of the old ones.
            pending += self._spawn(started, self._names())
            pending = [t for t in pending if t.is_alive()]
            if not pending:
                continue
            reason = self._park_reason(started)
            if reason == parked:
                continue
            parked = reason
            if reason:
                print(f"archive-batch: {reason}", flush=True)

    def _read_roster(self):
        """A fresh roster, or None. Records the failure instead of hiding it.

        Every roster read in this class goes through here. Before spec 5.5 the
        four readers each swallowed their own exception and returned a
        fallback, so one typo in the TOML parked every lane forever with no
        message anywhere. Parking is still the right behaviour; saying nothing
        was not.
        """
        try:
            roster = self.roster_fn()
        except Exception as exc:
            self._roster_broke(exc)
            return None
        self._roster_ok()
        return roster

    def _roster_broke(self, exc):
        message = str(exc) or repr(exc)
        with self._lock:
            self.roster_error = message
            fresh = message != self._roster_said
            self._roster_said = message
        if not fresh:
            # Once per distinct message. This runs on every worker's every
            # loop, so printing unconditionally would be tens of thousands of
            # identical lines a day -- which reads the same as silence.
            return
        print(f"archive-batch: the roster will not parse, so every lane parks "
              f"until it does: {message}", flush=True)
        self._publish_roster_error()

    def _roster_ok(self):
        with self._lock:
            if self.roster_error is None and self._roster_said is None:
                return
            self.roster_error = None
            self._roster_said = None
        print("archive-batch: the roster parses again; the lanes resume.",
              flush=True)
        self._publish_roster_error()

    def _publish_roster_error(self):
        """Make the file match the current state, whoever gets here first.

        Both callers decide under _lock and act after releasing it, so without
        this a writer preempted between the two can recreate a file that a
        recovering thread has already removed -- and _roster_ok's guard then
        never clears it again. Re-reading the live state inside this lock makes
        the file converge instead of depending on thread order.
        """
        with self._roster_file_lock:
            with self._lock:
                message = self.roster_error
            if message is None:
                self._clear_roster_error()
            else:
                self._write_roster_error(message)

    def _write_roster_error(self, message):
        """Publish the message for the daemon. Best effort, like the heartbeat.

        A run must not die because it could not write its own error file.
        """
        if not self.roster_error_path:
            return
        try:
            # Per-thread tmp name. _roster_broke writes outside the lock, and
            # six lanes can record two different parse errors within one poll:
            # on a shared tmp path one worker's os.replace publishes the
            # other's text and the loser fails with ENOENT, printing a write
            # error that has no real cause during the incident being
            # diagnosed.
            tmp = f"{self.roster_error_path}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(message + "\n")
            os.replace(tmp, self.roster_error_path)
        except Exception as exc:
            print(f"archive-batch: cannot write {self.roster_error_path}: "
                  f"{exc}", flush=True)

    def _clear_roster_error(self):
        if not self.roster_error_path:
            return
        try:
            os.remove(self.roster_error_path)
        except OSError:
            pass

    def _names(self):
        """Every rostered name right now, or () if the roster will not parse.

        A roster that stops parsing must not add or remove workers; the ones
        already running keep parking and re-checking. _read_roster records the
        parse error so the page and the console both learn about it.
        """
        roster = self._read_roster()
        return () if roster is None else [d.name for d in roster.denoisers]

    def _spawn(self, started, names):
        """Start a worker for each name not started yet. Returns the new threads."""
        made = []
        for name in names:
            if name in started:
                continue
            started.add(name)
            t = threading.Thread(target=self._worker, args=(name,), daemon=True)
            t.start()
            made.append(t)
        return made

    def _park_reason(self, names):
        """Why a lane cannot move, or None while every lane can.

        Three stalls look identical from outside -- every lane off, every
        encoder a lane may use off, every one of them quarantined -- and only
        the last clears itself, so the operator has to be told which. Reported
        per lane, not for the fleet: one wedged lane beside five healthy ones
        drains the queue and looks fine, and that is the case where a silent
        stall goes unnoticed longest.

        The encoder half exists because roster._validate cannot be the only
        guard: encoders are toggled from the dashboard mid-run, and until this
        the picker waited for ever without printing a line.
        """
        roster = self._read_roster()
        if roster is None:
            return None     # _roster_broke has already said so, in its own words
        lanes = [d for d in roster.enabled() if d.name in names]
        with self._lock:
            waiting = self.queue.qsize() + len(self._in_flight)
        if not waiting:
            return None     # nothing left to stall on; the run is finishing
        if not lanes:
            return (f"every denoiser is disabled and {waiting} clip(s) are "
                    f"still to do. Workers are parked, not stuck -- re-enable a "
                    f"denoiser in the roster to resume.")
        on = roster.enabled_encoders()
        live = [e for e in on if not self._is_quarantined(e.name)]
        starved = [d for d in lanes
                   if not any(d.allows(e.name) for e in live)]
        if not starved:
            return None
        # A lane that quarantine alone is starving needs nobody: it comes back
        # when the timer runs out. One that a disabled encoder is starving
        # needs a toggle. If both are stalled, name the one somebody has to
        # act on -- the other fixes itself either way.
        needs_hands = [d for d in starved
                       if not any(d.allows(e.name) for e in on)]
        if not needs_hands:
            return (f"every encoder these lanes may use is quarantined and "
                    f"{waiting} clip(s) are still to do: "
                    f"{', '.join(d.name for d in starved)}. This clears itself "
                    f"within {self.QUARANTINE_SECONDS:.0f}s if the host comes "
                    f"back; nothing needs doing.")
        allows = ", ".join(f"{d.name} allows {sorted(d.encoders)}"
                           for d in needs_hands)
        return (f"no enabled encoder is allowed by these lanes, so they can "
                f"never take a clip and {waiting} clip(s) are still to do: "
                f"{allows}. Enable one of those encoders or widen the "
                f"allowlist.")

    def _worker(self, name):
        while True:
            if self._stop.is_set() or self._empty():
                self._wake.set()    # nothing left to take: release parked workers
                return
            denoiser = self._current(name)
            if denoiser is None:
                # Disabled or removed. Do not exit: the roster is live and the
                # user may re-enable this device mid-run (spec 7.1 gate 7).
                # Why we woke does not matter -- re-check both conditions above.
                self._wake.wait(self.POLL_SECONDS)
                continue
            # want_denoise=True: a lane worker never takes an encode job. A
            # queue that still holds encode jobs is not empty, so returning
            # here would retire the lane; wait instead, because a denoise job
            # can still arrive by retry.
            clip = self._take(True)
            if clip is None:
                if not self._has(True):
                    self._wake.set()
                    return
                self._wake.wait(self.POLL_SECONDS)
                continue
            # Register NOW, not in _process: between get_nowait and that
            # registration the clip sits in neither the queue nor _in_flight,
            # and a retry arriving in the window would queue a second copy --
            # two lanes on one destination path.
            with self._lock:
                self._in_flight.add(clip.src)
            if self._should_bounce(clip, name):
                # This device already failed this clip and another device is
                # enabled. Put it back and pause so that device can claim it.
                # Bounded to one bounce per clip per worker, so if the other
                # device stays busy this one takes the clip anyway.
                # One lock across put and discard: split apart, another
                # worker could take the clip between them and its in-flight
                # mark would be erased by ours -- a retry could then queue a
                # second copy while the clip is being processed.
                with self._lock:
                    self.queue.put(clip)
                    self._in_flight.discard(clip.src)
                    self._encode_in_flight.discard(clip.src)
                time.sleep(self.BOUNCE_PAUSE)
                continue
            try:
                self._process(clip, denoiser)
            except Exception as exc:
                # A worker that dies here strands its GPU for the rest of the
                # run, silently. Never let that happen.
                print(f"archive-batch: worker {name} hit {exc!r} outside the "
                      f"runner; continuing.", flush=True)

    def _slot_worker(self, encoder_name):
        """One worker per encoder slot, taking only encode jobs.

        Bound to its encoder for the life of the run; it never picks. The order
        below is slot first, then job, because a worker holding a slot while it
        waits for work would idle the host -- so the slot goes back the moment
        there is no job for it.
        """
        while True:
            if self._stop.is_set() or self._empty():
                self._wake.set()
                return
            # Cheapest check first. A run with no encode job at all -- every
            # run today -- then costs one queue scan per slot per poll, rather
            # than a roster read and a semaphore round trip.
            if not self._has(False):
                if not self._has(True):
                    self._wake.set()
                    return
                self._wake.wait(self.POLL_SECONDS)
                continue
            encoder = self._current_encoder(encoder_name)
            if encoder is None or self._is_quarantined(encoder_name):
                # Disabled, removed from the roster, or quarantined. Park and
                # re-read, the same live-roster contract a lane worker has.
                self._wake.wait(self.POLL_SECONDS)
                continue
            taken = self._try_take(encoder)
            if taken is None:
                self._wake.wait(self.POLL_SECONDS)
                continue
            # Denoise jobs win. Leave the LAST free slot on this encoder for
            # the lanes while archive work is still queued or in flight: a
            # lane can only reach this host through a slot, and an encoder
            # with slots = 1 therefore runs encode jobs only once the denoise
            # work is done. Checked after the acquire, because "how many are
            # in use" is only meaningful once ours is one of them -- and only
            # when a lane could actually claim it, or the reserve is held for
            # nobody.
            if (self._slots_in_use(encoder.name) >= self._fixed[encoder.name][0]
                    and self._denoise_work_remains()
                    and self._lane_can_use(encoder.name)):
                self._release_slot(*taken)
                self._wake.wait(self.POLL_SECONDS)
                continue
            clip = self._take(False)
            if clip is None:
                # Raced with another slot worker for the last encode job.
                self._release_slot(*taken)
                continue
            name = f"{encoder.name}-{taken[1]}"
            try:
                self._process(clip, _SlotLane(name, encoder.host), taken=taken)
            except Exception as exc:
                print(f"archive-batch: slot worker {name} hit {exc!r} outside "
                      f"the runner; continuing.", flush=True)

    def _current_encoder(self, name):
        """This encoder from a freshly read roster, or None if it is gone or
        disabled. The encoder half of _current."""
        roster = self._read_roster()
        if roster is None:
            return None
        for e in roster.enabled_encoders():
            if e.name == name:
                return e
        return None

    def _should_bounce(self, clip, name):
        with self._lock:
            if name not in self._failed_on.get(clip.src, ()):
                return False
            if (clip.src, name) in self._bounced:
                return False    # already gave the others a turn; take it now
        if not self._other_enabled(name):
            return False        # no one else to hand it to
        with self._lock:
            self._bounced.add((clip.src, name))
        return True

    def _other_enabled(self, name):
        roster = self._read_roster()
        if roster is None:
            return False
        return any(d.name != name for d in roster.enabled())

    def _current(self, name):
        """Look this denoiser up in a freshly read roster; None if it is gone."""
        roster = self._read_roster()
        if roster is None:
            return None
        for d in roster.enabled():
            if d.name == name:
                return d
        return None

    @staticmethod
    def temp_dir_for(name, stem):
        """Must match make_runner in archive-batch.py, which builds
        Temp/<lane>/<stem>. The heartbeat carries it so the daemon never has to
        rebuild this path and drift from it."""
        return os.path.join(REPO, "Temp", name, stem)

    def _beat(self, name, clip, state):
        if not self.lanes_dir:
            return
        try:
            write_heartbeat(self.lanes_dir, lane=name, src=clip.src,
                            frames=clip.frames, state=state,
                            started_at=time.time(), batch_pid=os.getpid(),
                            attempt=self._attempts.get(clip.src, 0) + 1,
                            temp_dir=self.temp_dir_for(name, clip.stem),
                            pid_start=self._pid_start)
        except Exception as exc:
            # A dashboard must never be able to stop the run.
            print(f"archive-batch: heartbeat for {name} failed: {exc!r}",
                  flush=True)

    def _unbeat(self, name):
        if self.lanes_dir:
            clear_heartbeat(self.lanes_dir, name)

    def _take_slot(self, denoiser):
        """Block until a slot frees on an encoder this lane may use.

        There is a preference for the lane's own host, added 2026-09-02 after
        2026-08-25 deliberately left it out for simplicity. A lane that encodes
        where it denoised hands its y4m to a listener on its own address, which
        the kernel short-circuits to loopback, instead of pushing every
        denoised frame across the LAN. It is a tiebreak inside each idleness
        tier, never above it: promoting it over idle-first would put gpu4 on
        its own second slot while gpu5 sat empty, which is the exact
        regression the tier below was added to fix.

        There is also a preference for an encoder with nothing in flight, added
        2026-08-26. A slot is not a unit of capacity the way the first-free
        picker assumed: a second clip on a host that is already encoding does
        not get a second host's worth of CPU, it shares the cores, and both
        clips slow down. Measured on the gate run -- gpu4 carried two encodes
        at load 18 on 20 cores and fed gpu1's 4090 at 10.0 fps against the
        14.2 it produces, while gpu5 sat at load 1.7 with no encode at all.
        Roster order reached gpu4 first and it still had a free slot, so
        gpu5 was never asked. Filling breadth-first spends the fleet's cores
        before it doubles up on any one host.

        Returns (encoder, slot_index), or None when the run is winding down.
        """
        while not self._stop.is_set():
            self._expire_quarantine()
            roster = self._read_roster()
            if roster is not None:
                usable = [e for e in roster.enabled_encoders()
                          if denoiser.allows(e.name)
                          and not self._is_quarantined(e.name)]
                # Own box first. The sort is stable, so roster order still
                # decides inside each group -- both the lane's own encoders
                # and everybody else's stay in the order the file lists them.
                usable.sort(key=lambda e: e.host != denoiser.host)
                # Idle hosts first, then anything with a free slot. Reading the
                # in-use count and then acquiring is check-then-act, and two
                # workers can both see one encoder idle and both land on it.
                # That costs one placement, never correctness: the semaphore
                # and _slot_index stay authoritative, and the next clip sees
                # the true count. A lock spanning both would have to be held
                # across the acquire of every candidate in turn.
                for idle_only in (True, False):
                    for e in usable:
                        if idle_only and self._slots_in_use(e.name):
                            continue
                        taken = self._try_take(e)
                        if taken is not None:
                            return taken
            # Nothing free. Wait rather than spin: six workers polling a busy
            # pool would otherwise burn a core between them. On _stop and not
            # _wake, which the workers set for the rest of the run the moment
            # the queue drains and nobody ever clears -- waiting on that is
            # not a wait at all. _stop is the only event that means anything
            # here anyway: a drained queue frees no encode slot.
            self._stop.wait(0.25)
        return None

    def _quarantine_encoder(self, name, reason):
        """Take an encoder out of the picker for QUARANTINE_SECONDS.

        The refused clip goes back on the queue, so without this it is handed
        straight back to the same dead encoder: a hot loop, not a retry. Timed
        rather than disabled for the run, because gpu2 and gpu3 move -- a
        host that walks off the LAN must rejoin the pool when it walks back,
        with nobody editing the roster.
        """
        with self._lock:
            fresh = name not in self._quarantined
            self._quarantined[name] = self._clock() + self.QUARANTINE_SECONDS
        if not fresh:
            # Two lanes can refuse on the same encoder before either of them
            # gets here. One incident, one line.
            return
        print(f"archive-batch: encoder {name} refused before any work started, "
              f"so it leaves the pool for {self.QUARANTINE_SECONDS:.0f}s and no "
              f"clip is charged for it: {reason}", flush=True)

    def _is_quarantined(self, name):
        with self._lock:
            return name in self._quarantined

    def _expire_quarantine(self):
        """Put back every encoder whose timer has run out, saying so once."""
        now = self._clock()
        with self._lock:
            back = [n for n, until in self._quarantined.items() if until <= now]
            for name in back:
                del self._quarantined[name]
        for name in back:
            print(f"archive-batch: encoder {name} is out of quarantine and back "
                  f"in the pool.", flush=True)

    def _slots_in_use(self, name):
        """How many clips this encoder is carrying right now."""
        with self._lock:
            return len(self._ports.get(name, ()))

    def _try_take(self, encoder):
        """Reserve one slot on `encoder`, or None when it has none free."""
        sem = self._pool.get(encoder.name)
        # An encoder added to the file mid-run has no semaphore: the pool is
        # built once, like the slot count it replaces.
        if sem is None or not sem.acquire(blocking=False):
            return None
        # Slot count and port block from the startup snapshot, so editing
        # either mid-run cannot disagree with the semaphore that just admitted
        # us. Lowering slots used to admit a worker for which no index existed.
        slots, port_base = self._fixed[encoder.name]
        encoder = replace(encoder, slots=slots, port_base=port_base)
        try:
            return encoder, self._slot_index(encoder)
        except BaseException:
            # The only window in the run where a permit can be taken and never
            # given back: _process's try/finally does not start until this
            # returns.
            sem.release()
            raise

    def _slot_index(self, encoder):
        """Which port inside this encoder's block this acquisition gets.

        Slots are interchangeable, so any free index will do -- but two clips
        on one encode host must never share a port, or the second binds onto
        the first's listener.
        """
        with self._lock:
            used = self._ports.setdefault(encoder.name, set())
            for i in range(encoder.slots):
                if i not in used:
                    used.add(i)
                    return i
        # Unreachable: the semaphore admitted us, so an index is free.
        raise RuntimeError(f"no free slot index on encoder {encoder.name}")

    def _release_slot(self, encoder, slot):
        with self._lock:
            self._ports.get(encoder.name, set()).discard(slot)
        self._pool[encoder.name].release()

    def _process(self, clip, denoiser, taken=None):
        """Run one job. `taken` is a slot already reserved by the caller.

        A slot worker reserves its own encoder's slot before it takes a job,
        so that it never holds a slot while waiting for work. It passes the
        reservation in rather than letting the picker choose, because a slot
        worker is bound to one encoder for the life of the run.
        """
        held = taken is not None
        # Before the beat and before the slot wait, not after. The acquire
        # below can block for a whole clip, and for that entire window the
        # clip is already out of the queue -- so a retry arriving then would
        # see it nowhere, queue a second copy, and put two lanes on one
        # destination path.
        with self._lock:
            self._in_flight.add(clip.src)
            if clip.is_encode_job:
                self._encode_in_flight.add(clip.src)
        # Publish BEFORE acquiring, and say which of the two states this is.
        # The acquire below can block for a whole clip, and a lane waiting on an
        # encode slot must not be indistinguishable from one that is denoising:
        # its ETA would be wrong for the entire wait.
        if not held:
            self._beat(denoiser.name, clip, "waiting_for_slot")
            taken = self._take_slot(denoiser)
        if taken is None:
            # Winding down before a slot came free. Put the clip back
            # untouched: no attempt spent and nothing recorded, the same
            # contract as the YieldRequested arm below.
            with self._lock:
                self.queue.put(clip)
                self._in_flight.discard(clip.src)
                self._encode_in_flight.discard(clip.src)
            return
        encoder, slot = taken
        self._beat(denoiser.name, clip, "working")
        # After the acquire, not before. This feeds wall_s on the raising path
        # below, and that figure is meant to be how long the work took. Started
        # before the acquire, a clip that queued two hours for a slot and then
        # crashed instantly would be recorded as a two-hour failure, which reads
        # as a hang and is the wrong thing to go looking for.
        started = time.monotonic()
        raised = False
        reason = ""
        phases = {}
        # Defined before the try so the finally can read it even when the
        # runner raises something except Exception does not catch.
        ok = False
        # Set by the arms that put the clip back themselves: they requeue
        # under the lock together with the discard, so the finally must not
        # discard again -- a second discard could erase a taking worker's
        # fresh in-flight mark.
        requeued = False
        # Set on every path that reaches the retry decision after this
        # try/finally. Cleared, it means the runner raised past
        # `except Exception` -- SystemExit, KeyboardInterrupt -- and the
        # finally is the last code that runs for this clip, so the mark has
        # to go there or it is held for the rest of the run: the clip sits in
        # neither the queue nor _in_flight, and retry() refuses it for ever.
        decided_below = False
        try:
            result = self.runner(clip, denoiser, encoder, slot)
            # A runner may report the stage/work/publish split or not. Tests and
            # any older caller return five fields; accept both rather than make
            # the timing breakdown a hard part of the contract.
            if len(result) == 6:
                ok, wall_s, fps, out_bytes, reason, phases = result
            else:
                ok, wall_s, fps, out_bytes, reason = result
            decided_below = True
        except TransferOutage as exc:
            # Staging and publishing both target the source host, so this stops
            # every denoiser, not just this one. Put the clip back untouched and
            # wind down: recording it failed would spend an attempt on a clip
            # that was never tried (spec 6).
            with self._lock:
                self.queue.put(clip)
                self._in_flight.discard(clip.src)
                self._encode_in_flight.discard(clip.src)
            requeued = True
            print(f"archive-batch: {exc}\n"
                  f"archive-batch: the source host is unreachable. Stopping with "
                  f"{self.queue.qsize()} clip(s) queued and nothing recorded "
                  f"failed. Re-run to resume.", flush=True)
            self.stop()
            return
        except YieldRequested as exc:
            # Same shape as the TransferOutage arm above -- the clip goes back
            # untouched, no attempt is spent, nothing reaches state.jsonl --
            # and deliberately WITHOUT the stop(). That one line is the whole
            # difference: a transfer outage hits every lane through the one
            # source host, so the run winds down; a yield hits the lane the
            # operator named, and the other lanes must not notice.
            with self._lock:
                self.queue.put(clip)
                self._in_flight.discard(clip.src)
                self._encode_in_flight.discard(clip.src)
            requeued = True
            print(f"archive-batch: {exc}. {clip.src} is back on the queue with "
                  f"its attempt count unmoved.", flush=True)
            return
        except NoTrustedAddress as exc:
            # A PRE-FLIGHT refusal: raised while resolving the encoder's bind
            # address, before a single frame is staged, so it says nothing
            # about the clip. Same contract as the two arms above -- clip back
            # untouched, no attempt spent, nothing in state.jsonl -- plus the
            # quarantine, which the others do not need. Charged to the clip it
            # was a failure in 0.3 s, and a fast-failing encoder is always the
            # free one, so the picker fed it the whole queue twice over and
            # pending_clips then dropped those clips from every future run.
            with self._lock:
                self.queue.put(clip)
                self._in_flight.discard(clip.src)
                self._encode_in_flight.discard(clip.src)
            requeued = True
            self._quarantine_encoder(encoder.name, str(exc))
            return
        except Exception as exc:
            ok, wall_s, fps, out_bytes = False, time.monotonic() - started, 0.0, 0
            reason = repr(exc)
            raised = True
            decided_below = True
            with self._lock:
                self.failures.append((clip.src, denoiser.name, reason))
        finally:
            self._release_slot(encoder, slot)
            self._unbeat(denoiser.name)
            # A failed run decides on its retry only AFTER the state write
            # below, and until that put lands the clip must stay in
            # _in_flight: dropped here, the whole append_record + stats span
            # is a window where the clip sits in neither the queue nor the
            # set, and a retry() queues a second copy. So the mark is kept
            # on every non-requeued failure and dropped at the decision
            # points below.
            #
            # It is dropped here on the two paths that have no decision
            # point: a clean success, which the block below returns from
            # without touching the mark, and a runner that raised past
            # `except Exception`, where nothing below this line runs at all.
            # The arms that already requeued under their own lock are the
            # exception -- discarding again could erase the mark of the
            # worker that has since taken the clip.
            if ok or not (decided_below or requeued):
                with self._lock:
                    self._in_flight.discard(clip.src)
                    self._encode_in_flight.discard(clip.src)

        try:
            append_record(self.state_path,
                          Record(src=clip.src, status="done" if ok else "failed",
                                 denoiser=denoiser.name, encode_host=encoder.name,
                                 wall_s=round(wall_s, 2),
                                 fps=round(fps, 2), out_bytes=out_bytes,
                                 reason="" if ok else reason,
                                 stage_s=round(phases.get("stage_s", 0.0), 2),
                                 work_s=round(phases.get("work_s", 0.0), 2),
                                 publish_s=round(phases.get("publish_s", 0.0), 2)))
        except Exception as exc:
            # Losing the state file loses resume, so stop the run out loud rather
            # than let workers die one by one with nothing written down.
            print(f"archive-batch: cannot write state to {self.state_path}: {exc!r} "
                  f"-- stopping the run", flush=True)
            self.stop()

        with self._lock:
            if ok:
                self.done += 1
                return
            self.failed += 1
            if not raised:
                self.failures.append((clip.src, denoiser.name,
                                      reason or "runner reported failure"))
            attempts = self._attempts.get(clip.src, 0) + 1
            self._attempts[clip.src] = attempts
            self._failed_on.setdefault(clip.src, set()).add(denoiser.name)
            retry = attempts < MAX_ATTEMPTS and not self._stop.is_set()
            # The decision is made: either requeue below with the mark still
            # held, or drop it here -- both under this one lock, so no
            # retry() can see the clip in neither place.
            if not retry:
                self._in_flight.discard(clip.src)
                self._encode_in_flight.discard(clip.src)

        if retry:
            # Retry now rather than on the next run: over 15 days a device
            # specific failure would otherwise wait days for its second try.
            # _should_bounce steers it to a different denoiser if one is free.
            # One lock across put and discard: split apart, another worker
            # could take the clip between them and its in-flight mark would
            # be erased by ours -- the duplicate-retry window again.
            with self._lock:
                self.queue.put(clip)
                self._in_flight.discard(clip.src)
                self._encode_in_flight.discard(clip.src)
