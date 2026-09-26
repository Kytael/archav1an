"""Where the encode dashboard looks, and on which port it listens.

Every path here is one the batch already owns. The daemon reads them and
writes a few of them itself: `denoisers.toml` (the roster it edits through
rosterio), `batch.json` (the supervisor's claim on a live run) and the
control files it drops for the batch to pick up. It never sends the batch a
signal: it spawns or adopts it, and after that talks to it only through
files.
"""
import os
from dataclasses import dataclass

# Three dirnames, not the two archive-batch.py uses: that file sits in tools/
# and this one in tools/encode_dash/. Two would land on tools/ and the daemon
# would read a .archive-run no batch ever writes.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 9328 is free and sits in the 93xx range the fleet's other exporters use:
# Datasette 9325, OWUI telemetry 9326, rtk 9327. Checked on encoder-host 2026-08-16.
DEFAULT_PORT = 9328


@dataclass(frozen=True)
class Paths:
    run_dir: str
    state: str
    roster: str
    manifest: str
    # Directories, not files, unlike the three above. lanes holds one heartbeat
    # per denoiser; control is where this daemon drops request files for the
    # batch to pick up.
    lanes: str
    control: str
    # The batch writes this when its own load_roster fails. Separate from the
    # daemon's parse of the same file: the two processes read it at different
    # instants, and the batch's view is the one that says whether work is
    # happening (spec 5.5).
    roster_error: str
    # The batch writes this at startup and removes it on the way out. It is the
    # only liveness signal a PARKED run has: heartbeats exist only while a lane
    # holds a clip, and a run parked because every lane was yielded holds none.
    batch: str

    @classmethod
    def from_env(cls):
        """Resolve the run directory exactly as archive-batch.py:34 does.

        Diverging here would point the page at a different run from the one
        actually going, and the two would disagree with no error anywhere.
        """
        run_dir = os.environ.get("ARCHIVE_RUN_DIR") or os.path.join(REPO, ".archive-run")
        return cls(run_dir=run_dir,
                   state=os.path.join(run_dir, "state.jsonl"),
                   roster=os.path.join(run_dir, "denoisers.toml"),
                   manifest=os.path.join(run_dir, "manifest-raw.tsv"),
                   lanes=os.path.join(run_dir, "lanes"),
                   control=os.path.join(run_dir, "control"),
                   roster_error=os.path.join(run_dir, "roster-error.txt"),
                   batch=os.path.join(run_dir, "batch.json"))
