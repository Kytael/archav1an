"""Where the encode dashboard looks, and on which port it listens.

Every path here is one the batch already owns. The daemon reads them and, from
part 2, writes exactly one of them. It never talks to the batch process.
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
    # per denoiser; control is where part 2 drops request files for the batch to
    # pick up, and nothing in part 1 touches it.
    lanes: str
    control: str

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
                   control=os.path.join(run_dir, "control"))
