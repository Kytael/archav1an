#!/usr/bin/env python3
# tools/encode-dash.py
"""The archive run's dashboard. Runs beside archive-batch, never inside it.

    /opt/archav1an/venv/bin/python tools/encode-dash.py

Reads .archive-run/ and serves a page, a JSON snapshot and a Prometheus
endpoint. It holds no state of its own except the rate history, so it can be
restarted at any point in a fifteen-day run without touching the run.

It writes denoisers.toml and one request file per action into
.archive-run/control/. It also spawns the batch when the page asks and adopts
an existing run on restart, but it never signals the batch process directly:
the scheduler re-reads the roster per clip, and the batch polls that directory.
Stop stays graceful, through the control channel, not a signal.

Binding: the default is the Tailscale address, because that is how the fleet
reaches this host and because encoder-host's firewall already refuses high ports on
the LAN. There is no authentication, which matches every other service here --
llama-swap, ComfyUI, Datasette and opencode are all open on the tailnet. Unlike
those, this one edits the roster a running job reads, so anything on the
tailnet can park a lane; the tailnet is the trust boundary.
"""
import argparse
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.encode_dash import DEFAULT_PORT, Paths          # noqa: E402
from tools.encode_dash.liverate import RateTracker         # noqa: E402
from tools.encode_dash.model import snapshot               # noqa: E402
from tools.encode_dash.server import make_server           # noqa: E402
from tools.encode_dash.supervisor import Supervisor        # noqa: E402

# The checkout root. This file sits in tools/, so two dirname hops land on the
# root. The batch script lives in the same checkout, and the batch computes its
# own paths from its own __file__ -- but spawning with this cwd keeps every
# relative path a subprocess sees identical to the daemon's view.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_SCRIPT = os.path.join(REPO, "tools", "archive-batch.py")


def _tailscale_address():
    """This host's tailnet address, or None.

    Never guess a LAN address instead: encoder-host's firewall refuses high ports
    there, so binding one would produce a daemon that answers only itself.
    """
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                             text=True, timeout=5).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    # -4 prints one line per IPv4 address. A host with a second one would be
    # unusual here, and the first is the tailnet address in either case.
    return out[0].strip() if out else None


def make_snapshot(paths, tracker, supervisor, grafana_url):
    """The function the server calls for every poll.

    grafana_url is added here rather than inside snapshot() because it is not
    a measurement. Everything model.py produces is read off the run; this is a
    command-line flag, and mixing the two would put deployment settings in the
    middle of the thing that reports facts.

    A module-level factory rather than a closure in main() so it can be
    tested: every route the daemon serves goes through this one call, and the
    first version of it shipped a crash that no test could reach.
    """
    def _snapshot():
        # Reaps the batch this daemon spawned, if it has exited. Nothing else
        # waits on it -- there is no SIGCHLD handler here, and Popen only
        # reaps inside the next Popen -- so without this call a SIGKILLed
        # batch stays a zombie, os.kill(pid, 0) keeps succeeding on it, and
        # the stale batch.json it left behind reads as a live run for ever:
        # encode_batch_up pinned at 1, EncodeBatchDown never fired, Stop
        # offered for a run that is over.
        #
        # The result is deliberately not merged into the snapshot. The page
        # must call the run running only once the batch itself has claimed
        # batch.json, because a Stop written before that point is cleared as
        # stale by the batch's own startup.
        if supervisor is not None:
            supervisor.status()
        # time.time(), not monotonic: the heartbeat records wall clock because
        # a person reads it, and model._lane subtracts the two. Mixing the
        # clocks would give every lane a nonsense elapsed time. A run measured
        # in days does not care about a one-second NTP correction.
        return dict(snapshot(paths, tracker, time.time()),
                    grafana_url=grafana_url)
    return _snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=None,
                    help="bind address; default is this host's Tailscale IP, "
                         "falling back to 127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--grafana-url", default=None,
                    help="link the page's 'history' button at this URL. Left "
                         "out, the button is hidden: where the history lives "
                         "is a property of one deployment, not of this program.")
    ap.add_argument("--smooth", type=float, default=30.0,
                    help="seconds to average the live rate over. A windowed "
                         "lane bursts a whole window at once, so this must "
                         "cover at least one sweep or the figure alternates "
                         "between zero and a spike.")
    args = ap.parse_args()

    host = args.host or _tailscale_address() or "127.0.0.1"
    paths = Paths.from_env()
    tracker = RateTracker(smooth_s=args.smooth)

    # Own or adopt the batch process. Spawned with this checkout's interpreter
    # and the same environment, so ARCHIVE_RUN_DIR and friends reach the batch
    # exactly as the daemon sees them.
    supervisor = Supervisor([sys.executable, BATCH_SCRIPT], cwd=REPO,
                            batch_file=paths.batch)
    supervisor.adopt()

    srv = make_server(host, args.port,
                      make_snapshot(paths, tracker, supervisor,
                                    args.grafana_url),
                      roster_path=paths.roster,
                      control_dir=paths.control,
                      supervisor=supervisor)
    # Flushed, like the error path in server.py. This daemon's stdout is a log
    # file or a journal, never a terminal, and print block-buffers when it is
    # not a tty -- so without this the address line stays in the buffer for the
    # whole fifteen days. That line is the only record of which interface it
    # bound, and it is the first thing to read when a remote curl times out.
    print(f"[encode-dash] {paths.run_dir}", flush=True)
    print(f"[encode-dash] http://{host}:{args.port}  "
          f"(hostname {socket.gethostname()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[encode-dash] stopping", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
