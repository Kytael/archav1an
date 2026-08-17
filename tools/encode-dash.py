#!/usr/bin/env python3
# tools/encode-dash.py
"""The archive run's dashboard. Runs beside archive-batch, never inside it.

    /opt/archav1an/venv/bin/python tools/encode-dash.py

Reads .archive-run/ and serves a page, a JSON snapshot and a Prometheus
endpoint. It holds no state of its own except the rate history, so it can be
restarted at any point in a fifteen-day run without touching the run.

Part 1 is read-only: it cannot change the roster and cannot signal the batch.

Binding: the default is the Tailscale address, because that is how the fleet
reaches this host and because encoder-host's firewall already refuses high ports on
the LAN. There is no authentication, which matches every other service here --
llama-swap, ComfyUI, Datasette and opencode are all open on the tailnet. Unlike
those, from part 2 this one can stop a run; the tailnet is the trust boundary.
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


def _tailscale_address():
    """This host's the tailnet CGNAT range address, or None.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=None,
                    help="bind address; default is this host's Tailscale IP, "
                         "falling back to 127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--smooth", type=float, default=30.0,
                    help="seconds to average the live rate over. A windowed "
                         "lane bursts a whole window at once, so this must "
                         "cover at least one sweep or the figure alternates "
                         "between zero and a spike.")
    args = ap.parse_args()

    host = args.host or _tailscale_address() or "127.0.0.1"
    paths = Paths.from_env()
    tracker = RateTracker(smooth_s=args.smooth)

    srv = make_server(host, args.port,
                      # time.time(), not monotonic: the heartbeat records wall
                      # clock because a person reads it, and model._lane
                      # subtracts the two. Mixing the clocks would give every
                      # lane a nonsense elapsed time. A run measured in days
                      # does not care about a one-second NTP correction.
                      lambda: snapshot(paths, tracker, time.time()))
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
