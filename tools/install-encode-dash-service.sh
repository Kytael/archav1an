#!/bin/bash
# Install the encode dashboard as a systemd --user service on this host.
#
# The daemon is a sidecar, so nothing in a run depends on it. Without a unit it
# still dies at the next reboot, and a fifteen-day run outlives more than one
# reboot -- gpu1 reboots itself for Windows Update. A page that is missing
# exactly when a run goes wrong is worse than no page.
#
# This is not part of setup.sh on purpose. setup.sh builds encode dependencies
# into a prefix and runs on every host that denoises or encodes. The dashboard
# runs on one host and serves the whole fleet, so installing a service from
# setup.sh would leave five idle daemons fighting for one port.
#
# The unit is generated rather than checked in as a file, because ExecStart
# needs this checkout's absolute path and the two hosts do not agree on it
# (/home/user/reposetc/archav1an here, /home/user/archav1an on gpu1).
# Re-run after moving the checkout or the venv.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-/opt/archav1an/venv/bin/python}"
# Where this deployment keeps its history, for the page's "history" link. It
# arrives as an environment variable rather than a default in this file,
# because a default here would be one site's hostname committed to the repo,
# and from there into the published tree.
GRAFANA_URL="${GRAFANA_URL:-}"
ARGS=""
[ -n "$GRAFANA_URL" ] && ARGS=" --grafana-url $GRAFANA_URL"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/encode-dash.service"

[ -x "$VENV_PY" ] || {
    echo "no interpreter at $VENV_PY -- set VENV_PY to override" >&2
    exit 1
}
[ -f "$REPO/tools/encode-dash.py" ] || {
    echo "no tools/encode-dash.py under $REPO" >&2
    exit 1
}

# A daemon started by hand holds port 9328, so the unit would start, fail to
# bind and retry forever. Say so here rather than leaving that in the journal.
if pgrep -f "[e]ncode-dash\.py" > /dev/null; then
    if ! systemctl --user is-active --quiet encode-dash.service; then
        echo "encode-dash.py is already running outside systemd. Stop it first:" >&2
        echo "  pkill -f encode-dash.py" >&2
        exit 1
    fi
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT" <<EOF
[Unit]
Description=Encode dashboard for the archive batch (port 9328)
After=network-online.target

[Service]
Type=simple
# The daemon binds this host's Tailscale address, and at boot tailscaled does
# not have one yet. A user unit cannot order itself after a system unit, so the
# wait belongs here. It matters more than it looks: without it the daemon does
# not fail, it falls back to 127.0.0.1 and answers nobody but this host, which
# no restart policy can detect. TimeoutStartSec bounds the wait; a host with no
# tailscale at all needs --host on ExecStart instead.
TimeoutStartSec=120
ExecStartPre=/bin/sh -c 'until tailscale ip -4 > /dev/null 2>&1; do sleep 2; done'
ExecStart=$VENV_PY $REPO/tools/encode-dash.py$ARGS
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

echo "wrote $UNIT"
systemctl --user daemon-reload
systemctl --user enable encode-dash.service
# restart, not "enable --now": on a re-run after the checkout or the venv moved,
# --now sees an active service and does nothing, so the old ExecStart keeps
# running and the freshly written unit is a lie until the next reboot. The
# daemon holds no run state, so restarting one that was already healthy costs a
# second of page downtime and nothing else.
systemctl --user restart encode-dash.service
systemctl --user --no-pager --lines=0 status encode-dash.service || true

# Without linger the user manager stops at logout and takes the daemon with it.
if ! loginctl show-user "$(id -un)" 2>/dev/null | grep -q "^Linger=yes"; then
    echo
    echo "Linger is off, so this service stops when you log out. Enable it:"
    echo "  loginctl enable-linger $(id -un)"
fi

echo
echo "  journalctl --user -u encode-dash -f      # follow"
echo "  systemctl --user restart encode-dash     # after a code change"
echo "  systemctl --user disable --now encode-dash   # remove"
