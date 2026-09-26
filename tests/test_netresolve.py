import subprocess

import pytest

from tools.archive_batch.netresolve import NoTrustedAddress, resolve_stream_ip
from tools.archive_batch.roster import Encoder


class _Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


GPU2_ADDRS = """lo               UNKNOWN        127.0.0.1/8
eth2             UP             198.51.100.62/32
eth3             UP             10.0.0.17/24
"""


def test_a_fixed_stream_ip_wins_and_runs_no_probe():
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return _Proc(GPU2_ADDRS)

    enc = Encoder(name="gpu4", host="gpu4", stream_ip="10.0.0.14")
    assert resolve_stream_ip(enc, run=run) == "10.0.0.14"
    assert calls == []


def test_stream_net_picks_the_address_inside_it():
    enc = Encoder(name="gpu2", host="gpu2", stream_net="10.0.0.0/24")
    got = resolve_stream_ip(enc, run=lambda cmd, **kw: _Proc(GPU2_ADDRS))
    assert got == "10.0.0.17"


def test_a_moved_lease_resolves_to_the_new_address():
    moved = GPU2_ADDRS.replace("10.0.0.17/24", "10.0.0.201/24")
    enc = Encoder(name="gpu2", host="gpu2", stream_net="10.0.0.0/24")
    got = resolve_stream_ip(enc, run=lambda cmd, **kw: _Proc(moved))
    assert got == "10.0.0.201"


def test_the_tailnet_address_is_never_chosen():
    enc = Encoder(name="gpu2", host="gpu2", stream_net="10.0.0.0/24")
    got = resolve_stream_ip(enc, run=lambda cmd, **kw: _Proc(GPU2_ADDRS))
    assert not got.startswith("100.")


def test_off_lan_refuses_rather_than_binding_something_else():
    cafe = """lo               UNKNOWN        127.0.0.1/8
eth3             UP             10.44.9.87/16
"""
    enc = Encoder(name="gpu2", host="gpu2", stream_net="10.0.0.0/24")
    with pytest.raises(NoTrustedAddress, match="10.0.0.0/24"):
        resolve_stream_ip(enc, run=lambda cmd, **kw: _Proc(cafe))


def test_a_down_interface_is_ignored():
    down = GPU2_ADDRS.replace("eth3             UP", "eth3             DOWN")
    enc = Encoder(name="gpu2", host="gpu2", stream_net="10.0.0.0/24")
    with pytest.raises(NoTrustedAddress):
        resolve_stream_ip(enc, run=lambda cmd, **kw: _Proc(down))


def test_an_unreachable_host_raises_rather_than_returning_empty():
    enc = Encoder(name="gpu2", host="gpu2", stream_net="10.0.0.0/24")
    with pytest.raises(NoTrustedAddress, match="cannot list addresses"):
        resolve_stream_ip(enc,
                          run=lambda cmd, **kw: _Proc("", 255, "no route to host"))


def test_a_remote_host_is_probed_over_ssh():
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return _Proc(GPU2_ADDRS)

    enc = Encoder(name="gpu2", host="gpu2", stream_net="10.0.0.0/24")
    resolve_stream_ip(enc, run=run)
    assert seen[0][:2] == ["ssh", "gpu2"]


def test_a_local_host_is_probed_without_ssh():
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return _Proc(GPU2_ADDRS)

    enc = Encoder(name="encoder-host", host="local", stream_net="10.0.0.0/24")
    resolve_stream_ip(enc, run=run)
    assert seen[0][0] == "ip"


def test_a_hung_probe_refuses_rather_than_binding_something_else():
    """The runner turns this into a clip failure and picks another encoder, so
    a wedged ssh must raise like an unreachable host, not return empty."""
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    enc = Encoder(name="gpu2", host="gpu2", stream_net="10.0.0.0/24")
    with pytest.raises(NoTrustedAddress, match="the probe timed out"):
        resolve_stream_ip(enc, run=run)
