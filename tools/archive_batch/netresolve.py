"""Which address an encode host should bind, resolved fresh per clip.

gpu1, encoder-host, gpu4 and gpu5 hold fixed addresses and set stream_ip.
gpu2 and gpu3 get theirs from DHCP: a literal address in the roster goes
stale and the lane then dies until somebody edits the file. Those two set
stream_net instead and the address is looked up when it is needed.

This STRENGTHENS the bind guard in spec 5.5 rather than weakening it. The rule
stops being "bind this literal address" and becomes "bind an address inside
the trusted network, or do not run at all". gpu2 is a laptop: on a cafe
network it holds nothing inside 10.0.0.0/24, so there is nothing to bind
and the lane is skipped before a socket is opened. An address picked up on an
untrusted network can never be chosen, because it is never inside the CIDR.

Resolved per clip and never cached. A cache is a second thing that can be
stale, and the probe costs about 0.3 s against a clip that runs for minutes.
"""
import ipaddress
import subprocess


class NoTrustedAddress(Exception):
    """The host holds no address inside its stream_net, so the lane must not run."""


def resolve_stream_ip(encoder, run=subprocess.run):
    """The address `encoder` should bind. Raises NoTrustedAddress if there is none."""
    if encoder.stream_ip:
        return encoder.stream_ip
    if not encoder.stream_net:
        # A local encoder may set neither: dispatch resolves the callback
        # itself, which is what a pre-pool roster relies on.
        return ""
    net = ipaddress.ip_network(encoder.stream_net, strict=False)
    cmd = ["ip", "-4", "-br", "addr"]
    if encoder.is_remote:
        cmd = ["ssh", encoder.host, *cmd]
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise NoTrustedAddress(
            f"encoder '{encoder.name}': cannot list addresses on "
            f"{encoder.host}: the probe timed out")
    if proc.returncode != 0:
        raise NoTrustedAddress(
            f"encoder '{encoder.name}': cannot list addresses on "
            f"{encoder.host}: {proc.stderr.strip()}")
    for addr in _addresses(proc.stdout):
        if addr in net:
            return str(addr)
    raise NoTrustedAddress(
        f"encoder '{encoder.name}' holds no address in {encoder.stream_net} on "
        f"{encoder.host}. It is off the trusted network, so the lane is skipped "
        f"rather than bound to whatever else it holds.")


def _addresses(text):
    """Every IPv4 address on an interface that is UP, from `ip -4 -br addr`.

    Lines look like `eth3   UP   10.0.0.17/24`. A DOWN interface keeps its
    address listed, and binding one would fail at the socket instead of here,
    where the message can say why.
    """
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[1].upper() != "UP":
            continue
        for token in parts[2:]:
            try:
                out.append(ipaddress.ip_interface(token).ip)
            except ValueError:
                continue
    return out
