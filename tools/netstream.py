"""Raw-TCP transport for the split-host denoise pipeline.

    gpu1:   vspipe -c y4m denoise.vpy - | netstream.py send --host <encoder>
    encoder-host: netstream.py recv --port N | SvtAv1EncApp -i stdin ...

A single ssh stream caps at ~1.3 Gbps between the two boxes even over the 10G
LAN (the ceiling is ssh's own, not the wire or the cipher); a plain TCP socket
sustains 5+ Gbps. So the y4m payload gets its own connection and ssh is left
carrying only the control channel — remote launch, stderr, exit status.

The encoder side listens: it can bind before the ssh that starts vspipe, the
listener's lifetime is owned by the process that also owns the encoder, and a
failed run leaves nothing behind on the remote host.

`send` retries the *connect* so the two ends may start in either order. It
never retries mid-stream: BSVD carries state across frames and SvtAv1EncApp
sees one continuous y4m, so a broken connection means the encode is dead and
must fail loudly rather than resume with a hole in it.

Access control is the firewall's job: the listener accepts one connection from
anyone the host lets through, so scope the inbound rule to the denoise host.

Stdlib only — gpu1 has no socat, nc or pv.
"""
import argparse
import socket
import sys
import time

BUF = 4 << 20


def _report(label, nbytes, elapsed):
    rate = nbytes / elapsed / 1e6 if elapsed > 0 else 0.0
    print(f"[netstream] {label} {nbytes / 1e6:.0f} MB in {elapsed:.1f}s "
          f"({rate:.0f} MB/s)", file=sys.stderr)


def recv(args):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", args.port))
    except OSError as e:
        print(f"[netstream] Error: cannot listen on port {args.port} ({e})",
              file=sys.stderr)
        return 1
    srv.listen(1)
    srv.settimeout(args.accept_timeout)
    print(f"[netstream] listening on port {args.port}", file=sys.stderr)
    try:
        conn, peer = srv.accept()
    except socket.timeout:
        print(f"[netstream] Error: no connection within {args.accept_timeout}s",
              file=sys.stderr)
        return 1
    print(f"[netstream] accepted {peer[0]}:{peer[1]}", file=sys.stderr)

    out = sys.stdout.buffer
    view = memoryview(bytearray(BUF))
    total = 0
    t0 = time.monotonic()
    conn.settimeout(None)
    while True:
        n = conn.recv_into(view)
        if not n:
            break
        out.write(view[:n])
        total += n
    out.flush()
    _report("received", total, time.monotonic() - t0)
    return 0


def send(args):
    deadline = time.monotonic() + args.connect_timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            conn = socket.create_connection((args.host, args.port), timeout=10)
            break
        except OSError as e:
            if time.monotonic() >= deadline:
                print(f"[netstream] Error: could not connect to {args.host}:"
                      f"{args.port} after {attempt} attempts ({e})",
                      file=sys.stderr)
                return 1
            if attempt == 1:
                print(f"[netstream] {args.host}:{args.port} not ready ({e}); "
                      f"retrying for up to {args.connect_timeout}s",
                      file=sys.stderr)
            time.sleep(args.retry_interval)
    print(f"[netstream] connected to {args.host}:{args.port} "
          f"(attempt {attempt})", file=sys.stderr)

    conn.settimeout(None)
    src = sys.stdin.buffer
    view = memoryview(bytearray(BUF))
    total = 0
    t0 = time.monotonic()
    while True:
        n = src.readinto(view)
        if not n:
            break
        conn.sendall(view[:n])
        total += n
    conn.shutdown(socket.SHUT_WR)
    conn.close()
    _report("sent", total, time.monotonic() - t0)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("recv", help="listen, then copy the socket to stdout")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--accept-timeout", type=float, default=300.0)

    s = sub.add_parser("send", help="copy stdin to the socket, retrying connect")
    s.add_argument("--host", required=True)
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--connect-timeout", type=float, default=120.0,
                   help="keep retrying the connect for this long")
    s.add_argument("--retry-interval", type=float, default=1.0)

    args = ap.parse_args()
    return recv(args) if args.mode == "recv" else send(args)


if __name__ == "__main__":
    sys.exit(main())
