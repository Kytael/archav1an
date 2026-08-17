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


class _FrameCounter:
    """Frames seen so far, from the y4m header and arithmetic -- not from
    scanning for the FRAME marker.

    Scanning would give false positives on payload bytes that happen to spell
    FRAME, and would miss a marker straddling two reads. The header carries the
    geometry, so the frame stride is exact and a division answers it.

    `frames` is -1 until the header parses, which is how "cannot count" is told
    apart from "counted zero".
    """

    _HEADER_LIMIT = 512     # a y4m header is well under this; do not buffer a
                            # whole stream waiting for a newline that never comes

    def __init__(self):
        self.frames = -1
        self._stride = 0
        self._body = 0
        self._head = bytearray()

    def feed(self, chunk):
        if self._stride:
            self._body += len(chunk)
            self.frames = self._body // self._stride
            return
        self._head += bytes(chunk)
        end = self._head.find(b"\n")
        if end < 0:
            if len(self._head) > self._HEADER_LIMIT:
                self._head = bytearray()    # not y4m; give up quietly
                self.feed = lambda _chunk: None
            return
        stride = self._stride_from(bytes(self._head[:end]))
        if stride is None:
            # The stream still has to reach the encoder. A counter that cannot
            # work must go quiet, never take the encode down with it.
            self._head = bytearray()
            self.feed = lambda _chunk: None
            return
        self._stride = stride
        self._body = len(self._head) - (end + 1)
        self._head = bytearray()
        self.frames = self._body // self._stride

    @staticmethod
    def _stride_from(header):
        """Bytes per frame including the "FRAME\\n" marker, or None."""
        if not header.startswith(b"YUV4MPEG2"):
            return None
        width = height = 0
        depth = 1
        try:
            for tag in header.split():
                if tag[:1] == b"W":
                    width = int(tag[1:] or 0)
                elif tag[:1] == b"H":
                    height = int(tag[1:] or 0)
                elif tag[:1] == b"C":
                    # C420p10 and C420p16 are two bytes a sample; C420mpeg2 and
                    # friends are one. Getting this wrong halves or doubles
                    # every remote lane's rate.
                    depth = 2 if (b"p10" in tag or b"p12" in tag
                                  or b"p16" in tag) else 1
        except ValueError:
            # A truncated or corrupt W/H tag. Go quiet like any other header
            # this cannot read; never let the counter kill the encode.
            return None
        if width <= 0 or height <= 0:
            return None
        # 4:2:0 only, which is all this pipeline carries: svtav1-dispatch's
        # generated VPY converts to YUV420P10 precisely so the y4m stays 4:2:0.
        return len(b"FRAME\n") + width * height * 3 // 2 * depth


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
    # A remote denoise lane has no local frame counter otherwise: the remote
    # host's vspipe writes its log on the remote disk, and the ssh channel this
    # side captures carries only the remote dispatch's own output. Counting
    # arrivals here is also the truer figure -- it is frames that landed.
    #
    # The geometry is not passed in. dispatch does not know it: it has no width
    # or height anywhere and simply pipes y4m. The stream announces itself in
    # its first line, so this reads it there and needs no plumbing at all.
    counter = _FrameCounter() if args.progress else None
    next_report = t0 + args.progress_interval
    reported = -1
    while True:
        n = conn.recv_into(view)
        if not n:
            break
        out.write(view[:n])
        total += n
        if counter is not None:
            counter.feed(view[:n])
            now = time.monotonic()
            if now >= next_report and counter.frames != reported:
                next_report = now + args.progress_interval
                reported = counter.frames
                # Same shape vspipe -p emits, so the reader has one parser.
                print(f"Frame: {counter.frames}", file=sys.stderr, flush=True)
    out.flush()
    if counter is not None and counter.frames >= 0:
        print(f"Frame: {counter.frames}", file=sys.stderr, flush=True)
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
    r.add_argument("--progress", action="store_true",
                   help="print 'Frame: N' to stderr as frames arrive, for "
                        "the archive UI's live rate. Off by default so "
                        "nothing else changes.")
    r.add_argument("--progress-interval", type=float, default=1.0,
                   help="seconds between progress lines")

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
