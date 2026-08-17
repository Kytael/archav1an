import json
import socket
import threading
import urllib.error
import urllib.request

from tools.encode_dash.server import make_server

SNAP = {"batch": {"running": False, "pid": None}, "roster_error": None,
        "manifest_error": None, "encode": {"slots": 6, "lp_level": 4},
        "totals": {"clips": 1, "done": 0, "failed": 0, "queued": 1,
                   "frames": 10, "frames_done": 0, "fps_live": None,
                   "eta_finish": None},
        "lanes": [], "queue": [], "failures": []}


def _serve(snapshot_fn=lambda: SNAP):
    srv = make_server("127.0.0.1", 0, snapshot_fn)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode()


def _raw_get(port, target):
    """Everything the server put on the socket, unparsed. urllib normalises the
    request target and hides a malformed reply; both are what these tests are
    about."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        sock.sendall(f"GET {target} HTTP/1.0\r\nHost: x\r\n\r\n".encode())
        out = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return out
            out += chunk
    finally:
        sock.close()


def test_status_returns_the_snapshot_as_json():
    srv, base = _serve()
    try:
        status, ctype, body = _get(base + "/api/status")
        assert status == 200
        assert "application/json" in ctype
        assert json.loads(body)["totals"]["clips"] == 1
    finally:
        srv.shutdown()


def test_metrics_returns_prometheus_text():
    srv, base = _serve()
    try:
        status, ctype, body = _get(base + "/metrics")
        assert status == 200
        assert ctype.startswith("text/plain")
        assert "encode_batch_up 0" in body
    finally:
        srv.shutdown()


def test_root_serves_the_page():
    srv, base = _serve()
    try:
        status, ctype, body = _get(base + "/")
        assert status == 200
        assert "text/html" in ctype
        assert "<title>" in body
    finally:
        srv.shutdown()


def test_an_unknown_path_is_404_not_a_traceback():
    srv, base = _serve()
    try:
        _get(base + "/nope")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    else:
        raise AssertionError("expected 404")
    finally:
        srv.shutdown()


def test_a_path_cannot_climb_out_of_the_static_directory():
    """The static handler takes a name from the URL, and this listens on the
    tailnet with no authentication."""
    srv, base = _serve()
    try:
        _get(base + "/static/../../../../etc/passwd")
    except urllib.error.HTTPError as e:
        assert e.code in (400, 403, 404)
    else:
        raise AssertionError("expected a refusal")
    finally:
        srv.shutdown()


def test_no_encoding_of_a_climb_gets_out_of_the_static_directory():
    """urllib normalises `..` in the client, so the plain test above never puts
    one on the wire. These forms do: the handler reads a raw, still-encoded
    request target, so anything that decodes to a climb has to be refused after
    decoding, not before it.
    """
    srv, _ = _serve()
    port = srv.server_address[1]
    climbs = ["/static/../../../../etc/passwd",
              "/static/..%2f..%2f..%2f..%2fetc/passwd",
              "/static/%2e%2e%2f%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd",
              "/static/..%252f..%252fetc/passwd",
              "/static/..\\..\\..\\..\\etc\\passwd",
              "/static//etc/passwd",
              "/static/%2fetc%2fpasswd",
              "/static/%00../../../../etc/passwd",
              # A NUL is not a climb, but it must not be a 500 either: this
              # endpoint answers anything on the tailnet.
              "/static/app%00.html",
              "/static/%00",
              "/static/",
              "/static/.."]
    try:
        for path in climbs:
            reply = _raw_get(port, path)
            head = reply.split(b"\r\n", 1)[0]
            assert head.split()[1] in (b"400", b"403", b"404"), f"{path}: {head!r}"
            assert b"root:" not in reply, f"{path} served the file"
    finally:
        srv.shutdown()


def test_a_symlink_out_of_the_static_directory_is_refused(tmp_path, monkeypatch):
    """A guard on the text of the path alone does not see a symlink, and the
    static directory is on disk beside a daemon that anything on the tailnet can
    reach. The check has to be on the resolved file."""
    from tools.encode_dash import server as server_mod

    secret = tmp_path / "secret"
    secret.write_text("root:x:0:0:", encoding="utf-8")
    static = tmp_path / "static"
    static.mkdir()
    (static / "app.html").write_text("<title>t</title>", encoding="utf-8")
    (static / "escape.html").symlink_to(secret)
    monkeypatch.setattr(server_mod, "STATIC", str(static))

    srv, base = _serve()
    try:
        _get(base + "/static/escape.html")
    except urllib.error.HTTPError as e:
        assert e.code in (400, 403, 404)
    else:
        raise AssertionError("expected a refusal")
    finally:
        srv.shutdown()


def test_a_snapshot_that_raises_becomes_a_500_and_the_server_lives():
    """A daemon that dies on one bad read stops being a monitor, and /metrics
    is what the Prometheus alerts are built on."""
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("no run directory")
        return SNAP

    srv, base = _serve(flaky)
    try:
        try:
            _get(base + "/api/status")
        except urllib.error.HTTPError as e:
            assert e.code == 500
        else:
            raise AssertionError("expected 500")
        # Still serving, and the next request succeeds.
        assert _get(base + "/api/status")[0] == 200
    finally:
        srv.shutdown()


def test_a_failure_after_the_headers_does_not_splice_a_second_response():
    """Fault injection for the one case the ordering above cannot rule out: the
    body write itself failing. send_error at that point appends a whole second
    response to the first one's body, and the client reads that as a 200 whose
    body happens to contain "500 Internal Server Error" -- a failure that looks
    like success. Nothing may follow the first status line."""
    from tools.encode_dash import server as server_mod

    def half_sent(self, body, ctype):
        self.send_response(200)
        self._begun = True
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(b"part")
        raise ValueError("the disk went away")

    srv, _ = _serve()
    port = srv.server_address[1]
    original = server_mod._Handler._raw
    server_mod._Handler._raw = half_sent
    try:
        reply = _raw_get(port, "/api/status")
        assert reply.count(b"HTTP/1.0 ") == 1, reply[:400]
        assert b"500" not in reply, reply[:400]
        # And the daemon is still there.
        server_mod._Handler._raw = original
        assert _get(f"http://127.0.0.1:{port}/metrics")[0] == 200
    finally:
        server_mod._Handler._raw = original
        srv.shutdown()


def test_a_snapshot_that_will_not_serialise_is_a_500_not_a_torn_response():
    """json.dumps runs over a live snapshot, and one unexpected value in it
    raises. If that happened after the 200 had gone out the client would read a
    truncated body or a second response spliced onto the first, so the body has
    to be built before anything is sent."""
    srv, base = _serve(lambda: {"totals": object()})
    try:
        _get(base + "/api/status")
    except urllib.error.HTTPError as e:
        # A 200 whose body had a 500 spliced onto it would arrive here as a
        # plain 200, so the code alone tells the two apart.
        assert e.code == 500
    else:
        raise AssertionError("expected 500")
    finally:
        srv.shutdown()
