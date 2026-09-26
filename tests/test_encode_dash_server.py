import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from tools.archive_batch.roster import load_roster
from tools.encode_dash import rosterio
from tools.encode_dash.server import make_server
from tools.encode_dash.supervisor import AlreadyRunning, SupervisorError

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


def _raw_post(port, target, body, headers):
    """A POST with exactly the headers given and no others.

    urllib cannot express this: do_request_ inserts a Content-Type whenever a
    body is present, so 'no Content-Type at all' is unreachable through it and
    the test that claimed to cover that case was really testing the
    form-encoded one twice.
    """
    lines = [f"POST {target} HTTP/1.0", "Host: 127.0.0.1",
             f"Content-Length: {len(body)}"]
    lines += [f"{k}: {v}" for k, v in headers.items()]
    request = ("\r\n".join(lines) + "\r\n\r\n").encode() + body
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        sock.sendall(request)
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


def test_the_lane_catalogue_is_served_as_json():
    """The add-lane picker fetches this file and calls .json() on the reply.

    Before .json was in the content-type table it went out as
    application/octet-stream, which some browsers refuse to parse -- and the
    picker's failure path is to stay hidden, so the page would have looked
    exactly as it did before the feature, with no error anywhere.
    """
    srv, base = _serve()
    try:
        status, ctype, body = _get(base + "/static/lane-presets.json")
        assert status == 200
        assert "application/json" in ctype
        assert json.loads(body)["presets"]
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


ROSTER = '''\
[[denoiser]]
name    = "gpu1_4090"
host    = "gpu1"
backend = "trt"
device  = 0
tiling  = "none"
enabled = true

[[denoiser]]
name    = "igpu"
host    = "local"
backend = "migraphx"
device  = 0
tiling  = "none"
enabled = false

[encode]
host     = "local"
slots    = 6
lp_level = 4
'''


@pytest.fixture
def editable(tmp_path):
    path = tmp_path / "denoisers.toml"
    path.write_text(ROSTER)
    srv = make_server("127.0.0.1", 0, lambda: SNAP, roster_path=str(path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", str(path)
    srv.shutdown()


def _send(url, method, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_post_enabled_flips_the_lane(editable):
    base, path = editable
    status, out = _send(f"{base}/api/lane/igpu/enabled", "POST",
                        {"enabled": True, "rev": rosterio.rev(path)})
    assert status == 200
    assert out["rev"] == rosterio.rev(path)
    assert "enabled = true" in open(path).read()


def test_post_enabled_with_a_stale_rev_is_409(editable):
    base, path = editable
    stale = rosterio.rev(path)
    rosterio.set_enabled(path, "igpu", True)
    status, _ = _send(f"{base}/api/lane/igpu/enabled", "POST",
                      {"enabled": False, "rev": stale})
    assert status == 409


def test_post_enabled_with_a_string_is_400_not_coerced(editable):
    # bool("false") is True in Python. A client that sends the string
    # "false" -- a realistic curl/shell-script mistake on an unauthenticated
    # daemon -- must be rejected, not silently enable the lane it meant to
    # disable.
    base, path = editable
    before = open(path).read()
    status, body = _send(f"{base}/api/lane/igpu/enabled", "POST",
                        {"enabled": "false", "rev": rosterio.rev(path)})
    assert status == 400
    assert "must be true or false" in body
    assert open(path).read() == before


def test_post_enabled_on_an_unknown_lane_is_404(editable):
    base, path = editable
    status, _ = _send(f"{base}/api/lane/nosuch/enabled", "POST",
                      {"enabled": True, "rev": rosterio.rev(path)})
    assert status == 404


def test_disabling_the_last_lane_is_allowed(editable):
    """The page offers this switch on every lane, so the last one must work.
    The run parks until a lane comes back rather than refusing the edit."""
    base, path = editable
    status, _ = _send(f"{base}/api/lane/gpu1_4090/enabled", "POST",
                      {"enabled": False, "rev": rosterio.rev(path)})
    assert status == 200
    assert list(rosterio.load_roster(path).enabled()) == []


def test_post_lane_adds_a_disabled_lane(editable):
    base, path = editable
    status, _ = _send(f"{base}/api/lane", "POST",
                      {"name": "gpu3", "host": "gpu3", "backend": "trt",
                       "device": 0, "tiling": "auto", "window": 750,
                       "margin": 32, "stage_source": True,
                       "root": "reposetc/archav1an",
                       "rev": rosterio.rev(path)})
    assert status == 200
    text = open(path).read()
    assert 'name    = "gpu3"' in text
    # NOT "the file ends with enabled = false": add_lane inserts before the
    # [encode] table, so [encode] is still last. Assert the lane's state, not
    # the file's final line.
    added = [d for d in load_roster(path).denoisers if d.name == "gpu3"][0]
    assert added.enabled is False


def test_post_lane_with_a_bad_field_is_400(editable):
    base, path = editable
    status, body = _send(f"{base}/api/lane", "POST",
                         {"name": "gpu3", "host": "gpu3", "backend": "trt",
                          "device": 0, "tiling": "auto",
                          "rev": rosterio.rev(path)})
    assert status == 400
    assert "needs a window" in body


def test_post_lane_name_edits_the_lane(editable):
    base, path = editable
    status, _ = _send(f"{base}/api/lane/gpu1_4090", "POST",
                      {"name": "gpu1_4090", "device": 1,
                       "rev": rosterio.rev(path)})
    assert status == 200
    edited = [d for d in load_roster(path).denoisers
              if d.name == "gpu1_4090"][0]
    assert edited.device == 1


def test_editing_a_lane_does_not_disturb_the_others(editable):
    base, path = editable
    before = open(path).read()
    _send(f"{base}/api/lane/gpu1_4090", "POST",
          {"device": 1, "rev": rosterio.rev(path)})
    after = open(path).read()
    assert before.count("[[denoiser]]") == after.count("[[denoiser]]")
    assert 'name    = "igpu"' in after


def test_editing_an_unknown_lane_is_404(editable):
    base, path = editable
    status, body = _send(f"{base}/api/lane/nosuch", "POST",
                         {"device": 1, "rev": rosterio.rev(path)})
    assert status == 404
    assert "nosuch" in body


def test_an_edit_that_would_not_load_is_400_and_writes_nothing(editable):
    base, path = editable
    before = open(path).read()
    status, body = _send(f"{base}/api/lane/gpu1_4090", "POST",
                         {"tiling": "auto", "rev": rosterio.rev(path)})
    assert status == 400
    assert "needs a window" in body
    assert open(path).read() == before


def test_delete_lane_removes_it(editable):
    base, path = editable
    status, _ = _send(f"{base}/api/lane/igpu", "DELETE",
                      {"rev": rosterio.rev(path)})
    assert status == 200
    assert "igpu" not in open(path).read()


def test_post_lp_level(editable):
    base, path = editable
    status, _ = _send(f"{base}/api/encode/lp_level", "POST",
                      {"lp_level": 6, "rev": rosterio.rev(path)})
    assert status == 200
    assert "lp_level = 6" in open(path).read()


def test_post_lp_level_out_of_range_is_400(editable):
    base, path = editable
    status, body = _send(f"{base}/api/encode/lp_level", "POST",
                         {"lp_level": 9, "rev": rosterio.rev(path)})
    assert status == 400
    assert "parallelism" in body


def test_malformed_json_is_400(editable):
    base, _ = editable
    # The header is required now, and this test is about the BODY: without it
    # the request would be refused at the Content-Type check and never reach
    # the parser this test is named for.
    req = urllib.request.Request(f"{base}/api/encode/lp_level", data=b"{nope",
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400


def _send_raw(url, method, body, ctype):
    """A write with a chosen Content-Type, or none at all."""
    headers = {"Content-Type": ctype} if ctype is not None else {}
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_a_write_with_no_rev_is_400_and_changes_nothing(editable):
    """The guard is the whole point of the revision, so it cannot be optional.
    A write that quotes nothing does not know what it is overwriting."""
    base, path = editable
    before = open(path).read()
    for url, method, payload in (
            (f"{base}/api/encode/lp_level", "POST", {"lp_level": 6}),
            (f"{base}/api/lane/igpu/enabled", "POST", {"enabled": True}),
            (f"{base}/api/lane", "POST", {"name": "gpu3", "host": "local",
                                          "backend": "trt", "device": 0,
                                          "tiling": "none"}),
            (f"{base}/api/lane/igpu", "DELETE", {})):
        status, body = _send(url, method, payload)
        assert status == 400, f"{method} {url}"
        assert "rev" in body
    assert open(path).read() == before


def test_a_write_whose_rev_is_not_a_string_is_400(editable):
    """The old shape was [mtime_ns, size]. A client still sending that, or a
    number, or null, has not quoted a revision this daemon ever issued."""
    base, path = editable
    before = open(path).read()
    for bad in ([1787015143623006887, 431], 1787015143623006887, None, True):
        status, body = _send(f"{base}/api/encode/lp_level", "POST",
                             {"lp_level": 6, "rev": bad})
        assert status == 400, f"rev={bad!r}"
        assert "rev" in body
    assert open(path).read() == before


def test_a_text_plain_write_is_415_and_changes_nothing(editable):
    """The cross-origin shape, exactly as an attacking page would send it.

    text/plain is CORS-simple, so an auto-submitting
    `<form enctype="text/plain">` on any page the operator visits posts it with
    no preflight and no permission from this daemon -- which has no
    authentication. Requiring application/json makes the request non-simple, so
    the browser must preflight it and the attacking page cannot proceed.
    """
    base, path = editable
    before = open(path).read()
    payload = json.dumps({"enabled": False, "pad": "="}).encode()
    status, _ = _send_raw(f"{base}/api/lane/gpu1_4090/enabled", "POST",
                          payload, "text/plain;charset=UTF-8")
    assert status == 415
    assert open(path).read() == before


def test_a_form_encoded_write_is_415(editable):
    base, path = editable
    before = open(path).read()
    status, _ = _send_raw(f"{base}/api/encode/lp_level", "POST",
                          b"lp_level=6", "application/x-www-form-urlencoded")
    assert status == 415
    assert open(path).read() == before


def test_a_content_type_with_a_charset_parameter_is_accepted(editable):
    """"application/json; charset=utf-8" is a JSON body. Some stacks send it,
    and refusing it would break a legitimate client for no security gain."""
    base, path = editable
    payload = json.dumps({"lp_level": 6, "rev": rosterio.rev(path)}).encode()
    status, _ = _send_raw(f"{base}/api/encode/lp_level", "POST", payload,
                          "application/json; charset=utf-8")
    assert status == 200
    assert "lp_level = 6" in open(path).read()


def test_a_write_route_is_503_without_a_roster_path():
    srv, base = _serve()          # the three-argument helper already in this file
    status, _ = _send(f"{base}/api/encode/lp_level", "POST", {"lp_level": 4})
    assert status == 503
    srv.shutdown()


def test_an_unknown_post_path_is_404(editable):
    # A valid rev is now a precondition for reaching the router at all, so this
    # test has to supply one to still be about the path.
    base, path = editable
    status, _ = _send(f"{base}/api/nope", "POST", {"rev": rosterio.rev(path)})
    assert status == 404


def test_get_routes_still_work_with_a_roster_path(editable):
    base, _ = editable
    status, ctype, body = _get(f"{base}/api/status")
    assert status == 200 and json.loads(body) == SNAP


@pytest.fixture
def controllable(tmp_path):
    """A server with both a roster and a control directory.

    ROSTER verbatim. This fixture used to turn igpu on so that yielding
    gpu1_4090 would not leave the roster with no enabled lane, back when that
    was refused. Every lane off is a legal roster now, so the workaround is
    gone and these tests run against the file the operator actually has.
    """
    path = tmp_path / "denoisers.toml"
    path.write_text(ROSTER)
    control_dir = tmp_path / "control"
    srv = make_server("127.0.0.1", 0, lambda: SNAP, roster_path=str(path),
                      control_dir=str(control_dir))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", str(path), str(control_dir)
    srv.shutdown()


def test_a_yield_disables_the_lane_before_it_asks_for_the_kill(controllable):
    """Spec 5.3, and the order is not cosmetic. Reversed, the worker handles
    YieldRequested, loops, still finds itself enabled and retakes the same
    clip -- so the operator's GPU never comes back."""
    from tools.archive_batch import control

    base, path, control_dir = controllable
    status, out = _send(f"{base}/api/lane/gpu1_4090/yield", "POST",
                        {"rev": rosterio.rev(path)})
    assert status == 200
    assert out["rev"] == rosterio.rev(path)
    # Not a substring check. ROSTER already carries "enabled = false" for
    # igpu, so that would be satisfied by the fixture file before the request
    # was ever sent. gpu1_4090 is its only enabled lane, so the disable having
    # landed is exactly the roster reading back with nothing enabled.
    assert list(load_roster(path).enabled()) == []
    requests = control.take(control_dir)
    assert len(requests) == 1
    assert requests[0][1]["action"] == "yield"
    assert requests[0][1]["lane"] == "gpu1_4090"


def test_a_yield_with_a_stale_rev_neither_disables_nor_asks(controllable):
    """The refusal must happen before anything at all is written, or the lane
    is disabled on the strength of an edit the daemon then rejects."""
    from tools.archive_batch import control

    base, path, control_dir = controllable
    before = open(path).read()
    status, _ = _send(f"{base}/api/lane/gpu1_4090/yield", "POST",
                      {"rev": "1-1"})
    assert status == 409
    assert open(path).read() == before
    assert control.take(control_dir) == []


def test_a_yield_for_a_lane_that_is_not_in_the_roster_is_404(controllable):
    from tools.archive_batch import control

    base, path, control_dir = controllable
    status, _ = _send(f"{base}/api/lane/ghost/yield", "POST",
                      {"rev": rosterio.rev(path)})
    assert status == 404
    assert control.take(control_dir) == []


def test_a_stop_writes_one_request_and_needs_no_roster_rev(controllable):
    """A stop is not a roster edit. Requiring a rev would break the button
    exactly when the roster is being edited, which is one of the times the
    operator most wants it."""
    from tools.archive_batch import control

    base, _path, control_dir = controllable
    status, out = _send(f"{base}/api/run/stop", "POST", {})
    assert status == 200
    assert isinstance(out["id"], str)
    requests = control.take(control_dir)
    assert [r["action"] for _, r in requests] == ["stop"]
    assert requests[0][1]["id"] == out["id"]


def test_a_retry_carries_the_clip_path_through_the_url(controllable):
    """The src holds slashes, so the page percent-encodes it and this rejoins
    it. Matching on an exact segment count the way the roster routes do would
    answer a silent 404 for every clip that is not at the top level."""
    from tools.archive_batch import control
    from urllib.parse import quote

    base, _path, control_dir = controllable
    src = "SetB/2002/rose city setb/MVI_6077.MOV"
    status, out = _send(
        f"{base}/api/clip/{quote(src, safe='')}/retry", "POST", {})
    assert status == 200
    requests = control.take(control_dir)
    assert requests[0][1]["action"] == "retry"
    assert requests[0][1]["src"] == src


def test_a_retry_path_left_unencoded_still_reaches_the_route(controllable):
    from tools.archive_batch import control

    base, _path, control_dir = controllable
    status, _ = _send(f"{base}/api/clip/SetA/2001/a/x.MOV/retry", "POST", {})
    assert status == 200
    assert control.take(control_dir)[0][1]["src"] == "SetA/2001/a/x.MOV"


def test_a_control_write_as_text_plain_is_415_and_writes_nothing(controllable):
    """The same CSRF defence as the roster routes, and the only one these have.
    A text/plain POST is CORS-simple: an auto-submitting form on any page the
    operator visits could otherwise stop a fifteen-day run."""
    from tools.archive_batch import control

    base, _path, control_dir = controllable
    status, _ = _send_raw(f"{base}/api/run/stop", "POST", b'{"pad":"="}',
                          "text/plain;charset=UTF-8")
    assert status == 415
    assert control.take(control_dir) == []


def test_a_control_write_as_a_form_post_is_415_and_writes_nothing(controllable):
    from tools.archive_batch import control

    base, _path, control_dir = controllable
    status, _ = _send_raw(f"{base}/api/run/stop", "POST", b"pad=1",
                          "application/x-www-form-urlencoded")
    assert status == 415
    assert control.take(control_dir) == []


def test_a_control_route_answers_503_with_no_control_directory(editable):
    """The roster routes stay usable in this configuration; only the control
    ones are unavailable, and they say so rather than failing obscurely."""
    base, _path = editable
    status, _ = _send(f"{base}/api/run/stop", "POST", {})
    assert status == 503


def test_a_stop_still_works_when_the_daemon_has_no_roster_path(tmp_path):
    control_dir = tmp_path / "control"
    srv = make_server("127.0.0.1", 0, lambda: SNAP,
                      control_dir=str(control_dir))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        status, _ = _send(f"{base}/api/run/stop", "POST", {})
        assert status == 200
    finally:
        srv.shutdown()


def test_a_control_write_ignores_a_rev_the_page_sends_anyway(controllable):
    """app.js folds rev into every write body. Control does not use it, and it
    must not be an error to receive one."""
    base, path, _control_dir = controllable
    status, _ = _send(f"{base}/api/run/stop", "POST",
                      {"rev": rosterio.rev(path)})
    assert status == 200


def test_a_write_with_genuinely_no_content_type_is_415(editable):
    """Hand-written, because urllib always supplies one. Without this the case
    the docstring on _content_type_is_json calls out -- a body posted with no
    header -- had no coverage at all."""
    base, path = editable
    port = int(base.rsplit(":", 1)[1])
    before = open(path).read()
    raw = _raw_post(port, "/api/encode/lp_level",
                    json.dumps({"lp_level": 6}).encode(), {})
    assert b"415" in raw.split(b"\r\n", 1)[0], raw[:120]
    assert open(path).read() == before


def test_a_yield_returns_the_clip_to_the_queue(tmp_path):
    """Spec 9's one end-to-end: fake runner, real scheduler, real daemon.

    The ordering rule of spec 5.3 is what this proves. The daemon writes
    enabled = false BEFORE it drops the yield request; reversed, the worker
    handles YieldRequested, loops, still finds itself enabled, and takes the
    same clip straight back -- so the GPU never comes back and the page would
    look like the button did nothing.
    """
    import threading as _threading

    from tools.archive_batch import control as batch_control
    from tools.archive_batch.control import YieldRequested
    from tools.archive_batch.manifest import Clip
    from tools.archive_batch.roster import load_roster
    from tools.archive_batch.scheduler import Scheduler

    roster_path = tmp_path / "denoisers.toml"
    # ROSTER verbatim, so gpu1_4090 is the only enabled lane and this drives
    # the case the operator actually hits: yielding the LAST one. It used to be
    # refused, which is why this test once turned igpu on to dodge it. The run
    # must park with every lane off rather than stop, so the assertions below
    # check the roster reads back with nothing enabled and _stop is still
    # clear.
    roster_path.write_text(ROSTER)
    control_dir = tmp_path / "control"

    holding = _threading.Event()
    release = _threading.Event()
    killed = set()

    def runner(clip, denoiser, encoder, slot):
        holding.set()
        release.wait(20)
        if denoiser.name in killed:
            raise YieldRequested(f"{denoiser.name} was yielded")
        return True, 1.0, 1.0, 1, ""

    clip = Clip("SetA/2001/a/x.MOV", "SetA/2001/a", "x", 1, 100)
    s = Scheduler([clip], lambda: load_roster(str(roster_path)), runner,
                  state_path=tmp_path / "state.jsonl")
    s.POLL_SECONDS = 0.05

    srv = make_server("127.0.0.1", 0, lambda: SNAP,
                      roster_path=str(roster_path),
                      control_dir=str(control_dir))
    _threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    worker = _threading.Thread(target=s._worker, args=("gpu1_4090",),
                               daemon=True)
    worker.start()
    # Every assertion that can fail lives inside this try, so a failure here
    # still stops the worker and the server rather than leaving a daemon
    # thread tight-polling for the rest of the session -- the same guarantee
    # every other server-thread test in this file gets from srv.shutdown()
    # in a fixture teardown or a finally.
    try:
        assert holding.wait(10), "the lane never took the clip"

        status, _ = _send(f"{base}/api/lane/gpu1_4090/yield", "POST",
                          {"rev": rosterio.rev(str(roster_path))})
        assert status == 200

        # The batch's side of the channel, driven by hand rather than by a
        # thread: this test is about the daemon's ordering, not about the
        # poll interval.
        requests = batch_control.take(str(control_dir))
        assert [r["action"] for _, r in requests] == ["yield"]
        assert requests[0][1]["lane"] == "gpu1_4090"
        killed.add(requests[0][1]["lane"])
        release.set()

        # Assert the yield first, THEN release the worker. After the yield
        # the queue is non-empty and the lane is disabled, so _worker parks
        # by design and only stop() lets it return -- the trap documented at
        # tests/test_scheduler.py:394-401. Joining here instead would block
        # for the whole timeout and then fail, whatever the implementation
        # does.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and s.queue.qsize() != 1:
            time.sleep(0.01)
        assert s.queue.qsize() == 1, "the yielded clip must be back on the queue"
        assert s.queue.get_nowait() is clip
        assert s._attempts == {}, "a yield spends no attempt"
        assert s.failed == 0
        # Not a substring check: ROSTER already carries "enabled = false" for
        # igpu, so that would pass even if the yield had written nothing. This
        # is the whole point of the change -- gpu1_4090 was the last enabled
        # lane, and the run parks with none rather than refusing the edit.
        assert list(load_roster(str(roster_path)).enabled()) == [],             "the yield must disable the last enabled lane"
        assert not s._stop.is_set(), "a yield must not stop the run"
    finally:
        s.stop()
        worker.join(10)
        srv.shutdown()
    assert not worker.is_alive(), "the worker did not return after stop()"


class _FakeSupervisor:
    """A stand-in for tools.encode_dash.supervisor.Supervisor: answer with a
    pid, or refuse exactly the way the real one does when a run is on file."""

    def __init__(self, pid=None, error=None):
        self.pid = pid
        self.error = error

    def start(self):
        if self.error is not None:
            raise self.error
        return self.pid


def _serve_supervised(supervisor):
    srv = make_server("127.0.0.1", 0, lambda: SNAP, supervisor=supervisor)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def supervised():
    """A server whose supervisor answers with a known pid."""
    srv = _serve_supervised(_FakeSupervisor(pid=4242))
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_run_start_answers_200_with_the_pid(supervised):
    status, out = _send(f"{supervised}/api/run/start", "POST", {})
    assert status == 200
    assert out == {"ok": True, "pid": 4242}


def test_run_start_answers_409_when_a_batch_is_already_running():
    srv = _serve_supervised(_FakeSupervisor(
        error=AlreadyRunning("a batch is already running (pid 4242)")))
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        status, out = _send(f"{base}/api/run/start", "POST", {})
        assert status == 409
        assert "a batch is already running (pid 4242)" in out
    finally:
        srv.shutdown()


def test_run_start_answers_503_with_the_reason_it_cannot_manage_the_run_dir():
    """A run directory the daemon cannot write. adopt() logged this at boot
    and carried on read-only, so the click on Start is the first the operator
    sees of it: the sentence belongs in the reply, not a repr in a 500."""
    srv = _serve_supervised(_FakeSupervisor(error=SupervisorError(
        "cannot write /run/batch.json: Permission denied. The run directory "
        "is not writable by this daemon, so a batch cannot be claimed or "
        "recorded.")))
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        status, out = _send(f"{base}/api/run/start", "POST", {})
        assert status == 503
        assert "not writable by this daemon" in out
        assert "SupervisorError(" not in out, "a repr is not an explanation"
    finally:
        srv.shutdown()


def test_run_start_answers_503_with_no_supervisor():
    srv = make_server("127.0.0.1", 0, lambda: SNAP)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        status, _ = _send(f"{base}/api/run/start", "POST", {})
        assert status == 503
    finally:
        srv.shutdown()


def test_run_start_as_text_plain_is_415(supervised):
    status, _ = _send_raw(f"{supervised}/api/run/start", "POST", b"{}",
                          "text/plain")
    assert status == 415


# --- A pool roster has no [encode] table for /api/encode/lp_level to write ---

POOL_ROSTER = '''\
[[denoiser]]
name    = "igpu"
host    = "local"
backend = "migraphx"
device  = 0
tiling  = "none"
enabled = true

[[encoder]]
name      = "encoder-host"
host      = "local"
stream_ip = "10.0.0.10"
port_base = 5300
slots     = 6
lp_level  = 4
enabled   = true
'''


@pytest.fixture
def pool(tmp_path):
    path = tmp_path / "denoisers.toml"
    path.write_text(POOL_ROSTER)
    srv = make_server("127.0.0.1", 0, lambda: SNAP, roster_path=str(path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", str(path)
    srv.shutdown()


def test_post_lp_level_on_a_pool_roster_is_a_400_naming_the_gap(pool):
    """Not a 500 and not a silent success: the page shows this string, so it
    has to point at the row where each host's own level lives, rather than
    name a table the operator never wrote."""
    base, path = pool
    before = open(path).read()
    status, body = _send(f"{base}/api/encode/lp_level", "POST",
                         {"lp_level": 6, "rev": rosterio.rev(path)})
    assert status == 400
    assert "Encode hosts" in body
    assert open(path).read() == before


# --- The encode host switch -------------------------------------------------

def test_post_encoder_enabled_flips_the_host(pool):
    base, path = pool
    status, out = _send(f"{base}/api/encoder/encoder-host/enabled", "POST",
                        {"enabled": False, "rev": rosterio.rev(path)})
    assert status == 200
    assert out["rev"] == rosterio.rev(path)
    assert not load_roster(path).encoders[0].enabled


def test_post_encoder_enabled_does_not_reach_the_lane_table(pool):
    """The route picks the table, so a lane sharing the encoder's name is
    untouched. POOL_ROSTER's lane is "igpu", so this checks the weaker
    property that the lane table is not written at all."""
    base, path = pool
    before = {d.name: d.enabled for d in load_roster(path).denoisers}
    _send(f"{base}/api/encoder/encoder-host/enabled", "POST",
          {"enabled": False, "rev": rosterio.rev(path)})
    assert {d.name: d.enabled for d in load_roster(path).denoisers} == before


def test_post_encoder_enabled_on_a_missing_host_is_404(pool):
    # 404 and not 400: a name the roster does not carry is a missing thing,
    # which is what the lane route already answers for an unknown lane.
    base, path = pool
    status, body = _send(f"{base}/api/encoder/nope/enabled", "POST",
                         {"enabled": True, "rev": rosterio.rev(path)})
    assert status == 404
    assert "nope" in body


def test_post_encoder_enabled_with_a_string_is_400_not_coerced(pool):
    base, path = pool
    before = open(path).read()
    status, body = _send(f"{base}/api/encoder/encoder-host/enabled", "POST",
                         {"enabled": "false", "rev": rosterio.rev(path)})
    assert status == 400
    assert "must be true or false" in body
    assert open(path).read() == before


def test_post_encoder_enabled_with_a_stale_rev_is_409(pool):
    base, path = pool
    stale = rosterio.rev(path)
    rosterio.set_encoder_enabled(path, "encoder-host", False)
    status, _ = _send(f"{base}/api/encoder/encoder-host/enabled", "POST",
                      {"enabled": True, "rev": stale})
    assert status == 409


# --- Editing encode hosts, and routing lanes to them ------------------------

def test_post_encoder_adds_a_host_switched_off(pool):
    base, path = pool
    # stream_net, not omitted: a remote encoder with no address is refused by
    # the validator, because the listener has nothing to bind and the denoise
    # host has nothing to connect to.
    status, _ = _send(f"{base}/api/encoder", "POST",
                      {"name": "gpu1", "host": "gpu1", "port_base": 5350,
                       "slots": 2, "lp_level": 4,
                       "stream_net": "10.0.0.0/24",
                       "rev": rosterio.rev(path)})
    assert status == 200
    pool_now = {e.name: e for e in load_roster(path).encoders}
    assert "gpu1" in pool_now
    assert pool_now["gpu1"].enabled is False


def test_post_encoder_name_edits_that_host(pool):
    base, path = pool
    status, _ = _send(f"{base}/api/encoder/encoder-host", "POST",
                      {"slots": 3, "rev": rosterio.rev(path)})
    assert status == 200
    assert load_roster(path).encoders[0].slots == 3


def test_post_encoder_name_refuses_a_rename(pool):
    base, path = pool
    before = open(path).read()
    status, body = _send(f"{base}/api/encoder/encoder-host", "POST",
                         {"name": "other", "rev": rosterio.rev(path)})
    assert status == 400
    assert "allowlist" in body
    assert open(path).read() == before


def test_delete_encoder_removes_it(pool):
    base, path = pool
    # A second host first: removing the LAST encoder leaves a roster with no
    # pool, which the validator refuses on its own terms. This route is for
    # taking one box out, not for emptying the fleet.
    _send(f"{base}/api/encoder", "POST",
          {"name": "gpu1", "host": "gpu1", "port_base": 5350, "slots": 2,
           "lp_level": 4, "stream_net": "10.0.0.0/24",
           "rev": rosterio.rev(path)})
    status, _ = _send(f"{base}/api/encoder/encoder-host", "DELETE",
                      {"rev": rosterio.rev(path)})
    assert status == 200
    assert [e.name for e in load_roster(path).encoders] == ["gpu1"]


def test_delete_the_last_encoder_is_refused(pool):
    base, path = pool
    before = open(path).read()
    status, body = _send(f"{base}/api/encoder/encoder-host", "DELETE",
                         {"rev": rosterio.rev(path)})
    assert status == 400
    assert "no encoder" in body
    assert open(path).read() == before


def test_post_lane_can_set_the_allowlist(pool):
    """Routing, end to end through the route the checkboxes post to."""
    base, path = pool
    status, _ = _send(f"{base}/api/lane/igpu", "POST",
                      {"encoders": ["encoder-host"], "rev": rosterio.rev(path)})
    assert status == 200
    assert load_roster(path).denoisers[0].encoders == ("encoder-host",)


def test_post_lane_with_an_empty_allowlist_restores_any_encoder(pool):
    base, path = pool
    _send(f"{base}/api/lane/igpu", "POST",
          {"encoders": ["encoder-host"], "rev": rosterio.rev(path)})
    status, _ = _send(f"{base}/api/lane/igpu", "POST",
                      {"encoders": [], "rev": rosterio.rev(path)})
    assert status == 200
    lane = load_roster(path).denoisers[0]
    assert lane.encoders == () and lane.allows("anything")


def test_post_lane_routing_to_an_unknown_host_is_400(pool):
    base, path = pool
    before = open(path).read()
    status, body = _send(f"{base}/api/lane/igpu", "POST",
                         {"encoders": ["nosuchhost"], "rev": rosterio.rev(path)})
    assert status == 400
    assert "nosuchhost" in body
    assert open(path).read() == before


def test_get_presets_lists_the_run_linux_scripts():
    """The picker's only source. A route of its own, so the catalogue is not
    re-sent with every two-second status poll."""
    srv, base = _serve()
    try:
        status, ctype, body = _get(base + "/api/presets")
    finally:
        srv.shutdown()
    assert status == 200 and "application/json" in ctype
    presets = json.loads(body)["presets"]
    ids = {p["id"] for p in presets}
    assert "run_linux_dance_HQ_crf27.sh" in ids
    for p in presets:
        assert set(p) == {"id", "label", "quality", "photon_noise", "speed",
                          "params"}


def test_retry_all_is_one_request_that_names_no_clip(controllable):
    """Plural and src-less on purpose. The page cannot send the list: its
    failure panel is a capped preview, and the rows it dropped are still out of
    attempts. The batch reads state.jsonl instead."""
    from tools.archive_batch import control

    base, _path, control_dir = controllable
    status, out = _send(f"{base}/api/clips/retry-all", "POST", {})
    assert status == 200
    requests = control.take(control_dir)
    assert [r["action"] for _, r in requests] == ["retry-all"]
    assert "src" not in requests[0][1]
    assert requests[0][1]["id"] == out["id"]


def test_retry_all_as_text_plain_is_415_and_writes_nothing(controllable):
    """Bulk, and it survives a stopped run, so it gets the same CSRF gate the
    other control routes have."""
    from tools.archive_batch import control

    base, _path, control_dir = controllable
    status, _ = _send_raw(f"{base}/api/clips/retry-all", "POST", b'{"pad":"="}',
                          "text/plain;charset=UTF-8")
    assert status == 415
    assert control.take(control_dir) == []


def test_a_submission_carries_the_preset_the_page_picked(controllable):
    """The page sends one and the batch reads one. Dropped here, every queued
    folder silently got the fleet-fixed settings instead of the script the
    operator picked, and nothing said so."""
    from tools.archive_batch import control

    base, _path, control_dir = controllable
    status, _ = _send(f"{base}/api/jobs/submit", "POST",
                      {"host": "gpu1", "path": "/mnt/media/fresh",
                       "dest": "SetA/2026/New",
                       "preset": "run_linux_dance_HQ_crf27.sh"})
    assert status == 200
    requests = control.take(control_dir)
    assert requests[0][1]["preset"] == "run_linux_dance_HQ_crf27.sh"
