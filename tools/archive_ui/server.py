"""Routing, and nothing else.

Read-only in part 1: every route is a GET. The POST routes that steer the run
arrive in part 2, which is also when the page's switches stop being inert.
"""
import json
import os
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from .metrics import render as render_metrics

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_TYPES = {".html": "text/html; charset=utf-8",
          ".css": "text/css; charset=utf-8",
          ".js": "application/javascript; charset=utf-8"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "archive-ui"

    # True once a status line has gone out. See _fail.
    _begun = False

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/status":
                self._json(self.server.snapshot_fn())
            elif path == "/metrics":
                self._text(render_metrics(self.server.snapshot_fn()),
                           "text/plain; version=0.0.4; charset=utf-8")
            elif path in ("/", "/index.html"):
                self._static("app.html")
            elif path.startswith("/static/"):
                self._static(path[len("/static/"):])
            else:
                self._fail(404)
        except ConnectionError:
            # The browser navigated away mid-response. BrokenPipeError alone is
            # not enough: a page that closes its socket while a body is going
            # out raises ConnectionResetError here, and that is the common case
            # for a dashboard someone leaves and comes back to.
            pass
        except Exception as exc:
            # One unreadable file must not end the daemon: it is the thing that
            # tells you the run is in trouble, so it has to outlive the trouble.
            self._fail(500, repr(exc))

    def _fail(self, code, explain=None):
        if self._begun:
            # A status line is already on the wire, so send_error would append a
            # whole second response to the first one's body. A client reads that
            # as a 200 whose body happens to contain the words "500 Internal
            # Server Error", which is worse than a truncated body: it looks like
            # success. Abandon the response instead and let the close say it
            # failed -- HTTP/1.0 closes after every response, so the client sees
            # a short read rather than the next reply spliced on.
            return
        try:
            self.send_error(code, explain=explain)
        except ConnectionError:
            pass        # nothing left to answer on

    def _static(self, name):
        # Decoded first, then normalised. The other order is the classic hole:
        # ..%2f..%2f survives a check made before decoding and becomes a climb
        # after it. This listens on the tailnet with no authentication.
        safe = posixpath.normpath("/" + unquote(name)).lstrip("/")
        # A %00 decodes to a NUL, and both realpath and open raise ValueError on
        # one -- which the catch-all in do_GET would turn into a 500. A name no
        # file can have is a 404, and this endpoint answers anything on the
        # tailnet, so it should not be provokable into an error page.
        if "\x00" in safe:
            self._fail(404)
            return
        base = os.path.realpath(STATIC)
        # realpath, not abspath: abspath is string arithmetic and cannot see a
        # symlink, so a link dropped in static/ would hand out whatever it
        # points at. The name is already climb-free by here; this covers the
        # file the name lands on.
        full = os.path.realpath(os.path.join(base, safe))
        if not full.startswith(base + os.sep):
            self._fail(403)
            return
        try:
            with open(full, "rb") as fh:
                body = fh.read()
        except OSError:
            self._fail(404)
            return
        ctype = _TYPES.get(os.path.splitext(safe)[1], "application/octet-stream")
        self._raw(body, ctype)

    def _json(self, obj):
        # The body is built before anything is sent, here and in _text. A
        # snapshot with a value json cannot serialise then fails while the
        # response can still be turned into a clean 500.
        self._raw(json.dumps(obj).encode("utf-8"), "application/json")

    def _text(self, text, ctype):
        self._raw(text.encode("utf-8"), ctype)

    def _raw(self, body, ctype):
        self.send_response(200)
        # Set here rather than after end_headers: send_response only buffers the
        # status line, and a send_error after it would flush both status lines
        # inside one response.
        self._begun = True
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page polls every 2 s; a cached snapshot is a stale dashboard.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass            # a poll every 2 seconds for a fortnight is not a log


def make_server(host, port, snapshot_fn):
    """ThreadingHTTPServer so one slow snapshot does not block the next poll."""
    srv = ThreadingHTTPServer((host, port), _Handler)
    srv.daemon_threads = True
    srv.snapshot_fn = snapshot_fn
    return srv
