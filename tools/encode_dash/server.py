"""Routing, and nothing else.

The GET routes read; the POST and DELETE routes call rosterio and touch one
file. The one exception is /api/run/start, which talks to the supervisor: the
daemon spawns tools/archive-batch.py when the operator asks, and it never
kills what it finds. Enabling a lane is a line in a TOML that the scheduler
happens to re-read before every clip.
"""
import json
import os
import posixpath
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from tools.archive_batch.presets import catalogue
from tools.archive_batch.roster import RosterError

from tools.encode_dash import control
from tools.encode_dash.supervisor import AlreadyRunning, SupervisorError

from . import rosterio
from .metrics import render as render_metrics

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Guards the rosterio call inside _write. rosterio's own guard -- read the
# file, compare expect_rev, then commit -- is not atomic across threads: two
# requests can both pass the rev check against the same rev before either has
# committed, and the second write would clobber the first with no 409.
# ThreadingHTTPServer gives every request its own thread, so two browser tabs,
# or one impatient double-click, can hit that window. This lock closes it by
# letting only one write run at a time in this process; it says nothing about
# an outside editor changing the file on disk, which is what expect_rev is
# for.
_write_lock = threading.Lock()

_TYPES = {".html": "text/html; charset=utf-8",
          ".css": "text/css; charset=utf-8",
          ".js": "application/javascript; charset=utf-8",
          # static/lane-presets.json. It is data the page fetches once, not a
          # snapshot, so it is a file rather than an /api route.
          ".json": "application/json"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "encode-dash"
    # Pinned, not inherited. _fail abandons a response once its status line is
    # out and relies on the close to say it failed, which is only true at 1.0.
    # Raising this to 1.1 -- a natural thing to try for a page that polls every
    # two seconds -- keeps the connection open, so _begun survives into the next
    # request on that socket and the second request gets no reply at all.
    # Stating it here makes the comment in _fail true by construction.
    protocol_version = "HTTP/1.0"
    # Suppresses the "Python/3.14.7" that BaseHTTPRequestHandler appends to the
    # Server header. This listens on the tailnet with no authentication, and the
    # interpreter's patch version is not something a status page needs to say.
    sys_version = ""

    # True once a status line has gone out. See _fail.
    _begun = False

    # Per-connection read timeout. Without it a peer that opens a connection
    # and never sends (or declares a Content-Length and stalls) pins one
    # thread of the threading server forever.
    timeout = 30

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/status":
                self._json(self.server.snapshot_fn())
            elif path == "/api/presets":
                # Its own route rather than a field on /api/status: the
                # catalogue is the run_linux_*.sh scripts on disk and changes
                # only when somebody edits one, so shipping it with every
                # two-second poll would send the same 11 entries for ever.
                self._json({"presets": catalogue()})
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
            #
            # Printed here rather than through log_error, which send_error also
            # calls for every 404 -- and a 404 from anything scanning the tailnet
            # is noise, while a 500 is a fault. Without this line the only signal
            # that the daemon is failing is the scrape failing on the Pi, which
            # says a request went wrong and nothing about why.
            print(f"[encode-dash] {path}: {exc!r}", file=sys.stderr, flush=True)
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

    # A write is small, and this is the only body this daemon ever reads. The
    # cap is what stops an unauthenticated tailnet POST from asking the daemon
    # to buffer a gigabyte.
    MAX_BODY = 64 * 1024

    def do_POST(self):
        # Two frames, not one, and the split is by what the route actually
        # needs. A roster edit must quote the roster's revision and needs
        # roster_path; a control request writes into a different directory and
        # has no revision concept at all. Making /api/run/stop quote a roster
        # rev would mean the stop button stops working the moment somebody
        # edits the TOML in vim -- one of the times you most want it.
        path = self.path.split("?", 1)[0]
        if path == "/api/run/start":
            self._supervise(self._route_start)
        elif (path == "/api/run/stop" or path.startswith("/api/clip/")
                or path == "/api/clips/retry-all"
                or path == "/api/jobs/submit"):
            # A control request, like stop and retry: it writes into the
            # control directory and has no roster revision to quote.
            self._control(self._route_control)
        else:
            self._write(self._route_post)

    def do_DELETE(self):
        self._write(self._route_delete)

    def _write(self, route):
        """Shared frame for both write verbs: body, dispatch, error mapping."""
        path = self.path.split("?", 1)[0]
        try:
            if not self._content_type_is_json():
                return
            if self.server.roster_path is None:
                self._fail_json(503, "this daemon was started with no roster path")
                return
            body = self._body()
            if body is None:
                return
            rev = body.pop("rev", None)
            # Required on every write, never optional. §6: a write quotes the
            # revision it was based on, and a write that quotes nothing is a
            # write with no idea what it is overwriting -- the guard exists
            # precisely so two editors cannot clobber each other.
            #
            # It is also half the CSRF story. This daemon listens on the
            # tailnet with no authentication, so the only thing an attacking
            # page lacks is the token: same-origin policy stops it reading
            # /api/status, and without a rev it cannot forge one. The
            # Content-Type check above is the other half.
            #
            # isinstance, not truthiness: rev() returns a string, and a client
            # that sent the old [mtime, size] list, or a number, or null, has
            # not quoted a revision this daemon ever issued.
            if not isinstance(rev, str):
                self._fail_json(
                    400, "every write must quote the roster rev it was based "
                         "on, as a string; reload the page and retry")
                return
            # Serialised: see _write_lock's comment above. Only the rosterio
            # call needs to be inside it -- reading the body and mapping the
            # result don't touch the file.
            with _write_lock:
                result = route(path, body, rev)
            if result is None:
                # None is also what a successful commit's rev() could return
                # if the file vanished under it -- vanishingly unlikely, since
                # _commit just wrote it -- but every route function returns a
                # real rev on success, so in practice None here only ever
                # means no route matched this path.
                self._fail_json(404, "no such route")
                return
            self._json({"ok": True, "rev": result})
        except rosterio.StaleRoster as exc:
            # 409, and the page reloads. This is the vim-over-ssh case: the
            # operator edited the file by hand while the page was open.
            self._fail_json(409, str(exc))
        except (rosterio.LaneNotFound, rosterio.EncoderNotFound) as exc:
            # Both are 404: a name the roster does not carry is a missing
            # thing, not a malformed request. Without EncoderNotFound here it
            # would fall to the RosterError branch below and answer 400, so
            # the two tables would report the same mistake differently.
            self._fail_json(404, str(exc))
        except RosterError as exc:
            # roster.py's message, verbatim. §6: the form surfaces those
            # messages rather than restating the rules.
            self._fail_json(400, str(exc))
        except control.ControlError as exc:
            # A yield is a roster edit and a control request at once, so this
            # frame can raise it too.
            self._fail_json(400, str(exc))
        except ConnectionError:
            pass
        except Exception as exc:
            print(f"[encode-dash] {path}: {exc!r}", file=sys.stderr, flush=True)
            self._fail_json(500, repr(exc))

    def _control(self, route):
        """Shared frame for the control routes: body, dispatch, error mapping.

        The Content-Type gate is kept and the rev requirement is not. That gate
        is the CSRF defence that matters: application/json is not a CORS-simple
        content type, so it forces a preflight an attacking page cannot
        satisfy. The rev was the second half of that story for roster writes,
        and it is a roster concept with no meaning here.
        """
        path = self.path.split("?", 1)[0]
        try:
            if not self._content_type_is_json():
                return
            if self.server.control_dir is None:
                self._fail_json(
                    503, "this daemon was started with no control directory")
                return
            body = self._body()
            if body is None:
                return
            # app.js folds rev into every write body. Control does not use it,
            # and receiving one is not an error.
            body.pop("rev", None)
            result = route(path, body)
            if result is None:
                self._fail_json(404, "no such route")
                return
            self._json({"ok": True, "id": result})
        except control.ControlError as exc:
            self._fail_json(400, str(exc))
        except ConnectionError:
            pass
        except Exception as exc:
            print(f"[encode-dash] {path}: {exc!r}", file=sys.stderr, flush=True)
            self._fail_json(500, repr(exc))

    def _supervise(self, route):
        """Shared frame for the run-control routes: body, dispatch, error
        mapping. Unlike the roster routes, a supervisor route never carries a
        rev: starting a run is not an edit of the roster file, so the roster
        rev has no meaning here. The Content-Type gate still applies -- this
        daemon listens on the tailnet with no authentication, and
        application/json forces a preflight an attacking page cannot satisfy.
        """
        path = self.path.split("?", 1)[0]
        try:
            if not self._content_type_is_json():
                return
            if self.server.supervisor is None:
                self._fail_json(503, "this daemon was started with no supervisor")
                return
            body = self._body()
            if body is None:
                return
            result = route(path, body)
            if result is None:
                self._fail_json(404, "no such route")
                return
            self._json({"ok": True, "pid": result})
        except AlreadyRunning as exc:
            self._fail_json(409, str(exc))
        except SupervisorError as exc:
            # The daemon cannot manage the run directory. adopt() logged the
            # same fault at boot and carried on read-only, so this is the
            # first the operator hears of it: the sentence, not a repr.
            self._fail_json(503, str(exc))
        except ConnectionError:
            pass
        except Exception as exc:
            print(f"[encode-dash] {path}: {exc!r}", file=sys.stderr, flush=True)
            self._fail_json(500, repr(exc))

    def _route_start(self, path, body):
        """Start a batch run. The supervisor answers one run at a time, so a
        second POST while one is live answers 409 rather than spawning a
        second process. The body is ignored: starting a run carries no
        arguments today, and the Content-Type gate above is the whole CSRF
        defence.
        """
        return self.server.supervisor.start()

    def _fail_json(self, code, message):
        """_fail's counterpart for the write routes: a JSON body, not
        send_error's HTML page.

        send_error renders a whole HTML document and HTML-escapes `explain`
        into it. Task 8's page reads a write failure with `await r.text()`
        and drops that straight into the error line it shows the operator, so
        an HTML document there would bury "a tiled denoiser needs a window" in
        markup, and the escaping means a RosterError message would not reach
        the client verbatim -- the one thing §6 requires. This sends the
        message as-is in a small JSON object instead.

        Honours the same _begun contract as _fail, for the same reason: once
        a status line is on the wire, HTTP/1.0 has no way to retract it, and
        writing a second one here would splice a second response onto the
        first.
        """
        if self._begun:
            return
        body = json.dumps({"error": message}).encode("utf-8")
        try:
            self.send_response(code)
            self._begun = True
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            pass        # nothing left to answer on

    def _content_type_is_json(self):
        """True once the request declares a JSON body; answers 415 otherwise.

        An HTML form can only post application/x-www-form-urlencoded,
        multipart/form-data or text/plain, and all three are CORS-simple: the
        browser sends them cross-origin with no preflight and no permission
        from this daemon. So an auto-submitting
        `<form enctype="text/plain" action="http://encoder-host:9328/...">` on any
        page the operator happens to visit could reach these routes, and this
        daemon has no authentication to stop it. Requiring application/json
        makes every write a non-simple request, which forces a preflight the
        attacking page cannot satisfy.

        A missing header is refused too. It is not a legitimate shape here --
        every real client of these routes sets it -- and accepting it would
        reopen the hole for a body posted without one.
        """
        raw = self.headers.get("Content-Type") or ""
        # Strip the parameters: "application/json; charset=utf-8" is a JSON
        # body, and fetch() sends exactly that from some stacks.
        if raw.split(";", 1)[0].strip().lower() == "application/json":
            return True
        self._fail_json(415, "write routes require Content-Type: application/json")
        return False

    def _body(self):
        """The parsed JSON body, or None once a failure has been answered."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._fail_json(400, "bad Content-Length")
            return None
        # read(-1) would drain the connection until EOF, so a peer that
        # declares a negative length must be refused before any read happens.
        if length < 0:
            self._fail_json(400, "bad Content-Length")
            return None
        if length > self.MAX_BODY:
            self._fail_json(413, "body too large")
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._fail_json(400, f"body is not JSON: {exc}")
            return None
        if not isinstance(body, dict):
            self._fail_json(400, "body must be a JSON object")
            return None
        return body

    def _route_post(self, path, body, rev):
        roster = self.server.roster_path
        parts = path.strip("/").split("/")
        # /api/lane/<name>/yield
        if len(parts) == 4 and parts[:2] == ["api", "lane"] and parts[3] == "yield":
            name = unquote(parts[2])
            if self.server.control_dir is None:
                raise control.ControlError(
                    "this daemon was started with no control directory, so it "
                    "cannot ask for a kill")
            # Order is load-bearing, §5.3: disable first, then ask for the
            # kill. Reversed, the worker finishes handling YieldRequested,
            # loops, still finds itself enabled, and retakes the same clip --
            # so the GPU the operator asked for never comes back.
            #
            # set_enabled raising leaves nothing written anywhere, which is
            # what makes a stale rev a clean refusal rather than a half-done
            # yield.
            new_rev = rosterio.set_enabled(roster, name, False, expect_rev=rev)
            control.submit(self.server.control_dir, "yield", time.time(),
                           lane=name)
            return new_rev
        # /api/lane/<name>/enabled
        if len(parts) == 4 and parts[:2] == ["api", "lane"] and parts[3] == "enabled":
            name = unquote(parts[2])
            # Not bool(...): "false" the string is truthy, and coercing it
            # would flip a lane the client asked to turn off. set_enabled
            # rejects anything that isn't a real boolean instead.
            return rosterio.set_enabled(roster, name, body.get("enabled"),
                                        expect_rev=rev)
        if parts == ["api", "lane"]:
            return rosterio.add_lane(roster, body, expect_rev=rev)
        # /api/lane/<name> -- edit a lane already in the roster. Last of the
        # lane routes, so the three-segment shape cannot shadow the four
        # -segment ones above it.
        if len(parts) == 3 and parts[:2] == ["api", "lane"]:
            return rosterio.update_lane(roster, unquote(parts[2]), body,
                                        expect_rev=rev)
        # /api/encoder/<name>/enabled -- stop or start one encode host. A
        # roster write like the lane switch, not a control request: it takes
        # effect at that host's next clip and kills nothing in flight. Its own
        # route because encoder and denoiser names share no namespace: this
        # roster has both a lane and an encoder called "gpu4".
        if len(parts) == 4 and parts[:2] == ["api", "encoder"] \
                and parts[3] == "enabled":
            return rosterio.set_encoder_enabled(roster, unquote(parts[2]),
                                                body.get("enabled"),
                                                expect_rev=rev)
        # /api/encoder -- add one encode host, switched off.
        if parts == ["api", "encoder"]:
            return rosterio.add_encoder(roster, body, expect_rev=rev)
        # /api/encoder/<name> -- edit a host already in the pool. Last of the
        # encoder routes, so this three-segment shape cannot shadow the
        # four-segment /enabled above it.
        if len(parts) == 3 and parts[:2] == ["api", "encoder"]:
            return rosterio.update_encoder(roster, unquote(parts[2]), body,
                                           expect_rev=rev)
        if parts == ["api", "encode", "lp_level"]:
            return rosterio.set_lp_level(roster, body.get("lp_level"),
                                         expect_rev=rev)
        return None

    def _route_delete(self, path, body, rev):
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["api", "encoder"]:
            return rosterio.remove_encoder(self.server.roster_path,
                                           unquote(parts[2]), expect_rev=rev)
        if len(parts) == 3 and parts[:2] == ["api", "lane"]:
            return rosterio.remove_lane(self.server.roster_path,
                                        unquote(parts[2]), expect_rev=rev)
        return None

    def _route_control(self, path, body):
        control_dir = self.server.control_dir
        parts = path.strip("/").split("/")
        if parts == ["api", "run", "stop"]:
            return control.submit(control_dir, "stop", time.time())
        # /api/clip/<src>/retry. <src> is a clip path and holds slashes, so it
        # is matched by a fixed prefix and a fixed suffix and rejoined, not by
        # the exact-segment-count pattern the roster routes use. self.path is
        # the raw request target, so a %2F the page wrote survives split("/")
        # and unquote turns it back here; rejoining also makes an unencoded
        # path work rather than answering a silent 404.
        # /api/clips/retry-all. Plural, and no <src>: the page cannot send the
        # list, because its failure panel is a capped preview and the rows it
        # dropped are still out of attempts. The batch reads state.jsonl, which
        # is what the `N exhausted` count on that panel is computed from, so
        # the button means exactly what the number beside it says.
        if parts == ["api", "clips", "retry-all"]:
            return control.submit(control_dir, "retry-all", time.time())
        if len(parts) >= 4 and parts[:2] == ["api", "clip"] and parts[-1] == "retry":
            src = unquote("/".join(parts[2:-1]))
            if not src:
                raise control.ControlError("a retry must name a clip")
            return control.submit(control_dir, "retry", time.time(), src=src)
        # /api/jobs/submit -- queue a folder of encode jobs. A control request,
        # not a roster write: it names work, not configuration. The batch
        # probes the folder on its own thread and acks the count, so this
        # returns as soon as the request file is written.
        if parts == ["api", "jobs", "submit"]:
            folder = (body.get("path") or "").strip()
            if not folder:
                raise control.ControlError("a submission must name a folder")
            if not folder.startswith("/"):
                raise control.ControlError(
                    f"the folder must be an absolute path on the host that "
                    f"holds it: {folder}")
            dest = (body.get("dest") or "").strip()
            if not dest:
                raise control.ControlError(
                    "a submission must name a destination under encoded/")
            # preset is carried through, not dropped. The page sends one and
            # the batch reads one; left out here, every queued folder silently
            # got the fleet-fixed settings instead of the script the operator
            # picked, and nothing said so.
            return control.submit(control_dir, "submit", time.time(),
                                  path=folder, host=(body.get("host") or "").strip(),
                                  dest=dest,
                                  preset=(body.get("preset") or "").strip())
        return None


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer with a cap on live request threads.

    Without the cap, a peer that opens connections faster than they finish
    grows one thread per connection without bound. A refused connection gets a
    minimal 503 and an immediate close; real clients see it only under load
    that already exceeds what this daemon can serve.
    """

    MAX_THREADS = 32

    def process_request(self, request, client_address):
        if threading.active_count() >= self.MAX_THREADS:
            try:
                # This runs in the accept-loop thread before any handler
                # setup() has applied a socket timeout. Non-blocking, not a
                # short timeout: a timeout still serialises every new
                # connection behind it, so a peer advertising a zero receive
                # window could hold acceptance at one connection per timeout
                # indefinitely -- the cap's own courtesy reply turned into
                # the denial it exists to prevent. If the 55 bytes do not fit
                # the send buffer this instant, BlockingIOError lands in the
                # OSError arm and the peer gets the close without the 503.
                request.setblocking(False)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\n"
                    b"Connection: close\r\n\r\n")
            except OSError:
                pass
            finally:
                request.close()
            return
        super().process_request(request, client_address)


def make_server(host, port, snapshot_fn, roster_path=None, control_dir=None,
                supervisor=None):
    """ThreadingHTTPServer so one slow snapshot does not block the next poll.

    roster_path is optional so a read-only daemon -- and every test that only
    exercises GET -- still constructs. Without it the write routes answer 503
    rather than pretending to have written something. control_dir is optional
    for the same reason and answers 503 the same way. supervisor is optional
    too: a daemon started without one cannot spawn a batch, so /api/run/start
    answers 503 instead of pretending to have started one.
    """
    srv = _Server((host, port), _Handler)
    srv.daemon_threads = True
    srv.snapshot_fn = snapshot_fn
    srv.roster_path = roster_path
    srv.control_dir = control_dir
    srv.supervisor = supervisor
    return srv
