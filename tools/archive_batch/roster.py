"""The roster: which devices are in play right now.

A denoiser is data, not code, so adding a host is a config entry. An encoder is
data for the same reason, now that encoding is no longer pinned to one machine
-- see the Encoder docstring for what retired that rule.
"""
import ipaddress
import tomllib
import re

from dataclasses import dataclass


# shift_num: BSVD reads 16 frames of future context, so a smaller margin cannot
# reproduce a whole-clip run. Proven by gate 2 at margin 32.
MIN_MARGIN = 16
# Each encoder reserves a ten-port block, so its slot count can change without
# an elevated firewall edit on the encode host. encoder-host 5300, gpu2 5310,
# gpu4 5320, gpu5 5330.
BLOCK_PORTS = 10
TILING_MODES = {"none", "auto"}
# An explicit tile, "HxW" or a bare square size, as parse_tile_arg() in
# bsvd_windowed.py reads it. Only usable since engines are named for the shape
# that built them: before that, two tile sizes resolved to one cache file and
# the second silently evicted the first.
TILE_RE = re.compile(r"^\d+x\d+$|^\d+$")


class RosterError(Exception):
    """The roster is unusable and the run must not start."""


@dataclass(frozen=True)
class Denoiser:
    name: str
    host: str
    backend: str
    device: int
    tiling: str
    enabled: bool
    window: int = 0
    margin: int = 32
    # Set only when an old roster still carries `port` on a denoiser, so
    # _validate can refuse it by name. Never read for anything else.
    extra_port: int = 0
    # True when the remote does NOT hold the archive, so each clip must be
    # copied to it. gpu1 reads the archive in place and leaves this false,
    # which is what keeps 2.5 TB off a remote's root. A host that has only a
    # GPU to offer still qualifies under spec 2 -- it contributes no storage
    # and no encoding -- but it does cost one source transfer per clip.
    stage_source: bool = False
    # Where the checkout lives on the remote. dispatch defaults to
    # ~/archav1an, which is gpu1's layout; gpu2 and gpu3 keep theirs under
    # ~/reposetc/archav1an. Getting this wrong stages the clip into a
    # directory that rsync happily creates and then runs `cd` into a tree with
    # no dispatch in it, so the lane goes quiet instead of failing loudly.
    root: str = ""
    # Which encoders this lane may use. Empty means every enabled encoder.
    #
    # An allowlist on the lane rather than a denylist on the encoder, because
    # it fails safe: add a fifth, slow encoder later and a fast lane will not
    # use it until it is named. gpu1_4090 excludes gpu2, whose 16.23 fps
    # ceiling is a HOST property and not a per-stream rate -- it measures
    # 15.84 / 16.23 / 15.76 / 16.45 fps at one, two, three and five streams, so
    # a second slot there halves one encoder rather than adding a second.
    encoders: tuple = ()

    @property
    def is_remote(self):
        return self.host != "local"

    def allows(self, encoder_name):
        return not self.encoders or encoder_name in self.encoders


@dataclass(frozen=True)
class Encoder:
    """One host that can encode. A host may encode, denoise, both or neither.

    Encoding used to be pinned to encoder-host (spec 2026-08-11 5.2, "Encoding is
    never remote"). That rule is retired: the fleet produces about 39.9 fps of
    denoised frames against the 28.06 fps encoder-host manages while its own iGPU
    lane runs, so the encode side had to stop being one machine. See
    the design notes, which are not part of this tree
    """
    name: str
    host: str
    # 6, not 2: a slot is held for a whole clip, so a count below the number of
    # enabled denoisers blocks a lane.
    slots: int = 6
    # SVT-AV1's --lp: a parallelism LEVEL in [0, 6], not a thread count. See
    # docs/lp-and-encoder-parallelism.md.
    lp_level: int = 4
    # ON A REMOTE ENCODER: the address this encoder binds its y4m listener to,
    # and the address the denoise host connects to. Never 0.0.0.0 there, because
    # gpu2 is a laptop and a bind that fails off-LAN is the point -- the lane
    # refuses to start rather than listen on an untrusted network.
    #
    # ON A LOCAL ENCODER it is read by nothing: dispatch_cmd passes it only
    # under encoder.is_remote, the listener is dispatch's own netstream recv
    # with no --bind (so 0.0.0.0), and the connect address is CALLBACK_IP in
    # tools/archive-batch.py. Documented rather than deleted, because a reader
    # would otherwise take it for a bind address it is not.
    stream_ip: str = ""
    # The trusted network to pick a bind address from, when the host's address
    # is not fixed. gpu2 and gpu3 get theirs from DHCP, so a literal
    # stream_ip goes stale and the lane dies until the roster is edited.
    # Mutually exclusive with stream_ip.
    stream_net: str = ""
    # First port of this encoder's block. Slot N listens on port_base + N.
    # Blocks are ten wide so a slot-count change needs no firewall edit.
    port_base: int = 0
    # Where the checkout lives on a remote encode host, as Denoiser.root is for
    # a denoise host.
    root: str = ""
    enabled: bool = True

    @property
    def is_remote(self):
        return self.host != "local"

    def port_for(self, slot):
        return self.port_base + slot


@dataclass(frozen=True)
class Roster:
    denoisers: tuple
    encoders: tuple

    def enabled(self):
        return tuple(d for d in self.denoisers if d.enabled)

    def enabled_encoders(self):
        return tuple(e for e in self.encoders if e.enabled)


def load_roster(path):
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise RosterError(f"roster not found: {path}")
    except tomllib.TOMLDecodeError as e:
        raise RosterError(f"roster is not valid TOML: {e}")

    denoisers = tuple(_denoiser(entry) for entry in data.get("denoiser", []))
    encoders = tuple(_encoder(entry) for entry in data.get("encoder", []))
    if encoders and "encode" in data:
        raise RosterError(
            "roster has both [[encoder]] entries and a legacy [encode] table. "
            "Delete [encode]: keeping it would silently discard its slots and "
            "lp_level, which is the same quiet wrong value this loader refuses "
            "for a denoiser's port.")
    if not encoders and "encode" in data:
        # One local encoder from the pre-pool [encode] table. Named "local"
        # rather than for its machine, because nothing in the file says which
        # machine "local" is.
        # stream_ip stays unset on purpose: dispatch then resolves the callback
        # itself and behaves exactly as it did before the pool existed.
        # port_base does NOT -- see below.
        enc = data["encode"]
        encoders = (Encoder(name="local", host=enc.get("host", "local"),
                            slots=int(enc.get("slots", 6)),
                            lp_level=int(enc.get("lp_level", 4)),
                            # 5300 is what every pre-pool roster used on the
                            # denoiser side, so a migrated roster keeps the
                            # ports it already had. Not 0: port_for would then
                            # hand build_command `--remote-port 0` and today's
                            # gpu1 split lane would stream to nowhere.
                            port_base=5300),)
    roster = Roster(denoisers=denoisers, encoders=encoders)
    _validate(roster)
    return roster


def _denoiser(entry):
    try:
        return Denoiser(name=entry["name"], host=entry["host"],
                        backend=entry["backend"], device=int(entry.get("device", 0)),
                        tiling=entry.get("tiling", "none"),
                        enabled=bool(entry.get("enabled", True)),
                        window=int(entry.get("window", 0)),
                        margin=int(entry.get("margin", 32)),
                        extra_port=int(entry.get("port", 0)),
                        stage_source=bool(entry.get("stage_source", False)),
                        root=entry.get("root", ""),
                        encoders=tuple(entry.get("encoders", ())))
    except KeyError as e:
        raise RosterError(f"denoiser entry missing required key: {e}")


def _encoder(entry):
    try:
        return Encoder(name=entry["name"], host=entry["host"],
                       slots=int(entry.get("slots", 6)),
                       lp_level=int(entry.get("lp_level", 4)),
                       stream_ip=entry.get("stream_ip", ""),
                       stream_net=entry.get("stream_net", ""),
                       port_base=int(entry.get("port_base", 0)),
                       root=entry.get("root", ""),
                       enabled=bool(entry.get("enabled", True)))
    except KeyError as e:
        raise RosterError(f"encoder entry missing required key: {e}")


def _validate(roster):
    if not roster.denoisers:
        raise RosterError("no denoiser in roster")

    names = [d.name for d in roster.denoisers]
    if len(names) != len(set(names)):
        raise RosterError("duplicate denoiser name in roster")

    # There is deliberately no check that one of those denoisers is ENABLED.
    # Every lane off is a state the operator asks for: yielding the last enabled
    # lane parks the run until a lane comes back. That makes it a roster every
    # reader has to accept, not only the write that creates it. While it raised
    # here, the batch live read turned it into a banner saying the roster was
    # unreadable, and the daemon snapshot lit encode_roster_error -- a live
    # alert on the Pi -- so the operator was paged for doing what the page had
    # just offered them. run() gives every rostered name a worker rather than
    # only the enabled ones, so the parked run resumes by itself when a lane
    # comes back on. An empty roster stays refused above, because with no lane
    # there is no worker to park and nothing to switch on: run() would exit
    # rather than wait, and that is not a state anything recovers from.

    # Windowed tile-sequential denoise landed with gate 2 passing bit-identical
    # on 2026-08-11, so these keys now do something. They still have to be
    # coherent: a margin below the model's 16 frames of future context cannot
    # reproduce a whole-clip run, and an unknown tiling mode has no tile size.
    for d in roster.denoisers:
        if d.tiling not in TILING_MODES and not TILE_RE.match(d.tiling):
            raise RosterError(
                f"denoiser '{d.name}' has tiling '{d.tiling}'; expected one of "
                f"{sorted(TILING_MODES)}, an explicit 'HxW', or a square size")
        if d.tiling != "none":
            if d.window < 1:
                raise RosterError(
                    f"denoiser '{d.name}' is tiled but sets window {d.window}; a "
                    f"tiled denoiser needs a window or it buffers the whole clip")
            if d.margin < MIN_MARGIN:
                raise RosterError(
                    f"denoiser '{d.name}' sets margin {d.margin}, below the model's "
                    f"{MIN_MARGIN} frames of context; windowing would not be exact")
        elif d.window:
            raise RosterError(
                f"denoiser '{d.name}' sets window {d.window} without tiling; "
                f"windowing only applies to a tiled denoiser")

    for d in roster.denoisers:
        if d.extra_port:
            raise RosterError(
                f"denoiser '{d.name}' sets port {d.extra_port}. The port moved "
                f"to the encoder: the y4m listener now lives on the encode "
                f"host, so set port_base on an [[encoder]] entry instead. See "
                f"the design notes, which are not part of this tree")
        if d.root and not d.is_remote:
            raise RosterError(
                f"denoiser '{d.name}' sets root on a local host; root names the "
                f"checkout on a REMOTE denoise host")
        if d.stage_source and not d.is_remote:
            raise RosterError(
                f"denoiser '{d.name}' sets stage_source on a local host; the "
                f"source is already local, so there is nothing to stage")

    if not roster.encoders:
        raise RosterError("no encoder in roster")

    enc_names = [e.name for e in roster.encoders]
    known = set(enc_names)
    if len(enc_names) != len(known):
        raise RosterError("duplicate encoder name in roster")

    for d in roster.denoisers:
        for name in d.encoders:
            if name not in known:
                raise RosterError(
                    f"denoiser '{d.name}' allows encoder '{name}', which is not "
                    f"in the roster. Known encoders: {sorted(known)}")

    # As with denoisers, there is deliberately no check that one encoder is
    # ENABLED. Every encoder off parks the run; it does not break it -- every
    # lane waits together and one toggle starts them all again.
    #
    # A lane starved WHILE the fleet still encodes is the opposite case, and it
    # is refused here. The rest of the fleet keeps draining the queue, so the
    # run looks healthy while that one lane waits on an encoder that is never
    # coming; the scheduler's picker has no exit from that wait. Only enabled
    # lanes count: a lane that is off takes no clip, so it starves nothing, and
    # refusing it would reject a roster whose owner has simply turned a host
    # off. Reachable with the shipped file, where gpu1_4090 names three
    # encoders.
    live = {e.name for e in roster.encoders if e.enabled}
    if live:
        for d in roster.enabled():
            if d.encoders and not any(d.allows(name) for name in live):
                raise RosterError(
                    f"denoiser '{d.name}' allows only {sorted(d.encoders)}, and "
                    f"none of those encoders is enabled. That lane could never "
                    f"take a clip while {sorted(live)} carry the run, so it "
                    f"would wait for ever. Enable one of its encoders, widen "
                    f"its allowlist, or disable the lane.")

    seen = {}
    for e in roster.encoders:
        if e.slots < 1:
            raise RosterError(
                f"encoder '{e.name}' has slots {e.slots}; an encoder needs at "
                f"least one slot or it can never take a clip")
        # A level, not a thread count. The encoder clamps an out-of-range level
        # to 6 with a warning that scrolls past, so reject it here instead.
        if not 0 <= e.lp_level <= 6:
            raise RosterError(
                f"encoder '{e.name}' sets lp_level {e.lp_level}; --lp takes a "
                f"parallelism level in [0, 6], not a thread count.")
        if e.stream_ip and e.stream_net:
            raise RosterError(
                f"encoder '{e.name}' sets both stream_ip and stream_net; that "
                f"hides which one is in effect. Use stream_ip for a host whose "
                f"address is fixed and stream_net for one that moves.")
        if e.root and not e.is_remote:
            raise RosterError(
                f"encoder '{e.name}' sets root on a local host; root names the "
                f"checkout on a REMOTE encode host")
        if not 1 <= e.port_base <= 65535 - BLOCK_PORTS:
            raise RosterError(
                f"encoder '{e.name}' sets port_base {e.port_base}; it needs a "
                f"port block even on a local host, because a remote denoiser "
                f"still has to connect to it, and the block must fit under "
                f"65535")
        if e.stream_net:
            # Parsed here rather than in netresolve, which reads it per clip:
            # a typo that only raises there kills the lane on its first clip,
            # days after the operator stopped watching, and passes every smoke
            # test until then.
            try:
                ipaddress.ip_network(e.stream_net, strict=False)
            except ValueError as exc:
                raise RosterError(
                    f"encoder '{e.name}' has stream_net '{e.stream_net}', which "
                    f"is not a network: {exc}")
        if e.is_remote:
            # A local encoder may omit the address: dispatch resolves the
            # callback itself, which is what a pre-pool roster relies on.
            if not e.stream_ip and not e.stream_net:
                raise RosterError(
                    f"encoder '{e.name}' is remote and sets neither stream_ip "
                    f"nor stream_net; the listener needs an address to bind "
                    f"and the denoise host needs one to connect to")
        # BLOCK_PORTS, not e.slots. Both this file and the example roster
        # promise that a slot-count change needs no firewall edit, and that is
        # only true while the next encoder starts a whole block further on.
        # Walking only `slots` let an encoder at 5306 sit beside one at 5300
        # with six slots and collide the first time those slots went to seven.
        for slot in range(max(BLOCK_PORTS, e.slots)):
            port = e.port_for(slot)
            if port in seen:
                raise RosterError(
                    f"encoder '{e.name}' port block {e.port_base}.."
                    f"{e.port_base + e.slots - 1} overlaps encoder "
                    f"'{seen[port]}' at port {port}")
            seen[port] = e.name
