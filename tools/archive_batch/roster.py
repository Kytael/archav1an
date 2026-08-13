"""The denoiser roster: which devices are in play right now.

A denoiser is data, not code, so adding a host is a config entry. Encoding is
never remote: a remote host contributes a GPU and nothing else (spec 2, 5.2).
"""
import tomllib
from dataclasses import dataclass


# shift_num: BSVD reads 16 frames of future context, so a smaller margin cannot
# reproduce a whole-clip run. Proven by gate 2 at margin 32.
MIN_MARGIN = 16
TILING_MODES = {"none", "auto"}


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
    port: int = 0
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

    @property
    def is_remote(self):
        return self.host != "local"


@dataclass(frozen=True)
class EncodePool:
    host: str
    slots: int
    # SVT-AV1's --lp: a parallelism LEVEL in [0, 6], not a thread count. 0 lets
    # the encoder pick from the core count, which on a 16-thread host is level 5
    # and only reaches 6 at 24 threads. 6 is the default here because the pool
    # runs few slots: at 2 slots level 6 measured 26.27 fps against 23.05 at
    # level 4, and only 3 or more slots close that gap. See
    # docs/lp-and-encoder-parallelism.md. The cost is memory, ~4.8 GB a slot.
    lp_level: int = 6


@dataclass(frozen=True)
class Roster:
    denoisers: tuple
    encode: EncodePool

    def enabled(self):
        return tuple(d for d in self.denoisers if d.enabled)


def load_roster(path):
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise RosterError(f"roster not found: {path}")
    except tomllib.TOMLDecodeError as e:
        raise RosterError(f"roster is not valid TOML: {e}")

    denoisers = tuple(_denoiser(entry) for entry in data.get("denoiser", []))
    enc = data.get("encode", {})
    encode = EncodePool(host=enc.get("host", "local"),
                        slots=int(enc.get("slots", 2)),
                        lp_level=int(enc.get("lp_level", 6)))
    roster = Roster(denoisers=denoisers, encode=encode)
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
                        port=int(entry.get("port", 0)),
                        stage_source=bool(entry.get("stage_source", False)),
                        root=entry.get("root", ""))
    except KeyError as e:
        raise RosterError(f"denoiser entry missing required key: {e}")


def _validate(roster):
    names = [d.name for d in roster.denoisers]
    if len(names) != len(set(names)):
        raise RosterError("duplicate denoiser name in roster")

    active = roster.enabled()
    if not active:
        raise RosterError("no enabled denoiser in roster")

    if roster.encode.host != "local":
        raise RosterError("encode.host must be 'local': remote hosts contribute GPUs only")

    # Windowed tile-sequential denoise landed with gate 2 passing bit-identical
    # on 2026-08-11, so these keys now do something. They still have to be
    # coherent: a margin below the model's 16 frames of future context cannot
    # reproduce a whole-clip run, and an unknown tiling mode has no tile size.
    for d in roster.denoisers:
        if d.tiling not in TILING_MODES:
            raise RosterError(
                f"denoiser '{d.name}' has tiling '{d.tiling}'; expected one of "
                f"{sorted(TILING_MODES)}")
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

    ports = [d.port for d in roster.denoisers if d.is_remote]
    if any(p == 0 for p in ports):
        raise RosterError("a remote denoiser needs a port")
    if len(ports) != len(set(ports)):
        raise RosterError("duplicate port between remote denoisers")

    for d in roster.denoisers:
        if d.root and not d.is_remote:
            raise RosterError(
                f"denoiser '{d.name}' sets root on a local host; root names the "
                f"checkout on a REMOTE denoise host")
        if d.stage_source and not d.is_remote:
            raise RosterError(
                f"denoiser '{d.name}' sets stage_source on a local host; the "
                f"source is already local, so there is nothing to stage")

    # There is no thread budget to check. --lp is a level, and every level asks
    # for more threads than the machine has anyway: level 4 measured 87 threads
    # and level 6 measured 95, on 32 cores. The encoder clamps an out-of-range
    # level to 6 with a warning that scrolls past, so reject it here instead.
    if not 0 <= roster.encode.lp_level <= 6:
        raise RosterError(
            f"encode.lp_level is {roster.encode.lp_level}; --lp takes a parallelism "
            f"level in [0, 6], not a thread count. Use --pin for a core count.")
