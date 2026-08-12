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

    @property
    def is_remote(self):
        return self.host != "local"


@dataclass(frozen=True)
class EncodePool:
    host: str
    slots: int
    threads_per_slot: int


@dataclass(frozen=True)
class Roster:
    denoisers: tuple
    encode: EncodePool

    def enabled(self):
        return tuple(d for d in self.denoisers if d.enabled)


def load_roster(path, core_count):
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
                        threads_per_slot=int(enc.get("threads_per_slot", 16)))
    roster = Roster(denoisers=denoisers, encode=encode)
    _validate(roster, core_count)
    return roster


def _denoiser(entry):
    try:
        return Denoiser(name=entry["name"], host=entry["host"],
                        backend=entry["backend"], device=int(entry.get("device", 0)),
                        tiling=entry.get("tiling", "none"),
                        enabled=bool(entry.get("enabled", True)),
                        window=int(entry.get("window", 0)),
                        margin=int(entry.get("margin", 32)),
                        port=int(entry.get("port", 0)))
    except KeyError as e:
        raise RosterError(f"denoiser entry missing required key: {e}")


def _validate(roster, core_count):
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

    concurrent = min(roster.encode.slots, len(active))
    threads = concurrent * roster.encode.threads_per_slot
    if threads > core_count:
        raise RosterError(
            f"roster would oversubscribe {core_count} cores: {concurrent} concurrent "
            f"encoders x {roster.encode.threads_per_slot} threads = {threads}")
