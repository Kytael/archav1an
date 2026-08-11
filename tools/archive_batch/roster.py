"""The denoiser roster: which devices are in play right now.

A denoiser is data, not code, so adding a host is a config entry. Encoding is
never remote: a remote host contributes a GPU and nothing else (spec 2, 5.2).
"""
import tomllib
from dataclasses import dataclass


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

    # Windowed tile-sequential denoise is spec 5.5 and is not built yet. Accepting
    # the keys silently would run BSVD full-frame on an 8 GB card, which either
    # runs out of memory or falls back to CPU without saying so. Only enabled
    # entries are rejected, so a roster can carry a disabled entry ready for it.
    for d in roster.enabled():
        if d.window or d.tiling != "none":
            raise RosterError(
                f"denoiser '{d.name}' sets tiling/window, which is not implemented "
                f"yet (spec 5.5). Remove the keys or disable the entry.")

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
