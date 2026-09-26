"""Parse the probed manifest and decide processing order.

The manifest is produced once by walking gpu1 over ssh (see the spec, 5.3),
because probing a few thousand files across 9p is slow enough that resume must never
repeat it.
"""
import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

_YEAR = re.compile(r"^[0-9]{4}$")
_SET_RANK = {"SetA": 0, "SetB": 1}
_NO_YEAR = 9999


@dataclass(frozen=True)
class Clip:
    src: str        # path relative to ARCHIVE_ROOT, e.g. "SetA/2001/event-a/x.MOV"
    rel_dir: str    # directory part of src
    stem: str       # basename without extension
    size: int       # bytes
    frames: int     # video frame count
    # False marks an ENCODE JOB: no denoise pass at all, run straight through
    # SVT-AV1 on an encode host's CPU. Defaulted, so every manifest, state
    # record and test written before this field keeps its meaning.
    denoise: bool = True
    # The machine holding `src`. "" means SOURCE_HOST, which is every archive
    # clip. An encode job submitted from outside the archive names its own.
    src_host: str = ""
    # Which run_linux_*.sh preset encodes this job, or "" for the fleet-fixed
    # settings. Only ever set on an encode job: an archive clip's output must
    # not depend on which device took it, and a per-job setting there would
    # make it depend on when it was queued instead.
    preset: str = ""

    @property
    def is_encode_job(self):
        return not self.denoise


def parse_manifest(text):
    """Parse manifest TSV text into a tuple of Clip.

    Row layout is path, size, then one "rate,frames" column per video stream,
    then duration. A source with an embedded thumbnail stream has two rate
    columns, so frames always comes from the first. The trailing duration
    column is not read: ordering and the fps figures both work from frames.
    """
    clips = []
    skipped = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            # A truncated manifest would otherwise shrink the run silently, and
            # the missing clips would never be noticed.
            skipped.append(line[:120])
            continue
        src = fields[0]
        size = int(fields[1])
        frames = _frames_from(fields[2])
        p = PurePosixPath(src)
        clips.append(Clip(src=src, rel_dir=str(p.parent), stem=p.stem,
                          size=size, frames=frames))
    if skipped:
        print(f"[archive-batch] manifest: skipped {len(skipped)} malformed row(s), "
              f"first: {skipped[0]!r}", file=sys.stderr)
    return tuple(clips)


def parse_encode_manifest(text):
    """Parse the ENCODE manifest into a tuple of Clip with denoise=False.

    Its own function, not a mode flag on parse_manifest: the two files are
    different shapes, and one parser answering to both would have to guess
    which from the column count.

    The row is an archive row with three columns prepended --

        host <TAB> dest <TAB> preset <TAB> abs_path <TAB> size
          <TAB> rate,frames <TAB> ...

    -- because an archive clip's rel_dir IS its own parent directory, while an
    encode job's is a destination under encoded/ that its source path does not
    carry. For an archive clip those two strings were always the same and code
    derived either from either; for an encode job they are unrelated, so
    nothing may do that any more.

    `preset` sits third, among the columns this program writes, and not after
    the ones ffprobe produced. Those trail off: a source with an embedded
    thumbnail emits a second rate column, so a field read from the end is a
    duration for some files and a rate for others.
    """
    clips = []
    skipped = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        # Seven, not six: a row missing its host, dest or preset would
        # otherwise parse as a row shifted left, and the run would encode to a
        # destination built from a size.
        if len(fields) < 7:
            skipped.append(line[:120])
            continue
        host, dest, preset, src = fields[0], fields[1], fields[2], fields[3]
        try:
            size = int(fields[4])
        except ValueError:
            skipped.append(line[:120])
            continue
        clips.append(Clip(src=src, rel_dir=dest,
                          stem=PurePosixPath(src).stem, size=size,
                          frames=_frames_from(fields[5]),
                          denoise=False, src_host=host,
                          preset=preset.strip()))
    if skipped:
        print(f"[archive-batch] encode manifest: skipped {len(skipped)} "
              f"malformed row(s), first: {skipped[0]!r}", file=sys.stderr)
    return tuple(clips)


def _frames_from(field):
    parts = field.split(",")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def order_clips(clips):
    """SetA before SetB, years ascending, undated folders last, then
    longest clip first inside a folder.

    Longest-first matters because clip length spans seconds to tens of minutes: a
    shortest-first queue leaves a 30-minute clip starting on the slowest
    denoiser after every other one has drained.
    """
    return tuple(sorted(clips, key=_sort_key))


def _sort_key(clip):
    parts = PurePosixPath(clip.src).parts
    top = parts[0] if parts else ""
    second = parts[1] if len(parts) > 1 else ""
    year = int(second) if _YEAR.match(second) else _NO_YEAR
    return (_SET_RANK.get(top, 2), year, clip.rel_dir, -clip.frames, clip.src)
