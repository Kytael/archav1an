"""Targeted edits to denoisers.toml. Never a rewrite.

The roster is a file a person reads and edits in vim. It aligns its '='
signs and carries paragraphs of comments explaining why each lane is shaped
the way it is. A round-trip through a TOML writer would return valid TOML and
destroy all of that, so every function here locates the lines it was asked to
change and leaves every other byte alone.

Nothing is committed unvalidated: the new text goes to a temporary file in the
same directory, roster.load_roster reads it, and only a file that loads is
renamed over the original. os.replace is atomic on the same filesystem, so a
reader -- the scheduler, once per clip -- sees the old file or the new one and
never a half-written one.
"""
import os
import re
import tempfile

from tools.archive_batch.roster import RosterError, load_roster

# A [[denoiser]] block runs to the next table header of ANY kind, not to the
# next [[denoiser]]. The example roster ends with its [[encoder]] entries, so a
# block that ran to the next [[denoiser]] would make the last lane's block
# swallow the encode pool -- and removing that lane would delete it.
_TABLE = re.compile(r"^\s*\[")
_DENOISER = re.compile(r"^\s*\[\[denoiser\]\]\s*$")
_ENCODER = re.compile(r"^\s*\[\[encoder\]\]\s*$")
_NAME = re.compile(r"""^\s*name\s*=\s*["'](?P<v>[^"']*)["']""")


class StaleRoster(Exception):
    """The file changed since the caller read it. The caller must re-read."""


class LaneNotFound(RosterError):
    """No [[denoiser]] block carries that name."""


class EncoderNotFound(RosterError):
    """No [[encoder]] block carries that name."""


def rev(path):
    """An opaque revision token a write must match. None if the file is absent.

    A STRING, and it must stay one. This used to be [st_mtime_ns, st_size], and
    that shape cannot survive a browser: st_mtime_ns is around 1.79e18 while
    Number.MAX_SAFE_INTEGER is 9.007e15, so JSON.parse rounds it. Measured in
    Chrome, the daemon sent 1787015143623006887 and the page read back
    1787015143623007000; the page echoed the rounded number, the comparison in
    _read failed, and every write from the page answered 409. A decimal string
    round-trips through JSON exactly, by construction.

    So this is a token the client stores and echoes, never two numbers it is
    meant to reason about. Do not "tidy" it back into a list of ints.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{st.st_mtime_ns}-{st.st_size}"


def set_enabled(path, name, value, expect_rev=None):
    """Rewrite one lane's `enabled` line. Returns the new rev.

    The type check mirrors set_lp_level's: bool("false") is True in Python,
    so a caller that coerced instead of checked would flip a lane the client
    asked to turn off.
    """
    if not isinstance(value, bool):
        raise RosterError(f"enabled must be true or false, not {value!r}")
    lines = _read(path, expect_rev)
    _flip(lines, _block(lines, name), value)
    return _commit(path, lines)


def set_encoder_enabled(path, name, value, expect_rev=None):
    """Rewrite one ENCODE HOST's `enabled` line. Returns the new rev.

    The same edit as set_enabled on a different table, and deliberately a
    separate function rather than a flag on that one. The two names are in
    separate namespaces: this roster has a denoiser "gpu4" AND an encoder
    "gpu4", and they are different machines' worth of work. One function
    taking a bare name would have to guess which table the caller meant, and
    it would guess wrong on exactly the hosts that do both jobs.

    Turning a host off kills nothing in flight. Lanes stop picking it at the
    next clip and it drains, which is the same contract the lane switch has.
    """
    if not isinstance(value, bool):
        raise RosterError(f"enabled must be true or false, not {value!r}")
    lines = _read(path, expect_rev)
    _flip(lines, _block(lines, name, _ENCODER, EncoderNotFound, "encoder"),
          value)
    return _commit(path, lines)


def _flip(lines, extent, value):
    """Set `enabled` inside one block's extent, in place."""
    start, end = extent
    literal = "true" if value else "false"
    for i in range(start, end):
        # Keep everything after the boolean -- spacing and any inline
        # comment -- verbatim. An operator note like "# gpu3 is on TB4
        # power" must survive a toggle; only the value is this function's
        # business.
        rewritten = _rewrite_value(lines[i], "enabled", literal, r"(?:true|false)")
        if rewritten is not None:
            lines[i] = rewritten
            return
    # A block that never set it was relying on roster.py's default of True.
    lines.insert(start + 1, f"enabled = {literal}\n")


# The writable keys, in the order they are written, matching the shape of
# denoisers.example.toml. The server validates against this tuple.
#
# This list lives in TWO places, not one: static/app.js has its own copy in
# LANE_FIELDS, because the page builds its form before it has ever seen a
# snapshot. They agree today, and a key added to Denoiser has to be added to
# both or the form can never set it.
#
# No "port". The y4m listener moved to the encode host on 2026-08-25, so a port
# on a [[denoiser]] is refused by load_roster -- and this module writes the same
# file the batch reads, so offering one here parked every lane at the next clip
# boundary. A client that still sends it gets the unknown-key refusal.
FIELDS = ("name", "host", "backend", "device", "tiling", "window", "margin",
          "stage_source", "root", "encoders")

# Only keys whose value is meaningful get a line. Writing `window = 0` on an
# untiled lane would trip roster.py's rule that windowing needs tiling, and
# writing `root = ""` on a local one is noise.
#
# Public because model.py applies the same rule when it puts a lane's writable
# fields in the snapshot: a lane that sets no window must not arrive at the edit
# form showing "window 0", which is a value nobody typed and roster.py rejects
# on a tiled lane.
OMIT_WHEN_FALSY = {"window", "margin", "stage_source", "root", "encoders"}

# `encoders` is the routing control: which encode hosts this lane may stream to.
# An EMPTY list is not "no encoders" -- roster.py reads an absent or empty
# allowlist as "any enabled encoder" -- so it belongs in OMIT_WHEN_FALSY above,
# where an empty value removes the key and puts the lane back on that default.
# Writing `encoders = []` instead would say the same thing in a form nobody
# else in this file uses.


def add_lane(path, fields, expect_rev=None):
    """Append a [[denoiser]] block. Always disabled. Returns the new rev."""
    return _add(path, fields, expect_rev)


def add_encoder(path, fields, expect_rev=None):
    """Append an [[encoder]] block. Always disabled. Returns the new rev.

    After the last encoder rather than the last lane, so the file keeps its
    shape: denoisers, then the encode pool. A roster whose sections interleave
    is valid TOML and unreadable to the person who maintains it.
    """
    return _add(path, fields, expect_rev, header=_ENCODER, kind="encoder",
                field_list=ENCODER_FIELDS, omit=ENCODER_OMIT_WHEN_FALSY,
                pad=10)


def _add(path, fields, expect_rev, header=_DENOISER, kind="denoiser",
         field_list=None, omit=None, pad=7):
    lines = _read(path, expect_rev)
    block = _render(fields, kind, field_list, omit, pad)
    blocks = _blocks(lines, header)
    # After the last denoiser block, not at end of file: the roster ends with
    # its encode pool, and a [[denoiser]] appended after that is valid TOML
    # that reads as a file whose sections are out of order.
    at = blocks[-1][2] if blocks else len(lines)
    # Exactly one blank line on each side of the new block, which is how every
    # block already in this file is spaced. _render supplies the one below; the
    # one above is conditional because what sits at `at - 1` depends on the
    # file. In the fixture roster it is already a blank line, and an
    # unconditional leading "\n" -- what this used to do -- made two. In
    # denoisers.example.toml the last lane's extent runs through the paragraph
    # of comments that documents [encode], so `at - 1` is a comment line and a
    # block appended with no blank would be glued to it.
    if at and lines[at - 1].strip():
        block = ["\n"] + block
    lines[at:at] = block
    return _commit(path, lines)


def _literal(value):
    """One TOML value, as this file writes them. Shared by add and update so a
    root edited on the page is escaped exactly as one added on the page."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    # An allowlist. Flat, and always of strings: `encoders` is the only list
    # this file writes, and roster.py reads it as encoder names.
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(str(v)) for v in value) + "]"
    # TOML basic strings take backslash escapes, so both characters have to be
    # escaped or a Windows-shaped root would change meaning.
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render(fields, kind="denoiser", field_list=None, omit=None, pad=7):
    field_list = FIELDS if field_list is None else field_list
    omit = OMIT_WHEN_FALSY if omit is None else omit
    unknown = set(fields) - set(field_list)
    if unknown:
        raise RosterError(f"unknown {kind} key(s): {sorted(unknown)}")
    if not str(fields.get("name", "")).strip():
        raise RosterError(f"a{'n' if kind[0] in 'aeiou' else ''} {kind} needs a name")
    out = [f"[[{kind}]]\n"]
    for key in field_list:
        if key not in fields:
            continue
        value = fields[key]
        if key in omit and not value:
            continue
        out.append(f"{key:<{pad}} = {_literal(value)}\n")
    # §6: adding a lane does not test it, so it arrives switched off and the
    # operator turns it on when they are ready to find out. The same holds for
    # an encode host: a box that has never taken a clip is not one to put in
    # the pool unattended.
    out.append(f"{'enabled':<{pad}} = false\n")
    # A trailing blank line, unconditionally. This block is inserted directly
    # above whatever table header follows it, so without one its
    # `enabled = false` would sit hard against that header. The blank line ABOVE
    # the block is add_lane's business, because only add_lane can see what is
    # already there. This module exists to preserve a file a person reads in
    # vim, so it must not degrade that file's spacing on the way in.
    out.append("\n")
    return out


# The writable keys of an [[encoder]], in the order they are written. Its own
# list, not a slice of FIELDS: the two tables share no key but `name` and
# `host`, and load_roster REFUSES a `backend` written into an [[encoder]] --
# which the operator would only discover after typing it. app.js keeps a copy
# in ENCODER_FIELDS for the same reason LANE_FIELDS is a copy of FIELDS.
#
# No `enabled`: the row's switch owns it, and an edit carrying it would let a
# save undo a toggle made in between. No `slots`-less default either -- a host
# with no slots takes no work, which roster.py refuses outright.
ENCODER_FIELDS = ("name", "host", "root", "stream_ip", "stream_net",
                  "port_base", "slots", "lp_level")

# An encoder on the daemon's own box needs no address and no checkout path, so
# those three are removed rather than written empty. port_base, slots and
# lp_level are NOT here: a zero for any of them is a value roster.py rejects
# with its own message, which is more use than this module silently dropping
# the key and letting the default stand.
ENCODER_OMIT_WHEN_FALSY = {"root", "stream_ip", "stream_net"}

# Added here rather than in Task 1: nothing before this function needs it.
_ENCODE = re.compile(r"^\s*\[encode\]\s*$")

# What set_lp_level answers a pool roster, and what the page shows for it. It
# names the missing FEATURE rather than the missing table, because the operator
# of a pool roster never wrote an [encode] table and cannot act on being told
# there is none.
NO_ENCODE_TABLE = ("this roster has no [encode] table, so there is no single "
                   "pool --lp to set -- each encode host carries its own, on "
                   "its row in Encode hosts")


def has_encode_table(path):
    """Whether set_lp_level has a table to write. False for a pool roster.

    The page asks this rather than guessing from the parsed Roster: a legacy
    [encode] table loads as one encoder named "local", which a pool roster is
    free to be as well. The writer's own condition is the only honest answer to
    "can this control save?", so the control is driven by exactly it.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return any(_ENCODE.match(line) for line in fh)
    except OSError:
        return False


def set_lp_level(path, level, expect_rev=None):
    """Rewrite encode.lp_level. Returns the new rev.

    The range check is roster.py's, reached through _commit. Only the type is
    checked here, because a string would be written as a bare TOML token and
    fail as a parse error rather than as the range message the page shows.
    """
    if isinstance(level, bool) or not isinstance(level, int):
        raise RosterError(f"lp_level must be an integer in [0, 6], not {level!r}")
    lines = _read(path, expect_rev)
    start = None
    for i, line in enumerate(lines):
        if _ENCODE.match(line):
            start = i
            break
    if start is None:
        raise RosterError(NO_ENCODE_TABLE)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _TABLE.match(lines[j]):
            end = j
            break
    for i in range(start, end):
        # Keep everything after the number -- spacing and any inline comment
        # -- verbatim, the same standard set_enabled holds for its line.
        rewritten = _rewrite_value(lines[i], "lp_level", level, r"\S+")
        if rewritten is not None:
            lines[i] = rewritten
            break
    else:
        lines.insert(end, f"lp_level = {level}\n")
    return _commit(path, lines)


# Any value a TOML line can carry, for a rewrite that does not know the old
# type. The quoted alternatives come first so a string holding a space or a
# '#' is consumed whole and its inline comment is still left in the tail.
#
# The bracket alternative is what lets `encoders = ["a", "b"]` be rewritten.
# Without it the trailing \S+ matches only `["a",` and stops at the space, so
# the rest of the old list lands in the tail and is kept -- silently doubling
# half the allowlist. Flat lists only, which is all this file writes.
_ANY_VALUE = r"""(?:"(?:[^"\\]|\\.)*"|'[^']*'|\[[^\]]*\]|\S+)"""


def update_lane(path, name, fields, expect_rev=None):
    """Rewrite the values inside one [[denoiser]] block. Returns the new rev.

    Targeted, like every other write in this module. A key the block already
    sets has only its value replaced, so its spacing and any inline comment
    survive -- an operator's note about why a window is 300 must not be the
    price of correcting a margin. A key the block does not set is inserted in
    FIELDS order. A key sent empty is REMOVED, which is the only way to put a
    lane back on a default: an absent key and a key set to "" are different
    things to roster.py, and the page has no other way to say the first.

    The name is not editable and a request to change it is refused rather
    than silently ignored. The name is the key everything else uses: the
    worker's heartbeat at lanes/<name>.json, dispatch's --temp-tag, the
    scheduler's in-flight bookkeeping and this module's own block lookup. A
    rename mid-run would orphan all four while the lane kept working, and the
    page would show a lane that no longer exists beside one that has never
    run. Remove the lane and add it again.

    `enabled` is not in FIELDS, so it cannot be reached from here: the switch
    on each row owns it, and set_enabled is what it calls. A change made here
    takes effect at the next clip, exactly like a toggle, because the batch
    re-reads the roster before every clip and nothing kills work in flight.
    """
    return _update(path, name, fields, expect_rev,
                   rename_error=(
                       f"a lane cannot be renamed from '{name}' to "
                       f"'{fields.get('name')}': the name keys its heartbeat, "
                       f"its temp directory and the scheduler's bookkeeping. "
                       f"Remove the lane and add it again."))


def update_encoder(path, name, fields, expect_rev=None):
    """Rewrite the values inside one [[encoder]] block. Returns the new rev.

    update_lane's counterpart on the other table, and the same contract: only
    the values move, the block's spacing and comments survive, an empty value
    removes the key, and `enabled` is unreachable because the row's switch owns
    it. A change lands at that host's next clip.

    The name is refused here for a second reason on top of the lane's. Every
    lane's `encoders` allowlist names encoders by name, and roster.py checks
    those names against this table -- so a rename would dangle every allowlist
    that pointed here, and load_roster would refuse the file at commit with a
    message about the LANE rather than about the rename that caused it.
    """
    return _update(path, name, fields, expect_rev,
                   header=_ENCODER, missing=EncoderNotFound, kind="encoder",
                   field_list=ENCODER_FIELDS, omit=ENCODER_OMIT_WHEN_FALSY,
                   pad=10,
                   rename_error=(
                       f"an encode host cannot be renamed from '{name}' to "
                       f"'{fields.get('name')}': every lane's `encoders` "
                       f"allowlist names it, and a rename would leave those "
                       f"pointing at a host that no longer exists. Remove it "
                       f"and add it again."))


def _update(path, name, fields, expect_rev, rename_error,
            header=_DENOISER, missing=LaneNotFound, kind="denoiser",
            field_list=None, omit=None, pad=7):
    field_list = FIELDS if field_list is None else field_list
    omit = OMIT_WHEN_FALSY if omit is None else omit
    unknown = set(fields) - set(field_list)
    if unknown:
        raise RosterError(f"unknown {kind} key(s): {sorted(unknown)}")
    if "name" in fields and str(fields["name"]) != name:
        raise RosterError(rename_error)
    lines = _read(path, expect_rev)
    start, end = _block(lines, name, header, missing, kind)
    # FIELDS order, so several inserts in one edit land in the file's own
    # order rather than in whatever order the page's form iterated.
    for key in field_list:
        if key == "name" or key not in fields:
            continue
        value = fields[key]
        # "" is the page saying "unset this". A real 0 or False is not: those
        # are values roster.py has defaults for, and the omit set is the list
        # of keys where writing one is noise or an outright error.
        drop = value == "" or (key in omit and not value)
        at = _find_key(lines, start, end, key)
        if drop:
            if at is not None:
                del lines[at]
                end -= 1
            continue
        literal = _literal(value)
        if at is not None:
            rewritten = _rewrite_value(lines[at], key, literal, _ANY_VALUE)
            # _find_key matched `key =` and _rewrite_value wants a value after
            # it, so this is a line like `window =` with nothing on the right.
            # That file does not parse anyway, but the failure has to name the
            # line rather than write None into the list and traceback later.
            if rewritten is None:
                raise RosterError(
                    f"{kind} '{name}' has a '{key}' line this cannot "
                    f"rewrite: {lines[at].strip()!r}")
            lines[at] = rewritten
            continue
        lines.insert(_insert_at(lines, start, end, key),
                     f"{key:<{pad}} = {literal}\n")
        end += 1
    return _commit(path, lines)


def _find_key(lines, start, end, key):
    """Index of the line in this block that sets `key`, or None."""
    for i in range(start, end):
        if re.match(rf"^\s*{key}\s*=", lines[i]):
            return i
    return None


def _insert_at(lines, start, end, key):
    """Where a key the block does not set belongs: before the first key that
    comes after it in FIELDS order, and after the last assignment otherwise.

    `enabled` is on the end of the order without being in FIELDS, because
    _render puts it last and a key inserted below it would read as belonging
    to the next block.
    """
    order = list(FIELDS) + ["enabled"]
    after = order[order.index(key) + 1:]
    for i in range(start + 1, end):
        for later in after:
            if re.match(rf"^\s*{later}\s*=", lines[i]):
                return i
    last = start + 1
    for i in range(start + 1, end):
        if re.match(r"^\s*\w+\s*=", lines[i]):
            last = i + 1
    return last


def remove_encoder(path, name, expect_rev=None):
    """Delete one [[encoder]] block. Returns the new rev.

    Refused by the validator at commit if any lane's allowlist still names it:
    roster.py checks those names against this table, so removing a host that a
    lane routes to fails with that lane's message and nothing is written. Clear
    the routing first -- which is what the allowlist control is for.
    """
    return _remove(path, name, expect_rev, header=_ENCODER,
                   missing=EncoderNotFound, kind="encoder")


def remove_lane(path, name, expect_rev=None):
    """Delete a whole [[denoiser]] block. Returns the new rev."""
    return _remove(path, name, expect_rev)


def _remove(path, name, expect_rev, header=_DENOISER, missing=LaneNotFound,
            kind="denoiser"):
    lines = _read(path, expect_rev)
    start, end = _block(lines, name, header, missing, kind)
    del lines[start:end]
    return _commit(path, lines)


def _rewrite_value(line, key, value, value_pattern):
    """Rewrite one `key = <old value>` line and return the new line; `line`
    itself is untouched. Everything after the old value -- spacing, an
    inline comment -- is kept verbatim; only the value between `=` and that
    tail changes. Returns None if `line` does not set `key`, so the caller's
    loop can keep scanning.

    Shared by set_enabled and set_lp_level so a newline or tail edge case is
    fixed once here rather than twice at two call sites that used to carry
    the same regex by hand.
    """
    m = re.match(
        rf"^(?P<indent>\s*){key}(?P<gap>\s*=\s*){value_pattern}(?P<tail>.*)$",
        line)
    if not m:
        return None
    newline = "\n" if line.endswith("\n") else ""
    # The gap is captured, not normalised to " = ". This file aligns its '='
    # signs by hand and a person reads it in vim; rewriting `window  = 300`
    # as `window = 500` breaks that column for every line the page ever
    # touches, which is the degradation this whole module exists to avoid.
    return f"{m.group('indent')}{key}{m.group('gap')}{value}{m.group('tail')}{newline}"


def _read(path, expect_rev):
    current = rev(path)
    if current is None:
        raise RosterError(f"roster not found: {path}")
    # Plain string equality: rev() returns an opaque token, so there is nothing
    # to normalise before comparing. expect_rev of None still means "no check",
    # for the direct Python callers; the router requires one on every request.
    if expect_rev is not None and expect_rev != current:
        raise StaleRoster(
            "the roster changed since this page loaded it; reload and retry")
    with open(path, encoding="utf-8") as fh:
        return fh.readlines()


def _blocks(lines, header=_DENOISER):
    """[(name, start, end)] for each block of this kind. end is exclusive."""
    out = []
    for i, line in enumerate(lines):
        if not header.match(line):
            continue
        end = len(lines)
        for j in range(i + 1, len(lines)):
            if _TABLE.match(lines[j]):
                end = j
                break
        name = None
        for j in range(i + 1, end):
            m = _NAME.match(lines[j])
            if m:
                name = m.group("v")
                break
        out.append((name, i, end))
    return out


def _block(lines, name, header=_DENOISER, missing=LaneNotFound,
           kind="denoiser"):
    for bname, start, end in _blocks(lines, header):
        if bname == name:
            return start, end
    raise missing(f"no {kind} named '{name}' in the roster")


def _commit(path, lines):
    """Validate on a temporary copy, then rename it over the original."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".denoisers-", suffix=".toml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        # mkstemp creates the file at 0600. os.replace carries that mode onto
        # the roster, so without this an operator's first click through the
        # page would silently drop denoisers.toml from 644 to 600.
        os.chmod(tmp, os.stat(path).st_mode)
        # The whole point. An edit that would stop the run is refused here,
        # with roster.py's own message, rather than committed and discovered
        # by a scheduler that parks every lane in silence.
        load_roster(tmp)
        os.replace(tmp, path)
    except BaseException:
        # A refused edit must leave nothing behind: this directory is
        # .archive-run, which the operator reads by hand.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return rev(path)
