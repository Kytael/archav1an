import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_dispatch():
    spec = importlib.util.spec_from_file_location(
        "svtav1_dispatch", REPO / "tools" / "svtav1-dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_remote_source_is_used_verbatim_and_skips_staging():
    d = _load_dispatch()
    src, needs_staging = d.resolve_remote_src(
        remote_source="/mnt/media/dance/SetA/2003/event-c/MVI_0068.MOV",
        input_file="/tmp/staged/MVI_0068.MOV")
    assert src == "/mnt/media/dance/SetA/2003/event-c/MVI_0068.MOV"
    assert needs_staging is False


def test_without_remote_source_it_stages_into_temp_remote():
    d = _load_dispatch()
    src, needs_staging = d.resolve_remote_src(
        remote_source=None, input_file="/tmp/staged/MVI_0068.MOV")
    assert src == "Temp/_remote/MVI_0068.MOV"
    assert needs_staging is True


def test_a_tilde_root_survives_quoting_for_the_remote_shell():
    """dispatch's own default remote root is ~/archav1an, and the remote half
    reaches it with `cd <root>` through a shell.

    shlex.quote wrapped the whole path, so that became `cd '~/archav1an'` and
    a shell does not expand a quoted tilde: it looked for a directory named
    "~" and the lane died before it started. Proven against the real remote
    shell on 2026-08-24 -- the quoted form fails there, the unquoted one lands
    in /home/<user>/archav1an. Every host that set an absolute --remote-root
    was unaffected, which is why only the default was broken.
    """
    d = _load_dispatch()
    assert d.quote_remote_path("~/archav1an") == "~/archav1an"
    assert d.quote_remote_path("~/archav1an/Temp/_remote") == \
        "~/archav1an/Temp/_remote"
    assert d.quote_remote_path("~") == "~"
    assert d.quote_remote_path("~user/archav1an") == "~user/archav1an"


def test_an_absolute_root_is_quoted_exactly_as_before():
    d = _load_dispatch()
    assert d.quote_remote_path("/home/user/reposetc/archav1an") == \
        "/home/user/reposetc/archav1an"
    # Not a real archive folder: the public-tree scrubber maps every real
    # event name to one without a space, which left the exported copy of
    # this test asserting that a path with no space gets quoted.
    assert d.quote_remote_path("/mnt/media2/two words") == "'/mnt/media2/two words'"


def test_the_tail_of_a_tilde_path_is_still_quoted():
    """Only the tilde head is exempt. A space or a metacharacter further along
    still cannot reach the remote shell, or a checkout path would be an
    injection site on a machine reached with the operator's own key."""
    d = _load_dispatch()
    assert d.quote_remote_path("~/a b/c") == "~/'a b/c'"
    assert d.quote_remote_path("~/x;rm -rf /") == "~/'x;rm -rf /'"
    # Not a tilde path at all: quoted whole, which is the old behaviour.
    assert d.quote_remote_path("~ weird/x") == "'~ weird/x'"


def test_the_remote_command_leaves_path_to_prefer_prefix_bin():
    """PATH and LD_LIBRARY_PATH must move together, so only one place sets them.

    A non-interactive ssh on the Sparks omits /opt/archav1an/bin, so it is
    tempting to set PATH in this command. prefer_prefix_bin() already does it,
    as main()'s first statement, and it moves the loader path with it.
    Verified on gpu4 in this exact shape: shutil.which finds neither vspipe
    nor SvtAv1EncApp before that call and both after it.

    Setting PATH here as well would hardcode /opt/archav1an while
    prefer_prefix_bin honours $VS_PREFIX, so a host with a different prefix
    would run our binaries against the other prefix's libraries -- the
    mismatch that made ffmpeg write a complete output file and then exit 127
    on 2026-08-12, and read as a bad-clip problem for two runs.
    """
    d = _load_dispatch()
    cmd = d.build_remote_shell_cmd(
        remote_root="~/reposetc/ubuntav1an",
        inner="python3 tools/svtav1-dispatch.py --denoise-serve 1.2.3.4:5300")
    assert "PATH=" not in cmd
    # The tilde must still reach the remote shell unquoted, or cd looks for a
    # directory literally named "~" -- the bug 08d557c fixed.
    assert cmd.startswith("cd ~/reposetc/ubuntav1an && ")
    assert "PYTHONUNBUFFERED=1 exec " in cmd


def test_a_staged_copy_is_deleted_after_the_clip(monkeypatch):
    """Nothing cleaned these up, and 69 gate clips left 3.3 GB on gpu2 and
    775 MB on gpu4."""
    d = _load_dispatch()
    calls = []
    monkeypatch.setattr(d.subprocess, "call", lambda cmd: calls.append(cmd))
    d.remove_staged_source("gpuhost", "~/archav1an",
                           "Temp/_remote/MVI_0068.MOV", True)
    assert len(calls) == 1, calls
    assert calls[0][:2] == ["ssh", "gpuhost"]
    assert calls[0][2] == "rm -f ~/archav1an/Temp/_remote/MVI_0068.MOV"


def test_a_source_read_in_place_is_never_deleted(monkeypatch):
    """The whole safety of remove_staged_source. A --remote-source path is the
    operator's archive, read in place, and deleting one destroys an original --
    2.5 TB of footage sits behind that flag."""
    d = _load_dispatch()
    calls = []
    monkeypatch.setattr(d.subprocess, "call", lambda cmd: calls.append(cmd))
    d.remove_staged_source(
        "gpuhost", "~/archav1an",
        "/mnt/media/dance/SetA/2003/event-c/MVI_0068.MOV", False)
    assert calls == [], calls


def test_the_staged_path_is_quoted_for_a_name_with_a_space(monkeypatch):
    """Archive folders carry spaces, and the staged basename inherits them."""
    d = _load_dispatch()
    calls = []
    monkeypatch.setattr(d.subprocess, "call", lambda cmd: calls.append(cmd))
    src, staged = d.resolve_remote_src(None, "/tmp/two words/MVI_1.MOV")
    d.remove_staged_source("gpuhost", "/home/u/archav1an", src, staged)
    assert calls[0][2] == "rm -f /home/u/archav1an/Temp/_remote/MVI_1.MOV"
