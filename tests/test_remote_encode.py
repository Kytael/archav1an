import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "dispatch", REPO / "tools" / "svtav1-dispatch.py")
dispatch = importlib.util.module_from_spec(spec)


def _load():
    # The module runs main() only under __main__, so importing is safe.
    spec.loader.exec_module(dispatch)
    return dispatch


def test_remote_encode_cmd_puts_the_tilde_outside_the_quotes():
    d = _load()
    cmd = d.build_encode_serve_cmd(
        remote_python="python3", remote_root="~/archav1an",
        bind="10.0.0.17:5310", ivf="Temp/x/x.ivf", forward_args=["--lp", "6"])
    assert cmd.startswith("cd ~/archav1an && ")
    assert "'~/archav1an'" not in cmd


def test_remote_encode_cmd_quotes_an_absolute_root():
    d = _load()
    cmd = d.build_encode_serve_cmd(
        remote_python="python3", remote_root="/home/user/reposetc/archav1an",
        bind="10.0.0.17:5310", ivf="Temp/x/x.ivf", forward_args=[])
    assert "cd /home/user/reposetc/archav1an && " in cmd


def test_remote_encode_cmd_carries_the_bind_and_the_output():
    d = _load()
    cmd = d.build_encode_serve_cmd(
        remote_python="python3", remote_root="~/archav1an",
        bind="10.0.0.17:5310", ivf="Temp/x/x.ivf", forward_args=["--lp", "6"])
    assert "--encode-serve 10.0.0.17:5310" in cmd
    assert "-o Temp/x/x.ivf" in cmd
    assert "--lp 6" in cmd


# --- Fake process objects for the two ssh-owning orchestrators ---------------
#
# Nothing below may reach a real ssh, rsync or socket: subprocess.Popen,
# subprocess.run and subprocess.check_call are all replaced, and the two that
# must never fire raise instead of returning.

class _FakeStdin:
    def __init__(self):
        self.written = b""

    def write(self, data):
        self.written += data

    def close(self):
        pass


class _FakeProc:
    """Records how it was waited on, so an abandoned child is visible."""

    def __init__(self, returncode=0, wedged=False):
        self.returncode = returncode
        self.stdin = _FakeStdin()
        self.waits = []
        self.terminated = 0
        self._wedged = wedged

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self._wedged and timeout is not None:
            raise subprocess.TimeoutExpired("ssh", timeout)
        return self.returncode

    def terminate(self):
        self.terminated += 1


def _fence(monkeypatch, d, procs, calls):
    """Hand out `procs` in order and forbid every other subprocess entry point.

    subprocess.call is the one exception: the remote `rm -f` goes through it,
    and the tests below assert on it.
    """
    queue = list(procs)

    def fake_popen(cmd, **kw):
        assert queue, f"unexpected Popen: {cmd}"
        return queue.pop(0)

    def forbidden(cmd, *a, **kw):
        raise AssertionError(f"this test must not run: {cmd}")

    def fake_call(cmd, *a, **kw):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(d.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(d.subprocess, "run", forbidden)
    monkeypatch.setattr(d.subprocess, "check_call", forbidden)
    monkeypatch.setattr(d.subprocess, "call", fake_call)


def test_a_dead_local_source_does_not_abandon_the_encode_ssh(tmp_path, monkeypatch):
    """A BSVD OOM kills the local vspipe mid-clip. run_remote_denoise waits the
    remote out before reading its log; this path did not, so the tail was read
    while the remote still held the fd and the ssh was left running."""
    d = _load()
    ssh = _FakeProc()
    calls = []
    _fence(monkeypatch, d, [ssh], calls)

    def boom(*a, **kw):
        raise SystemExit(1)

    monkeypatch.setattr(d, "run_piped", boom)
    with pytest.raises(SystemExit):
        d.run_remote_encode("gpu2", "~/archav1an", "python3",
                            "10.0.0.17:5310", str(tmp_path / "out.ivf"),
                            [], ["vspipe"], str(tmp_path), "MVI_1463")
    assert ssh.waits == [30]


def test_a_dead_local_source_removes_the_short_remote_ivf(tmp_path, monkeypatch):
    """The rm was on the success path only, so every failed retry left a short
    IVF on the encode host."""
    d = _load()
    calls = []
    _fence(monkeypatch, d, [_FakeProc()], calls)

    def boom(*a, **kw):
        raise SystemExit(1)

    monkeypatch.setattr(d, "run_piped", boom)
    with pytest.raises(SystemExit):
        d.run_remote_encode("gpu2", "~/archav1an", "python3",
                            "10.0.0.17:5310", str(tmp_path / "out.ivf"),
                            [], ["vspipe"], str(tmp_path), "MVI_1463")
    assert any("rm -f" in " ".join(c) for c in calls)


def test_a_failed_split_encode_removes_the_short_remote_ivf(tmp_path, monkeypatch):
    """Case 4 exits on a nonzero encode status before it reaches its own rm."""
    d = _load()
    calls = []
    _fence(monkeypatch, d, [_FakeProc(returncode=3), _FakeProc()], calls)
    with pytest.raises(SystemExit):
        d.run_remote_encode_split(
            denoise_target="gpu1", denoise_root="~/archav1an",
            encode_target="gpu2", encode_root="~/archav1an",
            remote_python="python3", bind="10.0.0.17:5310",
            ivf_path=str(tmp_path / "out.ivf"), input_file="/src/MVI_1463.MOV",
            denoise_forward=[], encode_forward=[], temp_dir=str(tmp_path),
            stem="MVI_1463", remote_source="/src/MVI_1463.MOV")
    assert any("rm -f" in " ".join(c) for c in calls)


def test_the_split_denoise_wait_is_bounded(tmp_path, monkeypatch):
    """The only unbounded wait in the file. A denoise half that never makes
    progress held the encoder slot until archive-batch's dispatch_timeout,
    which is 24 h when the clip's frame count is unknown."""
    d = _load()
    dn, enc = _FakeProc(), _FakeProc()
    _fence(monkeypatch, d, [enc, dn], [])
    monkeypatch.setattr(d.subprocess, "check_call", lambda *a, **kw: 0)
    d.run_remote_encode_split(
        denoise_target="gpu1", denoise_root="~/archav1an",
        encode_target="gpu2", encode_root="~/archav1an",
        remote_python="python3", bind="10.0.0.17:5310",
        ivf_path=str(tmp_path / "out.ivf"), input_file="/src/MVI_1463.MOV",
        denoise_forward=[], encode_forward=[], temp_dir=str(tmp_path),
        stem="MVI_1463", remote_source="/src/MVI_1463.MOV")
    assert dn.waits and dn.waits[0] is not None


def test_a_wedged_split_denoise_is_killed_and_refused(tmp_path, monkeypatch):
    d = _load()
    dn, enc = _FakeProc(returncode=-15, wedged=True), _FakeProc()
    calls = []
    _fence(monkeypatch, d, [enc, dn], calls)
    with pytest.raises(SystemExit):
        d.run_remote_encode_split(
            denoise_target="gpu1", denoise_root="~/archav1an",
            encode_target="gpu2", encode_root="~/archav1an",
            remote_python="python3", bind="10.0.0.17:5310",
            ivf_path=str(tmp_path / "out.ivf"), input_file="/src/MVI_1463.MOV",
            denoise_forward=[], encode_forward=[], temp_dir=str(tmp_path),
            stem="MVI_1463", remote_source="/src/MVI_1463.MOV")
    assert dn.terminated == 1


def test_a_split_encode_timeout_says_timeout_not_exited_minus_one(tmp_path,
                                                                 monkeypatch,
                                                                 capsys):
    """enc_rc = -1 sent the reader looking for a crash that never happened.
    run_remote_encode prints a timeout-specific line for the same condition."""
    d = _load()
    dn, enc = _FakeProc(), _FakeProc(wedged=True)
    _fence(monkeypatch, d, [enc, dn], [])
    with pytest.raises(SystemExit):
        d.run_remote_encode_split(
            denoise_target="gpu1", denoise_root="~/archav1an",
            encode_target="gpu2", encode_root="~/archav1an",
            remote_python="python3", bind="10.0.0.17:5310",
            ivf_path=str(tmp_path / "out.ivf"), input_file="/src/MVI_1463.MOV",
            denoise_forward=[], encode_forward=[], temp_dir=str(tmp_path),
            stem="MVI_1463", remote_source="/src/MVI_1463.MOV")
    out = capsys.readouterr().out
    assert "did not finish within" in out
    assert "exited -1" not in out


def test_an_unset_quality_is_not_forwarded_as_the_string_none():
    """--quality and --speed default to None. Locally `if quality:` skips the
    flag; the forwarded list stringified it, so the encode half received the
    literal "None", found it truthy, and built `--crf None`. That fails on the
    encode host with an SVT parse error, far from the CLI that caused it."""
    d = _load()
    got = d.build_encode_forward(quality=None, speed=None, lp="16",
                                 photon_noise=None, encoder_params="",
                                 temp_tag=None)
    assert "None" not in got
    assert "--quality" not in got
    assert "--speed" not in got
    assert got == ["--lp", "16"]


def test_a_set_quality_and_speed_still_travel():
    d = _load()
    got = d.build_encode_forward(quality="30", speed="4", lp="6",
                                 photon_noise="12", encoder_params="--tune 2",
                                 temp_tag="slot3")
    assert got == ["--quality", "30", "--speed", "4", "--lp", "6",
                   "--photon-noise", "12", "--encoder-params", "--tune 2",
                   "--temp-tag", "slot3"]
