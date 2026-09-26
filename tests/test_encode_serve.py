import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DISPATCH = REPO / "tools" / "svtav1-dispatch.py"


def _run(args):
    return subprocess.run([sys.executable, str(DISPATCH), *args],
                          capture_output=True, text=True)


def test_encode_serve_and_remote_encode_are_exclusive():
    p = _run(["-i", "x.MOV", "-o", "y.mkv",
              "--encode-serve", "10.0.0.17:5310",
              "--remote-encode", "gpu2"])
    assert p.returncode != 0
    out = p.stdout + p.stderr
    # Not just "exclusive": USAGE carries that word already, under
    # "Denoisers (mutually exclusive)", so a bare substring check passed
    # before the flag even existed.
    assert "--encode-serve and --remote-encode are exclusive" in out


def test_encode_serve_needs_an_output_path():
    p = _run(["--encode-serve", "10.0.0.17:5310"])
    assert p.returncode != 0
    out = p.stdout + p.stderr
    assert "--encode-serve needs -o" in out


def test_encode_serve_rejects_a_bare_port():
    p = _run(["-i", "x.MOV", "-o", "y.mkv", "--encode-serve", "5310"])
    assert p.returncode != 0
    assert "IP:PORT" in p.stdout + p.stderr


def test_remote_encode_needs_an_ip():
    p = _run(["-i", "x.MOV", "-o", "y.mkv", "--remote-encode", "gpu2"])
    assert p.returncode != 0
    assert "--remote-encode-ip" in p.stdout + p.stderr


def test_encode_serve_binds_the_given_address_and_fails_loudly_if_absent():
    # 203.0.113.1 is TEST-NET-3: no host holds it, so the bind must fail. This
    # is the off-LAN guard: gpu2 is a laptop, and a lane that cannot bind its
    # LAN address must refuse rather than listen on an untrusted network.
    p = _run(["-i", "x.MOV", "-o", "y.ivf",
              "--encode-serve", "203.0.113.1:5310"])
    assert p.returncode != 0
    out = p.stdout + p.stderr
    assert "203.0.113.1" in out
    assert "bind" in out.lower()


def test_encode_serve_runs_without_an_input_file():
    """Task 9 sends no -i, because the frames arrive on a socket.

    main() refused any invocation without -i, so the remote command died
    before it bound anything -- a failure that would only have shown up on
    the first live two-host run.

    TEST-NET-3 is used so the bind fails immediately and this stays an
    argument-level test: reaching the bind at all is what proves -i is no
    longer required.
    """
    p = _run(["--encode-serve", "203.0.113.1:5310", "-o", "y.ivf"])
    out = p.stdout + p.stderr
    assert "-i and -o are required" not in out
    assert "cannot bind 203.0.113.1" in out


def test_encode_serve_does_not_need_vspipe(tmp_path):
    """The encode half runs netstream recv | SvtAv1EncApp. No vspipe anywhere.

    Requiring it would have failed every Spark, where vspipe is off the
    non-interactive PATH, on a mode that never calls it.

    The stub PATH holds the three binaries this mode does use, so the run
    reaches the bind and nothing else can answer for the guard under test.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for name in ("SvtAv1EncApp", "ffmpeg", "ffprobe"):
        stub = stub_bin / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    # VS_PREFIX points at nothing so prefer_prefix_bin() finds no prefix bin
    # to prepend; otherwise this host's own /opt/archav1an/bin comes back and
    # supplies the vspipe the test is here to do without.
    env = dict(os.environ, PATH=str(stub_bin), VS_PREFIX=str(tmp_path / "none"))
    env.pop("VSPIPE", None)
    p = subprocess.run([sys.executable, str(DISPATCH),
                        "--encode-serve", "203.0.113.1:5310", "-o", "y.ivf"],
                       capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    assert "vspipe not found" not in out
    assert "cannot bind 203.0.113.1" in out


def test_encode_serve_does_not_probe_the_audio_of_devnull():
    """input_file is os.devnull in this mode, so the shared setup was
    printing a fabricated "Audio: Opus 128k (2ch)" line into the encode
    host's log for a mode that never muxes audio."""
    p = _run(["--encode-serve", "203.0.113.1:5310", "-o", "y.ivf"])
    out = p.stdout + p.stderr
    assert "Audio:" not in out


def test_the_plain_source_branch_routes_to_a_remote_encoder():
    """An encode job: decode here, encode there, denoise nowhere.

    This was refused outright until 2026-09-04, and the refusal was right
    about the code as it stood -- only the denoise branches routed to the
    remote, so the plain source branch encoded HERE while reporting success.
    It also made every encode job a local-encoder job, because --remote-encode
    is what archive-batch passes for every host whose roster entry is not
    "local", so the whole pool answered with one error line.

    Read from the source rather than run: reaching this branch means decoding
    a real clip and opening an ssh, which is an end-to-end test and not this.
    """
    src = DISPATCH.read_text(encoding="utf-8")
    assert "--remote-encode needs a denoise flag" not in src, \
        "the old refusal is back, and every encode job is local again"
    # The plain-source branch: the one that pipes <stem>_src.vpy.
    at = src.index("_src_cmd = [vspipe_exe")
    branch = src[at:src.index("# --- Frame-count verification", at)]
    assert "run_remote_encode(" in branch, "no route to the encode host"
    assert "run_piped(_src_cmd, svt_cmd" in branch, "no local fallback left"


def test_a_port_already_in_use_is_not_diagnosed_as_off_lan():
    """EADDRINUSE is the likeliest bind failure in production: a killed or
    timed-out clip leaves a netstream recv holding the port, the scheduler
    releases the slot, and the next clip draws the same slot and the same
    port. Telling the operator the host does not hold its own LAN address
    sends them to the network while the real holder is a local process."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        p = _run(["--encode-serve", f"127.0.0.1:{port}", "-o", "y.ivf"])
    finally:
        holder.close()
    assert p.returncode != 0
    out = p.stdout + p.stderr
    assert "does not hold that address" not in out
    assert "already in use" in out


def test_denoise_serve_and_remote_encode_are_exclusive():
    """The sink chain gives --denoise-serve priority, so --remote-encode was
    silently ignored and the encode happened wherever the serve target points.
    Every other combination in this file is refused; this one was an
    omission."""
    p = _run(["-i", "x.MOV", "-o", "y.mkv", "--denoise-bsvd",
              "--denoise-serve", "10.0.0.17:5310",
              "--remote-encode", "gpu2",
              "--remote-encode-ip", "10.0.0.17"])
    assert p.returncode != 0
    out = p.stdout + p.stderr
    assert "--denoise-serve and --remote-encode are exclusive" in out
