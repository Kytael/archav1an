import re
import socket
import subprocess
import sys
import threading
from pathlib import Path

NETSTREAM = Path(__file__).resolve().parent.parent / "tools" / "netstream.py"

# W4 H2 4:2:0 10-bit: luma 8 samples + chroma 4 = 12, two bytes each = 24, plus
# the six bytes of "FRAME\n". Small enough that the arithmetic is checkable by
# eye, and the same shape vspipe emits.
HEADER = b"YUV4MPEG2 W4 H2 F30:1 Ip A1:1 C420p10\n"
FRAME = b"FRAME\n" + b"\x00" * 24


def _serve(payload, extra_args):
    """Run `netstream recv` against a client that sends `payload`.

    Returns its stderr. recv binds an explicit port, so we pick a free one by
    binding and releasing it first.
    """
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    proc = subprocess.Popen(
        [sys.executable, str(NETSTREAM), "recv", "--port", str(port)] + extra_args,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def client():
        for _ in range(40):
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                threading.Event().wait(0.05)
        else:
            return
        s.sendall(payload)
        s.close()

    t = threading.Thread(target=client)
    t.start()
    err = proc.communicate(timeout=60)[1].decode()
    t.join(5)
    return err


def test_recv_counts_frames_from_the_y4m_header_alone():
    """No geometry is passed in. dispatch does not know the dimensions -- it
    has no width or height anywhere -- but the stream announces them."""
    err = _serve(HEADER + FRAME * 7, ["--progress", "--progress-interval", "0"])
    counts = [int(m) for m in re.findall(r"Frame:\s*(\d+)", err)]
    assert counts, f"no Frame: line in stderr: {err}"
    assert max(counts) == 7


def test_eight_bit_frames_are_half_the_size():
    """C420mpeg2 is one byte a sample, so the same byte count is twice the
    frames. Getting this wrong halves or doubles every remote lane's rate."""
    header = b"YUV4MPEG2 W4 H2 F30:1 Ip A1:1 C420mpeg2\n"
    frame = b"FRAME\n" + b"\x00" * 12
    err = _serve(header + frame * 5, ["--progress", "--progress-interval", "0"])
    assert max(int(m) for m in re.findall(r"Frame:\s*(\d+)", err)) == 5


def test_recv_says_nothing_new_without_the_flag():
    """Opt-in, so the existing pipeline is byte-for-byte unchanged for anyone
    not asking for progress."""
    err = _serve(HEADER + FRAME * 3, [])
    assert "Frame:" not in err
    assert "received" in err, "the closing summary must still be there"


def test_an_unparseable_header_disables_counting_rather_than_failing():
    """The stream must reach the encoder even if this counter cannot work.
    A dashboard is never allowed to break the encode."""
    err = _serve(b"not-a-y4m-stream-at-all\n" + b"\x00" * 100,
                 ["--progress", "--progress-interval", "0"])
    assert "Frame:" not in err
    assert "received" in err
