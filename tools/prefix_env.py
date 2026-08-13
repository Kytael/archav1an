"""Put the hand-built stack at $VS_PREFIX ahead of whatever else is on PATH.

Every tool here finds its binaries with shutil.which, and nothing on any of
the hosts has the prefix on PATH. VS_PREFIX arrived in May 2026 and installs
moved to /opt/archav1an, so which() has been resolving to whatever pre-existed.
On encoder-host in August 2026 that still meant three March builds winning over
binaries setup.sh had rebuilt months later:

    vspipe          /usr/local/bin/vspipe          2026-03-27
    av1an           /usr/local/bin/av1an           2026-03-27
    SvtAv1EncApp    /usr/local/bin/SvtAv1EncApp    2026-03-27

The rule is that anything we compile is the default. The exception is narrow:
an external consumer that hard-links a soname we do not build. TensorRT is the
live example -- the onnxruntime provider demands libnvinfer.so.10 and we have
no say -- and those get pinned explicitly rather than papered over here.

Callers that already source activate-venv.sh are unaffected; this only re-adds
entries that are already first there.
"""

import os


def prefer_prefix_bin():
    """Prepend $VS_PREFIX/bin to PATH and append $VS_PREFIX/lib to the loader.

    PATH and LD_LIBRARY_PATH move together or not at all. Taking the prefix
    binaries without the prefix libraries gives them the SYSTEM libavcodec.so.63,
    which carries the same soname but an older ABI:

        /usr/lib/libavcodec.so.63             63.1.100   (distro)
        /opt/archav1an/lib/libavcodec.so.63   63.7.100   (ours)

    The binary starts, so the mismatch is invisible until a lazily-bound symbol
    only the newer library defines is called -- on 2026-08-12, av_bsf_graph_free
    during bitstream-filter teardown, which made ffmpeg write a complete output
    file and then exit 127. It hit only sources whose structure builds a bsf
    graph (the S2S clips carry a tmcd track), so it read as a bad-clip problem
    for two runs. Taking the libraries without the binaries is the same bug
    mirrored, which is why neither half is optional.

    Safe to call more than once, and safe to call from a module that another
    tool imports: both halves are idempotent.
    """
    prefix = os.environ.get("VS_PREFIX", "/opt/archav1an")
    prefix_bin = os.path.join(prefix, "bin")
    if not os.path.isdir(prefix_bin):
        return

    # Appended, not prepended: the MIGraphX lane hands us its own
    # LD_LIBRARY_PATH and must keep first claim. Appending is still enough,
    # because every LD_LIBRARY_PATH entry is searched before /usr/lib.
    prefix_lib = os.path.join(prefix, "lib")
    if os.path.isdir(prefix_lib):
        ld_parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
        if prefix_lib not in ld_parts:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts + [prefix_lib])

    parts = os.environ.get("PATH", "").split(os.pathsep)
    if parts and parts[0] == prefix_bin:
        return
    os.environ["PATH"] = os.pathsep.join(
        [prefix_bin] + [p for p in parts if p != prefix_bin])
