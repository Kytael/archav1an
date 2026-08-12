#!/usr/bin/env python3
"""Gate 2 (spec 7.1): does windowing change the picture?

Runs the same clip through the same engine twice -- once with a window large
enough to hold it (which IS the whole-clip tiled run) and once windowed -- and
compares the two frame by frame.

The question is not only "how big is the error" but "where is it". A bounded
receptive field predicts error spread evenly and sitting at fp16 epsilon. A
margin that is too small predicts error clustered at the window boundaries.
This reports both, so a pass is evidence and not a hope.

    python tools/gate2_windowing.py SOURCE.MOV --frames 900 --window 300 --margin 32
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))


# One definition of the source, used twice: exec'd here to build the clip the
# filter wraps, and written to disk for the reader subprocess to run. They must
# be the same clip, and the cheapest way to guarantee that is to have one text.
SOURCE_SCRIPT = '''\
import os as _os
import vapoursynth as vs
core = vs.core
for _d in ("/opt/archav1an/lib/vapoursynth", "/usr/lib/vapoursynth"):
    for _so in ("libffms2.so", "libvszip.so"):
        _p = _os.path.join(_d, _so)
        if _os.path.exists(_p):
            try:
                core.std.LoadPlugin(_p)
            except vs.Error:
                pass
from vstools import initialize_clip
clip = core.ffms2.Source(source={path!r})
clip = initialize_clip(clip)
if {frames!r}:
    clip = clip[:{frames!r}]
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
clip.set_output(0)
'''


def load_source(path, frames, script_path):
    text = SOURCE_SCRIPT.format(path=os.path.abspath(path), frames=frames)
    with open(script_path, "w") as fh:
        fh.write(text)
    ns = {}
    exec(compile(text, script_path, "exec"), ns)
    return ns["clip"]


def render(clip):
    """Render to fp16. float32 cost 22 GB per clip at 900x1080p, and both
    clips are held at once; the filter's own buffer is fp16 anyway, so this
    loses nothing that the comparison could have seen."""
    import numpy as np
    out = np.empty((clip.num_frames, 3, clip.height, clip.width), dtype=np.float16)
    for i, f in enumerate(clip.frames()):
        for c in range(3):
            out[i, c] = np.asarray(f[c])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--onnx", default=os.path.join(
        REPO, "models", "bsvd_realpair_ep14_stateful_v2_dyn_fp16.onnx"))
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--margin", type=int, default=32)
    # Same three forms dispatch accepts, read by the same parser: "auto",
    # "HxW", or a bare square size.
    ap.add_argument("--tile", default="576")
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    # device 0, not 1: CUDA enumerates only the NVIDIA card, so the 2070S is
    # ordinal 0 and device 1 fails with "invalid device ordinal".
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ep", default="TRT")
    ap.add_argument("--source-script", default=os.path.join(
        REPO, "Temp", "gate2_source.vpy"))
    # dispatch puts $VS_PREFIX/bin first on PATH before it spawns anything;
    # this harness is standalone, so it names the same binary itself rather
    # than inheriting whichever vspipe PATH happens to resolve.
    ap.add_argument("--vspipe", default=os.path.join(
        os.environ.get("VS_PREFIX", "/opt/archav1an"), "bin", "vspipe"))
    args = ap.parse_args()

    import numpy as np

    from bsvd_windowed import build_bsvd_windowed_tiled, parse_tile_arg

    tile = parse_tile_arg(args.tile)

    os.makedirs(os.path.dirname(os.path.abspath(args.source_script)), exist_ok=True)
    src = load_source(args.source, args.frames, args.source_script)

    common = dict(onnx_path=args.onnx, sigma=args.sigma, ep=args.ep,
                  device_id=args.device, tile=tile, overlap=args.overlap,
                  margin=args.margin, source_script=args.source_script,
                  vspipe=args.vspipe)

    n = src.num_frames
    print(f"[gate2] {n} frames, tile {tile}, margin {args.margin}, "
          f"window {args.window} vs whole-clip", flush=True)

    print("[gate2] rendering whole-clip (window >= n)...", flush=True)
    whole = render(build_bsvd_windowed_tiled(src, window=n + 1, **common))
    print("[gate2] rendering windowed...", flush=True)
    win = render(build_bsvd_windowed_tiled(src, window=args.window, **common))

    diff = np.abs(whole.astype(np.float32) - win.astype(np.float32))
    per_frame = diff.reshape(n, -1).max(axis=1)
    identical = int((per_frame == 0).sum())
    print(f"\n[gate2] bit-identical frames: {identical}/{n}")
    print(f"[gate2] max abs diff : {diff.max():.3e}")
    print(f"[gate2] mean abs diff: {diff.mean():.3e}")
    print(f"[gate2] fp16 epsilon : {np.finfo(np.float16).eps:.3e}")

    # Where is the error? A too-small margin puts it at the window joins.
    bounds = [i for i in range(args.window, n, args.window)]
    if bounds:
        near = np.zeros(n, dtype=bool)
        for b in bounds:
            near[max(0, b - 4):min(n, b + 4)] = True
        seam, body = per_frame[near], per_frame[~near]
        print(f"\n[gate2] frames near a window join: {near.sum()} "
              f"(joins at {bounds[:6]}{'...' if len(bounds) > 6 else ''})")
        print(f"[gate2] max diff at joins   : {seam.max():.3e}")
        print(f"[gate2] max diff elsewhere  : {body.max() if body.size else 0:.3e}")
        # A ratio needs a non-zero denominator to mean anything. Printing "inf"
        # for the perfect case reads as the worst case against the hint below,
        # so say which case it is in words instead.
        if body.size and body.max() > 0:
            print(f"[gate2] join/body ratio     : {seam.max() / body.max():.2f}"
                  f"   (>> 1 means a real seam; ~1 means no seam)")
        elif seam.max() > 0:
            print("[gate2] join/body ratio     : SEAM ONLY -- the joins differ and "
                  "everything else is exactly zero. The margin is too small.")
        else:
            print("[gate2] join/body ratio     : no difference anywhere -- joins "
                  "and body are both exactly zero.")

    worst = int(per_frame.argmax())
    print(f"\n[gate2] worst frame {worst} (window join at "
          f"{args.window * round(worst / args.window)}), diff {per_frame[worst]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
