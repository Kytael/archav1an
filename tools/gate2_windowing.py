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


def load_source(path, frames):
    import vapoursynth as vs
    core = vs.core
    for d in ("/opt/archav1an/lib/vapoursynth", "/usr/lib/vapoursynth"):
        for so in ("libffms2.so", "libvszip.so"):
            p = os.path.join(d, so)
            if os.path.exists(p):
                try:
                    core.std.LoadPlugin(p)
                except vs.Error:
                    pass
    from vstools import initialize_clip
    clip = core.ffms2.Source(source=path)
    clip = initialize_clip(clip)
    if frames:
        clip = clip[:frames]
    return core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")


def render(clip):
    import numpy as np
    out = np.empty((clip.num_frames, 3, clip.height, clip.width), dtype=np.float32)
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
    ap.add_argument("--tile", type=int, default=576)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--ep", default="TRT")
    args = ap.parse_args()

    import numpy as np

    from bsvd_windowed import build_bsvd_windowed_tiled

    common = dict(onnx_path=args.onnx, sigma=args.sigma, ep=args.ep,
                  device_id=args.device, tile=args.tile, overlap=args.overlap,
                  margin=args.margin)

    src = load_source(args.source, args.frames)
    n = src.num_frames
    print(f"[gate2] {n} frames, tile {args.tile}, margin {args.margin}, "
          f"window {args.window} vs whole-clip", flush=True)

    print("[gate2] rendering whole-clip (window >= n)...", flush=True)
    whole = render(build_bsvd_windowed_tiled(src, window=n + 1, **common))
    print("[gate2] rendering windowed...", flush=True)
    win = render(build_bsvd_windowed_tiled(src, window=args.window, **common))

    diff = np.abs(whole - win)
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
        ratio = (seam.max() / body.max()) if body.size and body.max() > 0 else float("inf")
        print(f"[gate2] join/body ratio     : {ratio:.2f}"
              f"   (>> 1 means a real seam; ~1 means no seam)")

    worst = int(per_frame.argmax())
    print(f"\n[gate2] worst frame {worst} (window join at "
          f"{args.window * round(worst / args.window)}), diff {per_frame[worst]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
