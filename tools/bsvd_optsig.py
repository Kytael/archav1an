"""BSVD optimal-σ pre-pass (preference-calibrated threshold rule, v1 2026-07).

Replaces the V3 NV-target linear predictor (sigma_estimator CNN + torch): the σ
policy is now a brightness step rule fit to the user's own preference labels
(see tools/optsig_pref/loco_report.md). Pure PyAV+numpy — no torch, no CNN —
so the pre-pass runs identically on gpu1 (CUDA) and encoder-host (MIGraphX).

The analysis window MUST match the labeling protocol: 180 frames starting at
40% of the clip (sequential decode; input seeking is keyframe-inaccurate on
intra codecs).
"""
import json
import subprocess

import av
import numpy as np


def _probe_nframes(path):
    out = subprocess.check_output(
        ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path], text=True)
    return int(out.strip())


def decode_window(path, start, length):
    c = av.open(path)
    out = []
    for n, fr in enumerate(c.decode(video=0)):
        if n < start:
            continue
        if len(out) >= length:
            break
        out.append(fr.to_ndarray(format="gray"))
    c.close()
    if not out:
        raise RuntimeError(f"no frames decoded from {path} (start={start})")
    return np.stack(out).astype(np.float32)


def brightness(frames):
    return float(frames.mean() / 255.0)


def compute_sigma_for_video(video_path, *, model_json, start_fraction=None,
                            length=None, _nframes=None, verbose=True):
    """Pre-pass: returns the σ the threshold rule assigns to this clip."""
    model = json.load(open(model_json))
    if model.get("kind") != "brightness-threshold":
        raise ValueError(f"unsupported optsig model kind: {model.get('kind')!r}")
    win = model.get("window", {})
    sf = start_fraction if start_fraction is not None else win.get("start_fraction", 0.40)
    ln = length if length is not None else win.get("length", 180)
    n = (_nframes or _probe_nframes)(video_path)
    start = max(0, min(round(sf * n), max(n - ln, 0)))
    frames = decode_window(video_path, start, ln)
    b = brightness(frames)
    sigma = model["sigma_low"] if b > model["threshold"] else model["sigma_high"]
    if verbose:
        import sys
        print(f"[bsvd-optsig] σ={sigma:.3f} (brightness={b:.4f} thr={model['threshold']:.4f}, "
              f"window {start}+{ln} of {n})", file=sys.stderr)
    return float(sigma)
