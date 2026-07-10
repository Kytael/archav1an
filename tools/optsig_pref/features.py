# tools/optsig_pref/features.py — PURE NUMPY, NO TORCH
import av, numpy as np

def decode_window(path, start, length):
    c = av.open(path); out = []
    for n, fr in enumerate(c.decode(video=0)):
        if n < start:
            continue
        if len(out) >= length:
            break
        out.append(fr.to_ndarray(format="gray"))
    c.close()
    return np.stack(out).astype(np.float32)

def brightness(frames):
    return float(frames.mean() / 255.0)

def _blocks(x, block):
    H, W = x.shape; hb, wb = H // block, W // block
    return (x[:hb*block, :wb*block].reshape(hb, block, wb, block)
              .transpose(0, 2, 1, 3).reshape(hb*wb, block*block))

def tnoise(frames, block=16, motion_frac_min=0.05, motion_cut=2.0):
    """8th-percentile of per-16x16-block temporal std of consecutive-frame diffs,
    / √2 (low tail selects quiet/static blocks). Returns (value|nan, static_frac);
    NaN when static_frac < motion_frac_min or <2 frames."""
    diffs = frames[1:] - frames[:-1]
    if len(diffs) == 0:
        return float("nan"), 0.0
    stds, static, total = [], 0, 0
    for d in diffs:
        b = _blocks(d, block)
        bmean = np.abs(b.mean(axis=1)); bstd = b.std(axis=1)
        stds.append(bstd); total += b.shape[0]
        static += int((bmean < motion_cut).sum())
    allstd = np.concatenate(stds)
    frac = static / max(total, 1)
    if frac < motion_frac_min:
        return float("nan"), frac
    return float(np.percentile(allstd, 8) / np.sqrt(2.0)), frac

def _haar_hh_mad(g):
    H, W = g.shape; H -= H % 2; W -= W % 2; g = g[:H, :W]
    hh = (g[0::2,0::2] - g[0::2,1::2] - g[1::2,0::2] + g[1::2,1::2]) / 2.0
    return float(np.median(np.abs(hh - np.median(hh))) / 0.6745)

def _box_down(g, k):
    H, W = g.shape; H -= H % k; W -= W % k
    return g[:H,:W].reshape(H//k, k, W//k, k).mean(axis=(1,3))

def spatial_std_flat(frame, block=16, pct=5):
    return float(np.percentile(_blocks(frame, block).std(axis=1), pct))

def mad4(frames):
    return float(np.mean([_haar_hh_mad(_box_down(f, 4)) for f in frames[:6]]))

def extract(path, start, length):
    f = decode_window(path, start, length)
    tn, frac = tnoise(f)
    return dict(brightness=brightness(f), tnoise=tn, tnoise_static_frac=frac,
                spatial_std5=float(np.mean([spatial_std_flat(x) for x in f[:6]])),
                mad4=mad4(f))
