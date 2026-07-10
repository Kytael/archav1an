# tools/optsig_pref/validate_tnoise.py
import csv, numpy as np
from scipy.stats import spearmanr
from tools.optsig_pref.features import decode_window, tnoise, _blocks

def spatial_noise_robust(frames, block=16, pct=10):
    vals = []
    for f in frames[:6]:
        s = _blocks(f, block).std(axis=1); s = s[s > 1e-6]
        if s.size: vals.append(np.percentile(s, pct))
    return float(np.mean(vals)) if vals else float("nan")

def main():
    rows = [r for r in csv.DictReader(open("tools/optsig_pref/windows.csv")) if r["included"] == "1"]
    tbl, T, S, nan_ct = [], [], [], 0
    for r in rows:
        f = decode_window(r["path"], int(r["start"]), int(r["length"]))
        tn, frac = tnoise(f); sp = spatial_noise_robust(f)
        if np.isnan(tn): nan_ct += 1
        tbl.append((r["stem"], tn, sp, frac))
        T.append(tn); S.append(sp)
    T, S = np.array(T), np.array(S)
    ok = ~np.isnan(T) & ~np.isnan(S)
    rho, _ = spearmanr(T[ok], S[ok])
    lines = [f"# tnoise vs spatial_noise_robust\n",
             f"Spearman rho = {rho:.3f}  (n={ok.sum()} of {len(rows)}; {nan_ct} tnoise=NaN)\n",
             "|clip|tnoise|spatial|static_frac|", "|--|--|--|--|"]
    for s, tn, sp, fr in tbl:
        lines.append(f"|{s}|{tn:.3f}|{sp:.3f}|{fr:.2f}|")
    open("tools/optsig_pref/tnoise_validation.md", "w").write("\n".join(lines) + "\n")
    print(f"Spearman(tnoise, spatial_noise_robust) = {rho:.3f}  | tnoise=NaN on {nan_ct}/{len(rows)} clips")

if __name__ == "__main__":
    main()
