# tools/optsig_pref/make_threshold_model.py
"""Derive the 2-regime brightness-threshold sigma rule from the user labels."""
import csv, json
import numpy as np

GRID = [round(0.01 * k, 2) for k in range(1, 9)]
LOW_LABEL_CUT = 0.03   # training labels <= this form the low-sigma (bright) regime

def interval_mae(p, lo, hi):
    if p < lo: return lo - p
    if p > hi: return p - hi
    return 0.0

def best_grid_value(los, his):
    return min(GRID, key=lambda g: float(np.mean([interval_mae(g, l, h) for l, h in zip(los, his)])))

def main():
    feats = {r["stem"]: float(r["brightness"]) for r in csv.DictReader(open("tools/optsig_pref/features.csv"))}
    labs = list(csv.DictReader(open("tools/optsig_pref/labels.csv")))
    b = np.array([feats[r["stem"]] for r in labs])
    y = np.array([float(r["sigma_pref"]) for r in labs])
    lo = np.array([float(r["sigma_lo"]) for r in labs])
    hi = np.array([float(r["sigma_hi"]) for r in labs])
    low = y <= LOW_LABEL_CUT
    assert low.sum() >= 2 and (~low).sum() >= 2, "degenerate regimes"
    thr = float((b[low].min() + b[~low].max()) / 2.0)
    assert b[low].min() > b[~low].max(), "regimes not separable by brightness"
    art = dict(kind="brightness-threshold",
               features=["brightness"],
               threshold=round(thr, 6),
               sigma_low=best_grid_value(lo[low], hi[low]),
               sigma_high=best_grid_value(lo[~low], hi[~low]),
               sigma_grid=GRID,
               window=dict(start_fraction=0.40, length=180),
               fit=dict(n=len(labs), n_low=int(low.sum()),
                        target="user-preference-labels-2026-07",
                        note="step rule; linear LOCO gate failed, see tools/optsig_pref/loco_report.md"))
    json.dump(art, open("models/bsvd_optsig_pref_v1.json", "w"), indent=1)
    print(json.dumps(art, indent=1))

if __name__ == "__main__":
    main()
