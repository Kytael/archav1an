# tools/optsig_pref/select_pilot.py
import csv, numpy as np
from tools.optsig_pref.features import extract

ANCHORS = ["6174", "8656", "0487", "7052"]   # bright-clean, bright-noisy, dark-clean, dark-noisy
K = 8

def main():
    rows = [r for r in csv.DictReader(open("tools/optsig_pref/windows.csv")) if r["included"] == "1"]
    feats = {}
    for r in rows:
        e = extract(r["path"], int(r["start"]), int(r["length"]))
        feats[r["stem"]] = (e["brightness"], e["tnoise"])
    stems = list(feats.keys())
    X = np.array([feats[s] for s in stems], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = {s: (np.array(feats[s]) - mu) / sd for s in stems}
    chosen = [a for a in ANCHORS if a in feats]
    while len(chosen) < K:
        best, bestd = None, -1.0
        for s in stems:
            if s in chosen:
                continue
            d = min(np.linalg.norm(Z[s] - Z[c]) for c in chosen)
            if d > bestd:
                bestd, best = d, s
        chosen.append(best)
    with open("tools/optsig_pref/pilot_clips.txt", "w") as f:
        f.write("\n".join(chosen) + "\n")
    print("pilot clips:", chosen)
    for s in chosen:
        print(f"  {s}: brightness={feats[s][0]:.3f} tnoise={feats[s][1]:.3f}")

if __name__ == "__main__":
    main()
