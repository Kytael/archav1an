# tools/optsig_pref/fit.py
"""Sign-constrained linear sigma predictor: fit + leave-one-clip-out validation."""
import csv, json
import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr, binomtest

GRID = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
# feature order is fixed: index 0 = brightness (weight <= 0), index 1 = tnoise (weight >= 0)
SIGN_LO = {0: -np.inf, 1: 0.0}
SIGN_HI = {0: 0.0, 1: np.inf}

def best_constant(y, grid=GRID):
    return float(grid[np.argmin([np.mean(np.abs(y - g)) for g in grid])])

def interval_mae(pred, lo, hi):
    if pred < lo: return float(lo - pred)
    if pred > hi: return float(pred - hi)
    return 0.0

def fit_fold(X, y):
    """Sign-constrained least squares on standardized features. X:(n,k), y:(n,)."""
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    n, k = Z.shape
    A = np.column_stack([Z, np.ones(n)])
    lo = np.array([SIGN_LO[i] for i in range(k)] + [-np.inf])
    hi = np.array([SIGN_HI[i] for i in range(k)] + [np.inf])
    r = lsq_linear(A, y, bounds=(lo, hi))
    w = r.x
    bound = [abs(w[i]) < 1e-8 for i in range(k)]
    return dict(w=w, mu=mu, sd=sd, bound=bound)

def predict(m, x):
    z = (np.asarray(x, float) - m["mu"]) / m["sd"]
    raw = float(np.dot(m["w"][:-1], z) + m["w"][-1])
    return float(np.clip(raw, GRID[0], GRID[-1]))

def loco(F, y, lo, hi):
    """LOCO for one feature set. F:(n,k). Returns per-clip preds + summary stats."""
    n = len(y)
    pm = np.zeros(n); pc = np.zeros(n)
    binds = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        m = fit_fold(F[tr], y[tr])
        pm[i] = predict(m, F[i]); binds.append(m["bound"])
        pc[i] = best_constant(y[tr])
    em = np.array([interval_mae(pm[i], lo[i], hi[i]) for i in range(n)])
    ec = np.array([interval_mae(pc[i], lo[i], hi[i]) for i in range(n)])
    e07 = np.array([interval_mae(0.07, lo[i], hi[i]) for i in range(n)])
    wins = int(np.sum(em < ec)); losses = int(np.sum(em > ec)); ties = n - wins - losses
    sign_p = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) if wins + losses else 1.0
    rho, _ = spearmanr(pm, y)
    binds = np.array(binds, dtype=float)
    return dict(pred=pm, pred_const=pc,
                mae=float(em.mean()), mae_bestconst=float(ec.mean()), mae_070=float(e07.mean()),
                wins=wins, losses=losses, ties=ties, sign_p=sign_p,
                spearman=float(rho), bound_frac=binds.mean(0).tolist())

def main():
    feats = {r["stem"]: r for r in csv.DictReader(open("tools/optsig_pref/features.csv"))}
    labs = list(csv.DictReader(open("tools/optsig_pref/labels.csv")))
    stems = [r["stem"] for r in labs]
    F2 = np.array([[float(feats[s]["brightness"]), float(feats[s]["tnoise"])] for s in stems])
    y = np.array([float(r["sigma_pref"]) for r in labs])
    lo = np.array([float(r["sigma_lo"]) for r in labs])
    hi = np.array([float(r["sigma_hi"]) for r in labs])

    full = loco(F2, y, lo, hi)
    bonly = loco(F2[:, :1], y, lo, hi)

    gate_full = (full["mae"] < full["mae_bestconst"]) and full["sign_p"] < 0.05
    gate_bonly = (bonly["mae"] < bonly["mae_bestconst"]) and bonly["sign_p"] < 0.05
    tnoise_earns = full["mae"] < bonly["mae"]
    ship = None
    if gate_full and tnoise_earns:
        ship = ("brightness+tnoise", F2, ["brightness", "tnoise"])
    elif gate_bonly:
        ship = ("brightness-only", F2[:, :1], ["brightness"])

    lines = ["# LOCO report — preference-labeled sigma predictor", ""]
    for name, r in (("brightness+tnoise", full), ("brightness-only", bonly)):
        lines += [f"## {name}",
                  f"- interval-MAE: model {r['mae']:.5f} | best-constant {r['mae_bestconst']:.5f} | fixed-0.07 {r['mae_070']:.5f}",
                  f"- wins/losses/ties vs best-constant: {r['wins']}/{r['losses']}/{r['ties']}  sign_p={r['sign_p']:.4f}",
                  f"- spearman(pred, label) = {r['spearman']:.3f}   boundary-binding frac: {r['bound_frac']}", ""]
    lines += ["## per-clip (full model)", "|stem|label|lo,hi|pred|const|", "|--|--|--|--|--|"]
    for i, s in enumerate(stems):
        lines.append(f"|{s}|{y[i]:.2f}|{lo[i]:.2f},{hi[i]:.2f}|{full['pred'][i]:.4f}|{full['pred_const'][i]:.2f}|")
    lines += ["", f"## decision inputs: gate_full={gate_full} tnoise_earns={tnoise_earns} gate_bonly={gate_bonly}",
              f"SHIP: {ship[0] if ship else 'NONE (gate failed)'}"]
    open("tools/optsig_pref/loco_report.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"SHIP: {ship[0] if ship else 'NONE'}")

    if ship:
        name, Fs, fnames = ship
        m = fit_fold(Fs, y)
        # NOTE: deliberately NOT the deployed filename — production ships the
        # threshold model (kind="brightness-threshold", make_threshold_model.py);
        # bsvd_optsig.py rejects kind="linear", so never overwrite v1.json here.
        art = dict(kind="linear",
                   features=fnames,
                   feature_means=[float(v) for v in m["mu"]],
                   feature_stds=[float(v) for v in m["sd"]],
                   weights=[float(v) for v in m["w"][:-1]],
                   bias=float(m["w"][-1]),
                   sigma_grid=[float(g) for g in GRID],
                   fit=dict(n=len(y), kind="sign-constrained-linear", target="user-preference-2026-07",
                            loco_mae=full["mae"] if name == "brightness+tnoise" else bonly["mae"]))
        json.dump(art, open("models/bsvd_optsig_pref_v1_linear.json", "w"), indent=1)
        print("wrote models/bsvd_optsig_pref_v1_linear.json")

if __name__ == "__main__":
    main()
