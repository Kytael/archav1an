# tools/optsig_pref/check_labels.py
import csv, sys
GRID = [round(0.01 * k, 2) for k in range(1, 9)]   # 0.01..0.08
def main():
    rows = list(csv.DictReader(open("tools/optsig_pref/labels.csv")))
    errs, warns = [], []
    if len(rows) != 21:
        errs.append(f"expected 21 rows, got {len(rows)}")
    for r in rows:
        s = r["stem"]
        try:
            p, lo, hi = (float(r["sigma_pref"]), float(r["sigma_lo"]), float(r["sigma_hi"]))
        except ValueError:
            errs.append(f"{s}: non-numeric/empty fields"); continue
        for name, v in (("pref", p), ("lo", lo), ("hi", hi)):
            if round(v, 2) not in GRID:
                errs.append(f"{s}: sigma_{name}={v} not on grid")
        if not (lo <= p <= hi):
            errs.append(f"{s}: lo<=pref<=hi violated ({lo},{p},{hi})")
        if min(abs(p - GRID[0]), abs(p - GRID[-1])) < 1e-9 or lo == GRID[0] or hi == GRID[-1]:
            warns.append(f"{s}: endpoint label (pref={p}, lo={lo}, hi={hi}) — interval-censored")
    for w in warns: print("WARN:", w)
    for e in errs: print("ERROR:", e)
    print(f"{len(rows)} rows, {len(errs)} errors, {len(warns)} endpoint warnings")
    sys.exit(1 if errs else 0)
if __name__ == "__main__":
    main()
