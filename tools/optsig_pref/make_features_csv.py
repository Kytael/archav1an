# tools/optsig_pref/make_features_csv.py
import csv
from tools.optsig_pref.features import extract
def main():
    rows = [r for r in csv.DictReader(open("tools/optsig_pref/windows.csv")) if r["included"] == "1"]
    with open("tools/optsig_pref/features.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["stem", "brightness", "tnoise", "tnoise_static_frac", "spatial_std5", "mad4"])
        for r in rows:
            e = extract(r["path"], int(r["start"]), int(r["length"]))
            wr.writerow([r["stem"], f"{e['brightness']:.6f}", f"{e['tnoise']:.6f}",
                         f"{e['tnoise_static_frac']:.4f}", f"{e['spatial_std5']:.6f}", f"{e['mad4']:.6f}"])
            print(r["stem"], "ok")
if __name__ == "__main__":
    main()
