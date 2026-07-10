import csv, subprocess
from tools.optsig_pref.corpus_paths import resolve_corpus  # (stem, path) via size-match

def window_for(n, length=180, frac=0.40, min_len=300):
    if n < min_len:
        return None
    return (round(frac * n), length)

def _nframes(path):
    out = subprocess.check_output(
        ["/usr/bin/ffprobe","-v","error","-select_streams","v:0","-count_packets",
         "-show_entries","stream=nb_read_packets","-of","csv=p=0", path], text=True)
    return int(out.strip())

def main(out_csv="tools/optsig_pref/windows.csv"):
    rows = []
    for stem, path in resolve_corpus():
        n = _nframes(path); w = window_for(n)
        if w is None:
            rows.append([stem, path, n, "", "", "0", f"N<300 ({n})"])
        else:
            rows.append([stem, path, n, w[0], w[1], "1", ""])
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["stem","path","n_frames","start","length","included","reason"])
        wr.writerows(rows)
    print(f"wrote {out_csv}: {sum(1 for r in rows if r[5]=='1')}/{len(rows)} included")

if __name__ == "__main__":
    main()
