#!/bin/bash
# Build .archive-run/manifest-raw.tsv by probing the source host.
#
# The manifest was previously built by an ad-hoc command that lived nowhere, so
# when the file was lost there was nothing to rebuild it from. Spec 5.3 says it
# is "built once by walking gpu1 over ssh"; this is that walk.
#
# Columns, tab separated, with a trailing tab:
#   relpath, size, then one "rate,frames" per video stream, then duration
# A source with an embedded thumbnail has two rate columns; the parser takes
# frames from the first (tools/archive_batch/manifest.py).
#
# Scope is spec 1: SetA and SetB for 2001-2007, plus the two named
# SetB folders. Deliberately excluded: SetA/2000, SetB/2009..2011,
# and the sibling Dance Projects, ginza and Workshops trees.
set -euo pipefail

HOST="${SOURCE_HOST:-gpu1}"
ROOT="${ARCHIVE_ROOT:-/mnt/media/dance}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/.archive-run/manifest-raw.tsv"
TMP="$OUT.partial"

mkdir -p "$(dirname "$OUT")"
if [ -s "$OUT" ]; then
    echo "Refusing to overwrite a non-empty $OUT ($(wc -l < "$OUT") rows)."
    echo "Move it aside first if you really mean to rebuild."
    exit 2
fi

echo "Probing $HOST:$ROOT -- a few thousand files took about 15 minutes when first built."

# Runs entirely on the source host: one ffprobe per file, and the 9p crossing
# is what makes this slow, so the loop must not run locally.
ssh "$HOST" ROOT="$ROOT" bash -s > "$TMP" <<'REMOTE'
set -u
cd "$ROOT" || exit 1
{
  for y in 2001 2002 2003 2004 2005 2006 2007; do
    [ -d "SetA/$y" ] && find "SetA/$y" -type f \( -iname '*.mov' -o -iname '*.mp4' \) -print0
    [ -d "SetB/$y" ] && find "SetB/$y" -type f \( -iname '*.mov' -o -iname '*.mp4' \) -print0
  done
  for d in "SetB/routine-one" "SetB/routine-two"; do
    [ -d "$d" ] && find "$d" -type f \( -iname '*.mov' -o -iname '*.mp4' \) -print0
  done
} | sort -z | while IFS= read -r -d '' f; do
    size=$(stat -c%s "$f" 2>/dev/null) || continue
    # One -show_entries with colon-separated sections. Passing the flag twice
    # makes ffprobe emit a blank column and repeat the stream row, which is
    # harmless to the parser (it reads frames from the first rate column) but
    # produces a ragged file.
    probe=$(ffprobe -v error -select_streams v \
              -show_entries stream=r_frame_rate,nb_frames:format=duration \
              -of csv=p=0 "$f" 2>/dev/null) || continue
    [ -n "$probe" ] || continue
    printf '%s\t%s\t%s\t\n' "$f" "$size" "$(printf '%s' "$probe" | tr '\n' '\t' | sed 's/\t$//')"
done
REMOTE

rows=$(wc -l < "$TMP")
if [ "$rows" -lt 1 ]; then
    echo "Probe produced no rows; leaving $TMP in place for inspection."
    exit 1
fi
mv "$TMP" "$OUT"

echo
echo "Wrote $OUT"
awk -F'\t' '{split($3,a,","); n++; b+=$2; f+=a[2]; d+=$4}
  END {printf "  files  : %d\n  size   : %.1f GiB\n  frames : %.2f M\n  hours  : %.1f\n",
       n, b/1073741824, f/1e6, d/3600}' "$OUT"
echo
echo "Spec 1 measured a few thousand files, a few TiB, millions of frames, days of footage."
