#!/bin/bash
# Clear everything the archive batch has produced, so the real run starts clean.
#
# Everything encoded before the first production run was gate and benchmark
# output, and some of it was produced by a different SvtAv1EncApp build (see
# a03f405). Mixing those into the archive would leave a handful of files nobody
# can account for later, so they go.
#
# Dry run by default. Pass --yes to actually delete.
set -euo pipefail

HOST="${SOURCE_HOST:-gpu1}"
ROOT="${ARCHIVE_ROOT:-/mnt/media/dance}"
ENCODED="$ROOT/encoded"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$REPO/.archive-run/state.jsonl"
STAGE="$REPO/Temp/_stage"

APPLY=0
[ "${1:-}" = "--yes" ] && APPLY=1

echo "=== published output on $HOST:$ENCODED ==="
ssh "$HOST" "find '$ENCODED' -type f -name '*.mkv' -printf '%s\t%P\n' 2>/dev/null" \
  | awk -F'\t' '{n++; t+=$1} END {printf "  %d file(s), %.2f GiB\n", n+0, t/1073741824}'

echo "=== local run state ==="
if [ -f "$STATE" ]; then
    echo "  $STATE: $(wc -l < "$STATE") record(s)"
else
    echo "  $STATE: absent"
fi
echo "  staged leftovers: $(find "$STAGE" -type f 2>/dev/null | wc -l) file(s)"

if [ "$APPLY" -ne 1 ]; then
    echo
    echo "Dry run. Re-run with --yes to delete the above."
    exit 0
fi

echo
echo "=== deleting ==="
# Remove only .mkv files and then the empty tree, so an unexpected file is
# left behind and noticed rather than silently destroyed.
ssh "$HOST" "find '$ENCODED' -type f -name '*.mkv' -delete 2>/dev/null; \
             find '$ENCODED' -mindepth 1 -type d -empty -delete 2>/dev/null; true"
leftover=$(ssh "$HOST" "find '$ENCODED' -type f 2>/dev/null | wc -l")
if [ "$leftover" != "0" ]; then
    echo "  WARNING: $leftover non-.mkv file(s) left under $ENCODED; inspect before rerunning"
    ssh "$HOST" "find '$ENCODED' -type f 2>/dev/null | head"
fi
rm -f "$STATE"
find "$STAGE" -type f -delete 2>/dev/null || true
echo "  published output removed, state cleared, stage swept"
echo
echo "The next run starts from clip 1 of $(wc -l < "$REPO/.archive-run/manifest-raw.tsv") ."
