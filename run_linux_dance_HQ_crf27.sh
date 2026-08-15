#!/bin/bash

# run_linux_dance_HQ_crf27.sh
# Single-pass SvtAv1EncApp encode — Dance / Performance CRF 27, full temporal context.
# Bypasses av1an chunking so encoder sees the entire clip without forced chunk resets.
# Place source files in Input/, encoded output goes to Output/.

cd "$(dirname "$0")"

# Activate Python venv
source "$(dirname "$(realpath "$0")")/activate-venv.sh"
touch "tools/sh-used-$(basename "$0").txt"

# --lp is a parallelism LEVEL in [0, 6], not a thread count, so the old
# LP=$(nproc) here only ever clamped to 6 with a warning. 0 lets the encoder
# choose from the core count: level 5 on a 16-thread host, 6 at 24 or more.
# See docs/lp-and-encoder-parallelism.md.
echo "Starting SvtAv1EncApp Batch (Dance HQ CRF 27) — single-pass..."
# Extra args passed to this script are forwarded to svtav1-dispatch.py (e.g. --denoise-scunet)
EXTRA_ARGS=("$@")

trap 'trap "" INT TERM; echo "Interrupted."; kill 0; exit 130' INT TERM

rm -f "tools/tag-manifest.txt"
mkdir -p Input Output

# How many clips to encode at once. One encoder does not fill a wide machine:
# on encoder-host (16c/32t) a single stream measured 22.86 fps and three measured
# 34.04, +49%. Past three the curve is flat -- five gave 34.72 -- so three is
# the useful setting here. Narrower hosts gain less or nothing: gpu1 (8c/16t)
# peaks at two, gpu3 loses throughput with every added stream. Override with
# JOBS=1 to get the old sequential behaviour back. See docs/encode-capacity.md.
JOBS=${JOBS:-3}

# A background job cannot append to the parent's array, so failures go to a
# file. mktemp keeps concurrent runs of this script from sharing it.
FAILED_LIST=$(mktemp "${TMPDIR:-/tmp}/dance-failed.XXXXXX")
# One marker file per running encode, naming the clip and its log, so the
# progress display knows which of the JOBS slots are busy.
STATUS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/dance-status.XXXXXX")
trap 'rm -f "$FAILED_LIST"; rm -rf "$STATUS_DIR"' EXIT

# Last progress record SvtAv1EncApp wrote to a log. It redraws with \r rather
# than \n, so the whole encode is one enormous line: split on \r and take the
# final record whole -- frames, fps, bitrate, size and elapsed time, with the
# encoder's own colour escapes left intact.
last_progress() {
    # Only the tail: a long clip writes one record per frame and the log runs
    # to megabytes, which is not worth re-reading twice a second for the last
    # 100 bytes of it.
    tail -c 65536 "$1" 2>/dev/null | tr '\r' '\n' | grep -o 'Encoding:.*' | tail -1
}

# Visible width of a string, ignoring colour escapes. Needed because a line
# that wraps would occupy two rows, and the redraw below moves the cursor back
# by a fixed number of rows -- a wrapped line desynchronises the whole block.
visible_len() {
    # Pure bash: this runs several times a second while the machine is busy
    # encoding, and forking a sed for each call was the wrong trade.
    local plain=$1 pre post
    while [[ $plain == *$'\e['* ]]; do
        pre=${plain%%$'\e['*}
        post=${plain#*$'\e['}
        post=${post#*m}
        plain=$pre$post
    done
    echo "${#plain}"
}

# Trim a progress record to fit, by dropping whole ` | ` fields from the end
# (elapsed time first, then size, then bitrate). Each field carries its own
# colour reset, so cutting on that boundary can never leave a half-written
# escape sequence the way a plain character truncation would.
fit_record() {
    local rec=$1 max=$2
    while [ "$(visible_len "$rec")" -gt "$max" ] && [[ "$rec" == *" | "* ]]; do
        rec="${rec% | *}"
    done
    printf '%s' "$rec"
}

# The progress display owns the terminal while encoding. Completion lines are
# routed through a file rather than echoed directly, because a line printed by
# the main shell or by a background job lands in the middle of the block and
# desynchronises the cursor-up redraw -- which showed up as stale rows left
# behind whenever a job finished and another started.
EVENTS="$STATUS_DIR/events"
: > "$EVENTS"
PROGRESS_ON=0
[ -t 1 ] && PROGRESS_ON=1
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-0.5}

event() {
    printf '%s\n' "$1" >> "$EVENTS"
    # With no display, nothing else is going to print these.
    [ "$PROGRESS_ON" = 1 ] || printf '%s\n' "$1"
}

SHOWN=0
DRAWN=0

# Print completions that appeared since the last tick. They scroll away above
# the live block, exactly as they would have if echoed directly -- but printed
# from here the row accounting stays correct.
emit_events() {
    local n line
    n=$(wc -l < "$EVENTS" 2>/dev/null)
    n=${n:-0}
    while [ "$SHOWN" -lt "$n" ]; do
        SHOWN=$((SHOWN + 1))
        line=$(sed -n "${SHOWN}p" "$EVENTS")
        printf '\033[K%s\n' "$line"
    done
}

draw_slots() {
    local m name log rec cols namew
    # Ask the controlling terminal, not stdout: this runs inside command
    # substitution, where stdout is a pipe, and `tput cols` there cannot
    # ioctl the terminal and silently returns the terminfo default of 80.
    cols=$(stty size </dev/tty 2>/dev/null | awk '{print $2}')
    [ -n "$cols" ] || cols=$(tput cols 2>/dev/null) || cols=120
    DRAWN=0
    for m in "$STATUS_DIR"/slot*; do
        [ -e "$m" ] || continue
        IFS='|' read -r name log < "$m" || continue
        rec=$(last_progress "$log")
        rec="${rec#Encoding: }"      # the clip name already says what it is
        if [ -z "$rec" ]; then rec="starting"; fi
        # Name first, then whatever record fits. Squeezing the name instead
        # left it reading "MVI_14" on an 80-column terminal, which is worse
        # than losing the elapsed-time field.
        if   [ "$cols" -ge 110 ]; then namew=28
        elif [ "$cols" -ge 90 ];  then namew=20
        else                           namew=14
        fi
        rec=$(fit_record "$rec" $(( cols - namew - 4 )))
        printf '\033[K  %-*.*s %s\033[0m\n' "$namew" "$namew" "$name" "$rec"
        DRAWN=$((DRAWN + 1))
    done
    while [ "$DRAWN" -lt "$JOBS" ]; do
        printf '\033[K  (idle)\n'
        DRAWN=$((DRAWN + 1))
    done
}

# A fixed block of JOBS lines means the cursor always moves back the same
# distance. Exits on a flag rather than a signal so it can flush the last
# completions and erase the block before returning.
progress_loop() {
    [ "$PROGRESS_ON" = 1 ] || return 0
    local stopping i
    printf '\n'
    while :; do
        stopping=0
        [ -f "$STATUS_DIR/stop" ] && stopping=1
        [ "$DRAWN" -gt 0 ] && printf '\033[%dA' "$DRAWN"
        emit_events
        if [ "$stopping" = 1 ]; then
            i=0
            while [ "$i" -lt "$JOBS" ]; do printf '\033[K\n'; i=$((i + 1)); done
            printf '\033[%dA' "$JOBS"
            return 0
        fi
        draw_slots
        sleep "$PROGRESS_INTERVAL"
    done
}

# Background encode jobs only: `jobs -rp` also lists the progress loop, and
# counting it would quietly run one fewer encode than asked for.
encodes_running() {
    local n=0 p
    for p in $(jobs -rp); do
        [ "$p" = "${PROGRESS_PID:-}" ] || n=$((n + 1))
    done
    echo "$n"
}

encode_one() {
    local f=$1 out=$2 tag=$3 log=$4
    printf '%s|%s\n' "$(basename -- "$f")" "$log" > "$STATUS_DIR/slot$BASHPID"
    if python3 tools/svtav1-dispatch.py \
        -i "$f" \
        -o "$out" \
        --quality 27 \
        --photon-noise 6 \
        --lp 0 \
        --speed 4 \
        --temp-tag "$tag" \
        --encoder-params "--tune 3 --hbd-mds 1 --keyint 305 --ac-bias 0.8 --sharp-tx 1 --sharpness 1 --tf-strength 2 --variance-boost-strength 1 --variance-octile 7 --enable-dlf 2" \
        "${EXTRA_ARGS[@]}" > "$log" 2>&1; then
        event "DONE:   $f"
    else
        event "FAILED: $f  (see $log)"
        printf '%s\n' "$f" >> "$FAILED_LIST"
    fi
    rm -f "$STATUS_DIR/slot$BASHPID"
}

progress_loop &
PROGRESS_PID=$!

while IFS= read -r -d '' f <&3; do
    filename=$(basename -- "$f")
    stem="${filename%.*}"
    rel_dir=$(dirname -- "$f")
    rel_dir="${rel_dir#Input}"
    rel_dir="${rel_dir#/}"

    if [ -n "$rel_dir" ]; then
        mkdir -p "Output/${rel_dir}"
        OUTPUT_FILE="Output/${rel_dir}/${stem}-av1.mkv"
    else
        OUTPUT_FILE="Output/${stem}-av1.mkv"
    fi

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping \"$f\" — output already exists."
        continue
    fi

    # Three encoders interleaving their progress on one terminal is unreadable,
    # and SvtAv1EncApp redraws with \r, so each clip gets its own log.
    LOG_FILE="${OUTPUT_FILE%.mkv}.log"
    # Working files go to Temp/<tag>/<stem>. Without a tag, two inputs sharing
    # a stem in different Input subdirectories would land in the same Temp
    # directory -- harmless when they ran one after another, a corrupted output
    # when they run together. The tag is a hash of the path, so it is unique
    # per input and stable across re-runs.
    TAG="j$(printf '%s' "$f" | md5sum | cut -c1-10)"

    # Wait for a free slot BEFORE launching, so at most JOBS run at once.
    # The progress loop is a background job too and must not occupy a slot.
    while [ "$(encodes_running)" -ge "$JOBS" ]; do wait -n; done
    [ "$PROGRESS_ON" = 1 ] || echo "START:  $f"
    encode_one "$f" "$OUTPUT_FILE" "$TAG" "$LOG_FILE" &

done 3< <(find Input -type f \( -iname "*.mkv" -o -iname "*.mp4" -o -iname "*.mov" -o -iname "*.m2ts" \) -print0 | sort -z)

# Every clip must finish before cleanup.py runs, or it deletes temp files out
# from under an encoder that is still using them. A bare `wait` would also wait
# on the progress loop, which never returns on its own, so wait on the encode
# jobs by pid and stop the display afterwards.
for job in $(jobs -p); do
    [ "$job" = "$PROGRESS_PID" ] || wait "$job"
done
touch "$STATUS_DIR/stop"
wait "$PROGRESS_PID" 2>/dev/null
mapfile -t FAILED_FILES < "$FAILED_LIST"

# --- CLEANUP ---
echo "Cleaning up temporary files and folders..."
python3 tools/cleanup.py

if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    echo "WARNING: ${#FAILED_FILES[@]} file(s) FAILED:"
    printf '  %s\n' "${FAILED_FILES[@]}"
    exit 1
fi
echo "All tasks finished."
