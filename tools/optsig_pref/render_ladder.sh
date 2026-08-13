#!/bin/bash
# render_ladder.sh <windows.csv> <out_dir> <all|stem1,stem2,...>
# Renders the 8-rung sigma ladder per selected included clip.
# Idempotent + failure-tolerant: verifies each output has the expected packet
# count; truncated/partial outputs are re-rendered, render failures are logged
# (not fatal) and the batch continues. Exits nonzero if any render failed.
set -uo pipefail
WCSV="$1"; OUT="$2"; SEL="${3:-all}"
VS_PREFIX="${VS_PREFIX:-/opt/archav1an}"
VS="$VS_PREFIX/bin/vspipe"
# Anything we compile is the default; the system copy is the fallback, not the
# other way round. This script used to name the two /usr/bin paths outright, so
# it ran the distro build even on a host that had just compiled its own.
# LD_LIBRARY_PATH has to move with the binaries: the prefix ffmpeg against
# the system libavcodec.so.63 shares a soname but not an ABI, and exits 127 at
# bitstream-filter teardown after writing a complete file. See tools/prefix_env.py.
FFMPEG="$VS_PREFIX/bin/ffmpeg";   [ -x "$FFMPEG" ]  || FFMPEG="$(command -v ffmpeg)"
FFPROBE="$VS_PREFIX/bin/ffprobe"; [ -x "$FFPROBE" ] || FFPROBE="$(command -v ffprobe)"
[ -d "$VS_PREFIX/lib" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$VS_PREFIX/lib"
SIGMAS="0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08"
TMP=/home/user/archav1an/Temp/optsig_pref; mkdir -p "$TMP"
FAILS="$TMP/failures.txt"; : > "$FAILS"
pkts() { "$FFPROBE" -v error -select_streams v:0 -count_packets \
  -show_entries stream=nb_read_packets -of csv=p=0 "$1" 2>/dev/null || echo 0; }
tail -n +2 "$WCSV" | while IFS=, read -r stem path nframes start length included reason; do
  [ "$included" = "1" ] || continue
  if [ "$SEL" != "all" ]; then case ",$SEL," in *",$stem,"*) ;; *) continue ;; esac; fi
  read -r W H < <("$FFPROBE" -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$path" | tr ',' ' ')
  RATE=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$path")
  CACHE="$TMP/${stem}.ffindex"
  mkdir -p "$OUT/$stem"
  for s in $SIGMAS; do
    tag=${s#0.}
    o="$OUT/$stem/${stem}_s${tag}.mkv"
    if [ -f "$o" ]; then
      pk=$(pkts "$o")
      if [ "$pk" = "$length" ]; then echo "SKIP $o ($pk pkts)"; continue; fi
      echo "REDO $o (had $pk != $length pkts)"; rm -f "$o"
    fi
    echo "=== RENDER $stem sigma=$s -> $o ==="
    if CLIP_PATH="$path" WIN_START="$start" WIN_LEN="$length" BSVD_SIGMA="$s" FFMS_CACHE="$CACHE" \
         "$VS" tools/optsig_pref/ladder.vpy - 2>"$TMP/${stem}_s${tag}.vspipe.log" | \
         "$FFMPEG" -hide_banner -loglevel error -f rawvideo -pix_fmt gbrp -s "${W}x${H}" -r "$RATE" -i - \
           -c:v ffv1 -level 3 -y "$o"; then
      pk=$(pkts "$o")
      if [ "$pk" != "$length" ]; then
        echo "FAIL $stem s=$s (got $pk != $length pkts)"; rm -f "$o"; echo "$stem $s pkts=$pk" >> "$FAILS"
      fi
    else
      echo "FAIL $stem s=$s (render error; see $TMP/${stem}_s${tag}.vspipe.log)"; rm -f "$o"; echo "$stem $s render-error" >> "$FAILS"
    fi
  done
done
if [ -s "$FAILS" ]; then echo "BATCH DONE WITH FAILURES:"; cat "$FAILS"; exit 1; fi
echo "BATCH DONE (all renders verified)"
