#!/bin/bash
## Re-encode rollout videos (OpenCV writes mp4v, which VS Code / browsers
## cannot play) to H.264 in rollouts/h264/.  Originals are left untouched:
## analysis/score_grasp.py reads them and they are the record.  Skips files
## already converted.  Runs niced on 2 threads so a live rollout (50 Hz
## control loop, sensitive to CPU contention) is not starved.
##
##   bash analysis/reencode_rollouts.sh            # everything under rollouts/
##   bash analysis/reencode_rollouts.sh 20260919   # only files matching a pattern
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/rollouts"; DST="$SRC/h264"; mkdir -p "$DST"
pat="${1:-}"
n=0; skip=0
for f in "$SRC"/rollout_*"$pat"*.mp4; do
  [ -e "$f" ] || continue
  out="$DST/$(basename "$f")"
  if [ -s "$out" ] && [ "$out" -nt "$f" ]; then skip=$((skip+1)); continue; fi
  nice -n 19 ffmpeg -hide_banner -loglevel error -y -i "$f" \
    -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart -threads 2 \
    "$out" && n=$((n+1)) && echo "encoded $(basename "$f")" || echo "FAILED  $(basename "$f")"
done
echo "done: $n encoded, $skip already up to date -> $DST"
