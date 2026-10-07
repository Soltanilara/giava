#!/bin/bash
## Re-encode rollout videos.  OpenCV writes mp4v (MPEG-4 Part 2), which almost
## nothing previews.  Two copies, originals left untouched (score_grasp.py
## reads them and they are the record):
##
##   rollouts/h264/   H.264 -- browsers, Slack, the review HTML pages, and
##                    vlm_annotate.py / episode_features.py read these.
##   rollouts/webm/   VP9 -- the one VS Code's built-in player can open (it
##                    ships without H.264 and MPEG-4 decoders).
##
## The recorder cannot write either directly: this OpenCV has no H.264 encoder,
## and its VP9 path runs at ~11 fps against a 50 Hz loop (measured 2026-10-07).
##
## Skips files already converted.  Runs niced on 2 threads so a live rollout
## (50 Hz control loop, sensitive to CPU contention) is not starved.
##
##   bash analysis/reencode_rollouts.sh            # everything under rollouts/
##   bash analysis/reencode_rollouts.sh 20260919   # only files matching a pattern
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/rollouts"; H264="$SRC/h264"; WEBM="$SRC/webm"; mkdir -p "$H264" "$WEBM"
pat="${1:-}"
n=0; skip=0
for f in "$SRC"/rollout_*"$pat"*.mp4; do
  [ -e "$f" ] || continue
  base="$(basename "$f" .mp4)"
  out="$H264/$base.mp4"
  if [ -s "$out" ] && [ "$out" -nt "$f" ]; then skip=$((skip+1)); else
    nice -n 19 ffmpeg -hide_banner -loglevel error -y -i "$f" \
      -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart -threads 2 \
      "$out" && n=$((n+1)) && echo "h264 $base" || echo "FAILED h264 $base"
  fi
  out="$WEBM/$base.webm"
  if [ -s "$out" ] && [ "$out" -nt "$f" ]; then skip=$((skip+1)); else
    nice -n 19 ffmpeg -hide_banner -loglevel error -y -i "$f" \
      -c:v libvpx-vp9 -crf 32 -b:v 0 -deadline good -cpu-used 4 -row-mt 1 -an -threads 2 \
      "$out" && n=$((n+1)) && echo "webm $base" || echo "FAILED webm $base"
  fi
done
echo "done: $n encoded, $skip already up to date -> $H264, $WEBM"
