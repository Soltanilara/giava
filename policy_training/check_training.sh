#!/bin/bash
# Is the fine-tuning actually still running?  Answers from the log's own step
# counter and the process table, not from the presence of a checkpoint --
# `checkpoints/last` is rewritten at EVERY save, so it exists long before a
# run is finished and is not evidence of completion.
cd "$(dirname "$0")" || exit 1
for L in logs/*_dagger_r1.log; do
  J=$(basename "$L" .log)
  STEP=$(tail -c 400 "$L" | tr '\r' '\n' | grep -o "[0-9]*/[0-9]* \[[^]]*\]" | tail -1)
  ALIVE=$(pgrep -f "job-name $J" >/dev/null && echo RUNNING || echo "not running")
  AGE=$(( $(date +%s) - $(stat -c %Y "$L") ))
  printf "%-44s %-10s  %s   (log touched %ss ago)\n" "$J" "$ALIVE" "${STEP:-no progress line}" "$AGE"
  printf "%-44s checkpoints: %s\n" "" "$(ls outputs/train/$J/checkpoints/ 2>/dev/null | tr '\n' ' ')"
done
