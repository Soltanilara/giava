#!/bin/bash
# Run several trainings back to back on ONE gpu, unattended.
#
#   ./train_chain.sh <dataset_run_dir> [gpu]
#
# Runs, in order:
#   1. all successful episodes          -> <task>_act_50ep
#   2. the first 25 successful episodes -> <task>_act_25ep
#
# Two runs on the same data is the data-efficiency curve: if 25 is nearly as
# good as 50, the remaining pieces need half the collection time.  They run
# SEQUENTIALLY on one gpu on purpose -- two ACT jobs sharing a 3090 each run
# at roughly half speed, so nothing is gained by overlapping them, and the
# other gpu stays free for rollouts.
#
# DO NOT start this while data_collection.py is running on the same gpu: the
# recorder solves IK on the gpu every tick, and a training job beside it
# pushes solve times past the 20 ms control period.
set -u

ROOT="${1:?usage: train_chain.sh <dataset_run_dir> [gpu]}"
GPU="${2:-0}"
PY=/home/devi/miniconda3/envs/gym_av312/bin/python
CAMS="right_wrist top_scene oak_left"
STEPS=100000
STAMP=$(date +%Y%m%d_%H%M%S)
cd "$(dirname "$0")" || exit 1
mkdir -p logs

TASK=$($PY -c "import json,sys; print(json.load(open('$ROOT/meta.json'))['task'])") || exit 1

## The 25-episode subset is the FIRST 25 successes, in recording order -- not a
## random draw, so it is reproducible and matches "what if I had stopped at 25".
EP25=$($PY - "$ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for c in (root / "episode_outcomes.jsonl", root.parent / "episode_outcomes.jsonl"):
    if c.exists():
        recs = {}
        for line in c.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[int(r["episode_index"])] = r
        good = sorted(i for i, r in recs.items() if r.get("outcome") == "success")
        print(" ".join(str(i) for i in good[:25]))
        break
else:
    sys.exit("no episode_outcomes.jsonl")
EOF
) || exit 1

run () {
  local name="$1"; shift
  local log="logs/${name}_${STAMP}.log"
  echo "=== $(date '+%F %T')  START $name  (gpu $GPU) -> $log"
  CUDA_VISIBLE_DEVICES="$GPU" $PY train_real.py \
      --root "$ROOT" --policy act --cameras $CAMS \
      --steps "$STEPS" --save-freq 10000 --batch-size 8 --num-workers 8 \
      --successes-only --job-name "$name" "$@" > "$log" 2>&1
  echo "=== $(date '+%F %T')  DONE  $name  exit=$?"
}

echo "dataset : $ROOT"
echo "task    : $TASK"
echo "25-ep   : $EP25"
echo

run "${TASK}_act_50ep"
run "${TASK}_act_25ep" --episodes $EP25

echo "=== $(date '+%F %T')  CHAIN COMPLETE"
