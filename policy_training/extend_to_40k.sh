#!/bin/bash
# Wait for tonight's two fine-tunes to reach 20000 steps, then CONTINUE each
# one to 40000 in the same output directory.
#
#   nohup setsid ./extend_to_40k.sh > logs/extend_chain.log 2>&1 &
#
# WHY --resume AND NOT --init-from.  Resuming restores the AdamW moments and
# the step counter, so step 20001 carries on exactly where 20000 left off and
# the new checkpoints are numbered 025000..040000 beside the existing ones.
# --init-from would reload the weights but start Adam from zero, which puts a
# transient at the join for no reason.  lerobot trains `range(step, steps)`,
# so a LARGER --steps on a resume is precisely how a finished run is extended.
#
# The learning rate stays 1e-6: load_training_state() overwrites the freshly
# built optimizer's param_groups (lr included) from the checkpoint.
#
# SAFETY: each job is only extended if its last checkpoint really reached
# 20000.  A run that died early is left alone and reported, because resuming
# a crash without looking at it is how a bad run gets silently doubled.
set -u
cd "$(dirname "$0")" || exit 1
PY=/home/devi/miniconda3/envs/gym_av312/bin/python
DC=/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts

step_of () {   # last checkpointed step, or 0
  "$PY" -c "import json,sys
try: print(json.load(open('outputs/train/$1/checkpoints/last/training_state/training_step.json'))['step'])
except Exception: print(0)" 2>/dev/null
}

wait_for_pid () {
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
}

extend () {
  local PID=$1 JOB=$2 GPU=$3; shift 3
  echo "[chain] waiting for $JOB (pid $PID) ..."
  wait_for_pid "$PID"
  local S; S=$(step_of "$JOB")
  echo "[chain] $JOB exited; last checkpoint step = $S"
  if [ "$S" -lt 20000 ]; then
    echo "[chain] !! $JOB stopped at $S, short of 20000 -- NOT extending."
    echo "[chain]    look at logs/$JOB.log before resuming it by hand."
    return 1
  fi
  echo "[chain] extending $JOB to 40000 on GPU $GPU"
  ## --init-from its OWN checkpoint: build_policy_config then takes the
  ## architecture from that config.json instead of re-deriving a default
  ## ACTConfig, so any non-default field in the original run cannot drift and
  ## break the weight load.  --resume overwrites pretrained_path with the
  ## same directory, so this only fixes the config source.
  CUDA_VISIBLE_DEVICES=$GPU "$PY" train_real.py "$@" \
      --init-from "outputs/train/$JOB/checkpoints/last/pretrained_model" \
      --resume --steps 40000 --save-freq 5000 \
      --batch-size 8 --num-workers 8 --job-name "$JOB" \
      >> "logs/$JOB.log" 2>&1
  echo "[chain] $JOB finished stage 2 at step $(step_of "$JOB")"
}

extend 1502025 transfer_flower_20260910_act_dagger_r1 1 \
    --root "$DC/dataset/lerobot/transfer_flower/20260903_214256" \
    --policy act --successes-only \
    --dagger-from 50 --dagger-min-interventions 1 --lr 1e-6 &
FLOWER=$!

extend 3275737 shape_sorter_cube_20260910_act_dagger_r1 0 \
    --root "$DC/dataset/lerobot/shape_sorter_canon/20260907_204429_r1" \
    --policy act --cameras right_wrist top_scene oak_left \
    --episodes $(cat cube_episodes.txt) --lr 1e-6 &
CUBE=$!

wait $FLOWER $CUBE
echo "[chain] both stages complete"
