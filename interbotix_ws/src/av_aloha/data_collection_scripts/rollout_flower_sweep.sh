#!/bin/bash
# Roll out every transfer_flower variant back to back, 5 episodes each.
#
#   ./rollout_flower_sweep.sh [n_episodes]
#
# The FIRST run saves the starting scenes; every later run replays them with
# --replicate, so the same block positions are used throughout.  At n=5 the
# placement noise between runs is far larger than any difference between the
# models, so unpaired trials would compare the table, not the policies.
#
# Each run stops for ENTER before every episode and prompts for scores after,
# so this is a queue to work through, not something to walk away from.
set -u

N="${1:-5}"
B=/home/devi/giava/policy_training/outputs/train
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/devi/miniconda3/envs/gym_av312/bin/python
PY_VLA=/home/devi/miniconda3/envs/vla312/bin/python
COMMON="--mode right --episodes $N --seconds ${SECONDS_CAP:-20} --stall-seconds 5 --score --video --engage"
cd "$HERE" || exit 1

run () {
  local label="$1" ckpt="$2" python="$3"; shift 3
  echo
  echo "############################################################"
  echo "#  $label"
  echo "#  $ckpt"
  echo "############################################################"
  if [ ! -f "$ckpt/config.json" ]; then
    echo "!! missing checkpoint, skipping"; return
  fi
  CUDA_VISIBLE_DEVICES=0 "$python" rollout_policy.py --checkpoint "$ckpt" $COMMON "$@"
  echo "--- $label done.  ENTER for the next model, Ctrl-C to stop here."
  read -r _
}

## 1. pixels baseline -- also defines the scene positions everything reuses.
run "1/4  ACT 3-camera (pixels only)" \
    "$B/transfer_flower_20260903_act/checkpoints/last/pretrained_model" "$PY"

REF=$(ls -1dt "$HERE"/rollouts/snapshots_* 2>/dev/null | head -1)
if [ -n "$REF" ]; then
  echo "[sweep] replicating scenes from $REF"
  REPL=(--replicate "$REF")
else
  echo "[sweep] no snapshots from run 1 -- later runs will NOT be paired"
  REPL=()
fi

## 2. the object-centric question: identical model + a 12-d centroid vector.
run "2/4  ACT 3-camera + env-state (object-centric)" \
    "$B/transfer_flower_20260903_act_envstate_50ep/checkpoints/last/pretrained_model" \
    "$PY" "${REPL[@]}"

## 3. camera ablation, trained rather than occluded at test time.
run "3/4  ACT 2-camera" \
    "$B/transfer_flower_20260903_act_2cam/checkpoints/last/pretrained_model" \
    "$PY" "${REPL[@]}"

## 4. SmolVLA needs the vla312 env (transformers 5 / hub 1.x); it has never
## been rolled out on hardware, so treat a failure here as untested plumbing
## rather than a result about the policy.
run "4/4  SmolVLA (env vla312 -- UNTESTED on hardware)" \
    "$B/transfer_flower_20260903_smolvla_50ep/checkpoints/last/pretrained_model" \
    "$PY_VLA" "${REPL[@]}"

echo
echo "############################################################"
echo "sweep complete -- summarise with:"
echo "    $PY summarize_rollouts.py --since $(date +%Y-%m-%d)"
echo "############################################################"
