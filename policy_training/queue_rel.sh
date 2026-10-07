#!/usr/bin/env bash
## Sequential queue. 62 GB host RAM holds two ACT runs at useful speed, not
## three (measured 2026-09-23 23:10: three at once -> GPU0 0%, 2.7-5.3 s/step).
## Stages: absolute head pair -> env-state-as-input arm.  (Relative pair moved to lara94, 2026-09-24.)
set -u
cd /home/devi/giava/policy_training
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate gym_av312 2>/dev/null
B=/home/devi/giava/interbotix_ws/src/av_aloha/data_collection/dataset/lerobot
LOG=logs/queue_rel.log
running() { ps -eo cmd | grep -q "^python train_.*job-name $1"; }
echo "$(date) queue armed; waiting for sorter166_head_abs_*" >> $LOG
while running sorter166_head_abs_; do sleep 120; done
echo "$(date) relative pair runs on lara94 (moved 2026-09-24 08:20); skipping to env" >> $LOG
echo "$(date) stage 3: env-state input arm" >> $LOG
CUDA_VISIBLE_DEVICES=1 nohup python train_real.py \
  --root $B/shape_sorter_envstate/20260907_204429_r2 --env-state \
  --policy act --cameras right_wrist top_scene oak_left --successes-only \
  --chunk-size 100 --n-action-steps 100 --steps 60000 --batch-size 8 --num-workers 12 --save-freq 10000 \
  --job-name sorter172_act_env > logs/sorter172_act_env.log 2>&1 &
echo "$(date) env pid $!" >> $LOG
