#!/bin/bash
## GPU0 queue, rev 2 (2026-09-15 00:55).  The A4000 does 1.17 step/s, so the
## cube env-state run comes here instead of behind cube r2 on GPU1.
##   joint runs (running) -> EE-rgb + cube env-state together -> EE-envstate
source ~/miniconda3/etc/profile.d/conda.sh; conda activate gym_av312
cd /home/devi/giava/policy_training
while pgrep -f "job-name hull22_act_joint" > /dev/null; do sleep 120; done
until grep -q "CUBE PREP DONE" logs/cube_prep.log; do sleep 60; done
echo "[queue0] $(date) joint runs finished -- launching EE-rgb + cube env-state"
CUDA_VISIBLE_DEVICES=0 python train_real.py --root /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_v2_ee/20260914_234059 --policy act --successes-only --steps 50000 --num-workers 8 --job-name hull22_act_ee_rgb > logs/hull22_act_ee_rgb.log 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=0 python train_real.py --root /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/shape_sorter_envstate/20260907_204429_r2 --policy act --env-state --cameras right_wrist top_scene oak_left --pieces cube --successes-only --dagger-from 162 --steps 50000 --num-workers 8 --job-name shape_sorter_cube_20260915_act_envstate > logs/shape_sorter_cube_20260915_act_envstate.log 2>&1 &
wait
echo "[queue0] $(date) EE-rgb + cube env-state finished -- launching EE-envstate"
CUDA_VISIBLE_DEVICES=0 python train_real.py --root /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_v2_ee_envstate/20260914_234059 --policy act --env-state --successes-only --steps 50000 --num-workers 8 --job-name hull22_act_ee_envstate > logs/hull22_act_ee_envstate.log 2>&1
echo "[queue0] $(date) all GPU0 jobs done"
