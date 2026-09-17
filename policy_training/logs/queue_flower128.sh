#!/bin/bash
## The env-state half of the 2x2, queued behind the RGB half.  Four ACT runs
## at ~8 GB each do not fit on one 24 GB card, so they go two at a time --
## and both pairs then share a GPU identically, which keeps wall-clock
## comparable between them (it does not affect the results, only the timing).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate gym_av312
cd /home/devi/giava/policy_training
while pgrep -f "job-name flower128_act_joint_rgb|job-name flower128_act_ee_rgb" > /dev/null; do sleep 120; done
echo "[queue] $(date) RGB pair finished -- launching env-state pair"
CUDA_VISIBLE_DEVICES=0 python train_real.py --root /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_merged_envstate/20260916_centre_plus_rim --policy act --env-state --successes-only --steps 60000 --num-workers 8 --job-name flower128_act_joint_envstate > logs/flower128_act_joint_envstate.log 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=0 python train_real.py --root /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_merged_ee_envstate/20260916_centre_plus_rim --policy act --env-state --successes-only --steps 60000 --num-workers 8 --job-name flower128_act_ee_envstate > logs/flower128_act_ee_envstate.log 2>&1 &
wait
echo "[queue] $(date) all four flower128 runs done"
