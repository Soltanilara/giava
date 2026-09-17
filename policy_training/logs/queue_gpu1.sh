#!/bin/bash
## GPU1 queue, rev 2 (2026-09-15 01:05): after the cube r2 fine-tune, SmolVLA
## on the SAME 22 hull episodes as the four ACT variants -- the pretraining
## arm of the structure-vs-pretraining question.  10k steps: the 50-episode
## SmolVLA run did 1.52 s/step on the 3090; the A4000 is slower and this has
## to be done by ~20:00.  vla312 env (transformers), not gym_av312.
cd /home/devi/giava/policy_training
while pgrep -f "job-name shape_sorter_cube_20260915_act_dagger_r2" > /dev/null; do sleep 120; done
echo "[queue1] $(date) cube r2 finished -- launching SmolVLA hull22"
CUDA_VISIBLE_DEVICES=1 /home/devi/miniconda3/envs/vla312/bin/python train_real.py --root /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_v2/20260914_234059 --policy smolvla --cameras right_wrist top_scene low_scene --successes-only --steps 10000 --save-freq 2500 --batch-size 8 --num-workers 6 --job-name hull22_smolvla > logs/hull22_smolvla.log 2>&1
echo "[queue1] $(date) SmolVLA exit code $?"
