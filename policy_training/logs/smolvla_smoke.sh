#!/bin/bash
# Wait for the 50ep ACT run (pid 759193, GPU0) to exit, then smoke-test SmolVLA
# fine-tuning: 20 steps, checkpoint at 10, so from_pretrained / forward /
# backward / save are all exercised before committing GPU0 to a real run.
while kill -0 759193 2>/dev/null; do sleep 20; done
echo "ACT 50ep exited at $(date)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cd /home/devi/giava/policy_training
rm -rf outputs/train/smoke_smolvla
CUDA_VISIBLE_DEVICES=0 /home/devi/miniconda3/envs/vla312/bin/python train_real.py \
  --root /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/20260903_214256 \
  --policy smolvla --cameras right_wrist top_scene low_scene \
  --steps 20 --save-freq 10 --batch-size 8 --num-workers 4 \
  --job-name smoke_smolvla
echo "SMOKE EXIT CODE $?"
