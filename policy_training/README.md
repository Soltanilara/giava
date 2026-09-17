# Real-robot policy training (ACT / Diffusion / SmolVLA)

Train on datasets recorded by `data_collection.py` (LeRobot v3 format, one
run directory per session under
`interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/<task>/<timestamp>/`).

## Quick start

```bash
conda activate gym_av312
cd /home/devi/giava/policy_training

# ACT, all cameras, all episodes
python train_real.py --root <dataset_run_dir> --policy act

# Diffusion, 3 cameras, only episodes saved with `ss`
python train_real.py --root <dataset_run_dir> --policy diffusion \
    --cameras left_wrist right_wrist top_scene --successes-only

# Resume after an interruption
python train_real.py --root <dataset_run_dir> --policy act --resume
```

Checkpoints land in `outputs/train/<task>_<policy>/checkpoints/`.

## SmolVLA (fine-tune from `lerobot/smolvla_base`)

```bash
conda activate vla312          # NOT gym_av312 -- see below
CUDA_VISIBLE_DEVICES=0 python train_real.py --root <dataset_run_dir> --policy smolvla \
    --cameras right_wrist top_scene low_scene --steps 20000 --save-freq 5000 --batch-size 8
```

`--policy smolvla` loads the released checkpoint's config, weights and
processor pipelines and overrides only the I/O features; lerobot replaces the
SO-100 normalization stats with this dataset's. The VLM stays frozen
(`train_expert_only=True`), so only the ~100M-param action expert + state
projection train. The language prompt is the dataset's `task` string -- for
single-task runs it is a constant and irrelevant; give multi-object datasets a
real instruction at collection time.

**Why a separate env:** SmolVLA needs `transformers>=5.4`, which needs
`huggingface-hub>=1.5`; `gym_av312` pins hub 0.36 for the data-collection
stack and must not be upgraded. `vla312` is a clone of `gym_av312` with
`transformers 5.5 / huggingface-hub 1.30 / num2words` added (2026-09-06).
Note `conda create --clone` left pip broken in the clone; it was repaired with
`get-pip.py --force-reinstall`.

## Why not plain `lerobot-train`

The recorder stores per-camera/per-arm float64 epoch timestamps and EE poses
in every frame (for sync analysis and replay). lerobot's factory would feed
all of those to the policy as state inputs — epoch timestamps (~1.7e9) would
poison normalization. `train_real.py` explicitly restricts inputs to
`observation.state` + chosen cameras (add `--use-ee-pose` to opt EE poses in)
and hands everything else to lerobot's own training loop unchanged.

## One dataset, both policies

Yes — the same dataset trains both. Actions are absolute joint positions +
gripper for both; only the chunking differs and both derive it from the
dataset's own fps:

| | ACT | Diffusion |
|---|---|---|
| action horizon | 2 s chunk (`2*fps` ticks), executed open-loop | 64-tick horizon, executes 32 then re-plans |
| obs history | 1 step | 2 steps |
| steps (default) | 100k | 100k |

## Expected wall-clock (RTX 3090, 480×640 images)

Roughly, for 30–50 episodes (~40–80k frames):

- **ACT, 3 cameras, batch 8**: ~1.5–2.5 it/s → 100k steps ≈ **11–18 h**;
  usable checkpoints usually appear by 30–50k (~4–8 h). With all 6 cameras,
  roughly double. Cut `--steps` to 60k for a first pass.
- **Diffusion, 3 cameras, batch 8**: similar per-step cost, but typically
  needs the full 100k+ → plan for **overnight to ~1 day**.

Rule of thumb: launch in the evening, evaluate the 30–50k and final
checkpoints the next day. Train the first policy after ~30 episodes as a
pipeline smoke test — don't wait for the full dataset.

## Deployment consistency (matters as much as training)

At inference, reproduce the collection-time conditions:

- same fps (`GIAVA_CONTROL_HZ`), same `moving_time` / clamps / command path
  as `data_collection.py` (read the dataset's `teleop_config.json`);
- frames chosen by the same nearest-timestamp selection
  (`select_synchronized_frames`), not latest-wins;
- absolute joint-position actions sent through the same driver-feasibility
  clamp;
- ACT: execute the chunk at dataset fps; optionally enable temporal
  ensembling later (requires `n_action_steps=1` in lerobot).

## Related tools

- `../interbotix_ws/src/av_aloha/data_collection_scripts/measure_latency.py`
  — offline latency/sync report for any recorded run (tick jitter, camera
  sync + staleness, actuation lag, optional visual lag).
- `<run_dir>/meta/robustness.jsonl` — per-episode quality record (overruns,
  clamp/gate counts, camera sync spread, IK timing) written at save time.
- `<run_dir>/teleop_config.json` — the resolved teleop config + GIAVA_* env
  that produced the run.
