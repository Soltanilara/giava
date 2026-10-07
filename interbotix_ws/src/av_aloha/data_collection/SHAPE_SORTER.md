# Shape-sorter insertion (active vision) — workflow

Pick one piece among three on the table and insert it into its hole in the
sorter box, with the middle camera arm following the headset. Task
definition, per-piece detector and env-state features live in
`shape_sorter.py`; everything below just calls it.

## 0. Pieces

`shape_sorter.CLASSES` mirrors the box's top face: **green cube, red
triangle, blue flower** (one color per shape). Same-color/different-shape
pieces from the set are indistinguishable to the detector — keep them off the
table during collection; use them as rollout distractors.

## 1. Calibrate the holes (once per box/camera placement)

Record one throwaway episode (or grab any top_scene frame with the box in
place), then:

```bash
python measure_targets.py --dataset dataset/lerobot/shape_sorter/<run>
# click ROI corner 1, ROI corner 2, then each hole in the printed order; s = save
```

Writes `assets/shape_sorter_targets.json` + a `.png` overlay and prints which
pieces the detector currently sees. Re-run if the box or camera moves.

## 2. Collect

```bash
python data_collection.py --mode right_av --task shape_sorter --target cube --episodes 30
#   ... episodes ...   `target triangle` at the prompt switches piece between episodes
#   omit --target to be asked for the piece at startup; --episodes N is a soft
#   per-piece goal (the save line reads "12/30 cube", at N it reminds you to
#   switch or q); q parks all arms at --quit-pose (rest) and ends the session
```

- Session keys: `r` record, `ss`/`sf` save, `d` discard, `i` re-park, `sync`
  (re-read the arms after moving them from another terminal, no motion),
  `q` quit to `--quit-pose`, `qh` quit **holding position** (next launch with
  the same start pose starts instantly). `--start-pose` takes a shared name
  or per-arm `right=forward,middle=far_scene`.
- `--mode right_av` = right arm (6 + gripper) + middle camera arm (7) → 14-dim
  state/action; `left_av` is the mirror. Cameras: right_wrist, top_scene,
  low_scene, oak_left/oak_right (headset stereo).
- The per-frame task string is `insert the red cube into the square hole`
  etc. — that is how `build_envstate_dataset.py` and `rollout_policy.py` know
  which piece an episode is about. Every episode in a shape_sorter run must be
  recorded with a target.
- Verdicts as before: `ss` success, `sf` failure (kept, labelled in
  `episode_outcomes.jsonl`), `d` discard. Train on successes with
  `train_real.py --successes-only`; an episode with a recovery that ends in a
  successful insertion is a success.
- Randomize all three piece positions every episode; keep the box fixed.

## 3. Build the env-state variant

```bash
python build_envstate_dataset.py --src dataset/lerobot/shape_sorter/<run> \
    --dst dataset/lerobot/shape_sorter_envstate/<run> --scene shape_sorter
python build_envstate_dataset.py --src dataset/lerobot/shape_sorter/<run> \
    --dst dataset/lerobot/shape_sorter_onehot/<run> --scene shape_sorter_onehot
```

- `shape_sorter` (8-d, unnormalized, ~[-1, 1]): target piece `cx, cy, found,
  size`, its hole `cx, cy`, and the piece→hole vector. No class one-hot on
  purpose — pure geometry, so a held-out shape sees an in-distribution vector.
- `shape_sorter_onehot` (4-d): the target class one-hot and nothing else —
  "which piece", not "where". This is the **baseline** for pooled training:
  with three pieces on the table a pixel-only policy cannot know which one the
  episode is about, so plain pixels is ill-posed for a multi-piece dataset.
  A held-out class has no one-hot, so this condition is fine-tune-only.

## 4. Train (env `gym_av312`)

```bash
cd policy_training
CAMS="--cameras right_wrist top_scene oak_left"
# base policies on the two training pieces (successful episodes only)
python train_real.py --root <onehot_run>   --policy act --env-state $CAMS --pieces cube triangle --successes-only --job-name ss_onehot_base
python train_real.py --root <envstate_run> --policy act --env-state $CAMS --pieces cube triangle --successes-only --job-name ss_geom_base
# held-out piece: zero-shot = roll out ss_geom_base with --target flower, no training.
# fine-tune = continue from the base checkpoint on the pooled data (base pieces + the
# 10 new ones), a few k steps.  Pooled, not the 10 alone: lerobot re-derives the
# normalization stats from the fine-tune dataset, and 10 episodes' stats drift.
python train_real.py --root <envstate_run> --policy act --env-state $CAMS --successes-only \
    --init-from outputs/train/ss_geom_base/checkpoints/last/pretrained_model \
    --steps 5000 --save-freq 1000 --job-name ss_geom_ft_flower
# control: the 10 new episodes from scratch
python train_real.py --root <envstate_run> --policy act --env-state $CAMS --pieces flower --successes-only --steps 20000 --job-name ss_geom_scratch_flower
```

`--pieces` selects episodes by the task string in `episode_outcomes.jsonl`;
`--init-from` loads config + weights from a checkpoint (features must match).

## 5. Roll out and score

```bash
python rollout_policy.py --checkpoint <ckpt> --mode right_av --target cube --score [--engage]
```

Keys during an episode: `p` pause/resume (arm holds, paused time does not
count, the queued chunk is dropped on resume; counted as an intervention),
`s` end the episode early (then score it), `q` quit. `--seconds 40
--stall-seconds 5` gives the policy time without paying for stalls. After each
episode the auto-metrics draft the stage letters (ENTER accepts, letters
override, `-` = none, `x` = discard a rig-fault trial); flags `c` = corrected
by hand, `w` = grasped the wrong piece.

`--target` selects the scene: env-state features are computed from top_scene
every tick with the same extractor (required for `--env-state` checkpoints;
the rollout refuses a checkpoint whose env dim does not match), and scoring
uses the shape-sorter stages `a=approach g=grasp t=transport h=hole
(hovered over the CORRECT hole) i=inserted`. `summarize_rollouts.py` tables
them per stage set with Wilson intervals.

## Experiment

One run: two training pieces (~30 demos each) + one held-out piece (~10).
Conditions on the same episodes:

| condition | env input | tests |
|---|---|---|
| one-hot | which piece (4-d) | pooled baseline; fine-tune onto held-out |
| geometry | which + where (8-d) | in-dist vs one-hot; **zero-shot** on held-out; fine-tune |
| scratch-10 | geometry, 10 demos only | control for the fine-tune |

Score every rollout at 10-15 trials per cell. "Hole" rate is where the
object-centric question lives; "inserted" is a precision question no input
representation fixes.
