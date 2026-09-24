# GIAVA — system documentation

Written 2026-09-21 by reading the code and the measurements in this repo.
Every number here has a file behind it.

| document | what it answers |
|---|---|
| [`SUBMODULES.md`](SUBMODULES.md) | What are pyroki, curobo, lerobot, gym_av_aloha and the Interbotix SDK, which files matter in each, and where they touch this work. Includes the curobo-vs-pyroki analysis: why collision was broken, why it is slow, and the three ways to fix the speed. |
| [`DATA_COLLECTION.md`](DATA_COLLECTION.md) | The runtime tree after the reorg — the five entry points, what each core file does, what one tick does, and which directories are exploration rather than critical path. |
| [`IK.md`](IK.md) | Solving IK here: the formulation, what the ablation study found, the frame problem, the collision story end to end, the gate layer, and what it costs in production. |
| [`TUBE_MPC.md`](TUBE_MPC.md) | Tube MPC in five steps, the GIAVA integration decisions, and the two-config-file bug that is the best case study in the repo. |
| [`CAMERAS_AND_THREADS.md`](CAMERAS_AND_THREADS.md) | The eleven threads, how a frame gets from sensor to episode, frame synchronisation and its honest limit, and why `q` is not Ctrl-C. |
| [`CHANGELOG_JULY_ONWARD.md`](CHANGELOG_JULY_ONWARD.md) | What changed since July and why, including the cross-cutting fixes that are not any one commit. |

Also in-tree, and still the primary sources:

- `LESSONS.md` — April–September narrative, written from measurements.
- `tube_mpc/MATH.md` — the full derivation and proof sketch.
- `ik/study/{BASELINE,RESULTS_FINAL,COLLISION_STUDY,TRAJECTORIES}.md` — the raw studies.
- `data_collection_scripts/{FRAMES,TELEOP_MATH,HEADSET,SHAPE_SORTER}.md`.
- `reconstruction/README.md`, `calibration/README.md`, `rgbd/README.md`.

## The system in one picture

```
  VR headset ──pose──> teleop.py ──EE targets──┐
                                               │
                        ┌──────────────────────▼──────────────────────┐
                        │  study_ik.py — ONE coupled solve, 3 arms     │
                        │    pose ×3 (50/10) + limits + smoothing 0.05 │
                        │    + centering 0.5 + 180-sphere collision    │
                        │    + tabletop half-space                     │
                        │    pyroki / jaxls LM, JIT, GPU               │
                        └──────────────────────┬──────────────────────┘
                                               │  13.8 ms mean
                        clamps (step, driver limits)
                                               │
                        [tube MPC filter]  GIAVA_TUBE_MPC=1
                                               │
                        GATES: capsule / exact GJK  +  table z-floor
                                               │
                        interbotix_xs_sdk ──U2D2──> Dynamixel servos
                                               │
  cameras ──worker threads──> frame history ──sync──> episode buffer
                                               │
                        EpisodeSaver thread ──> LeRobot dataset
```

45.9 Hz achieved against a 50 Hz target.  Three safety layers with three
different time horizons: the IK cost is reactive and soft, tube MPC is
predictive over a horizon, the gate is instantaneous and hard.

## Known stale documentation

Three files describe a repo that no longer exists and should be fixed or
labelled:

1. `README.md` (root) names eight scripts that do not exist.
2. `av_aloha/README.md` documents `archive/` and `act_code/`, both deleted in `75c1286`.
3. `ACT_MODIFICATIONS.md` is about the YOLO ACT fork, also deleted in `75c1286`.

`instruction.md` and `experiments.txt` are the **gaze project's** runbook and
sweep commands — accurate for `gaze_av_aloha/`, unrelated to the real-robot work.
