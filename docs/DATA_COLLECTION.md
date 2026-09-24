# Data collection: what runs, what supports it, what was exploration

The reorg of 2026-09-12 (`da9bf2b`) split one 215-file directory into sibling
trees.  The rule it used is worth stating, because it is the rule that keeps
the tree honest:

> `data_collection_scripts/` holds **exactly the transitive closure of the
> entry points we actually run.**  If nothing imports a file and you don't
> reach for it every session, it belongs in `debug/` or `analysis/`.

One deliberate exception: `move_arms.py` is standalone but used constantly, so
it lives with the runtime.

```
av_aloha/
  data_collection_scripts/   the runtime — teleop, recording, replay, rollout
  ik/                        the solver + the study that chose it
  calibration/               camera/robot calibration (writes JSON)
  reconstruction/            pixels → metres (consumes that JSON)
  rgbd/                      reading RGB-D back out of recorded datasets
  debug/                     interactive probes, hardware bring-up
  analysis/                  dataset/rollout analysis, relabelling
  config/  launch/           ROS driver modes and launch files
```

`archive/` and `act_code/` are **gone** as of `75c1286` — the `av_aloha/README.md`
section describing `archive/` is stale and should be deleted.

---

## The five entry points

Everything else in `data_collection_scripts/` is imported by one of these.

| command | what it does |
|---|---|
| `python data_collection.py [ep]` | the teleop + recording loop. The big one (166 KB). |
| `python teleop.py` | teleoperation with no recording. Same control path, no dataset. |
| `python replay_episode.py` | replay a recorded episode on hardware. |
| `python rollout_policy.py` | run a trained checkpoint on the arms. |
| `python dataset.py --verify` | build / inspect / verify a LeRobot dataset. |

### `dataset_builders/` — offline, moved out 2026-09-21

These derive a new dataset from a recorded one.  They are exploration tooling,
not runtime, and they now live in their own tree with a `_giava_paths.py` shim:

```
build_envstate_dataset.py        object-centric env-state
build_mask_dataset.py            object-mask camera stream
build_relative_action_dataset.py per-step delta actions
build_ee_action_dataset.py       end-effector-space actions
build_eedist_dataset.py          gripper↔object distance feature
```

**What could not move, and why it matters.** The *feature extractors*
(`scene_features.py`, `wrist_features.py`, `eedist_features.py`) stay in the
runtime, because `rollout_policy.py` imports all three: a rollout has to
compute the same feature online that the builder computed offline, or the
policy sees a different input at inference than it was trained on.  Builder
and extractor are two halves of one thing.

`build_mask_dataset.py` had the same edge — `rollout_policy` imported
`mask_for` from it.  That rule now lives in `object_mask.py` in the runtime,
and both the builder and the rollout import it from there.

---

## The runtime, by layer

### Configuration and geometry

| file | what it is |
|---|---|
| `arm_config.py` | **the robot definition.** Arm names → driver name, model, joint names, EE link. All the named poses (`forward`, `rest`, `high`, `low`, `med`, `witness`, `forward_demo`, …) with the story of each in comments. |
| `data_col_config.py` | `TeleopConfig` — every runtime knob. 47 KB, mostly because each value carries why it is that value. |
| `paths.py` | where things are. Reach assets via `paths.ASSETS_DIR`, never `Path(__file__).parent / "assets"`. |
| `collision_modes.py` | the `--collision sphere\|capsule\|gjk` and `--table on\|off` switches. **Must run before `study_ik` is imported** — both read config at module import. |
| `jax_platform.py` | GPU/CPU backend selection, applied before any JAX import. |
| `transform_utils.py` | SE(3) helpers. |

### Control

| file | what it is |
|---|---|
| `robot_kinematics.py` | **pure kinematics** — `build_robot_model`, `compute_fk_and_ee`. No ROS, no Interbotix SDK, so offline tools can get FK without hardware. Split out of `robot_control.py` 2026-09-21. |
| `robot_control.py` | the driver seam. `create_and_configure_robots()`, `apply_profile_limits()` (reads the achieved velocity cap **off the live motors**), `sync_robot_state()`, `move_arms_together()`, SDK log filtering. |
| `study_ik.py` | the deployed coupled three-arm IK. Wraps `ik/` and owns the driver↔URDF frame conversion for the middle waist. |
| `teleop.py` | headset pose → EE target math, and the standalone teleop entry point. |
| `gripper.py` | gripper command path (`current_based_position`). |
| `servo_health.py` | per-tick watchdog off `joint_states`: stall gate, overload detection, `FaultRecorder` (20 s pre-fault buffer → `fault_logs/`). |
| `tube_mpc_hook.py` | the adapter between the `tube_mpc` package and this rig. `GIAVA_TUBE_MPC=1`. |

### Safety gates — the layer *after* the solver

These are hard yes/no on the final clamped command.  They need no gradients,
only a conservative distance.

| file | what it is |
|---|---|
| `capsule_gate.py` | inter-arm gate. Coarse capsules by default; `GIAVA_CAPSULE_GATE_FINE=1` promotes near-misses to exact GJK. |
| `gjk.py` | the exact convex distance algorithm. Iterative, data-dependent branching, no useful gradient — which is exactly why it can be a gate and can never be an IK residual. |
| `multi_capsule.py` | the fitted multi-capsule decomposition the gate uses. |
| `table_gate.py` | the tabletop z-floor gate. **The one that actually fires** — inter-arm blocks are rare in single-arm work, table blocks are routine. |

The distinction that makes this tree make sense:

```
IK soft cost   shapes how the motion FEELS      must be differentiable
               (sphere model, inscribed)         optimistic by ≤18 mm → margin 20 mm
GATE           decides WHERE YOU STOP            must be conservative
               (capsule, or exact GJK)           cannot use an optimistic model
```

### Sensing

| file | what it is |
|---|---|
| `camera_manager.py` | all camera I/O: RealSense D405s, the OAK stereo pair, worker threads, frame history, synchronised selection, intrinsics recording, rectification. See [`CAMERAS_AND_THREADS.md`](CAMERAS_AND_THREADS.md). |
| `gvlink_headset.py` | the Unity headset protocol (gvlink v3) — pose in, video out. |
| `webrtc_headset.py` | the older WebRTC transport. |
| `headset_link.py`, `headset_utils.py` | shared headset helpers. |
| `oak_preview.py`, `oak_stereo_calibrate.py`, `stereo_calib_live.py` | OAK preview and checkerboard stereo calibration. |
| `scene_features.py`, `wrist_features.py`, `eedist_features.py` | feature extractors for the object-centric datasets. |

### Recording

| file | what it is |
|---|---|
| `dataset.py` | `LeRobotDataset` wrapper + the async `EpisodeSaver` writer thread + `--verify`. |
| `log.py` | per-episode telemetry → `meta/robustness.jsonl`, and the config snapshot. |
| `scene_snapshots.py` | per-episode scene images for the ghost-alignment window. |
| `place_grid.py` | the placement lattice overlay (green/purple cells). |

### Task and operator tooling

`shape_sorter.py`, `auto_reset.py` (scripted pick-place on the px→EE fit),
`policy_driver.py` (policy stepping inside the teleop loop, for DAgger),
`canonicalize_waist.py`, `waist_travel.py`, `move_arms.py`.

### Documentation in-tree

`FRAMES.md` (coordinate frames — read this before touching anything spatial),
`TELEOP_MATH.md`, `HEADSET.md`, `SHAPE_SORTER.md`.

---

## What each tick does

The loop in `data_collection.py` runs against a 20 ms budget (50 Hz).
Achieved: **45.9 Hz median** across 141 recorded episodes.

```
 1  read joint_states                 (ROS subscriber, driver-published)
 2  servo health check                stall gate, overload, fault buffer
 3  read headset pose                 gvlink
 4  headset pose → EE targets         teleop.py: scaling, rate limits, enable gate
 5  COUPLED IK SOLVE (all 3 arms)     study_ik.py → pyroki/jaxls LM
 6  clamps                            per-joint step limit, driver limit margin
 7  [optional] tube-MPC filter        GIAVA_TUBE_MPC=1
 8  GATES                             capsule/GJK inter-arm, table z-floor
 9  publish goal positions            interbotix_xs_sdk
10  grab synchronised camera frames   from the worker threads' history
11  append frame to episode buffer    if recording
12  sleep to budget                   overrun → sleep is zero, logged
```

Step 5 is the cost centre: mean 13.8 ms, p95 23.9 ms, 12.8 % of ticks overrun.
Steps 10–11 are cheap *because* the cameras are on their own threads and the
episode writer is on its own thread — neither blocks the tick.

Everything in that list is counted in `meta/robustness.jsonl` per episode:

```
measured_fps  overruns  ik_solve_ms{n,mean,p95,max}  ik_overrun_ticks
capsule_gate_blocks  capsule_gate_scaled  table_gate_blocks  table_gate_scaled
driver_clamp_ticks  driver_clamp_joints  clamp_saturated_ticks
tube_fallback_ticks  cam_sync_spread_s  policy_ms
teleop_enable_count  teleop_disable_count
dagger_human_frames  dagger_policy_frames
collision_mode  table_mode  arms  cameras
```

That list exists because of the fps bug: a rate written into metadata is a
claim; the achieved rate is a measurement, and only the measurement is stored.

---

## The supporting trees

### `ik/` — a hard runtime dependency, not research

`study_ik.py` imports five modules from it at runtime: `robot_model`,
`baseline`, `variants`, `collision_models`, `table_collision`.  **Treat those
five as production.**

`ik/study/` is the ablation that chose the weights and the collision model.
Its *scripts* are not run day to day, but `ik/study/results/` holds the frozen
model inputs the runtime reads (`multi_capsule_decomposition.json`,
`sphere_decomposition.json`, `sphere_pair_pruning.npz`, `hull_leak.json`), so
it is not disposable either.  `ik/benchmark/` is the earlier, self-contained
PyRoki performance benchmark.

### `calibration/` → `reconstruction/` — the geometry chain

```
world coords → robot pose → camera pose → calibrated/synced images
                                → camera extrinsics → 3D reconstruction
              └──── calibration/ ────┘  └── reconstruction/ ──┘
```

`calibration/` establishes and measures; it writes JSON.  `reconstruction/`
consumes that JSON and does geometry; it never writes or repairs a
calibration — a missing extrinsic raises and names the command that would
produce it.  `reconstruction/selftest.py` is 33 checks against synthetic
ground truth the code under test cannot see.

GIAVA has no TCP frame, and solve-before-anchor is non-negotiable.

### `debug/` — interactive probes

27 tools, none imported by anything.  Hardware bring-up (`dynamixel.py`,
`jog_waist.py`, `verify_middle_urdf.py`), frame work (`frame_calibrate.py`,
`measure_waist_reclock.py`, `shift_waist_frame.py`), headset
(`headset_frame_probe.py`, `probe_frame_age.py`), cameras (`oak_pose_finder.py`,
`oak_quality_debug.py`), collision (`capsule_gate_view.py`,
`measure_hull_leak.py`), and `tube_mpc_bench.py`.

Each carries `_giava_paths.py`, which walks up for `giava.urdf` to find the
repo.  It does **not** count `parents[N]` — that idiom was in this tree at two
different depths and both were wrong the moment a directory moved.

### `analysis/` — and the duplication worth fixing

There are **two** analysis directories and they are not the same thing:

| | contents | status |
|---|---|---|
| `av_aloha/analysis/` | `ab_report`, `analyze_replays`, `audit_idle_frames`, `manifest_training_runs`, `measure_latency`, `placement_map`, `relabel_episode(s)`, `review_rollouts`, `summarize_rollouts`, `true_fps` | committed (`e30bf4e`) |
| `data_collection_scripts/analysis/` | `vlm_annotate.py`, `vlm_annotate_demos.py`, `score_grasp.py`, `fit_px2ee.py`, `placement_report.py`, `episode_features.py`, `split_cameras.py`, `reencode_rollouts.sh` | **untracked**, written 2026-09-19/20 |

The second one is the newer labelling/scoring work.  It should move into
`av_aloha/analysis/` under the reorg's own rule, or the rule should be
restated.  Either way it needs committing — right now it exists only on this
disk.

---

## Curiosity branches — real work, not on the critical path

These are worth labelling as such rather than deleting.  None of them is
imported by an entry point.

| where | what it was | verdict |
|---|---|---|
| `reconstruction/pointcloud.py` | dense tabletop point cloud, table/object segmentation, curved-surface fit (2026-09-20, three commits) | newest exploration; geometrically sound, no consumer yet |
| `rgbd/` | reading RGB-D back out of recorded datasets | useful, occasional |
| `ik/benchmark/` | the pre-study PyRoki performance benchmark | superseded by `ik/study/`, self-contained, keep as provenance |
| `ik/study/playground.py`, `view_*.py` | Viser interactive solver/collision viewers | the best way to *see* what the solver does; keep |
| `debug/make_synth.py` | synthetic-data generation | one-off |
| `analysis/vlm_annotate*.py` | VLM-based episode annotation | active, uncommitted |
| `gaze_av_aloha/`, `pretrain/` | the gaze / foveated-ViT line (Jinyu, Ian) | separate research programme, not yours; `pretrain/` is MAE pretraining for it and nothing imports it |
| `block_square/` (repo root) | a 1-episode / 65-frame HF dataset clone with its own `.git`, remote `huggingface.co/datasets/deviamar/block_square` | stray clone, untracked. The *real* block_square data is under `data_collection_scripts/old_dataset/` |
| `pyroki_ik_script.py` (repo root, 58 KB) | the original single-file IK exploration | superseded by `ik/`; still referenced by `tube_mpc/{README,MATH}.md` as the source of the GJK/sphere pipeline for Phase 2 |
| `curobo/` | read for collision/speed ideas | never deployed; the VOXEL sphere fit was borrowed and reimplemented |

---

## Stale documentation to fix

Three files describe a repo that no longer exists:

1. **`README.md`** (repo root) names `reset_arm.py`, `move_arm.py`,
   `record_episodes.py`, `robot_factory.py`, `arm_controller.py`, `config.py`,
   `teleop_utils.py`, `upload_dataset.py` — **none of which exist.**  The real
   entry points are the five above.
2. **`av_aloha/README.md`** documents `archive/` and lists `act_code/` as an
   entry point.  Both were deleted in `75c1286`.
3. **`ACT_MODIFICATIONS.md`** is about `modeling_act copy.py`, the YOLO-feature
   ACT fork, also deleted in `75c1286`.  Keep it only as a record of what that
   variant did; say so at the top.

`instruction.md` and `experiments.txt` are the gaze-project runbook and its
sweep commands — accurate for `gaze_av_aloha/`, unrelated to the real-robot
work.  Label them, don't delete them.


---

## Import conventions (2026-09-21)

Two idioms were removed across 24 files because they had never executed:

- **`if __package__: from .x import y` / `else: from x import y`** — 42 blocks.
  `data_collection_scripts/` has no `__init__.py` and nothing in the repo
  imports it as a package, so `__package__` was always `''` and the relative
  branch was dead.  Entry points are `python teleop.py`; the absolute form is
  the only one that ever ran.
- **`try: import rospy / except ImportError: rospy = None`** in
  `robot_control.py` — these deferred an import error to first use.  They were
  load-bearing only because `robot_control.py` mixed two layers: the hardware
  seam *and* pure kinematics that offline tools wanted.  Splitting
  `robot_kinematics.py` out removed the reason for them.

What remains, and should: `try/except ImportError` where the handler does real
work (the `sys.path` setup in `study_ik.py`, `place_grid.py`,
`rollout_policy.py`), and genuinely optional dependencies (loguru).

**The rule going forward:** absolute imports everywhere in these flat trees;
`import _giava_paths` at the top of any tool outside `data_collection_scripts/`.
Guard an import only when the module is genuinely optional — not to paper over
a module that is doing two jobs.
