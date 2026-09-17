# av_aloha — directory map

Reorganised 2026-09-12.  `data_collection_scripts/` had grown to ~215 Python
files, of which 40 were actually reachable from the things we run every day.
The other 175 were research phases, one-off probes, migrations that had already
been applied, and sim-era code from April–July.  They are still here — nothing
was deleted — but they are no longer in the way.

```
av_aloha/
  data_collection_scripts/   the runtime: teleop, recording, replay, rollouts
  ik/                        the IK solver, and the study that produced it
  calibration/               camera/robot calibration pipeline (PHASE 1-8)
  reconstruction/            stage two: pixels to metres
  rgbd/                      reading RGB-D out of recorded datasets
  debug/                     interactive probes and hardware debug tools
  analysis/                  dataset and rollout analysis, relabelling
  archive/                   dead code, kept only until you delete it
```

## Where data lives

Assets are central, at the **repo root**, not in any package:

| path | what |
| --- | --- |
| `<repo>/assets/` | the one asset location: `meshes/` (robot STLs), sim XMLs, `shape_sorter_targets.json`, `top_scene_static_blue.npy` |
| `<repo>/aloha_assets` | **symlink** to `assets/meshes` — this is how `giava.urdf`'s `filename="aloha_assets/*.stl"` resolves.  Do not delete it |

Reach assets from code via `paths.ASSETS_DIR`, never `Path(__file__).parent /
"assets"`.

Generated run data sits with whatever produces it, and is gitignored:

| path | written by | when |
| --- | --- | --- |
| `calibration/data/oak_calib/` | `teleop.py --cameras oak` | OAK stereo checkerboard captures |
| `debug/bench_logs/` | `debug/tube_mpc_bench.py` | every bench run |
| `data_collection_scripts/fault_logs/` | `servo_health.py` (`FaultRecorder`) | on a watchdog trip — 20 s pre-fault buffer |
| `data_collection_scripts/rollouts/snapshots_<tag>/` | `rollout_policy.py` | **every run by default**; `--no-snapshots` turns it off |
| `data_collection_scripts/trajectories/logs/` | `data_collection.py` when `TRACK_LOG` | teleop tracking logs |

## What "in use" means

`data_collection_scripts/` now holds exactly the transitive closure of the
entry points we actually run:

| entry point | what it is |
| --- | --- |
| `data_collection.py` | the teleop + recording loop |
| `teleop.py` | teleoperation without recording |
| `replay_episode.py` | replay a recorded episode on hardware |
| `rollout_policy.py` | run a trained policy |
| `dataset.py` | LeRobot dataset build / verify |
| `build_envstate_dataset.py` | object-centric env-state datasets |
| `act_code/act_train_*.py`, `act_code/act_rollout_*.py` | ACT training and rollout |

Everything else in that directory is imported by one of those, with one
deliberate exception: `move_arms.py` is a standalone tool used constantly
during collection and rollouts, so it lives with them rather than in `debug/`.

If you add a file here and nothing imports it and you don't reach for it every
session, it belongs in `debug/` or `analysis/`.

## ik/

`ik/` is the solver.  `data_collection_scripts/study_ik.py` imports five
modules from it at runtime — `robot_model`, `baseline`, `variants`,
`collision_models`, `table_collision` — so **`ik/` is a hard dependency of data
collection**, not research code.  Treat those five as production.

`ik/study/` is the ablation study that chose the solver's weights and collision
model (`RESULTS_FINAL.md`, `COLLISION_STUDY.md`).  `ik/study/results/` holds the
frozen model inputs the runtime reads — `multi_capsule_decomposition.json`,
`hull_leak.json` — so the directory is not disposable either, even though its
scripts are not run day to day.  `ik/benchmark/` is the earlier PyRoki
performance benchmark and is self-contained.

## Imports across directory boundaries

The tools outside `data_collection_scripts/` are run directly
(`python3 debug/move_arms.py`), so Python puts only their own folder on
`sys.path`.  Anything they import from the runtime — `arm_config`,
`robot_control`, `camera_manager` — needs help.  Each folder carries a
`_giava_paths.py` that supplies it:

```python
import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import arm_config
```

It finds the repo by walking up for `giava.urdf`, the same marker
`data_collection_scripts/paths.py` uses, and honours `GIAVA_ROOT`.  It does not
count `parents[N]` — that idiom was in this tree at two different depths and
both were wrong the moment a directory moved.  Please don't reintroduce it.

`calibration/` predates the shim and uses its own `SCRIPTS_DIR` /
`_find_repo_root()` in `common.py`, to the same effect.

## archive/

`archive/` is a holding pen, laid out by where each file came from
(`archive/act_code/`, `archive/test_scripts/`, `archive/debugging_scripts/`,
`archive/old code/`).  See `archive/README.md` for what is in it and why.
Nothing in the runtime imports anything from it — that is checked, not assumed.
Delete it when you are satisfied.
