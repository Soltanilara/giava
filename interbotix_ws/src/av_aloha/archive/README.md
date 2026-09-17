# archive — pending deletion

Moved here 2026-09-12 out of `data_collection_scripts/`.  Nothing in the live
tree imports anything in this directory; that is checked, not assumed.  Layout
mirrors where each file came from, so provenance survives.

**This is a holding pen, not a library.** Delete it when you're satisfied
nothing here is worth keeping — `git rm -r archive/`.  Everything is in git
history regardless.

## Why each group is here

### `old code/` — sim era
`sim_env.py`, `record_sim_episodes.py`, `diff_ik.py`, `grad_ik.py`,
`image_recorders.py`, `sleep.py`, `launch_robot.sh`.  MuJoCo sim and the
pre-PyRoki IK experiments.  Superseded by `ik/` and the real-robot recorders.
Last touched July.

### `debugging_scripts/` — May–August one-offs
Nothing newer than 2026-08-18, most of it May–June: curobo experiments
(`curobo_expl.py`, `curobo_teleop.py`), single-arm keyboard IK, an OpenCV
smoke test, a YOLO scratchpad, gripper-value fixups, a commented-out file.
The debug tools that are still current are in `debug/`.

### `test_scripts/` — April/May smoke tests
Joint/gripper readouts, ZED and OAK connection tests, sim-reward tests, the
early WebRTC sender/receiver pair.  Written against the sim and the old
headset path.  The three that are still current (`make_synth.py`,
`test_replay_loop.py`, `test_replay_modes.py`, all 2026-08-21) went to
`debug/`.

### `act_code/` — superseded ACT experiments
`act_experiments.py`, `act_grid_search.py`, `act_rollout_experiment.py`,
`act_rollout_gridsearch.py`, `act_train_rgb+yolo.py`.  The June/August
gridsearch generation, replaced by `act_train_rgb.py` /
`act_train_objcentric.py` and their rollout counterparts, which stayed in
`data_collection_scripts/act_code/`.  Note their hardcoded `zoom=1.8`
assumption about the June recorder.

### Migrations that have already run
`convert_depth_datasets.py`, `fix_rgb_only_metadata.py`,
`migrate_checkpoints.py`, plus the reports they wrote
(`depth_conversion_report.json`, `rgb_only_fix_report.json`).  One-shot format
conversions to LeRobot v3.0 / v0.6.0 external normalization.  Keep the reports
if you want the record of what was converted.

### One-time hardware calibrations, already folded in
`set_waist_homing_offset.py` ("one-time fix"), `make_middle_offsets.py`,
`joint_range_calibration.py`.  The 2026-08-21 fold moved these offsets into
`giava.urdf` itself; `middle_joint_offsets.json` has been intentionally empty
since.  `debug/verify_middle_urdf.py` is the tool that still matters here.

### Superseded odds and ends
- `constants.py` — sim-era task configs and `aloha.xml` paths.  Its one live
  consumer needed a single scalar (`REAL_DT`), now inlined in
  `debug/headset_control.py`.
- `middle_arm_control.py`, `upload_dataset.py`, `data_analysis.py` (hardcoded
  absolute dataset path), `oak_performance.py`, `oak_to_png.py`,
  `test_oak.py` (the top-level copy; `test_scripts/test_oak.py` is the other).
- `act_rollout_gridsearch.py` at the root is the **top-level** copy;
  `act_code/act_rollout_gridsearch.py` is the different, later one.

### Not code
`pyzed-4.1-*.whl` (852K, ZED SDK — we're on OAK now), `usbreset` + `usbreset.c`
(the binary was committed), `teleop_timing.log`, `top_frame_overlay.png`,
`placement_grid_check.png`.
