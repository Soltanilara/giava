# What changed since July 2026

Reconstructed from the git history and the measurements in the repo, not from
memory.  Roughly: in July the loop ran at 1.7 Hz with 32 mm of position error
and no collision handling at all.  It now runs at 45.9 Hz with 0.25 mm of
position error, a validated collision model, three safety layers, and per-
episode telemetry on everything.

The headline, measured:

| | May–July | now |
|---|---|---|
| control rate | 1.7 Hz | **45.9 Hz** median |
| IK solve | 167.6 ms | **13.8 ms** mean |
| position error | 32 mm median, 470 mm max | **0.25 mm** max |
| orientation error | 9.2° median, 126° p95 | **0.02°** max |
| worst self-collision clearance | −157 mm (blind) | **+3 mm**, gated |
| what the dataset records about itself | a hardcoded `fps` | 25 measured fields per episode |

---

## July — modularisation, and the IK rewrite begins

**`de05030`, `15175ae`, `0140611`** — data collection made modular: left,
right and bimanual configurations from one code path instead of three copies.

**`406277f`, `3c65f4e`** — datasets stopped being tracked by git.  Overdue;
`old_dataset/` alone is 51 GB with zero tracked files, so without the ignore
rule git rescanned it on every `status` and a stray `git add -A` would try to
stage all of it.

**`4ace0b7` "Latest commit (IK)"** — 4,470 lines.  The URDF work: the 7-DoF
camera arm description rebuilt (`wx250s_7dof.urdf.xacro`, meshes moved under
`urdf/`), and `giava.urdf` brought into line.  This is the commit that made a
single robot description possible.

**`84d3ad1` "pyroki ik script"** — `pyroki_ik_script.py`, 1,443 lines in one
file.  The exploration that everything in `ik/` descends from.  It still
exists at the repo root and is still cited by `tube_mpc/MATH.md` as the source
of the GJK/sphere pipeline for the collision phase.

---

## August — the two studies, and the collision fix

This is the month that changed the system, and both halves were *measurement*
before they were code.

**`d71569c` "Add ik_study"** — 9,462 lines.  A controlled ablation: a frozen
27-trajectory benchmark suite (hash `3815490046334af4`, verified per run), a
clean baseline, one-residual-at-a-time variants, weight sweeps, a
manipulability deep-dive, and a Viser playground to watch solvers live.

What it found, and each of these was a real change of belief:

- The clean pose-only baseline is **not accuracy-limited** — sub-µm at
  0.5–1.2 ms.  Per-tick IK was never the hard part.
- It fails in three specific ways: branch dead-ends (`forearm_roll` pinning at
  ±π, 38.5 mm *permanent* error), wrist self-motion flips (241 jumps, up to
  97°/tick), and contact blindness.
- **Smoothing (0.05) + centering (0.5) together are strictly better than
  baseline on every aggregate simultaneously** — 8× tracking, 9× orientation,
  jerk ÷5 — while being marginally *faster*.  Neither alone does this.
- **Manipulability was rejected outright.**  14× solve time, zero measured
  benefit, harmful in every combination.  The prior belief was "marginally
  better but 5–10× slower"; the measurement was "not better, and 14× slower".
- pos 50 / ori 10 chosen on the *winning* combination rather than the
  baseline, because a sweep on a broken baseline measures error re-allocation
  inside a failure mode, not a real tradeoff.

**`5a730d1` "Fix URDF finger joints for pyroki + refit collision model"** — a
small diff with a large consequence: the finger joints were wrong, so every
collision fit built on them was wrong.

**The collision study** (documented in `COLLISION_STUDY.md`, landed with the
study tree).  The finding worth carrying: **collision was not broken because
collision costs are expensive.  It was broken because the geometry was
wrong.**

PyRoki's default capsule fit is the minimum bounding *cylinder*, used as a
capsule — caps added beyond the ends.  For plate-like links that orients the
axis along the thin dimension: the base plate became a 390 mm slab, a 5 mm
camera cover a 144 mm blob, a 27 mm finger read 129 mm thick.  Twelve link
pairs were permanently inside any margin (constant gradient bias, zero signal),
and kept pairs were conservative by several cm — 155 mm of bimanual workspace
falsely consumed.  That is the "hands jerking away from each other", quantified
at 59–76 mm.

The fix: 180 inscribed spheres, using cuRobo's VOXEL algorithm reimplemented
dependency-free with three fixes this robot's meshes needed.  Against 70
mesh-ground-truth cases, mean error 98 mm → **11 mm**, false-positive
collisions 15 → **0**, stolen workspace 94 mm median → **0**.  Compile time
~18 minutes → **2.5 s** (the segment-segment closest-point routine was the
bomb, not the pair count).

Deployed result: **zero tracking cost on every clean trajectory**, worst
clearance −157 mm → +3 mm.

**`5d56b4d` "Collision gate, GJK, scene-camera calibration; correct base
separation"** — 24,314 lines, the largest commit of the period.  Three things
at once:

- `study_ik.py` — the coupled three-arm solver replacing per-arm solves.
  Coupling is what makes inter-arm collision terms mean anything.
- The **gate layer**: `capsule_gate.py`, `gjk.py`, `multi_capsule.py`.  A hard
  yes/no on the final clamped command, deliberately *different in kind* from
  the solver's soft cost — the sphere model is inscribed and optimistic, so it
  can never be the last line of defence.
- `reconstruction/` — 2,600 lines including a 582-line selftest with 33 checks
  against synthetic ground truth the code under test cannot see.
- The **base separation was corrected**: the left arm is at (+0.535, −0.021,
  0.045) with yaw π−0.24°, touch-calibrated — not the symmetric value that had
  been assumed.

**`39366fe` "Tabletop world-collision"** — both halves of table avoidance
behind one switch: the soft half-space cost in the solver
(`ik/table_collision.py`) and the hard z-floor gate (`table_gate.py`), with
`--table on|off`.  The table gate is **the one that actually fires** —
inter-arm blocks are rare in single-arm work; a gripper driven into the table
through a 353:1 non-backdrivable gearbox does not give way, it stalls and
heats.

**`b52483b` "GPU solver, coupled camera arm, recording gate, replay fixes"** —
1,736 lines.  `jax_platform.py` (backend selection before any JAX import, and
printing which backend *actually* initialised, because "cuda,cpu" falls back
silently by design).  `gvlink_headset.py`, 556 lines — the new Unity headset
protocol.  `replay_episode.py` substantially rewritten.  `audit_idle_frames.py`.

**`f6c030e`, `2c8e048`** — the headset link tracked to gvlink protocol v3,
plus `HEADSET.md` and the frame probes.

---

## September — hardware truth, the reorg, and the first result that worked

**`cac87db` "Calibrated the stereo oak cameras"** — and found the left and
right lenses were **swapped**.  Measured: the board sat 102 px further left in
the frame called "left" across 25 of 25 pairs, and the installed calibration
carried `T[0] = +62.63 mm` where a correctly ordered pair must be negative.
The right lens's image had been going to the left eye — inverted depth, for
months, **invisible to every calibration metric**, because a swap leaves
rectified rows perfectly aligned.  Fixed at the single point where the two
frames are named.

**2026-09-09 — the tabletop gate hardware-validated** on the real arms, and
the tube-MPC command-mode `W` problem diagnosed (see below).

**2026-09-11 — the middle waist re-clocked by hand.**  The camera waist sat
0.5° from the encoder wrap, so boot readings landed on either branch:
episodes 0–67 recorded near −3.03 rad and 68+ near +3.26 for the same physical
pose.  It surfaced as a normalisation problem —
`observation.state[middle_base]` std 2.768 against 0.109 of real motion — and
would have reached the network as "which half of the session was this" at 25×
the amplitude of the signal.  `Homing_Offset` is inert in `ext_position` mode
and capped at ±90° anyway, so **the software knob existed and could not do the
job**; the motor was physically re-clocked.  Rest is now 175° from the seam.

**`da9bf2b` "Reorg: split data_collection_scripts into sibling trees"** —
215 Python files, of which 40 were reachable from anything run daily.  The
other 175 were research phases, one-off probes, applied migrations and sim-era
code.  Split into `ik/`, `calibration/`, `reconstruction/`, `rgbd/`, `debug/`,
`analysis/`, with `_giava_paths.py` in each folder that walks up for
`giava.urdf` rather than counting `parents[N]` (that idiom was in the tree at
two different depths and both were wrong the moment a directory moved).

**`3db9602` "tube_mpc: add the tube-MPC package"** — a robust-MPC reference
filter with recursive-feasibility guarantees, dependent only on
numpy/scipy/osqp/pyyaml.  Measured on the right arm against an operator lunge
0.4 rad past the limits: naive clamping violates position limits on **115
steps** (worst 0.41 rad past); tube MPC has **zero** violations at ~1.3 ms mean
solve.  `GIAVA_TUBE_MPC=1` wires it into the live loop.

The most instructive thing in the package is the *two config files*.
Calibrating `W` on the real arm gave `w_v = 0.038–0.081` rad/s, inflated by
280 ms of command→encoder lag.  Installed in a command-mode loop the readiness
check still said READY while reserving **94.3 % of the narrowest joint's
range** against a disturbance that, in that loop, does not occur.  Hence two
files, and a paragraph in each saying exactly what is and is not measured.

**`299d3a5`, `e30bf4e`** — the runtime modules written since the reorg, plus
the analysis and debug instruments (12 analysis tools, 27 debug probes) and the
training wrapper.

**`75c1286` "Remove superseded code"** — `archive/` (78 files), `act_code/`,
and the YOLO ACT fork deleted.  Note this leaves `av_aloha/README.md` and
`ACT_MODIFICATIONS.md` describing files that no longer exist.

**`fbbc306` "Middle arm: torque on at launch"** — and, more usefully, the
*reasons* the modes are what they are written into
`config/puppet_modes_middle.yaml`: why `ext_position` was the workaround for
the seam, why plain `position` is now viable and would be an improvement
(it restores the servo's own limit enforcement, which `ext_position` does
not apply), and why `camera_yaw` must stay in `ext_position`.

**`bdb9c2e`, `ad9a6f3`, `2cec489`** — `reconstruction/pointcloud.py`: dense
tabletop point cloud with table/object segmentation, then a curved table
surface fit, then table-side-up rendering.  The middle commit explicitly
corrects a noise claim made in the first, which is the right way to do this.

---

## The cross-cutting changes that are not one commit

### Instrumentation, because of the fps bug

The single most consequential defect of the period: the recorder wrote
`fps: 30` while the loop captured at ~0.7 Hz.  It propagated into the dataset
metadata, into a report ("each demonstration lasts 1.5 to 3 seconds"), and into
rollout configuration (`rollout_seconds=3.0`) — so the policy was asked to
execute in 3 seconds a motion demonstrated over two minutes.

Nothing caught it because the pipeline was **self-consistent everywhere the
same code ran**: collection and replay both went through the slow teleop loop,
so a replay took two minutes and looked exactly like the demonstration.  It
broke at the one boundary where a different loop ran with a hardcoded rate.

Everything below exists because of that:

- `meta/robustness.jsonl` — 25 fields per episode: `measured_fps`, `overruns`,
  `ik_solve_ms{n,mean,p95,max}`, `ik_overrun_ticks`, gate blocks and scalings,
  driver clamp counts, `tube_fallback_ticks`, `cam_sync_spread_s`, DAgger
  frame counts, and which collision/table mode produced the session.
- `analysis/measure_latency.py`, `analysis/true_fps.py`.
- a per-run config snapshot.
- epoch timestamps stored as float64.

**A rate written into metadata is a claim.  Measure the achieved rate and
store that.**

### Servo profile: `GIAVA_PROFILE_MODE=velocity`

`Drive_Mode` bit 2 decides whether `Profile_Velocity` is a *duration* or a
*speed cap*.  The Interbotix layer writes `moving_time * 1000` either way — so
`moving_time = 0.14` silently became a 3.36 rad/s cap instead of a 140 ms
move, and the loop's step clamp was **5.9× larger** than what the servo could
execute.  Recorded on hardware: `middle_base` at 2095 mA against a ~2300 mA
overload latch, sitting at zero effort afterwards while the command walked 48°
away — and the wedged bus stalled the 50 Hz loop for up to 1.65 s, taking the
arms, the headset and the camera stream down together.

`apply_profile_limits()` now reads the achieved cap **off the live motors**
rather than trusting config.  The legacy default froze arms and latched
overloads; velocity mode is required, not preferred.

### Frame synchronisation

The D405s do not support `inter_cam_sync_mode` (verified, all four, firmware
5.12.14.100), so there is no genlock.  Latest-frame-wins was replaced with
nearest-to-a-common-reference selection from a per-camera history, bounding
each frame to half a frame period of the reference.  The residual 11.6 ms
inter-camera spread is **physical and stated**, not hidden — and it is stable
(std 0.027 ms over 150 samples), so it is a calibratable constant.
`cam_sync_spread_s` is recorded per timestep.

### Clean shutdown

Camera workers are now joined *before* their devices are destroyed, and the
OAK's blocking queues are closed to wake the worker first (setting the event
alone does not).  This removed the `FATAL: exception not rethrown` core dump
and the `[depthai] Device has crashed` log on every exit.  `shutdown_cameras`
is registered with `atexit` so it also covers Ctrl-C and tracebacks, and it is
idempotent.

### Console output

Three filters (`camera_manager._info`, `robot_control._quiet_sdk`,
`quiet_solver_logs`) built on one principle: routine detail is hidden,
anything describing a failure or a behaviour-changing fallback is always
shown, suppressed lines are **counted** so "nothing was printed" and "nothing
happened" stay different statements, and everything is reprinted in full if
the thing it was quieting raises.

---

## The first result that worked (2026-09-12)

The DAgger round produced the first rollout in this repo's history where the
policy did the task.  Recorded in `LESSONS.md` §7–9, with the caveats intact:

- **Training loss did not predict rollout behaviour.**  Six variants within
  0.006 of each other behaved very differently closed-loop.
- **Rollout scores have day-scale variance large enough to invert a
  conclusion.**  The same checkpoint scored 8/17 and 4/17 on identical scenes
  a day apart.  Single rollouts per condition are coin flips; scene-paired
  blinded A/B is worth roughly triple the episode count.
- **Warm-up contaminates the first episode of every run.**  Across 53 rollout
  runs the first episode reached a median 47.3 Hz against 49.1 Hz for later
  ones, and *every* rate-abort on record (11 of 11) was an episode 0.

---

## What is not yet committed

The working tree carries ~1,000 lines of uncommitted change and seven
untracked files.  This matters: some of it is a week old and exists only on
this disk.

| | |
|---|---|
| `giava.urdf` | +265 lines — the largest uncommitted change |
| `data_collection.py` | −447/+... net simplification |
| `rollout_policy.py` | +517 lines |
| `build_envstate_dataset.py`, `scene_features.py`, `dataset.py`, `collision_modes.py` | smaller edits |
| `vx300s.yaml`, `vx300s.urdf.xacro`, `wx250s_7dof.urdf{,.xacro}` | driver/URDF config |
| `policy_training/train_real.py` | +28 |
| **untracked** | `auto_reset.py`, `build_{eedist,mask,relative_action}_dataset.py`, `eedist_features.py`, `wrist_features.py`, and the whole `data_collection_scripts/analysis/` tree (VLM annotation, grasp scoring, px→EE fit, placement reports) |

The `analysis/` tree in particular is the labelling work from 2026-09-19/20 and
has never been committed.
