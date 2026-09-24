# The five external codebases, and what each one is actually for

Four are git submodules (`.gitmodules`); the fifth (the Interbotix ROS stack)
is vendored directly into `interbotix_ws/src/`.  This is the map: what each
one is, the handful of files that matter, and where it touches your work.

| | what it is | language / runtime | is it in the live loop? |
|---|---|---|---|
| `pyroki/` | differentiable kinematics + IK | Python, JAX | **yes** — every teleop tick |
| `lerobot/` | datasets + policies | Python, PyTorch | **yes** — recording and rollout |
| `interbotix_ws/src/interbotix_*` | the servo driver | C++, ROS Noetic | **yes** — every command |
| `gym_av_aloha/` | MuJoCo sim of this robot | Python, MuJoCo | no — reference only |
| `curobo/` | GPU motion planning | Python, CUDA/Torch | **no** — read, never deployed |

Submodule pins (`git submodule status`):

```
pyroki          7b23e3e  heads/main
lerobot         eee04e9  v0.6.0-1   (one local commit)
gym_av_aloha    e13410c  mj3.3.3-6
curobo          89b1449  v0.8.0-29
```

---

## 1. `pyroki/` — the IK solver you actually run

PyRoki (Berkeley, arXiv 2505.03728) is a kinematics toolkit built on JAX.  The
idea: parse a URDF into a *differentiable* forward-kinematics function, express
whatever you want as least-squares residuals, and hand the whole thing to a
Levenberg–Marquardt solver that JIT-compiles to CPU or GPU.

### The mental model

```
URDF ──parse──> pk.Robot ──FK──> link poses
                   │
   your residuals ─┴─> jaxls.LeastSquaresProblem ──LM──> q*
   (pose, limits, smoothing, centering, collision)
```

Everything is a *cost*.  There is no "IK algorithm" separate from the costs —
IK is just "the pose residual happens to dominate".  That is why adding
collision or smoothing is a one-line change and why each addition costs solve
time: it is another block of Jacobian.

### Key files

| file | what it holds |
|---|---|
| `src/pyroki/_robot.py` | `pk.Robot` — joints, limits, `forward_kinematics()`. The object everything else takes. |
| `src/pyroki/_robot_urdf_parser.py` | URDF → robot. Where "joints not in topological order" warnings come from. |
| `src/pyroki/costs.py` | the stock cost library: `pose_cost`, `limit_cost`, `self_collision_cost`, `manipulability_cost`, `rest_cost`. |
| `src/pyroki/_residuals/_pose_residual_analytic_jac.py` | the pose residual with a **hand-derived Jacobian** — meaningfully faster than autodiff, and what we use. |
| `src/pyroki/collision/_robot_collision.py` | `RobotCollision`: `from_urdf()` (capsules) and `from_sphere_decomposition()` (you supply spheres). Pair bookkeeping lives here. |
| `src/pyroki/collision/_geometry.py` | `Capsule`, `Sphere` primitives and their closed-form distance functions. |
| `src/pyroki/collision/_collision.py` | `colldist_from_sdf` — the smooth margin hinge that turns a distance into a cost. |
| `examples/pyroki_snippets/_solve_ik_with_collision.py` | the canonical pattern our solver is descended from. |

### How it is used here

`interbotix_ws/src/av_aloha/ik/` is our layer on top:

```
ik/robot_model.py        loads giava.urdf once, caches pk.Robot   (single source of truth)
ik/baseline.py           the frozen pose-only problem (the study's control)
ik/variants.py           baseline + exactly one extra residual    (the ablation)
ik/collision_models.py   our 180-sphere model + corrected capsule fit
ik/table_collision.py    tabletop half-space as a world-collision cost
```

and `data_collection_scripts/study_ik.py` assembles the deployed
configuration — one **coupled** solve for all three arms per tick.  Coupled
matters: inter-arm collision terms are meaningless if each arm is solved in
its own problem.

Full story: [`IK.md`](IK.md).

---

## 2. `curobo/` — read for ideas, never deployed

NVIDIA cuRobo: GPU motion generation.  You never used it in the loop, and the
one attempt (`archive/debugging_scripts/curobo_teleop.py`, deleted in
`75c1286`) did not survive.  But it is worth understanding, because it is the
serious answer to exactly the two problems pyroki has here — collision
representation and speed — and one of its ideas is already running on your
robot.

### The architecture, in one picture

```
mesh ──sphere fit──> link spheres ─┐
                                   ├─> CUDA kernel: all-pairs sphere distance
world ──voxels/ESDF──> signed dist ┘        (thousands of configs at once)
                                   │
   many random seeds ──> particle optimizer (MPPI) ──> L-BFGS polish ──> rank
```

Three design choices, all different from pyroki's:

1. **Everything is spheres.**  Robot links *and* world obstacles.  A
   sphere-vs-sphere test is one subtraction; a sphere-vs-ESDF lookup is one
   texture fetch.  No segment-segment closest-point routine, no mesh queries.
2. **Custom CUDA kernels, not autodiff.**  `_src/curobolib/` hand-writes the
   collision cost *and its gradient*.  pyroki differentiates through the
   geometry with JAX; cuRobo writes the derivative down.
3. **Many seeds in parallel, then rank.**  `num_seeds=12` (or 32) IK problems
   solve simultaneously on the GPU and the best is returned.  This is how it
   escapes the local minima that trap a single warm-started LM — which is
   precisely your "branch dead-end" failure mode.

### Key files, if you read it

| file | why |
|---|---|
| `curobo/sphere_fit.py` → `_src/geom/sphere_fit/fit_spheres.py` | **the one you already borrowed.** `SphereFitType.VOXEL`: voxelize the interior, take inscribed radii. Our `ik/collision_models.py` mirrors this dependency-free (warp/torch would not import in the env). |
| `_src/geom/sphere_fit/sphere_count.py` | how it picks a sphere count from bounding volume — the question our Phase-6 Pareto answered by hand. |
| `_src/solver/solver_ik.py` | multi-seed IK: seed generation, batch padding, ranking by pose error. |
| `_src/optim/particle/` | MPPI — the sampling stage that makes it robust to bad initialisation. |
| `_src/optim/gradient/` | L-BFGS polish stage. |
| `_src/optim/multi_stage_optimizer.py` | how the two stages compose. |
| `_src/geom/collision/` | the sphere↔ESDF / sphere↔sphere checks. |
| `_src/motion/motion_planner.py` | full trajectory-level planning (not just per-tick IK). |

### What you already took from it

The collision study's 180-sphere model **is** cuRobo's VOXEL algorithm,
re-implemented without warp/torch, plus three fixes our meshes needed:
per-body convex hulls (the STLs are thin-shell housings, 12–23 % fill, so raw
inscribed spheres landed in the walls), plate-aware fitting for hulls thinner
than 25 mm, and spread-aware selection.  Documented in
`ik/study/COLLISION_STUDY.md` Part 2.

### "Collision wasn't working and it was slow" — where that stands now

Worth being precise, because the first half is **fixed** and the second half is
only partly:

**Collision was not working — that was geometry, and it is solved.**  PyRoki's
default `RobotCollision.from_urdf()` fits one capsule per link as the *minimum
bounding cylinder*, then adds hemispherical caps beyond the ends.  For a
plate-like link this orients the axis along the *thin* dimension: the base
plate (299×204×79 mm) became r = 154 mm, an effective 390 mm slab; a 5 mm
camera cover became a 144 mm blob; a 27 mm finger read 129 mm thick.  Twelve
link pairs were then permanently inside any margin — constant gradient bias
with zero signal — and the *kept* pairs were conservative by several cm, which
is the "hands jerk away from each other" you felt.  Measured against mesh
ground truth (70 cases): mean error 98 mm for pyroki capsules, **11 mm** for
the sphere model; false-positive collisions 15 → **0**; bimanual workspace
stolen 94 mm median → **0**.

Because the sphere model is *inscribed* it is optimistic by ≤ 18.3 mm, so
"collision" must mean `distance < margin`, never `distance < 0` — hence the
20 mm margin.  That is a real constraint, not a fudge.

**Slow is partly fixed and partly structural.**  The compile bomb is gone:
the capsule model took ~18 minutes of XLA compile (the segment-segment
closest-point routine, not the pair count); the sphere model compiles in
2.5 s.  Per-solve, the study measured 5.2 ms mean on a quiet CPU.  But
production, measured across **600,945 recorded ticks in 1,231 episodes**:

```
IK solve mean   13.8 ms   (per-episode median; p10 10.4, p90 14.6)
IK solve p95    23.9 ms
IK solve max    45.2 ms typical, 191 ms worst on record
ticks overrunning the 20 ms budget:  12.8 %
achieved loop rate:  45.9 Hz median against a 50 Hz target
```

So the 20 ms budget is missed on about one tick in eight.  Where it goes is
measured (`COLLISION_STUDY.md` Phase 10): **the Jacobian is ~92 % of the
collision cost.**  jaxls picks forward-mode when residual dim > tangent dim
(5,884 pairs ≫ 23 joints), so each LM iteration costs ~23 JVPs through
FK → 180 sphere transforms → all pair distances → hinge.

### Three things cuRobo suggests, ranked by payoff

1. **Aggregation — already measured, not yet deployed.**  Replace the
   P-vector collision residual with the scalar `r = w·√(Σhᵢ² + ε)`.  The
   least-squares objective is *identical* (`r² = w²Σhᵢ²`), but auto-diff flips
   to reverse mode: **one VJP instead of 23 JVPs**.  Measured 1.8× faster
   overall (2.3–2.5 ms vs 4.1–4.4 ms median), behaviour identical to 0.1 mm.
   The price is a rank-1 Gauss–Newton block for collision, which costs LM
   iterations when many pairs are active at once (70 vs 39 under adversarial
   conflict).  This is the cheapest real win available and it is already
   written up — `COLLISION_STUDY.md` Phase 10.
2. **Hand-written collision gradient.**  cuRobo's actual trick. You would
   write the sphere-pair distance derivative analytically instead of
   differentiating through FK, the way pyroki already does for the *pose*
   residual (`_pose_residual_analytic_jac.py` — the precedent exists in-tree).
   Bigger job, bigger win, and it removes the forward/reverse-mode question
   entirely.
3. **Multi-seed solve for the dead-end problem.**  Your branch dead-ends
   (forearm_roll pinning at ±π, 38.5 mm permanent error on `teleop_grasp_yaw`)
   are exactly what multi-seed fixes.  You currently solve this differently and
   more cheaply — the centering residual at w = 0.5 prevents the bad branch
   from forming at ~0.2 mm bias and zero compute — so this is the *least*
   urgent of the three.  Worth knowing it is the standard answer.

What is **not** worth importing: cuRobo's world model.  ESDF voxel grids buy
you arbitrary scene obstacles; your world is one table plane, already handled
analytically by `ik/table_collision.py` as a half-space.

---

## 3. `lerobot/` — datasets and policies

HuggingFace LeRobot, pinned at **v0.6.0** plus exactly one local commit
(`eee04e928`), which is deliberately minimal so `git diff v0.6.0` is the
complete list of your deviations:

- `__init__.py` — package shim so `import lerobot` resolves from the repo root.
- `datasets/lerobot_dataset.py` — restores the **pre-v3 writer surface** the
  data collection scripts use: `episode_buffer`, `episode_data_index`,
  `image_writer`, `create_episode_buffer`, `start/stop_image_writer`,
  `add_frame(frame, task)`.
- `datasets/utils.py` — dict-style access on `DatasetInfo` (`get/keys/items`).

### Key files

| file | what it holds |
|---|---|
| `src/lerobot/datasets/lerobot_dataset.py` | `LeRobotDataset` — the v3 on-disk format: `data/chunk-*/*.parquet`, `videos/chunk-*/<key>/*.mp4`, `meta/{info,episodes,tasks,episodes_stats}.jsonl`. |
| `src/lerobot/datasets/dataset_writer.py` | the streaming encoder our `EpisodeSaver` clones per episode. |
| `src/lerobot/datasets/compute_stats.py` | `meta/stats` — **every image feature needs an entry or the policy factory raises KeyError.** This is the trap when you add a camera stream. |
| `src/lerobot/policies/factory.py` | `make_policy` / `make_pre_post_processors`. Import this **before** `lerobot.configs` or you get a segfault. |
| `src/lerobot/policies/act/` | ACT — your main workhorse. |
| `src/lerobot/policies/{pi0,pi05,smolvla}/` | the VLA policies (the lara94 box). |
| `src/lerobot/processor/` | the v3 pre/post-processing pipeline. Build it with `make_pre_post_processors`, never by hand. |

### How it is used here

- `data_collection_scripts/dataset.py` wraps `LeRobotDataset` for recording,
  and adds the async `EpisodeSaver` writer thread and `--verify`.
- `policy_training/train_real.py` is your training wrapper.
- `data_collection_scripts/rollout_policy.py` loads a checkpoint and drives
  the arms.
- `build_*_dataset.py` derive new datasets (env-state, masks, relative
  actions, ee-distance) from a recorded one.

Two facts that cost you time and are worth keeping in front of you: video
files roll over at ~200 MB so multiple `.mp4` per camera is *normal*; and a new
image stream needs a `meta/stats` entry or the factory fails.

---

## 4. `gym_av_aloha/` — the MuJoCo sim, now reference material

The AV-ALOHA simulation environment (Chuang et al.): a MuJoCo model of this
exact robot — two 6-DoF arms plus the 7-DoF camera arm — with six tasks
(peg insertion, cube transfer, thread needle, pour test tube, hook package,
slot insertion) and the published sim datasets.

### Key files

| file | what |
|---|---|
| `gym_av_aloha/env/sim_env.py` | the gym environment. |
| `gym_av_aloha/env/sim_config.py` | joint names, camera definitions, task constants. |
| `gym_av_aloha/env/tasks/*.py` | one file per task. |
| `gym_av_aloha/kinematics/diff_ik.py`, `grad_ik.py` | the pre-pyroki IK — historical interest: `diff_ik` is the Jacobian-pseudoinverse approach, `grad_ik` the gradient-descent one. |
| `gym_av_aloha/vr/headset*.py` | the original VR teleop path, superseded here by `gvlink_headset.py`. |
| `scripts/convert_lerobot_to_avaloha.py` | format conversion for the gaze pipeline. |

### Where it still touches your work

Two places, both read-only:

- `calibration/camera_mount.py` cites `aloha.xml`'s `<camera name="wrist_cam_left">`
  as the *nominal* wrist-camera mount pose — the sim model is the design
  intent the hand-eye calibration measures deviation from.
- `debug/oak_to_headset.py` notes it **used to** import `gym_av_aloha.vr.headset`
  and no longer does (fails on Python 3.12).

It is not on the runtime path.  Keep it: it is the sim half of "one robot
description that sim and hardware agree on", and `assets/` at the repo root
holds the shared meshes and task XMLs.

---

## 5. Interbotix ROS stack — the thing that actually moves the motors

Vendored, not a submodule.  ROS Noetic, and the only C++ in the system.

```
interbotix_ws/src/
  interbotix_ros_core/interbotix_ros_xseries/
      interbotix_xs_sdk/          <- THE DRIVER NODE (C++)
      dynamixel_workbench_toolbox/ <- Dynamixel register-level library
      interbotix_xs_msgs/          <- the ROS message/service contract
  interbotix_ros_manipulators/interbotix_ros_xsarms/
      interbotix_xsarm_control/    <- launch files + per-model configs
      interbotix_xsarm_descriptions/ <- vx300s URDF xacro
  interbotix_ros_toolboxes/
      interbotix_xs_toolbox/       <- the PYTHON side you import
  wx250s_7dof_description/         <- the custom 7-DoF camera arm
```

### The chain, end to end

```
your python                          interbotix_xs_sdk (C++)          hardware
──────────                           ───────────────────────          ────────
InterbotixArmXSInterface
  .set_joint_positions(q)  ──ROS──>  xs_sdk_obj.cpp
                                       writes Goal_Position    ──U2D2──> Dynamixel
                                       over the DXL bus                  servos
joint_states  <───────ROS────────      reads Present_Position ◀─────────
```

### Key files

| file | why it matters |
|---|---|
| `interbotix_xs_sdk/src/xs_sdk_obj.cpp` | the driver. Reads the mode config, sets up groups, writes goals, publishes `joint_states`. |
| `interbotix_xs_sdk/config/mode_configs_template.yaml` | the template your `av_aloha/config/puppet_modes_*.yaml` are instances of. |
| `interbotix_xs_sdk/99-interbotix-udev.rules` | what creates `/dev/ttyDXL_puppet_{left,right,middle}`. |
| `interbotix_xs_toolbox/interbotix_xs_modules/` | `InterbotixRobotXSCore`, `InterbotixArmXSInterface` — the Python classes `robot_control.create_robot()` builds. |
| `interbotix_xsarm_control/config/vx300s.yaml` | per-model motor config (**modified** here). |
| `interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro` | the stock arm description (**modified** here). |
| `wx250s_7dof_description/urdf/wx250s_7dof.urdf.xacro` | the 7-DoF camera arm (**modified** here). |
| `av_aloha/config/puppet_modes_*.yaml` | **your** per-arm mode configs — operating mode, profile type, torque-on-at-launch. |
| `av_aloha/launch/3arms_teleop.launch` | brings up all three drivers. |

### The two register facts that cost you the most

Both are written into `config/puppet_modes_middle.yaml` as comments, which is
the right place for them:

1. **`Drive_Mode` bit 2 decides what `Profile_Velocity` means.**  Under a
   *time*-based profile it is a duration in ms; under a *velocity*-based
   profile it is a speed cap.  The Interbotix layer writes `moving_time * 1000`
   either way — so `moving_time = 0.14` silently became a 3.36 rad/s cap
   instead of a 140 ms move, and the loop's step clamp was 5.9× larger than
   what the servo could execute.  Result: tracking error growing without bound
   at stall current, `middle_base` at 2095 mA against a ~2300 mA overload
   latch, a wedged bus stalling the 50 Hz loop for up to 1.65 s.  The fix is
   `GIAVA_PROFILE_MODE=velocity` and `apply_profile_limits()` reading the
   achieved cap **off the live motors** rather than trusting config.
2. **`Homing_Offset` is inert in `ext_position` mode** (and capped at ±90°
   anyway).  The camera waist sitting on the encoder seam therefore could not
   be fixed in software — the motor was re-clocked by hand on 2026-09-11.
   `ext_position` is kept because the seam is now 175° out of the working
   range; the yaml records that plain `position` mode is now viable and would
   restore the servo's own limit enforcement.

Two `rospy.logerr` lines at every startup are **expected, not faults** —
"please set profile type to time" (you run velocity on purpose) and "please set
the gripper's operating mode to pwm or current" (the gripper is put into
`current_based_position` immediately after).  `robot_control._quiet_sdk()`
filters exactly those two and reprints everything if construction raises.
