# Solving IK on this robot

Three arms, 23 actuated joints, one solve per 20 ms tick.  This is what that
solve is, how it got to be what it is, and where it still costs you.

Source docs this compresses: `ik/study/BASELINE.md`, `RESULTS_FINAL.md`,
`COLLISION_STUDY.md`, `TRAJECTORIES.md`.  Those hold the numbers; this holds
the shape.

---

## 1. What the problem is

Inverse kinematics: given a desired end-effector pose, find joint angles that
achieve it.  On this robot the honest statement is harder in three ways.

**It is over-determined and under-determined at once.**  23 joints, 3 targets
× 6 DoF = 18 constraints.  Five dimensions of null space — the arm can move
while the hand stays put.  Something has to decide what it does with that
freedom, and "whatever the solver happens to do" is a decision too.

**It is a tracking problem, not a one-shot problem.**  You are not solving
"reach this pose"; you are solving "reach this pose, 50 times a second,
starting from where you were last tick".  That changes everything: the warm
start is the previous *commanded* configuration, so the solver inherits its own
history, and a bad branch persists.

**The target comes from a human head and hands.**  It is noisy, it can be
unreachable, and it can ask for two arms to occupy the same space.  The solver
has to do something reasonable with a command that has no solution.

---

## 2. How PyRoki formulates it

Not as an algorithm — as a **least-squares problem built out of costs**:

```
q* = argmin  Σ ‖ rᵢ(q) ‖²
        q
```

with residuals `rᵢ` you choose.  FK is differentiable (JAX), so `∂r/∂q` comes
free, and `jaxls` runs Levenberg–Marquardt on it.  LM interpolates between
Gauss–Newton (fast near a solution) and gradient descent (safe far from one)
via a damping parameter λ.

The entire deployed problem, from `ik/variants.py::ComboIK`:

```python
costs = []
for i in range(3):                                   # left, right, middle
    costs.append(pk.costs.pose_cost_analytic_jac(    # ── the objective
        robot, joint_var, target_SE3[i], link_idx[i],
        pos_weight=…, ori_weight=…))
costs.append(pk.costs.limit_constraint(robot, joint_var))   # ── hard
costs.append(_smoothing_residual(joint_var, prev_q=q_init, scale=…))
costs.append(_centering_residual(robot, joint_var, weight=…, joint_mask=arm))
costs.append(pk.costs.self_collision_cost(robot, robot_coll, joint_var,
                                          margin=0.020, weight=100.0))
costs.append(<table half-space cost>)                # if --table on
```

Read that as five statements about what you want:

| cost | the statement | weight |
|---|---|---|
| `pose_cost_analytic_jac` ×3 | put the hands and camera where I asked | pos 50 / ori 10 (camera 10/25) |
| `limit_constraint` | never exceed a joint limit | hard (augmented Lagrangian) |
| smoothing | don't move much further than last tick | 0.05, velocity-scaled |
| centering | stay off your stops | 0.5, range-normalised, arm joints only |
| self-collision | don't run into yourself | 100, margin 20 mm, soft hinge |
| table | don't press into the table | 100, margin 20 mm |

`pose_cost_analytic_jac` is the version with a **hand-derived Jacobian**
instead of autodiff — meaningfully faster, and the precedent for doing the
same to the collision cost.

Two things are *not* in the list, and both were removed on measurement:
**manipulability** (14× solve time, zero measured benefit, harmful in every
combination) and **capsule collision** (see §5).

---

## 3. What the study actually found

27 frozen trajectories at 50 Hz, hash-verified identical across every run
(`3815490046334af4`), CPU-pinned, one residual changed at a time.

### The clean baseline is not accuracy-limited

Pose-only IK tracks smooth reachable motion to sub-micrometre at 0.5–1.2 ms.
Per-tick IK on this robot was never the hard part.  It fails in three specific
ways:

**(a) Branch dead-ends.**  Warm-started LM follows a path into a configuration
branch where a provably reachable target would need a joint past its limit.
`forearm_roll` pins at ±π and the solver sticks — 38.5 mm of *permanent*
error on `teleop_grasp_yaw`.  Not a convergence failure; it converges happily
to the wrong basin and stays there.

**(b) Wrist self-motion flips.**  `forearm_roll` and `wrist_rotate` trade
against each other near alignment: 3.6–97° in a single tick, 241 jumps across
the suite, jerk to 3×10⁴.  This is the null space being spent badly.

**(c) Contact blindness.**  By construction, before collision costs.

### Two residuals fix (a) and (b), and they are complementary

| variant | pos p95 | ori p95 | grasp p95 | jumps | jerk | solve |
|---|---|---|---|---|---|---|
| baseline | 4.72 mm | 2.14° | 28.7 mm | 241 | 1905 | 0.68 ms |
| + smoothing only | 10.9 | 4.48 | 17.9 | 18 | 71 | 0.56 |
| + centering only | 4.72 | 2.14 | 28.7 | 238 | 1857 | 0.71 |
| **smoothing 0.05 + centering 0.5** | **0.60** | **0.24** | **0.83** | **138** | **378** | **0.56** |

Centering alone changes nothing measurable; smoothing alone trades tracking
for smoothness.  **Together they are strictly better than baseline on every
aggregate simultaneously** — 8× tracking, 9× orientation, jerk ÷5 — while
being marginally *faster* (3.8 vs 7.4 LM iterations).

Why they compose: smoothing suppresses the per-tick excursions (b); centering
keeps joints off their stops so the dead-end branches of (a) never form.  Each
fixes what the other cannot.  Centering looks useless alone because its job is
*preventive* — it only shows up once smoothing has removed the noise it was
hiding under.

The deployed stack is also **simpler** than what it replaced: no velocity
clamp, no low-pass filter, no reseed logic was needed for any suite trajectory.

---

## 4. The frame problem, which is not an IK problem

The solver works in URDF coordinates; the driver works in its own.  For 22 of
23 joints those agree.  For `middle_base` they do not, and `study_ik.py` owns
the conversion:

```
urdf = driver + (π − h)      h = the waist's total driver-frame shift
```

`h` has **two** contributors that are not interchangeable: a physical re-clock
of the motor (unlimited, invisible to every register, −π here since
2026-09-11) and `Homing_Offset` (a register, and **inert under ext_position**).
`robot_control.resolve_middle_waist_shift()` combines them.

`urdf_to_driver()` is frame-aware: the driver's waist can boot 2π-shifted, so
the command uses the 2π-equivalent *nearest the current driver reading* —
never a full-turn jump, whatever frame the servo woke up in.

The lesson underneath: **a correction that lives outside the model is
invisible to everything that loads the model.**  `middle_joint_offsets.json`
was applied only by `study_ik.py` and `calibration/kinematics.py`, so RViz, the
collision viewer and pyroki's own collision model all drew a *different robot*
than the one being commanded.  Folding it into `giava.urdf` made there be one
robot.  The file is kept, now near-identity, so a freshly measured offset can
be applied without a URDF edit.

---

## 5. Collision: why it was broken, and what fixed it

This is the part worth reading twice, because the failure was not where it
looked.

### The symptom

Turning on pyroki's stock self-collision cost made everything worse: 27× solve
time, ~18 minute XLA compile, tracking degraded 2.8×, the rest pose blocked,
33 % non-convergence, and the hands visibly jerking away from each other.  The
natural read is "collision costs are expensive" or "the penalty shape is
wrong".  Both are false.

### The actual cause: the capsule fit

`RobotCollision.from_urdf()` fits one capsule per link as the **minimum
bounding cylinder** (`trimesh.bounds.minimum_cylinder`), then uses it as a
capsule — hemispherical caps added *beyond* the cylinder ends.

For a box a×b×t with a ≥ b ≫ t, minimising cylinder volume πr²h compares

```
axis ⊥ plate:  r ≈ √(a²+b²)/2,  h = t     →  V ∝ (a²+b²)·t
axis ∥ a:      r ≈ √(b²+t²)/2,  h ≈ a     →  V ∝ (b²+t²)·a
```

For the finger (99×54×27 mm) these differ by ~3 % and the **⊥ orientation
wins**.  Then the caps add a full radius on each flat side, so the 27 mm finger
reads 129 mm thick.  Measured fits: the base plate became a 390 mm slab; the
5 mm camera cover a 144 mm blob.

Consequences, both measured:

- **12 link pairs were permanently inside any margin.**  Zero discriminative
  signal, pure constant gradient bias — this is what corrupted the run.
- The *kept* pairs were conservative by several cm.  Home gripper↔gripper:
  truth 186 mm, capsule 31 mm.  **155 mm of bimanual workspace falsely
  consumed** — the hand-jerk, quantified.

### The fix: 180 inscribed spheres

cuRobo's VOXEL algorithm (interior grid → inscribed radii), reimplemented
dependency-free, with three fixes this robot's meshes needed: per-body convex
hulls (the STLs are thin-shell housings, 12–23 % fill, so raw inscribed
spheres landed inside the walls), plate-aware fitting (hulls < 25 mm get a
mid-plane 2-D grid at r = max(t/2, 12 mm)), and spread-aware selection.

Against 70 mesh-ground-truth cases:

| | pyroki capsule | corrected capsule | **spheres** |
|---|---|---|---|
| mean \|error\| | 97.9 mm | 73.9 mm | **11.1 mm** |
| false-positive collisions | 15 | 16 | **0** |
| stolen workspace (med/max) | 94 / 254 mm | 78 / 159 mm | **0 / 27 mm** |
| signed bias | −98 (conservative) | −74 | **+7.8 (optimistic)** |

A *corrected* single capsule (axis along the longest OBB dimension, exact
vertex containment) halves the error and still fails — one capsule inflates a
box's thin dimension to its mid dimension, which *inverts* the home
cross-finger case (truth +14 mm → −34 mm).  Single primitives are
structurally unable to meet the criteria.

### The margin is not a fudge

Inscribed spheres sit *inside* the true surface, so the model **overestimates**
distance: `d_model = d_true + ε`, ε ≥ 0, mean ≈ 8 mm, **max 18.3 mm** measured
on contact cases.  A margin-activated cost catches every true contact iff
`m > max ε`.  Hence **m ∈ [20, 25] mm**, and "collision" must mean
`d < margin`, never `d < 0`.

### The cost shape, and why soft beats hard

Per active pair, `colldist_from_sdf`:

```
h(d) = 0                  d ≥ m          (off)
     = (m − d)² / (2m)    0 < d < m      (smooth quadratic ramp)
     = m/2 − d            d ≤ 0          (linear in penetration)
```

C¹ at both joints, which is what keeps boundary approach from chattering.
The augmented-Lagrangian *constraint* form (hard standoff at the margin) was
tried and **empirically rejected**: it fights boundary-riding (81 LM
iterations, 80 % non-convergence on the tight bimanual test, 9 mm forced
deviation) and *diverges* under infeasible commands.  For teleoperation,
"resist and bound" is the right semantics — a crossing command is operator
error, not a task.

### Result

| variant | pos p95 | ori p95 | worst clearance | solve mean |
|---|---|---|---|---|
| smooth_center | 0.60 mm | 0.24° | −157 mm | 0.56 ms |
| **+ sphere collision** | **0.60 mm** | **0.24°** | **+3.0 mm** | 5.18 ms |
| + capsule collision | 13.05 | 4.52 | +3.1 | 43.1 (33 % nonconv) |

Zero tracking cost on every clean trajectory, worst clearance −157 → +3 mm.
Bimanual deviation: 0.0 mm on clear tests, 2.6 mm on the boundary-riding test
— exactly what is needed to hold +15 mm clearance, against 59–76 mm for
capsules.

One pleasant accident: collision *incidentally* prevents the yaw-grasp
dead-end, because the bad branch swings the hand into the camera's protection
band.

One honest caveat: with a soft cost and an optimistic model, a deliberately
interpenetrating command is **bounded to ≈1–2 cm true penetration**, not
prevented.  Which is why there is a gate.

---

## 6. The gate — a second layer, deliberately different

The solver's cost and the gate are **not two settings of one knob**:

```
IK SOFT COST        shapes how motion FEELS         must be differentiable
                    sphere, INSCRIBED (optimistic)  → cannot prove separation
                                                    → can never be last defence

GATE                decides WHERE YOU STOP          needs no gradient
                    capsule (circumscribed)         → conservative, can prove it
                    or exact GJK                    → iterative, branchy,
                                                      no useful gradient
                                                    → cannot be an IK residual
```

`--collision sphere|capsule|gjk` in `collision_modes.py` selects a *coherent
configuration of both layers*, and prints exactly what it changed so a
session's feel can be attributed correctly.  The mode is recorded in the
episode log.

- `sphere` — sphere solver cost, coarse-capsule gate.
- `capsule` — capsule solver cost (the study rejected this in sim; the flag
  exists so you can feel it), coarse-capsule gate.
- `gjk` (**default**) — sphere solver cost, **exact GJK gate**: measured to
  stop the arms 27 mm closer in real geometry while never touching.

`--table on` (default) enables both halves of tabletop avoidance: the soft
half-space cost in the solver and the hard z-floor gate.  **The table gate is
the one that actually fires** — inter-arm blocks are rare in single-arm work,
table blocks with a fingertip limiter are routine, and the failure it prevents
(a gripper driven into the table through a 353:1 non-backdrivable gearbox)
does not give way: it stalls and heats.  `--table off` prints a warning
because a stale shell variable disarming it silently is exactly the failure
mode to avoid.

---

## 7. What it costs in production

The study measured 5.2 ms mean on a quiet CPU with the bare deployment stack.
Production carries the table cost too, runs on GPU, and competes with cameras,
encoding and the headset.  Across **1,231 episodes / 600,945 ticks**:

```
IK solve mean      13.8 ms   (per-episode median; p10 10.4, p90 14.6)
IK solve p95       23.9 ms
IK solve max       45.2 ms typical, 191 ms worst on record
overrun ticks      12.8 %    (against a 20 ms budget)
achieved rate      45.9 Hz   median, p10 43.2, p90 46.8
```

So the budget is missed on roughly one tick in eight.  That is not a crisis —
the loop degrades by sleeping zero, not by falling over — but it is the thing
standing between you and a true 50 Hz.

Where it goes (Phase-10 microbench, jitted, ms/call):

| | spheres 5,884 pairs | capsules 421 pairs |
|---|---|---|
| residual eval | 0.048 | 0.076 |
| **Jacobian (what LM needs)** | **0.569** | **1.437** |
| aggregated scalar grad | 0.360 | 0.544 |

**The Jacobian is ~92 % of the collision cost.**  jaxls picks forward mode
when residual dim > tangent dim (5,884 ≫ 23), so every LM iteration costs
~23 JVPs through FK → 180 sphere transforms → all pair distances → hinge.

Note the capsule Jacobian is 2.5× the sphere Jacobian *with 14× fewer pairs* —
the segment-segment closest-point routine is expensive to differentiate, and
it is also the ~18-minute compile bomb.  Spheres compile in 2.5 s.

---

## 8. Three ways to make it faster, ranked

**1. Aggregation — measured, written up, not deployed.**  Replace the
P-vector collision residual with the scalar `r = w·√(Σhᵢ² + ε)`.  The
least-squares objective is *identical* (`r² = w²Σhᵢ² + w²ε`), but autodiff
flips to reverse mode: **one VJP instead of 23 JVPs**.  Measured 1.8× faster
overall (2.3–2.5 vs 4.1–4.4 ms median), behaviour identical to 0.1 mm.  The
price is a rank-1 Gauss–Newton block for collision, which costs LM iterations
when many pairs are simultaneously active (70 vs 39 under adversarial
conflict).  For a loop that overruns 12.8 % of ticks, this is the obvious
first move.  `COLLISION_STUDY.md` Phase 10.

**2. An analytic collision Jacobian.**  cuRobo's real trick: write the
sphere-pair distance derivative down instead of differentiating through FK.
PyRoki already does exactly this for the pose residual
(`_pose_residual_analytic_jac.py`), so the pattern is in-tree.  Bigger job,
bigger win, and it makes the forward/reverse-mode question moot.

**3. Multi-seed solve.**  Solve N randomly-seeded IK problems in parallel and
rank by pose error — how cuRobo escapes local minima.  This targets the branch
dead-ends, which **you have already solved more cheaply** with the centering
residual (0.2 mm bias, zero compute).  Least urgent; worth knowing it is the
standard answer.

Things that are *not* the problem, all measured: pair count (5,884 vs 14,491
differ ~14 % in Jacobian time — pruning buys principled signal, not runtime);
sphere count (any meaningful reduction re-creates capsule-style conservatism
exactly where it hurts, at the gripper forks); and the penalty shape.

---

## 9. Remaining limitations, stated plainly

1. **Bounded, not prevented.**  A deliberately interpenetrating command is
   held to ≈1–2 cm true penetration at w = 100.  w = 300 tightens it at
   ~55 ms during conflicts.  The gate is what makes this acceptable.
2. **`reach_limit` recovery regressed** 6.8 → 27–32 mm p95 (boundary stretch
   + extension self-proximity + smoothing lag compound).  Over-reach is
   operator error in deployment; documented, accepted.
3. **138 mild wrist jumps remain** (≈3.6°/tick ≈ the velocity limit).  A
   hardware velocity clamp masks them, or raise smoothing toward 0.1 at ~2 mm
   extra lag.
4. **Tracker noise** is improved ÷18 but not eliminated.  Noise belongs to a
   target-side filter, not the optimizer.
5. **The table cost has never been swept.**  It was hardware-validated
   2026-09-09 and its margin was chosen, not measured — unlike the inter-arm
   margin, which has a full grid behind it.  That is the next study.
6. **`wrist_twist`'s forced ±180° unwind** is inherent to the commanded path.
   No residual removes it; only a rate limit.
