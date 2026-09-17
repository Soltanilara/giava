# Phase 1 — Baseline IK: exact specification

Status: **reference implementation** (`baseline.py`). Once signed off, this
problem definition is frozen; every later variant is baseline + exactly one
documented change.

## 1. The objective being optimized

One nonlinear least-squares problem over all 23 actuated joints, solved once
per control tick:

```
min_q  Σ_{a ∈ {left, right, middle}}  || W_a · r_a(q) ||²
s.t.   lower_j ≤ q_j ≤ upper_j        (augmented-Lagrangian constraint)
```

- `q ∈ R²³`: 6 joints per hand arm, 2 finger joints per hand, 7 camera-arm
  joints. All three arms are solved **coupled** in a single problem (they do
  not share joints, but they share the solve, the damping schedule, and the
  termination decision).
- `r_a(q) = log( T_a(q)⁻¹ · T_a* ) ∈ R⁶` — the SE(3) logarithm of the pose
  error between the FK pose `T_a(q)` of the arm's target link and the
  commanded pose `T_a*`.
- Target links: `left_gripper_base`, `right_gripper_base`, `middle_camera_cover`.

## 2. Position residual

The first three components of the SE(3) log: `r_pos = ρ(q) · pos_weight`.

Precisely: PyRoki computes `(T_actual⁻¹ T_target).log()`; the translational
part of that twist is **not** the raw Euclidean displacement — it is the
body-frame translational velocity of the geodesic connecting the two poses
(`V⁻¹ · Δt` with the SO(3) left-Jacobian inverse `V⁻¹`). For pose errors up to
a few cm / tens of degrees it is numerically close to the Euclidean error, and
it vanishes exactly when positions coincide. Evaluation metrics (Phase 3)
therefore use plain Euclidean distance, computed *outside* the solver — the
residual and the metric are deliberately not the same expression.

Implementation: `pk.costs.pose_cost_analytic_jac` — residual **and its
Jacobian** are analytic PyRoki code (`_pose_residual_analytic_jac.py`); no
automatic differentiation is involved in the pose term.

## 3. Orientation residual

The last three components of the same SE(3) log: `r_ori = ω(q) · ori_weight`,
where `ω` is the axis-angle (SO(3) log) of the relative rotation
`R_actual⁻¹ R_target`, in radians. Zero iff orientations coincide; norm equals
the geodesic angle. No Euler angles anywhere.

## 4. Position/orientation weighting

PyRoki's canonical basic-IK defaults, applied identically to all three arms:

| | value | unit |
|---|---|---|
| `pos_weight` | **50.0** | 1/m |
| `ori_weight` | **10.0** | 1/rad |

Interpretation: the optimizer trades errors at the ratio at which weighted
residuals are equal — `50·e_pos = 10·e_ori` → **1 rad ≙ 0.20 m**, i.e.
1° ≙ 3.5 mm. Whether that trade is right for teleoperation is exactly the
Phase 5 question; the baseline simply fixes the library default.

Weights are runtime arguments of the compiled solve (changing them does not
recompile), but Phases 3–4 run everything at the values above.

Note (historical): the solver deployed when this was written
(`three_arm_ik.py`) used ori_weight 2–5, i.e. it valued orientation 2–5× *less*
than the PyRoki default. This is one of the things Phase 5 adjudicated — the
study's answer was ori 10, and `three_arm_ik.py` has since been deleted in
favour of `study_ik.py`, which is now the only solver anything runs.

## 5. Constraints and joint limits

`pk.costs.limit_constraint(robot, joint_var)` — PyRoki-native. Residual per
joint: `max(0, q − upper) + max(0, lower − q)` (zero inside the range).
jaxls treats it as a **constraint** (`leq_zero`), handled by an augmented
Lagrangian: penalty + multiplier terms added to the least-squares objective,
multipliers updated in outer iterations until the constraint holds. This is
firmer than a fixed-weight penalty but still not a projection: violations can
transiently survive if the LM budget is exhausted. Phase 3 measures actual
violations rather than assuming zero.

No other constraints. Fingers are included in `q` but no cost touches them
except the limit constraint (they are downstream of every target link), so LM
damping leaves them at their initialization.

Velocity limits: **none in the baseline** (no smoothing term, no post-solve
clamp). The baseline is pure per-tick pose IK; anything temporal is an
ablation.

## 6. Solver configuration

jaxls Levenberg-Marquardt, matching PyRoki's basic-IK example exactly:

| setting | value | origin |
|---|---|---|
| algorithm | Levenberg-Marquardt (trust-region damping) | jaxls default when `trust_region` is given |
| linear solver | `dense_cholesky` (23×23 normal equations) | PyRoki example |
| `lambda_initial` | **1.0** | PyRoki example (jaxls default would be 5e-4) |
| `lambda_factor` | 2.0 | jaxls default |
| `lambda_min` / `lambda_max` | 1e-5 / 1e6 | jaxls default |
| `step_quality_min` | 1e-3 | jaxls default |
| dtype | float32 | JAX default |
| device | fixed per experiment run, recorded in results (CPU primary; see Phase 3 conditions) | study decision |

## 7. Initialization

Warm start: `q_init` = the configuration commanded at the previous tick
(first tick: the URDF home configuration). The augmented-Lagrangian
multipliers start fresh every tick (no carry-over; that is how the PyRoki
examples use jaxls).

This is the one unavoidable temporal coupling in the baseline: LM is local, so
the warm start picks the solution basin. It is initialization, not a cost
term, and is identical across all variants — so ablation deltas are still
attributable to the residual being added.

## 8. Convergence criteria (jaxls, all native defaults)

Early termination fires when **any** of:

| criterion | condition | value |
|---|---|---|
| cost tolerance | `|Δcost| / cost < tol` (non-constraint cost) | 1e-5 |
| gradient tolerance | `‖retract(x, ATb) − x‖_∞ < tol`, checked from iteration 10 | 1e-4 |
| parameter tolerance | `‖δ‖₂ < (‖x‖₂ + tol) · tol` | 1e-6 |

## 9. Stopping criteria

`max_iterations = 100` per inner LM solve (jaxls default; kept for the
baseline — a deployment budget like 20–30 would be a *variant* decision, not a
baseline one). For the constrained problem this bounds each inner solve, with
augmented-Lagrangian multiplier updates between inner solves.

Reported per solve (`SolveResult`): iterations of the final inner solve, which
termination criterion fired (all-false + iterations == max means budget
exhausted), wall-clock ms including device sync and excluding JIT compile
(`warmup()` is called first).

## 10. What is PyRoki-native vs. custom

| piece | origin |
|---|---|
| pose residual + analytic Jacobian | PyRoki (`pose_cost_analytic_jac`) |
| joint-limit constraint | PyRoki (`limit_constraint`) + jaxls augmented Lagrangian |
| LM solver, damping, termination | jaxls (native defaults except `lambda_initial=1.0` from the PyRoki example) |
| weights 50/10 | PyRoki basic-IK example defaults |
| problem assembly, warm start bookkeeping, timing | custom (this study), ~40 lines |

Explicitly **absent** (deferred to Phase 4 ablations): smoothing /
previous-configuration regularization, joint centering, rest pose, collision
(self or world), manipulability, limit barrier, velocity/acceleration clamps,
target/output filtering, re-seeding or recovery logic.

## 11. Known baseline properties to keep in mind when reading results

- Without smoothing, nothing ties tick t to tick t−1 except the warm start;
  configuration flips between equally good basins are *expected* and are part
  of what the smoothing ablation measures.
- Without any redundancy resolution, the 7-DoF camera arm's null space is
  fixed only by the warm start + LM damping; slow null-space drift is possible.
- `limit_constraint` reacts to violation; it does not keep a margin.
