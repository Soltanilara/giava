# Phase 8 — Final synthesis: what each residual buys, what it costs, and
# what to deploy

Study of 2026-08-08. Conditions throughout: CPU-pinned (`JAX_PLATFORMS=cpu`),
JAX 0.9.0 float32, suite rev 2.1 (27 trajectories @ 50 Hz, hash
`3815490046334af4`, verified identical for every run), solver frozen per
BASELINE.md. All raw data under `results/`; plots under `results/plots/`;
phase docs: RESULTS_BASELINE / PHASE4 / PHASE5 .md.

## The master table

Clean-tracking mean = the 21 feasible trajectories; grasp = the 3-style
reach-and-grasp family. Δ columns vs baseline.

| variant | pos p95 mm (clean mean) | ori p95 ° | grasp pos p95 mm | solve mean ms | worst-traj p95 ms | jumps | jerk RMS | trap (yaw grasp) |
|---|---|---|---|---|---|---|---|---|
| baseline | 4.72 | 2.14 | 28.7 | 0.68 | 3.2 | 241 | 1905 | stuck @38.5 mm |
| + smoothing (0.5, deployed) | 10.9 | 4.48 | 17.9 | 0.56 | 2.9 | 18 | 71 | 15.3 mm |
| + centering (0.05, deployed) | 4.72 | 2.14 | 28.7 | 0.71 | 4.1 | 238 | 1857 | stuck @38.5 mm |
| + collision (5, 3 cm) | 16.1 | 6.68 | 70.4 | 18.5 | 95.1 | 399 | 8716 | stuck |
| + manipulability (0.02) | 7.4 | 2.70 | 29.5 | 9.7 | 62.6 | 252 | 2103 | stuck @38.4 mm |
| **smoothing 0.05 + centering 0.5** | **0.60** | **0.24** | **0.83** | **0.56** | **3.7** | **138** | **378** | **0.9 mm** |
| smooth_center + manipulability | 3.41 | 0.69 | 3.3 | 7.5 | 41.6 | 202 | 395 | 4.7 mm |

## Answers to the nine questions

### 1. What does the clean PyRoki baseline do well?
Everything smooth and reachable: exact tracking (sub-µm) at 0.5–1.2 ms/solve,
including through the waist singularity and the camera arm's entire suite,
with clean 120 ms recovery from workspace-boundary excursions. Per-tick pose
IK on this robot is *not* accuracy-limited.

### 2. Where does it fail?
Three modes, all measured: (a) **branch dead-ends** — warm-started LM follows
a path into a configuration branch where the (provably reachable) target
needs a joint past its limit; forearm_roll pins at ±π and the solver sticks
(38.5 mm permanent, `teleop_grasp_yaw`); (b) **wrist self-motion flips** —
forearm_roll/wrist_rotate trade against each other near alignment (3.6–97°/
tick, 241 jumps suite-wide, jerk to 3·10⁴); (c) **contact blindness** (by
construction).

### 3. What does each residual contribute?
- **Smoothing**: converts the discrete flip/excursion failure into a small
  continuous lag; the *only* single residual that improves the grasp family.
- **Centering (at 0.5)**: holds joints off their stops so dead-end branches
  never form — eliminates failure (a) at ~0.2 mm bias and zero compute.
- **Collision (as configured)**: negative contribution in both directions —
  see 5.
- **Manipulability**: bias + compute; no failure mode improved at any weight.

### 4. Which residuals provide meaningful benefit?
Smoothing (weight-corrected to 0.05) and centering (weight-corrected to
0.5). Together they are **strictly better than baseline on every aggregate
metric simultaneously** — tracking 8×, orientation 9×, jumps ÷1.7, jerk ÷5,
limit dwell ÷4 — while being marginally *faster* (3.8 vs 7.4 LM iterations).
They are complementary, not redundant: smoothing suppresses excursions,
centering prevents dead-ends; each fixes what the other cannot.

### 5. Which are not worth their computational cost?
- **Manipulability**: 14× solve time (structural — present at weight 0),
  worst-trajectory p95 62.6 ms vs a 20 ms tick; measured benefit ≈ none
  (Yoshikawa floor 0.009→0.009 at usable weights). Even added *on top of*
  the winning pair it only degrades it (0.60→3.41 mm, +7 ms). **Verdict:
  "marginally better but 5–10× slower" was optimistic — it is *not better*
  here, and 14× slower.**
- **Collision (as configured)**: 27× solve time and it fails both jobs —
  soft w=5 loses to pose w=50 on real interpenetration, while pairs
  permanently inside the 3 cm margin (intra-gripper fingers at 5.7 mm)
  distort everything else and *create* contact on quiet trajectories.
  Needs margin-aware pair pruning + the constraint form → deferred to the
  dedicated collision study, with those two items as its agenda.

### 6. What position/orientation weighting?
**pos 50 / ori 10 (the PyRoki default), chosen on the fixed solver.** The
sweep on the baseline showed the ratio mostly re-allocates dead-end error
(deployed ori=2 hid the trap as ~25° orientation error). With the trap fixed
(smooth_center), the ratio becomes a genuine, gentle tradeoff and ori=10
lands sub-mm *and* sub-degree p95 simultaneously across all three grasp
styles (0.5–1.1 mm / 0.4–0.5°); ori=5 favors position, ori≥20 buys
~0.3° at growing transient position spikes. 1 rad ≙ 0.2 m is well-placed
for this robot.

### 7. What manipulability formulation and weight, if any?
**None, for deployment.** The anatomy (measured): translational Yoshikawa
√det(JJᵀ) maximized via residual w/(m+1e-6); Jacobian = `jax.jacfwd` over
the *full-model* FK; its own gradient by plain autodiff (second-order FK);
per LM iteration and arm the gradient call costs 1.06 ms ≈ 70× an FK call —
that, not iteration count, is the whole cost (iterations actually drop).
If it is ever wanted: the **arm-restricted jacfwd is a free 2.5× speedup**
(bit-identical results), and the **hinge form** `w·relu(m₀−m)/m₀` removes
the everywhere-bias (near_singular 8.9→0.7 mm, Lissajous exact) — but on
this suite even that buys nothing centering-at-0.5 doesn't already provide.

### 8. Best practical combination?
```
pose_cost_analytic_jac ×3      pos_weight = 50, ori_weight = 10
limit_constraint               (pyroki augmented Lagrangian)
smoothing residual             (w/(v_nom·dt))·(q − q_prev), w = 0.05  → scale 1.25
centering residual             0.5 · (q − mid)/half_range, arm joints only
LM, dense_cholesky, λ₀ = 1.0, warm start = previous commanded q, 50 Hz
```
Measured on the full suite: clean-mean 0.60 mm / 0.24° p95, grasp family
≤1.1 mm / ≤0.5° including the former trap, 0.56 ms mean / 3.7 ms worst-
trajectory p95 solve, recovery 120 ms, zero non-convergence. This is also
*simpler* than the deployed stack (no velocity clamp, no LPF, no reseed
logic were needed for any suite trajectory — whether they are needed for
real VR input remains an on-robot question).

### 9. Remaining limitations
1. **No contact safety** — clearance on the designed-contact stress is
   −157 mm (unchanged). Collision handling is unsolved pending the
   dedicated study (pair pruning, constraint-form, GPU timing).
2. **`rot_forward`-style transients**: smooth_center keeps a single ≤29 mm
   position spike during the hardest displaced-posture palm rotation
   (baseline: 18 mm + recurring 12.8 mm p95). Believed benign (one
   reconfiguration event); on-robot verification advised.
3. **Residual wrist wander**: 138 mild jumps (≈3.6°/tick ≈ the velocity
   limit) remain; a velocity clamp on hardware will mask them, or raise
   smoothing toward 0.1 at ~2 mm extra lag.
4. **Tracker noise**: jitter jerk is improved (÷18) but not eliminated;
   noise belongs to a target-side filter, not the optimizer.
5. **All timing is CPU-under-contention** (RTX 3090 busy with training);
   absolute ms have ~±30% session noise (one reproducible ~70 ms outlier
   per suite run, unexplained), but every comparison is same-session.
6. `wrist_twist`'s forced ±180° unwind is inherent to the commanded path;
   no residual can remove it (only rate-limit it).

## Deviations from the original brief (full list)
1. 50 Hz benchmark (deployment rate) instead of the old harness's 20 Hz.
2. `jitter_teleop` added (flagged) as a VR-realism stress.
3. Suite rev 2.1 contact-purity respace of rot_* (user-approved,
   pre-ablation).
4. The four collision-bearing Phase 7 combinations deferred to the collision
   study (component established broken-as-configured; ~18 min compile each).
5. Pos/ori weighting selected on the *winning combination* rather than the
   baseline, because the baseline sweep measures error re-allocation inside
   a failure mode rather than a real tradeoff.
