# Phase 4 — one-residual ablations (measured 2026-08-08)

Same conditions as RESULTS_BASELINE.md: CPU, suite rev 2.1 (hash
`3815490046334af4`, verified identical for every variant), baseline problem +
exactly one residual at the user's deployed/preset weight (`variants.py`).
Raw: `results/<variant>/`. Comparison: `python compare.py`.

## Suite-aggregate table (clean-tracking trajectories; Δ vs baseline)

| metric | baseline | +centering (0.05) | +collision (5, 3 cm) | +manipulability (0.02) | +smoothing (0.5) |
|---|---|---|---|---|---|
| pos p95 mm (mean) | 4.72 | 4.72 (±0) | 16.1 (+11.3) | 7.4 (+2.7) | 10.9 (+6.2) |
| pos max mm (worst) | 51.7 | 51.7 | 126.0 (+74) | 51.4 | 78.8 (+27) |
| ori p95 ° (mean) | 2.14 | 2.14 (±0) | 6.68 (+4.5) | 2.70 (+0.6) | 4.48 (+2.3) |
| solve mean ms | 0.68 | 0.71 | 18.5 (**27×**) | 9.7 (**14×**) | 0.56 |
| solve p95 ms (worst traj) | 3.2 | 4.1 | 95.1 | 62.6 | 2.9 |
| config jumps (total) | 241 | 238 | 399 (+158) | 252 | **18 (−223)** |
| jerk RMS (mean) | 1905 | 1857 | 8716 (+6812) | 2103 | **71 (−1834)** |
| limit dwell (frac) | 0.044 | 0.046 | 0.288 | 0.038 | **0.006** |
| worst clearance mm | −157 | −157 | −154 | −133 | −157 |
| manip hands (suite min) | 0.0017 | 0.0017 | 0.0018 | 0.0054 | 0.0017 |

Compile time (one-off, CPU): baseline/smoothing/centering seconds;
manipulability ~2 min; **collision ~18 min** (435-pair capsule cost).

## Verdicts, one per residual

### Smoothing (w=0.5, velocity-scaled) — does its job, at a real price
Buys: flips 241→18, jerk ÷27, limit dwell ÷7, jitter noise absorbed, and it
*helps* the grasp family (teleop_grasp 47.7→22.8 mm p95; yaw-trap residual
error 38.5→15.3 mm — the spring drags the wrist through reconfigurations more
gradually instead of letting LM teleport between branches).
Costs: pure tracking lag everywhere (clean-mean pos p95 +6.2 mm, ori +2.3°;
trans_y exact→21.9 mm) — at 50 Hz the deployed scaling w/(v·dt)=12.5 is a
heavy low-pass. The Phase 5 sweep must find the knee of this Pareto.
Solve time: *faster* than baseline (fewer LM iterations: 7.4→3.5).

### Joint centering (w=0.05) — inert
No metric moved (±noise). The pull is ~1000× weaker than the pose term; it
neither prevented the forearm_roll dead-end nor reduced limit dwell. Only
effect: LM iterations 7.4→6.3. Phase 5 sweeps to 2.0 to find where it wakes.

### Collision (pyroki self_collision_cost, w=5, margin 3 cm) — broken as configured
Fails in *both* directions:
- Does not prevent designed penetration (`arms_converge` still −133 mm — a
  soft w=5 term loses to pose w=50, consistent with the old benchmark's note
  on soft collision costs).
- Degrades everything else: clean-mean pos p95 ×3.4, `teleop_grasp_side`
  exact→126 mm/47°, jumps +158, and clearance on quiet trajectories went
  *negative* (trans_x +5.7→−21.6 mm — it pushed the arms *into* contact).
- 27× solve time (mean 18.5 ms ≈ the whole 20 ms tick; max 183 ms), ~18 min
  compile.

Root cause (diagnosed, matches the old ik_benchmark README): the raw pair set
contains capsule pairs that are *permanently* inside the 3 cm margin (the
intra-gripper finger pair sits at 5.7 mm by construction), so the cost
gradient pushes constantly against geometry that can never comply, distorting
the solution everywhere. A fair collision residual needs margin-aware pair
pruning (+ likely the constraint form for rigid guarantees). **Deferred to
the dedicated collision study per user; the Phase 4 conclusion is only that
the out-of-the-box configuration is not usable.**

### Manipulability (pyroki Yoshikawa, w=0.02, hand arms) — pays 14×, buys ~nothing
- Solve: 0.68→9.7 ms mean (14×), worst-trajectory p95 62.6 ms; +2 min compile.
- Manipulability itself barely moves: near_singular min 0.0086→0.0087,
  trans_y 0.044→0.044; only self_fold improved (0.0017→0.0054, with pos
  error 39→71 mm as the price).
- Does not fix the yaw-trap (38.4 mm, forearm_roll still pinned), does not
  reduce flips (252), *worsens* its own showcase (`near_singular`
  0→8.9 mm p95, 9 new jumps), and adds a ~5 mm standing error to the parked
  hands throughout the mid_* suite (the 1/w gradient never vanishes, so it
  always drags the arms away from the pose target).
- At w=0.02 the answer to "does it change the configuration?" is: barely —
  it mostly adds bias and cost. Phase 5 sweeps w to 0.5; Phase 6 dissects
  where the 14× goes and whether a hinge/arm-local formulation salvages it.

## Cross-cutting observations

1. Iteration count is not the cost driver for the heavy variants — collision
   runs 17 LM iters at 18.5 ms while baseline runs 7.4 at 0.68 ms: the cost
   is per-iteration residual/Jacobian work (autodiff through 435 capsule
   pairs / through jacfwd-FK), not extra iterations.
2. Smoothing is the only residual that touched the branch-flip failure mode,
   and it did so by slowing traversal, not by choosing better branches.
3. Nothing fixed the `teleop_grasp_yaw` dead-end. Branch selection appears
   to need a mechanism outside a local per-tick cost (multi-start/re-seed —
   the deployed solver's recovery logic, to be evaluated in Phase 7+ on top
   of the winning combination).
