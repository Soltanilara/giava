# Phase 5 — weight sensitivity (measured 2026-08-08)

Grids and pre-registered subsets per `sweep_weights.py`; raw CSVs in
`results/sweeps/`. Every sweep's zero/deployed anchor reproduces the Phase 3/4
runs exactly (deterministic pipeline).

## 1. Smoothing: the knee is at w ≈ 0.05–0.1, not the deployed 0.5

pos p95 [mm] (jumps) by weight w (residual scale = w/(2.0·0.02)):

| trajectory | 0 | 0.02 | **0.05** | **0.1** | 0.2 | 0.5 (deployed) | 1.0 |
|---|---|---|---|---|---|---|---|
| trans_y | 0.0 (71) | 0.3 (64) | 1.5 (52) | 5.7 (43) | 20.9 (0) | 21.9 (0) | 27.9 (0) |
| trans_lissajous | 0.0 (18) | 10.6 (15) | 1.2 (9) | 3.6 (0) | 15.0 (0) | 31.5 (0) | 39.3 (0) |
| rot_home | 0.0 (28) | 0.2 (30) | 1.4 (29) | 5.2 (22) | 16.2 (19) | 29.8 (0) | 45.3 (0) |
| teleop_grasp | 47.7 (13) | 0.2 (5) | **1.0 (0)** | 1.8 (0) | 4.1 (0) | 22.8 (0) | 51.5 (0) |
| jitter_teleop | 49.0 (28) | 0.9 (2) | **1.6 (0)** | 2.3 (0) | 4.8 (0) | 22.8 (0) | 51.7 (0) |
| reach_limit | 6.8 (8) | 6.8 (10) | **6.9 (10)** | 11.8 (5) | 19.5 (8) | 49.1 (0) | 83.0 (0) |

- At w=0.05–0.1 the grasp-family excursions *vanish* (47.7 → 1.0 mm: a weak
  spring prevents the wrist from entering the flip-and-recover excursion at
  all), clean-trajectory lag stays ≈1–2 mm, and boundary recovery is
  untouched. The deployed w=0.5 pays 20–30 mm of lag for nothing extra.
- Residual mild jumps at 0.05 (52 on trans_y) are the 3.6°/tick wrist-wander
  kind, not branch flips; jerk falls 10–500× vs baseline on the violent
  trajectories.
- Nonmonotonic blip at w=0.02 on trans_lissajous (10.6 mm, jerk 1636): too
  weak to suppress a mid-trajectory flip, strong enough to retime it —
  weights below the knee are *worse* than either side.
- Jitter jerk remains high at any tolerable weight (tracker noise cannot be
  absorbed by a spring without lag) — noise belongs to a target/output
  filter, outside the optimizer.

**Recommended operating point: w = 0.05 (scale 1.25), re-examined in
Phase 7 on top of the winning combination.**

## 2. Pos/ori weighting: on coupled trajectories it re-allocates the
##    dead-end error; it cannot remove it

pos p95 mm / ori p95 ° at pos_weight=50 (baseline solver, no smoothing):

| ori weight | 2 (deployed) | 5 | 10 (pyroki) | 20 | 40 |
|---|---|---|---|---|---|
| teleop_grasp | 2.5 / 26.3 | 14.3 / 24.6 | 47.7 / 20.5 | 110 / 11.3 | 142 / 3.3 |
| teleop_grasp_yaw | 2.1 / 24.7 | 12.1 / 23.2 | 38.5 / 18.7 | 88 / 10.0 | 124 / 3.3 |
| rot_forward | 0.7 / 8.0 | 4.1 / 7.3 | 12.8 / 5.7 | 27.4 / 3.1 | 38.2 / 1.1 |
| trans_lissajous, rot_home, grasp_side | 0 / 0 at every weight |

- Feasible-in-branch trajectories are exactly tracked at *every* weighting —
  the ratio only matters when the solver cannot satisfy both, i.e. inside
  the branch dead-end. There, total error is roughly conserved and the
  weights just choose its split (ori=2 → 26° hidden in orientation; ori=40
  → 14 cm hidden in position).
- **The deployed ori_weight=2 doesn't solve the trap — it hides it as ~25°
  orientation error** while position looks fine.
- Consequence (methodological): the meaningful pos/ori selection must happen
  *after* the trap is fixed. Re-swept on the best combination in Phase 7.

## 3. Centering: inert until 0.2, then at 0.5 it *prevents the branch
##    dead-end* — the surprise winner

pos p95 mm (limit margin, rad) by weight:

| trajectory | 0 | 0.05 | 0.2 | **0.5** | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| teleop_grasp_yaw | 38.5 (0.00) | 38.5 (0.00) | 38.5 (0.00) | **0.4 (0.24)** | 1.5 (0.24) | 5.5 (0.25) |
| rot_forward | 12.8 (0.00) | 12.8 (0.00) | 12.8 (0.00) | **0.6 (0.44)** | 2.1 (0.46) | 6.5 (0.51) |
| trans_y | 0.0 | 0.0 | 0.04 | 0.23 | 0.9 | 3.2 |
| near_singular | 0.0 | 0.0 | 0.03 | 0.19 | 0.7 | 2.9 |
| wrist_twist | 3.7 | 3.7 | 3.7 | 3.7 | 3.7 | 8.0 |

- At w=0.5 the pull keeps forearm_roll off its stops (margin 0.24 rad
  instead of pinned), so the dead-end branch never forms: the previously
  *unfixable* teleop_grasp_yaw goes 38.5 → 0.4 mm. Cost: ~0.2 mm bias on
  clean trajectories and zero compute (0.7 ms solves).
- Caveats: rot_forward's mild jump count rises (24→41) at 0.5 — the pull can
  trigger extra small reconfigurations; and it does nothing for wrist_twist's
  deliberate saturation until absurd weights. To be watched in combination.

**Recommended: w = 0.5, validated in combination (Phase 7).**

## 4. Manipulability: no weight is good; the cost is structural

| | w=0 | 0.005 | 0.02 | 0.1 | 0.5 |
|---|---|---|---|---|---|
| near_singular pos p95 mm | 0.0 | 0.6 | 8.9 | 111 | 275 |
| near_singular manip min | 0.009 | 0.009 | 0.009 | 0.02 | 0.04 |
| trap (yaw) pos p95 mm | 38.5 | 38.5 | 38.4 | 53 | 180 |
| solve ms (typical) | 7–36 | 7–33 | 7–31 | 7–27 | 7–17 |

- Below 0.1 it changes nothing measurable except adding bias; above, it
  destroys tracking while raising the Yoshikawa floor only 0.009→0.04.
  It never fixes the trap; centering at 0.5 strictly dominates it on every
  metric measured, at ~1/14 the compute.
- **The 10–30× solve cost is present even at weight 0** (the jacfwd-of-FK
  autodiff work is in the compiled graph regardless of the runtime weight) —
  confirmed directly by the w=0 column. Phase 6 dissects this and tests
  cheaper formulations, but the bar it must beat is now "a free residual
  that already does its practical job better".
