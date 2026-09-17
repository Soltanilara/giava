# Phase 2 — Trajectory benchmark suite (rev 2.1, signed off)

Rev 1 was reviewed by the user in Viser on 2026-08-08 and approved with four
changes (rev 2): longer/larger translations (15–18 s, ±9–13 cm), three
reach-and-grasp approach styles, a middle-arm mirror suite (`mid_*`), and the
rev-1 `reach_limit` direction fix.

Rev 2.1 (2026-08-08, user-approved, applied *before any ablation ran*):
baseline measurement showed all three rot_* trajectories drove capsule
contact (fingertip-vs-fingertip at home standoff 57 mm; hands-vs-camera-body
at the −18 cm low posture), which would have confounded category C with
collision avoidance. Amendments, each verified contact-free (+5.7 mm floor =
the intra-gripper finger pair) by baseline rollout:

- rot_* hand positions get a mirrored ±6 cm outward x-spread;
- `rot_low` drop reduced −0.18 → −0.12 z (−0.18 sits exactly at camera
  height; offset grid-searched).

**The suite is frozen at rev 2.1** (hash `3815490046334af4`). No trajectory
may be modified to make a later variant look better. Near-contact behaviour
remains covered by arms_converge, self_fold, and the grasp family's camera
proximity; a deeper collision study is planned separately.

Global conventions (full detail in `trajectories.py`):

- Control rate **50 Hz** (`DT = 0.02 s`) — matches deployment
  (`data_col_config.control_dt = 1/50`); the old `ik_benchmark` used 20 Hz.
- Arm order (left, right, middle). Non-movers hold home as actively tracked
  targets. In `mid_*` trajectories the hands first park at
  home + (±0.20, 0, −0.15) via a 2 s min-jerk intro; `eval_start` excludes the
  intro from aggregates.
- Every run starts at the URDF home configuration with zero initial error.
- `s(u) = 10u³ − 15u⁴ + 6u⁵` (min-jerk; peak velocity 1.875·D/T).

Reachability oracle (`validate_trajectories.py`): multi-seed cold-start LM
(home + 6 random seeds, 200 iterations, threshold 3 mm / 1°). Measured
**max hand reach 0.758 m** (home = 79% of max), **max camera reach 0.658 m**
(home = 69%). "clean" = 100% of sampled waypoints reachable.

## A — pure translation (hands)

| name | definition | peak speed | duration | oracle |
|---|---|---|---|---|
| `trans_x` | p = p_home + 0.12·sin(2π·0.2·t)·x̂ | 0.15 m/s | 15 s / 750 | clean |
| `trans_y` | same, ŷ, ±0.12 | 0.15 m/s | 15 s / 750 | clean |
| `trans_z` | same, ẑ, ±0.09 | 0.11 m/s | 15 s / 750 | clean |

Purpose: per-axis translational tracking, fixed orientation. Watch: lag at
sine peaks, orientation drift, left/right asymmetry.

## B — general translation (hands)

| name | definition | peak speed | duration | oracle |
|---|---|---|---|---|
| `trans_diag` | ±0.13 m along (1,1,1)/√3, min-jerk, 2.5 s half-period × 3 | 0.10 m/s | 15 s / 750 | clean |
| `trans_lissajous` | Δp = (0.10 sin 2π·0.20t, 0.10 sin 2π·0.10t, 0.07 sin 2π·0.26t) | ~0.13 m/s peak/axis | 18 s / 900 | clean |

Purpose: does tracking degrade when motion is not axis-aligned?

## C — pure rotation at three workspace positions (hands)

R(t) = R_home·exp(30°·sin(2π t′/4 s)·ê_j), ê_j cycling gripper-local x
(finger axis), y (palm normal), z (approach axis), 4 s each; position fixed.
Peak angular rate 0.82 rad/s. 12 s / 600 steps each. All held positions
include the rev-2.1 mirrored ±6 cm outward x-spread.

| name | held position (before spread) | oracle |
|---|---|---|
| `rot_home` | p_home | clean |
| `rot_low` | p_home + (0, 0, −0.12) | clean |
| `rot_forward` | p_home + (0, −0.10, −0.08) | clean |

## D — teleop reach-and-grasp (three approach styles, coupled pos+ori)

Shared protocol, 6 s / 300 steps each, objects at (±0.20, −0.12, 0.12)
(markers in Viser): position home →(min-jerk 2.5 s) pre-grasp (10 cm standoff)
→(min-jerk 2 s) grasp (2 cm standoff) → hold; orientation
slerp(R_home → R_grasp) min-jerk-warped over t ∈ [1.2, 4.5] s — overlapping
the position motion (never "translate then rotate"). Peak hand speed
≈ 0.34 m/s in the first segment.

| name | approach direction d̂ | oracle |
|---|---|---|
| `teleop_grasp` | (0, −sin 35°, −cos 35°) — top-front pitch | clean |
| `teleop_grasp_side` | (0, −1, 0) — pitch to horizontal, side grasp | clean |
| `teleop_grasp_yaw` | Rz(∓40°)·(0, −sin 55°, −cos 55°) — pitch + yaw, mirrored, each hand from its outer side | clean |

R_grasp: approach axis ẑ = d̂, finger axis horizontal ⟂ d̂. Purpose: the
Phase 5 pos/ori weighting testbed, with graded orientation demand across the
three styles (total reorientation ~100–130°).

## E — stress tests

| name | what | duration | oracle verdict |
|---|---|---|---|
| `arms_converge` | hands cross 3.2 cm past each other (±3 cm y-stagger), hold, part | 11 s / 550 | per-arm reachable; grippers physically overlap at apex (collision metric measures it) |
| `self_fold` | right hand dives to 5 cm from its own base column | 12 s / 600 | partially: worst waypoint ~20–30 mm unreachable — intended |
| `reach_limit` | hands 0.35 m along shoulder→EE ray (demand 0.95 m vs 0.758 m max), hold 2 s, return | 12 s / 600 | infeasible at apex (worst 212 mm); 48% clean |
| `wrist_twist` | roll about approach axis 0→+200°→−200°→0 | 12 s / 600 | every pose reachable (+200° ≡ −160°); the path forces saturation at +180° then a ~320° unwind |
| `near_singular` | targets to 0.55 m above each arm's own base (waist singularity), hold 2 s | 12 s / 600 | clean — stress is conditioning, not reachability |
| `middle_gaze_sweep` | camera orbits its gaze point ±35°, hands hold home | 12 s / 600 | clean |
| `jitter_teleop` | `teleop_grasp` + iid noise (σ_pos 3 mm, σ_rot 1°, seed 0) | 6 s / 300 | clean |

## M — middle-arm mirror suite (hands parked at home + (±0.20, 0, −0.15))

All start with the 2 s park intro (excluded from aggregates via `eval_start`).

| name | definition | oracle |
|---|---|---|
| `mid_trans_x` | camera ±0.10 m sine along x̂, 15 s | clean |
| `mid_trans_y` | same, ŷ, ±0.10 | clean |
| `mid_trans_z` | same, ẑ, ±0.08 | clean |
| `mid_trans_diag` | ±0.10 m along (1,1,1)/√3, min-jerk × 3 cycles | clean |
| `mid_trans_lissajous` | (0.08, 0.08, 0.05) m at (0.20, 0.10, 0.26) Hz, 18 s | clean |
| `mid_rot_home` | ±30° about camera-local x (optical roll), y, z at home | clean |
| `mid_rot_low` | same at home + (0, −0.10, −0.05) | clean |
| `mid_approach` | coupled look-at + approach to 15 cm from (0, −0.18, 0.12); position over [0,5] s, gaze pan over [1,4.5] s | clean |
| `mid_reach_limit` | 0.40 m along shoulder→EE ray (demand 0.86 m vs 0.658 m max), hold, return | infeasible at apex (worst 226 mm) — by design |

The camera has no gripper, so the grasp analog is `mid_approach` (move in and
look) — flagged interpretation, approved direction from the user's request to
mirror "all the trajectories" for the middle arm.

## Interpretation notes / deviations (explicit)

1. **50 Hz, not 20 Hz** — matches the real control loop.
2. `wrist_twist` is a *path* stress (limit dwell + forced unwind), not per-pose
   infeasibility — established by the oracle, documented accordingly.
3. `jitter_teleop` is an addition beyond the phase spec (VR-tracker realism);
   easy to exclude from aggregates if unwanted.
4. Infeasible holds (`reach_limit`, `mid_reach_limit`) are scored as
   boundary/recovery behaviour, not tracking failure: tracking aggregates
   exclude steps whose commanded target lies beyond (max_reach − 5 mm) from
   the shoulder; recovery time and post-return error are reported separately.
5. Pre-registered sweep subset (`CORE_SUBSET`, chosen before any ablation
   results exist): trans_lissajous, rot_home, teleop_grasp, arms_converge,
   reach_limit, mid_approach.

## Suite totals

27 trajectories, ~16,750 timesteps per full-suite run (~5.6 min of robot
time). Baseline solve cost ≈ 0.7 ms/step (CPU) → ~12 s per full-suite rollout
plus metric computation.
