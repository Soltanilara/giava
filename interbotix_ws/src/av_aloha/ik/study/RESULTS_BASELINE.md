# Phase 3 — Baseline performance (measured 2026-08-08)

Conditions: CPU (`JAX_PLATFORMS=cpu`, RTX 3090 occupied by training), JAX
0.9.0, float32, suite rev 2.1 (hash `3815490046334af4`, 27 trajectories,
50 Hz), solver per BASELINE.md (pose 50/10 + limit constraint, LM ≤100
iters, warm start). Numbers below are the rev-2.1 re-measurement; the C
category was respaced (±6 cm outward; rot_low drop −0.12) after the rev-2
run showed finger/camera capsule contact — see TRAJECTORIES.md.
Raw data: `results/baseline/` (summary.csv + per-step npz per trajectory).
Runs are deterministic: two full-suite runs reproduced every metric except
wall-clock noise.

## Headline

| group | pos p95 (mm) | ori p95 (°) | solve mean (ms) | verdict |
|---|---|---|---|---|
| A/B translations (5) | 0.0001 | 0.00001 | 0.6–0.75 | essentially exact |
| C rotations: home, low | 0.0001 | 0.00001 | 0.6–0.74 | exact |
| C rot_forward | 12.8 | 5.7 | 0.76 | recurring transient (below) |
| D teleop_grasp | 47.7 | 20.5 | 0.82 | 1 s transient, ends at 0.0 mm |
| D teleop_grasp_side | 0.0001 | 0.00001 | 0.59 | exact |
| D teleop_grasp_yaw | **38.5 (persistent)** | 18.7 | 1.16 | **stuck — branch dead-end** |
| E stress | (by design) | | 0.5–1.2 | see below |
| M middle-arm (9) | 0.0001 | 0.00001 | ~0.5 | exact; boundary case recovers |

Compute across the whole suite: **mean 0.5–1.2 ms, p95 ≲ 2 ms, worst
observed 5.6 ms** (one 73 ms outlier on `mid_trans_x` in both runs — see
notes), against a 20 ms budget at 50 Hz. Non-convergence (budget exhausted):
0 everywhere except teleop_grasp_yaw 0.7% and self_fold 4.3%.

## What the clean baseline does well

1. **Reachable, smoothly-varying targets are tracked to numerical precision**
   — sub-µm/µdeg, including through the waist singularity (`near_singular`:
   manipulability drops 5× to 0.009, tracking unaffected) and the camera
   arm's whole mirror suite.
2. **Boundary recovery is clean**: `reach_limit` and `mid_reach_limit`
   recover in 120 ms after the infeasible hold, settling to ~0 error. The
   deployed solver's "parks at the boundary and stays stuck" trap did **not**
   reproduce in the smoothing-free baseline — evidence the trap involves the
   smoothing spring, to be tested directly in Phase 4.
3. **Fast and boringly consistent**: ~0.5–1.2 ms/solve, 3–26 LM iterations.

## Where it fails (the ablation targets)

1. **Branch dead-end (the big one).** `teleop_grasp_yaw` sticks at 38.4 mm /
   ~19° with both `forearm_roll` joints pinned at ±π. The commanded grasp
   pose is reachable (multi-seed oracle: 100%), but not in the branch the
   warm-started path leads into; LM cannot cross the cost barrier out.
   `rot_forward` shows the milder recurring form (transients to 21 mm during
   palm-normal rotation, recovering each cycle); `teleop_grasp` the
   transient form (0 → 50 mm → 0 within ~1 s, with an 84 rad/s wrist flip).
   Approach style determines severity: side grasp exact, top-front
   transient, pitch+yaw stuck.
2. **Configuration flips / self-motion chatter.** With nothing tying tick to
   tick but the warm start: 71 jumps on `trans_y`, 32 on `rot_home`, 18 on
   `trans_lissajous`; jerk RMS up to 655 rad/s³ (`rot_home`) vs 0.4 on quiet
   trajectories; `vel_max` up to 84 rad/s during flips. Undeployable raw —
   this is what smoothing is *for*, now quantified.
3. **No concept of contact.** Real capsule interpenetration wherever targets
   demand or graze it: `arms_converge` −133 mm (by design), `self_fold`
   −157 mm (by design), but also **`teleop_grasp` −41 mm hands-vs-camera-body**
   and `rot_home` −18.6 mm fingertip-vs-fingertip (see suite notes below).
4. **Joint-limit dwell** is common under orientation stress
   (`limit_violation_frac` 12–51% of steps on the failing trajectories,
   magnitude ~1e-5 rad — the augmented Lagrangian holds the boundary, but
   the pose error the saturation causes is the real symptom).

## Suite properties discovered during measurement

1. `rot_*` (rev 2): the grippers' fingertips start 57 mm apart and ±30°
   rotations swung them into capsule contact (−19 to −45 mm); at the −18 cm
   low posture the fingers swept the camera body. **Resolved in rev 2.1**
   (user-approved, pre-ablation): ±6 cm outward respace + rot_low drop
   −0.12; all C trajectories now hold the +5.7 mm no-contact floor, so
   category C purely measures orientation tracking for every variant.
2. `teleop_grasp*`: the hands legitimately pass close to / through the camera
   body — deployment-relevant; the `mid_clearance_min_mm` metric tracks
   camera-involved pairs separately. Kept by design (collision-ablation
   testbed; deeper collision study deferred per user).
3. `mid_*`: the parked hands' forearm grazes the base capsule by a constant
   −1.9 mm. Benign hold artifact of the park pose; camera-involved clearance
   is reported separately so it cannot mask the moving arm.

## Notes

- One 73 ms solve outlier on `mid_trans_x` step ~0 of the mid suite,
  reproduced in both runs; all other 16k steps ≤ 5.6 ms. Not yet explained
  (suspect allocator/GC hiccup at the first post-compile call of the mid
  group); tracked, excluded from conclusions (medians/p95 unaffected).
- `manip_hands_min` spans 0.0017 (`self_fold`) to 0.044; `near_singular`
  bottoms at 0.0086 with zero tracking impact — a first hint that raw
  manipulability magnitude is not by itself predictive of failure.
- Baseline leaves fingers untouched (no cost references them) — confirmed:
  finger joints stay at initialization all suite long.
