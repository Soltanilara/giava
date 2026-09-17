# PyRoki IK performance benchmark (three-arm GIAVA)

A reproducible harness for answering: *which cost terms and weights give
teleoperation that is accurate, smooth, safe, and responsive enough to collect
good demonstration data?*

It builds directly on the existing `three_arm_ik_collision.py` design (fixed
array shapes so weight changes never retrigger a JAX compile) and adds
workloads, measurement, and search.

## Layout

| file | role |
| --- | --- |
| `solver.py` | Configurable three-arm IK. Cost *structure* is static (one compile per structure); all weights, margins and controller gains are runtime arguments. |
| `workloads.py` | Scripted target trajectories that stress different failure modes. |
| `metrics.py` | Pose error, smoothness, manipulability, clearance, limit margin, lag, timing, and the composite teleoperation score. |
| `run_benchmark.py` | `compare` (variants head-to-head) and `sweep` (random search over weights). |

## Cost terms

| term | residual | why it matters for teleoperation |
| --- | --- | --- |
| `pose` | `log(T_actual^-1 T_target)`, split into pos/ori with separate weights | tracking accuracy |
| `limit_constraint` | pyroki augmented-Lagrangian joint limit | hard feasibility |
| `smoothness` | `w/(v_max·dt) · (q - q_prev)` | temporal continuity; the single most important term for preventing configuration flips between ticks |
| `manipulability` | `1 / Yoshikawa(J_translational)` | keeps the arm out of singular configurations where control authority collapses |
| `self_collision` | signed distance vs `margin`, all active link pairs | arms must not hit each other or the camera arm |
| `world_collision` | signed distance vs a `CollGeom` obstacle | table / object avoidance |
| `rest` | `q - q_nominal` | resolves redundancy consistently, so the same hand pose gives the same arm posture |
| `limit_barrier` (custom) | `relu(margin - d_limit)/margin`, arm joints only | pushes back *before* a joint hits its stop, instead of only reacting on violation. Targets the "arm parks on a limit and stops responding" failure mode. |

`limit_barrier` and the velocity-normalized `smoothness` scaling are the custom
terms; the rest wrap pyroki builtins.

Two loop-level knobs sit outside the optimizer, since they change felt behaviour
as much as any weight: `--target-lpf` (EMA on the incoming VR pose, fights
tracker jitter) and `--output-lpf` (EMA on the commanded joints), plus the
post-solve velocity clamp (`--velocity-limit`, `--no-clamp`).

## Variants compared

| preset | structure |
| --- | --- |
| `pure` | pose + joint limits only — raw absolute-pose IK, no temporal coupling |
| `pure_smooth` | + previous-configuration regularization |
| `collision` | + self- and world-collision |
| `manipulability` | + manipulability |
| `custom` | + rest posture and soft limit barrier |
| `full` | everything |

## Workloads

| name | what it probes |
| --- | --- |
| `smooth_track` | nominal regime: slow Lissajous reach + wrist rotation |
| `fast_track` | 4× faster: velocity budget, lag, saturation |
| `step_response` | instantaneous target jumps: rise time, overshoot, jerk |
| `jitter_track` | smooth motion + VR tracker noise: jerk amplification |
| `crossing_arms` | both hands converge on the midline: self-collision stress |
| `obstacle_sweep` | a sphere crosses the workspace while the hands track |
| `workspace_edge` | targets driven outside the reachable set: limits and singularities |
| `replay:<file.npz>` | recorded VR targets — `positions` (T,3,3), `wxyzs` (T,3,4) |

Replaying real recorded targets is the highest-value workload: dump those arrays
from the VR loop and pass `--workloads replay:episode.npz` for a like-for-like
comparison against the scripted set.

## Metrics

- **Tracking** — position error (mean/p95/max, mm) and geodesic orientation
  error (deg), per arm. Orientation error uses the relative rotation angle, not
  Euler differences.
- **Smoothness** — joint velocity/acceleration/jerk RMS, and end-effector
  Cartesian jerk RMS (what the operator actually sees).
- **Continuity** — `config_jumps`: joint-space steps more than 5× the run's
  median step, i.e. elbow flips and solution-branch switches.
- **Responsiveness** — `lag_ms` from cross-correlation of target vs achieved
  end-effector motion; `vel_saturation_frac` for time spent pinned at the cap.
- **Configuration quality** — Yoshikawa manipulability (translational and full
  6D) and Jacobian condition number, min and mean.
- **Safety** — min self/world clearance and the fraction of time in violation;
  min joint-limit margin and fraction of time within 0.05 rad of a limit.
- **Compute** — per-step solve time mean/p95/max, `realtime_factor`
  (`solve_ms / dt`), and solver iteration count.

Gripper finger joints are excluded from motion-quality and limit metrics — their
0–0.041 rad range would otherwise register as permanently "near limit".

Link pairs whose capsule approximations overlap at the home pose (or in
essentially every configuration) are auto-removed from the collision model, for
both measurement and the collision cost. Without this the self-collision cost
pushes against a violation it can never resolve. Disable with `--no-auto-ignore`.

## Score

`metrics.teleop_score` — lower is better. Each term is divided by an
operator-relevant target value (5 mm p95 position error, 2° orientation, 100 ms
lag, 1 configuration jump, 20 ms solve time, …) so mm, deg, ms and m/s³ become
comparable, then weighted. Constraint violations (limit, self-collision, world
collision) are penalized as fractions of time; minimum manipulability is a
bonus. The targets live in `DEFAULT_SCORE_TERMS` / `PENALTY_TERMS` /
`BONUS_TERMS` — edit them to match what you care about, since the ranking is
only as meaningful as those numbers.

## Usage

```bash
# head-to-head over all variants and workloads
python run_benchmark.py compare --steps 200 --out results/compare

# isolate one question
python run_benchmark.py compare --variants pure pure_smooth --workloads jitter_track

# find good weights within one cost structure
python run_benchmark.py sweep --structure full --samples 60 \
    --workloads smooth_track fast_track jitter_track crossing_arms \
    --out results/sweep

# controller-level knobs
python run_benchmark.py compare --variants full --target-lpf 0.6 --velocity-limit 3.0
```

`compare` writes `<out>_summary.csv` (per variant, averaged over workloads) and
`<out>_per_workload.csv`. `sweep` writes `<out>_sweep.csv` with every sampled
weight set and its metrics, sorted by score — the raw material for the
accuracy-vs-smoothness Pareto plot.

## Safety limits: velocity vs acceleration

The post-solve velocity clamp (`--velocity-limit`) and the acceleration limit
(`--accel-limit`) protect against different things, and it is worth being precise
about which:

- **Collision costs** prevent geometric contact. They do nothing about motor
  overload, which is an electrical/thermal failure, so enabling collision
  avoidance is not a reason to loosen the rate limits.
- **Motor overload is a torque problem**, and joint torque tracks *acceleration*
  (plus gravity and friction), not velocity. A constant-speed joint draws little
  current; an abruptly accelerating one draws a lot.
- A **velocity ceiling** bounds acceleration only indirectly — it caps how far a
  single tick can move, so it caps Δv per tick at `2·v_max/dt` in the worst case,
  which is a very loose bound. It pays for that everywhere in the workspace by
  making fast hand motion lag.
- An **acceleration limit** bounds the damaging quantity directly, which allows a
  higher velocity ceiling for the same peak torque: faster where it is safe,
  still gentle on the transitions.

`joint_acc_max` and `joint_acc_p95` are the motor-stress proxies (peak ≈
instantaneous current, RMS ≈ thermal load). They are kinematic proxies: there is
no dynamics model here, so they rank settings against each other and do not
predict amps. Validating them against measured motor current on the real arms is
the obvious next step, and the only way to set the limit non-arbitrarily.

Sweep both together:

```bash
python run_benchmark.py limits --structure full \
    --velocities 1 2 4 8 0 --accels 0 40 20 10 5 \
    --workloads fast_track step_response --out results/limits
```

`--velocities 0` means no velocity clamp; `--accels 0` means no acceleration
limit.

## Timings: CPU vs GPU

This machine now has `jax[cuda12]` on an RTX 3090. The speedup is **not**
uniform, and which backend is faster depends on the cost structure:

| variant | CPU | GPU |
| --- | --- | --- |
| `pure_smooth` | 1.1 ms | 5.8 ms |
| `full` | 193 ms | 54 ms |

Small problems are dominated by kernel-launch and host-device sync overhead, so
the GPU makes the light variants *slower*; it only pays off once the collision
costs (435 self-collision pairs) dominate. If the chosen configuration ends up
being a light one, run the teleoperation loop on CPU.

Every solve here blocks on `block_until_ready()`, matching a synchronous control
loop. An asynchronous loop that pipelines the next solve against the current
command would hide some of this latency.
