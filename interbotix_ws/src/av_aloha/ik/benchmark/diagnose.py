"""Diagnostics for two observed failures.

1. `recovery` : drive the target outside the workspace, then bring it back to a
   reachable pose, and check whether the solver recovers.  Run with and without
   the manipulability cost, because `manipulability_residual` is 1/(w + 1e-6)
   and therefore explodes at exactly the singular configurations a fully
   extended arm reaches.

2. `collision` : run the arms into each other at several self-collision weights
   and report the actual capsule penetration depth.

  python diagnose.py recovery
  python diagnose.py collision
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import metrics as M  # noqa: E402
import workloads as W  # noqa: E402
from run_benchmark import TARGET_LINKS, load_robot  # noqa: E402
from solver import ControllerConfig, Weights, preset  # noqa: F401  # noqa: E402
from solver import ThreeArmIK  # noqa: E402

URDF = "/home/devi/giava/giava.urdf"

# The user's reported-good weights.
USER = Weights(
    position=(50.0, 50.0, 40.0),
    orientation=(10.0, 10.0, 1.0),
    manipulability=(0.163, 0.163, 0.0),
    smoothness=1.22,
    self_collision=20.0,
    world_collision=20.0,
    rest=0.0,
    limit_barrier=0.0,
    collision_margin=0.03,
)
CTRL = ControllerConfig(dt=0.05, velocity_limit=6.0, accel_limit=40.0, nominal_velocity=2.0)


def setup():
    robot, coll = load_robot(URDF, auto_ignore=True)
    structure, _ = preset("full")
    ik = ThreeArmIK(robot, coll, TARGET_LINKS, structure)
    ev = M.StateEvaluator(robot, coll, ik.target_indices)
    q0 = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
    base_pos, base_wxyz = W.home_poses(robot, q0, ik.target_indices)
    return robot, ik, ev, q0, base_pos, base_wxyz


def recovery() -> None:
    """Out-and-back: is the failure permanent, and is manipulability implicated?"""
    robot, ik, ev, q0, base_pos, base_wxyz = setup()

    steps_out, steps_back = 40, 60
    for label, manip_w in (("manip=0.163 (user)", 0.163), ("manip=0 (off)", 0.0)):
        weights = USER.replace(manipulability=(manip_w, manip_w, 0.0))
        ik.reset(q0)
        ik.warmup(weights, CTRL)
        ik.reset(q0)

        print(f"\n=== {label} ===")
        print(f"{'phase':10s} {'t':>3s} {'pos_err_L':>10s} {'manip_L':>9s} "
              f"{'|q|':>8s} {'nan?':>5s}")
        for t in range(steps_out + steps_back):
            if t < steps_out:
                # push the left/right targets far outside the reachable set
                reach = 0.9 * (t + 1) / steps_out
                tp = base_pos.copy()
                tp[0] += np.array([reach, reach * 0.5, 0.3])
                tp[1] += np.array([reach, -reach * 0.5, 0.3])
                phase = "OUT"
            else:
                tp = base_pos.copy()  # fully reachable again: the home pose
                phase = "BACK"
            q = ik.step(tp, base_wxyz, weights, CTRL)
            if t % 10 == 0 or t in (steps_out - 1, steps_out, len(range(steps_out + steps_back)) - 1):
                pe, _ = ev.pose_error(q, tp, base_wxyz)
                mt, _, _ = ev.manipulability(q)
                print(f"{phase:10s} {t:3d} {pe[0] * 1e3:10.1f} {mt[0]:9.5f} "
                      f"{np.linalg.norm(q):8.3f} {str(bool(np.any(~np.isfinite(q)))):>5s}")

        pe, _ = ev.pose_error(q, base_pos, base_wxyz)
        print(f"  -> final error back at the home target: "
              f"L={pe[0] * 1e3:.1f} mm  R={pe[1] * 1e3:.1f} mm "
              f"({'RECOVERED' if pe[0] * 1e3 < 20 else 'STUCK'})")


def collision() -> None:
    """Do the arms actually stop, and how deep does the capsule penetration get?"""
    robot, ik, ev, q0, base_pos, base_wxyz = setup()
    wl = W.crossing_arms(base_pos, base_wxyz, steps=120, dt=CTRL.dt, reach=0.35)

    print(f"{'w_self_coll':>12s} {'min_clear_m':>12s} {'frac<0':>8s} "
          f"{'pos_err_p95':>12s} {'margin_used':>12s}")
    for w_self in (0.0, 20.0, 200.0, 2000.0):
        for margin in (0.03, 0.08):
            weights = USER.replace(self_collision=w_self, collision_margin=margin)
            ik.reset(q0)
            ik.warmup(weights, CTRL)
            ik.reset(q0)
            clears, errs = [], []
            for t in range(len(wl)):
                q = ik.step(wl.positions[t], wl.wxyzs[t], weights, CTRL)
                clears.append(ev.self_clearance(q))
                pe, _ = ev.pose_error(q, wl.positions[t], wl.wxyzs[t])
                errs.append(pe[:2].max())
            c = np.asarray(clears)
            print(f"{w_self:12.0f} {np.min(c):12.4f} {np.mean(c < 0):8.3f} "
                  f"{np.percentile(errs, 95) * 1e3:12.1f} {margin:12.2f}")




def causes() -> None:
    """Isolate why an arm stays stuck after an out-of-reach excursion.

    Candidates: the smoothness cost anchoring to the previous configuration, the
    manipulability cost, and plain local-minimum capture (tested by re-solving
    from the home configuration instead of from the stuck one).
    """
    robot, ik, ev, q0, base_pos, base_wxyz = setup()
    steps_out, steps_back = 40, 60

    def run(weights, reinit_at_back=False):
        ik.reset(q0)
        ik.warmup(weights, CTRL)
        ik.reset(q0)
        for t in range(steps_out + steps_back):
            if t < steps_out:
                reach = 0.9 * (t + 1) / steps_out
                tp = base_pos.copy()
                tp[0] += np.array([reach, reach * 0.5, 0.3])
                tp[1] += np.array([reach, -reach * 0.5, 0.3])
            else:
                tp = base_pos.copy()
                if reinit_at_back and t == steps_out:
                    ik.reset(q0)  # warm-start from home instead of the stuck pose
            q = ik.step(tp, base_wxyz, weights, CTRL)
        pe, _ = ev.pose_error(q, base_pos, base_wxyz)
        return pe[0] * 1e3, pe[1] * 1e3

    print(f"{'configuration':44s} {'err_L_mm':>9s} {'err_R_mm':>9s}  verdict")
    trials = [
        ("user weights (smooth=1.22, manip=0.163)", USER, False),
        ("manip off", USER.replace(manipulability=(0.0,) * 3), False),
        ("smoothness 0.3", USER.replace(smoothness=0.3), False),
        ("smoothness 0.05", USER.replace(smoothness=0.05), False),
        ("smoothness 0.05 + manip off", USER.replace(smoothness=0.05, manipulability=(0.0,) * 3), False),
        ("rest cost 0.05 added", USER.replace(rest=0.05), False),
        ("user weights, re-init from home on return", USER, True),
    ]
    for label, w, reinit in trials:
        eL, eR = run(w, reinit)
        ok = "recovered" if max(eL, eR) < 20 else "STUCK"
        print(f"{label:44s} {eL:9.1f} {eR:9.1f}  {ok}")




def reseed() -> None:
    """Does the multi-start re-seed actually fix the stuck-arm failure?"""
    robot, ik, ev, q0, base_pos, base_wxyz = setup()
    steps_out, steps_back = 40, 60

    def run(ctrl):
        ik.reset(q0)
        ik.warmup(USER, ctrl)
        ik.reset(q0)
        reseeds, worst_jump = 0, 0.0
        prev = ik.q.copy()
        for t in range(steps_out + steps_back):
            if t < steps_out:
                reach = 0.9 * (t + 1) / steps_out
                tp = base_pos.copy()
                tp[0] += np.array([reach, reach * 0.5, 0.3])
                tp[1] += np.array([reach, -reach * 0.5, 0.3])
            else:
                tp = base_pos.copy()
            q = ik.step(tp, base_wxyz, USER, ctrl)
            reseeds += int(getattr(ik, "last_reseeded", False))
            worst_jump = max(worst_jump, float(np.max(np.abs(q - prev))))
            prev = q.copy()
        pe, _ = ev.pose_error(q, base_pos, base_wxyz)
        return pe[0] * 1e3, pe[1] * 1e3, reseeds, worst_jump

    print(f"{'controller':32s} {'err_L':>7s} {'err_R':>7s} {'reseeds':>8s} "
          f"{'max_dq_rad':>11s}  verdict")
    for label, ctrl in (
        ("reseed off", CTRL),
        ("reseed on (>30mm, 5 ticks)",
         ControllerConfig(dt=CTRL.dt, velocity_limit=6.0, accel_limit=40.0,
                          nominal_velocity=2.0, reseed_error_m=0.03,
                          reseed_after_ticks=5)),
    ):
        eL, eR, n, jump = run(ctrl)
        ok = "recovered" if max(eL, eR) < 20 else "STUCK"
        print(f"{label:32s} {eL:7.1f} {eR:7.1f} {n:8d} {jump:11.4f}  {ok}")




def clutch() -> None:
    """Compare the three strategies against the out-and-back excursion.

    plain    : chase the unreachable target (what happens today)
    reseed   : chase it, then escape the trapped configuration afterwards
    clutch   : refuse to chase it; park at the boundary and wait for re-engage
    """
    robot, ik, ev, q0, base_pos, base_wxyz = setup()
    steps_out, steps_back = 40, 60

    def run(ctrl):
        ik.reset(q0)
        ik.warmup(USER, ctrl)
        ik.reset(q0)
        max_stretch, frozen_ticks, worst_jump = 0.0, 0, 0.0
        prev = ik.q.copy()
        for t in range(steps_out + steps_back):
            if t < steps_out:
                reach = 0.9 * (t + 1) / steps_out
                tp = base_pos.copy()
                tp[0] += np.array([reach, reach * 0.5, 0.3])
                tp[1] += np.array([reach, -reach * 0.5, 0.3])
            else:
                tp = base_pos.copy()
            q = ik.step(tp, base_wxyz, USER, ctrl)
            frozen_ticks += int(getattr(ik, "frozen", False))
            mt, _, _ = ev.manipulability(q)
            max_stretch = max(max_stretch, float(np.linalg.norm(q)))
            worst_jump = max(worst_jump, float(np.max(np.abs(q - prev))))
            prev = q.copy()
        pe, _ = ev.pose_error(q, base_pos, base_wxyz)
        return pe[0] * 1e3, pe[1] * 1e3, frozen_ticks, worst_jump

    base = dict(dt=CTRL.dt, velocity_limit=6.0, accel_limit=40.0, nominal_velocity=2.0)
    trials = [
        ("plain (chase it)", ControllerConfig(**base)),
        ("reseed recovery", ControllerConfig(**base, reseed_error_m=0.03,
                                             reseed_after_ticks=5)),
        ("clutch (freeze at boundary)", ControllerConfig(**base, freeze_error_m=0.03,
                                                         reengage_m=0.05)),
        ("clutch + reseed", ControllerConfig(**base, freeze_error_m=0.03,
                                             reengage_m=0.05, reseed_error_m=0.03,
                                             reseed_after_ticks=5)),
    ]
    print(f"{'strategy':30s} {'err_L':>7s} {'err_R':>7s} {'frozen':>7s} "
          f"{'max_dq':>8s}  verdict")
    for label, ctrl in trials:
        eL, eR, fr, jump = run(ctrl)
        ok = "recovered" if max(eL, eR) < 20 else "STUCK"
        print(f"{label:30s} {eL:7.1f} {eR:7.1f} {fr:7d} {jump:8.4f}  {ok}")




def centering() -> None:
    """Joint-limit centering vs manipulability, on the failure and on quality.

    Reports where in its travel each joint sits ("range used"), which is the
    quantity centering targets directly and manipulability only approaches
    indirectly.
    """
    robot, coll = load_robot(URDF, auto_ignore=True)
    from solver import CostStructure
    structure = CostStructure(smoothness=True, manipulability=True,
                              self_collision=True, world_collision=True,
                              rest=True, limit_barrier=True, centering=True)
    ik = ThreeArmIK(robot, coll, TARGET_LINKS, structure)
    ev = M.StateEvaluator(robot, coll, ik.target_indices)
    q0 = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
    base_pos, base_wxyz = W.home_poses(robot, q0, ik.target_indices)
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    mid = 0.5 * (lower + upper)
    half = np.maximum(0.5 * (upper - lower), 1e-6)
    mask = ev.arm_joint_mask

    steps_out, steps_back = 40, 60

    def run(weights):
        ik.reset(q0); ik.warmup(weights, CTRL); ik.reset(q0)
        used, manips = [], []
        for t in range(steps_out + steps_back):
            if t < steps_out:
                reach = 0.9 * (t + 1) / steps_out
                tp = base_pos.copy()
                tp[0] += np.array([reach, reach * 0.5, 0.3])
                tp[1] += np.array([reach, -reach * 0.5, 0.3])
            else:
                tp = base_pos.copy()
            q = ik.step(tp, base_wxyz, weights, CTRL)
            used.append(np.max(np.abs((q - mid) / half)[mask]))
            mt, _, _ = ev.manipulability(q)
            manips.append(float(np.min(mt[:2])))
        pe, _ = ev.pose_error(q, base_pos, base_wxyz)
        return pe[0] * 1e3, pe[1] * 1e3, float(np.max(used)), float(np.min(manips))

    base = USER.replace(manipulability=(0.0,) * 3, centering=0.0)
    trials = [
        ("neither", base),
        ("manipulability 0.163 (yours)", base.replace(manipulability=(0.163, 0.163, 0.0))),
        ("centering 0.02", base.replace(centering=0.02)),
        ("centering 0.10", base.replace(centering=0.10)),
        ("centering 0.50", base.replace(centering=0.50)),
        ("centering 0.10 + manip 0.163",
         base.replace(centering=0.10, manipulability=(0.163, 0.163, 0.0))),
    ]
    print(f"{'configuration':32s} {'err_L':>7s} {'err_R':>7s} "
          f"{'max_range_used':>15s} {'min_manip':>10s}  verdict")
    for label, w in trials:
        eL, eR, used, mn = run(w)
        ok = "recovered" if max(eL, eR) < 20 else "STUCK"
        print(f"{label:32s} {eL:7.1f} {eR:7.1f} {used:15.3f} {mn:10.4f}  {ok}")




def centering_multi() -> None:
    """Recovery RATE over many excursions, not a single trial.

    A single out-and-back is not evidence: the outcome sits on a bifurcation
    (identical settings have been observed to recover in one run and stay stuck
    in another, because float32 GPU reductions tip the basin choice).  This runs
    several distinct excursions per configuration and reports how often each
    recovers.
    """
    robot, coll = load_robot(URDF, auto_ignore=True)
    from solver import CostStructure
    structure = CostStructure(smoothness=True, manipulability=True,
                              self_collision=True, world_collision=True,
                              rest=True, limit_barrier=True, centering=True)
    ik = ThreeArmIK(robot, coll, TARGET_LINKS, structure)
    ev = M.StateEvaluator(robot, coll, ik.target_indices)
    q0 = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
    base_pos, base_wxyz = W.home_poses(robot, q0, ik.target_indices)

    rng = np.random.default_rng(0)
    # Distinct excursion directions/magnitudes: each is a different way of
    # leaving the workspace, so they are independent tests of the same failure.
    scenarios = [rng.normal(size=3) for _ in range(8)]
    scenarios = [d / np.linalg.norm(d) * m
                 for d, m in zip(scenarios, [0.6, 0.7, 0.8, 0.9, 1.0, 0.7, 0.85, 0.95])]

    def run(weights, direction, ctrl):
        ik.reset(q0); ik.warmup(weights, ctrl); ik.reset(q0)
        steps_out, steps_back = 40, 60
        for t in range(steps_out + steps_back):
            if t < steps_out:
                f = (t + 1) / steps_out
                tp = base_pos.copy()
                tp[0] += direction * f
                tp[1] += direction * f * np.array([1, -1, 1])
            else:
                tp = base_pos.copy()
            q = ik.step(tp, base_wxyz, weights, ctrl)
        pe, _ = ev.pose_error(q, base_pos, base_wxyz)
        return max(pe[0], pe[1]) * 1e3

    base_w = USER.replace(manipulability=(0.0,) * 3, centering=0.0)
    plain = ControllerConfig(dt=CTRL.dt, velocity_limit=6.0, accel_limit=40.0,
                             nominal_velocity=2.0)
    clutch_ctrl = ControllerConfig(dt=CTRL.dt, velocity_limit=6.0, accel_limit=40.0,
                                   nominal_velocity=2.0, freeze_error_m=0.03,
                                   freeze_after_ticks=5, reengage_m=0.05)
    reseed_ctrl = ControllerConfig(dt=CTRL.dt, velocity_limit=6.0, accel_limit=40.0,
                                   nominal_velocity=2.0, reseed_error_m=0.03,
                                   reseed_after_ticks=5)

    trials = [
        ("baseline (no centering, no manip)", base_w, plain),
        ("manipulability 0.163", base_w.replace(manipulability=(0.163, 0.163, 0.0)), plain),
        ("centering 0.05", base_w.replace(centering=0.05), plain),
        ("centering 0.10", base_w.replace(centering=0.10), plain),
        ("centering 0.25", base_w.replace(centering=0.25), plain),
        ("centering 0.50", base_w.replace(centering=0.50), plain),
        ("clutch only", base_w, clutch_ctrl),
        ("reseed only", base_w, reseed_ctrl),
        ("centering 0.10 + reseed", base_w.replace(centering=0.10), reseed_ctrl),
    ]
    n = len(scenarios)
    print(f"{'configuration':36s} {'recovered':>10s} {'median_err_mm':>14s} "
          f"{'worst_mm':>9s}")
    for label, w, ctrl in trials:
        errs = [run(w, d, ctrl) for d in scenarios]
        ok = sum(e < 20 for e in errs)
        print(f"{label:36s} {ok:>4d}/{n:<5d} {np.median(errs):14.1f} "
              f"{max(errs):9.1f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "recovery"
    {"recovery": recovery, "collision": collision, "causes": causes,
     "reseed": reseed, "clutch": clutch,
     "centering": centering,
     "centering_multi": centering_multi}[mode]()
