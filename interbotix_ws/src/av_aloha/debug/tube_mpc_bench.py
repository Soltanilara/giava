"""Tube-MPC bench: one real arm, a scripted joint reference, every number
that decides whether the filter is fit to sit inside the teleop loop.

Measures, per run:
  latency     command -> encoder lag per joint (cross-correlation of the
              command and measured velocity streams, same method as
              measure_latency.py), plus loop dt and overruns
  tracking    joint |q_cmd - q_meas|; end-effector POSITION (mm) and
              ORIENTATION (deg) error of the achieved pose against the
              reference pose, through the same pk.Robot FK the IK uses
  solve       tube-MPC solve time (mean / p99 / max vs the 20 ms budget)
  velocity    measured joint speed (Present_Velocity register, not a finite
              difference) against v_max and against what the reference asked
  current     Present_Current per joint (mA): peak, RMS, and the fraction of
              the run above the servo_health warning level

Modes (--mode):
  calib      gentle sinusoid straight to the driver, NO filter.  Writes the
             .npz that `python -m tube_mpc.calibrate_w` reads (q_cmd, q_meas,
             v_meas, dt).  Run this first on every machine.
  baseline   reference -> driver clamp -> driver, i.e. what data_collection
             does today with GIAVA_TUBE_MPC unset.
  tube       reference -> tube MPC -> driver clamp (should be a no-op) -> driver.

--lunge pushes the reference PAST a joint limit for the middle third of the
run: the test the filter exists for.  Refused in baseline mode on hardware
(it would drive the joint into its hard stop at full speed); use --sim.

Usage (data_collection_scripts/, gym_av312 env, driver launched):
  python tube_mpc_bench.py --mode tube                     # print the plan, no motion
  python tube_mpc_bench.py --mode calib --go               # gentle, for W
  python tube_mpc_bench.py --mode baseline --go
  python tube_mpc_bench.py --mode tube --go
  python tube_mpc_bench.py --mode tube --amp 0.4 --freq 1.0 --go   # fast
  python tube_mpc_bench.py --mode tube --lunge --go
  python tube_mpc_bench.py --sim --mode tube --lunge       # no robot
  python tube_mpc_bench.py --analyze bench_logs/<file>.npz

Every run writes bench_logs/<mode>_<arm>_<stamp>.npz and prints the report;
--analyze re-prints it from the file.  Ctrl-C stops commanding; the arm
holds where it is.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_config import ARM_CONFIG  # noqa: E402
from tube_mpc_hook import build_arm_filter, describe  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_logs")
# servo_health's warning level; imported lazily so --sim needs no ROS
EFFORT_WARN_MA_FALLBACK = 700.0


# --------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------
def make_reference(q0, lim, t, amp, freq, joints, lunge_joint=None,
                   lunge_rad=0.3, ramp_s=1.0):
    """(T, n) joint reference: q0 + per-joint sinusoid on `joints`, ramped in
    over ramp_s.  Phases are staggered so the joints do not all reverse at
    once (that is one big current spike, not a tracking test).

    With lunge_joint set, the middle third of the run replaces that joint's
    reference with q_max + lunge_rad: infeasible on purpose."""
    n = len(q0)
    ref = np.tile(q0, (len(t), 1))
    ramp = np.clip(t / ramp_s, 0.0, 1.0)
    for k, j in enumerate(joints):
        # keep the sinusoid inside the limits with 0.05 rad to spare
        room = min(lim.q_max[j] - q0[j], q0[j] - lim.q_min[j]) - 0.05
        a = min(amp, max(room, 0.0))
        ref[:, j] = q0[j] + a * ramp * np.sin(2 * np.pi * freq * t + 2 * np.pi * k / max(len(joints), 1))
    if lunge_joint is not None:
        T = len(t)
        sl = slice(T // 3, 2 * T // 3)
        ref[sl, lunge_joint] = lim.q_max[lunge_joint] + lunge_rad
    return ref


# --------------------------------------------------------------------------
# plants
# --------------------------------------------------------------------------
class RealArm:
    """The interbotix arm behind the same three calls the hook needs."""

    def __init__(self, arm, moving_time, accel_time):
        import rospy
        from robot_control import create_and_configure_robots, get_joint_positions
        from data_col_config import TeleopConfig

        if not rospy.core.is_initialized():
            rospy.init_node("tube_mpc_bench", anonymous=True)
        self.arm = arm
        self.bot = create_and_configure_robots((arm,))[arm]
        self.n = ARM_CONFIG[arm]["num_joints"]
        self.names = ARM_CONFIG[arm]["joint_names"]
        self._get_q = get_joint_positions
        self.moving_time = moving_time
        self.accel_time = accel_time
        gi = self.bot.arm.group_info
        self.driver_lo = np.asarray(gi.joint_lower_limits, dtype=float)[: self.n]
        self.driver_hi = np.asarray(gi.joint_upper_limits, dtype=float)[: self.n]
        tc = TeleopConfig()
        self.driver_margin = tc.driver_limit_margin
        self.driver_max_step = float(tc.driver_max_step)
        try:
            from servo_health import EFFORT_WARN_MA
            self.effort_warn = float(EFFORT_WARN_MA)
        except Exception:
            self.effort_warn = EFFORT_WARN_MA_FALLBACK
        # the driver validates against its last ACCEPTED command; start equal
        self.bot.arm.joint_commands = list(self._get_q(self.bot))

    def read(self):
        js = self.bot.arm.core.joint_states
        q = np.asarray(js.position[: self.n], dtype=float)
        v = np.asarray(js.velocity[: self.n], dtype=float) if len(js.velocity) >= self.n else np.zeros(self.n)
        e = np.asarray(js.effort[: self.n], dtype=float) if len(js.effort) >= self.n else np.zeros(self.n)
        return q, v, e

    def driver_ref(self):
        getter = getattr(self.bot.arm, "get_joint_commands", None)
        if getter is not None:
            try:
                return np.asarray(getter(), dtype=float)[: self.n]
            except Exception:
                pass
        return np.asarray(self.bot.arm.joint_commands, dtype=float)[: self.n]

    def send(self, q_cmd):
        ok = self.bot.arm.set_joint_positions(
            q_cmd.tolist(), moving_time=self.moving_time,
            accel_time=self.accel_time, blocking=False)
        return ok is not False

    def hold(self):
        q, _, _ = self.read()
        self.bot.arm.set_joint_positions(q.tolist(), moving_time=None,
                                         accel_time=None, blocking=False)


class SimArm:
    """No ROS: the tube_mpc double integrator with box noise as the plant,
    plus a first-order servo lag so latency/tracking numbers are non-trivial."""

    def __init__(self, arm, dt, lag_ticks=2):
        from tube_mpc.config import FilterConfig, resolve_urdf
        from tube_mpc.urdf_limits import parse_urdf_limits
        from tube_mpc_hook import DEFAULT_CONFIG, PREFIX_OF

        cfg = FilterConfig.load(os.environ.get("GIAVA_TUBE_CONFIG", DEFAULT_CONFIG))
        self.names, self.lim = parse_urdf_limits(resolve_urdf(cfg.urdf_path), PREFIX_OF[arm], cfg.a_max)
        self.n = self.lim.n
        self.dt = dt
        self.driver_lo, self.driver_hi = self.lim.q_min.copy(), self.lim.q_max.copy()
        self.driver_margin = 0.03
        self.driver_max_step = 0.396
        self.effort_warn = EFFORT_WARN_MA_FALLBACK
        self.q = 0.5 * (self.lim.q_min + self.lim.q_max)
        self.v = np.zeros(self.n)
        self.goal = self.q.copy()
        self.alpha = 1.0 / max(lag_ticks, 1)     # servo pole
        self.rng = np.random.default_rng(0)

    def read(self):
        # "current" ~ |acceleration| + holding torque proxy, just so the
        # column is populated in sim
        return (self.q.copy(), self.v.copy(),
                200.0 + 60.0 * np.abs(self.v) + self.rng.normal(0, 5, self.n))

    def driver_ref(self):
        return self.goal.copy()

    def send(self, q_cmd):
        self.goal = np.asarray(q_cmd, dtype=float).copy()
        return True

    def tick(self):
        # first-order chase of the goal, velocity-limited, plus box noise
        v_des = np.clip(self.alpha * (self.goal - self.q) / self.dt,
                        -self.lim.v_max, self.lim.v_max)
        a = np.clip((v_des - self.v) / self.dt, -self.lim.a_max, self.lim.a_max)
        self.v = self.v + self.dt * a + self.rng.uniform(-1, 1, self.n) * 0.01
        self.q = self.q + self.dt * self.v + self.rng.uniform(-1, 1, self.n) * 0.002

    def hold(self):
        self.goal = self.q.copy()


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def driver_clamp(q_cmd, ref, plant):
    """data_collection's driver-feasibility clamp, verbatim in spirit."""
    c = np.clip(q_cmd, ref - plant.driver_max_step, ref + plant.driver_max_step)
    return np.clip(c, plant.driver_lo + plant.driver_margin,
                   plant.driver_hi - plant.driver_margin)


def run(args):
    dt = 1.0 / args.hz
    plant = SimArm(args.arm, dt) if args.sim else RealArm(args.arm, args.moving_time, args.accel_time)
    n = plant.n
    names = plant.names

    filt = None
    if args.mode == "tube":
        filt = build_arm_filter(args.arm, plant.driver_lo, plant.driver_hi,
                                plant.driver_margin, config_path=args.config,
                                state_source=args.state, lookahead=args.lookahead,
                                v_scale=args.v_scale)
        lim = filt.model.limits
    else:
        from tube_mpc.model import JointLimits
        lim = JointLimits(q_min=plant.driver_lo + plant.driver_margin,
                          q_max=plant.driver_hi - plant.driver_margin,
                          v_max=np.full(n, np.pi), a_max=np.full(n, 12.0))

    q0, _, _ = plant.read()
    T = int(round(args.duration * args.hz))
    t = np.arange(T) * dt
    joints = [int(j) for j in args.joints.split(",")] if args.joints else [1, 2, 4]
    lunge_joint = args.lunge_joint if args.lunge else None
    ref = make_reference(q0, lim, t, args.amp, args.freq, joints, lunge_joint, args.lunge_rad)

    peak_v = args.amp * 2 * np.pi * args.freq
    peak_a = peak_v * 2 * np.pi * args.freq
    print(f"plan: {args.mode} on {args.arm} ({'SIM' if args.sim else 'REAL'}), "
          f"{args.duration:.0f} s at {args.hz:.0f} Hz, joints {[names[j] for j in joints]}")
    print(f"      amp {args.amp:.2f} rad, {args.freq:.2f} Hz -> peak ref velocity "
          f"{peak_v:.2f} rad/s ({100 * peak_v / np.pi:.0f}% of v_max), peak accel {peak_a:.1f} rad/s^2")
    if lunge_joint is not None:
        print(f"      LUNGE: {names[lunge_joint]} reference -> q_max + {args.lunge_rad:.2f} rad "
              f"for the middle third")
    if filt is not None:
        print(f"      filter: state={filt.state_source} lookahead={filt.lookahead} "
              f"v_max={lim.v_max.max():.2f} rad/s "
              f"w_q={np.asarray(filt.cfg.w_q).ravel()[:1]} w_v={np.asarray(filt.cfg.w_v).ravel()[:1]}")
    if args.lunge and args.mode == "baseline" and not args.sim:
        sys.exit("refusing: --lunge in baseline mode on hardware drives the joint into its "
                 "hard stop at driver speed. Use --mode tube, or --sim.")
    if not args.go and not args.sim:
        print("dry run (no --go): nothing sent.")
        return None

    log = {k: np.zeros((T, n)) for k in ("q_ref", "q_cmd", "q_meas", "v_meas", "effort", "v_plan")}
    log["t"] = np.zeros(T)
    log["solve_ms"] = np.zeros(T)
    log["loop_ms"] = np.zeros(T)
    log["fallback"] = np.zeros(T, dtype=bool)
    log["clamped"] = np.zeros(T, dtype=bool)
    log["refused"] = np.zeros(T, dtype=bool)
    log["status"] = np.empty(T, dtype=object)

    if filt is not None:
        filt.reset(q0)
    t_start = time.monotonic()
    next_tick = t_start
    aborted = None
    try:
        for k in range(T):
            tk0 = time.perf_counter()
            q, v, e = plant.read()
            q_target = ref[k]

            if filt is not None:
                q_cmd, info = filt.step(q, v, q_target)
                log["solve_ms"][k] = info["solve_ms"]
                log["fallback"][k] = info["fallback"]
                log["status"][k] = info["status"]
                log["v_plan"][k] = info["v_plan"]
            else:
                q_cmd = q_target.copy()
                log["status"][k] = "passthrough"

            # the driver clamp stays in both modes -- in tube mode it must
            # never fire, and the count proves it
            clamped = driver_clamp(q_cmd, plant.driver_ref(), plant)
            log["clamped"][k] = bool(np.any(np.abs(clamped - q_cmd) > 1e-6))
            q_cmd = clamped

            log["refused"][k] = not plant.send(q_cmd)
            if args.sim:
                plant.tick()

            log["t"][k] = time.monotonic() - t_start
            log["q_ref"][k], log["q_cmd"][k] = q_target, q_cmd
            log["q_meas"][k], log["v_meas"][k], log["effort"][k] = q, v, e
            log["loop_ms"][k] = (time.perf_counter() - tk0) * 1e3

            if not args.sim and np.max(np.abs(e)) > args.effort_abort:
                aborted = f"effort {np.max(np.abs(e)):.0f} mA > --effort-abort {args.effort_abort:.0f}"
                break

            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0 and not args.sim:
                time.sleep(sleep)
    except KeyboardInterrupt:
        aborted = "Ctrl-C"
    finally:
        plant.hold()

    if aborted:
        k_end = k
        print(f"\nSTOPPED at tick {k_end}/{T}: {aborted}. Arm holding.")
        for key in list(log):
            log[key] = log[key][:k_end]
    log = {k: (np.asarray(v, dtype=str) if k == "status" else v) for k, v in log.items()}
    log.update(dt=dt, mode=args.mode, arm=args.arm, names=np.array(names),
               sim=bool(args.sim), amp=args.amp, freq=args.freq,
               lunge=bool(args.lunge), lunge_joint=lunge_joint if lunge_joint is not None else -1, lunge_rad=args.lunge_rad,
               q_min=lim.q_min, q_max=lim.q_max, v_max=lim.v_max,
               effort_warn=plant.effort_warn,
               filter=describe(filt) if filt is not None else "none")
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{args.mode}_{args.arm}_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    np.savez_compressed(path, **log)
    print(f"\nlog: {path}")
    if args.mode == "calib":
        cpath = path.replace(".npz", "_calib.npz")
        np.savez_compressed(cpath, q_cmd=log["q_cmd"], q_meas=log["q_meas"],
                            v_meas=log["v_meas"], dt=dt)
        print(f"calibration input: {cpath}\n  -> python -m tube_mpc.calibrate_w {cpath} "
              f"--config {args.config or '/home/devi/giava/tube_mpc/config.giava.yaml'} --write")
    return path


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def _xcorr_lag(a, b, max_lag):
    """Lag (ticks) at which b best matches a shifted forward: b[k] ~ a[k-lag]."""
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0, 0.0
    best, best_c = 0, -np.inf
    for lag in range(0, max_lag + 1):
        c = float(np.dot(a[: len(a) - lag], b[lag:]) / (len(a) - lag))
        if c > best_c:
            best, best_c = lag, c
    return best, best_c / (a.std() * b.std())


def _fk_series(arm, q_series):
    """EE (pos, R) per row through the same pk.Robot FK the IK uses."""
    try:
        from robot_control import build_robot_model
        from scipy.spatial.transform import Rotation as R
    except Exception as exc:
        print(f"(FK unavailable: {exc}; skipping end-effector metrics)")
        return None
    robot, arm_data = build_robot_model("all")
    idx = arm_data[arm]["joint_indices"]
    ee = arm_data[arm]["ee_index"]
    q_full = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32).copy()
    pos, rot = [], []
    for q in q_series:
        qf = q_full.copy()
        qf[idx] = q
        fk = np.asarray(robot.forward_kinematics(qf))[ee]
        pos.append(np.asarray(fk[4:], float))
        rot.append(R.from_quat(np.roll(np.asarray(fk[:4], float), -1)).as_matrix())
    return np.array(pos), np.array(rot)


def analyze(path):
    d = np.load(path, allow_pickle=True)
    names = list(d["names"])
    n = len(names)
    dt = float(d["dt"])
    t, q_ref, q_cmd, q_meas = d["t"], d["q_ref"], d["q_cmd"], d["q_meas"]
    v_meas, eff = d["v_meas"], d["effort"]
    T = len(t)
    mode = str(d["mode"])
    print(f"\n=== {os.path.basename(path)}: {mode} on {d['arm']} "
          f"({'sim' if d['sim'] else 'real'}), {T} ticks, {T * dt:.1f} s ===")
    print(f"filter: {d['filter']}")

    # -- loop timing
    dts = np.diff(t)
    if len(dts):
        print(f"\nLOOP    dt mean {dts.mean() * 1e3:.2f} ms p95 {np.quantile(dts, .95) * 1e3:.2f} "
              f"max {dts.max() * 1e3:.2f} (nominal {dt * 1e3:.0f}); "
              f"overruns {(dts > 1.5 * dt).sum()}/{len(dts)}; "
              f"compute mean {d['loop_ms'].mean():.2f} ms max {d['loop_ms'].max():.2f} ms")
    if mode == "tube":
        s = d["solve_ms"]
        st = d["status"]
        print(f"SOLVE   mean {s.mean():.2f} ms p50 {np.median(s):.2f} p99 {np.quantile(s, .99):.2f} "
              f"max {s.max():.2f} (budget {dt * 1e3:.0f}); fallbacks {d['fallback'].sum()}/{T}; "
              f"statuses {dict(zip(*np.unique(st, return_counts=True)))}")
    print(f"DRIVER  clamp fired {d['clamped'].sum()}/{T} ticks"
          + ("  <-- should be 0 in tube mode" if mode == "tube" and d["clamped"].sum() else "")
          + f"; commands refused {d['refused'].sum()}/{T}")

    # -- limits: the point of the whole exercise
    over_hi = np.maximum(q_meas - d["q_max"], 0).max(axis=0)
    over_lo = np.maximum(d["q_min"] - q_meas, 0).max(axis=0)
    viol = int(((q_meas > d["q_max"]) | (q_meas < d["q_min"])).any(axis=1).sum())
    print(f"LIMITS  measured-position violations: {viol}/{T} ticks; worst overshoot "
          f"{max(over_hi.max(), over_lo.max()):.4f} rad")
    if bool(d["lunge"]):
        j = int(d["lunge_joint"])
        gap = d["q_max"][j] - q_meas[:, j]
        sl = slice(T // 3, 2 * T // 3)
        print(f"LUNGE   {names[j]}: reference {float(d['lunge_rad']) if 'lunge_rad' in d else 0.3:.2f} rad past q_max; "
              f"closest approach {gap[sl].min():+.4f} rad from the limit "
              f"(negative = crossed), cmd-vs-ref max intervention "
              f"{np.abs(q_cmd[sl, j] - q_ref[sl, j]).max():.3f} rad")

    # -- latency
    v_cmd = np.gradient(q_cmd, dt, axis=0)
    lags = []
    for j in range(n):
        lag, c = _xcorr_lag(v_cmd[:, j], v_meas[:, j], max_lag=int(0.4 / dt))
        lags.append((lag, c))
    print("\nLATENCY command->encoder lag per joint (ticks | ms | corr):")
    for j in range(n):
        lag, c = lags[j]
        if abs(c) < 0.3:
            print(f"   {names[j]:<22s} (joint barely moved; corr {c:.2f})")
        else:
            print(f"   {names[j]:<22s} {lag:3d} | {lag * dt * 1e3:5.0f} ms | {c:.2f}")

    # -- joint tracking
    e_cm = np.abs(q_cmd - q_meas)
    e_rm = np.abs(q_ref - q_meas)
    e_rc = np.abs(q_ref - q_cmd)
    print("\nTRACKING per joint (rad): cmd-meas mean/p95 | ref-meas mean/p95 | ref-cmd max (filter intervention)")
    for j in range(n):
        print(f"   {names[j]:<22s} {e_cm[:, j].mean():.4f}/{np.quantile(e_cm[:, j], .95):.4f} | "
              f"{e_rm[:, j].mean():.4f}/{np.quantile(e_rm[:, j], .95):.4f} | {e_rc[:, j].max():.4f}")

    # -- end effector
    fk_ref = _fk_series(str(d["arm"]), q_ref)
    if fk_ref is not None:
        p_ref, R_ref = fk_ref
        p_cmd, R_cmd = _fk_series(str(d["arm"]), q_cmd)
        p_meas, R_meas = _fk_series(str(d["arm"]), q_meas)

        def ang(Ra, Rb):
            tr = np.clip(np.einsum("nij,nij->n", Ra, Rb), -1.0, 3.0)
            return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))

        pe_cm = np.linalg.norm(p_cmd - p_meas, axis=1) * 1e3
        pe_rm = np.linalg.norm(p_ref - p_meas, axis=1) * 1e3
        pe_rc = np.linalg.norm(p_ref - p_cmd, axis=1) * 1e3
        ae_cm, ae_rm, ae_rc = ang(R_cmd, R_meas), ang(R_ref, R_meas), ang(R_ref, R_cmd)
        # "lag-compensated" tracking: shift the measured stream back by the
        # median lag so latency and shape error are reported separately
        lag_med = int(np.median([l for l, c in lags if abs(c) >= 0.3] or [0]))
        if lag_med > 0 and T > 2 * lag_med:
            pe_rm_lc = np.linalg.norm(p_ref[:-lag_med] - p_meas[lag_med:], axis=1) * 1e3
            ae_rm_lc = ang(R_ref[:-lag_med], R_meas[lag_med:])
        else:
            pe_rm_lc, ae_rm_lc = pe_rm, ae_rm
        print(f"\nEND EFFECTOR ({ARM_CONFIG[str(d['arm'])]['ee_link']})")
        print(f"   position  mm   cmd-meas {pe_cm.mean():6.1f}/{np.quantile(pe_cm, .95):6.1f} | "
              f"ref-meas {pe_rm.mean():6.1f}/{np.quantile(pe_rm, .95):6.1f} | "
              f"ref-meas lag-comp({lag_med} ticks) {pe_rm_lc.mean():6.1f}/{np.quantile(pe_rm_lc, .95):6.1f} | "
              f"ref-cmd max {pe_rc.max():6.1f}")
        print(f"   orientation deg cmd-meas {ae_cm.mean():6.2f}/{np.quantile(ae_cm, .95):6.2f} | "
              f"ref-meas {ae_rm.mean():6.2f}/{np.quantile(ae_rm, .95):6.2f} | "
              f"ref-meas lag-comp {ae_rm_lc.mean():6.2f}/{np.quantile(ae_rm_lc, .95):6.2f} | "
              f"ref-cmd max {ae_rc.max():6.2f}")
        print("   (mean/p95; cmd-meas = servo, ref-cmd = filter, ref-meas = what the operator feels)")

    # -- velocity & current
    v_ref = np.gradient(q_ref, dt, axis=0)
    print("\nVELOCITY per joint (rad/s): |v_meas| peak / p95 | |v_ref| peak | v_max")
    for j in range(n):
        print(f"   {names[j]:<22s} {np.abs(v_meas[:, j]).max():5.2f} / {np.quantile(np.abs(v_meas[:, j]), .95):5.2f} | "
              f"{np.abs(v_ref[:, j]).max():5.2f} | {d['v_max'][j]:.2f}")
    warn = float(d["effort_warn"])
    print(f"\nCURRENT per joint (mA): peak | RMS | ticks above warn ({warn:.0f})")
    for j in range(n):
        ej = np.abs(eff[:, j])
        print(f"   {names[j]:<22s} {ej.max():6.0f} | {np.sqrt((ej ** 2).mean()):6.0f} | "
              f"{(ej > warn).sum()}/{T}")
    # what the current cost, in the units the overload latch counts
    print(f"   total I^2*t over run: {float((eff.astype(float) ** 2).sum() * dt / 1e6):.1f} A^2 s "
          f"(compare runs of equal length only)")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", metavar="NPZ", help="re-print the report for a saved log")
    ap.add_argument("--mode", choices=("calib", "baseline", "tube"), default="tube")
    ap.add_argument("--arm", choices=("left", "right", "middle"), default="right")
    ap.add_argument("--sim", action="store_true", help="no ROS: simulate the plant")
    ap.add_argument("--go", action="store_true", help="actually move the arm")
    ap.add_argument("--hz", type=float, default=float(os.environ.get("GIAVA_CONTROL_HZ", "50")))
    ap.add_argument("--duration", type=float, default=15.0, help="seconds")
    ap.add_argument("--amp", type=float, default=0.15, help="sinusoid amplitude, rad")
    ap.add_argument("--freq", type=float, default=0.3, help="sinusoid frequency, Hz")
    ap.add_argument("--joints", default=None, help="comma-separated joint indices (default 1,2,4)")
    ap.add_argument("--lunge", action="store_true", help="push the reference past a joint limit")
    ap.add_argument("--lunge-joint", type=int, default=4, help="joint index for --lunge (default wrist_angle)")
    ap.add_argument("--lunge-rad", type=float, default=0.3)
    ap.add_argument("--config", default=None, help="tube_mpc FilterConfig yaml (default config.giava.yaml)")
    ap.add_argument("--state", choices=("command", "measured"), default=None)
    ap.add_argument("--lookahead", type=int, default=None)
    ap.add_argument("--v-scale", type=float, default=None,
                    help="scale the filter's velocity limits; use 0.3-0.5 for first hardware runs")
    ap.add_argument("--moving-time", type=float, default=0.14, help="as data_collection sends it")
    ap.add_argument("--accel-time", type=float, default=0.04)
    ap.add_argument("--effort-abort", type=float, default=1200.0, help="mA; stop the run above this")
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return
    path = run(args)
    if path:
        analyze(path)


if __name__ == "__main__":
    main()
