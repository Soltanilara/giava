import json
import logging
import os as _os
import time as _time
from dataclasses import dataclass, field, fields, is_dataclass
import numpy as np

logging.basicConfig(
    filename="teleop_timing.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

@dataclass
class CameraStats:
    frames_missing: int = 0
    timestamps_missing: int = 0
    lag_dt: list[float] = field(default_factory=list)

@dataclass
class ArmStats:
    ik_attempts: int = 0
    ik_successes: int = 0
    ik_failures: int = 0

    ik_position_errors: list[float] = field(default_factory=list)
    ik_orientation_errors: list[float] = field(default_factory=list)

    joint_step_norms: list[float] = field(default_factory=list)

    cmd_track_err: list[float] = field(default_factory=list)
    # Asymmetry discriminators (see log_episode_info): commanded waist angle
    # and target positions per tick.
    waist_cmds: list[float] = field(default_factory=list)
    target_positions: list = field(default_factory=list)

@dataclass
class SessionStats:
    ## Wall clock at episode start, set by reset_episode_log.  Optional so a
    ## SessionStats built by hand (tests, older callers) still summarises --
    ## it just reports no measured rate rather than a wrong one.
    t_start: float = None
    frames_added: int = 0
    ## Frames dropped BEFORE teleop was first enabled in this episode
    ## (GIAVA_RECORD_GATE=teleop).  These are the operator walking from the
    ## keyboard back to the controllers: the arm parked at the reset pose,
    ## action == reset pose, repeated.  Recording them makes the reset pose
    ## the single most common action in the dataset and teaches a policy to
    ## fall back to it whenever the scene looks static.  Counted so the cost
    ## of the habit is visible per episode.
    frames_skipped_pre_teleop: int = 0

    teleop_enable_count: int = 0
    teleop_disable_count: int = 0

    cameras: dict[str, CameraStats] = field(default_factory=dict)
    arms: dict[str, ArmStats] = field(default_factory=dict)

    loop: list[float] = field(default_factory=list)
    headset: list[float] = field(default_factory=list)
    ik_solve: list[float] = field(default_factory=list)
    # Coupled-IK solve time per tick and how often it exceeded the control
    # period (sphere collision is cheap on average but has a long tail).
    ik_solve_ms: list[float] = field(default_factory=list)
    ik_overrun_ticks: int = 0
    ## --policy (DAgger): wall time of the whole policy path per tick --
    ## state assembly, frame copy, observation build, inference or queue pop.
    ## Beside ik_solve_ms so the two command sources can be compared directly.
    policy_ms: list[float] = field(default_factory=list)
    ## --policy (DAgger): frames driven by the policy vs by the operator.
    ## Persisted so "how much of this episode was a correction" survives the
    ## session instead of scrolling off the terminal.
    dagger_policy_frames: int = 0
    dagger_human_frames: int = 0
    # Ticks where the post-solve joint clamp was saturating (only counted when
    # TeleopConfig.enable_joint_clamp is on).
    clamp_saturated_ticks: int = 0
    # Ticks where a command had to be pulled back into the driver's feasible
    # set (position or per-tick velocity), and which joints were responsible.
    driver_clamp_ticks: int = 0
    ## GIAVA_TUBE_MPC=1 only: ticks on which the tube filter's QP did not
    ## return a converged plan and the shifted-plan fallback was applied.
    ## Per-arm solve/intervention distributions ride in
    ## controller_gate.tube_mpc (see tube_mpc_hook.stats_summary).
    tube_fallback_ticks: int = 0
    ## Ticks on which the capsule gate refused the assembled command --
    ## the arms held instead of moving toward an inter-arm contact.
    capsule_gate_blocks: int = 0
    ## Ticks where the gate SHORTENED the step rather than refusing it:
    ## the arms slid up to the margin and stopped there. Common and
    ## healthy near the boundary; blocks are the harsher outcome.
    capsule_gate_scaled: int = 0
    capsule_gate_min_alpha: float = 1.0
    ## Same two counters, for the tabletop floor gate (table_gate.py) --
    ## hardware-validated but never swept, so still worth watching separately
    ## from the inter-arm gate.
    table_gate_blocks: int = 0
    table_gate_scaled: int = 0
    table_gate_min_alpha: float = 1.0
    driver_clamp_joints: dict[str, int] = field(default_factory=dict)
    # Residual misalignment of each timestep's camera frames, in seconds:
    # max minus min of the timestamps actually chosen.  Only populated when
    # GIAVA_SYNC_FRAMES is on (camera_manager.select_synchronized_frames).
    # These cameras free-run, so this is the software alignment achieved --
    # it is recorded rather than assumed.
    sync_spreads: list[float] = field(default_factory=list)
    cmd: list[float] = field(default_factory=list)
    overruns: int = 0
    ## Which collision configuration produced this session (collision_modes.py).
    collision_mode: str = "unknown"
    ## Whether tabletop avoidance was on for this session (--table).
    table_mode: str = "unknown"

def reset_episode_log(active_cameras, active_arms):
    stats = SessionStats()
    ## Stamp the collision configuration on every episode, from the
    ## environment collision_modes.select() populated at startup. Done here
    ## rather than at the call sites so no future one can forget: an
    ## episode's feel is uninterpretable without knowing which model
    ## produced it.
    stats.collision_mode = _os.environ.get("GIAVA_COLLISION", "unknown")
    stats.table_mode = _os.environ.get("GIAVA_TABLE", "unknown")

    ## WALL-CLOCK START, so the episode's ACHIEVED rate can be measured rather
    ## than assumed.  The recorded `fps` is round(1/control_dt) -- a TARGET.
    ## When the loop body overruns that budget the sleep is zero and the real
    ## period is however long the work took, so the target and the truth
    ## diverge silently and by an unbounded amount.  They did: datasets from
    ## 2026-04..06 claim 30-50 fps and were measured at 0.6-2.8 Hz, 18-48x out,
    ## which reached rollout as a policy executing its waypoints ~22x too fast.
    ## Measuring costs one clock read per episode.
    stats.t_start = _time.monotonic()

    for camera in active_cameras:
        stats.cameras[camera] = CameraStats()

    for arm in active_arms:
        stats.arms[arm] = ArmStats()

    return stats

def _mean(values):
    return np.mean(values) if values else 0.0

def _max(values):
    return np.max(values) if values else 0.0

## ---------------------------------------------------------------------------
## PER-EPISODE ROBUSTNESS RECORD
##
## log_episode_info() below prints a one-line summary into teleop_timing.log --
## useful live, useless for analysis: the log lives beside the SCRIPT, not the
## dataset, and free-text key=value lines need re-parsing.  These helpers write
## the same facts (plus the resolved config that produced them) as JSON, one
## line per episode, into <dataset_root>/meta/robustness.jsonl -- so "which
## episodes were in sync / clamp-free / hold-free" becomes a query against the
## dataset itself, and every dataset is self-describing: the exact control
## rate, scales, clamps and env overrides that shaped it travel with it.


def _dist(values):
    """Summary stats for a list of per-tick samples: n/mean/p95/max."""
    if not values:
        return None
    v = np.sort(np.asarray(values, dtype=float))
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "p95": float(v[min(len(v) - 1, int(0.95 * len(v)))]),
        "max": float(v[-1]),
    }


def _jsonable(value):
    """Best-effort conversion of config values to JSON-serializable form."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def teleop_config_snapshot(cfg):
    """The RESOLVED TeleopConfig plus every GIAVA_* env override.

    The config is env-tunable (GIAVA_CONTROL_HZ and friends), so the dataclass
    values alone do not say where they came from -- record both.  Properties
    (driver_max_step) are evaluated too: the number the clamps actually used
    matters more than the formula.
    """
    snap = {}
    if cfg is not None and is_dataclass(cfg):
        for f in fields(cfg):
            snap[f.name] = _jsonable(getattr(cfg, f.name))
        for name in ("driver_max_step",):
            try:
                snap[name] = _jsonable(getattr(cfg, name))
            except Exception:
                pass
    return {
        "teleop_config": snap,
        "env": {k: v for k, v in sorted(_os.environ.items())
                if k.startswith("GIAVA_")},
    }


def episode_stats_summary(stats):
    """SessionStats -> a JSON-serializable dict of counters and distributions."""
    ## The two numbers that make a rate claim checkable.  duration_s is wall
    ## clock over the whole episode, so it includes every overrun; measured_fps
    ## is what a consumer of this dataset would actually have to replay at.
    _dur = None
    if getattr(stats, "t_start", None) is not None:
        _dur = max(0.0, _time.monotonic() - stats.t_start)
    out = {
        "duration_s": round(_dur, 3) if _dur else None,
        "measured_fps": (round(stats.frames_added / _dur, 3)
                         if _dur and stats.frames_added else None),
        "frames_added": stats.frames_added,
        "frames_skipped_pre_teleop": stats.frames_skipped_pre_teleop,
        "teleop_enable_count": stats.teleop_enable_count,
        "teleop_disable_count": stats.teleop_disable_count,
        "overruns": stats.overruns,
        "ik_overrun_ticks": stats.ik_overrun_ticks,
        "clamp_saturated_ticks": stats.clamp_saturated_ticks,
        "driver_clamp_ticks": stats.driver_clamp_ticks,
        "driver_clamp_joints": dict(stats.driver_clamp_joints),
        "tube_fallback_ticks": stats.tube_fallback_ticks,
        "capsule_gate_blocks": stats.capsule_gate_blocks,
        "capsule_gate_scaled": stats.capsule_gate_scaled,
        "table_gate_blocks": stats.table_gate_blocks,
        "table_gate_scaled": stats.table_gate_scaled,
        "collision_mode": stats.collision_mode,
        "table_mode": stats.table_mode,
        "loop_s": _dist(stats.loop),
        "headset_s": _dist(stats.headset),
        "ik_solve_ms": _dist(stats.ik_solve_ms),
        "policy_ms": _dist(stats.policy_ms),
        "dagger_policy_frames": stats.dagger_policy_frames,
        "dagger_human_frames": stats.dagger_human_frames,
        "cmd_s": _dist(stats.cmd),
        "cam_sync_spread_s": _dist(stats.sync_spreads),
        "cameras": {
            name: {"frames_missing": cs.frames_missing,
                   "timestamps_missing": cs.timestamps_missing}
            for name, cs in stats.cameras.items()
        },
        "arms": {},
    }
    for name, a in stats.arms.items():
        out["arms"][name] = {
            "ik_attempts": a.ik_attempts,
            "ik_failures": a.ik_failures,
            "ik_pos_err": _dist(a.ik_position_errors),
            "ik_ori_err_rad": _dist(a.ik_orientation_errors),
            "joint_step_norm": _dist(a.joint_step_norms),
            "cmd_track_err": _dist(a.cmd_track_err),
        }
    return out



## How far the achieved rate may drift from the declared one before it is
## worth interrupting the operator.  Loose enough that ordinary jitter is
## silent, tight enough that a real mismatch is not.
FPS_WARN_FRAC = float(_os.environ.get("GIAVA_FPS_WARN_FRAC", "0.10"))


def _check_declared_fps(dataset_root, measured):
    """Compare the episode's achieved rate against the dataset's `fps` claim.

    THE WHOLE POINT OF THIS FUNCTION.  `fps` in meta.json is
    round(1/control_dt) -- an intention.  Nothing previously compared it to
    reality, and for four months reality was 18-48x slower: datasets claiming
    30-50 fps were captured at 0.6-2.8 Hz.  Everything downstream inherited
    it -- the report's "1.5 to 3 second" episodes, and rollouts configured to
    match, which asked policies to execute their waypoints ~22x too fast.

    The measurement was never missing.  The loop timed itself, saved the
    timings, and logged an overrun on every tick.  What was missing was any
    code that put the measurement and the claim in the same place.  This is
    that code.

    Writes `measured_fps` into meta.json alongside `fps` so a dataset carries
    its own achieved rate, and prints once per episode when they diverge.
    Never raises: the arms are live.
    """
    try:
        if not measured:
            return
        meta_path = _os.path.join(str(dataset_root), "meta.json")
        if not _os.path.exists(meta_path):
            return
        with open(meta_path) as f:
            meta = json.load(f)
        declared = meta.get("fps")

        ## Keep a running mean over the session rather than the last episode:
        ## one slow episode should move it, not define it.
        n = int(meta.get("measured_fps_n") or 0)
        prev = float(meta.get("measured_fps") or 0.0)
        running = (prev * n + float(measured)) / (n + 1)
        meta["measured_fps"] = round(running, 3)
        meta["measured_fps_n"] = n + 1
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if not declared:
            return
        frac = abs(running - declared) / declared
        if frac > FPS_WARN_FRAC:
            print("=" * 68)
            print(f"[fps] DECLARED {declared} fps, MEASURED {running:.2f} fps "
                  f"({declared / running:.1f}x out) over {n + 1} episode(s).")
            print( "      `fps` is round(1/control_dt) -- a target. When the loop")
            print( "      body overruns, the sleep is zero and the real period is")
            print( "      however long the work took. Training and rollout both")
            print( "      read `fps` as truth, so this gap becomes a policy")
            print(f"      executing its waypoints {declared / running:.1f}x too fast.")
            print( "      Fix the loop, or record at a rate the loop can hold.")
            print("=" * 68)
    except Exception as exc:
        print(f"[log] fps check failed (non-fatal): {exc}")


def write_episode_robustness(dataset_root, episode_index, outcome, stats,
                             gate_deltas=None, extra=None):
    """Append one episode's robustness record to meta/robustness.jsonl.

    Never raises: the arms are live when this runs, and a bookkeeping failure
    must not take the session down.  Returns the path, or None on failure.
    """
    try:
        record = {
            "episode_index": int(episode_index),
            "outcome": outcome,
            "wall_time": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": episode_stats_summary(stats),
        }
        if gate_deltas is not None:
            record["controller_gate"] = gate_deltas
        ## Per-episode facts that are NOT counters -- the start pose, chiefly.
        ## It lives here rather than in teleop_config.json because the
        ## operator can change it between episodes ('i' keys), so one
        ## session-level value would be wrong for most of the run.
        if extra:
            record.update(extra)
        meta_dir = _os.path.join(str(dataset_root), "meta")
        _os.makedirs(meta_dir, exist_ok=True)
        path = _os.path.join(meta_dir, "robustness.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

        _check_declared_fps(dataset_root, record["stats"].get("measured_fps"))
        return path
    except Exception as exc:
        print(f"[log] could not write robustness record: {exc}")
        return None


def write_session_config(dataset_root, cfg, extra=None):
    """Write the resolved teleop config once, at session start.

    Goes to <dataset_root>/teleop_config.json, beside camera_intrinsics.json:
    both answer the same question later -- WHAT produced this dataset.
    """
    try:
        snap = teleop_config_snapshot(cfg)
        if extra:
            snap.update(extra)
        path = _os.path.join(str(dataset_root), "teleop_config.json")
        with open(path, "w") as f:
            json.dump(snap, f, indent=2)
        return path
    except Exception as exc:
        print(f"[log] could not write teleop config snapshot: {exc}")
        return None


def log_episode_info(episode_idx, episode_stats):
    msg = [
        f"episode={episode_idx:04d}",
        f"frames_added={episode_stats.frames_added}",
        f"frames_skipped_pre_teleop="
        f"{getattr(episode_stats, 'frames_skipped_pre_teleop', 0)}",
        f"teleop_enable={episode_stats.teleop_enable_count}",
        f"teleop_disable={episode_stats.teleop_disable_count}",
        f"overruns={episode_stats.overruns}",
        f"loop_ms={1000*_mean(episode_stats.loop):.2f}",
        f"headset_ms={1000*_mean(episode_stats.headset):.2f}",
        f"ik_ms={1000*_mean(episode_stats.ik_solve):.2f}",
        f"cmd_ms={1000*_mean(episode_stats.cmd):.2f}",
    ]

    # Coupled-IK timing detail: the mean hides the tail that actually causes
    # missed ticks, so report p95/max and the overrun count explicitly.
    if episode_stats.ik_solve_ms:
        _ms = sorted(episode_stats.ik_solve_ms)
        _p95 = _ms[min(len(_ms) - 1, int(0.95 * len(_ms)))]
        msg.extend([
            f"ik_solve_mean_ms={_mean(episode_stats.ik_solve_ms):.2f}",
            f"ik_solve_p95_ms={_p95:.2f}",
            f"ik_solve_max_ms={_ms[-1]:.2f}",
            f"ik_overruns={episode_stats.ik_overrun_ticks}"
            f"/{len(episode_stats.ik_solve_ms)}",
        ])
    if episode_stats.clamp_saturated_ticks:
        msg.append(f"clamp_saturated={episode_stats.clamp_saturated_ticks}")
    ## Lead with the collision configuration: an episode's feel is only
    ## interpretable against the model that produced it.
    _cm = getattr(episode_stats, "collision_mode", None)
    if _cm and _cm != "unknown":
        msg.append(f"collision={_cm}")
    if episode_stats.capsule_gate_blocks:
        msg.append(f"capsule_gate_blocks={episode_stats.capsule_gate_blocks}")
    if episode_stats.capsule_gate_scaled:
        msg.append(
            f"capsule_gate_scaled={episode_stats.capsule_gate_scaled}"
            f"(min step {episode_stats.capsule_gate_min_alpha * 100:.0f}%)")
    _tm = getattr(episode_stats, "table_mode", None)
    if _tm and _tm not in ("unknown", "off"):
        msg.append(f"table={_tm}")
    if episode_stats.table_gate_blocks:
        msg.append(f"table_gate_blocks={episode_stats.table_gate_blocks}")
    if episode_stats.table_gate_scaled:
        msg.append(
            f"table_gate_scaled={episode_stats.table_gate_scaled}"
            f"(min step {episode_stats.table_gate_min_alpha * 100:.0f}%)")
    if episode_stats.driver_clamp_ticks:
        worst = sorted(episode_stats.driver_clamp_joints.items(),
                       key=lambda kv: -kv[1])[:3]
        msg.append(f"driver_clamped={episode_stats.driver_clamp_ticks}")
        msg.append("driver_clamp_joints=" + ",".join(f"{k}:{v}" for k, v in worst))

    if episode_stats.sync_spreads:
        sp = sorted(episode_stats.sync_spreads)
        msg.append(
            f"cam_sync_spread_ms={1000*_mean(sp):.2f}"
            f"/p95={1000*sp[min(len(sp)-1, int(0.95*len(sp)))]:.2f}"
            f"/max={1000*sp[-1]:.2f}")

    for cam_name, cam_stats in episode_stats.cameras.items():

        msg.extend([
            f"{cam_name}_missing={cam_stats.frames_missing}",
            f"{cam_name}_missing_ts={cam_stats.timestamps_missing}",
            f"{cam_name}_lag_ms={1000*_mean(cam_stats.lag_dt):.2f}",
        ])

    for arm_name, arm_stats in episode_stats.arms.items():

        success_rate = (
            arm_stats.ik_successes / arm_stats.ik_attempts
            if arm_stats.ik_attempts > 0
            else 0.0
        )

        # Asymmetry discriminators: wild waist range with a quiet target =
        # pose-dependent conditioning (target near the waist axis); wild
        # target = input noise (controller tracking / head-composition leak).
        if arm_stats.waist_cmds:
            _w = arm_stats.waist_cmds
            msg.append(f"{arm_name}_waist_range_deg={np.degrees(max(_w) - min(_w)):.1f}")
        if arm_stats.target_positions:
            _t = np.asarray(arm_stats.target_positions)
            msg.append(f"{arm_name}_target_p2p_mm={np.max(np.ptp(_t, axis=0)) * 1e3:.1f}")

        msg.extend([
            f"{arm_name}_ik_attempts={arm_stats.ik_attempts}",
            f"{arm_name}_ik_successes={arm_stats.ik_successes}",
            f"{arm_name}_ik_failures={arm_stats.ik_failures}",
            f"{arm_name}_ik_success_rate={success_rate:.3f}",

            f"{arm_name}_mean_pos_err={_mean(arm_stats.ik_position_errors):.5f}",
            f"{arm_name}_max_pos_err={_max(arm_stats.ik_position_errors):.5f}",

            f"{arm_name}_mean_ori_err_rad={_mean(arm_stats.ik_orientation_errors):.5f}",
            f"{arm_name}_max_ori_err_rad={_max(arm_stats.ik_orientation_errors):.5f}",

            f"{arm_name}_mean_joint_step={_mean(arm_stats.joint_step_norms):.5f}",
            f"{arm_name}_max_joint_step={_max(arm_stats.joint_step_norms):.5f}",

            f"{arm_name}_mean_track_err={_mean(arm_stats.cmd_track_err):.5f}",
            f"{arm_name}_max_track_err={_max(arm_stats.cmd_track_err):.5f}",
        ])

    logging.info(
        "episode_summary %s",
        " ".join(msg),
    )