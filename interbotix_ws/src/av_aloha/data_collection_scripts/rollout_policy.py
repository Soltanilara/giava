"""Roll out a trained ACT / diffusion policy on the real arm.

    python rollout_policy.py --checkpoint <.../checkpoints/050000/pretrained_model>

DRY RUN BY DEFAULT.  The loop runs, the policy is queried and everything is
logged, but no command reaches a servo until --engage is passed.  Watch a dry
run first: it prints the step the policy WANTS to take each tick, which is the
cheapest way to find out that a checkpoint has learned to lunge.

WHY THIS EXISTS RATHER THAN act_rollout_gridsearch.py
=====================================================
That script hardcodes two RealSense cameras in its own CAMERA_SERIALS, opens
them directly with pyrealsense2, and applies digital_zoom() to top_scene.  The
current recorder does none of those things: it opens cameras through
camera_manager's worker threads, records frames UNZOOMED (digital_zoom is
imported by data_collection.py and never called), and picks each tick's frames
with select_synchronized_frames() rather than latest-wins.  Every one of those
differences is a train/deploy mismatch that a policy cannot tell you about --
it just performs badly.  So this shares the recorder's camera path instead of
reimplementing it.

WHICH CAMERAS: not a flag.  The checkpoint's own config lists the image
features it was trained on, and that list is what gets opened.  A 2-camera
checkpoint opens 2 cameras; passing a camera it never saw is refused rather
than silently ignored.
"""

import argparse
import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch

try:
    import rospy
except ImportError:  # pragma: no cover - the dry-run import check
    rospy = None

## JAX PLATFORM -- must be set BEFORE anything that reaches jax is imported
## (robot_control -> pyroki -> jax), which is why this sits above the project
## imports rather than next to the gate code that uses it.
##
## CPU on purpose.  The safety gates solve tiny problems every tick, and on
## GPU the per-call launch overhead dominates: measured 11.62 ms/tick on CUDA
## against 1.67 ms/tick on CPU for the same two gates.  CPU is also the only
## choice that does not compete with a training run for GPU memory.
## Override with GIAVA_JAX_PLATFORM=gpu (an explicit JAX_PLATFORMS always
## wins -- apply() only ever setdefault()s).
try:
    from .jax_platform import apply as _apply_jax_platform
except ImportError:
    from jax_platform import apply as _apply_jax_platform
_apply_jax_platform("cpu")

if __package__:
    from .arm_config import ARM_CONFIG, POSES
    from .camera_manager import (
        CameraConfig,
        get_active_cameras,
        join_camera_workers,
        select_synchronized_frames,
        setup_cameras,
    )
    from .canonicalize_waist import canonical_branch
    from .data_col_config import ARM_MODES, TeleopConfig
    from .gripper import GRIPPER_CLOSED, GRIPPER_OPEN, command_gripper
    from .robot_control import (
        apply_profile_limits,
        create_and_configure_robots,
        get_pose,
        move_arms_together,
        move_to_named_poses,
    )
else:
    from arm_config import ARM_CONFIG, POSES
    from camera_manager import (
        CameraConfig,
        get_active_cameras,
        join_camera_workers,
        select_synchronized_frames,
        setup_cameras,
    )
    from canonicalize_waist import canonical_branch
    from data_col_config import ARM_MODES, TeleopConfig
    from gripper import GRIPPER_CLOSED, GRIPPER_OPEN, command_gripper
    from robot_control import (
        apply_profile_limits,
        create_and_configure_robots,
        get_pose,
        move_arms_together,
        move_to_named_poses,
    )


frame_lock = threading.Lock()
camera_shutdown = threading.Event()
latest_key = None
latest_line = None


def keyboard_listener():
    ## One reader for stdin.  Scoring prompts must go through latest_line
    ## rather than calling input() themselves: two threads reading stdin
    ## race, and the loser silently eats the operator's keystroke.
    global latest_key, latest_line
    while True:
        try:
            raw = input()
        except (EOFError, OSError):
            return
        latest_line = raw
        latest_key = raw.strip().lower()


def read_line(prompt):
    """Block until the listener thread delivers one line. Returns raw text."""
    global latest_key, latest_line
    latest_key = None
    latest_line = None
    print(prompt, end="", flush=True)
    while latest_line is None:
        if rospy is not None and rospy.is_shutdown():
            return ""
        time.sleep(0.05)
    return latest_line


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def resolve_checkpoint(path):
    """Accept any of the three things an operator will actually type.

    A checkpoint directory (.../checkpoints/050000), its pretrained_model
    subdirectory, or the job output directory (whose checkpoints/last symlink
    is the newest).  Returns the directory that holds config.json.
    """
    p = Path(path).expanduser().resolve()
    candidates = [
        p,
        p / "pretrained_model",
        p / "checkpoints" / "last" / "pretrained_model",
    ]
    for c in candidates:
        if (c / "config.json").exists() and (c / "model.safetensors").exists():
            return c
    raise SystemExit(
        f"No policy checkpoint under {p}.\n"
        f"Looked for config.json + model.safetensors in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nPoint --checkpoint at a checkpoints/<step>/pretrained_model "
          "directory, or at the job output directory to use checkpoints/last."
    )


def load_policy(ckpt_dir, device, temporal_ensemble=None):
    """(policy, preprocessor, postprocessor, image_keys, state_dim, action_dim).

    Loaded through PreTrainedConfig so the policy TYPE comes off the
    checkpoint -- an ACT and a diffusion checkpoint load through the same call
    and the operator never has to remember which one this directory is.
    """
    ## IMPORT ORDER IS LOAD-BEARING.  Importing anything out of
    ## lerobot.configs BEFORE lerobot.policies.factory segfaults this build
    ## (verified: `from lerobot.configs.policies import PreTrainedConfig`
    ## followed by `from lerobot.policies.factory import get_policy_class`
    ## dies with SIGSEGV before either name is bound; the reverse order is
    ## fine, and so is factory on its own).  Not a path or shadowing problem
    ## -- it reproduces with no local directory on sys.path.  Keep factory
    ## first.
    from lerobot.policies.factory import (
        get_policy_class,
        make_pre_post_processors,
    )
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    if temporal_ensemble is not None:
        ## ACT's own closed-loop mode: infer EVERY tick, execute the
        ## exponentially-weighted average of every chunk that covers this
        ## step (w_i = exp(-coeff * age), oldest heaviest).  Closed-loop
        ## without chunk-boundary kinks, and older, cleaner-view predictions
        ## keep a vote -- the thing a short n_action_steps throws away.  The
        ## ensembler is built in the policy's __init__ from the config, so
        ## the config MUST be edited before from_pretrained and passed in.
        if cfg.type != "act":
            raise SystemExit("--temporal-ensemble is an ACT feature")
        cfg.temporal_ensemble_coeff = float(temporal_ensemble)
        cfg.n_action_steps = 1
    policy = get_policy_class(cfg.type).from_pretrained(ckpt_dir, config=cfg)
    policy.to(device)
    policy.eval()

    ## Built by lerobot's OWN factory rather than by calling
    ## PolicyProcessorPipeline.from_pretrained twice by hand.  The two
    ## pipelines need different converters -- the preprocessor takes a batch
    ## dict, the postprocessor takes a raw action Tensor
    ## (policy_action_to_transition / transition_to_policy_action) -- and
    ## from_pretrained defaults BOTH to the batch converters.  Hand-built that
    ## way, postprocessor(action) dies with "EnvTransition must be a
    ## dictionary. Got Tensor" on the first inference.  The factory also runs
    ## _reconnect_relative_absolute_steps() across the pair, which is what
    ## keeps relative-action policies consistent.
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=str(ckpt_dir))

    image_keys = sorted(k for k in cfg.input_features
                        if k.startswith("observation.images."))
    state_dim = int(np.prod(cfg.input_features["observation.state"].shape))
    action_dim = int(np.prod(cfg.output_features["action"].shape))
    return (policy, preprocessor, postprocessor, image_keys,
            state_dim, action_dim, cfg)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def read_arm_state(bot, n_joints, has_gripper):
    """observation.state for one arm, in the recorder's layout.

    MUST match dataset.build_frame's `_flatten("joints", "gripper")`: joints
    then gripper, measured (not commanded), float32.  A different order here
    is a silent disaster -- the policy gets a valid-looking vector with the
    gripper where a wrist angle should be.
    """
    js = bot.dxl.joint_states
    q = np.asarray(js.position[:n_joints], dtype=np.float32)
    if not has_gripper:
        return q
    g = np.float32(js.position[n_joints])
    return np.concatenate([q, [g]], axis=0)


# ---------------------------------------------------------------------------
# Safety gates
# ---------------------------------------------------------------------------

class SafetyGates:
    """The capsule + table gates from data collection, on the rollout command.

    WHY THIS BELONGS HERE.  Both gates were ACTIVE for all 50 episodes of the
    training set and fired ZERO times (robustness.jsonl: capsule_gate_blocks,
    capsule_gate_scaled, table_gate_blocks, table_gate_scaled all 0).  So they
    contributed nothing to the recorded actions -- the demonstrations are
    unmodified operator commands -- and running them at rollout therefore
    introduces NO train/test mismatch: for any command inside the demonstrated
    envelope alpha is 1.0 and the gate is invisible, exactly as during
    collection.  They engage only when the policy commands something no
    demonstration did, which is both the thing that can break the rig and a
    measurement worth logging.

    Each gate returns the largest fraction of the step that stays clear.  The
    two are chained (both shrink-only, so the result is at least as
    conservative as either), and the SENT command is what the next tick
    measures its step from -- otherwise a held command would be re-proposed
    from a position the arm never reached.

    NOT applied to parking moves: the 'rest' pose sits 1.3 mm above the table,
    inside the gate's own 15 mm structural margin, so gating the park would
    refuse to let the arm stand down.  Collection parks outside the gated
    loop for the same reason.
    """

    def __init__(self, mode, arm_names, want_capsule=True, want_table=True,
                 jax_platform="cpu", inactive_pose="rest"):
        ## CPU by default: the GPU is usually busy training, and the gates
        ## cost ~0.25 ms/tick on CPU against a 20 ms budget.  Must be set
        ## before anything imports jax.
        try:
            from .jax_platform import apply as _apply_jax
        except ImportError:
            from jax_platform import apply as _apply_jax
        _apply_jax(jax_platform)

        try:
            from .arm_config import ARM_CONFIG, URDF_PATH
            from .robot_control import build_robot_model
            from .study_ik import CoupledStudyIK
        except ImportError:
            from arm_config import ARM_CONFIG, URDF_PATH
            from robot_control import build_robot_model
            from study_ik import CoupledStudyIK
        from yourdfpy import URDF

        self.robot, self.arm_data = build_robot_model(mode)
        urdf = URDF.load(URDF_PATH)

        self.capsule = self.table = None
        if want_capsule:
            try:
                from .capsule_gate import build_gate as bc
            except ImportError:
                from capsule_gate import build_gate as bc
            self.capsule = bc(self.robot, urdf)
        if want_table:
            try:
                from .table_gate import build_gate as bt
            except ImportError:
                from table_gate import build_gate as bt
            self.table = bt(self.robot, urdf)

        self.ik = CoupledStudyIK(
            self.robot, URDF_PATH,
            ee_links={a: ARM_CONFIG[a]["ee_link"]
                      for a in ("left", "right", "middle")},
            control_dt=0.02)

        self.n_act = self.robot.joints.num_actuated_joints
        self.idx = {a: self.arm_data[a]["joint_indices"] for a in arm_names}
        ## INACTIVE ARMS ARE NOT AT URDF ZERO.  Left at zeros, the left arm's
        ## gripper sits at x=+0.05 z=+0.47 -- mid-air, right through the
        ## workspace -- and the inter-arm capsule gate then blocks perfectly
        ## legal motion against an arm that is actually parked at rest.
        ## Measured on a --mode right_av rollout before this fix: 252 of 1892
        ## ticks BLOCKED (min_alpha 0.0), which does not make the rollout
        ## safer, it makes it a different policy.  Assume they are at
        ## `inactive_pose`; they are not sensed, so if one is parked somewhere
        ## else the gate is checking the wrong geometry either way.
        self.q_cmd = np.zeros(self.n_act, dtype=float)
        try:
            from .arm_config import ARM_CONFIG as _AC, POSES as _POSES
        except ImportError:
            from arm_config import ARM_CONFIG as _AC, POSES as _POSES
        inactive = [a for a in ("left", "right", "middle") if a not in arm_names]
        for a in inactive:
            if inactive_pose not in _POSES[a]:
                raise SystemExit(
                    f"--inactive-pose={inactive_pose!r} is not a pose {a} has. "
                    f"Available for {a}: {sorted(_POSES[a])}")
            idx = [self.robot.joints.actuated_names.index(j)
                   for j in _AC[a]["joint_names"]]
            n = _AC[a]["num_joints"]
            self.q_cmd[idx] = np.asarray(_POSES[a][inactive_pose],
                                         dtype=float)[:n]
        if inactive:
            print(f"[gates] inactive arm(s) {inactive} assumed at "
                  f"'{inactive_pose}' (NOT sensed -- --inactive-pose to change)")
        self.reset_counters()

    def reset_counters(self):
        self.blocks = {"capsule": 0, "table": 0}
        self.scaled = {"capsule": 0, "table": 0}
        self.min_alpha = {"capsule": 1.0, "table": 1.0}
        self.limiters = {}

    def seed(self, arm, q_meas):
        """Start the step reference at the measured pose (episode start)."""
        self.q_cmd[self.idx[arm]] = np.asarray(q_meas, dtype=float)

    def filter(self, arm, q_target):
        """(q_sent, alpha, info). alpha 0 means HOLD -- do not command."""
        idx = self.idx[arm]
        q_prev_full = self.q_cmd.copy()
        q_next_full = q_prev_full.copy()
        q_next_full[idx] = np.asarray(q_target, dtype=float)
        u_prev = self.ik.driver_to_urdf(q_prev_full)
        u_next = self.ik.driver_to_urdf(q_next_full)

        alpha = 1.0
        info = {}
        for name, gate in (("capsule", self.capsule), ("table", self.table)):
            if gate is None:
                continue
            a, dist, who = gate.largest_safe_fraction(u_prev, u_next)
            a = float(a)
            if a < 1.0:
                self.min_alpha[name] = min(self.min_alpha[name], a)
                info[name] = {"alpha": round(a, 4),
                              "dist_mm": round(float(dist) * 1e3, 2),
                              "limiter": str(who)}
                if who is not None:
                    key = f"{name}:{who}"
                    self.limiters[key] = self.limiters.get(key, 0) + 1
                if a <= 0.0:
                    self.blocks[name] += 1
                else:
                    self.scaled[name] += 1
            alpha = min(alpha, a)

        if alpha <= 0.0:
            return self.q_cmd[idx].copy(), 0.0, info
        q_sent = (q_prev_full[idx]
                  + alpha * (np.asarray(q_target, dtype=float)
                             - q_prev_full[idx]))
        self.q_cmd[idx] = q_sent
        return q_sent, alpha, info

    def summary(self):
        return {"blocks": dict(self.blocks), "scaled": dict(self.scaled),
                "min_alpha": {k: round(v, 4) for k, v in self.min_alpha.items()},
                "limiters": dict(self.limiters),
                "capsule": self.capsule is not None,
                "table": self.table is not None}


# ---------------------------------------------------------------------------
# Rollout scoring
# ---------------------------------------------------------------------------

## Per-scene operator stages, in order.  The shape sorter scores "hole"
## (hovered over the CORRECT hole) separately from "inserted": which-hole is
## the perceptual question the object-centric input is meant to answer,
## insertion is a precision question no input representation fixes.
STAGE_SETS = {
    "flower": [("a", "approach"), ("g", "grasp"),
               ("t", "transport"), ("p", "placement")],
    "shape_sorter": [("a", "approach"), ("g", "grasp"), ("t", "transport"),
                     ("h", "hole"), ("i", "inserted")],
}


def make_extractor(scene, target=None, env_dim=None):
    """The scene's object-centric feature extractor (see scene_features.py /
    shape_sorter.py).  Used BOTH to feed observation.environment_state to a
    checkpoint that takes it and to track the object for --score, so the
    rollout measures with the same eyes the policy has.  For the shape sorter
    the checkpoint's env dim says which variant it trained on: 8 = geometry
    (centroid + hole), 4 = class one-hot; no env input = geometry, for
    tracking only."""
    if scene == "shape_sorter":
        try:
            import shape_sorter as ss
        except ImportError:
            from . import shape_sorter as ss
        if env_dim == ss.ONEHOT_FEATURE_DIM:
            return ss.OneHotExtractor(target)
        return ss.Extractor(target)
    try:
        import scene_features as sf
    except ImportError:
        from . import scene_features as sf
    return sf.Extractor()


def object_track(frames_by_tick, extractor):
    """[(tick, t, cx, cy, found)] for the top_scene samples of one episode.

    Returns [] if the scene has no calibration yet (shape_sorter needs
    assets/shape_sorter_targets.json for its ROI and hole centres).  Automatic
    tracking is a convenience; losing it must never cost the operator a trial
    they just ran, so this degrades to manual scoring instead of raising."""
    out = []
    for tick, t, img in frames_by_tick:
        try:
            cx, cy, _area, ok = extractor.detect(img)
        except (FileNotFoundError, KeyError) as exc:
            print(f"    [score] no automatic tracking: {exc}")
            print(f"    [score] scoring by hand -- run measure_targets.py to "
                  f"enable the automatic measurements")
            return []
        out.append((tick, t, cx, cy, bool(ok)))
    return out


def auto_metrics(track, rows, arm, px_per_cm=None, targets=None):
    """Measurements that do not need an operator: where the object ended up,
    how far it travelled, and when the gripper actually closed.

    `targets` = {name: (x, y)} from the extractor; each becomes a
    final_dist_<name>_px.  Reported in PIXELS always; centimetres only when
    --px-per-cm is given, so a distance is never printed in units nobody
    calibrated.
    """
    m = {}
    seen = [p for p in track if p[4]]
    m["obj_found_frac"] = (len(seen) / len(track)) if track else None
    if seen:
        sx, sy = seen[0][2], seen[0][3]
        ex, ey = seen[-1][2], seen[-1][3]
        m["obj_start_px"] = [round(sx, 1), round(sy, 1)]
        m["obj_end_px"] = [round(ex, 1), round(ey, 1)]
        m["obj_displacement_px"] = round(float(np.hypot(ex - sx, ey - sy)), 1)
        for name, (tx, ty) in (targets or {}).items():
            d = float(np.hypot(ex - tx, ey - ty))
            m[f"final_dist_{name}_px"] = round(d, 1)
            m[f"initial_dist_{name}_px"] = round(float(np.hypot(sx - tx, sy - ty)), 1)
            if px_per_cm:
                m[f"final_dist_{name}_cm"] = round(d / px_per_cm, 2)
        m["obj_last_seen_t"] = round(seen[-1][1], 2)

    ## Gripper: OPEN=0.0, CLOSED=-1.5 (gripper.py). "Closed" here is past the
    ## midpoint, which is a command to squeeze, not necessarily a grasp -- the
    ## operator's g flag is what says whether anything was actually held.
    key = f"{arm}_gripper_cmd"
    g = [(r["t"], r[key]) for r in rows if key in r]
    if g:
        vals = [v for _t, v in g]
        m["grip_min"] = round(min(vals), 3)
        m["grip_max"] = round(max(vals), 3)
        closed = [t for t, v in g if v < -0.75]
        m["t_first_close"] = round(closed[0], 2) if closed else None
        m["t_last_close"] = round(closed[-1], 2) if closed else None
        m["close_frac"] = round(len(closed) / len(g), 3)
    return m


## Pixel thresholds for the auto-suggested stages.  Rough on purpose -- they
## draft the operator's answer, they do not replace it.
SUGGEST_MOVED_PX = 15       # target piece displaced at all
SUGGEST_NEAR_PX = 30        # "at the target": ~3 cm on top_scene


def suggest_stages(metrics, stages):
    """Draft {stage: bool} from the automatic measurements.  Reasoning:
    grasp = gripper closed AND the target moved; transport = it ended closer
    to the target than it started (by half); placement/hole = it ended within
    SUGGEST_NEAR_PX; inserted = it got there and then vanished from top_scene
    with the gripper open (dropped into the box)."""
    names = [n for _k, n in stages]
    tgt = next((k[len("final_dist_"):-len("_px")] for k in metrics
                if k.startswith("final_dist_") and k.endswith("_px")
                and k != "final_dist_oval_px"), None)
    moved = (metrics.get("obj_displacement_px") or 0) > SUGGEST_MOVED_PX
    closed = metrics.get("t_first_close") is not None
    final = metrics.get(f"final_dist_{tgt}_px") if tgt else None
    initial = metrics.get(f"initial_dist_{tgt}_px") if tgt else None
    grasp = closed and moved
    transport = grasp and final is not None and initial and final < 0.5 * initial
    ## The final distance is measured from the LAST FRAME THE OBJECT WAS SEEN.
    ## In the sorter the cube is in the gripper for most of the episode --
    ## obj_found_frac ran 10-38% over the 2026-09-16 r2 run -- so that last
    ## sighting can predate the insertion attempt entirely, and `near` then
    ## says "not placed" for an episode the operator noted as a success.
    ## Every cube run on record scored placement 0 while the notes describe
    ## insertions; this is why.  Below half visibility the metric earns no
    ## opinion, and score_episode makes the operator answer instead.
    track_ok = (metrics.get("obj_found_frac") or 0) >= 0.5
    near = (grasp and track_ok and final is not None
            and final < SUGGEST_NEAR_PX)
    gone = (metrics.get("obj_last_seen_t") is not None
            and metrics.get("elapsed_s") is not None
            and metrics["elapsed_s"] - metrics["obj_last_seen_t"] > 1.5
            and (metrics.get("close_frac") or 0) < 0.95)
    guess = {
        "approach": closed or moved, "grasp": grasp, "transport": transport,
        "placement": near, "hole": near, "inserted": near and gone,
    }
    return {n: bool(guess.get(n, False)) for n in names}


def score_episode(metrics, ep, n_eps, stages):
    """Ask the operator what actually happened. Returns a dict, or None to
    discard the episode (a rig fault, not a policy failure)."""
    print(f"\n  --- score episode {ep + 1}/{n_eps} ---")
    bits = []
    if metrics.get("obj_displacement_px") is not None:
        bits.append(f"obj moved {metrics['obj_displacement_px']:.0f} px")
    for k in sorted(metrics):
        if k.startswith("final_dist_") and k.endswith("_px"):
            name = k[len("final_dist_"):-len("_px")]
            d = f"{metrics[k]:.0f} px"
            if metrics.get(f"final_dist_{name}_cm") is not None:
                d += f" ({metrics[f'final_dist_{name}_cm']:.1f} cm)"
            bits.append(f"final obj->{name} {d}")
    if metrics.get("t_first_close") is not None:
        bits.append(f"gripper closed t={metrics['t_first_close']:.1f}s")
    elif "grip_min" in metrics:
        bits.append("gripper never closed")
    if metrics.get("obj_found_frac") is not None:
        bits.append(f"obj visible {metrics['obj_found_frac']:.0%}")
    if bits:
        print("      auto: " + "  |  ".join(bits))

    ## FLAGS are not stages.  'c' = the operator intervened (paused and
    ## corrected the scene, or helped the grasp) and the policy carried on
    ## from there -- the stages after the correction are then the policy's,
    ## but the episode is not a clean trial.  'w' (shape sorter) = it grasped
    ## a piece that was not the target: distinct from a missed grasp, it is
    ## the multimodality / wrong-conditioning failure the env input exists to
    ## prevent, so it gets its own column instead of a free-text note.
    flags = [("c", "corrected")]
    if len(stages) == 5:
        flags.append(("w", "wrong_piece"))
    guess = suggest_stages(metrics, stages)
    guess_letters = "".join(k for k, n in stages if guess[n])
    if metrics.get("interventions"):
        guess_letters += "c"
    print("      stages reached?  " +
          "  ".join(f"{k}={name}" for k, name in stages) +
          "   flags: " + "  ".join(f"{k}={n}" for k, n in flags))
    _seen = metrics.get("obj_found_frac")
    _blind = _seen is not None and _seen < 0.5
    if _blind:
        print(f"      [!] the object was visible in only {_seen:.0%} of frames, "
              f"so the final-distance number above is measured from a stale "
              f"sighting.  The suggestion CANNOT see whether it went in --")
        print(f"      [!] type the stages yourself; ENTER alone is refused here.")
    print(f"      auto suggests '{guess_letters or '-'}'   "
          f"({'letters required' if _blind else 'ENTER=accept'}, "
          f"letters=override, -=none, x=discard)")
    raw = read_line("    > ").strip().lower()
    while _blind and raw == "":
        print("      [!] no default here -- type the letters reached, "
              "'-' for none, or x to discard.")
        raw = read_line("    > ").strip().lower()
    if raw == "x":
        print("      [discarded]")
        return None
    if raw == "":
        raw = guess_letters
    got = {name: (k in raw) for k, name in stages}
    flagged = {name: (k in raw) for k, name in flags}
    note = read_line("    note (ENTER to skip) > ").strip()
    print("      [scored] " +
          " ".join(f"{n}={'1' if v else '0'}" for n, v in got.items()) +
          "".join(f"  {n}" for n, v in flagged.items() if v) +
          (f"  note: {note}" if note else ""))
    return {"stages": got,
            "furthest": max((i + 1 for i, (_k, n) in enumerate(stages)
                             if got[n]), default=0),
            **flagged,
            "note": note}


# ---------------------------------------------------------------------------
## Warm-up passes allowed before episode 0 starts its clock.  Bounded so a
## genuinely slow policy cannot stall the run forever waiting to get fast.
_WARMUP_MAX_PASSES = 12

## The rate guard's measurement window, in ticks.  Starts after the opening
## transient rather than at t0; see the guard for why.
_RATE_WINDOW_START = 30
_RATE_WINDOW_END = 90


def _slug(text, limit=32):
    """A filename-safe fragment of `text`, or "" if there is nothing usable.

    Deliberately lossy: lowercase, non-alphanumerics collapsed to single
    underscores, truncated on a word boundary.  It only has to be recognisable
    at a glance in a directory listing -- the authoritative task string is in
    the run's jsonl, so nothing downstream parses this back."""
    if not text:
        return ""
    out, prev_us = [], True     # prev_us=True so a leading separator is eaten
    for ch in str(text).lower():
        if ch.isalnum():
            out.append(ch); prev_us = False
        elif not prev_us:
            out.append("_"); prev_us = True
    slug = "".join(out).strip("_")
    if len(slug) <= limit:
        return slug
    cut = slug[:limit]
    return (cut.rsplit("_", 1)[0] if "_" in cut else cut).strip("_")


# Rollout video
# ---------------------------------------------------------------------------

## Starting-scene snapshots + ghosted alignment preview live in
## scene_snapshots.py, shared with data_collection.py: a scene replicated for
## a DAgger correction is then the same scene an evaluation replicates.
from scene_snapshots import save_snapshot, load_snapshot, alignment_view  # noqa: E402


class RolloutVideo:
    """Write what the POLICY SAW to an mp4, off the control thread.

    Evidence for a rollout has to be the policy's own view: a third-person
    clip shows that the arm missed, this shows what it was looking at when it
    decided to.  Frames are the same arrays build_observation just handed the
    network, so the video cannot drift from the input the way a separately
    opened camera would.

    The control loop must never pay for encoding.  submit() only copies onto a
    bounded queue; a worker thread tiles, annotates and encodes.  If the
    encoder falls behind, frames are DROPPED and counted -- the arm is not
    stalled to finish a video.
    """

    def __init__(self, path, cameras, fps, scale=0.5, queue_size=90):
        import queue as _queue

        self.path = Path(path)
        self.cameras = list(cameras)
        self.fps = float(fps)
        self.scale = float(scale)
        self.dropped = 0
        self.written = 0
        self._writer = None
        self._q = _queue.Queue(maxsize=queue_size)
        self._Empty = _queue.Empty
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frames, episode, tick, t):
        """Copy frames onto the queue. Never blocks; drops when full."""
        if self._stop.is_set():
            return
        try:
            self._q.put_nowait(
                ({c: frames[c].copy() for c in self.cameras if
                  frames.get(c) is not None}, episode, tick, t))
        except Exception:
            self.dropped += 1

    def _compose(self, frames, episode, tick, t):
        import cv2

        tiles = []
        for cam in self.cameras:
            img = frames.get(cam)
            if img is None:
                continue
            ## Frames are RGB (realsense is opened rgb8, the OAK worker
            ## converts); cv2 writes BGR.
            img = img[..., ::-1]
            if self.scale != 1.0:
                img = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                                 interpolation=cv2.INTER_AREA)
            img = np.ascontiguousarray(img)
            cv2.putText(img, cam, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, cam, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        if not tiles:
            return None
        h = max(t_.shape[0] for t_ in tiles)
        tiles = [t_ if t_.shape[0] == h else
                 cv2.copyMakeBorder(t_, 0, h - t_.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
                 for t_ in tiles]
        canvas = np.hstack(tiles)
        label = f"ep {episode}  tick {tick:4d}  t={t:5.2f}s"
        cv2.putText(canvas, label, (6, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (6, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                    cv2.LINE_AA)
        return canvas

    def _run(self):
        import cv2

        while not (self._stop.is_set() and self._q.empty()):
            try:
                item = self._q.get(timeout=0.2)
            except self._Empty:
                continue
            try:
                canvas = self._compose(*item)
                if canvas is None:
                    continue
                if self._writer is None:
                    h, w = canvas.shape[:2]
                    ## H.264 FIRST.  "mp4v" is MPEG-4 Part 2, which produces a
                    ## structurally valid .mp4 that Chrome, Firefox, Safari,
                    ## QuickTime, Slack and most previewers refuse to decode --
                    ## the file looks fine on disk and plays nowhere except
                    ## VLC/mpv.  That is what made these rollouts unviewable.
                    ## avc1 needs an OpenCV built with H.264; keep mp4v as the
                    ## fallback so a run never loses its video over a codec,
                    ## but say so, because the result will not preview.
                    for _tag in ("avc1", "mp4v"):
                        self._writer = cv2.VideoWriter(
                            str(self.path), cv2.VideoWriter_fourcc(*_tag),
                            self.fps, (w, h))
                        if self._writer.isOpened():
                            if _tag != "avc1":
                                print(f"[video] H.264 unavailable in this "
                                      f"OpenCV build -- falling back to {_tag}. "
                                      f"{self.path.name} will need VLC/mpv, or "
                                      f"re-encode: ffmpeg -i IN -c:v libx264 OUT")
                            break
                        self._writer.release()
                        self._writer = None
                    if self._writer is None:
                        print(f"[video] could not open {self.path}")
                        self._stop.set()
                        return
                self._writer.write(canvas)
                self.written += 1
            except Exception as exc:  # a broken frame must not kill a rollout
                self.dropped += 1
                if self.dropped in (1, 50):
                    print(f"[video] encode error: {exc}")

    def close(self):
        self._stop.set()
        self._thread.join(timeout=10.0)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self.written, self.dropped


def build_observation(image_keys, cameras, state, device, task=None, latest=None):
    """Policy-ready batch, or (None, reason) when a camera has no frame yet.

    Frames come from select_synchronized_frames -- the same nearest-timestamp
    selection the recorder used -- so the images the policy sees are aligned
    the way its training images were.  There is deliberately NO latest-wins
    fallback: that was the pre-sync behaviour, no current dataset was recorded
    with it, and a switch that silently changes what the policy sees relative
    to its training data is not a debug aid.
    """
    with frame_lock:
        if latest is not None:
            ## LATEST-WINS, the recorder's DAgger path.  select_synchronized_
            ## frames matches every camera to the newest instant ALL of them
            ## have covered -- so one lagging camera drags every frame the
            ## policy sees back to that lagging moment, with a small "spread"
            ## that looks healthy.  This bypasses that entirely.
            frames = {c: (latest.get(c).copy() if latest.get(c) is not None else None)
                      for c in cameras}
            info = {"reference_s": None, "spread_s": None, "per_camera": {},
                    "cameras_missing": [c for c in cameras if frames[c] is None]}
        else:
            frames, _ts, info = select_synchronized_frames(cameras)
            frames = {c: (f.copy() if f is not None else None)
                      for c, f in frames.items()}
        ## Frame AGE, not just spread: how far behind wall-clock the frames
        ## the policy is about to act on actually are.  All camera clocks are
        ## mapped to the host epoch (camera_manager), so this is comparable.
        _now = time.time()
        info["age_s"] = ((_now - info["reference_s"]) if info.get("reference_s")
                         else None)
        info["per_camera_age_s"] = {c: round(_now - v["timestamp_s"], 4)
                                    for c, v in (info.get("per_camera") or {}).items()}

    missing = [c for c in cameras if frames.get(c) is None]
    if missing:
        return None, f"no frame yet from {', '.join(missing)}", info, None

    obs = {"observation.state":
           torch.from_numpy(state).unsqueeze(0).to(device)}
    ## LANGUAGE-CONDITIONED POLICIES NEED THE INSTRUCTION EVERY TICK.  lerobot
    ## routes a top-level "task" into complementary_data (converters.py
    ## _COMPLEMENTARY_KEYS), where SmolVLA's tokenizer step reads it; without
    ## it the preprocessor raises KeyError: 'task' on the first inference.
    ## ACT ignores complementary data, so this is harmless for every policy.
    if task is not None:
        obs["task"] = [task]
    for key in image_keys:
        cam = key.removeprefix("observation.images.")
        ## HWC uint8 -> CHW float in [0,1], batched.  This is exactly what
        ## LeRobotDataset hands the policy at training time; the preprocessor
        ## does normalization from there.
        obs[key] = (
            torch.from_numpy(np.ascontiguousarray(frames[cam]))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
    return obs, None, info, frames


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

def clamp_action(action, measured, lower, upper, max_step, n_joints,
                 has_gripper):
    """Bound one action before it reaches a servo. Returns (q, gripper, hit).

    TWO INDEPENDENT BOUNDS, both necessary:

      step   -- |command - MEASURED| per tick.  A policy that has not
                converged emits jumps; the driver would accept a large one and
                run the arm there at profile speed.  Clamping against the
                measured position (not the last command) means the bound also
                holds when the arm is lagging or has been stopped by contact.

                CHOOSING max_step.  This bound is NOT a small number by
                nature: the recorded action LEADS the measured state by
                whatever the servos are lagging, so a healthy in-distribution
                tick already carries a sizeable |command - measured|.
                Measured on transfer_flower/20260903_214256 (50 episodes,
                37k ticks), the fraction of TRAINING ticks a given bound
                would have clamped:

                    0.05 rad -> 46.6 %      <- fights normal behaviour
                    0.10 rad ->  7.3 %
                    0.15 rad ->  3.1 %      <- default
                    0.20 rad ->  2.1 %
                    0.40 rad ->  0.5 %

                A bound that trips on half the training distribution does not
                make the arm safer, it makes it a different arm: every command
                gets dragged back toward the current position, the motion is
                slowed and distorted, and the policy is driven out of the
                distribution it was fitted on -- which is the failure the
                clamp was supposed to prevent.  Pick a value that only the
                genuine tail reaches.  The driver has its own, looser bound
                underneath this one (driver_max_step ~0.396 rad/tick), which
                fired 13 times in those same 37k ticks.
      limit  -- the driver's own joint_lower/upper_limits, so a clamped step
                can never walk the arm into a limit over many ticks.
    """
    action = np.asarray(action, dtype=float).reshape(-1)
    q = action[:n_joints]
    hit = {}

    step = q - measured[:n_joints]
    over = np.abs(step) > max_step
    if np.any(over):
        hit["step"] = float(np.max(np.abs(step)))
        q = measured[:n_joints] + np.clip(step, -max_step, max_step)

    clipped = np.clip(q, lower, upper)
    if not np.allclose(clipped, q):
        hit["limit"] = True
    q = clipped

    gripper = None
    if has_gripper:
        gripper = float(np.clip(action[n_joints],
                                min(GRIPPER_CLOSED, GRIPPER_OPEN),
                                max(GRIPPER_CLOSED, GRIPPER_OPEN)))
    return q, gripper, hit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Roll out a trained policy on the real arm. Dry run "
                    "unless --engage is given.")
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoints/<step>/pretrained_model, or the job "
                         "output dir to use checkpoints/last")
    ap.add_argument("--engage", action="store_true",
                    help="ACTUALLY COMMAND THE ARM. Without it nothing moves.")
    ap.add_argument("--mode", default="right",
                    help=f"arm mode ({sorted(ARM_MODES)})")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="wall-clock cap per rollout")
    ap.add_argument("--hz", type=float, default=None,
                    help="control rate; default: the checkpoint's dataset fps")
    ap.add_argument("--start-pose", default=None,
                    help="pose to park at before each rollout -- one name for "
                         "every arm, or per-arm as "
                         "'right=forward,middle=forward_demo'; default: the "
                         "start_pose recorded in the training run's "
                         "teleop_config.json, else 'forward'")
    ap.add_argument("--dataset", default=None,
                    help="training run dir, to read fps + start pose from. "
                         "Default: whatever the checkpoint's train_config.json "
                         "says it trained on.")
    ap.add_argument("--quit-pose", default="rest",
                    help="pose to park at when the session ends; 'rest' lets "
                         "the arm down onto its stops instead of holding its "
                         "own weight on servo torque")
    ap.add_argument("--latest-frames", action="store_true",
                    help="feed the policy each camera's NEWEST frame (the "
                         "recorder's DAgger path) instead of the "
                         "timestamp-synchronized set. Use to test whether "
                         "frame staleness from a lagging camera is what the "
                         "policy is suffering from.")
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                    help="ACT temporal ensembling: run the network every tick "
                         "and execute the weighted average of all chunks that "
                         "cover this step (paper value 0.01). The principled "
                         "closed-loop mode -- unlike a short --n-action-steps "
                         "it keeps older, cleaner-view predictions in the vote.")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="how many actions of each predicted chunk to execute "
                         "before re-querying the policy. Default: whatever the "
                         "checkpoint was configured with (ACT ships chunk_size, "
                         "i.e. FULLY OPEN LOOP for the whole chunk). Lowering "
                         "it makes the rollout more closed-loop at the cost of "
                         "more inference calls; it needs no retraining.")
    ap.add_argument("--max-step", type=float, default=0.15,
                    help="max |command - measured| per joint per tick, rad. "
                         "SIZE THIS FROM THE TRAINING DATA, not from intuition "
                         "-- see the note in clamp_action().")
    ap.add_argument("--checkpoint-b", default=None, metavar="DIR",
                    help="SECOND checkpoint, for a scene-paired A/B. Each "
                         "scene is then rolled out TWICE, once per policy, "
                         "back to back and in a randomised order, and the "
                         "operator is not told which is which. This is the "
                         "only design that survives what the logs show: the "
                         "SAME checkpoint scored 8/17 and 4/17 on the same "
                         "sub-pixel-replicated scenes a day apart, so a "
                         "comparison against yesterday's numbers measures the "
                         "day, not the policy.")
    ap.add_argument("--ab-seed", type=int, default=0,
                    help="seed for the per-scene A/B order (default 0)")
    ap.add_argument("--unblind", action="store_true",
                    help="A/B: print which checkpoint is driving. Off by "
                         "default -- you score these by eye, and knowing "
                         "which one is the fine-tune is exactly the bias the "
                         "pairing exists to remove. The true identity is "
                         "always written to the score record either way.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None,
                    help="directory for rollout logs (default: rollouts/)")
    ap.add_argument("--video", action="store_true",
                    help="save an mp4 per rollout of what the POLICY SAW "
                         "(its own camera frames, tiled and annotated with "
                         "episode/tick/time). Encoding runs on a worker "
                         "thread and drops frames rather than stalling the "
                         "control loop.")
    ap.add_argument("--task-string", default=None,
                    help="language instruction handed to the policy every "
                         "tick. Only VLAs (SmolVLA, pi0) use it; ACT ignores "
                         "it. Default: the shape_sorter target's own string, "
                         "else the single task the training dataset recorded.")
    ap.add_argument("--inactive-pose", default="rest",
                    help="where the arms this mode does NOT drive are assumed "
                         "to be parked, for the inter-arm capsule gate. They "
                         "are not sensed; the default matches --quit-pose.")
    ap.add_argument("--place-grid", action="store_true",
                    help="between episodes, show top_scene with the hull-study "
                         "lattice (place_grid.py) and name the nearest cell; "
                         "if --cells gives a lattice name (B2) for this "
                         "episode only that cell is drawn.  Needs top_scene.")
    ap.add_argument("--replicate", default=None, metavar="DIR",
                    help="snapshots/ directory of an EARLIER rollout run. "
                         "Before each episode the live cameras are shown "
                         "ghosted against that run's episode-N snapshot, so "
                         "the objects can be put back exactly where they were "
                         "-- which is what makes the two runs a PAIRED "
                         "comparison instead of two independent samples.")
    ap.add_argument("--no-snapshots", action="store_true",
                    help="do not save the per-episode starting-scene PNGs "
                         "(they are what a later --replicate run needs)")
    ap.add_argument("--witness-cameras", nargs="+", default=[],
                    metavar="CAM",
                    help="extra cameras to OPEN AND RECORD but never feed to "
                         "the policy -- a third-person witness view for "
                         "reviewing rollouts afterwards. 'oak_left' is the "
                         "camera arm's own eye: park the middle arm where you "
                         "would stand and it films the trial from your point "
                         "of view, hands-free. Requires --video to be saved.")
    ap.add_argument("--video-scale", type=float, default=0.5,
                    help="resize factor for each camera tile in the mp4")
    ap.add_argument("--min-rate-frac", type=float, default=0.7,
                    help="abort an episode whose achieved tick rate falls "
                         "below this fraction of the target. A policy trained "
                         "at 50 Hz and executed at 4 Hz holds every action "
                         "~12x too long: the rollout is not a slower version "
                         "of the same thing, it is a different experiment. "
                         "0 disables the guard.")
    ap.add_argument("--jax", default="cpu", choices=["cpu", "gpu"],
                    help="jax backend for the gates + the EE-action IK "
                         "(cpu default: the GPU is usually training; an "
                         "EE-action checkpoint solves IK every tick and "
                         "wants gpu when one is free)")
    ap.add_argument("--gates", default="on",
                    choices=["on", "off", "table", "capsule"],
                    help="safety gates on the commanded step: 'on' (default) "
                         "runs both the inter-arm capsule gate and the "
                         "tabletop gate, exactly as data collection did. They "
                         "fired 0 times across the whole training set, so a "
                         "well-behaved policy never notices them; a firing "
                         "means the policy left the demonstrated envelope and "
                         "is logged as such. 'off' only if you know why.")
    ap.add_argument("--score", action="store_true",
                    help="after each rollout, print automatic measurements "
                         "and prompt for per-stage outcomes plus a free-text "
                         "note; appends to rollout_scores.jsonl. Forces the "
                         "top_scene camera open (for object tracking) even "
                         "when the checkpoint does not use it.")
    ap.add_argument("--stall-seconds", type=float, default=0.0,
                    help="end the episode when no state dimension has moved "
                         "by more than --stall-rad for this long (0 = off). "
                         "A policy holding still is not going to finish; this "
                         "lets --seconds be generous without paying for every "
                         "stall in wall-clock.")
    ap.add_argument("--stall-rad", type=float, default=0.02,
                    help="motion threshold for --stall-seconds, rad (gripper "
                         "included)")
    ap.add_argument("--target", default=None,
                    help="shape_sorter piece being inserted this session "
                         "(cube/triangle/flower/cylinder). Selects the "
                         "shape_sorter scene: the env-state features (if the "
                         "checkpoint takes them), --score object tracking and "
                         "the scoring stages all follow the piece. Without it "
                         "the flower scene is assumed.")
    ap.add_argument("--cells", default=None,
                    help="comma-separated placement labels, one per episode, "
                         "in the order you will place the object (e.g. "
                         "'1,2,3,...,21'). Recorded as `cell` on the "
                         "trajectory and score records so a placement map can "
                         "be plotted without parsing free-text notes. Fewer "
                         "labels than episodes cycles them; omitted = the "
                         "episode index.")
    ap.add_argument("--px-per-cm", type=float, default=None,
                    help="top_scene scale, to report distances in cm as well "
                         "as pixels. Calibrate once: measure the flower->oval "
                         "target centre distance on the sheet in cm; the same "
                         "distance is 32.5 px in the image, so "
                         "px_per_cm = 32.5 / that measurement.")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.mode not in ARM_MODES:
        raise SystemExit(f"--mode must be one of {sorted(ARM_MODES)}")
    arm_names = ARM_MODES[args.mode]

    ckpt_dir = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    print(f"[policy] loading {ckpt_dir}")
    (policy, preprocessor, postprocessor, image_keys,
     state_dim, action_dim, pcfg) = load_policy(
         ckpt_dir, device, temporal_ensemble=args.temporal_ensemble)
    if args.temporal_ensemble is not None:
        print(f"[policy] TEMPORAL ENSEMBLING coeff={args.temporal_ensemble} "
              f"(n_action_steps forced to 1; the network runs every tick)")

    ## Object-centric input.  A checkpoint trained with --env-state expects
    ## observation.environment_state every tick, computed from top_scene by
    ## the SAME extractor build_envstate_dataset.py used -- there is no
    ## fallback, a missing key would fail inside the preprocessor anyway.
    ## The scene follows --target: flower without it, shape_sorter with it.
    ENV_KEY = "observation.environment_state"
    scene = "shape_sorter" if args.target else "flower"
    stages = STAGE_SETS[scene]
    env_key = ENV_KEY if ENV_KEY in pcfg.input_features else None
    want = (int(np.prod(pcfg.input_features[env_key].shape))
            if env_key is not None else None)
    extractor = make_extractor(scene, args.target, want)
    if env_key is not None:
        if want != extractor.FEATURE_DIM:
            raise SystemExit(
                f"checkpoint takes a {want}-d {ENV_KEY} but the {scene} "
                f"extractor produces {extractor.FEATURE_DIM}-d -- wrong "
                f"--target / scene for this checkpoint?")
        print(f"[env-state] feeding {ENV_KEY} ({want}-d, scene={scene}"
              + (f", target={args.target}" if args.target else "") + ")")
    elif args.target:
        print(f"[scene] shape_sorter target={args.target} (scoring only; "
              f"this checkpoint takes no {ENV_KEY})")

    ## ACTION HORIZON, at inference time only.
    ##
    ## `n_action_steps` is how many actions of each predicted chunk get
    ## executed before the policy is asked again -- and it is read at
    ## queue-refill time (ACT's select_action), so it can be changed on a
    ## trained checkpoint without retraining anything.
    ##
    ## THIS MATTERS MORE THAN IT LOOKS.  lerobot ships ACT with
    ## n_action_steps == chunk_size, which means the policy looks at the
    ## cameras ONCE per chunk and then runs open loop until the queue drains.
    ## At 50 Hz with chunk 100 that is a full 2 SECONDS of blind execution:
    ## nothing that happens in the scene -- a slipped grasp, a nudged object,
    ## the arm being a centimetre off -- can be reacted to until the chunk
    ## ends.  Smaller values re-plan more often and cost one forward pass each
    ## time (measured ~11 ms warm on this box, well inside a 20 ms tick).
    if args.n_action_steps is not None and args.temporal_ensemble is not None:
        raise SystemExit("--n-action-steps and --temporal-ensemble are exclusive: "
                         "ensembling requires inference every tick (n=1)")
    if args.n_action_steps is not None:
        n_req = int(args.n_action_steps)
        n_max = int(getattr(pcfg, "chunk_size", None)
                    or getattr(pcfg, "horizon", n_req))
        if not 1 <= n_req <= n_max:
            raise SystemExit(
                f"--n-action-steps must be in [1, {n_max}] for this "
                f"checkpoint, got {n_req}")
        if getattr(pcfg, "temporal_ensemble_coeff", None) is not None and n_req != 1:
            raise SystemExit(
                "this checkpoint uses temporal ensembling, which requires "
                "--n-action-steps 1")
        policy.config.n_action_steps = n_req
        if hasattr(policy, "_action_queue"):
            policy._action_queue.clear()
        print(f"[policy] n_action_steps {n_max} -> {n_req} "
              f"(re-plans every {n_req} ticks)")
    else:
        _n = getattr(policy.config, "n_action_steps", None)
        print(f"[policy] n_action_steps={_n} (checkpoint default)")

    ## --------------------------------------------------------------- ##
    ## A/B: the second policy, loaded into its own bundle.  Both stay
    ## resident; ~1.2 GB extra on the GPU buys back the ability to run the
    ## two policies MINUTES apart on the same scene instead of a day apart.
    ## --------------------------------------------------------------- ##
    ab = None
    ## True when EITHER side of an A/B wants the env-state feature, so
    ## top_scene is opened for the pair rather than for A alone.
    _ab_needs_top = False
    if args.checkpoint_b:
        ckpt_b = resolve_checkpoint(args.checkpoint_b)
        print(f"[policy] loading B {ckpt_b}")
        (policy_b, pre_b, post_b, image_keys_b,
         state_dim_b, action_dim_b, pcfg_b) = load_policy(
             ckpt_b, device, temporal_ensemble=args.temporal_ensemble)
        if image_keys_b != image_keys:
            raise SystemExit(
                f"--checkpoint-b takes different cameras:\n  A {image_keys}\n"
                f"  B {image_keys_b}\nThe two must be interchangeable to be "
                f"paired on one scene.")
        if (state_dim_b, action_dim_b) != (state_dim, action_dim):
            raise SystemExit(
                f"--checkpoint-b has state/action dims "
                f"{(state_dim_b, action_dim_b)}, A has "
                f"{(state_dim, action_dim)}")
        if (ENV_KEY in pcfg_b.input_features) != (env_key is not None):
            ## ALLOWED, and it is the whole point of an object-centric
            ## ablation: A takes observation.environment_state and B does not.
            ## The extractor runs either way (it is cheap and top_scene is
            ## already open), and each policy is fed only what its own config
            ## declares -- so the two see identical images, identical state,
            ## and differ in exactly the feature under test.
            print(f"[A/B] the two checkpoints differ on {ENV_KEY}: "
                  f"A={'yes' if env_key else 'no'}, "
                  f"B={'yes' if ENV_KEY in pcfg_b.input_features else 'no'}. "
                  f"That IS the comparison; each is fed its own inputs.")
        if args.n_action_steps is not None:
            policy_b.config.n_action_steps = int(args.n_action_steps)
            if hasattr(policy_b, "_action_queue"):
                policy_b._action_queue.clear()
        _env_b = ENV_KEY if ENV_KEY in pcfg_b.input_features else None
        _want_b = (int(np.prod(pcfg_b.input_features[_env_b].shape))
                   if _env_b is not None else None)
        if _env_b is not None and _want_b != extractor.FEATURE_DIM:
            raise SystemExit(
                f"--checkpoint-b takes a {_want_b}-d {ENV_KEY} but the "
                f"{scene} extractor produces {extractor.FEATURE_DIM}-d")
        ab = {
            "a": (policy, preprocessor, postprocessor, pcfg, ckpt_dir,
                  env_key, want),
            "b": (policy_b, pre_b, post_b, pcfg_b, ckpt_b, _env_b, _want_b),
        }
        ## top_scene has to be read whenever EITHER policy wants the feature.
        _ab_needs_top = (env_key is not None) or (_env_b is not None)
        ## Which arm drives which episode.  Every scene gets both, adjacent,
        ## in an order drawn per scene -- so a drift over the session (warming
        ## servos, a tiring operator) hits both arms equally instead of
        ## landing on whichever ran second.
        _rng = random.Random(args.ab_seed)
        n_scenes = (args.episodes + 1) // 2
        ## BALANCED, not coin-flipped.  Independent flips leave the order
        ## lopsided at this sample size -- seed 0 put B first in 16 of 25
        ## scenes -- and "ran second" is a real effect here: the object is
        ## re-placed, the servos are warmer, the operator has just watched
        ## the same scene once.  Exactly half the scenes start with each arm,
        ## with only WHICH half left to the seed.
        _first = ["a"] * (n_scenes // 2) + ["b"] * (n_scenes - n_scenes // 2)
        _rng.shuffle(_first)
        ab["plan"] = []
        for _f in _first:
            ab["plan"] += [_f, "b" if _f == "a" else "a"]
        ab["plan"] = ab["plan"][:args.episodes]
        ## The blinded name the operator sees.  Per SCENE, so the two
        ## episodes of a pair are "P1" and "P2" in the order they are run and
        ## nothing carries across scenes.
        print(f"[A/B] {n_scenes} scene(s) x 2 policies = {args.episodes} "
              f"episodes, order seeded with --ab-seed {args.ab_seed}")
        if args.unblind:
            print("[A/B] UNBLINDED -- the driving checkpoint is printed")
        else:
            print("[A/B] blinded: each episode is announced as P1/P2 only; "
                  "the true checkpoint goes to the score record")
        print(f"[A/B]   A = {ckpt_dir}")
        print(f"[A/B]   B = {ckpt_b}")

    cameras = [k.removeprefix("observation.images.") for k in image_keys]
    print(f"[policy] type={pcfg.type}  device={device}")
    print(f"[policy] cameras (from the checkpoint): {cameras}")
    print(f"[policy] state_dim={state_dim}  action_dim={action_dim}")

    ## The policy's state/action dims must match the arms being driven, or the
    ## vector is being interpreted as something it is not.  Checked here, not
    ## at the first command.
    want_state = sum(ARM_CONFIG[a]["num_joints"]
                     + (1 if ARM_CONFIG[a]["has_gripper"] else 0)
                     for a in arm_names)
    ## The dim check itself is below, after meta.json says whether this
    ## checkpoint's action is joint-space or an end-effector pose.

    ## fps and start pose come from the RUN THAT PRODUCED THE CHECKPOINT.
    ## Rolling out at a rate the demonstrations were not recorded at rescales
    ## every action the policy emits -- a 50 Hz policy stepped at 25 Hz moves
    ## the arm at half speed through a trajectory whose timing was the whole
    ## demonstration.  So this is READ FROM THE CHECKPOINT rather than left to
    ## the operator to remember: train_config.json records the dataset root it
    ## trained on, and that directory carries the fps and the start pose.
    fps, start_pose, ds_root = None, None, None
    if args.dataset:
        ds_root = Path(args.dataset)
    else:
        try:
            tcfg = json.loads((ckpt_dir / "train_config.json").read_text())
            ds_root = Path(tcfg["dataset"]["root"])
            print(f"[cfg] training dataset (from the checkpoint): {ds_root}")
            if not ds_root.exists():
                print("[cfg] ...which no longer exists; "
                      "pass --dataset or --hz")
                ds_root = None
        except Exception as exc:
            print(f"[cfg] checkpoint does not name its dataset ({exc}); "
                  f"pass --dataset or --hz")
    ## Defaults, so a run whose dataset cannot be resolved still reaches the
    ## checks below instead of raising NameError on the first reference.
    waist_ref, waist_name, waist_ds_reclock = None, "middle_base", 0.0
    ee_action_space = None
    if ds_root is not None:
        try:
            fps = float(json.loads(
                (ds_root / "meta" / "info.json").read_text())["fps"])
        except Exception as exc:
            print(f"[cfg] could not read fps from {ds_root}: {exc}")
        try:
            tc = json.loads((ds_root / "teleop_config.json").read_text())
            start_pose = tc.get("start_pose")
        except Exception:
            pass
        ## WAIST BRANCH.  middle_base is multi-turn and its driver frame boots
        ## 2*pi-shifted at random, so canonicalize_waist.py put the training
        ## data on ONE branch and recorded which.  The live servo can boot on
        ## the other one; without the same mapping here the policy meets a
        ## 6.28 rad offset on that joint from the first tick.  Absent key =
        ## the dataset was never canonicalized = do nothing.
        try:
            _m = json.loads((ds_root / "meta.json").read_text())
            waist_ref = _m.get("waist_reference")
            waist_name = _m.get("waist_joint", "middle_base")
            waist_ds_reclock = float(_m.get("waist_reclock", 0.0) or 0.0)
            ee_action_space = _m.get("action_space")
        except Exception:
            waist_ref, waist_name = None, "middle_base"
            waist_ds_reclock = 0.0
            ee_action_space = None
    ## EE-ACTION CHECKPOINT (build_ee_action_dataset.py): the action is
    ## [qw,qx,qy,qz,x,y,z,gripper] for ONE arm, 8-d, and is turned into joint
    ## targets by the coupled IK each tick (below).  The dataset's meta.json
    ## says so; the dim alone does not.
    ee_action = bool(ee_action_space and str(ee_action_space).startswith("ee_pose."))
    ee_arm = str(ee_action_space).split(".")[1].split("+")[0] if ee_action else None
    if ee_action and (len(arm_names) != 1 or arm_names[0] != ee_arm
                      or action_dim != 8):
        raise SystemExit(
            f"EE-action checkpoint ({ee_action_space}, action_dim={action_dim}) "
            f"needs --mode {ee_arm} and an 8-d action; got --mode {args.mode}.")
    if ee_action and args.checkpoint_b:
        raise SystemExit("A/B is not wired for EE-action checkpoints yet -- "
                         "compare against a joint checkpoint with --replicate.")
    if state_dim != want_state or (action_dim != want_state and not ee_action):
        raise SystemExit(
            f"--mode {args.mode} has {want_state} state/action dims, but this "
            f"checkpoint has state={state_dim}, action={action_dim}. It was "
            f"trained on a different arm set.")
    if ee_action:
        print(f"[policy] EE-ACTION checkpoint: {ee_action_space} -> coupled IK "
              f"per tick for arm {ee_arm!r}")

    if args.hz:
        fps = float(args.hz)
    if fps is None:
        raise SystemExit(
            "Control rate unknown. Pass --dataset <training run dir> so it "
            "can be read from meta/info.json, or set --hz explicitly. "
            "Rolling out at the wrong rate silently rescales every action.")
    start_pose = args.start_pose or start_pose or "forward"

    def parse_pose_spec(label, spec):
        """'forward' -> every arm there; 'right=forward,middle=forward_demo'
        -> each arm to its own.

        PER-ARM MATTERS.  The camera arm's 'forward' was re-taught after the
        shape_sorter demonstrations were recorded, so a policy trained on
        that data needs the middle arm at 'forward_demo' while the right arm
        goes to 'forward'.  Parking both at one name puts the camera where
        the policy has never seen it -- and the camera arm carries oak_left,
        so that is an observation shift, not a cosmetic one.
        data_collection.py already accepted this syntax; without it here the
        whole string reached get_pose() verbatim and the run died at the
        first park."""
        spec = str(spec)
        if "=" not in spec:
            out = {a: spec for a in arm_names}
        else:
            out = {}
            for part in spec.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    raise SystemExit(f"{label}={spec!r}: {part!r} is not arm=pose")
                arm, name = (x.strip() for x in part.split("=", 1))
                if arm not in arm_names:
                    raise SystemExit(
                        f"{label}: arm {arm!r} is not active in --mode "
                        f"{args.mode} (active: {arm_names})")
                out[arm] = name
            missing = [a for a in arm_names if a not in out]
            if missing:
                raise SystemExit(
                    f"{label}={spec!r} says nothing about {missing}. Name "
                    f"every active arm, or give one pose for all of them.")
        ## Checked against EVERY active arm before the servos are energised,
        ## for the reason data_collection.py validates up front: discovering
        ## a bad pose name mid-move is the wrong time.
        for a, name in out.items():
            if name not in POSES[a]:
                raise SystemExit(f"{label}: {name!r} is not defined for arm "
                                 f"'{a}'. Available: {sorted(POSES[a])}")
        return out

    start_poses = parse_pose_spec("--start-pose", start_pose)
    quit_poses = parse_pose_spec("--quit-pose", args.quit_pose)

    def park(poses):
        """Every active arm to ITS pose, together."""
        return move_arms_together(
            {a: robots[a] for a in arm_names},
            {a: get_pose(a, poses[a]) for a in arm_names})

    control_dt = 1.0 / fps
    cfg = TeleopConfig(control_dt=control_dt)
    print(f"[cfg] {fps:g} Hz  start_pose={start_pose}  "
          f"max_step={args.max_step:g} rad/tick")

    if not args.engage:
        print("\n" + "=" * 70)
        print("DRY RUN -- no command will reach a servo.")
        print("The loop, the cameras and the policy all run; actions are")
        print("printed and logged only.  Add --engage to drive the arm.")
        print("=" * 70)
    else:
        print("\n" + "!" * 70)
        print("ENGAGED -- THIS WILL MOVE THE ARM.")
        print("Keep a hand near the e-stop.  Ctrl-C stops commanding at once.")
        print("!" * 70)

    if rospy is None:
        raise SystemExit("rospy is required.")
    rospy.init_node("rollout_policy", anonymous=True)

    robots = create_and_configure_robots(arm_names)
    apply_profile_limits(robots, cfg, arm_names=arm_names)

    ## Cameras: exactly the ones the checkpoint names.  get_active_cameras
    ## derives the wrist cameras from the arm mode, so the scene flags only
    ## have to carry top/low.
    need_top = ("top_scene" in cameras) or args.score or env_key is not None \
        or args.place_grid \
        or _ab_needs_top
    camera_config = CameraConfig(
        top_active=need_top,
        low_active="low_scene" in cameras)
    active_cameras = get_active_cameras(args.mode, camera_config)

    ## Witness cameras are opened here rather than inferred from the arm mode:
    ## the whole point is a view the POLICY does not have.  get_active_cameras
    ## only yields the OAK pair when the middle arm is in the mode, so a
    ## right-only rollout would otherwise never film itself.
    witness = [c for c in args.witness_cameras if c not in cameras]
    ignored = [c for c in args.witness_cameras if c in cameras]
    if ignored:
        print(f"[witness] {ignored} already feed the policy; recorded anyway")
    for c in witness:
        if c not in active_cameras:
            active_cameras = list(active_cameras) + [c]
    if witness:
        if not args.video:
            print(f"[witness] {witness} requested but --video is off -- "
                  f"nothing will be recorded from them")
        else:
            print(f"[witness] recording {witness} alongside the policy's view "
                  f"(NOT fed to the network)")

    record_cameras = list(cameras) + [c for c in witness if args.video]
    unexpected = sorted(set(active_cameras) - set(cameras) - set(witness))
    missing = sorted(set(cameras) - set(active_cameras))
    if missing:
        raise SystemExit(
            f"The checkpoint needs {missing}, which --mode {args.mode} does "
            f"not provide. Roll out with the arm mode it was trained on.")
    if unexpected:
        ## Opening a camera the policy never saw is harmless but wasteful, and
        ## more importantly it means the operator thinks a view is being used
        ## that is not.  Say so.
        print(f"[camera] {unexpected} opened but NOT fed to this checkpoint "
              f"as an image"
              + (" (top_scene is for --score tracking / env-state features)"
                 if need_top and "top_scene" in unexpected else ""))
    if need_top and "top_scene" not in active_cameras:
        raise SystemExit(
            f"--score / env-state need top_scene for object tracking, which "
            f"--mode {args.mode} does not provide.")

    ## Index of the waist inside the concatenated state/action vector, in the
    ## recorder's layout (per arm in ARM_MODES order, joints then gripper).
    ## THE PHYSICAL RE-CLOCK, which canonicalization cannot absorb.
    ##
    ## canonical_branch only ever adds multiples of 2*pi, so for any shift
    ## smaller than pi it rounds to k=0 and silently changes nothing.  The
    ## 2026-09-11 re-clock is -pi: a dataset recorded before it describes the
    ## SAME physical yaw with a number pi away from what the servo reports
    ## today.  Without this, a 14-dim policy (right + camera arm) would drive
    ## the camera arm half a turn from everything it was trained on.
    ##
    ## The gap is `live theta - the theta already baked into this dataset`,
    ## which is 0 for a dataset rewritten by shift_waist_frame.py and the full
    ## theta for one recorded earlier.  Applied BEFORE canonicalization: the
    ## reference branch is a number in the DATASET's frame.
    try:
        from robot_control import MIDDLE_WAIST_RECLOCK_RAD as _live_reclock
    except Exception:
        _live_reclock = float(os.environ.get("GIAVA_MIDDLE_WAIST_RECLOCK") or 0.0)
    waist_shift = float(_live_reclock) - float(waist_ds_reclock)

    waist_idx = None
    ## Needed whenever EITHER correction applies -- a dataset that was never
    ## canonicalized still needs the re-clock.
    if waist_ref is not None or abs(waist_shift) > 1e-9:
        _i = 0
        for a in arm_names:
            for jn in ARM_CONFIG[a]["joint_names"]:
                if jn == waist_name:
                    waist_idx = _i
                _i += 1
            if ARM_CONFIG[a]["has_gripper"]:
                _i += 1
        if waist_idx is None:
            print(f"[waist] {waist_name} is not in --mode {args.mode}; "
                  f"no waist correction needed")
            waist_ref = None
            waist_shift = 0.0
        else:
            if abs(waist_shift) > 1e-9:
                print(f"[waist] physical re-clock {waist_shift:+.4f} rad "
                      f"(live {_live_reclock:+.4f} - dataset "
                      f"{waist_ds_reclock:+.4f}) applied to {waist_name} "
                      f"before canonicalization")
            print(f"[waist] canonicalizing {waist_name} (state index "
                  f"{waist_idx}) to branch {waist_ref:+.3f} rad, as the "
                  f"training data was")

    latest_frames = {cam: None for cam in active_cameras}
    latest_timestamps = {cam: None for cam in active_cameras}
    pipelines = setup_cameras(active_cameras, camera_shutdown, frame_lock,
                              latest_frames, latest_timestamps)

    out_dir = Path(args.out or (Path(__file__).parent / "rollouts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    ## THE INSTRUCTION.  A VLA is conditioned on it; an ACT policy never sees
    ## it.  Prefer an explicit --task-string, then the shape_sorter piece
    ## (whose string IS the target), then whatever single task the training
    ## dataset recorded.  A multi-task dataset without --task-string is
    ## ambiguous and says so rather than guessing.
    task_string = args.task_string
    if task_string is None and args.target:
        try:
            import shape_sorter as _ss
        except ImportError:
            from . import shape_sorter as _ss
        task_string = _ss.task_string(args.target)
    if task_string is None and ds_root is not None:
        try:
            import pandas as _pd
            _t = _pd.read_parquet(ds_root / "meta" / "tasks.parquet").reset_index()
            _names = [str(x) for x in _t["task"]]
            if len(_names) == 1:
                task_string = _names[0]
            elif len(_names) > 1:
                print(f"[task] dataset has {len(_names)} task strings; pass "
                      f"--task-string if this policy is language-conditioned")
        except Exception as exc:
            print(f"[task] could not read tasks.parquet ({exc})")
    if task_string is not None:
        print(f"[task] instruction: {task_string!r}")

    ## cv2 is imported lazily everywhere else in this file (it pulls Qt and
    ## is only needed when something is drawn); the setup preview below draws
    ## every episode, so bind it once here.
    import cv2

    ## Scene snapshots: every camera that is open and shows the workspace.
    ## The wrist camera moves with the arm, so it is a poor alignment
    ## reference and is left out.
    snap_cameras = [c for c in active_cameras if not c.endswith("_wrist")]
    ## Placement labels.  The episode index alone would work, but naming the
    ## cell explicitly means the run records WHERE the object was rather than
    ## only WHEN it was tried -- and the two stop matching the moment an
    ## episode is aborted and re-run.
    _cells = ([c.strip() for c in args.cells.split(",") if c.strip()]
              if args.cells else None)
    ## In A/B every scene is rolled out twice, so the SCENE index -- which
    ## snapshot to replicate, which grid cell to call it -- advances at half
    ## the episode rate.  Everything the operator sets up keys off this; only
    ## the log/video filenames key off the episode.
    def _scene_of(ep):
        return ep // 2 if ab is not None else ep

    def _cell_of(ep):
        if not _cells:
            return ep
        return _cells[_scene_of(ep) % len(_cells)]
    if _cells:
        print(f"[cells] {len(_cells)} placement labels; episode -> cell: "
              + ", ".join(f"ep{i:02d}->{_cell_of(i)}"
                          for i in range(min(4, args.episodes)))
              + (" ..." if args.episodes > 4 else ""))

    ## RUN TAG = task + timestamp, and it goes on EVERY artefact this run
    ## writes.  A directory of `snapshots_20260911_092907` tells you when a
    ## rollout happened and nothing about what it was trying to do, so finding
    ## "the cube runs from Tuesday" meant opening images until one looked
    ## right.  The task is the thing you actually search by.
    run_tag = f"{_slug(args.target or task_string) or 'rollout'}_{stamp}"

    snap_dir = None if args.no_snapshots else out_dir / f"snapshots_{run_tag}"
    ref_snap_dir = Path(args.replicate) if args.replicate else None
    if ref_snap_dir is not None and not ref_snap_dir.exists():
        raise SystemExit(f"--replicate: {ref_snap_dir} does not exist")
    if snap_dir is not None:
        print(f"[snapshot] starting scenes -> {snap_dir}  "
              f"(cameras: {snap_cameras})")
    if ref_snap_dir is not None:
        print(f"[replicate] reference snapshots: {ref_snap_dir}")
    log_path = out_dir / f"rollout_{run_tag}.jsonl"
    score_path = out_dir / "rollout_scores.jsonl"
    print(f"[log] {log_path}")
    if args.score:
        print(f"[scores] {score_path}")

    threading.Thread(target=keyboard_listener, daemon=True).start()

    n_joints = {a: ARM_CONFIG[a]["num_joints"] for a in arm_names}
    has_grip = {a: ARM_CONFIG[a]["has_gripper"] for a in arm_names}
    limits = {}
    for a in arm_names:
        gi = robots[a].arm.group_info
        limits[a] = (np.asarray(gi.joint_lower_limits, dtype=float),
                     np.asarray(gi.joint_upper_limits, dtype=float))

    ## PREFLIGHT: a rollout shares the GPU with whatever else is running, and
    ## the image->CUDA path in build_observation is the tick's dominant cost.
    ## Measured 2026-09-06: ~1 ms with the GPU free, 216 ms with two training
    ## runs on it -- which drags a 50 Hz loop down to 4.3 Hz and silently
    ## turns every rollout into an invalid measurement.  Say so BEFORE the
    ## operator spends a session on it.
    if args.device.startswith("cuda"):
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            others = [l for l in out.splitlines()
                      if l.strip() and not l.startswith(str(os.getpid()))]
            if others:
                print("\n[preflight] WARNING: other processes are on the GPU:")
                for l in others:
                    print(f"             {l}")
                print("[preflight] The per-tick image->CUDA transfer is the "
                      "loop's dominant cost. Contention has been measured to "
                      "drop a 50 Hz rollout to 4.3 Hz, which changes the "
                      "policy's effective control rate and INVALIDATES the "
                      "comparison. Stop the other jobs, or expect the "
                      "rate guard below to abort.\n")
        except Exception:
            pass

    gates = None
    if args.gates != "off":
        print(f"[gates] building ({args.gates}) ...")
        gates = SafetyGates(args.mode, arm_names, inactive_pose=args.inactive_pose,
                            want_capsule=args.gates in ("on", "capsule"),
                            want_table=args.gates in ("on", "table"),
                            jax_platform=args.jax)
        print(f"[gates] {gates.summary()}")
    elif ee_action:
        raise SystemExit("an EE-action checkpoint needs the gates' coupled IK "
                         "solver: run with --gates on (default).")
    else:
        print("[gates] DISABLED -- nothing bounds a lunging checkpoint but "
              "--max-step and the driver limits.")

    global latest_key
    aborted = False

    try:
        for ep in range(args.episodes):
            if aborted:
                break
            print(f"\n=== rollout {ep + 1}/{args.episodes} "
                  f"({'ENGAGED' if args.engage else 'dry run'})")
            print(f"    parking at {start_pose!r}...")
            if args.engage:
                park(start_poses)
                rospy.sleep(0.2)

            ## A/B: swap in this episode's policy BEFORE anything touches
            ## it.  Rebinding the names the tick loop already uses keeps the
            ## control path identical for both arms -- there is no second
            ## code path that could differ.
            ab_arm = ab_label = None
            if ab is not None:
                ab_arm = ab["plan"][ep]
                ab_label = f"P{(ep % 2) + 1}"
                (policy, preprocessor, postprocessor,
                 pcfg, ckpt_dir, env_key, want) = ab[ab_arm]
                for _other in ("a", "b"):
                    ab[_other][0].reset()
                print(f"    [A/B] scene {_scene_of(ep):02d}, "
                      f"{'second' if ep % 2 else 'first'} of the pair: "
                      f"{ab_label}"
                      + (f"  = arm {ab_arm.upper()}  {ckpt_dir}"
                         if args.unblind else "  (blinded)"))

            ## The queue holds the tail of the PREVIOUS rollout's action
            ## chunk; carrying it into a fresh scene would execute actions
            ## planned for a layout that no longer exists.
            policy.reset()

            print("    press ENTER to start, or type q + ENTER to quit"
                  "   (during the episode: p pause/resume, s end early)")
            if ref_snap_dir is not None:
                print(f"    [replicate] matching ep{_scene_of(ep):02d} of {ref_snap_dir.name}"
                      f" -- place the objects until the ghosting disappears")
            latest_key = None
            _preview = None
            while latest_key is None and not rospy.is_shutdown():
                ## Live alignment preview while waiting for ENTER.  Costs
                ## nothing: the control loop has not started, and the frames
                ## come from the camera workers that are already running.
                if snap_cameras:
                    with frame_lock:
                        live = {c: (latest_frames.get(c).copy()
                                    if latest_frames.get(c) is not None else None)
                                for c in snap_cameras}
                    ref = (load_snapshot(ref_snap_dir, _scene_of(ep), snap_cameras)
                           if ref_snap_dir is not None else {})
                    _preview = alignment_view(live, ref, snap_cameras)
                    if _preview is not None:
                        cv2.imshow("setup", _preview)
                        cv2.waitKey(1)
                if args.place_grid:
                    ## Hull-study lattice on the live top_scene, same drawing
                    ## and same detector as place_grid.py (which cannot run
                    ## alongside: one RealSense owner at a time).
                    with frame_lock:
                        _ts = latest_frames.get("top_scene")
                        _ts = _ts.copy() if _ts is not None else None
                    if _ts is not None:
                        try:
                            import place_grid as _pgm
                            import scene_features as _sfm
                        except ImportError:
                            from . import place_grid as _pgm
                            from . import scene_features as _sfm
                        _only = None
                        _labels = ([c.strip() for c in args.cells.split(",")]
                                   if args.cells else [])
                        _grid = _pgm.cells_for(_labels)
                        if _labels:
                            _lab = _labels[_scene_of(ep) % len(_labels)]
                            if _lab in {c[0] for c in _grid}:
                                _only = _lab
                        _img = _pgm.draw(cv2.cvtColor(_ts, cv2.COLOR_RGB2BGR),
                                         _grid, _sfm.detect_object(_ts),
                                         8.0, _only)
                        cv2.imshow("placement", _img)
                        cv2.waitKey(1)
                time.sleep(0.05)
            if snap_cameras:
                try:
                    cv2.destroyWindow("setup")
                    cv2.waitKey(1)
                except Exception:
                    pass
            if args.place_grid:
                try:
                    cv2.destroyWindow("placement")
                    cv2.waitKey(1)
                except Exception:
                    pass
            if latest_key == "q":
                aborted = True
                break
            latest_key = None

            ## Snapshot AFTER the operator committed, so it records the scene
            ## the policy actually starts from.
            if snap_dir is not None:
                with frame_lock:
                    _snap = {c: (latest_frames.get(c).copy()
                                 if latest_frames.get(c) is not None else None)
                             for c in snap_cameras}
                save_snapshot(snap_dir, ep, _snap)

            if gates is not None:
                gates.reset_counters()
                for a in arm_names:
                    gates.seed(a, read_arm_state(
                        robots[a], n_joints[a], has_grip[a])[:n_joints[a]])

            rate_aborted = False
            score_frames = []
            extractor.reset()   # hold-last carry must not cross episodes
            video = None
            if args.video:
                video = RolloutVideo(
                    out_dir / f"rollout_{run_tag}_ep{ep:02d}.mp4",
                    record_cameras, fps, scale=args.video_scale)
                print(f"    [video] {video.path.name}")

            ## WARM THE POLICY BEFORE THE CLOCK STARTS.  The first forward
            ## pass on a CUDA device pays one-off costs -- lazy context init,
            ## cuDNN algorithm selection for these exact input shapes -- worth
            ## ~500 ms.  Measured: tick #1 took 506 ms while every other tick
            ## held 19.9 ms, and that single stall dragged a genuine 50.8 Hz
            ## loop down to the 31 Hz that tripped the rate guard and aborted
            ## every episode.  One throwaway inference here, then reset() to
            ## drop the action queue it just filled, and the episode starts
            ## against a warm kernel cache.
            _warm_state = np.concatenate(
                [read_arm_state(robots[a], n_joints[a], has_grip[a])
                 for a in arm_names]).astype(np.float32)
            if waist_idx is not None:
                _warm_state = _warm_state.copy()
                _w = _warm_state[waist_idx] - waist_shift
                _warm_state[waist_idx] = (
                    canonical_branch(_w, waist_ref)
                    if waist_ref is not None else _w)
            _warm_obs, _why, _, _wf = build_observation(
                image_keys, cameras, _warm_state, device, task=task_string,
                latest=(latest_frames if args.latest_frames else None))
            if _warm_obs is not None:
                if env_key is not None:
                    _ts = (_wf or {}).get("top_scene")
                    if _ts is None:
                        with frame_lock:
                            _ts = latest_frames.get("top_scene")
                            _ts = _ts.copy() if _ts is not None else None
                    if _ts is not None:
                        _warm_obs[env_key] = (
                            torch.from_numpy(extractor(_ts)).unsqueeze(0).to(device))
                        extractor.reset()
                _t_warm = time.monotonic()
                postprocessor(policy.select_action(preprocessor(_warm_obs)))
                policy.reset()
                _dt = time.monotonic() - _t_warm
                _passes = 1

                ## ONE PASS IS NOT ENOUGH ON THE FIRST EPISODE, and episode 0
                ## paid for it every single run: measured across 53 runs, the
                ## first episode of a run reaches a median 47.3 Hz against
                ## 49.1 Hz for later ones, and on the worst of them it came in
                ## at 32.8 Hz -- under the rate guard, which then aborted at
                ## tick 60.  Every rate-abort on record (11 of them) is an
                ## episode 0.  The cost is not one inference: cuDNN autotunes
                ## per shape, the caching allocator grows, and the first few
                ## calls after that are still paying for it.
                ##
                ## So keep going until a pass fits comfortably inside the
                ## control period, rather than assuming one did the job.  This
                ## runs BEFORE t0, so it costs startup time and nothing else --
                ## and it buys back a real episode 0 instead of a discarded one.
                if ep == 0:
                    _budget = 0.5 / max(1e-6, fps)
                    while _passes < _WARMUP_MAX_PASSES and _dt > _budget:
                        _t = time.monotonic()
                        postprocessor(policy.select_action(preprocessor(_warm_obs)))
                        policy.reset()
                        _dt = time.monotonic() - _t
                        _passes += 1
                    if _dt > _budget:
                        print(f"    [warmup] still {_dt * 1e3:.0f} ms/pass after "
                              f"{_passes} passes, against a {_budget * 1e3:.0f} ms "
                              f"budget. This episode may not hold {fps:.0f} Hz --"
                              f" that is a real throughput problem, not warm-up.")
                if ep == 0 or _dt > 0.05:
                    print(f"    [warmup] {_passes} pass(es), last "
                          f"{_dt * 1e3:.0f} ms (excluded from the rate)")
            else:
                print(f"    [warmup] skipped: {_why}")

            t0 = time.monotonic()
            next_tick = t0
            rate_t0 = None
            tick = 0
            skipped = 0
            clamp_hits = {"step": 0, "limit": 0}
            rows = []
            ended = "time"
            ros_down = False
            paused = False
            pause_t0 = None
            interventions = 0
            stall_ref = None    # (state, t) the stall clock is measured from

            ## ROS SHUTDOWN IS NOT AN EPISODE OUTCOME.  When roscore goes or
            ## the operator Ctrl-Cs, is_shutdown() flips and this loop falls
            ## straight out with `ended` still at its default "time" -- so
            ## every REMAINING episode span up, exited in ~13 ms with 0 ticks,
            ## and got written to rollout_scores.jsonl looking like a policy
            ## that did nothing.  Three such records on 2026-09-16.  Detect it,
            ## mark the episode, and stop the run instead of burning through
            ## the rest.
            while True:
                if rospy.is_shutdown():
                    ros_down, ended, aborted = True, "ros_shutdown", True
                    print("\n    [ros] node is shutting down -- ending the run. "
                          "This episode is NOT a result; nothing is scored.")
                    break
                ## 'p' toggles a pause: nothing is commanded, the arm holds
                ## where it is, and the operator can fix the scene or help a
                ## grasp.  Paused time does not count against --seconds.  On
                ## resume the policy's queued chunk is dropped (it was planned
                ## for a scene that no longer exists) and the gates re-seed
                ## from the measured pose in case the arm was nudged.
                if latest_key == "p":
                    latest_key = None
                    paused = not paused
                    if paused:
                        pause_t0 = time.monotonic()
                        interventions += 1
                        print("    [paused] arm holds; fix the scene, then p")
                    else:
                        t0 += time.monotonic() - pause_t0
                        next_tick = time.monotonic()
                        policy.reset()
                        if gates is not None:
                            for a in arm_names:
                                gates.seed(a, read_arm_state(
                                    robots[a], n_joints[a], has_grip[a])[:n_joints[a]])
                        stall_ref = None
                        print("    [resumed]")
                if paused:
                    if latest_key in ("q", "s"):
                        print(f"    stopped by operator ('{latest_key}')")
                        aborted = latest_key == "q"
                        ended = "operator"
                        break
                    time.sleep(0.05)
                    continue

                if time.monotonic() - t0 >= args.seconds:
                    print(f"    time limit ({args.seconds:g}s) reached")
                    break
                if latest_key in ("q", "s"):
                    print(f"    stopped by operator ('{latest_key}')")
                    aborted = latest_key == "q"
                    ended = "operator"
                    break

                state = np.concatenate(
                    [read_arm_state(robots[a], n_joints[a], has_grip[a])
                     for a in arm_names]).astype(np.float32)

                ## state_raw stays in DRIVER coordinates: it is what the
                ## clamp measures against and what the servo understands.
                ## Only the copy handed to the policy is canonicalized.
                state_raw = state
                if waist_idx is not None:
                    state = state.copy()
                    _w = state_raw[waist_idx] - waist_shift
                    state[waist_idx] = (
                        canonical_branch(_w, waist_ref)
                        if waist_ref is not None else _w)

                if args.stall_seconds > 0:
                    now = time.monotonic()
                    if (stall_ref is None
                            or np.max(np.abs(state - stall_ref[0])) > args.stall_rad):
                        stall_ref = (state.copy(), now)
                    elif now - stall_ref[1] >= args.stall_seconds:
                        print(f"    [stall] no joint moved > {args.stall_rad:g} rad "
                              f"for {args.stall_seconds:g}s -- ending episode")
                        ended = "stall"
                        break

                obs, why, sync_info, frames = build_observation(
                    image_keys, cameras, state, device, task=task_string,
                    latest=(latest_frames if args.latest_frames else None))
                if obs is None:
                    ## Do NOT command on a stale observation: the arm would be
                    ## driven from an image of where things used to be.
                    skipped += 1
                    if skipped in (1, 10, 100) or skipped % 250 == 0:
                        print(f"    waiting on cameras: {why} "
                              f"({skipped} ticks)")
                    next_tick += control_dt
                    time.sleep(max(0.0, next_tick - time.monotonic()))
                    continue

                if env_key is not None:
                    ## Prefer the synchronized frame when top_scene is also a
                    ## policy camera, so the features describe the same
                    ## instant as the image tokens; otherwise the latest.
                    ts_img = frames.get("top_scene")
                    if ts_img is None:
                        with frame_lock:
                            ts_img = latest_frames.get("top_scene")
                            ts_img = (ts_img.copy() if ts_img is not None
                                      else None)
                    if ts_img is None:
                        skipped += 1
                        if skipped in (1, 10, 100) or skipped % 250 == 0:
                            print(f"    waiting on cameras: no top_scene "
                                  f"frame for env-state ({skipped} ticks)")
                        next_tick += control_dt
                        time.sleep(max(0.0, next_tick - time.monotonic()))
                        continue
                    obs[env_key] = (torch.from_numpy(extractor(ts_img))
                                    .unsqueeze(0).to(device))

                action = postprocessor(policy.select_action(preprocessor(obs)))
                action = np.asarray(
                    action.detach().float().cpu().numpy()).reshape(-1)

                _ee_row = None
                if ee_action:
                    ## The 8-d output is a commanded END-EFFECTOR pose in the
                    ## URDF base frame (pyroki wxyz_xyz, the same thing the
                    ## recorder wrote as observation.ee_pose).  Solve it with
                    ## the coupled IK teleop recorded the demonstrations with,
                    ## seeded from the last command (gates.q_cmd, which
                    ## gates.seed() set to the measured pose at episode
                    ## start), so the arm reaches the pose the way the
                    ## demonstrator's arm did.  From here the tick is a joint
                    ## policy's tick: clamp, gate, command.
                    _wxyz = action[:4].astype(float)
                    _wxyz /= max(1e-9, float(np.linalg.norm(_wxyz)))
                    _t_ik = time.monotonic()
                    _q_full = np.asarray(gates.ik.solve(
                        gates.q_cmd,
                        {ee_arm: (action[4:7].astype(float), _wxyz)}), dtype=float)
                    _ee_row = {"ee_target": action[:7].tolist(),
                               "ik_ms": round((time.monotonic() - _t_ik) * 1e3, 2)}
                    action = np.concatenate(
                        [_q_full[gates.idx[ee_arm]], action[7:8]]).astype(np.float32)

                ## Back to the servo's own branch before anything commands
                ## it: the policy speaks canonical, the motor speaks driver.
                if waist_idx is not None:
                    ## anchor on the measured value IN THE DATASET'S FRAME,
                    ## then shift the command back into the servo's.
                    _anchor = state_raw[waist_idx] - waist_shift
                    _a = (canonical_branch(action[waist_idx], _anchor)
                          if waist_ref is not None else action[waist_idx])
                    action[waist_idx] = _a + waist_shift

                i = 0
                row = {"t": time.monotonic() - t0, "tick": tick,
                       "state": state_raw.tolist(), "action_raw": action.tolist(),
                       "sync_spread_s": sync_info.get("spread_s"),
                       "frame_age_s": sync_info.get("age_s"),
                       "per_camera_age_s": sync_info.get("per_camera_age_s")}
                if _ee_row:
                    row.update(_ee_row)
                _age = sync_info.get("age_s")
                if _age is not None and _age > 0.15 and tick % 25 == 0:
                    print(f"    [frames] STALE: policy is seeing frames "
                          f"{_age * 1e3:.0f} ms old  "
                          f"{sync_info.get('per_camera_age_s')}")
                for a in arm_names:
                    n = n_joints[a]
                    width = n + (1 if has_grip[a] else 0)
                    meas = state_raw[i:i + width].astype(float)
                    q, grip, hit = clamp_action(
                        action[i:i + width], meas, *limits[a],
                        args.max_step, n, has_grip[a])
                    for k in hit:
                        clamp_hits[k] = clamp_hits.get(k, 0) + 1
                    ## Gate the CLAMPED command, so the gate sees the step
                    ## that would actually be sent.
                    gate_hold = False
                    if gates is not None:
                        q, alpha, ginfo = gates.filter(a, q)
                        if ginfo:
                            row[f"{a}_gated"] = ginfo
                        gate_hold = alpha <= 0.0

                    row[f"{a}_cmd"] = q.tolist()
                    if grip is not None:
                        row[f"{a}_gripper_cmd"] = grip
                    if hit:
                        row[f"{a}_clamped"] = {k: v for k, v in hit.items()}

                    if args.engage and not gate_hold:
                        robots[a].arm.set_joint_positions(
                            q.tolist(),
                            moving_time=cfg.moving_time,
                            accel_time=cfg.accel_time,
                            blocking=False,
                        )
                        if grip is not None:
                            command_gripper(robots[a], grip)
                    i += width

                rows.append(row)
                if video is not None:
                    ## Witness frames come from the worker's latest, not from
                    ## the synchronized set: they are evidence for a human,
                    ## and holding up the policy's frame selection to wait on
                    ## a camera the network never sees would be backwards.
                    vframes = frames
                    if witness:
                        vframes = dict(frames)
                        with frame_lock:
                            for c in witness:
                                w = latest_frames.get(c)
                                if w is not None:
                                    vframes[c] = w.copy()
                    video.submit(vframes, ep, tick, row["t"])
                ## Scoring samples top_scene on its own, straight from the
                ## camera worker, so adding it never changes the
                ## nearest-timestamp set the POLICY's frames are chosen from.
                if args.score and tick % 25 == 0:
                    with frame_lock:
                        ts_img = latest_frames.get("top_scene")
                        ts_img = ts_img.copy() if ts_img is not None else None
                    if ts_img is not None:
                        score_frames.append((tick, row["t"], ts_img))
                tick += 1

                if tick % max(1, int(fps)) == 0:
                    step_norm = np.linalg.norm(
                        np.asarray(row[f"{arm_names[0]}_cmd"])
                        - state_raw[:n_joints[arm_names[0]]])
                    print(f"    t={row['t']:5.1f}s  tick={tick:4d}  "
                          f"|cmd-meas|={step_norm:.4f} rad  "
                          f"clamped step/limit="
                          f"{clamp_hits['step']}/{clamp_hits['limit']}")

                ## Rate guard, measured on a STEADY-STATE window: ticks
                ## 30..90 rather than 0..60.  The old window started at t0,
                ## so any one-off cost in the first half second -- the video
                ## writer opening its file, the first log flush, an allocator
                ## growth -- was averaged over only 60 ticks and could drag
                ## the mean under the threshold on its own.  A genuinely
                ## doomed episode is slow throughout and still trips this;
                ## a slow first half-second no longer does.
                if args.min_rate_frac > 0 and tick == _RATE_WINDOW_START:
                    rate_t0 = time.monotonic()
                if (args.min_rate_frac > 0 and tick == _RATE_WINDOW_END
                        and rate_t0 is not None):
                    achieved = ((_RATE_WINDOW_END - _RATE_WINDOW_START)
                                / max(1e-6, time.monotonic() - rate_t0))
                    if achieved < args.min_rate_frac * fps:
                        print(f"\n    [rate] ABORTING: achieved "
                              f"{achieved:.1f} Hz vs target {fps:.0f} Hz "
                              f"({achieved / fps:.0%}).")
                        print(f"    [rate] The policy was trained at "
                              f"{fps:.0f} Hz. Executing it this slowly is a "
                              f"DIFFERENT experiment, not a slower one -- do "
                              f"not score it.")
                        print(f"    [rate] Usual cause: something else is on "
                              f"the GPU (check nvidia-smi). Re-run "
                              f"--min-rate-frac 0 to override.")
                        rate_aborted = True
                        break

                next_tick += control_dt
                slack = next_tick - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    ## Behind: resnap rather than burst to catch up, the same
                    ## rule the recorder's loop uses.
                    next_tick = time.monotonic()

            video_written = video_dropped = 0
            if video is not None:
                video_written, video_dropped = video.close()

            ## END-OF-EPISODE SCENE, the pair to the start snapshot taken
            ## before the run.  Written with an `_end` suffix so the START
            ## filename is untouched -- load_snapshot() and therefore
            ## --replicate look for exactly `ep{ep}_{cam}.png`, and renaming
            ## it would silently break replication of every earlier run.
            if snap_dir is not None:
                with frame_lock:
                    _snap_end = {c: (latest_frames.get(c).copy()
                                     if latest_frames.get(c) is not None else None)
                                 for c in snap_cameras}
                save_snapshot(snap_dir, ep, _snap_end, suffix="_end")

            record = {
                "episode": ep, "cell": _cell_of(ep), "checkpoint": str(ckpt_dir),
                "scene_index": _scene_of(ep),
                "ab_arm": ab_arm, "ab_label": ab_label,
                "policy": pcfg.type, "engaged": bool(args.engage),
                "fps": fps, "start_pose": start_pose,
                "start_poses": start_poses,
                "cameras": cameras, "witness_cameras": witness,
                "latest_frames": bool(args.latest_frames),
                "ticks": tick,
                "camera_stall_ticks": skipped,
                "clamped": clamp_hits,
                "seconds": time.monotonic() - t0,
                "achieved_hz": round(tick / max(1e-6, time.monotonic() - t0), 2),
                "target_hz": fps,
                "rate_aborted": rate_aborted,
                "ended": ended,
                "interventions": interventions,
                "video": (str(video.path) if video is not None else None),
                "video_frames": video_written,
                "video_dropped": video_dropped,
                "gates": (gates.summary() if gates is not None else None),
                "rows": rows,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"    done: {tick} ticks, {skipped} camera stalls, "
                  f"clamped step/limit {clamp_hits['step']}/"
                  f"{clamp_hits['limit']}")

            if gates is not None:
                gs = gates.summary()
                fired = sum(gs["blocks"].values()) + sum(gs["scaled"].values())
                if fired:
                    print(f"    [gates] FIRED {fired} ticks "
                          f"blocks={gs['blocks']} scaled={gs['scaled']} "
                          f"min_alpha={gs['min_alpha']} -- the policy "
                          f"commanded outside the demonstrated envelope")

            if args.score and (rate_aborted or ros_down):
                print(f"    [score] SKIPPED -- "
                      + ("ROS shut down mid-episode"
                         if ros_down else "the rate guard aborted this episode")
                      + "; scoring it would put an invalid trial in "
                        "rollout_scores.jsonl.")
            elif args.score:
                achieved = tick / max(1e-6, time.monotonic() - t0)
                try:
                    _targets = extractor.targets_px()
                except (FileNotFoundError, KeyError):
                    _targets = {}
                metrics = auto_metrics(
                    object_track(score_frames, extractor), rows,
                    arm_names[0], args.px_per_cm, _targets)
                metrics["achieved_hz"] = round(achieved, 2)
                metrics["elapsed_s"] = round(time.monotonic() - t0, 2)
                metrics["interventions"] = interventions
                metrics["ended"] = ended
                verdict = score_episode(metrics, ep, args.episodes, stages)
                if verdict is not None:
                    with open(score_path, "a") as f:
                        f.write(json.dumps({
                            "cell": _cell_of(ep),
                            "episode": ep,
                            ## The pairing key: two records with the same
                            ## scene_index are the same physical scene.
                            "scene_index": _scene_of(ep),
                            "ab_arm": ab_arm,
                            "ab_label": ab_label,
                            "ab_blinded": (ab is not None and not args.unblind),
                            "log": str(log_path),
                            "video": (str(video.path)
                                      if video is not None else None),
                            "checkpoint": str(ckpt_dir),
                            "cameras": cameras,
                            "scene": scene,
                            "target": args.target,
                            "task_string": task_string,
                            "snapshot_dir": (str(snap_dir) if snap_dir else None),
                            "replicate_of": (str(ref_snap_dir) if ref_snap_dir else None),
                            "ended": ended,
                            "interventions": interventions,
                            "env_state": env_key is not None,
                            "mode": args.mode,
                            "n_action_steps": getattr(
                                policy.config, "n_action_steps", None),
                            "temporal_ensemble": args.temporal_ensemble,
                            "engaged": bool(args.engage),
                            "seconds": round(time.monotonic() - t0, 2),
                            "ticks": tick,
                            "camera_stall_ticks": skipped,
                            "clamped": clamp_hits,
                            "gates": (gates.summary()
                                      if gates is not None else None),
                            "auto": metrics,
                            **verdict,
                            "wall_time": time.time(),
                        }) + "\n")
            if video is not None:
                print(f"    [video] {video_written} frames written, "
                      f"{video_dropped} dropped -> {video.path}")
            if ros_down:
                ## Do not start the next episode: every one of them would
                ## exit instantly the same way.
                print(f"[abort] ROS is down -- stopping after episode {ep}; "
                      f"{args.episodes - ep - 1} episode(s) not run.")
                break

    except KeyboardInterrupt:
        ## Ctrl-C must stop COMMANDING immediately.  Torque stays on, so the
        ## arm holds where it is rather than dropping.
        print("\n[abort] Ctrl-C -- no further commands will be sent.")
    finally:
        try:
            if args.engage:
                print(f"[shutdown] parking at {args.quit_pose!r}...")
                park(quit_poses)
        except Exception as exc:
            print(f"[shutdown] could not park: {exc}")
        camera_shutdown.set()
        join_camera_workers(timeout=2.0)
        for name, p in (pipelines or {}).items():
            try:
                if isinstance(p, dict):
                    for pipe in p.values():
                        pipe.stop()
                elif isinstance(p, (tuple, list)) and p:
                    close = getattr(p[0], "close", None)
                    if callable(close):
                        close()
            except Exception as exc:
                print(f"[camera] {name} shutdown: {exc}")
        print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
