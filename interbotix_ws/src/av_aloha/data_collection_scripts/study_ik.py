"""Coupled three-arm IK for data collection — the ik_study winner, adapted.

Wraps the study's validated deployment configuration
(`ik/study/RESULTS_FINAL.md`, `ik/study/COLLISION_STUDY.md`):

    pose_cost_analytic_jac ×3      pos 50 / ori 10
    limit_constraint               (pyroki augmented Lagrangian)
    smoothing                      w = 0.05 (velocity-scaled prev-config)
    centering                      w = 0.5  (range-normalized, arm joints)
    sphere self-collision          180 spheres · corpus-pruned pairs ·
                                   soft margin hinge · margin 20 mm · w = 100

One coupled solve per control tick replaces the previous per-arm
`solve_single_arm_ik` calls: all three arms share one least-squares problem,
which is what makes inter-arm collision terms meaningful.

Notes for this integration:
- End-effector links come from the CALLER (arm_config.ARM_CONFIG); the
  middle arm tracks `middle_camera_cover` (same as the study benchmarks).
  It must be downstream of the `middle_pan` joint or IK cannot control the
  camera-yaw motor (the collision model is link-based and unaffected by
  the IK target choice).
- Arms without a target this tick hold their current commanded pose (FK of
  prev_q) as an active target, so the coupled solve keeps them still.
- The study's weights are used verbatim and are deliberately NOT read from
  TeleopConfig (whose pos 40 / ori 0.25 / dq 0.18 belong to the old per-arm
  solver).  Expect much stiffer orientation tracking than before — that is
  the study's central correction, not a bug.  Tune here if the feel is off.
- Runs on the GPU by default (jax_platform.py).  The study's "CPU is faster"
  result was measured on the BARE pose solver; this one carries the
  180-sphere self-collision cost and the tabletop term, which is the regime
  where the arithmetic outweighs kernel-launch overhead.  `--jax cpu` (or
  GIAVA_JAX_PLATFORM=cpu) restores the studied configuration, and the backend
  that actually initialised is printed at construction -- CUDA falls back to
  CPU silently otherwise.
"""

from __future__ import annotations

import os

## Backend selection, before ANY jax import below (jaxlie pulls it in).
## GPU-first -- see jax_platform.py for why the study's CPU result does not
## carry over to the collision-carrying solver this file actually builds.
## `--jax cpu` on the entry point, or GIAVA_JAX_PLATFORM=cpu, restores it.
try:
    from .jax_platform import apply as _jax_apply
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _HERE = str(_Path(__file__).resolve().parent)
    if _HERE not in _sys.path:
        _sys.path.insert(0, _HERE)
    from jax_platform import apply as _jax_apply
_jax_apply()

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

## The solver modules live in av_aloha/ik/, a sibling of this directory --
## the research phases that produced them are in ik/study/, and are NOT
## imported here.  Resolved relative to this file rather than the CWD so the
## entry points work from anywhere.
_IK = Path(__file__).resolve().parent.parent / "ik"
## The pruned-sphere and table collision models read their fitted parameters
## from the study's results tree; that directory moved with the reorg, and
## the two call sites below went stale (NameError on 2026-09-14).
if not (_IK / "variants.py").exists():
    raise SystemExit(
        f"the IK solver modules are missing from {_IK} -- study_ik.py needs "
        f"collision_models.py, table_collision.py and variants.py there.")
if str(_IK) not in sys.path:
    sys.path.insert(0, str(_IK))

import jaxlie  # noqa: E402
from yourdfpy import URDF  # noqa: E402

from collision_models import (pruned_sphere_collision,  # noqa: E402
                              pruned_tight_capsule_collision)
from table_collision import table_halfspace, table_robot_collision  # noqa: E402
from variants import ComboIK  # noqa: E402

ARM_ORDER = ("left", "right", "middle")

## The middle waist ('middle_base') is a multiturn joint whose DRIVER frame is
## rotated pi from the URDF frame (measured driver range [-6.35, 0], forward
## pose at driver -3.10 == URDF ~0; FK-verified).  The solver works in URDF
## coordinates; convert at the boundary:
##   urdf  = driver + pi        (driver in [-2pi, 0] -> urdf in [-pi, pi])
##   driver = urdf - pi         (urdf in [-pi, pi]  -> driver in [-2pi, 0],
##                               always inside the [-6.35, 0] driver limits)
WAIST_URDF_OFFSET = np.pi

## Per-joint driver<->URDF offsets for the middle arm (beyond the waist).
##     urdf_j = sign_j * (driver_j - offset_j)
## Calibrated with make_middle_offsets.py, stored by JOINT NAME in
## middle_joint_offsets.json; absent or empty file = identity (the waist keeps
## its own dedicated machinery above).
##
## EMPTY SINCE 2026-08-21, and that is the intended end state.  The 2026-08
## calibration was folded into giava.urdf itself -- axes negated on
## middle_shoulder/middle_upper_arm (they were modelled `0 -1 0` against the
## vendor description's `0 1 0`), Rot(axis, -offset) post-multiplied into the
## other four joints' origins, limits transformed to match.  The mechanism
## stays here because it is the only correct place for a genuine hardware
## offset, but prefer the URDF: a correction living in this file is invisible
## to everything that loads giava.urdf directly (RViz, pyroki's collision
## model, view_collision.py, capsule_gate_view.py), which is exactly how sim
## and real came to be two different robots.
MIDDLE_OFFSETS_FILE = Path(__file__).resolve().parent / "middle_joint_offsets.json"


def _running_in_sim() -> bool:
    """True when the interbotix SIM driver is up (node 'xs_sdk_sim').

    The assembly offsets describe the REAL arm only; the sim driver speaks the
    URDF frame natively, so applying them there would INTRODUCE the mismatch.
    Detection by node name means no file renaming between sim and real."""
    try:
        import rosnode
        return any("xs_sdk_sim" in n for n in rosnode.get_node_names())
    except Exception:
        return False


def _load_middle_offsets():
    """{giava_joint_name: (sign, offset)} with driver = sign*urdf + offset.

    File entries may be a bare number (offset, sign +1) or
    {"sign": -1, "offset": x}.  Keys starting with "_" are notes."""
    if not MIDDLE_OFFSETS_FILE.exists():
        return {}
    if _running_in_sim():
        print("[study_ik] sim driver detected -- middle joint offsets NOT applied")
        return {}
    import json
    try:
        d = json.load(open(MIDDLE_OFFSETS_FILE))
        out = {}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                out[k] = (int(v.get("sign", 1)), float(v.get("offset", 0.0)))
            else:
                out[k] = (1, float(v))
        print("[study_ik] middle offsets (REAL arm): " + "  ".join(
            f"{k}[s={sg:+d},o={of:+.3f}]" for k, (sg, of) in out.items()))
        return out
    except Exception as exc:
        print(f"[study_ik] could not read {MIDDLE_OFFSETS_FILE}: {exc}")
        return {}

## ------------------------------------------------------------------------- ##
## TUNING KNOBS — edit these while testing on the real arms.
##
## Everything that shapes the feel of teleoperation is here, including the pose
## weights, which used to be inherited silently from ik/baseline.py
## (DEFAULT_POS_WEIGHT / DEFAULT_ORI_WEIGHT).  They are stated explicitly now so
## there is one place to change, and so the deployed value is visible rather
## than buried two files away.
##
## An env var overrides any of them without editing code, e.g.
##     GIAVA_IK_ORI_W=3 GIAVA_IK_SMOOTHING_W=0.3 python data_collection.py
## ------------------------------------------------------------------------- ##


def _env(name: str, default: float) -> float:
    """Read an override from the environment, else use the default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[study_ik] ignoring non-numeric {name}={raw!r}")
        return default


## Pose tracking.  ori 10 is the ik_study value; the benchmark's weight search
## (ik/benchmark/results/opt2_best.json, two independent runs) converged on
## ~4.7 instead, and ori 10 is the leading suspect for jerky motion and the
## shoulder swinging -- at that stiffness the solver contorts the whole arm to
## satisfy wrist orientation.
POS_W = _env("GIAVA_IK_POS_W", 50.0)
ORI_W = _env("GIAVA_IK_ORI_W", 10.0)

## Per-arm override for the MIDDLE (camera) arm.  Its job differs from the
## grippers: gaze DIRECTION is the product, position is only composition.  A
## high position weight makes the arm chase every centimetre of head height;
## a higher orientation weight makes it hold the look direction instead.
## Defaults fall back to the shared weights (no behavior change until set).
## Defaults changed 2026-08 (was: inherit 50/10 from the grippers).  Head yaw
## is not a pure rotation -- the head pivots about the neck with a ~10 cm lever
## arm, so every head rotation also TRANSLATES the head.  At pos 50 / ori 10
## the solver spends the arm's motion chasing that incidental translation
## instead of rotating the camera, which is exactly "the camera arm does not do
## what my head does".  Gaze direction is the product; position is composition.
POS_W_MIDDLE = _env("GIAVA_IK_POS_W_MIDDLE", 10.0)
ORI_W_MIDDLE = _env("GIAVA_IK_ORI_W_MIDDLE", 25.0)

## Smoothing: fraction of the per-tick velocity budget, so it is directly
## comparable with the benchmark (identical formulation, v_nom = 2.0 rad/s).
## The benchmark's search converged on ~0.9 from two independent runs with
## different objectives; 0.05 here is 18x lower, which is the other half of the
## jerkiness suspicion.
SMOOTHING_W = _env("GIAVA_IK_SMOOTHING_W", 0.05)

## Range-normalized joint centering (dimensionless).  Already fairly strong.
CENTERING_W = _env("GIAVA_IK_CENTERING_W", 0.5)

## Sphere self-collision (ik_study's 180-sphere pruned model — NOT the
## benchmark's capsule model, so the benchmark's collision weight does not
## transfer).  Left at the study-validated values.
COLLISION_W = _env("GIAVA_IK_COLLISION_W", 100.0)
COLLISION_MARGIN = _env("GIAVA_IK_COLLISION_MARGIN", 0.020)  # m

## Which geometry the SOLVER's soft collision cost feels.
##
##   sphere   (default) the studied deployment model: 180 inscribed spheres,
##            corpus-pruned pairs.  0.60 mm clean p95, 5.2 ms mean solve.
##   capsule  the circumscribed long-axis capsule fit.  The study measured
##            59-76 mm hand-jerk, 33 % non-convergence and 43 ms solves at
##            these exact settings ON CPU SIM over the frozen suite -- it is
##            switchable here so that verdict can be FELT on the real arms
##            rather than inherited from sim.  Expect the solver to shy away
##            from close approaches much earlier (the capsules are fat), and
##            watch ik_overruns in the episode stats: 43 ms against a 20 ms
##            budget at 50 Hz means dropped ticks.
##
## This is the COST the solver trades against pose tracking, not a guarantee.
## The hard inter-arm guarantee is capsule_gate.py, which runs on the final
## clamped command regardless of what is selected here.
COLLISION_MODEL = os.environ.get("GIAVA_IK_COLLISION_MODEL", "sphere")

## Tabletop world-collision (table_collision.py). HARDWARE-VALIDATED
## 2026-09-09 on the real arms -- the tabletop behaviour was confirmed to be
## what was intended.  What has still NOT been run is a Phase-9-style
## margin x weight sweep, unlike every other weight above, so these numbers
## are a validated working point rather than a tuned optimum.  OFF by default; opt in per-run with
##     GIAVA_IK_TABLE_ENABLE=1 python data_collection.py
## and inspect it in ik/study/view_table_collision.py before trusting it on
## hardware.  Margin/weight default to the self-collision winner's values as
## a starting point only.
TABLE_ENABLE = os.environ.get("GIAVA_IK_TABLE_ENABLE", "0") == "1"
TABLE_W = _env("GIAVA_IK_TABLE_W", 100.0)
TABLE_MARGIN = _env("GIAVA_IK_TABLE_MARGIN", 0.020)  # m
TABLE_Z = _env("GIAVA_IK_TABLE_Z", 0.0)  # m, world frame -- same convention
                                          # as calibration/base_validation.py
                                          # --table-z

## Levenberg-Marquardt budget.  ComboIK defaults to 100; the study's winner
## converges in a handful of iterations from a warm start, and the benchmark
## measured a 20-iteration cap as bit-identical on tracking at a fraction of the
## wall clock.  Capping is what keeps the worst-case tick inside the control
## period once the sphere-collision terms are active (study: 5.2 ms median but
## 64 ms worst-trajectory p95 -- over three control periods at 50 Hz).
MAX_ITERATIONS = int(_env("GIAVA_IK_MAX_ITERATIONS", 20))

## The control period the smoothing residual is scaled against.  ik_study's
## variants.DT is fixed at 0.02 (50 Hz); the smoothing scale is w/(v_nom*DT), so
## running the loop at a different rate without correcting the weight silently
## changes the smoothing strength -- at 25 Hz each tick allows twice the motion
## for the same penalty, i.e. half the intended smoothing.  CoupledStudyIK
## rescales the weight so the *physical* velocity budget stays what the study
## validated, whatever rate the loop runs at.
STUDY_DT = 0.02


def describe_weights() -> str:
    """One-line summary, printed at startup so the deployed values are logged."""
    table = (
        f" table=hw-validated,w={TABLE_W:g},margin={TABLE_MARGIN * 1e3:g}mm,"
        f"z={TABLE_Z * 1e3:g}mm"
        if TABLE_ENABLE else ""
    )
    return (
        f"pos={POS_W:g} ori={ORI_W:g} "
        f"pos_mid={POS_W_MIDDLE:g} ori_mid={ORI_W_MIDDLE:g} smoothing={SMOOTHING_W:g} "
        f"centering={CENTERING_W:g} collision={COLLISION_W:g}"
        f"[{COLLISION_MODEL}] "
        f"margin={COLLISION_MARGIN * 1e3:g}mm max_iter={MAX_ITERATIONS}{table}"
    )


class CoupledStudyIK:
    """Drop-in coupled solver for the data collection loop."""

    def __init__(
        self,
        robot,
        urdf_path: str,
        ee_links: Dict[str, str],
        control_dt: float = STUDY_DT,
        max_iterations: int = MAX_ITERATIONS,
        waist_driver_shift: float = 0.0,
    ) -> None:
        """`waist_driver_shift`: the middle waist's total driver-frame shift
        in radians, i.e. driver_new = driver_old + shift.

        TWO THINGS FEED IT and they are not interchangeable: a physical
        re-clock of the motor (unlimited, invisible to every register, and
        since 2026-09-11 equal to -pi here) and Homing_Offset (a register,
        inert under ext_position).  robot_control.resolve_middle_waist_shift
        combines them; pass its result.

        The legacy relation was urdf = driver + pi at shift 0; with a shift h
        it becomes urdf = driver + (pi - h).  At h = -pi that offset is 2*pi
        and, after the [-pi, pi] wrap in driver_to_urdf, the two frames
        coincide."""
        self.waist_urdf_offset = float(np.pi - waist_driver_shift)
        self.robot = robot
        urdf = URDF.load(urdf_path)
        if COLLISION_MODEL == "sphere":
            robot_coll = pruned_sphere_collision(urdf, _IK / "study" / "results")
        elif COLLISION_MODEL == "capsule":
            robot_coll = pruned_tight_capsule_collision(urdf)
            print("[study_ik] SOLVER COLLISION = CAPSULE (hardware trial). "
                  "Sim verdict at these settings: 59-76 mm hand-jerk, 33% "
                  "nonconvergence, 43 ms solves. Watch ik_overruns; "
                  "GIAVA_IK_COLLISION_MODEL=sphere restores the deployed "
                  "model.")
        else:
            raise ValueError(
                f"GIAVA_IK_COLLISION_MODEL must be sphere|capsule, got "
                f"'{COLLISION_MODEL}'")
        self._link_idx = np.asarray(
            [robot.links.names.index(ee_links[a]) for a in ARM_ORDER],
            dtype=np.int32,
        )
        self._waist_idx = robot.joints.actuated_names.index("middle_base")
        # Per-joint sign/offset over the FULL actuated set (identity elsewhere):
        #     driver = sign * urdf + offset   <=>   urdf = sign * (driver - offset)
        # Normally the identity now -- the 2026-08 calibration lives in
        # giava.urdf (see MIDDLE_OFFSETS_FILE above).  Kept so a freshly
        # measured hardware offset can be applied without a URDF edit.
        n_act = robot.joints.num_actuated_joints
        self._joint_offsets = np.zeros(n_act, dtype=np.float32)
        self._joint_signs = np.ones(n_act, dtype=np.float32)
        for name, (sg, off) in _load_middle_offsets().items():
            if name == "middle_base":
                continue  # waist handled by waist_urdf_offset
            try:
                j = robot.joints.actuated_names.index(name)
                self._joint_signs[j] = sg
                self._joint_offsets[j] = off
            except ValueError:
                print(f"[study_ik] offsets file names unknown joint '{name}'")
        extras = {
            # Rescaled so w/(v_nom*STUDY_DT) equals the study's scale at the
            # *actual* control period: w_eff = w * STUDY_DT / control_dt.
            "smoothing": SMOOTHING_W * STUDY_DT / max(control_dt, 1e-6),
            "centering": CENTERING_W,
            "collision": COLLISION_W,
        }
        table_coll, table_geom = None, None
        if TABLE_ENABLE:
            extras["table"] = TABLE_W
            table_coll = table_robot_collision(urdf, _IK / "study" / "results")
            table_geom = table_halfspace(TABLE_Z)
            print("[study_ik] TABLE COLLISION ENABLED -- hardware-validated "
                  "2026-09-09 on the real arms; not swept "
                  "(see ik/table_collision.py).")
        self._ik = ComboIK(
            robot,
            tuple(ee_links[a] for a in ARM_ORDER),
            extras=extras,
            robot_coll=robot_coll,
            collision_margin=COLLISION_MARGIN,
            max_iterations=max_iterations,
            table_coll=table_coll,
            table_geom=table_geom,
            table_margin=TABLE_MARGIN,
        )
        self.control_dt = control_dt
        self.max_iterations = max_iterations
        # Kept for diagnostics: min_clearance() lets the control loop verify
        # the collision terms are actually seeing what the operator sees.
        self.robot_coll = robot_coll
        # ARM_ORDER = (left, right, middle)
        self._pos_weights = np.array([POS_W, POS_W, POS_W_MIDDLE], dtype=np.float32)
        self._ori_weights = np.array([ORI_W, ORI_W, ORI_W_MIDDLE], dtype=np.float32)
        self.last_solve_ms: float = float("nan")
        self.last_iterations: int = 0
        print(f"[study_ik] weights: {describe_weights()}")
        ## Name the backend that ACTUALLY initialised.  "cuda,cpu" falls back
        ## silently by design (a missing driver must not stop data
        ## collection), which is exactly how a GPU that never engaged stays
        ## invisible -- so it is printed rather than assumed.
        from jax_platform import describe as _jax_describe
        print(_jax_describe())

    # ------------------------------------------------------------------ #
    def driver_to_urdf(self, q_driver: np.ndarray) -> np.ndarray:
        """Driver joint vector -> URDF joint vector (waist frame shift)."""
        q = np.asarray(q_driver, dtype=np.float32).copy()
        q = self._joint_signs * (q - self._joint_offsets)
        w = q[self._waist_idx] + self.waist_urdf_offset
        # wrap into [-pi, pi] in case the driver value sits at a 2pi-shifted
        # equivalent (e.g. freshly re-homed edge cases)
        q[self._waist_idx] = (w + np.pi) % (2 * np.pi) - np.pi
        return q

    def urdf_to_driver(self, q_urdf: np.ndarray,
                       ref_driver: np.ndarray | None = None) -> np.ndarray:
        """URDF joint vector -> driver joint vector (waist frame shift).

        FRAME-AWARE: the driver's waist frame can boot 2pi-shifted (encoder
        wrap; Homing_Offset is inert in extended-position mode).  The command
        therefore uses the 2pi-equivalent of the URDF angle NEAREST to the
        current driver reading (ref_driver) — never a full-turn jump,
        whatever frame the servo woke up in."""
        q = np.asarray(q_urdf, dtype=np.float32).copy()
        q = self._joint_signs * q + self._joint_offsets
        d = q[self._waist_idx] - self.waist_urdf_offset
        if ref_driver is not None:
            ref = float(np.asarray(ref_driver)[self._waist_idx])
            k = np.round((ref - d) / (2 * np.pi))
            d = d + 2 * np.pi * k
        q[self._waist_idx] = d
        return q

    def min_clearance(self, q_driver: np.ndarray) -> float:
        """Minimum signed clearance [m] of the SOLVER's collision model at a
        DRIVER configuration.

        Positive = separation, negative = overlap.  Interpretation depends on
        GIAVA_IK_COLLISION_MODEL: the sphere fit is INSCRIBED (optimistic by
        up to ~18 mm on true contacts -- why the cost activates at margin
        20 mm rather than zero); the capsule fit is CIRCUMSCRIBED (a negative
        reading does not necessarily mean the meshes touch)."""
        q_urdf = self.driver_to_urdf(q_driver)
        d = self.robot_coll.compute_self_collision_distance(
            self.robot, np.asarray(q_urdf, dtype=np.float32)
        )
        return float(np.min(np.asarray(d)))

    def hold_poses(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Current EE poses of all three target links (FK of q, URDF coords)."""
        fk = self.robot.forward_kinematics(np.asarray(q, dtype=np.float32))
        pos = np.zeros((3, 3))
        wxyz = np.zeros((3, 4))
        for k, idx in enumerate(self._link_idx):
            se3 = jaxlie.SE3(fk[int(idx)])
            pos[k] = np.asarray(se3.translation())
            wxyz[k] = np.asarray(se3.rotation().wxyz)
        return pos, wxyz

    def warmup(self, q0: np.ndarray) -> None:
        """Trigger the JIT compile (a few seconds) before the control loop.

        q0 is in DRIVER coordinates (as read from the robot)."""
        q_urdf = self.driver_to_urdf(q0)
        pos, wxyz = self.hold_poses(q_urdf)
        self._ik.solve(q_urdf, pos, wxyz,
                       pos_weights=self._pos_weights,
                       ori_weights=self._ori_weights)

    def solve(
        self,
        prev_q: np.ndarray,
        targets: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]],
    ) -> np.ndarray:
        """One coupled tick.

        targets: {arm: (position (3,), wxyz (4,))} for arms being commanded;
        omitted/None arms hold their current commanded pose.
        Returns the full actuated configuration (finger joints untouched by
        any cost; the caller slices per-arm joints as before).
        """
        prev_urdf = self.driver_to_urdf(prev_q)
        pos, wxyz = self.hold_poses(prev_urdf)
        for k, arm in enumerate(ARM_ORDER):
            t = targets.get(arm)
            if t is not None:
                pos[k] = np.asarray(t[0], dtype=np.float64)
                wxyz[k] = np.asarray(t[1], dtype=np.float64)
        res = self._ik.solve(prev_urdf, pos, wxyz,
                             pos_weights=self._pos_weights,
                             ori_weights=self._ori_weights)
        self.last_solve_ms = res.solve_ms
        self.last_iterations = res.iterations
        return self.urdf_to_driver(res.q, ref_driver=prev_q)


def build_study_ik(robot, control_dt: float = STUDY_DT,
                   waist_driver_shift: float = 0.0,
                   urdf_path: Optional[str] = None,
                   max_iterations: int = MAX_ITERATIONS) -> "CoupledStudyIK":
    """The deployed solver, built from arm_config — one call, no weights to pass.

    This exists because "which weights?" is the question that produced three
    different IK solvers in this directory, all of them called from teleop
    loops, none of them agreeing.  Every weight the study validated lives in
    the module constants above; the only things a caller legitimately knows are
    its control period and the middle waist's Homing_Offset, so those are the
    only things this takes.

    ALWAYS ALL THREE ARMS, whatever mode the caller runs.  The self-collision
    cost and the gates are statements about the robot, not about the arms one
    session happens to drive: leaving an arm out of the problem does not make
    it leave the workspace.  Arms with no target in a given tick hold the
    configuration in `prev_q` (CoupledStudyIK.solve), so a single-arm caller
    passes one target and gets collision avoidance against the other two for
    free -- provided it seeded their joints with where they actually are.

    Returns a CoupledStudyIK; call `.solve(prev_q, {arm: (pos, wxyz)})` with
    DRIVER-coordinate joints, exactly as data_collection.py does."""
    from arm_config import ARM_CONFIG as _AC
    from arm_config import URDF_PATH as _UP
    return CoupledStudyIK(
        robot,
        urdf_path or _UP,
        ee_links={a: _AC[a]["ee_link"] for a in ARM_ORDER},
        control_dt=control_dt,
        max_iterations=max_iterations,
        waist_driver_shift=waist_driver_shift,
    )
