"""Shared plumbing for the GIAVA calibration / validation pipeline.

Paths, provenance metadata, never-overwrite JSON writing, and the SE(3)
helpers every calibration script uses.


FRAME NAMING CONVENTION  --  used everywhere in this package, no exceptions
==========================================================================
A variable or JSON key named ``T_a_b`` is a 4x4 homogeneous matrix that

  * maps a point expressed in frame ``b`` into frame ``a``:

        p_a = T_a_b @ p_b

  * equivalently, IS THE POSE OF FRAME ``b`` EXPRESSED IN FRAME ``a``:
    its translation column is b's origin written in a's coordinates, and
    its rotation block's columns are b's axes written in a's coordinates.

Composition reads left to right with matching inner names:

        T_world_camera = T_world_ee @ T_ee_camera

and inversion swaps them: ``T_b_a = inv(T_a_b)``.

Every transform this package writes to disk carries an explicit
``"convention"`` string spelling this out, so a file is never ambiguous
about direction even when read years later without this docstring.


UNITS
=====
Metres and radians, always -- matching giava.urdf, pyroki and the whole
existing GIAVA control stack.  Quaternions are stored **wxyz** (scalar
first), which is what pyroki / jaxlie forward kinematics returns and what
``CoupledStudyIK.solve`` expects as a target.  Anything that talks to
``scipy.spatial.transform.Rotation`` must convert -- scipy is xyzw.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

## ------------------------------------------------------------------ ##
## Paths
## ------------------------------------------------------------------ ##

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ -- where the rest of the robot stack lives.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
def _find_repo_root(start: Path) -> Path:
    """Walk up for giava.urdf rather than counting parents[N] -- the count was
    already wrong once, the day this package moved up a level."""
    for cand in (start, *start.parents):
        if (cand / "giava.urdf").exists():
            return cand
    raise SystemExit(f"could not find giava.urdf above {start}")


REPO_ROOT = _find_repo_root(HERE)

URDF_PATH = os.environ.get("GIAVA_URDF", str(REPO_ROOT / "giava.urdf"))

## Calibration RESULTS live under calibration/data/ -- separate from source,
## and gitignored by the repo-wide `*.json` rule (see calibration/.gitignore,
## which re-includes the hand-maintained config files only).
DATA_ROOT = Path(os.environ.get("GIAVA_CALIB_DATA", str(HERE / "data")))

## Per-subsystem output directories (phase 9 layout).
DIR_ROBOT = DATA_ROOT / "robot"            # motion validation runs
DIR_CAMERAS = DATA_ROOT / "cameras"        # per-camera intrinsics (factory + charuco)
DIR_CHARUCO = DATA_ROOT / "charuco"        # captured board images
DIR_SYNC = DATA_ROOT / "sync"              # two-camera timing logs
DIR_COMPARE = DATA_ROOT / "comparisons"    # factory-vs-opencv reports
DIR_VISER = DATA_ROOT / "viser_captures"   # manual captures from the viewer

ALL_DIRS = (DIR_ROBOT, DIR_CAMERAS, DIR_CHARUCO, DIR_SYNC, DIR_COMPARE, DIR_VISER)


def ensure_dirs() -> None:
    """Create the calibration data tree (idempotent)."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """Run-directory stamp, matching the repo convention (dataset.py:165)."""
    return time.strftime("%Y%m%d_%H%M%S")


## ROS noetic's python packages (rospy, interbotix_xs_modules) reach the
## interpreter through PYTHONPATH, which only exists once ROS has been
## sourced.  data_collection.py appends these paths defensively so it works
## either way; the hardware-touching calibration scripts do the same, so
## `python calibration/x.py --from-robot` behaves identically whether or not
## the shell happened to source ROS first.
ROS_PATHS = (
    "/opt/ros/noetic/lib/python3/dist-packages",
    str(REPO_ROOT / "interbotix_ws" / "devel" / "lib" / "python3" /
        "dist-packages"),
)


def ensure_ros_path() -> None:
    """Put the ROS dist-packages directories on sys.path (idempotent)."""
    import sys as _sys
    for p in ROS_PATHS:
        if Path(p).is_dir() and p not in _sys.path:
            _sys.path.append(p)


## ------------------------------------------------------------------ ##
## Interactive prompts
## ------------------------------------------------------------------ ##

class NoStdinError(RuntimeError):
    """Raised when a prompt is needed but stdin cannot be read."""


_CONDA_RUN_HINT = """
  Could not read from stdin, so this prompt cannot be answered.

  The usual cause is `conda run`, which does NOT forward stdin -- every
  prompt in these scripts dies with EOFError under it.  ACTIVATE the
  environment instead of wrapping the command:

      conda activate gym_av312
      cd /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts
      python calibration/<script>.py ...

  For a non-interactive run, pass the flag that skips the prompt
  (--yes / --no-prompt), or use --dry-run.
"""


def wait_for_enter(message: str = "press ENTER to continue... ") -> None:
    """Pause for the operator, with a useful error when stdin is absent."""
    try:
        input(f"      {message}")
    except EOFError:
        raise NoStdinError(_CONDA_RUN_HINT) from None


def confirm(message: str, expect: str = "yes") -> bool:
    """Ask for an explicit typed confirmation before moving hardware."""
    try:
        return input(f"  {message}").strip().lower() == expect
    except EOFError:
        raise NoStdinError(_CONDA_RUN_HINT) from None


## ------------------------------------------------------------------ ##
## Provenance
## ------------------------------------------------------------------ ##

def git_rev() -> str:
    """Short HEAD sha, or 'unknown'.  Same call as ik/study/run_study.py."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> Optional[bool]:
    """True when the working tree has uncommitted changes (None if unknown).

    Worth recording: a calibration produced from a dirty tree cannot be
    reproduced from the sha alone."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def provenance(method: str, **extra: Any) -> Dict[str, Any]:
    """The metadata block every calibration artifact carries (phase 9).

    `method` names HOW the numbers were obtained, e.g.
    "realsense_factory_intrinsics" or "opencv_charuco".  Anything
    method-specific goes in **extra."""
    meta: Dict[str, Any] = {
        "method": method,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "unix_time": time.time(),
        "host": socket.gethostname(),
        "user": _safe_user(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git": git_rev(),
        "git_dirty": git_dirty(),
        "urdf": str(URDF_PATH),
        "units": "metres, radians",
    }
    meta.update(extra)
    return meta


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


## ------------------------------------------------------------------ ##
## Never-overwrite JSON writing
## ------------------------------------------------------------------ ##

class OutputExistsError(RuntimeError):
    """Raised instead of clobbering an existing calibration result."""


class _NumpyJSON(json.JSONEncoder):
    """numpy scalars/arrays are ubiquitous here; make them JSON-native."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: Path, overwrite: bool = False) -> Path:
    """Write `obj` as pretty JSON, refusing to clobber by default.

    Calibration results are experimental records: silently replacing one
    destroys the ability to compare against it later.  Pass
    overwrite=True (wired to an explicit --overwrite flag) to replace."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise OutputExistsError(
            f"{path} already exists.\n"
            f"  Calibration results are never overwritten silently.\n"
            f"  Re-run without --out to get a fresh timestamped file, "
            f"or pass --overwrite to replace this one deliberately."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=_NumpyJSON))
    return path


def append_json_run(path: Path, run: Dict[str, Any],
                    header: Optional[Dict[str, Any]] = None) -> int:
    """Append one run to an accumulating JSON file. Returns the run count.

    Deliberate accumulation, not the silent overwriting save_json refuses:
    repeated measurements only mean something TOGETHER, so they belong in
    one file where the spread across them is visible.

    The write is atomic -- a temp file in the same directory, then a
    rename. A crash midway through would otherwise truncate every earlier
    run, which is a bad way to lose an afternoon of measurements."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any]
    if path.exists():
        try:
            doc = json.loads(path.read_text())
        except Exception as exc:
            raise RuntimeError(
                f"{path} exists but is not readable JSON ({exc}). Move it "
                f"aside rather than risk appending to a damaged file.")
        if not isinstance(doc.get("runs"), list):
            raise RuntimeError(
                f"{path} has no 'runs' list -- it was not written by "
                f"append_json_run. Use a different --out.")
    else:
        doc = dict(header or {})
        doc["runs"] = []

    run = dict(run)
    run.setdefault("run_index", len(doc["runs"]))
    run.setdefault("recorded", time.strftime("%Y-%m-%d %H:%M:%S"))
    doc["runs"].append(run)
    doc["n_runs"] = len(doc["runs"])

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, cls=_NumpyJSON))
    os.replace(tmp, path)          # atomic on the same filesystem
    return len(doc["runs"])


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def latest_matching(directory: Path, pattern: str) -> Optional[Path]:
    """Newest file matching `pattern` in `directory`, or None.

    Lets the comparison and calibration tools default to "the most recent
    result" without the operator pasting timestamps around."""
    hits = sorted(Path(directory).glob(pattern))
    return hits[-1] if hits else None


## ------------------------------------------------------------------ ##
## SE(3) helpers
## ------------------------------------------------------------------ ##
##
## Kept deliberately small and dependency-light (numpy only): the frame
## report must run even in an environment without jax.  Anything that
## needs jaxlie already has it.

def quat_wxyz_to_matrix(wxyz: Sequence[float]) -> np.ndarray:
    """Rotation matrix from a scalar-first quaternion."""
    w, x, y, z = (float(v) for v in wxyz)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion from a rotation matrix (Shepperd's method).

    Branching on the largest diagonal term keeps it numerically stable at
    the 180-degree cases where the naive trace formula loses precision."""
    R = np.asarray(R, dtype=float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = (0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q) * (1.0 if q[0] >= 0 else -1.0)


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF rpy (fixed-axis roll-pitch-yaw) -> rotation matrix: Rz @ Ry @ Rx.

    This is the URDF/ROS convention: extrinsic rotations about the FIXED
    x, then y, then z axes, which composes as Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Inverse of rpy_to_matrix -- returns (roll, pitch, yaw) in radians."""
    R = np.asarray(R, dtype=float)
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) > 1.0 - 1e-9:  # gimbal lock: roll and yaw are degenerate
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def make_T(translation: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 from a 3-vector and a 3x3 rotation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def T_from_pos_quat(pos: Sequence[float], wxyz: Sequence[float]) -> np.ndarray:
    return make_T(pos, quat_wxyz_to_matrix(wxyz))


def T_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """Build a transform from the URDF's own (xyz, rpy) spelling."""
    return make_T(xyz, rpy_to_matrix(rpy))


def invert_T(T: np.ndarray) -> np.ndarray:
    """Rigid-transform inverse: T_b_a from T_a_b (no general matrix inverse)."""
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def T_to_dict(T: np.ndarray, frame_a: str, frame_b: str) -> Dict[str, Any]:
    """Serialize a transform with its direction stated in full.

    The `convention` string is deliberately verbose and self-contained:
    whoever reads this JSON must not have to guess which way it points."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    return {
        "name": f"T_{frame_a}_{frame_b}",
        "parent_frame": frame_a,
        "child_frame": frame_b,
        "convention": (
            f"T_{frame_a}_{frame_b} maps points from '{frame_b}' into "
            f"'{frame_a}':  p_{frame_a} = T @ p_{frame_b}.  Equivalently it "
            f"is the pose of '{frame_b}' expressed in '{frame_a}'.  "
            f"Inverse direction = numpy.linalg.inv of this matrix."
        ),
        "matrix_4x4_row_major": T.tolist(),
        "translation_xyz_m": T[:3, 3].tolist(),
        "quaternion_wxyz": matrix_to_quat_wxyz(R).tolist(),
        "rpy_rad": matrix_to_rpy(R).tolist(),
        "rpy_deg": np.degrees(matrix_to_rpy(R)).tolist(),
    }


def T_from_dict(d: Dict[str, Any]) -> np.ndarray:
    """Read back a transform written by T_to_dict."""
    return np.asarray(d["matrix_4x4_row_major"], dtype=float)


def rotation_angle_deg(R: np.ndarray) -> float:
    """Magnitude of a rotation, in degrees -- the geodesic angle."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def format_T(T: np.ndarray, indent: str = "    ") -> str:
    """Human-readable multi-line rendering for console reports."""
    T = np.asarray(T, dtype=float)
    rpy = np.degrees(matrix_to_rpy(T[:3, :3]))
    lines = [
        f"{indent}position xyz [m]  : "
        f"[{T[0, 3]:+.5f}, {T[1, 3]:+.5f}, {T[2, 3]:+.5f}]",
        f"{indent}rotation rpy [deg]: "
        f"[{rpy[0]:+8.3f}, {rpy[1]:+8.3f}, {rpy[2]:+8.3f}]",
        f"{indent}quaternion wxyz   : "
        + "[" + ", ".join(f"{v:+.5f}" for v in matrix_to_quat_wxyz(T[:3, :3])) + "]",
        f"{indent}matrix:",
    ]
    for row in T:
        lines.append(indent + "  [" + "  ".join(f"{v:+9.5f}" for v in row) + "]")
    return "\n".join(lines)
