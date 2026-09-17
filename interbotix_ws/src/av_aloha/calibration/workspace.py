"""Environment keep-out limits, and swept-volume checks.

WHY THIS EXISTS
===============
The collision model the IK uses (`study_ik` / `CoupledStudyIK.min_clearance`,
the frozen 180-sphere decomposition) checks the robot against ITSELF and
nothing else.  It does not know the table exists.  It does not know about
the frame bars, the monitor, the workstation, or anything else bolted
around the arms.

So a reported clearance of "+19 mm" means *the arm is not intersecting
itself*.  It says nothing whatsoever about whether the arm is about to
swing into a bar.  Any tool that moves the robot through a large sweep has
to answer that question separately, and the only source of truth is a
measurement of the actual rig.

This module holds that measurement (`workspace_limits.json`) and checks
candidate motions against it.  Until the box is filled in, the limits are
UNKNOWN and the motion tools say so loudly rather than implying safety they
cannot verify.

CONVENTION
    The box is axis-aligned in the WORLD frame -- giava.urdf's root link
    `base`: +x operator's left, +y operator's backward, +z up.  Limits are
    metres.  A null bound means "not measured", and is treated as unknown
    rather than infinite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

LIMITS_FILE = HERE / "workspace_limits.json"

## Links sampled when sweeping an arm.  The end-effector is NOT the widest
## part of the arm through a waist rotation -- the elbow and forearm swing
## through their own arcs -- so a footprint computed from the EE alone
## understates the swept volume.
def arm_links(frames, arm: str) -> List[str]:
    """Every link belonging to one arm, for swept-volume sampling."""
    return [l for l in frames.link_names if l.startswith(f"{arm}_")]


def load_limits(path: Path = LIMITS_FILE) -> Dict[str, Any]:
    """Read workspace_limits.json ({} when absent)."""
    from common import load_json
    p = Path(path)
    if not p.exists():
        return {}
    raw = load_json(p)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def limits_known(limits: Dict[str, Any]) -> bool:
    box = (limits or {}).get("keep_out_box") or {}
    return any(box.get(k) is not None
               for k in ("x_min", "x_max", "y_min", "y_max",
                         "z_min", "z_max"))


def describe_limits(limits: Dict[str, Any]) -> str:
    box = (limits or {}).get("keep_out_box") or {}
    if not limits_known(limits):
        return ("NOT MEASURED -- no environment limits are recorded, so "
                "nothing can be checked against the rig.")
    def f(k):
        v = box.get(k)
        return "  (unset)" if v is None else f"{v:+8.3f}"
    return (f"x [{f('x_min')}, {f('x_max')}]  "
            f"y [{f('y_min')}, {f('y_max')}]  "
            f"z [{f('z_min')}, {f('z_max')}]  metres")


def violations(points: np.ndarray,
               limits: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """Which bounds a set of world-frame points exceeds.

    Returns [(bound_name, worst_value, limit)] -- empty when inside, or when
    nothing has been measured (in which case the CALLER must treat it as
    unknown, not as safe)."""
    box = (limits or {}).get("keep_out_box") or {}
    if not points.size:
        return []
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    out = []
    for axis, i in (("x", 0), ("y", 1), ("z", 2)):
        lo, hi = box.get(f"{axis}_min"), box.get(f"{axis}_max")
        if lo is not None and p[:, i].min() < lo:
            out.append((f"{axis}_min", float(p[:, i].min()), float(lo)))
        if hi is not None and p[:, i].max() > hi:
            out.append((f"{axis}_max", float(p[:, i].max()), float(hi)))
    return out


def swept_points(frames, bridge, q_driver_base: np.ndarray, joint_index: int,
                 values_rad: Sequence[float], links: Sequence[str],
                 samples_between: int = 6) -> np.ndarray:
    """World-frame points swept by `links` as one joint moves through
    `values_rad`, INCLUDING the intermediate configurations.

    The motion between two commanded angles is a real sweep, not a
    teleport: checking only the commanded stops would miss everything the
    arm passes through on the way."""
    dense: List[float] = []
    vals = list(values_rad)
    for a, b in zip(vals[:-1], vals[1:]):
        dense += list(np.linspace(a, b, samples_between + 1, endpoint=False))
    dense.append(vals[-1])

    pts = []
    for v in dense:
        q = np.asarray(q_driver_base, dtype=np.float64).copy()
        q[joint_index] = v
        fk = frames.fk(bridge.to_urdf(q))
        for link in links:
            pts.append(fk[link][:3, 3])
    return np.asarray(pts, dtype=float)


def footprint(points: np.ndarray, about_xy: Optional[np.ndarray] = None
              ) -> Dict[str, Any]:
    """Bounding box of a swept point cloud, plus radius about an axis."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    out = {
        "x_min_m": float(p[:, 0].min()), "x_max_m": float(p[:, 0].max()),
        "y_min_m": float(p[:, 1].min()), "y_max_m": float(p[:, 1].max()),
        "z_min_m": float(p[:, 2].min()), "z_max_m": float(p[:, 2].max()),
        "n_points": int(len(p)),
    }
    out["x_extent_m"] = out["x_max_m"] - out["x_min_m"]
    out["y_extent_m"] = out["y_max_m"] - out["y_min_m"]
    if about_xy is not None:
        r = np.linalg.norm(p[:, :2] - np.asarray(about_xy, float), axis=1)
        out["max_radius_m"] = float(r.max())
    return out


def report_footprint(fp: Dict[str, Any], limits: Dict[str, Any],
                     points: np.ndarray) -> bool:
    """Print the swept envelope and check it. Returns True when verified
    safe against MEASURED limits; False when limits are unknown."""
    print()
    print("  SWEPT ENVELOPE of this motion (world frame, every arm link,")
    print("  including the configurations passed through in between):")
    print()
    print(f"      x  {fp['x_min_m']:+.3f} .. {fp['x_max_m']:+.3f} m"
          f"      (extent {fp['x_extent_m']*1e3:6.0f} mm)")
    print(f"      y  {fp['y_min_m']:+.3f} .. {fp['y_max_m']:+.3f} m"
          f"      (extent {fp['y_extent_m']*1e3:6.0f} mm)")
    print(f"      z  {fp['z_min_m']:+.3f} .. {fp['z_max_m']:+.3f} m")
    if "max_radius_m" in fp:
        print(f"      max radius about the waist axis: "
              f"{fp['max_radius_m']*1e3:.0f} mm")

    print()
    print(f"  Environment limits: {describe_limits(limits)}")
    if not limits_known(limits):
        print(f"""
  *** THE ENVIRONMENT IS NOT MODELLED. ***
  The IK collision model checks the robot against ITSELF only -- it does
  not know the table, the frame bars, or anything else around the arms.
  Nothing here has verified this sweep will clear them.

  Measure the free space around this arm once and record it in
      {LIMITS_FILE}
  and every motion tool will check against it from then on.

  Until then: compare the envelope above against the rig BY EYE before
  answering the prompt.
""")
        return False

    bad = violations(points, limits)
    if bad:
        print()
        print("  *** THIS MOTION LEAVES THE MEASURED FREE SPACE ***")
        for name, worst, limit in bad:
            print(f"      {name}: reaches {worst:+.3f} m, limit {limit:+.3f} m"
                  f"   ({abs(worst-limit)*1e3:.0f} mm over)")
        print()
        return False
    print("  the swept envelope is inside the measured free space.")
    return True


def points_for_configs(frames, bridge, q_drivers: Sequence[np.ndarray],
                       links: Sequence[str]) -> np.ndarray:
    """World-frame points of `links` over a list of DRIVER configurations.

    For Cartesian paths, where the configurations come from IK rather than
    from sweeping one joint.  Pass every waypoint the arm will be commanded
    through, not just the endpoints."""
    pts = []
    for q in q_drivers:
        fk = frames.fk(bridge.to_urdf(np.asarray(q, dtype=np.float64)))
        for link in links:
            pts.append(fk[link][:3, 3])
    return np.asarray(pts, dtype=float)


def check_configs(frames, bridge, q_drivers, arm: str,
                  limits: Optional[Dict[str, Any]] = None):
    """(ok, violations, points) for a planned Cartesian path.

    `ok` is True only when limits are MEASURED and nothing exceeds them.
    Unknown limits give ok=False with an empty violation list -- unverified
    is not the same as safe, and callers must not conflate them."""
    limits = load_limits() if limits is None else limits
    pts = points_for_configs(frames, bridge, q_drivers, arm_links(frames, arm))
    if not limits_known(limits):
        return False, [], pts
    bad = violations(pts, limits)
    return (not bad), bad, pts
