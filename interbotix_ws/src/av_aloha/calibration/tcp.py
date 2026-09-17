"""Tool centre point: commanding the grasp point instead of the flange.

THE PROBLEM THIS SOLVES
=======================
`arm_config.ARM_CONFIG[arm]["ee_link"]` is `*_gripper_base` -- the gripper
MOUNTING PLATE.  Every pose the IK tracks, every pose FK reports, and every
pose written into a dataset is that plate, not the point between the
fingers.

Two consequences, both of which show up the moment you put a ruler on the
robot:

1.  The grasp point sits about 72 mm further out along the plate's LOCAL
    +z (the approach axis).  So the world-frame z of the fingertips is not
    the world-frame z of `gripper_base`, and the gap between them depends
    on the wrist orientation -- it is the full 72 mm when the gripper
    points straight down, and zero when it points horizontally.

2.  Commanding a pure translation of the PLATE does not translate the
    FINGERS by the same amount unless the orientation holds exactly.  A
    lever arm turns orientation error into position error: at 72 mm, a
    10 deg tilt moves the fingertips 12.5 mm.  That is why "go up 5 cm"
    can undershoot at the fingertips while the plate itself tracked well.

Commanding in TCP space fixes both.  The target becomes the grasp point:

    T_world_flange = T_world_tcp @ inv(T_flange_tcp)

so the solver has to put the flange wherever it must to place the FINGERS
where you asked, and orientation drift is compensated automatically
instead of leaking into fingertip position.

WHAT THE DEFAULT OFFSET IS, AND ISN'T
=====================================
The default is DERIVED, not measured.  `giava.urdf` was flattened from the
upstream interbotix vx300s description and lost the gripper tail, including
its canonical grasp frame `ee_gripper_link`.  That frame sat at

    0.042825 + 0.025875 + 0.0385 = 0.1072 m  along gripper_link's +x

which, re-expressed in `gripper_base` coordinates, is (0, 0.00006, 0.0722).
It assumes GIAVA's custom fingers grasp where the stock ones did.

MEASURE IT.  Put the gripper somewhere you can reach, note where the
fingers actually close, and record the offset in tcp_offsets.json.  Until
then everything here is labelled `derived` rather than `measured`.

CONVENTION
    T_flange_tcp maps points from the TCP frame into the flange
    (`*_gripper_base`) frame, and is the pose of the TCP expressed in the
    flange.  Metres.  The TCP keeps the flange's ORIENTATION by default --
    only the origin moves -- so "TCP rpy" and "flange rpy" are the same
    unless you give a rotation too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import T_from_xyz_rpy, invert_T, load_json, make_T  # noqa: E402

TCP_CONFIG = HERE / "tcp_offsets.json"

## Upstream interbotix vx300s.urdf.xacro, the chain giava.urdf dropped:
##   gripper_link --[ee_arm      +0.042825 x]--> ee_arm_link
##                --[gripper_bar  0.0       x]--> gripper_bar_link
##                --[ee_bar      +0.025875 x]--> fingers_link
##                --[ee_gripper  +0.0385   x]--> ee_gripper_link
UPSTREAM_TCP_ALONG_GRIPPER_LINK_X = 0.042825 + 0.0 + 0.025875 + 0.0385

## giava.urdf's fixed joint gripper_link -> gripper_base.
_FLANGE_FROM_GRIPPER_LINK = T_from_xyz_rpy(
    (0.035, 0.0, 0.0), (-1.570000, 0.000796, -1.570796))


def _derived_offset() -> np.ndarray:
    """T_flange_tcp from the upstream description, as a translation only."""
    p = (invert_T(_FLANGE_FROM_GRIPPER_LINK)
         @ np.array([UPSTREAM_TCP_ALONG_GRIPPER_LINK_X, 0.0, 0.0, 1.0]))[:3]
    return make_T(p, np.eye(3))


## Arms that have a gripper at all.  The middle arm carries a camera; its
## "ee_link" is a mount frame and no TCP concept applies.
TCP_ARMS = ("left", "right")


def load_overrides(path: Path = TCP_CONFIG) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    data = load_json(p)
    return {k: v for k, v in data.get("arms", {}).items()
            if isinstance(v, dict) and v.get("translation_xyz_m")}


def resolve(arm: str, path: Path = TCP_CONFIG) -> Dict[str, Any]:
    """Effective TCP offset for one arm, measured values preferred."""
    if arm not in TCP_ARMS:
        return {"arm": arm, "T_flange_tcp": None, "provenance": "n/a",
                "measured": False,
                "note": ("no gripper on this arm -- its ee_link is a camera "
                         "mount frame, so there is no tool centre point")}
    override = load_overrides(path).get(arm)
    if override is not None:
        rpy = override.get("rpy_rad") or (0.0, 0.0, 0.0)
        return {
            "arm": arm,
            "T_flange_tcp": T_from_xyz_rpy(override["translation_xyz_m"], rpy),
            "provenance": override.get("provenance", "measured"),
            "measured": bool(override.get("measured", True)),
            "note": override.get("note", "from tcp_offsets.json"),
            "source": str(path),
        }
    return {
        "arm": arm,
        "T_flange_tcp": _derived_offset(),
        "provenance": "derived_from_upstream_urdf",
        "measured": False,
        "note": ("derived from the upstream interbotix ee_gripper_link, "
                 "which giava.urdf dropped. Assumes GIAVA's custom fingers "
                 "grasp where the stock ones did. NOT measured."),
        "source": "built-in default (tcp.py)",
    }


def tcp_pose(T_world_flange: np.ndarray, arm: str,
             path: Path = TCP_CONFIG) -> np.ndarray:
    """T_world_tcp from the flange pose."""
    off = resolve(arm, path)["T_flange_tcp"]
    return T_world_flange if off is None else T_world_flange @ off


def flange_target(T_world_tcp_target: np.ndarray, arm: str,
                  path: Path = TCP_CONFIG) -> np.ndarray:
    """The flange pose that puts the TCP at the requested pose.

    This is the whole point: the IK tracks the flange, so a TCP command has
    to be converted before it is handed over."""
    off = resolve(arm, path)["T_flange_tcp"]
    return (T_world_tcp_target if off is None
            else T_world_tcp_target @ invert_T(off))


def describe(arm: str, path: Path = TCP_CONFIG) -> str:
    e = resolve(arm, path)
    if e["T_flange_tcp"] is None:
        return f"{arm}: no TCP ({e['note']})"
    t = e["T_flange_tcp"][:3, 3]
    return (f"{arm}: TCP at ({t[0]*1e3:+.1f}, {t[1]*1e3:+.1f}, "
            f"{t[2]*1e3:+.1f}) mm in the flange frame "
            f"[{e['provenance']}, measured={e['measured']}]")
