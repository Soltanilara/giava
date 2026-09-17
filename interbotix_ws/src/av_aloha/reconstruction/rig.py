"""Assembling a StereoRig out of what calibration/ wrote.

Two reference frames are possible and the difference matters, so this
module never picks one silently:

  ANCHORED (ref_frame = "base", the robot world frame)
      Requires an entry for each camera in
      `calibration/ee_camera_transforms.json` with rigid_to = "base".
      Written by `calibration/scene_extrinsics.py write` after the rig has
      been anchored to the robot.  Measurements come out directly
      comparable with forward kinematics -- this is what you need to ask
      "does the URDF agree with reality".

  STEREO-ONLY (ref_frame = the reference camera's optical frame)
      Requires only `data/extrinsics/stereo_*.json`, the camera-to-camera
      calibration.  **Lengths, sizes and shapes are already fully correct
      in this frame** -- the metric scale comes from the ChArUco square,
      not from the robot.  Only the expression in robot coordinates is
      missing.  So you can measure link lengths and check the rulers
      before ever touching the arms.

Asking for the anchored rig when no anchor exists raises, with the command
that would produce one.  It does not quietly fall back -- a measurement
labelled "base" that is actually in camera coordinates is the worst
possible failure mode here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_CALIB = _HERE.parent / "calibration"
for _p in (str(_HERE), str(_CALIB), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from camera import PinholeCamera, load_intrinsics  # noqa: E402
from stereo import StereoRig  # noqa: E402

SCENE_CAMERAS = ("top_scene", "low_scene")


def extrinsics_dir() -> Path:
    from common import DATA_ROOT
    return Path(DATA_ROOT) / "extrinsics"


def latest_stereo(path: Optional[Path] = None) -> Path:
    """Newest stereo extrinsics file, or a clear error saying how to make one."""
    from common import latest_matching
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist")
        return p
    d = extrinsics_dir()
    hit = latest_matching(d, "stereo_*.json") if d.exists() else None
    if hit is None:
        raise FileNotFoundError(
            f"no camera-to-camera calibration in {d}.\n"
            f"  Produce one:\n"
            f"    python calibration/scene_extrinsics.py collect "
            f"--cameras top_scene low_scene\n"
            f"    python calibration/scene_extrinsics.py solve --dir <session>")
    return hit


## ------------------------------------------------------------------ ##
## Loaders
## ------------------------------------------------------------------ ##

def load_stereo_rig(cameras: Sequence[str] = SCENE_CAMERAS,
                    stereo_file: Optional[Path] = None,
                    prefer_intrinsics: str = "charuco") -> StereoRig:
    """Rig posed in the REFERENCE CAMERA's optical frame."""
    from common import load_json

    path = latest_stereo(stereo_file)
    data = load_json(path)
    entries = data.get("cameras", {})
    ref = data.get("reference_camera")

    missing = [c for c in cameras if c not in entries]
    if missing:
        raise KeyError(
            f"{path} has no extrinsics for {missing} "
            f"(has: {sorted(entries)}). Re-run the collect/solve pair with "
            f"all the cameras you want in the rig.")

    cams: List[PinholeCamera] = []
    for name in cameras:
        cam = load_intrinsics(name, prefer=prefer_intrinsics)
        cam.T_ref_cam = np.asarray(
            entries[name]["T_ref_cam_row_major"], dtype=float)
        cam.ref_frame = ref
        cam.provenance = entries[name].get("provenance", str(path.name))
        cams.append(cam)
    return StereoRig(cams, ref_frame=ref, source=f"stereo:{path.name}")


def load_world_rig(cameras: Sequence[str] = SCENE_CAMERAS,
                   prefer_intrinsics: str = "charuco",
                   q_urdf: Optional[np.ndarray] = None) -> StereoRig:
    """Rig posed in the robot world frame `base`.

    Static cameras come straight out of `ee_camera_transforms.json`.  A
    camera bolted to an arm (a wrist D405) can also join the rig, but its
    pose depends on the joint angles, so `q_urdf` must be supplied -- and
    the resulting rig is only valid for THAT configuration.  That is a
    real constraint, not an inconvenience: a wrist camera moves, so a
    "rig" containing one is a snapshot."""
    import camera_mount as CM

    cams: List[PinholeCamera] = []
    frames = None
    for name in cameras:
        mount = CM.resolve_mount(name)
        if mount["T_parent_optical"] is None:
            raise ValueError(
                f"camera '{name}' has no calibrated pose "
                f"(provenance='{mount['provenance']}').\n"
                f"  Anchor the scene rig to the robot and record it:\n"
                f"    python calibration/scene_extrinsics.py anchor "
                f"--dir <session>\n"
                f"    python calibration/scene_extrinsics.py write "
                f"--stereo <f> --anchor <f>")

        parent = mount["rigid_to"]
        if parent == "base":
            T_base_cam = np.asarray(mount["T_parent_optical"], dtype=float)
        else:
            if q_urdf is None:
                raise ValueError(
                    f"camera '{name}' is mounted on '{parent}', which moves. "
                    f"Pass q_urdf to fix the configuration this rig "
                    f"describes.")
            if frames is None:
                from kinematics import RobotFrames
                frames = RobotFrames()
            T_base_cam = (frames.link_pose(q_urdf, parent)
                          @ np.asarray(mount["T_parent_optical"], dtype=float))

        cam = load_intrinsics(name, prefer=prefer_intrinsics)
        cam.T_ref_cam = T_base_cam
        cam.ref_frame = "base"
        cam.provenance = mount.get("provenance", "unknown")
        cams.append(cam)
    return StereoRig(cams, ref_frame="base",
                     source="ee_camera_transforms.json")


def load_rig(cameras: Sequence[str] = SCENE_CAMERAS,
             frame: str = "auto",
             prefer_intrinsics: str = "charuco",
             q_urdf: Optional[np.ndarray] = None) -> StereoRig:
    """The rig, in the best frame available.

    frame="base"    insist on robot world coordinates (raises if unanchored)
    frame="stereo"  insist on the reference camera's frame
    frame="auto"    world if anchored, otherwise stereo-only -- and the
                    result always states which, in `rig.ref_frame`
    """
    if frame == "base":
        return load_world_rig(cameras, prefer_intrinsics, q_urdf)
    if frame == "stereo":
        return load_stereo_rig(cameras, prefer_intrinsics=prefer_intrinsics)
    if frame != "auto":
        raise ValueError(f"frame must be base|stereo|auto, got '{frame}'")

    try:
        return load_world_rig(cameras, prefer_intrinsics, q_urdf)
    except (ValueError, KeyError, FileNotFoundError) as e:
        first = str(e).splitlines()[0]
        rig = load_stereo_rig(cameras, prefer_intrinsics=prefer_intrinsics)
        rig.source += f"  (not anchored to the robot: {first})"
        return rig


def rig_status(cameras: Sequence[str] = SCENE_CAMERAS) -> Dict[str, Any]:
    """What exists and what is missing, without raising."""
    import camera_mount as CM

    out: Dict[str, Any] = {"cameras": {}, "stereo_file": None,
                           "anchored": True}
    try:
        out["stereo_file"] = str(latest_stereo())
    except FileNotFoundError as e:
        out["stereo_error"] = str(e).splitlines()[0]

    for name in cameras:
        entry: Dict[str, Any] = {}
        try:
            cam = load_intrinsics(name)
            entry["intrinsics"] = cam.source
            entry["resolution"] = [cam.width, cam.height]
        except FileNotFoundError as e:
            entry["intrinsics"] = None
            entry["intrinsics_error"] = str(e).splitlines()[0]
        try:
            mount = CM.resolve_mount(name)
            entry["pose_known"] = mount["T_parent_optical"] is not None
            entry["rigid_to"] = mount["rigid_to"]
            entry["provenance"] = mount["provenance"]
        except KeyError:
            entry["pose_known"] = False
        if not entry.get("pose_known"):
            out["anchored"] = False
        out["cameras"][name] = entry
    return out
