"""Where each camera sits relative to the robot  (PHASE 3).

WHAT EXISTS IN THE REPOSITORY TODAY
===================================
Nothing calibrated.  There is no hand-eye result, no ``T_ee_camera``
anywhere in GIAVA, and no camera optical frame in ``giava.urdf``.  What the
repository does contain is *modelling* evidence, and this module surfaces it
explicitly as such:

  * The wrist RealSense D405s are present in ``giava.urdf`` only as a
    **visual/collision mesh** (``aloha_assets/d405_solid.stl``) rigidly
    attached to ``left_gripper_base`` / ``right_gripper_base``.  They have no
    link, no joint, and no optical frame -- so nothing in TF or FK reports
    a wrist-camera pose.
  * The MuJoCo sim model (``gym_av_aloha/.../aloha.xml``) defines cameras
    ``wrist_cam_left`` / ``wrist_cam_right`` in the same body, at a position
    that matches the URDF mesh origin to sub-micron.  That gives an
    *orientation* for the optical axis which the mesh origin alone does not.
  * The OAK-D stereo pair on the middle (camera) arm has NO pose
    information of any kind relative to ``middle_camera``.
  * ``top_scene`` and ``low_scene`` are static room cameras, not mounted on
    any arm; their pose in world is unknown until extrinsic calibration.

So every transform below is either NOMINAL (derived from model files, never
physically validated) or MISSING.  None is a measurement.  The whole point
of this module is to make that distinction impossible to overlook, and to
give a place for a real calibration to land later.


PROVENANCE LEVELS
=================
``urdf_mesh``     geometry origin of a mesh in giava.urdf.  Locates the
                  camera BODY; its orientation is the STL's authoring
                  frame, which is NOT the optical frame.
``mujoco_model``  a camera element in the MuJoCo scene.  Gives an optical
                  axis, but authored by hand for simulation -- unvalidated
                  against the physical rig.
``calibrated``    produced by a calibration procedure and written to
                  ee_camera_transforms.json.  None exist yet.
``unknown``       no information in the repository.

Only ``calibrated`` entries should ever be trusted for metric work.


DIRECTION CONVENTION
====================
Every transform here is named ``T_<parent>_<child>`` and means

    p_parent = T_parent_child @ p_child

i.e. it IS the pose of the child frame expressed in the parent frame.  See
``common.py`` for the full statement.  The composition this pipeline is
being built toward is

    T_world_camera = T_world_ee @ T_ee_camera

with ``T_world_ee`` from forward kinematics (``kinematics.RobotFrames``) and
``T_ee_camera`` from here.


OPTICAL FRAME CONVENTION
========================
Camera frames are OpenCV optical frames: **+x right, +y down, +z forward
along the viewing axis**.  This is what ``cv2.calibrateCamera``,
``solvePnP`` and the intrinsics in this pipeline assume.  It is NOT the
ROS ``*_optical_frame`` naming difference (ROS uses the same axes) and it
is NOT MuJoCo's convention (MuJoCo cameras look along **-z** with +y up),
which is why the MuJoCo-derived values below carry an explicit flip.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    T_from_xyz_rpy,
    T_to_dict,
    format_T,
    load_json,
    rotation_angle_deg,
)

## Camera identity comes from the production camera layer -- never redefined.
sys.path.insert(0, str(SCRIPTS_DIR))
from camera_manager import CAMERA_SERIALS  # noqa: E402

## Operator-maintained overrides (calibrated values land here).
## Overridable so an alternative calibration can be loaded without editing
## the tracked file -- used by the self-tests, and the way to try a new
## camera pose against the old one before committing to it.
TRANSFORM_CONFIG = Path(os.environ.get(
    "GIAVA_EE_TRANSFORMS", str(HERE / "ee_camera_transforms.json")))


## ------------------------------------------------------------------ ##
## Source constants -- copied from the model files, with their location,
## so the derivation below is auditable rather than a pasted matrix.
## ------------------------------------------------------------------ ##

## giava.urdf, <link name="right_gripper_base"> / "left_gripper_base",
## the <visual> whose mesh is aloha_assets/d405_solid.stl.
D405_MESH_XYZ = (0.0, -0.082475, -0.009595)
D405_MESH_RPY = (0.436333, 0.0, -3.141593)          # 0.436333 rad = 25.0 deg

## gym_av_aloha/gym_av_aloha/assets/aloha.xml, <camera name="wrist_cam_left">
## inside body "left_gripper_base" (identical for wrist_cam_right).
D405_MUJOCO_POS = (0.0, -0.0824748, -0.0095955)
D405_MUJOCO_EULER_XYZ = (2.70525955359, 0.0, 0.0)   # = pi - 0.436333
D405_MUJOCO_FOVY_DEG = 58.0

## MuJoCo cameras look down -z with +y up; OpenCV optical frames look down
## +z with +y down.  The change of basis between them is a pi rotation
## about the shared +x (image right) axis.
R_OPENCV_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0])


def _d405_nominal_optical() -> np.ndarray:
    """T_gripperbase_cameraoptical for a wrist D405 (NOMINAL, unvalidated)."""
    T = T_from_xyz_rpy(D405_MUJOCO_POS, D405_MUJOCO_EULER_XYZ)
    out = T.copy()
    out[:3, :3] = T[:3, :3] @ R_OPENCV_FROM_MUJOCO
    return out


def _d405_mesh_origin() -> np.ndarray:
    """T_gripperbase_meshorigin -- where the STL body sits.  Orientation is
    the STL's authoring frame and is NOT the optical axis (it differs from
    the optical frame by exactly 180 deg about x)."""
    return T_from_xyz_rpy(D405_MESH_XYZ, D405_MESH_RPY)


## ------------------------------------------------------------------ ##
## The mount table
## ------------------------------------------------------------------ ##

CAMERA_MOUNTS: Dict[str, Dict[str, Any]] = {
    "left_wrist": {
        "hardware": "Intel RealSense D405",
        "serial": CAMERA_SERIALS.get("left_wrist"),
        "arm": "left",
        ## The frame the camera is rigidly attached to.  This is also the
        ## arm's ee_link in arm_config.py, so T_ee_camera is exactly this.
        "rigid_to": "left_gripper_base",
        "T_parent_optical": _d405_nominal_optical(),
        "provenance": "mujoco_model",
        "validated": False,
        "note": (
            "Position matches the giava.urdf d405_solid.stl mesh origin to "
            "<1 um; orientation comes from the MuJoCo wrist_cam_left camera "
            "converted to the OpenCV optical convention. Never measured on "
            "hardware. The mesh origin itself differs by 180 deg about x -- "
            "see mesh_origin_T below."
        ),
        "mesh_origin_T": _d405_mesh_origin(),
        "nominal_fovy_deg": D405_MUJOCO_FOVY_DEG,
    },
    "right_wrist": {
        "hardware": "Intel RealSense D405",
        "serial": CAMERA_SERIALS.get("right_wrist"),
        "arm": "right",
        "rigid_to": "right_gripper_base",
        "T_parent_optical": _d405_nominal_optical(),
        "provenance": "mujoco_model",
        "validated": False,
        "note": (
            "Identical mount to left_wrist -- giava.urdf gives both gripper "
            "bases byte-identical d405 mesh origins."
        ),
        "mesh_origin_T": _d405_mesh_origin(),
        "nominal_fovy_deg": D405_MUJOCO_FOVY_DEG,
    },
    "oak_left": {
        "hardware": "Luxonis OAK-D (CAM_B / left)",
        "serial": None,  # camera_manager opens the first available device
        "arm": "middle",
        "rigid_to": "middle_camera",
        "T_parent_optical": None,
        "provenance": "unknown",
        "validated": False,
        "note": (
            "NO pose information exists. middle_camera / middle_camera_body / "
            "middle_camera_cover are all the SAME frame (both fixed joints "
            "are exact identity), and none of them is an optical frame. The "
            "unused zedm_macro.urdf.xacro describes a ZED-mini, which is not "
            "the camera currently mounted. Needs hand-eye calibration."
        ),
    },
    "oak_right": {
        "hardware": "Luxonis OAK-D (CAM_C / right)",
        "serial": None,
        "arm": "middle",
        "rigid_to": "middle_camera",
        "T_parent_optical": None,
        "provenance": "unknown",
        "validated": False,
        "note": (
            "As oak_left. The stereo baseline between the two eyes IS known "
            "(64.3 mm, from the external stereo.npz calibration), but "
            "neither eye is located relative to the robot."
        ),
    },
    "top_scene": {
        "hardware": "Intel RealSense D405",
        "serial": CAMERA_SERIALS.get("top_scene"),
        "arm": None,
        ## Static room camera: it is rigid to the WORLD, not to any arm, so
        ## the meaningful transform is T_world_camera directly.
        "rigid_to": "base",
        "T_parent_optical": None,
        "provenance": "unknown",
        "validated": False,
        "note": (
            "Static scene camera, not mounted on an arm. Its pose in the "
            "world frame is unknown and must come from extrinsic "
            "calibration (e.g. a board visible to both this camera and a "
            "wrist camera, or a board held at known robot poses)."
        ),
    },
    "low_scene": {
        "hardware": "Intel RealSense D405",
        "serial": CAMERA_SERIALS.get("low_scene"),
        "arm": None,
        "rigid_to": "base",
        "T_parent_optical": None,
        "provenance": "unknown",
        "validated": False,
        "note": "Static scene camera. As top_scene.",
    },
}


## ------------------------------------------------------------------ ##
## Calibrated overrides
## ------------------------------------------------------------------ ##

def load_overrides(path: Path = TRANSFORM_CONFIG) -> Dict[str, Any]:
    """Read ee_camera_transforms.json ({} when absent or empty)."""
    p = Path(path)
    if not p.exists():
        return {}
    data = load_json(p)
    return {k: v for k, v in data.get("cameras", {}).items()
            if isinstance(v, dict) and v.get("matrix_4x4_row_major")}


def resolve_mount(camera: str, path: Path = TRANSFORM_CONFIG) -> Dict[str, Any]:
    """Effective mount for one camera, calibrated values preferred.

    The returned dict always states which source won, so a caller can
    refuse to proceed on a nominal value if it needs a real one."""
    if camera not in CAMERA_MOUNTS:
        raise KeyError(f"unknown camera '{camera}'. "
                       f"Known: {sorted(CAMERA_MOUNTS)}")
    entry = dict(CAMERA_MOUNTS[camera])
    override = load_overrides(path).get(camera)
    if override is not None:
        entry["T_parent_optical"] = np.asarray(
            override["matrix_4x4_row_major"], dtype=float)
        entry["provenance"] = override.get("provenance", "calibrated")
        entry["validated"] = bool(override.get("validated", True))
        entry["rigid_to"] = override.get("rigid_to", entry["rigid_to"])
        entry["note"] = override.get("note", "from ee_camera_transforms.json")
        entry["source"] = str(path)
    else:
        entry["source"] = "built-in nominal (camera_mount.py)"
    return entry


def T_world_camera(frames, q_urdf, camera: str,
                   path: Path = TRANSFORM_CONFIG) -> np.ndarray:
    """Compose  T_world_camera = T_world_parent @ T_parent_camera.

    `frames` is a kinematics.RobotFrames, `q_urdf` a URDF-coordinate joint
    vector.  Raises when the mount transform is unknown -- this pipeline
    does not invent extrinsics."""
    mount = resolve_mount(camera, path)
    T_parent_cam = mount["T_parent_optical"]
    if T_parent_cam is None:
        raise ValueError(
            f"camera '{camera}' has no mount transform "
            f"(provenance='{mount['provenance']}'). Calibrate it and record "
            f"the result in {path}, or use a camera that has one.")
    T_world_parent = frames.link_pose(q_urdf, mount["rigid_to"])
    return T_world_parent @ T_parent_cam


## ------------------------------------------------------------------ ##
## Report
## ------------------------------------------------------------------ ##

def _describe(camera: str, entry: Dict[str, Any]) -> str:
    lines = [
        "=" * 74,
        f"CAMERA: {camera}",
        "=" * 74,
        f"  hardware       : {entry['hardware']}",
        f"  serial         : {entry['serial'] or '(not addressable by serial)'}",
        f"  mounted on arm : {entry['arm'] or '(static -- not on an arm)'}",
        f"  rigid to frame : {entry['rigid_to']}",
        f"  provenance     : {entry['provenance'].upper()}"
        f"   validated={entry['validated']}",
        f"  source         : {entry['source']}",
    ]
    T = entry["T_parent_optical"]
    parent = entry["rigid_to"]
    if T is None:
        lines += [
            "",
            f"  T_{parent}_{camera}: *** NOT AVAILABLE ***",
            "      No transform exists in the repository for this camera.",
        ]
    else:
        lines += [
            "",
            f"  T_{parent}_{camera}   (maps camera-frame points into "
            f"'{parent}';",
            f"      equivalently the camera's pose expressed in '{parent}')",
            format_T(T, indent="      "),
            "",
            "      camera optical axes expressed in "
            f"'{parent}' (OpenCV convention):",
            f"        +x image-right   : {np.round(T[:3, 0], 5).tolist()}",
            f"        +y image-down    : {np.round(T[:3, 1], 5).tolist()}",
            f"        +z view direction: {np.round(T[:3, 2], 5).tolist()}",
        ]
        if entry.get("mesh_origin_T") is not None:
            M = entry["mesh_origin_T"]
            dt = np.linalg.norm(M[:3, 3] - T[:3, 3]) * 1e3
            da = rotation_angle_deg(M[:3, :3].T @ T[:3, :3])
            lines += [
                "",
                "      cross-check vs the giava.urdf d405 mesh origin:",
                f"        translation difference : {dt:.4f} mm",
                f"        rotation difference    : {da:.3f} deg",
                "        (a 180 deg rotation is EXPECTED -- the STL's "
                "authoring frame",
                "         is not the optical frame; the translation "
                "agreement is the",
                "         meaningful check)",
            ]
    if entry.get("note"):
        lines += ["", "  NOTE: " + _wrap(entry["note"], 68, "        ")]
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> str:
    import textwrap
    return textwrap.fill(text, width=width,
                         subsequent_indent=indent).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=None,
                    help="report a single camera (default: all)")
    ap.add_argument("--config", default=str(TRANSFORM_CONFIG),
                    help="calibrated-override file to consult")
    ap.add_argument("--world", action="store_true",
                    help="also compose T_world_camera using forward "
                         "kinematics at the chosen configuration")
    ap.add_argument("--pose", default=None,
                    help="named pose from arm_config.POSES for --world "
                         "(default: URDF home configuration)")
    ap.add_argument("--json", default=None,
                    help="also write the full report to this JSON file")
    args = ap.parse_args()

    cams = [args.camera] if args.camera else list(CAMERA_MOUNTS)
    cfg = Path(args.config)

    print()
    print("#" * 74)
    print("#  GIAVA camera mounting transforms  (PHASE 3)")
    print("#" * 74)
    print()
    print("  Direction convention: T_a_b maps points from frame b into "
          "frame a,")
    print("  and is the pose of b expressed in a.  "
          "T_world_camera = T_world_ee @ T_ee_camera.")
    print()
    print(f"  Override file: {cfg}"
          f"  ({'present' if cfg.exists() else 'ABSENT -- all values nominal'})")
    print()

    resolved = {}
    for cam in cams:
        entry = resolve_mount(cam, cfg)
        resolved[cam] = entry
        print(_describe(cam, entry))
        print()

    if args.world:
        _report_world(resolved, args.pose)

    n_cal = sum(1 for e in resolved.values() if e["provenance"] == "calibrated")
    print("-" * 74)
    print(f"  {n_cal} of {len(resolved)} cameras have a CALIBRATED mount "
          f"transform.")
    if n_cal < len(resolved):
        print("  The rest are nominal or missing and must not be used for "
              "metric work.")
    print("-" * 74)
    print()

    if args.json:
        _write_report(resolved, Path(args.json))


def _report_world(resolved: Dict[str, Any], pose: Optional[str]) -> None:
    from kinematics import RobotFrames

    frames = RobotFrames()
    if pose:
        q_driver = frames.q_from_named_pose(pose)
        bridge = _bridge(frames)
        q_urdf = bridge.to_urdf(q_driver)
        label = f"named pose '{pose}' (driver coords -> URDF)"
    else:
        q_urdf = frames.home_q()
        label = ("pyroki's default configuration for giava.urdf "
                 "(NOT all zeros)")

    print("=" * 74)
    print(f"T_world_camera at {label}")
    print("=" * 74)
    for cam, entry in resolved.items():
        if entry["T_parent_optical"] is None:
            print(f"\n  {cam}: skipped -- no mount transform")
            continue
        T = frames.link_pose(q_urdf, entry["rigid_to"]) @ entry["T_parent_optical"]
        print(f"\n  {cam}  (= T_world_{entry['rigid_to']} @ "
              f"T_{entry['rigid_to']}_{cam})")
        print(format_T(T, indent="      "))
    print()


def _bridge(frames):
    from kinematics import JointFrameBridge
    return JointFrameBridge(frames.robot)


def _write_report(resolved: Dict[str, Any], path: Path) -> None:
    from common import provenance, save_json

    out = {
        "metadata": provenance("camera_mount_report"),
        "convention": (
            "T_a_b maps points from frame b into frame a; equivalently it is "
            "the pose of b expressed in a. T_world_camera = T_world_ee @ "
            "T_ee_camera."
        ),
        "optical_frame_convention": "OpenCV: +x right, +y down, +z forward",
        "cameras": {},
    }
    for cam, e in resolved.items():
        rec = {
            "hardware": e["hardware"],
            "serial": e["serial"],
            "arm": e["arm"],
            "rigid_to": e["rigid_to"],
            "provenance": e["provenance"],
            "validated": e["validated"],
            "source": e["source"],
            "note": e.get("note"),
        }
        if e["T_parent_optical"] is not None:
            rec["transform"] = T_to_dict(e["T_parent_optical"],
                                         e["rigid_to"], cam)
        else:
            rec["transform"] = None
        out["cameras"][cam] = rec
    save_json(out, path)
    print(f"  report written to {path}")


if __name__ == "__main__":
    main()
