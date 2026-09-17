"""GIAVA reconstruction -- stage two: pixels to metres.

`calibration/` establishes the front half of the chain (robot pose, camera
pose, synchronised images, extrinsics) and writes JSON.  This package
consumes that JSON and does the geometry: epipolar correspondence,
triangulation, plane intersection, and metric measurement, each with its
own error bars.

Nothing here writes or repairs a calibration.  A missing extrinsic raises,
naming the command that would produce it.

    from reconstruction import load_rig

    rig = load_rig(["top_scene", "low_scene"])       # world frame if anchored
    curve = rig.epipolar_curve("top_scene", (312, 205), "low_scene")
    tri   = rig.triangulate({"top_scene": (312, 205),
                             "low_scene": (401, 88)})
    print(tri.describe(rig.ref_frame))
"""

from camera import (PinholeCamera, from_intrinsics_dict, load_charuco,
                    load_factory, load_intrinsics, normalize_model)
from rig import (SCENE_CAMERAS, load_rig, load_stereo_rig, load_world_rig,
                 rig_status)
from stereo import (StereoRig, Triangulation, distance, distance_with_error)

__all__ = [
    "PinholeCamera", "StereoRig", "Triangulation", "SCENE_CAMERAS",
    "load_rig", "load_stereo_rig", "load_world_rig", "load_intrinsics",
    "load_factory", "load_charuco", "from_intrinsics_dict",
    "normalize_model", "rig_status", "distance", "distance_with_error",
]
