"""Gripper-centric object features from the right_wrist camera.

WHY THIS EXISTS -- read before changing the layout.

`scene_features.py` computes everything from `top_scene`, a top-down camera.
Measured over all 113,343 frames of transfer_flower_merged, that 12-d vector
has numerical **rank 4**: `flower_cx/cy` and `oval_cx/cy` are exactly constant
(the targets never move) and the four `obj_to_*` offsets are the object
centroid plus a constant.  Worse, the persistent rollout failure is closing the
gripper ABOVE or BELOW the object -- a height error -- and a top-down camera
encodes table x-y with no height at all.  There is also no gripper position in
that vector, so the policy is told where the object is but not where its own
hand is, and cannot form the error signal that would close the loop.

This module fixes both problems by changing the coordinate frame rather than
adding more numbers.  In the wrist camera the gripper IS the origin, so:

  * the object's offset from the grasp aim-point is the alignment error,
    directly, with no kinematics to invert;
  * apparent blob area is a monotone proxy for distance -- measured 501 px at
    the start of an approach rising to ~12,500 px at grasp -- which is exactly
    the height signal top_scene cannot carry.

GRASP_AIM_PX and GRASP_AREA below were measured at the first gripper-close
frame of 129 episodes of transfer_flower_merged.  They are strikingly tight:
cx IQR +/-9 px, cy IQR +/-6 px, area IQR +/-7%, across a placement range 4.7x
wider than the original collection grid.  That tightness is the point -- the
grasp is a FIXED POINT in wrist coordinates no matter where the object sits on
the table, which is the property that should generalize.

Every feature here is referenced to that fixed point, so a value of 0 means
"aligned / at grasp distance".  Nothing in the vector is constant and nothing
is an affine copy of anything else -- run `--rank` on the built dataset to
confirm before training (see build_envstate_dataset.py).

Values are hand-scaled into roughly [-1, 1] ON PURPOSE, exactly as in
scene_features.py: ACT's `normalization_mapping` has no ENV entry, so
`observation.environment_state` reaches the network UNNORMALIZED.
"""

from __future__ import annotations

import cv2
import numpy as np

import scene_features

## Feature layout, in order.  Anything reading the vector by index must use
## these names.
FEATURE_NAMES = [
    ## --- gripper-centric, from right_wrist.  This is the new information. ---
    "w_err_x",      # object centroid - grasp aim, x.  0 => aligned
    "w_err_y",      # object centroid - grasp aim, y.  0 => aligned
    "obj_found",    # 1.0 if the wrist camera sees it this frame
    "w_range",      # sqrt(area / grasp_area) - 1.  0 => at grasp distance,
                    # negative => too far.  This is the height signal.
    "w_cos2t",      # principal-axis orientation as a double angle, so a blob
    "w_sin2t",      # and its 180-degree rotation map to the same value
    ## --- global context, from top_scene.  Where on the table we are. ---
    "t_obj_cx",
    "t_obj_cy",
    "t_obj_found",
]
FEATURE_DIM = len(FEATURE_NAMES)

## Measured at the first gripper-close frame of 129 episodes of
## transfer_flower_merged/20260916_centre_plus_rim (2026-09-18).
## REMEASURE with tools/measure_grasp_aim.py if the wrist camera is remounted
## or the gripper fingers change.
GRASP_AIM_PX = (344.4, 305.0)
GRASP_AREA = 12462.0

## Blue in OpenCV HSV (H is 0..179).  Same hue band as scene_features, but the
## saturation/value floor is deliberately NOT reused: the object fills a large
## part of the wrist frame and is lit differently up close.  Validated at 100%
## detection with zero missed frames over a full 812-frame episode.
HUE_LO, HUE_HI = 95, 135
SAT_MIN, VAL_MIN = 140, 70
AREA_MIN = 80


def _blue_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((H >= HUE_LO) & (H <= HUE_HI) &
         (S >= SAT_MIN) & (V >= VAL_MIN)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def detect_object(rgb):
    """(cx_px, cy_px, area_px, cos2t, sin2t, found) for the blue object.

    Largest blue blob in the FULL wrist frame -- no ROI and no static-blue
    subtraction, both of which are top_scene concepts.  The wrist camera
    moves, so a static mask is meaningless; and the printed blue targets do
    appear in frame near the destination, but the held object is centimetres
    from the lens and dominates on area there.
    """
    m = _blue_mask(rgb)
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(m)
    best_i, best_a = -1, 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= AREA_MIN and a > best_a:
            best_i, best_a = i, a
    if best_i < 0:
        return 0.0, 0.0, 0, 0.0, 0.0, False

    cx, cy = float(cents[best_i][0]), float(cents[best_i][1])

    ## Principal axis from central moments of that blob only.
    blob = (_lab == best_i).astype(np.uint8)
    mo = cv2.moments(blob, binaryImage=True)
    if mo["m00"] > 0:
        mu20 = mo["mu20"] / mo["m00"]
        mu02 = mo["mu02"] / mo["m00"]
        mu11 = mo["mu11"] / mo["m00"]
        ## Double angle directly from the second moments: atan2(2*mu11,
        ## mu20-mu02) already IS 2*theta, so cos/sin of it are the
        ## 180-degree-symmetric orientation without ever forming theta.
        two_t = np.arctan2(2.0 * mu11, mu20 - mu02)
        c2, s2 = float(np.cos(two_t)), float(np.sin(two_t))
    else:
        c2, s2 = 0.0, 0.0
    return cx, cy, best_a, c2, s2, True


def wrist_features(wrist_rgb, top_rgb, last=None):
    """FEATURE_DIM float32 vector for one (right_wrist, top_scene) pair.

    `last` is the previous frame's vector; on a miss the gripper-centric
    dimensions are HELD rather than snapped to zero, because zero here means
    "perfectly aligned at grasp distance" and teaching the network that
    occlusion means a perfect grasp is far worse than no feature at all.
    """
    h, w = wrist_rgb.shape[:2]
    cx, cy, area, c2, s2, found = detect_object(wrist_rgb)

    if found:
        ex = (cx - GRASP_AIM_PX[0]) / (w / 2.0)
        ey = (cy - GRASP_AIM_PX[1]) / (h / 2.0)
        rng = float(np.sqrt(max(area, 1) / GRASP_AREA) - 1.0)
    elif last is not None:
        ex, ey, rng, c2, s2 = (float(last[0]), float(last[1]), float(last[3]),
                               float(last[4]), float(last[5]))
    else:
        ex, ey, rng, c2, s2 = 0.0, 0.0, -1.0, 0.0, 0.0

    ## Global context: where the object is on the table, from the top camera.
    ## Only the three non-degenerate dimensions of the old vector are kept --
    ## the static targets and the affine offsets are deliberately dropped.
    tcx, tcy, tarea, tfound = scene_features.detect_object(top_rgb)
    th, tw = top_rgb.shape[:2]
    if tfound:
        tx, ty = scene_features._norm_xy(tcx, tcy, tw, th)
    elif last is not None:
        tx, ty = float(last[6]), float(last[7])
    else:
        tx, ty = 0.0, 0.0

    return np.array([ex, ey, 1.0 if found else 0.0, rng, c2, s2,
                     tx, ty, 1.0 if tfound else 0.0], dtype=np.float32)


class Extractor:
    """Same interface as scene_features.Extractor / shape_sorter.Extractor,
    except __call__ takes TWO frames.  build_envstate_dataset.py dispatches on
    the module-level NEEDS_TWO_CAMERAS flag below."""

    FEATURE_NAMES = FEATURE_NAMES
    FEATURE_DIM = FEATURE_DIM
    target = None
    ## On the CLASS, because callers hold an instance and test
    ## getattr(extractor, "NEEDS_TWO_CAMERAS"); the module-level copy below
    ## is for build_envstate_dataset, which holds the module.
    NEEDS_TWO_CAMERAS = True
    CAMERAS = ("right_wrist", "top_scene")

    def __init__(self):
        self.last = None

    def reset(self):
        self.last = None

    def __call__(self, wrist_rgb, top_rgb):
        v = wrist_features(wrist_rgb, top_rgb, self.last)
        self.last = v
        return v

    def detect(self, rgb):
        """Object tracking for --score.  rollout_policy.object_track feeds
        this TOP_SCENE frames (the scoring view is fixed regardless of what
        the policy consumes), so delegate to the top_scene detector rather
        than running the wrist rule on the wrong camera."""
        return scene_features.detect_object(rgb)

    def targets_px(self):
        return {"flower": scene_features.FLOWER_TARGET_PX,
                "oval": scene_features.OVAL_TARGET_PX}


## Read by build_envstate_dataset.py and rollout_policy.py to decide how many
## camera streams to feed the extractor.
NEEDS_TWO_CAMERAS = True
CAMERAS = ("right_wrist", "top_scene")


## Value of the vector when nothing is detected and there is no previous frame
## to hold (i.e. the first frame of an episode).  NOT all zeros: w_range = 0
## means "at grasp distance", so a miss encoded as zeros would read as a
## perfect grasp.  build_envstate_dataset.py reads this.
MISS_VECTOR = [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
