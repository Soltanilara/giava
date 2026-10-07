"""Object-centric scene features from the top_scene camera.

ONE extractor, imported by BOTH the offline dataset builder and the rollout
script.  The 271 study computed features in the trainer and again, separately,
in the rollout script; two copies of a heuristic drift apart and the policy
then sees different numbers at test time than it trained on.  Import this, do
not re-implement it.

The features are hand-scaled into roughly [-1, 1] ON PURPOSE.  ACT's
`normalization_mapping` has no ENV entry, so `observation.environment_state`
reaches the network UNNORMALIZED -- whatever is written here is what the
linear projection sees.  Values must therefore already sit on the same scale
as the mean/std-normalized state and image tokens, or the env token is
effectively muted.

Scene layout (transfer_flower, top_scene 640x480, blinds fixed):
  - blue flower BLOCK on the wood table  -- the object, moves
  - blue clover OUTLINE on a white sheet -- flower target, static
  - blue OVAL on the same sheet          -- oval target, static
  - blue distractors OUTSIDE the workspace ROI: the "SHALL" box (right),
    a device at bottom-centre, shelf boxes along the top edge.  The ROI is
    what separates them from the object; do not widen it without re-running
    validate_scene_features.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

## Feature layout, in order.  Anything reading the vector by index must use
## these names.
FEATURE_NAMES = [
    "obj_cx", "obj_cy", "obj_found", "obj_size",
    "flower_cx", "flower_cy", "oval_cx", "oval_cy",
    "obj_to_flower_dx", "obj_to_flower_dy",
    "obj_to_oval_dx", "obj_to_oval_dy",
]
FEATURE_DIM = len(FEATURE_NAMES)

## Workspace: the table region the object can occupy.  Measured from the
## Sept-2026 collection; excludes every large blue distractor.
WORKSPACE_ROI = (200, 150, 500, 300)          # x0, y0, x1, y1

## Blue in OpenCV HSV (H is 0..179).
HUE_LO, HUE_HI = 95, 135
OBJ_SAT_MIN, OBJ_VAL_MIN = 140, 70
OBJ_AREA_MIN = 80

## Static targets, measured from this dataset (see validate_scene_features.py).
## Pixel coords in the 640x480 top_scene frame.
FLOWER_TARGET_PX = (313.2, 208.4)
OVAL_TARGET_PX = (310.8, 240.8)

## The printed targets are STATIC, so they are removed by a precomputed
## background mask rather than by a position+area rule.  The rule version
## failed exactly when the block is placed ON the flower target: the two blobs
## merge and an area threshold cannot separate "printed target, partly
## shadowed" from "target plus block".  Subtracting the static pixels leaves
## the block's own pixels in both cases.
##
## Built by tools/build_static_blue.py: blue in >80% of sampled frames,
## dilated 3x3.  REBUILD IT if the sheet or the camera moves.
from paths import ASSETS_DIR  # noqa: E402

STATIC_MASK_PATH = ASSETS_DIR / "top_scene_static_blue.npy"
_STATIC = {"mask": None}


def static_blue_mask():
    if _STATIC["mask"] is None:
        if not STATIC_MASK_PATH.exists():
            raise FileNotFoundError(
                f"{STATIC_MASK_PATH} missing -- run tools/build_static_blue.py "
                "against the dataset this policy trains on.")
        _STATIC["mask"] = np.load(STATIC_MASK_PATH).astype(np.uint8)
    return _STATIC["mask"]


def _norm_xy(x, y, w, h):
    """Pixel -> [-1, 1], origin at image centre."""
    return (2.0 * x / w - 1.0, 2.0 * y / h - 1.0)


def _blue_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((H >= HUE_LO) & (H <= HUE_HI) &
         (S >= OBJ_SAT_MIN) & (V >= OBJ_VAL_MIN)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def detect_object(rgb):
    """(cx_px, cy_px, area_px, found) for the blue block, or found=False.

    Largest blue blob inside the workspace ROI once the static printed
    targets are subtracted.  Works whether the block is on the table or
    placed on a target.
    """
    x0, y0, x1, y1 = WORKSPACE_ROI
    mask = _blue_mask(rgb)
    mask = mask * (1 - static_blue_mask())
    roi = np.zeros_like(mask)
    roi[y0:y1, x0:x1] = 1
    mask = mask * roi
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < OBJ_AREA_MIN:
            continue
        if best is None or area > best[2]:
            best = (float(cents[i][0]), float(cents[i][1]), area)
    if best is None:
        return 0.0, 0.0, 0, False
    return best[0], best[1], best[2], True


def detect_object_outline(rgb, roi=None):
    """(contour Nx2 px, centroid, area, found) for the blue object.

    Same detection as detect_object -- identical mask, identical
    largest-blob rule -- but returns the blob's OUTLINE rather than only its
    centre.  A centroid says where the flower was; the outline says how it
    was lying, and rotation is what the rollout notes keep blaming for a
    failed grasp ("rotated 45 degrees", "on its side").

    `roi` overrides WORKSPACE_ROI for one call.  That exists because the ROI
    is a 300x150 window in a 640x480 frame -- 47% of the width and 31% of the
    height -- and a placement grid that spans the workspace will put objects
    outside it, where they are simply invisible. Pass a wider box to check
    coverage before trusting a session's worth of trials to the default.
    """
    x0, y0, x1, y1 = roi if roi is not None else WORKSPACE_ROI
    mask = _blue_mask(rgb)
    mask = mask * (1 - static_blue_mask())
    box = np.zeros_like(mask)
    box[y0:y1, x0:x1] = 1
    mask = mask * box
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= OBJ_AREA_MIN and a > best_area:
            best, best_area = c, a
    if best is None:
        return None, (0.0, 0.0), 0, False
    m = cv2.moments(best)
    cx = m["m10"] / m["m00"] if m["m00"] else float(best[:, 0, 0].mean())
    cy = m["m01"] / m["m00"] if m["m00"] else float(best[:, 0, 1].mean())
    return best.reshape(-1, 2), (float(cx), float(cy)), int(best_area), True


def scene_features(rgb, last=None):
    """FEATURE_DIM float32 vector for one top_scene frame.

    `last` is the previous frame's vector; when the object is occluded (the
    gripper closes over it, which is precisely when the policy is deciding
    whether it has a grasp) the position is HELD and obj_found drops to 0
    rather than jumping to the origin.  A coordinate that teleports to (0,0)
    on every grasp teaches the network that occlusion means "object at image
    centre", which is worse than no feature at all.
    """
    h, w = rgb.shape[:2]
    cx, cy, area, found = detect_object(rgb)

    if found:
        ox, oy = _norm_xy(cx, cy, w, h)
        size = float(np.sqrt(area / float(w * h)) * 10.0)
    elif last is not None:
        ox, oy, size = float(last[0]), float(last[1]), float(last[3])
    else:
        ox, oy, size = 0.0, 0.0, 0.0

    fx, fy = _norm_xy(*FLOWER_TARGET_PX, w, h)
    vx, vy = _norm_xy(*OVAL_TARGET_PX, w, h)

    return np.array([
        ox, oy, 1.0 if found else 0.0, size,
        fx, fy, vx, vy,
        fx - ox, fy - oy,
        vx - ox, vy - oy,
    ], dtype=np.float32)


class Extractor:
    """Same interface as shape_sorter.Extractor, so build_envstate_dataset.py
    and rollout_policy.py can swap scenes without knowing which one they
    hold.  Carries the hold-last state for one episode / one rollout."""

    FEATURE_NAMES = FEATURE_NAMES
    FEATURE_DIM = FEATURE_DIM
    target = None

    def __init__(self):
        self.last = None

    def reset(self):
        self.last = None

    def __call__(self, rgb):
        v = scene_features(rgb, self.last)
        self.last = v
        return v

    def detect(self, rgb):
        return detect_object(rgb)

    def targets_px(self):
        """name -> (x, y) for auto metrics; names are the keys the flower-era
        rollout_scores.jsonl records already carry."""
        return {"flower": FLOWER_TARGET_PX, "oval": OVAL_TARGET_PX}


## No-detection fallback for the first frame of an episode; see
## build_envstate_dataset.py.  Position and size go to zero, which for
## this layout means "image centre, no extent" -- the historical
## behaviour, preserved exactly.
MISS_VECTOR = [0.0, 0.0, 0.0, 0.0,
               _norm_xy(*FLOWER_TARGET_PX, 640, 480)[0],
               _norm_xy(*FLOWER_TARGET_PX, 640, 480)[1],
               _norm_xy(*OVAL_TARGET_PX, 640, 480)[0],
               _norm_xy(*OVAL_TARGET_PX, 640, 480)[1],
               _norm_xy(*FLOWER_TARGET_PX, 640, 480)[0],
               _norm_xy(*FLOWER_TARGET_PX, 640, 480)[1],
               _norm_xy(*OVAL_TARGET_PX, 640, 480)[0],
               _norm_xy(*OVAL_TARGET_PX, 640, 480)[1]]
