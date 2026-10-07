"""Shape-sorter task: pieces, task strings, per-class detector, env features.

ONE module, imported by data_collection.py (task string per episode),
build_envstate_dataset.py (offline features), rollout_policy.py (online
features + scoring) and measure_targets.py (hole calibration).  Same rule as
scene_features.py: never re-implement any of this elsewhere, or train and
deploy drift apart.

Task: three pieces on the table at random positions, one is the TARGET for
this episode; pick it up and insert it into the matching hole in the sorter
box.  The target is fixed per episode and named in the episode's task string,
which is how the offline builder and the rollout know which piece to track.

COLOR == SHAPE in the object set, on purpose: the detector classifies by hue,
so the pieces on the table during COLLECTION must have one color per shape.
The three top-face cutouts of the lab's box are a red triangle, a green cube
and a blue flower, and CLASSES mirrors that.  Same-color/different-shape pieces
from the set (a green cylinder next to the green cube) are invisible to the
detector as distinct objects -- use them only as rollout distractors, where
"does the policy still take the right SHAPE" is the question being asked.

Feature vector (FEATURE_DIM, hand-scaled to ~[-1, 1] because ACT's ENV input
is NOT normalized -- see scene_features.py):

    obj_cx, obj_cy, obj_found, obj_size     -- the target piece (held-last on occlusion)
    hole_cx, hole_cy                        -- the matching hole (static, from calibration)
    obj_to_hole_dx, obj_to_hole_dy

Deliberately NO class one-hot.  The vector is pure task geometry -- "this
thing goes there" -- so a held-out shape presents the policy with a vector
from the SAME distribution it trained on, and zero-shot transfer is a fair
test of whether the policy learned "carry the object to the target" rather
than a per-shape lookup.  The shape itself is visible in the images.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

## name -> color word (for the task string), hole name, HSV hue ranges
## (OpenCV H in 0..179; red wraps, hence two ranges).  Dict order is the
## canonical class order everywhere.
CLASSES = {
    "cube":     {"color": "green", "hole": "square",   "hue": [(45, 90)]},
    "triangle": {"color": "red",   "hole": "triangle", "hue": [(0, 8), (168, 179)]},
    "flower":   {"color": "blue",  "hole": "flower",   "hue": [(95, 135)]},
}
CLASS_NAMES = list(CLASSES)

SAT_MIN, VAL_MIN = 120, 60
OBJ_AREA_MIN = 80

TASK_NAME = "shape_sorter"

FEATURE_NAMES = [
    "obj_cx", "obj_cy", "obj_found", "obj_size",
    "hole_cx", "hole_cy",
    "obj_to_hole_dx", "obj_to_hole_dy",
]
FEATURE_DIM = len(FEATURE_NAMES)

## Written by measure_targets.py from a top_scene frame of THIS rig setup:
##   {"image_size": [w, h], "roi": [x0, y0, x1, y1],
##    "holes": {"square": [x, y], "triangle": [x, y], ...}}
## Re-measure if the box or the camera moves.
from paths import ASSETS_DIR  # noqa: E402

TARGETS_PATH = ASSETS_DIR / "shape_sorter_targets.json"
_TARGETS = {"data": None}


# ---------------------------------------------------------------------------
# Task strings
# ---------------------------------------------------------------------------

def task_string(target: str) -> str:
    """The per-episode language instruction, e.g.
    'insert the red cube into the square hole'."""
    c = CLASSES[target]
    return f"insert the {c['color']} {target} into the {c['hole']} hole"


_TASK_RE = re.compile(r"\binsert the (\w+) (" + "|".join(CLASS_NAMES) + r")\b")


def target_from_task(task: str):
    """Inverse of task_string(); None if the string is not one of ours."""
    m = _TASK_RE.search(task or "")
    return m.group(2) if m else None


# ---------------------------------------------------------------------------
# Calibrated targets
# ---------------------------------------------------------------------------

def load_targets():
    if _TARGETS["data"] is None:
        if not TARGETS_PATH.exists():
            raise FileNotFoundError(
                f"{TARGETS_PATH} missing -- run measure_targets.py on a "
                "top_scene frame of this setup to click the ROI and the "
                "hole centres.")
        d = json.loads(TARGETS_PATH.read_text())
        missing = [c["hole"] for c in CLASSES.values()
                   if c["hole"] not in d.get("holes", {})]
        if missing:
            raise KeyError(f"{TARGETS_PATH} has no hole(s) {missing}; "
                           "re-run measure_targets.py")
        _TARGETS["data"] = d
    return _TARGETS["data"]


def hole_px(target: str):
    """(x, y) pixel centre of the hole the `target` class goes into."""
    return tuple(load_targets()["holes"][CLASSES[target]["hole"]])


def workspace_roi():
    return tuple(load_targets()["roi"])


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def class_mask(rgb, cls: str):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(H.shape, dtype=bool)
    for lo, hi in CLASSES[cls]["hue"]:
        m |= (H >= lo) & (H <= hi)
    m &= (S >= SAT_MIN) & (V >= VAL_MIN)
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def detect_class(rgb, cls: str, roi=None):
    """(cx_px, cy_px, area_px, found) for the largest `cls`-colored blob
    inside the workspace ROI."""
    x0, y0, x1, y1 = roi if roi is not None else workspace_roi()
    mask = class_mask(rgb, cls)
    win = np.zeros_like(mask)
    win[y0:y1, x0:x1] = 1
    mask = mask * win

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


def detect_all(rgb, roi=None):
    """{class: (cx, cy, area, found)} for every class -- for tools/debugging."""
    return {c: detect_class(rgb, c, roi) for c in CLASS_NAMES}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _norm_xy(x, y, w, h):
    return (2.0 * x / w - 1.0, 2.0 * y / h - 1.0)


def features(rgb, target: str, last=None):
    """FEATURE_DIM float32 vector for one top_scene frame.

    `last` = previous frame's vector: on occlusion the position is HELD and
    obj_found drops to 0 (same reasoning as scene_features.scene_features --
    a coordinate that teleports to the origin on every grasp teaches the
    network that occlusion means 'object at image centre')."""
    h, w = rgb.shape[:2]
    cx, cy, area, found = detect_class(rgb, target)

    if found:
        ox, oy = _norm_xy(cx, cy, w, h)
        size = float(np.sqrt(area / float(w * h)) * 10.0)
    elif last is not None:
        ox, oy, size = float(last[0]), float(last[1]), float(last[3])
    else:
        ox, oy, size = 0.0, 0.0, 0.0

    hx, hy = _norm_xy(*hole_px(target), w, h)
    return np.array([
        ox, oy, 1.0 if found else 0.0, size,
        hx, hy,
        hx - ox, hy - oy,
    ], dtype=np.float32)


class Extractor:
    """The interface build_envstate_dataset.py and rollout_policy.py program
    against, so the flower scene and this one are swappable.  `target` is
    fixed for the extractor's lifetime (one episode / one rollout)."""

    FEATURE_NAMES = FEATURE_NAMES
    FEATURE_DIM = FEATURE_DIM

    def __init__(self, target: str):
        if target not in CLASSES:
            raise ValueError(f"unknown target {target!r}; one of {CLASS_NAMES}")
        self.target = target
        self.last = None

    def reset(self):
        self.last = None

    def __call__(self, rgb):
        v = features(rgb, self.target, self.last)
        self.last = v
        return v

    def detect(self, rgb):
        return detect_class(rgb, self.target)

    def targets_px(self):
        """name -> (x, y) for auto metrics: the hole this target goes into."""
        return {"target": hole_px(self.target)}


## "Which piece" WITHOUT "where": a one-hot of the target class and nothing
## else.  This is the fair baseline for a policy trained on several pieces at
## once -- with three pieces on the table a pixel-only policy has no way to
## know which one this episode is about, so plain pixels is ill-posed for
## pooled training, not a baseline.  Against the geometry extractor above it
## isolates what the centroid + hole vector add over task identity alone.
## (With a fixed box the hole coordinates are a function of the class, so the
## one-hot and hole_cx/cy carry the same information; the difference is the
## piece position.)  A held-out class has no one-hot: this condition can only
## be fine-tuned onto a new piece, never tested zero-shot.
ONEHOT_FEATURE_NAMES = [f"is_{c}" for c in CLASS_NAMES]
ONEHOT_FEATURE_DIM = len(ONEHOT_FEATURE_NAMES)


class OneHotExtractor:
    FEATURE_NAMES = ONEHOT_FEATURE_NAMES
    FEATURE_DIM = ONEHOT_FEATURE_DIM

    def __init__(self, target: str):
        if target not in CLASSES:
            raise ValueError(f"unknown target {target!r}; one of {CLASS_NAMES}")
        self.target = target
        self._v = np.zeros(ONEHOT_FEATURE_DIM, dtype=np.float32)
        self._v[CLASS_NAMES.index(target)] = 1.0

    def reset(self):
        pass

    def __call__(self, rgb):
        return self._v.copy()

    def detect(self, rgb):
        return detect_class(rgb, self.target)

    def targets_px(self):
        return {"target": hole_px(self.target)}
