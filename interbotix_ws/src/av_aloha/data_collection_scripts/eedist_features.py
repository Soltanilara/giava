"""Metric gripper->object vector, computed live during a rollout.

The training column (build_eedist_dataset.py) is

    [ee_to_obj_dx, ee_to_obj_dy, ee_to_obj_dz, obj_found]   metres, base frame

built offline as `obj_xyz - ee_xyz`, where `obj_xyz` is the flower's top_scene
centroid pushed through the empirical pixel->grasp-pose fit
(assets/flower_px2ee.json) and `ee_xyz` is the recorded
`observation.ee_pose.right` -- forward kinematics of the COMMANDED joints.

This module reproduces that at 50 Hz.  Both halves must match how the labels
were made or the policy meets a different feature at test time than it trained
on: the same fit, and FK of the same commanded configuration (not the measured
one, which lags it).

The fit assumes the object is ON THE TABLE, so while the piece is held the
vector reads "displacement since pickup" rather than "distance to object" --
`obj_found` is in the vector so the policy can tell the regimes apart, and
hold-last keeps the last seen position when the top camera loses it.
"""

from __future__ import annotations

import json

import numpy as np

import scene_features

from paths import ASSETS_DIR

FEATURE_NAMES = ["ee_to_obj_dx", "ee_to_obj_dy", "ee_to_obj_dz", "obj_found"]
FEATURE_DIM = len(FEATURE_NAMES)

## Matches build_eedist_dataset: a miss on the first frame of an episode has no
## previous vector to hold, and zero is the honest reading for "no offset known".
MISS_VECTOR = [0.0, 0.0, 0.0, 0.0]

NEEDS_EE = True          # rollout_policy.env_vector must supply ee_xyz
CAMERAS = ("top_scene",)


class Extractor:
    """Same shape as scene_features.Extractor, except __call__ also takes the
    end-effector position for this tick."""

    FEATURE_NAMES = FEATURE_NAMES
    FEATURE_DIM = FEATURE_DIM
    NEEDS_EE = True
    target = None

    def __init__(self):
        fit = json.loads((ASSETS_DIR / "flower_px2ee.json").read_text())
        self.W = np.asarray(fit["W_affine"], dtype=np.float64)
        self.n_fit = fit.get("n_episodes")
        self.last = None
        self.last_obj = None

    def reset(self):
        self.last = None
        self.last_obj = None

    def px_to_xyz(self, uv):
        return np.array([uv[0], uv[1], 1.0]) @ self.W

    def __call__(self, top_rgb, ee_xyz):
        cx, cy, _area, found = scene_features.detect_object(top_rgb)
        if found:
            obj = self.px_to_xyz((cx, cy))
            self.last_obj = obj
        elif self.last_obj is not None:
            ## HOLD THE OBJECT, NOT THE VECTOR.  build_eedist_dataset holds the
            ## top_scene centroid (the env-state column's own hold-last), so
            ## obj_xyz stays put while the gripper moves and the vector keeps
            ## shrinking as the arm approaches.  Holding the VECTOR instead
            ## would make the object follow the gripper -- a 7 cm disagreement
            ## with the training column on occluded frames, measured.
            obj = self.last_obj
        else:
            v = np.asarray(MISS_VECTOR, dtype=np.float32)
            self.last = v
            return v
        d = obj - np.asarray(ee_xyz, dtype=float)
        v = np.array([d[0], d[1], d[2], 1.0 if found else 0.0], dtype=np.float32)
        self.last = v
        return v

    def detect(self, rgb):
        """Object tracking for --score, which feeds top_scene frames."""
        return scene_features.detect_object(rgb)

    def targets_px(self):
        return {"flower": scene_features.FLOWER_TARGET_PX,
                "oval": scene_features.OVAL_TARGET_PX}
