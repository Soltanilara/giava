"""The object-mask rule, shared by the offline builder and the live rollout.

This is deliberately its own module.  `build_mask_dataset.py` renders the mask
into a dataset ahead of training; `rollout_policy.py` has to produce the SAME
mask online, or the policy sees a different input at inference than it was
trained on.  One rule, one place -- the builder can then live with the other
offline tooling without the runtime depending on it.
"""

from __future__ import annotations

import cv2
import numpy as np

MASK_KEY = "observation.images.obj_mask"


def mask_for(rgb, camera, dilate):
    """Binary object mask as uint8 0/255, matching the camera's detector.

    top_scene uses scene_features' rule (static printed targets subtracted,
    workspace ROI applied); right_wrist uses wrist_features' rule (full frame,
    no static mask -- the camera moves, so a static mask is meaningless).
    """
    if camera == "top_scene":
        import scene_features
        m = scene_features._blue_mask(rgb)
        m = m * (1 - scene_features.static_blue_mask())
        x0, y0, x1, y1 = scene_features.WORKSPACE_ROI
        roi = np.zeros_like(m)
        roi[y0:y1, x0:x1] = 1
        m = m * roi
    else:
        import wrist_features
        m = wrist_features._blue_mask(rgb)

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ## Keep only the largest blob: the detector's own "this is the object"
    ## decision, so the mask and the env-state vector cannot disagree.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]))
        m = (lab == big).astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return (m * 255).astype(np.uint8)
