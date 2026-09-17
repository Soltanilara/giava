"""Starting-scene snapshots and the ghosted alignment preview.

ONE implementation, imported by rollout_policy.py (evaluation) and
data_collection.py (DAgger), so an episode recorded under --replicate in one
is reproducible in the other.  A paired comparison is only paired if both
sides put the object in the same place; at real-robot trial counts placement
noise is larger than the effect being measured, so this is not cosmetic.

    save_snapshot(dir, ep, {camera: RGB})      one PNG per camera
    load_snapshot(dir, ep, cameras)            {camera: RGB} or {}
    alignment_view(live, ref, cameras)         tiled BGR image for cv2.imshow
    list_episodes(dir)                         sorted episode indices present
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np


def save_snapshot(snap_dir, ep, frames, suffix=""):
    """One PNG per camera of the scene as the episode STARTED (suffix "") or
    ENDED (suffix "_end")."""
    import cv2

    snap_dir = Path(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    for cam, img in (frames or {}).items():
        if img is not None:
            cv2.imwrite(str(snap_dir / f"ep{ep:02d}_{cam}{suffix}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_snapshot(snap_dir, ep, cameras):
    """{camera: RGB image} for episode `ep` of an earlier run, or {}."""
    import cv2

    out = {}
    for cam in cameras:
        f = Path(snap_dir) / f"ep{ep:02d}_{cam}.png"
        if f.exists():
            img = cv2.imread(str(f))
            if img is not None:
                out[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return out


def list_episodes(snap_dir):
    """Episode indices that have at least one snapshot in `snap_dir`."""
    eps = set()
    for p in glob.glob(str(Path(snap_dir) / "ep*_*.png")):
        m = re.match(r"ep(\d+)_", os.path.basename(p))
        if m:
            eps.add(int(m.group(1)))
    return sorted(eps)


def alignment_view(live, ref, cameras, scale=0.6, banner=None):
    """Tiled live view, each camera GHOSTED against its reference frame.

    A 50/50 blend rather than a difference image on purpose: a difference goes
    black when it matches, which gives nothing to aim at, while a blend shows
    the object doubled until it is right and single when it is."""
    import cv2

    tiles = []
    for cam in cameras:
        img = live.get(cam)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        r = ref.get(cam)
        label = cam
        if r is not None:
            r = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
            if r.shape != img.shape:
                r = cv2.resize(r, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 0.5, r, 0.5, 0)
            label = f"{cam}  [ghosted vs reference]"
        img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for t in tiles]
    out = np.hstack(tiles)
    if banner:
        cv2.putText(out, banner, (6, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, banner, (6, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
