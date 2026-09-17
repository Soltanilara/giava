"""Live ghost of top_scene against the June training frame, for placing the block.

    python place_block_view.py                 # window, ghosted live vs reference
    python place_block_view.py --snap out.png  # one frame to a file, no window

WHY A GHOST AND NOT A DIFFERENCE
================================
Same reason scene_snapshots.alignment_view uses one: a difference image goes
black when it matches, which gives you nothing to aim at.  A 50/50 blend shows
the block DOUBLED until it is right and SINGLE when it is.

WHAT IT CORRECTS FOR
====================
The June recorder applied digital_zoom(frame, zoom=1.8) before recording
(data_collection.py@44895c6 line 421).  A raw pyrealsense grab is therefore NOT
comparable with the training frames -- it shows far more of the room and the
table sits much smaller.  This applies the same 1.8 so live and reference are
in the same frame.

Note the old rollout scripts used zoom=1.6, not 1.8, which is a ~12% scale
error against what those policies were trained on.  1.8 is the number that
matches the data; see ZOOM below.

THE TARGET BOX
==============
Across the 71-episode transfer_flower training set, 55 episodes placed the
block inside x 484-501, y 210-228 -- a 16x18 px box, 2.5% of the frame.  That
box is drawn.  Land the block in it and the policy is in distribution; land it
elsewhere and you are testing generalisation the data never covered.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

## The zoom the RECORDER used for the June data.  Not 1.6 -- that is what the
## old rollout scripts used, and the mismatch was never intentional.
ZOOM = 1.6

## Measured from the training masks; see the module docstring.
BOX = (484, 210, 501, 228)
CENTRE = (493, 218)

REF = os.path.join(str(_giava_paths.SCRIPTS_DIR),
                   "reference_frames", "june_transfer_flower_top_scene.png")
SERIAL = "230322270396"          # top_scene, camera_manager.CAMERA_SERIALS


def digital_zoom(frame, zoom=ZOOM):
    """Centre crop by `zoom` and resize back -- the recorder's exact operation."""
    import cv2
    h, w = frame.shape[:2]
    nw, nh = int(w / zoom), int(h / zoom)
    x1, y1 = (w - nw) // 2, (h - nh) // 2
    return cv2.resize(frame[y1:y1 + nh, x1:x1 + nw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


def annotate(img):
    """Draw the in-distribution box and centre onto a BGR image."""
    import cv2
    x1, y1, x2, y2 = BOX
    ## Generous outer rectangle so the thin one stays visible against the
    ## ghosted background, which is busy by construction.
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 3)
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 255, 0), 1)
    cx, cy = CENTRE
    cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
    cv2.putText(img, "place block here (55/71 eps)", (x1 - 120, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "place block here (55/71 eps)", (x1 - 120, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snap", default=None,
                    help="write one annotated frame here and exit (no window)")
    ap.add_argument("--zoom", type=float, default=ZOOM)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="blend weight on the LIVE frame (1.0 = live only)")
    args = ap.parse_args()

    import cv2
    import pyrealsense2 as rs

    ref = cv2.imread(args.ref)
    if ref is None:
        raise SystemExit(f"could not read reference {args.ref}")

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(SERIAL)
    ## Must match camera_manager's recorded stream or the geometry differs.
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
    pipe.start(cfg)

    windowed = args.snap is None
    if windowed:
        print("[place] q or ESC to quit, s to save a snapshot, "
              "g to toggle the ghost")
    ghost = True
    try:
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            live = np.asanyarray(color.get_data())          # RGB
            live = digital_zoom(live, args.zoom)
            img = cv2.cvtColor(live, cv2.COLOR_RGB2BGR)

            if ghost:
                r = ref if ref.shape == img.shape else cv2.resize(
                    ref, (img.shape[1], img.shape[0]))
                img = cv2.addWeighted(img, args.alpha, r, 1.0 - args.alpha, 0)
            img = annotate(img)
            label = (f"top_scene  zoom {args.zoom}  "
                     + ("[ghosted vs June reference]" if ghost else "[live only]"))
            for col, th in (((0, 0, 0), 3), ((0, 255, 255), 1)):
                cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, th, cv2.LINE_AA)

            if not windowed:
                cv2.imwrite(args.snap, img)
                print(f"[place] wrote {args.snap}")
                return
            cv2.imshow("place block  (ghost vs June training frame)", img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("g"):
                ghost = not ghost
            if k == ord("s"):
                out = os.path.join(str(_giava_paths.SCRIPTS_DIR),
                               "reference_frames", "placement_check.png")
                cv2.imwrite(out, img)
                print(f"[place] saved {out}")
    finally:
        pipe.stop()
        if windowed:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
