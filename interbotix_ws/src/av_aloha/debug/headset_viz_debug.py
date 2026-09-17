"""Live viser view of what the headset is actually sending -- head, hand, and
controller frames plus hand keypoints -- through OUR real gvlink_headset.py,
not a synthetic mock. Built to check the receive_data() fix (2026-08-24):
hand-tracking sessions used to send both "controllers" to the same point
(ControllerState defaults to the origin when never populated), which read as
overlapping hands and a self-collision refusal. This shows the RAW packet
gvlink_headset.py resolves that from, so the fix is visible directly rather
than inferred from robot behaviour.

Sends a synthetic test pattern as video (like gvlink's own mock_robot.py
--source pattern) so the viewer has something to display and keeps its
input uplink flowing -- no camera, no arms, no ROS needed.

Run:  python headset_viz_debug.py [--port 8099]
Then connect the headset to this machine same as for data_collection.py.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gvlink_headset import GvLinkHeadset
from gvlink.viz import HeadsetViz


def _pattern(w: int, h: int, t: float) -> np.ndarray:
    """Cheap moving test pattern -- just enough that a black-screen/frozen-
    stream problem is visually obvious, distinct from a pose problem."""
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    phase = t * 0.5
    img = np.stack([
        ((np.sin(xv * 20 + phase) + 1) * 0.5 * 255),
        ((np.sin(yv * 20 - phase) + 1) * 0.5 * 255),
        np.full_like(xv, 128.0),
    ], axis=-1).astype(np.uint8)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--src", default="640x480")
    args = ap.parse_args()
    w, h = (int(v) for v in args.src.split("x"))

    headset = GvLinkHeadset(name="headset-viz-debug", src_size=(w, h))
    headset.run_in_thread()
    print(f"[headset_viz_debug] gvlink up; connect the headset to this "
          f"machine's IP same as for data_collection.py")

    viz = HeadsetViz(port=args.port)
    print(f"[headset_viz_debug] viser: {viz.url}")

    t0 = time.monotonic()
    last_seq = None
    try:
        while True:
            t = time.monotonic() - t0
            img = _pattern(w, h, t)
            headset.send_images(img, img)

            pkt = headset._fresh_input()
            if pkt is not None and pkt.seq != last_seq:
                last_seq = pkt.seq
                viz.update(pkt)
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        headset.close()


if __name__ == "__main__":
    main()
