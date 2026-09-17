from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract_rgbd_episode import compute_depth_visualization_range, make_depth_preview, require_cv2
from rgbd_episode_reader import RGBDEpisodeReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open one RGB/depth frame and click a pixel to inspect depth in meters.",
    )
    parser.add_argument("dataset_root", help="Path to a LeRobot dataset run root.")
    parser.add_argument("--camera", default="right_wrist", help="Camera name, e.g. right_wrist or top_scene.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to inspect.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2 = require_cv2()

    with RGBDEpisodeReader(
        dataset_root=args.dataset_root,
        camera=args.camera,
        episode_index=args.episode,
    ) as reader:
        frame = reader.get_frame(args.frame, rgb_order="rgb")
        depth_min_m, depth_max_m = compute_depth_visualization_range([frame.depth])
        depth_preview = make_depth_preview(frame.depth, depth_min_m, depth_max_m)

    rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    depth_bgr = cv2.cvtColor(depth_preview, cv2.COLOR_RGB2BGR)
    combined = np.concatenate([rgb_bgr, depth_bgr], axis=1)
    rgb_width = rgb_bgr.shape[1]

    window_name = f"Depth Viewer: {args.camera} episode={args.episode} frame={args.frame}"

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if x >= rgb_width:
            x_depth = x - rgb_width
        else:
            x_depth = x

        if x_depth < 0 or x_depth >= frame.depth.shape[1] or y < 0 or y >= frame.depth.shape[0]:
            return

        depth_value = float(frame.depth[y, x_depth])
        valid = np.isfinite(depth_value) and depth_value > 0.0
        label = f"x={x_depth} y={y} depth={depth_value:.4f} m" if valid else f"x={x_depth} y={y} depth=invalid"
        print(label)

        display = combined.copy()
        cv2.circle(display, (x_depth, y), 4, (0, 255, 0), -1)
        cv2.circle(display, (x_depth + rgb_width, y), 4, (0, 255, 0), -1)
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, combined)
    print("Click a pixel in either half of the window to inspect depth. Press q or Esc to quit.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
