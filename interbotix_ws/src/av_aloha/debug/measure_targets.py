"""Click the workspace ROI and the sorter-box hole centres on a top_scene frame.

    python measure_targets.py --dataset dataset/lerobot/shape_sorter/<run>   # first top_scene frame
    python measure_targets.py --image some_top_scene_frame.png

Writes assets/shape_sorter_targets.json (what shape_sorter.py reads) and an
annotated PNG beside it so the result can be eyeballed.  Re-run whenever the
box or the camera moves; the features are only as good as these pixels.

Clicks, in order:  ROI corner 1, ROI corner 2, then one click per hole in the
order printed.  Keys: r = restart, s/ENTER = save, q = quit without saving.
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

import shape_sorter as ss


def first_top_scene_frame(run_dir: Path):
    import av
    paths = sorted(glob.glob(
        str(run_dir / "videos" / "observation.images.top_scene" / "*" / "*.mp4")))
    if not paths:
        raise SystemExit(f"no top_scene video under {run_dir}")
    with av.open(paths[0]) as c:
        for frame in c.decode(video=0):
            return frame.to_ndarray(format="rgb24")
    raise SystemExit("empty video")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="LeRobot run dir (task/<timestamp>)")
    src.add_argument("--image", help="RGB image file of the top_scene view")
    ap.add_argument("--out", default=str(ss.TARGETS_PATH))
    args = ap.parse_args()

    if args.dataset:
        rgb = first_top_scene_frame(Path(args.dataset))
    else:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"cannot read {args.image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    holes = [ss.CLASSES[c]["hole"] for c in ss.CLASS_NAMES]
    steps = ["ROI corner 1", "ROI corner 2"] + [f"hole: {n}" for n in holes]
    clicks: list[tuple[int, int]] = []

    def redraw():
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        if len(clicks) >= 2:
            (x0, y0), (x1, y1) = clicks[0], clicks[1]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 1)
        for (x, y), name in zip(clicks[2:], holes):
            cv2.circle(img, (x, y), 5, (0, 0, 255), 1)
            cv2.putText(img, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1)
        nxt = steps[len(clicks)] if len(clicks) < len(steps) else "s = save"
        cv2.putText(img, f"click: {nxt}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.putText(img, f"click: {nxt}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1)
        return img

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(steps):
            clicks.append((int(x), int(y)))
            print(f"  {steps[len(clicks) - 1]:16s} -> ({x}, {y})")

    win = "measure_targets"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("order: " + " -> ".join(steps))
    while True:
        cv2.imshow(win, redraw())
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            print("quit, nothing written")
            return
        if k == ord("r"):
            clicks.clear()
            print("  restart")
        if k in (ord("s"), 13) and len(clicks) == len(steps):
            break
    cv2.destroyAllWindows()

    (x0, y0), (x1, y1) = clicks[0], clicks[1]
    roi = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    data = {
        "image_size": [w, h],
        "roi": roi,
        "holes": {name: [x, y] for (x, y), name in zip(clicks[2:], holes)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    png = out.with_suffix(".png")
    cv2.imwrite(str(png), redraw())
    print(f"wrote {out}\n      {png}")

    ## Sanity: which pieces does the detector see in this frame, with this ROI?
    ss._TARGETS["data"] = data
    for c, (cx, cy, area, found) in ss.detect_all(rgb).items():
        print(f"  {c:9s} {'found' if found else '  -  '} "
              + (f"at ({cx:.0f}, {cy:.0f}) area={area}" if found else ""))


if __name__ == "__main__":
    main()
