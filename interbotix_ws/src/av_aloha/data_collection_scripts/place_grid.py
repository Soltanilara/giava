#!/usr/bin/env python3
"""Live placement guide for the transfer_flower interpolation study.

Opens top_scene on its own -- no arms, no ROS, no teleop -- draws the
detector's WORKSPACE_ROI, the study hull and the 4x4 lattice over the live
image, and runs the SAME detector the policy's environment_state uses
(scene_features.detect_object).  Put the object down, read off which cell you
are on and how many pixels you are from its centre, nudge, then record.

Why this exists: the lattice is specified in top_scene pixels, and pixels are
not something you can hit by eye.  Without it you would be placing "roughly
where the figure showed", which is exactly the sloppy coverage the study is
trying to measure against.

    python place_grid.py                  # the study lattice
    python place_grid.py --cell B2        # highlight one cell, hide the rest
    python place_grid.py --tol 6          # tighter "on target" tolerance

Keys:  q / ESC quit      s  save a snapshot next to the script
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None

try:
    from . import scene_features as sf
    from .camera_manager import (CAMERA_SERIALS, RS_COLOR_FPS,
                                 RS_COLOR_HEIGHT, RS_COLOR_WIDTH)
except ImportError:
    import scene_features as sf
    from camera_manager import (CAMERA_SERIALS, RS_COLOR_FPS,
                                RS_COLOR_HEIGHT, RS_COLOR_WIDTH)

## THE STUDY HULL, in top_scene pixels.  Chosen to sit inside
## scene_features.WORKSPACE_ROI with margin for the object's own radius, clear
## of the printed target sheet on the left (it ends at x=342) and clear of the
## table frame on the right.  4.7x the area of the 107 demos collected to date,
## which spanned only x 395..445, y 213..260.
HULL = (362, 180, 470, 282)                      # x0, y0, x1, y1
GRID = 4                                         # 4x4 -> 12 perimeter, 4 interior

CYAN, MAG, AMBER = (200, 220, 60), (255, 110, 190), (60, 170, 235)   # BGR
WHITE, GREEN, RED = (255, 255, 255), (120, 235, 140), (70, 70, 235)


def lattice():
    """[(name, x, y, is_perimeter)] for the 4x4 grid over HULL."""
    x0, y0, x1, y1 = HULL
    xs = np.linspace(x0, x1, GRID)
    ys = np.linspace(y0, y1, GRID)
    out = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            edge = i in (0, GRID - 1) or j in (0, GRID - 1)
            out.append((f"{'ABCDEFGH'[i]}{j + 1}", float(x), float(y), edge))
    return out


def random_interior(n, seed=0, margin=14):
    """n placements drawn uniformly inside HULL (inset by `margin` px so the
    object's own radius stays inside), named R1..Rn, all non-perimeter.
    Seeded, so the same labels mean the same pixels across sessions and
    policies -- which is what makes them a paired test rather than "somewhere
    in the middle".  Points closer than 14 px to a lattice cell are redrawn:
    the lattice already covers those."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = HULL
    cells = lattice()
    out = []
    while len(out) < n:
        x = float(rng.uniform(x0 + margin, x1 - margin))
        y = float(rng.uniform(y0 + margin, y1 - margin))
        if min(np.hypot(c[1] - x, c[2] - y) for c in cells) < 14.0:
            continue
        if any(np.hypot(o[1] - x, o[2] - y) < 14.0 for o in out):
            continue
        out.append((f"R{len(out) + 1}", x, y, False))
    return out


def edge_midpoints():
    """M1..M12: the midpoints of the hull's rim BETWEEN adjacent demonstrated
    cells.  Every training placement sits on a lattice cell, so these are on
    the boundary the data covers but offset from any episode -- the cheapest
    test of whether the policy interpolates ALONG the rim rather than
    memorising the twelve cells on it."""
    x0, y0, x1, y1 = HULL
    xs = np.linspace(x0, x1, GRID)
    ys = np.linspace(y0, y1, GRID)
    out = []
    for i in range(GRID - 1):            # top and bottom edges
        out.append(((xs[i] + xs[i + 1]) / 2, y0))
        out.append(((xs[i] + xs[i + 1]) / 2, y1))
    for j in range(GRID - 1):            # left and right edges
        out.append((x0, (ys[j] + ys[j + 1]) / 2))
        out.append((x1, (ys[j] + ys[j + 1]) / 2))
    return [(f"M{n + 1}", float(x), float(y), False)
            for n, (x, y) in enumerate(out)]


def cells_for(labels, seed=0):
    """The lattice plus however many R<n> points the labels ask for."""
    want = [int(l[1:]) for l in labels if l[:1] == "R" and l[1:].isdigit()]
    out = lattice() + (random_interior(max(want), seed) if want else [])
    if any(l[:1] == "M" and l[1:].isdigit() for l in labels):
        out = out + edge_midpoints()
    return out


def draw(bgr, cells, det, tol, only=None):
    x0, y0, x1, y1 = sf.WORKSPACE_ROI
    ## ROI: dashed, because it is a limit rather than a thing to aim at.
    for x in range(x0, x1, 14):
        cv2.line(bgr, (x, y0), (min(x + 7, x1), y0), AMBER, 1)
        cv2.line(bgr, (x, y1), (min(x + 7, x1), y1), AMBER, 1)
    for y in range(y0, y1, 14):
        cv2.line(bgr, (x0, y), (x0, min(y + 7, y1)), AMBER, 1)
        cv2.line(bgr, (x1, y), (x1, min(y + 7, y1)), AMBER, 1)
    cv2.putText(bgr, "WORKSPACE_ROI", (x0 + 3, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, AMBER, 1, cv2.LINE_AA)

    cv2.rectangle(bgr, HULL[:2], HULL[2:], (120, 200, 120), 1)

    cx, cy, area, found = det
    near = None
    if found:
        pool = [c for c in cells if only is None or c[0] == only]
        near = min(pool, key=lambda c: (c[1] - cx) ** 2 + (c[2] - cy) ** 2)
        ## Distance to the cell you are CLOSEST to -- not to the cell you
        ## meant.  Pass --cell when you want the second thing.
        d = float(np.hypot(near[1] - cx, near[2] - cy))

    for name, x, y, edge in cells:
        if only is not None and name != only:
            continue
        p = (int(round(x)), int(round(y)))
        col = CYAN if edge else MAG
        hot = found and near is not None and near[0] == name
        if edge:
            cv2.circle(bgr, p, 7, col, 2 if not hot else -1)
        else:
            cv2.drawMarker(bgr, p, col, cv2.MARKER_TILTED_CROSS, 15,
                           2 if not hot else 4)
        cv2.circle(bgr, p, int(tol), col, 1)
        cv2.putText(bgr, name, (p[0] - 9, p[1] - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    if not found:
        cv2.putText(bgr, "OBJECT NOT DETECTED -- outside the ROI, or occluded",
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
        return bgr

    cv2.circle(bgr, (int(cx), int(cy)), 4, WHITE, -1)
    cv2.line(bgr, (int(cx), int(cy)),
             (int(round(near[1])), int(round(near[2]))), WHITE, 1)
    ok = d <= tol
    role = "perimeter" if near[3] else "HOLDOUT -- do not record here"
    cv2.putText(bgr, f"{near[0]}  {d:4.1f} px  ({role})", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN if ok else RED, 2,
                cv2.LINE_AA)
    cv2.putText(bgr, f"obj ({cx:.0f},{cy:.0f})  area {area}", (14, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    if ok and near[3]:
        cv2.putText(bgr, "ON TARGET -- record", (14, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
    return bgr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", help="highlight only this cell, e.g. B2")
    ap.add_argument("--random", type=int, default=0,
                    help="also draw N seeded random interior points R1..RN")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--midpoints", action="store_true",
                    help="also draw M1..M8, the rim midpoints between cells")
    ap.add_argument("--tol", type=float, default=8.0,
                    help="px from the cell centre that counts as on target")
    args = ap.parse_args()

    if rs is None:
        raise SystemExit("pyrealsense2 is not importable -- activate the env.")
    cells = (lattice() + (random_interior(args.random, args.seed) if args.random else [])
             + (edge_midpoints() if args.midpoints else []))
    names = {c[0] for c in cells}
    if args.cell and args.cell not in names:
        raise SystemExit(f"--cell must be one of {sorted(names)}")

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(CAMERA_SERIALS["top_scene"])
    cfg.enable_stream(rs.stream.color, RS_COLOR_WIDTH, RS_COLOR_HEIGHT,
                      rs.format.rgb8, RS_COLOR_FPS)
    pipe.start(cfg)
    print("top_scene open.  q/ESC quit, s saves a snapshot.")
    print(f"hull {HULL}   tolerance {args.tol:.0f} px"
          + (f"   showing only {args.cell}" if args.cell else ""))
    try:
        while True:
            fr = pipe.wait_for_frames().get_color_frame()
            if not fr:
                continue
            rgb = np.asanyarray(fr.get_data())
            det = sf.detect_object(rgb)
            bgr = draw(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy(),
                       cells, det, args.tol, args.cell)
            cv2.imshow("placement grid -- top_scene", bgr)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                p = f"placement_grid_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(p, bgr)
                print(f"  saved {p}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
