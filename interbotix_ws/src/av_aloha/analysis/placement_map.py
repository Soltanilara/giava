"""Placement map: which start positions a policy succeeded from.

    # BEFORE the session -- does your grid even fit the detector's window?
    python placement_map.py --check-grid rollouts/snapshots_<stamp>/ep00_top_scene.png

    # AFTER -- one figure per run
    python placement_map.py --run rollouts/rollout_<stamp>.jsonl

WHY A MAP AND NOT A SUCCESS RATE
================================
The flower rollouts fail in a patterned way, not a random one -- the episode
notes already say so ("rotated 45 degrees, grasp failed", "on its side",
"out of distribution placement... too far", against "middle of the grid,
both succeeded").  A single success rate averages that structure away, and
at the trial counts a real robot allows it also cannot separate two policies:
every pairwise Fisher test on the existing transfer_flower runs came back
p = 0.11 to 1.00.  A map asks the question the data can actually answer --
WHERE does it fail -- and that shows up with far fewer trials than a mean
shift does.

WHAT IT DRAWS
=============
Each trial contributes the object's OUTLINE at its start position, taken from
that episode's start snapshot, over the scene itself.  Outline colour is the
verdict; a line to the final centroid shows how far the object actually
travelled.  So one image answers "which placements work", "how were they
lying when they failed", and "how close did the near-misses get".

THE DETECTOR'S WINDOW IS THE REAL CONSTRAINT
============================================
`scene_features.WORKSPACE_ROI` is (200, 150, 500, 300) -- a 300x150 box
inside a 640x480 frame, 47% of the width and 31% of the height.  Objects
outside it are invisible to the detector, and widening it is NOT free:
measured on this rig, opening the ROI to the full frame made detect_object
lock onto a larger blue blob at (552, 476) instead of the flower at
(404, 254).  The window is excluding real distractors, so a placement grid
has to fit INSIDE it.  --check-grid draws the box and your proposed points on
a real snapshot so that is settled before a session, not after.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scene_features import WORKSPACE_ROI, detect_object_outline  # noqa: E402

## Verdict colours, BGR.  Deliberately the same hue at two lightnesses: the
## eye groups by hue, so success and failure read as one variable rather than
## two unrelated categories, and the map stays legible in greyscale.
C_OK = (150, 70, 20)         # dark blue  -- succeeded
C_FAIL = (245, 200, 140)     # light blue -- failed
C_ROI = (90, 90, 90)
C_TRACK = (60, 180, 250)     # start -> final travel


def grid_points(roi=WORKSPACE_ROI, inset=0.10):
    """The 21 placements: 9 cell centres + 12 edge midpoints.

    Laid out on a 5x5 lattice with the 4 corners dropped -- cell centres at
    the odd indices, edge midpoints at the even ones.  `inset` keeps the
    outer ring off the ROI boundary, because an object centred exactly on the
    edge is half outside the detector's window.
    """
    x0, y0, x1, y1 = roi
    dx, dy = (x1 - x0) * inset, (y1 - y0) * inset
    xs = np.linspace(x0 + dx, x1 - dx, 5)
    ys = np.linspace(y0 + dy, y1 - dy, 5)
    pts, n = [], 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if (i % 2) and (j % 2):
                continue                      # drop the in-between diagonals
            if (i % 2 == 0) and (j % 2 == 0):
                kind = "cell"                 # 3x3 centres -> 9
            else:
                kind = "edge"                 # midpoints    -> 12
            n += 1
            pts.append({"n": n, "x": float(x), "y": float(y), "kind": kind})
    return pts


def check_grid(img_path, out_path=None, roi=WORKSPACE_ROI):
    """Draw the ROI and the 21 proposed placements on a real snapshot."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"could not read {img_path}")
    vis = img.copy()
    x0, y0, x1, y1 = roi
    cv2.rectangle(vis, (x0, y0), (x1, y1), C_ROI, 1)
    cv2.putText(vis, f"WORKSPACE_ROI {x1-x0}x{y1-y0}", (x0, max(12, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_ROI, 1, cv2.LINE_AA)
    pts = grid_points(roi)
    for p in pts:
        c = (int(p["x"]), int(p["y"]))
        col = C_OK if p["kind"] == "cell" else C_TRACK
        cv2.drawMarker(vis, c, col, cv2.MARKER_CROSS, 9, 1)
        cv2.putText(vis, str(p["n"]), (c[0] + 5, c[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    ## What the detector currently sees in this very frame, so the operator
    ## can confirm the flower is being found at all before trusting a grid.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cnt, cen, area, found = detect_object_outline(rgb, roi=roi)
    if found:
        cv2.polylines(vis, [cnt.astype(np.int32)], True, C_OK, 2)
        cv2.putText(vis, f"detected area {area}px", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_OK, 1, cv2.LINE_AA)
    else:
        cv2.putText(vis, "NOTHING DETECTED in this frame", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)
    out = Path(out_path or "placement_grid_check.png")
    cv2.imwrite(str(out), vis)
    print(f"[grid] {len(pts)} points ({sum(p['kind']=='cell' for p in pts)} "
          f"cell centres + {sum(p['kind']=='edge' for p in pts)} edge midpoints)")
    print(f"[grid] ROI is {x1-x0} x {y1-y0} px -- every placement must sit "
          f"inside it or the detector cannot see it")
    print(f"[grid] wrote {out}")
    print("\nPlacement order (label -> pixel, top_scene):")
    for p in pts:
        print("  %2d  %-5s  (%3d, %3d)" % (p["n"], p["kind"], p["x"], p["y"]))
    return pts


def _verdict(score_row, stage):
    st = (score_row or {}).get("stages") or {}
    if stage in st:
        return bool(st[stage])
    return bool((score_row or {}).get("furthest", 0) >= 4)


def build_map(run_logs, stage="grasp", out_path=None, roi=WORKSPACE_ROI):
    """One map from one OR MORE run logs.

    Several logs because a grid rarely gets finished in one sitting -- the arm
    faults, the operator adds placements, a session gets split.  Episode
    numbers restart at 0 in every run, so they cannot identify a placement
    across runs; the `cell` label can, which is what --cells is for and what
    this keys on.
    """
    if isinstance(run_logs, (str, Path)):
        run_logs = [run_logs]
    run_logs = [Path(r) for r in run_logs]

    base = None
    drawn = ok = 0
    rows = []

    for run_log in run_logs:
        stamp = run_log.stem.replace("rollout_", "")
        snap_dir = run_log.parent / f"snapshots_{stamp}"
        if not snap_dir.is_dir():
            raise SystemExit(
                f"no snapshot dir {snap_dir} (was --no-snapshots set?)")

        scores = {}
        sp = run_log.parent / "rollout_scores.jsonl"
        if sp.exists():
            for line in open(sp):
                r = json.loads(line)
                if Path(r.get("log", "")).name == run_log.name:
                    scores[r["episode"]] = r

        for start in sorted(snap_dir.glob("ep*_top_scene.png")):
            if start.stem.endswith("_end"):
                continue
            ep = int(start.stem[2:4])
            img = cv2.imread(str(start))
            if img is None:
                continue
            if base is None:
                base = img.copy()

            cnt, cen, area, found = detect_object_outline(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), roi=roi)
            sc = scores.get(ep)
            label = str((sc or {}).get("cell", ep))
            if not found:
                print(f"  [{stamp}] ep{ep:02d} (cell {label}): object NOT "
                      f"FOUND in the start frame -- placed outside the "
                      f"{roi[2]-roi[0]}x{roi[3]-roi[1]} ROI?")
                rows.append({"run": stamp, "episode": ep, "cell": label,
                             "ok": None, "start_px": None, "travel_px": None,
                             "note": "not detected"})
                continue

            good = _verdict(sc, stage)
            col = C_OK if good else C_FAIL
            cv2.polylines(base, [cnt.astype(np.int32)], True, col, 2)
            cv2.putText(base, label, (int(cen[0]) + 6, int(cen[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

            dist = None
            endp = snap_dir / f"ep{ep:02d}_top_scene_end.png"
            if endp.exists():
                eimg = cv2.imread(str(endp))
                if eimg is not None:
                    _c2, cen2, _a2, f2 = detect_object_outline(
                        cv2.cvtColor(eimg, cv2.COLOR_BGR2RGB), roi=roi)
                    if f2:
                        cv2.arrowedLine(base,
                                        (int(cen[0]), int(cen[1])),
                                        (int(cen2[0]), int(cen2[1])),
                                        C_TRACK, 1, tipLength=0.25)
                        dist = float(np.hypot(cen2[0] - cen[0],
                                              cen2[1] - cen[1]))
            drawn += 1
            ok += bool(good)
            rows.append({"run": stamp, "episode": ep, "cell": label,
                         "ok": bool(good),
                         "start_px": [round(cen[0], 1), round(cen[1], 1)],
                         "travel_px": None if dist is None else round(dist, 1)})

    if base is None:
        raise SystemExit("no start snapshots found")

    x0, y0, x1, y1 = roi
    cv2.rectangle(base, (x0, y0), (x1, y1), C_ROI, 1)
    cv2.putText(base, f"{ok}/{drawn} {stage}   dark=ok  light=failed",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1,
                cv2.LINE_AA)

    tag = "_".join(r.stem.replace("rollout_", "") for r in run_logs)[:60]
    out = Path(out_path or run_logs[0].with_name(f"placement_map_{tag}.png"))
    cv2.imwrite(str(out), base)
    jout = out.with_suffix(".json")
    json.dump(rows, open(jout, "w"), indent=2)
    missed = sum(1 for r in rows if r["ok"] is None)
    print(f"[map] {drawn} trials plotted, {ok} succeeded on '{stage}'"
          + (f", {missed} not detected" if missed else ""))
    print(f"[map] wrote {out}")
    print(f"[map] wrote {jout}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-grid", metavar="TOP_SCENE_PNG",
                    help="draw the ROI + 21 proposed placements on a snapshot")
    ap.add_argument("--run", metavar="ROLLOUT_JSONL", nargs="+",
                    help="build the placement map from one or more run logs "
                         "(cells identify placements across runs)")
    ap.add_argument("--stage", default="grasp",
                    help="which stage counts as success (default grasp)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--roi", default=None,
                    help="override WORKSPACE_ROI as x0,y0,x1,y1")
    a = ap.parse_args()
    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else WORKSPACE_ROI
    if a.check_grid:
        check_grid(a.check_grid, a.out, roi)
    elif a.run:
        build_map(a.run, a.stage, a.out, roi)
    else:
        ap.error("need --check-grid or --run")


if __name__ == "__main__":
    main()
