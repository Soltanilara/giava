"""Measure WHERE the flower was placed, per rollout episode, in centimetres.

WHY A SEPARATE DETECTOR.  The task ends with the object sitting ON the printed
flower target, and both are blue -- so `scene_features.detect_object`, which
subtracts the static printed marks, erases most of the very thing we want to
measure (a placed object leaves only an ~90 px sliver, and its centroid is the
centroid of the sliver, not of the object).

The separation that does work is BRIGHTNESS.  Measured on the 2026-09-19
snapshots, inside the same blue hue band:

    printed target   V percentiles  [102 118 129]
    placed object    V reaches      163

so a value floor at V>=135 keeps ~220 px of object and ~5 px of printed target.
That is the whole trick: hue finds blue, value separates the light plastic
flower from the dark printed clover underneath it.

WHAT IS REPORTED, per episode with a placement:

    dx, dy   centimetres from the flower target, in the operator's words
             (left/right, high/low -- image axes, as seen in top_scene)
    dist     radial error in cm
    rot      degrees from the demonstrator's mean placement orientation,
             signed cw/ccw.  From the blob's second moments as a DOUBLE angle,
             so a flower and the same flower turned 180 degrees agree -- the
             piece is near-symmetric and a single angle would flip at random.
    frac     object area as a fraction of the reference area: << 1 means the
             arm is occluding it and the measurement is not trustworthy.

Scale comes from assets/flower_px2ee.json (2.38 mm/px in x, 2.73 in y) -- the
same empirical pixel->metres fit auto_reset drives the arm with.

    python analysis/placement_report.py                       # every 2026-09-19 run
    python analysis/placement_report.py --runs 20260919_124148
    python analysis/placement_report.py --csv out.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import scene_features as sf  # noqa: E402
from paths import ASSETS_DIR  # noqa: E402

HUE_LO, HUE_HI = 95, 135
SAT_MIN = 140
VAL_MIN = 135          # the object/printed-target split; see module docstring
AREA_MIN = 100
AREA_MAX = 520

## SEARCH ONLY WHERE A PLACEMENT CAN BE.  The robot arms carry bright blue
## housings that pass the same hue+value test as the piece: on 2026-09-19 the
## detector locked onto the camera arm at (352,187) and reported a 10.9 cm
## "placement error" for an episode where the piece was somewhere else
## entirely (area 117 px of arm, against ~204 px for the real piece).  The
## workspace ROI is far too generous for this job -- a placement is by
## definition near the target, so the box is +/-34 px in x and +/-30 in y
## around it, about 16 x 16 cm.  Anything outside is not a placement, and a
## blob far from the expected area is not the piece.
ROI = (int(sf.FLOWER_TARGET_PX[0]) - 34, int(sf.FLOWER_TARGET_PX[1]) - 30,
       int(sf.FLOWER_TARGET_PX[0]) + 34, int(sf.FLOWER_TARGET_PX[1]) + 30)


def _fit():
    d = json.loads((ASSETS_DIR / "flower_px2ee.json").read_text())
    return d["m_per_px"]["x"] * 100.0, d["m_per_px"]["y"] * 100.0   # cm per px


def detect_placed(bgr):
    """(cx, cy, area, angle_deg, found) for the bright-blue object."""
    rgb = bgr[..., ::-1]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((H >= HUE_LO) & (H <= HUE_HI) & (S >= SAT_MIN) & (V >= VAL_MIN)).astype(np.uint8)
    x0, y0, x1, y1 = ROI
    box = np.zeros_like(m)
    box[y0:y1, x0:x1] = 1
    m = cv2.morphologyEx(m * box, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m)
    best, ba = -1, 0
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if AREA_MIN <= a <= AREA_MAX and a > ba:
            best, ba = i, a
    if best < 0:
        return 0.0, 0.0, 0, 0.0, False
    cx, cy = float(ce[best][0]), float(ce[best][1])
    ang = petal_phase((lab == best).astype(np.uint8), cx, cy)
    return cx, cy, ba, ang, True


def petal_phase(blob, cx, cy):
    """Orientation MOD 90 degrees, from the 4th angular harmonic of the blob's
    radius profile.

    Second moments cannot do this job.  The piece is a four-lobed flower, so
    its second-moment ellipse is nearly a circle (mu20 ~ mu02, mu11 ~ 0) and
    the principal axis is numerically meaningless -- measured on the 09-19
    placements it scattered with a median |rotation| of 17-21 deg and a p90 of
    80 deg, which is noise, not orientation.  The 4th harmonic locks onto the
    four petals instead: same placements give a concentration R of 0.74 and a
    median deviation of 5 deg.

    Mod 90 is the honest range: a four-fold symmetric piece turned 90 degrees
    is the same piece, so no method can distinguish them and none should
    pretend to.
    """
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(c) < 12:
        return 0.0
    d = c - np.array([cx, cy])
    th = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])
    g = np.linspace(-np.pi, np.pi, 181)[:-1]
    idx = np.argsort(th)
    rr = np.interp(g, th[idx], r[idx], period=2 * np.pi)
    z = np.sum(rr * np.exp(-4j * g))
    return float((np.degrees(np.angle(z)) / 4.0) % 90.0)


def _policy_of(rec):
    """Job name from the checkpoint path, or '?' for an episode with no score
    record (aborted before scoring, or a run still in progress)."""
    parts = Path((rec or {}).get("checkpoint") or "").parts
    return parts[-4] if len(parts) >= 4 else "?"


## The piece, measured from 77 placed blobs: 16 px across = 3.81 cm.  The
## operator's notes measure offsets in FRACTIONS OF THE BLOCK ("half flower
## too right", "quarter too high"), so that is the unit this reports in, with
## centimetres alongside.
BLOCK_CM = 3.81

## "Ideal" calibrated against the episodes the operator themself called ideal:
## those measured 0.50-1.02 cm and 0-4 deg, while the ones they quantified as
## off begin at "0.5 cm too high" (measured 0.47) and "~1 cm too high" (1.02).
## A quarter block and 10 degrees sits on that boundary.
IDEAL_CM = BLOCK_CM / 4.0
IDEAL_DEG = 10.0

FRACS = [(0.20, "a touch"), (0.38, "a quarter block"), (0.60, "half a block"),
         (0.85, "three-quarters of a block"), (1.35, "a full block")]


def frac_words(cm):
    f = abs(cm) / BLOCK_CM
    for lim, name in FRACS:
        if f < lim:
            return name
    return f"{f:.1f} blocks"


def words(dx_cm, dy_cm, rot_deg=None):
    """The placement in the operator's vocabulary, or 'ideal placement'."""
    dist = float(np.hypot(dx_cm, dy_cm))
    rot = 0.0 if rot_deg is None else abs(rot_deg)
    if dist <= IDEAL_CM and rot <= IDEAL_DEG:
        return "ideal placement"
    out = []
    if abs(dy_cm) >= 0.25:
        out.append(f"{frac_words(dy_cm)} too {'low' if dy_cm > 0 else 'high'}")
    if abs(dx_cm) >= 0.25:
        out.append(f"{frac_words(dx_cm)} to the {'right' if dx_cm > 0 else 'left'}")
    if rot > IDEAL_DEG:
        out.append(f"rotated {rot:.0f}\u00b0 {'ccw' if rot_deg > 0 else 'cw'}")
    return ", ".join(out) if out else "just off target"


def signed_rot(ang, ref):
    """Smallest signed rotation from ref, MOD 90 -- the piece has four-fold
    symmetry, so +/-45 deg is the whole meaningful range.  Positive = ccw in
    image coordinates (y grows downward, so the sign is flipped)."""
    return -(((ang - ref + 45.0) % 90.0) - 45.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run stamps, e.g. 20260919_124148")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--ref-run", default=None, help="run whose placements define the reference orientation")
    args = ap.parse_args()

    cmx, cmy = _fit()
    tx, ty = sf.FLOWER_TARGET_PX

    dirs = sorted(glob.glob(str(HERE / "rollouts" / "snapshots_rollout_*")))
    if args.runs:
        dirs = [d for d in dirs if any(r in d for r in args.runs)]
    scores = {}
    for line in open(HERE / "rollouts" / "rollout_scores.jsonl"):
        if line.strip():
            r = json.loads(line)
            sd = r.get("snapshot_dir")
            if sd:
                scores[(Path(sd).name, r["episode"])] = r

    rows = []
    for d in dirs:
        for e in sorted(glob.glob(f"{d}/*top_scene_end.png")):
            ep = int(re.search(r"ep(\d+)_", Path(e).name).group(1))
            rec = scores.get((Path(d).name, ep))
            img = cv2.imread(e)
            if img is None:
                continue
            cx, cy, area, ang, ok = detect_placed(img)
            rows.append({
                "run": Path(d).name.replace("snapshots_rollout_", ""), "ep": ep,
                "policy": _policy_of(rec),
                "cell": (rec or {}).get("cell"), "placed": bool(((rec or {}).get("stages") or {}).get("placement")),
                "found": ok, "area": area, "cx": cx, "cy": cy, "ang": ang,
                "dx_cm": (cx - tx) * cmx if ok else None,
                "dy_cm": (cy - ty) * cmy if ok else None,
                "note": ((rec or {}).get("note") or "")[:60],
            })

    placed = [r for r in rows if r["found"] and r["placed"]]
    if not placed:
        print("no episodes with both a detection and a placement flag")
        return
    ref_area = float(np.median([r["area"] for r in placed]))
    ## circular median, mod 90
    _z = np.exp(1j * np.radians(np.array([r["ang"] for r in placed]) * 4.0)).mean()
    ref_ang = float((np.degrees(np.angle(_z)) / 4.0) % 90.0)
    ref_R = float(abs(_z))
    for r in rows:
        r["frac"] = (r["area"] / ref_area) if r["found"] else 0.0
        r["rot"] = signed_rot(r["ang"], ref_ang) if r["found"] else None
        r["dist_cm"] = float(np.hypot(r["dx_cm"], r["dy_cm"])) if r["found"] else None

    print(f"reference: flower target ({tx:.0f},{ty:.0f}) px · {cmx*10:.2f}×{cmy*10:.2f} mm/px · "
          f"median placed area {ref_area:.0f} px · reference orientation {ref_ang:.0f}° "
          f"(4-fold, concentration R={ref_R:.2f})\n")
    by = {}
    for r in rows:
        by.setdefault(r["policy"], []).append(r)
    for pol, rs in by.items():
        good = [r for r in rs if r["found"] and r["placed"] and r["frac"] > 0.5]
        print(f"=== {pol}   {len(rs)} episodes, {sum(1 for r in rs if r['placed'])} placed, "
              f"{len(good)} measurable")
        if good:
            dx = np.array([r["dx_cm"] for r in good]); dy = np.array([r["dy_cm"] for r in good])
            di = np.array([r["dist_cm"] for r in good]); ro = np.array([abs(r["rot"]) for r in good])
            print(f"    offset  dx {dx.mean():+.2f} ± {dx.std():.2f} cm   dy {dy.mean():+.2f} ± {dy.std():.2f} cm")
            print(f"    radial error  median {np.median(di):.2f} cm   p90 {np.percentile(di,90):.2f} cm")
            print(f"    |rotation|    median {np.median(ro):.0f}°        p90 {np.percentile(ro,90):.0f}°")
            n_ideal = sum(1 for r in good if r["dist_cm"] <= IDEAL_CM and abs(r["rot"]) <= IDEAL_DEG)
            print(f"    IDEAL placements  {n_ideal}/{len(good)} measured ({n_ideal/len(good):.0%})"
                  f"   [within {IDEAL_CM:.2f} cm (a quarter block) and {IDEAL_DEG:.0f}\u00b0]")
        for r in sorted(rs, key=lambda r: r["ep"]):
            if not r["placed"]:
                continue
            if not r["found"]:
                print(f"    {r['cell'] or '?':>3s} ep{r['ep']:02d}  NOT DETECTED (arm occluding, or placed outside the workspace)")
            elif r["frac"] <= 0.5:
                print(f"    {r['cell'] or '?':>3s} ep{r['ep']:02d}  partly occluded ({r['frac']:.0%} of reference area) -- skipped")
            else:
                print(f"    {str(r['cell'] or '?'):>3s} ep{r['ep']:02d}  "
                      f"{words(r['dx_cm'], r['dy_cm'], r['rot']):<50s} "
                      f"({r['dist_cm']:.2f} cm, {abs(r['rot']):.0f}\u00b0)")
        print()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
