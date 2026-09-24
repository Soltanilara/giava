"""Exact, per-episode measurements of the things the operator writes notes about.

WHY NOT ASK A MODEL.  Most of the operator's vocabulary is already determined
by data the rig records: "got blocked by the collision gate" is a gate event,
"approached too low" is the wrist camera's vertical error at the moment the
gripper closed, "nudged it away" is the object's displacement between the first
frame and the grasp, "rotated 45 cc" is the change in the block's petal phase,
"shaky" is command reversals.  Measuring those is exact, free, and consistent
across 240 episodes; asking a VLM is none of those things.  What is left for a
model is genuine judgement -- intent, whether a correction helped, whether the
arm looked hesitant -- and that is a much smaller prompt.

CAMERA-FRAME CONVENTIONS, because they are what the notes mean.  "Too high /
too low" in the notes is not world z: it is along the VERTICAL OF THE WRIST
CAMERA, which is what the operator is looking at.  So:

    approach_dx   +right / -left      of the grasp aim point, wrist frame
    approach_dy   +below / -above     of the grasp aim point, wrist frame
    approach_range  <0 the gripper closed while still FAR from the block
                    >0 it closed while already past it

All three come from wrist_features' calibrated aim point, so "too low" here
means the same thing it means in the env-state vector the policies are fed.

    python analysis/episode_features.py --limit 5
    python analysis/episode_features.py --policy rel2_rgb --csv out.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import scene_features as sf  # noqa: E402
import wrist_features as wf  # noqa: E402
from analysis.placement_report import detect_placed, petal_phase, _fit  # noqa: E402

OUT = HERE / "rollouts" / "episode_features.jsonl"


def block_in_workspace(bgr):
    """(cx, cy, area, petal_phase) for the block ANYWHERE on the table.

    Not `detect_placed`, which searches only the 16 cm box around the target
    -- at the start of an episode the block is at its placement cell, well
    outside that.  Not `scene_features.detect_object` either, which subtracts
    the static printed marks and so erases the block once it is sitting on
    one.  This is the brightness split (the piece is bright cyan, the print is
    dark navy) over the whole workspace, which works in both places.
    """
    hsv = cv2.cvtColor(bgr[..., ::-1], cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((H >= 95) & (H <= 135) & (S >= 140) & (V >= 135)).astype(np.uint8)
    x0, y0, x1, y1 = sf.WORKSPACE_ROI
    box = np.zeros_like(m); box[y0:y1, x0:x1] = 1
    m = cv2.morphologyEx(m * box, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m)
    best, ba = -1, 0
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if 100 <= a <= 520 and a > ba:
            best, ba = i, a
    if best < 0:
        return None
    cx, cy = float(ce[best][0]), float(ce[best][1])
    return (cx, cy, ba, petal_phase((lab == best).astype(np.uint8), cx, cy))


def gripper_events(rows, arm="right"):
    g = np.array([x.get(f"{arm}_gripper_cmd", 0.0) for x in rows], dtype=float)
    if not len(g):
        return {}
    op = g.max()
    closed = g < op - 0.5
    closes = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
    opens = np.flatnonzero(closed[:-1] & (~closed[1:])) + 1
    return {"n_grasp_attempts": int(len(closes)),
            "first_close_tick": int(closes[0]) if len(closes) else None,
            "last_open_tick": int(opens[-1]) if len(opens) else None,
            "closed_frac": float(closed.mean())}


def motion_features(rows):
    """Shakiness, drift and how hard the command fought the arm."""
    S = np.array([x["state"] for x in rows])[:, :6]
    A = np.array([x["action_raw"] for x in rows])[:, :6]
    d = np.diff(S, axis=0)
    ## HESITATION.  A clean reach moves each joint monotonically; a hesitant
    ## one reverses.  Counting sign flips per joint per tick turns the
    ## operator's "shaky" into a number.
    sign = np.sign(d)
    flips = (sign[1:] * sign[:-1] < 0).mean() if len(sign) > 1 else 0.0
    tail = S[int(len(S) * 0.7):]
    path = float(np.abs(np.diff(tail, axis=0)).sum())
    net = float(np.abs(tail[-1] - tail[0]).sum()) if len(tail) > 1 else 0.0
    gated = [x.get("right_gated") for x in rows if x.get("right_gated")]
    blocks = sum(1 for gg in gated if any((v or {}).get("alpha", 1) <= 0 for v in gg.values()))
    limiters = {}
    for gg in gated:
        for nm, v in gg.items():
            w = (v or {}).get("limiter")
            if w:
                limiters[f"{nm}:{w}"] = limiters.get(f"{nm}:{w}", 0) + 1
    return {"reversal_frac": round(float(flips), 4),
            "tail_path_rad": round(path, 3), "tail_net_rad": round(net, 3),
            "straightness": round(net / path, 3) if path > 1e-9 else None,
            "cmd_vs_measured": round(float(np.abs(A - S).mean()), 4),
            "gate_ticks": len(gated), "gate_blocks": int(blocks),
            "gate_limiter": max(limiters, key=limiters.get) if limiters else None}


def video_features(rec, video, cams, close_tick):
    """What the two cameras say about the block: visibility, where it was when
    the gripper closed, and how far/how much it turned over the episode."""
    out = {}
    if "right_wrist" in cams:
        k = cams.index("right_wrist")
    else:
        return out
    kt = cams.index("top_scene") if "top_scene" in cams else None
    n = len(rec["rows"])
    c = av.open(str(video))
    tw = None
    seen = 0; total = 0
    first_top = last_top = None
    at_close = None
    for i, f in enumerate(c.decode(video=0)):
        if i >= n:
            break
        img = f.to_ndarray(format="bgr24")
        if tw is None:
            tw = img.shape[1] // len(cams)
        if i % 3 == 0 or i == close_tick:
            w = img[:, k * tw:(k + 1) * tw][..., ::-1]
            w = cv2.resize(w, (640, 480), interpolation=cv2.INTER_LINEAR)
            cx, cy, area, c2, s2, found = wf.detect_object(w)
            total += 1
            seen += int(found)
            if i == close_tick and found:
                at_close = {"approach_dx": round((cx - wf.GRASP_AIM_PX[0]) / 320.0, 3),
                            "approach_dy": round((cy - wf.GRASP_AIM_PX[1]) / 240.0, 3),
                            "approach_range": round(float(np.sqrt(max(area, 1) / wf.GRASP_AREA) - 1.0), 3),
                            "block_aspect": None}
        if kt is not None and (i == 0 or i == n - 1):
            cxy = block_in_workspace(img[:, kt * tw:(kt + 1) * tw])
            if i == 0 and cxy:
                first_top = cxy
            if i == n - 1 and cxy:
                last_top = cxy
    c.close()
    out["wrist_visible_frac"] = round(seen / max(total, 1), 3)
    if at_close:
        out.update(at_close)
    if first_top and last_top:
        cmx, cmy = _fit()
        out["block_moved_cm"] = round(float(np.hypot((last_top[0] - first_top[0]) * cmx,
                                                     (last_top[1] - first_top[1]) * cmy)), 2)
        d = ((last_top[3] - first_top[3] + 45.0) % 90.0) - 45.0
        out["block_rot_deg"] = round(-d, 1)          # + = ccw in image coords
    out["block_seen_at_end"] = bool(last_top)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy"); ap.add_argument("--run"); ap.add_argument("--limit", type=int)
    ap.add_argument("--csv")
    args = ap.parse_args()
    scores = {}
    for line in open(HERE / "rollouts" / "rollout_scores.jsonl"):
        if line.strip():
            r = json.loads(line)
            sd = r.get("snapshot_dir")
            if sd:
                scores[(Path(sd).name.replace("snapshots_rollout_", ""), r["episode"])] = r
    rows_out = []
    for lp in sorted(glob.glob(str(HERE / "rollouts" / "rollout_rollout_*.jsonl"))):
        run = Path(lp).stem.replace("rollout_rollout_", "")
        if args.run and args.run != run:
            continue
        for line in open(lp):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("rows"):
                continue
            job = (r.get("checkpoint") or "?").split("/")[-4]
            if args.policy and args.policy not in job:
                continue
            vid = HERE / "rollouts" / "h264" / f"rollout_rollout_{run}_ep{r['episode']:02d}.mp4"
            if not vid.is_file():
                continue
            ge = gripper_events(r["rows"])
            mf = motion_features(r["rows"])
            vf = video_features(r, vid, r.get("cameras") or [], ge.get("first_close_tick") or -1)
            rec = scores.get((run, r["episode"]), {})
            row = {"run": run, "episode": r["episode"], "job": job, "cell": rec.get("cell"),
                   "seconds": round(r.get("seconds") or 0, 1), "ended": r.get("ended"),
                   "n_action_steps": r.get("n_action_steps"), **ge, **mf, **vf}
            rows_out.append(row)
            print(f"{job[:26]:26s} {str(row['cell']):>3} att={row.get('n_grasp_attempts')} "
                  f"vis={row.get('wrist_visible_frac')} dx={row.get('approach_dx')} dy={row.get('approach_dy')} "
                  f"rng={row.get('approach_range')} rev={row['reversal_frac']} gate={row['gate_blocks']} "
                  f"moved={row.get('block_moved_cm')}cm rot={row.get('block_rot_deg')}")
            if args.limit and len(rows_out) >= args.limit:
                break
        if args.limit and len(rows_out) >= args.limit:
            break
    with open(OUT, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(rows_out)} episodes -> {OUT}")
    if args.csv and rows_out:
        import csv
        keys = sorted({k for r in rows_out for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_out)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
