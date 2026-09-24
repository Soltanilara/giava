"""Grasp-alignment error for recorded rollouts -- a continuous metric.

WHY.  Binary success at n=20 placements cannot separate policies: the same
checkpoint scored 3/10 and 5/10 on identical scenes on different days.  But
two policies that both succeed 12/20 can differ by 15 px in WHERE they close
the gripper, and that difference is measurable on every rollout, RGB-only
policies included, because it is read off the recorded wrist video after the
fact.  It also measures the failure the operator keeps seeing -- closing above
or below the object -- directly: `w_range` at the close tick is the height
error.

WHAT.  For every episode in a rollout log that has a video, find the tick the
gripper closed, crop the right_wrist tile out of the composite recording,
and run wrist_features.detect_object on it (upscaled back to 640x480, the
resolution GRASP_AIM_PX / GRASP_AREA were measured at).  Reports, per episode:

    close_tick     first tick the gripper command dropped
    w_err_x/y      object centroid minus grasp aim-point, [-1,1]; 0 = aligned
    w_err_px       that error in pixels (Euclidean)
    w_range        sqrt(area/grasp_area) - 1; 0 = at grasp distance,
                   negative = closed too far away / too high
    approach_min_px  best alignment in the 25 ticks BEFORE the close --
                   separates "aimed well but closed late" from "never aimed"
    found          detector saw the object at the close tick

Recorded rollouts have one video frame per tick (rows[i] <-> frame i), and
the composite is the policy's cameras side by side, in `cameras` order, at
--video-scale (default 0.5 -> 320x240 tiles).

    python analysis/score_grasp.py rollouts/rollout_<...>.jsonl [...]
    python analysis/score_grasp.py --summary            # group by checkpoint
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
import wrist_features as wf  # noqa: E402

OUT = HERE / "rollouts" / "grasp_scores.jsonl"
NATIVE = (640, 480)
PRE_WINDOW = 25          # ticks before the close to search for best alignment
DROP = 0.5               # gripper command falls by this much => closed


def close_tick(rows, arm="right"):
    """First tick at which the gripper command dropped from its opening
    value -- the same rule build_relative_action_dataset / the wrist
    aim-point measurement used on the training data."""
    g = np.array([r.get(f"{arm}_gripper_cmd", np.nan) for r in rows], dtype=float)
    if np.all(np.isnan(g)):
        return None
    g0 = np.nanmax(g[: max(5, len(g) // 10)])
    hits = np.flatnonzero(g < g0 - DROP)
    return int(hits[0]) if len(hits) else None


def wrist_tile(frame_bgr, cameras, scale):
    """Crop the right_wrist tile from the composite and return it as RGB at
    native resolution."""
    if "right_wrist" not in cameras:
        return None
    k = cameras.index("right_wrist")
    tw = int(round(NATIVE[0] * scale))
    tile = frame_bgr[:, k * tw:(k + 1) * tw]
    tile = cv2.resize(tile, NATIVE, interpolation=cv2.INTER_LINEAR)
    return tile[..., ::-1]


def features_at(rgb):
    cx, cy, area, c2, s2, found = wf.detect_object(rgb)
    if not found:
        return None
    w, h = NATIVE
    ex = (cx - wf.GRASP_AIM_PX[0]) / (w / 2.0)
    ey = (cy - wf.GRASP_AIM_PX[1]) / (h / 2.0)
    rng = float(np.sqrt(max(area, 1) / wf.GRASP_AREA) - 1.0)
    px = float(np.hypot(cx - wf.GRASP_AIM_PX[0], cy - wf.GRASP_AIM_PX[1]))
    return {"w_err_x": round(ex, 4), "w_err_y": round(ey, 4),
            "w_err_px": round(px, 1), "w_range": round(rng, 4),
            "area_px": int(area), "cx": round(cx, 1), "cy": round(cy, 1)}


def score_episode(rec, log_path: Path, video_scale: float):
    video = rec.get("video")
    if not video:
        ## The recorder names videos <log-stem>_epNN.mp4 next to the log.
        cand = log_path.with_name(f"{log_path.stem}_ep{rec['episode']:02d}.mp4")
        video = str(cand) if cand.exists() else None
    if not video or not Path(video).exists():
        return {"skipped": "no video"}
    rows = rec.get("rows") or []
    ct = close_tick(rows)
    if ct is None:
        return {"skipped": "gripper never closed"}

    want = set(range(max(0, ct - PRE_WINDOW), ct + 1))
    feats = {}
    c = av.open(video)
    for i, fr in enumerate(c.decode(video=0)):
        if i > ct:
            break
        if i in want:
            tile = wrist_tile(fr.to_ndarray(format="bgr24"), rec["cameras"], video_scale)
            if tile is not None:
                feats[i] = features_at(tile)
    c.close()

    at_close = feats.get(ct)
    pre = [f["w_err_px"] for i, f in feats.items() if i < ct and f]
    out = {"close_tick": ct, "close_t_s": round(rows[ct]["t"], 2),
           "found": at_close is not None,
           "approach_min_px": round(min(pre), 1) if pre else None,
           "approach_found_frac": round(sum(1 for i in want if i < ct and feats.get(i)) / max(1, PRE_WINDOW), 2)}
    if at_close:
        out.update(at_close)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="rollout_*.jsonl files (default: all under rollouts/)")
    ap.add_argument("--video-scale", type=float, default=0.5)
    ap.add_argument("--summary", action="store_true", help="group grasp_scores.jsonl by checkpoint")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    if args.summary:
        return summary()

    logs = [Path(p) for p in (args.logs or sorted(glob.glob(str(HERE / "rollouts" / "rollout_*.jsonl"))))]
    logs = [p for p in logs if p.name != "rollout_scores.jsonl"]
    done = set()
    if OUT.exists():
        for l in open(OUT):
            if l.strip():
                r = json.loads(l)
                done.add((r["log"], r["episode"]))

    n_new = 0
    print(f"{'log':44s} {'ep':>2s} {'ckpt':28s} {'close':>6s} {'err_px':>7s} {'w_range':>8s} {'pre_min':>8s} found")
    with open(OUT, "a") as fo:
        for lp in logs:
            recs = [json.loads(l) for l in open(lp) if l.strip()]
            for rec in recs:
                if "episode" not in rec or (str(lp), rec["episode"]) in done:
                    continue
                s = score_episode(rec, lp, args.video_scale)
                ck = "/".join(Path(rec.get("checkpoint", "?")).parts[-4:-2])
                if "skipped" in s:
                    print(f"{lp.name[:44]:44s} {rec['episode']:2d} {ck[:28]:28s}   -- {s['skipped']}")
                    continue
                print(f"{lp.name[:44]:44s} {rec['episode']:2d} {ck[:28]:28s} {s['close_tick']:6d} "
                      f"{str(s.get('w_err_px', '-')):>7s} {str(s.get('w_range', '-')):>8s} "
                      f"{str(s.get('approach_min_px', '-')):>8s} {s['found']}")
                row = {"log": str(lp), "episode": rec["episode"], "checkpoint": rec.get("checkpoint"),
                       "cell": rec.get("cell"), "ended": rec.get("ended"), **s}
                if not args.no_write:
                    fo.write(json.dumps(row) + "\n")
                n_new += 1
    print(f"\n{n_new} episode(s) scored -> {OUT}")


def summary():
    if not OUT.exists():
        raise SystemExit(f"{OUT} does not exist yet")
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r.get("checkpoint") or "?", []).append(r)
    print(f"{'checkpoint':52s} {'n':>3s} {'found':>5s} {'err_px med [IQR]':>22s} {'w_range med [IQR]':>24s} {'pre_min med':>11s}")
    for ck, rs in sorted(by.items()):
        f = [r for r in rs if r.get("found")]
        def med_iqr(k, fmt):
            v = np.array([r[k] for r in f if r.get(k) is not None], dtype=float)
            if not len(v):
                return "-"
            q = np.percentile(v, [25, 50, 75])
            return f"{q[1]:{fmt}} [{q[0]:{fmt}},{q[2]:{fmt}}]"
        name = "/".join(Path(ck).parts[-4:-2]) if ck != "?" else ck
        print(f"{name[:52]:52s} {len(rs):3d} {len(f):5d} {med_iqr('w_err_px', '.0f'):>22s} "
              f"{med_iqr('w_range', '.2f'):>24s} {med_iqr('approach_min_px', '.0f'):>11s}")


if __name__ == "__main__":
    main()
