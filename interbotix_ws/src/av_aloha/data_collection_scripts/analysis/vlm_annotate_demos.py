"""Label TELEOPERATED demonstrations with the same schema as the rollouts.

WHY BOTH.  The rollouts say what going wrong looks like.  The demonstrations
say what the operator's own good behaviour looks like -- and the two labelled
with one schema can be compared directly: if the demonstrations read
`approach: on_target, grasp: centred` and a policy's rollouts read `too_low`
four times in ten, that gap is a measurement rather than an impression.  For
the DAgger episodes it is sharper still, because those contain the operator's
corrections FROM the states the policy gets itself into.

WHY A SEPARATE READER.  A rollout is recorded as one composite mp4 -- every
camera tiled into a single frame, which is exactly what the annotator wants to
show a model.  A LeRobot demonstration is the opposite: one AV1 video per
camera, sliced into episodes by `meta/episodes/*.parquet`.  This reads those,
cuts the right frame index out of each camera, and tiles them in the same
order with the same labels, so the model sees the identical format and the
labels stay comparable.

    python analysis/vlm_annotate_demos.py --root dataset/lerobot/transfer_flower_merged/<run> --limit 8
    python analysis/vlm_annotate_demos.py --root <run> --episodes 50 106   # the DAgger block
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from analysis.placement_report import _fit  # noqa: E402
from analysis.vlm_annotate import PROMPT, SCHEMA, ask, scale_bar  # noqa: E402

OUT = HERE / "rollouts" / "vlm_labels_demos.jsonl"
CAM_ORDER = ["low_scene", "right_wrist", "top_scene"]


def episode_table(root: Path):
    fs = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).sort_values("episode_index")


def frame_span(root: Path, ep: int):
    """(from_index, to_index) of one episode in the dataset's global frame order."""
    t = episode_table(root)
    row = t[t.episode_index == ep].iloc[0]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def read_frames(root: Path, cam: str, want_global: set[int]):
    """Decode specific GLOBAL frame indices from one camera's videos.

    The camera's mp4s are a rollover sequence, so the global index has to be
    walked file by file -- there is no per-file offset recorded that is worth
    trusting over simply counting."""
    got, i = {}, 0
    for p in sorted(glob.glob(str(root / "videos" / f"observation.images.{cam}" / "*" / "*.mp4"))):
        c = av.open(p)
        for f in c.decode(video=0):
            if i in want_global:
                got[i] = f.to_ndarray(format="bgr24")
            i += 1
            if len(got) == len(want_global):
                break
        c.close()
        if len(got) == len(want_global):
            break
    return got


def compose(per_cam: dict, cams, idx, ep, t):
    tiles = []
    for cam in cams:
        img = per_cam[cam].get(idx)
        if img is None:
            return None
        img = img.copy()
        for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, cam, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, th, cv2.LINE_AA)
        tiles.append(img)
    canvas = np.hstack(tiles)
    lab = f"ep {ep}  frame {t}"
    for col, th in (((0, 0, 0), 3), ((0, 255, 255), 1)):
        cv2.putText(canvas, lab, (6, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", help="one episode, or a from/to range")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    cams = [c for c in CAM_ORDER if f"observation.images.{c}" in info["features"]]
    print(f"[demos] {root}  cameras {cams}")

    tbl = episode_table(root)
    eps = list(tbl.episode_index.astype(int))
    if args.episodes:
        eps = ([args.episodes[0]] if len(args.episodes) == 1
               else [e for e in eps if args.episodes[0] <= e <= args.episodes[1]])
    done = set()
    if OUT.exists() and not args.redo:
        for line in open(OUT):
            if line.strip():
                d = json.loads(line)
                done.add((d["root"], d["episode"]))
    eps = [e for e in eps if (str(root), e) not in done]
    if args.limit:
        eps = eps[: args.limit]
    print(f"[demos] {len(eps)} episode(s) to label\n")

    pq = sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))
    act = np.concatenate([np.stack(pd.read_parquet(f, columns=["action"]).action.values) for f in pq])
    mmx = _fit()[0] * 10.0
    fo = open(OUT, "a")
    for n, ep in enumerate(eps, 1):
        try:
            a, b = frame_span(root, ep)
            g = act[a:b, -1]
            closed = g < g.max() - 0.5
            idx = np.flatnonzero(closed)
            close = int(idx[0]) if len(idx) else (b - a) // 3
            end = b - a - 1
            local = {0: "1_start", max(0, close - 50): "2_pre_grasp", close: "3_grasp",
                     min(end, close + 60): "4_lift", min(end, (close + end) // 2): "5_carry",
                     max(0, end - 120): "6_pre_place", end: "7_end"}
            glob_idx = {a + k: v for k, v in local.items()}
            per_cam = {c: read_frames(root, c, set(glob_idx)) for c in cams}
            with tempfile.TemporaryDirectory() as td:
                paths = []
                for gi in sorted(glob_idx):
                    canvas = compose(per_cam, cams, gi, ep, gi - a)
                    if canvas is None:
                        continue
                    p = Path(td) / f"{glob_idx[gi]}.png"
                    cv2.imwrite(str(p), scale_bar(canvas, cams, mmx))
                    paths.append(p)
                if len(paths) < 4:
                    raise ValueError(f"only {len(paths)} frames decoded")
                lab = ask(paths)
            row = {"root": str(root), "episode": ep, "source": "demonstration", **lab}
            fo.write(json.dumps(row) + "\n"); fo.flush()
            gq = lab.get("grasp") or {}; pl = lab.get("placement") or {}
            print(f"[{n}/{len(eps)}] ep{ep:03d}  app={(lab.get('approach') or {}).get('offset','?'):12s} "
                  f"grasp={str(gq.get('where_on_block')):9s} held={str(gq.get('held')):5s} "
                  f"rel={str(pl.get('released')):5s} fail={lab.get('primary_failure')}")
        except Exception as exc:
            print(f"[{n}/{len(eps)}] FAILED ep{ep}: {str(exc)[:120]}")
    fo.close()


if __name__ == "__main__":
    main()
