"""Split a recorded rollout into one H.264 video per camera, upscaled.

The recorder writes ONE composite mp4 per episode: the policy's cameras tiled
side by side at --video-scale (0.5 by default), so each tile is 320x240 inside
a 960x240 or 1280x240 frame.  That is the right format for seeing what the
policy saw all at once, and the wrong one for studying a single view -- a
320x240 wrist tile in a corner of a wide strip is hard to read.

This crops each tile out and writes it as its own file, upscaled, in H.264 so
it plays in VS Code (OpenCV writes mp4v, which VS Code cannot decode).  The
camera order comes from the episode's own log record, never guessed.

    python analysis/split_cameras.py --policy rel2 --cell A1
    python analysis/split_cameras.py --log rollouts/rollout_..._.jsonl --ep 3
    python analysis/split_cameras.py --policy wristmask --cameras right_wrist --scale 3
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "rollouts" / "cameras"


def episodes(log_glob="rollouts/rollout_*.jsonl"):
    for lp in sorted(glob.glob(str(HERE / log_glob))):
        if Path(lp).name == "rollout_scores.jsonl":
            continue
        for line in open(lp):
            if line.strip():
                r = json.loads(line)
                if r.get("rows") is not None or "cameras" in r:
                    r["_log"] = lp
                    yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", help="substring of the checkpoint job name, e.g. rel2, wristmask")
    ap.add_argument("--cell", help="placement label, e.g. A1")
    ap.add_argument("--log"); ap.add_argument("--ep", type=int)
    ap.add_argument("--cameras", nargs="*", default=None, help="default: every camera the policy saw")
    ap.add_argument("--scale", type=float, default=2.0, help="upscale factor on the tile")
    ap.add_argument("--limit", type=int, default=8, help="max episodes to split in one call")
    args = ap.parse_args()

    picks = []
    for r in episodes():
        if args.log and Path(r["_log"]).name != Path(args.log).name:
            continue
        if args.ep is not None and r.get("episode") != args.ep:
            continue
        if args.policy and args.policy not in (r.get("checkpoint") or ""):
            continue
        if args.cell and str(r.get("cell")) != args.cell:
            continue
        picks.append(r)
    if not picks:
        raise SystemExit("no episode matched -- check --policy / --cell / --log")
    picks = picks[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    for r in picks:
        cams = r.get("cameras") or []
        vid = r.get("video")
        if not vid:
            stem = Path(r["_log"]).with_suffix("")
            vid = f"{stem}_ep{r['episode']:02d}.mp4"
        if not Path(vid).exists():
            print(f"[skip] ep{r['episode']:02d}: no video at {vid}")
            continue
        want = args.cameras or cams
        job = Path(r.get("checkpoint") or "?").parts
        job = job[-4] if len(job) >= 4 else "unknown"
        for cam in want:
            if cam not in cams:
                print(f"[skip] {cam} is not one of this episode's cameras {cams}")
                continue
            k = cams.index(cam)
            name = f"{job}_{r.get('cell') or 'x'}_ep{r['episode']:02d}_{cam}.mp4"
            out = OUT / name
            ## in_w/in_h are the composite's dimensions; each tile is 1/len(cams) wide
            vf = (f"crop=iw/{len(cams)}:ih:{k}*iw/{len(cams)}:0,"
                  f"scale=iw*{args.scale}:ih*{args.scale}:flags=neighbor")
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", vid,
                   "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", str(out)]
            if subprocess.run(["nice", "-n", "15"] + cmd).returncode == 0:
                print(f"[ok] {name}  ({out.stat().st_size/1e6:.1f} MB)")
            else:
                print(f"[FAIL] {name}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
