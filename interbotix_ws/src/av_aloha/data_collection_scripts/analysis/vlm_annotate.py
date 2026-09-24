"""Label recorded rollout episodes with a vision-language model.

WHAT THIS IS FOR.  The operator's own notes are the richest labels in the
project (292 of them) but they are prose, uneven in detail, and written from
memory at the end of an episode.  This asks a VLM the same questions in a
fixed schema for EVERY episode, so the failure modes can be counted instead of
grepped -- and so the subtask spans a language-conditioned policy needs can be
produced at scale.

HOW IT WORKS.  Five frames per episode, chosen from the trajectory rather than
by clock: the first frame, 0.8 s before the gripper first closes, the close
itself, midway through the carry, and the last frame.  Each frame is the
recorder's composite -- all three camera views side by side -- so the model
sees what the policy saw.  They go to `claude -p` (no API key needed; it uses
the local CLI session) with a schema it must fill.

WHAT IT IS NOT.  The model is not the ground truth.  It drafts; the operator
confirms.  Validate a batch against the stage flags and the placement detector
before trusting a run of hundreds -- `--validate` prints that comparison.

    python analysis/vlm_annotate.py --limit 12 --validate      # try a batch
    python analysis/vlm_annotate.py --policy rel2_rgb          # one policy
    python analysis/vlm_annotate.py                            # everything not yet done
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.placement_report import _fit  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "rollouts" / "vlm_labels.jsonl"

SCHEMA = """{
 "approach": {
   "offset": "on_target" | "too_left" | "too_right" | "too_high" | "too_low" | "too_far" | "from_outside",
   "quality": "direct" | "with_corrections" | "hesitant_shaky" | "never_reached",
   "far_from_block_at_close": true|false
 },
 "contact_before_grasp": {
   "happened": true|false,
   "finger": null | "left" | "right" | "both",
   "effect_on_block": null | "stayed_in_place" | "slid_flat" | "nudged_away" | "tipped_on_side" | "knocked_far",
   "moved_cm": null | <number>,
   "rotated_deg": null | <number, + = counter-clockwise>
 },
 "grasp": {
   "attempted": true|false,
   "height": null | "too_high" | "too_low" | "correct",
   "where_on_block": null | "edge" | "corner" | "bottom_half" | "centred",
   "held": true|false
 },
 "block_state": {
   "visible_in_wrist_camera": "always" | "mostly" | "briefly_lost" | "rarely",
   "ended_sideways": true|false,
   "long_side_along_gripper_vertical": null | true | false
 },
 "gripper_alignment": "aligned_with_block" | "aligned_with_target" | "misaligned" | "rotated_needlessly",
 "events": {
   "blocked_by_collision_gate": true|false,
   "drifted_from_block": true|false,
   "twisted_into_itself": true|false,
   "proceeded_empty": true|false,
   "dropped_block": true|false,
   "drop_result": null | "recovered" | "left_on_table" | "fell_on_side" | "out_of_reach",
   "upright_attempted": true|false,
   "upright_result": null | "succeeded" | "fell_again" | "gave_up" | "needed_second_attempt"
 },
 "corrections": {
   "attempted": true|false,
   "count": <integer>,
   "outcome": null | "improved" | "no_change" | "made_worse"
 },
 "placement": {
   "reached": true|false,
   "released": true|false,
   "offset": null | "ideal" | "too_high" | "too_low" | "too_left" | "too_right",
   "rotation_deg": null | <number, + = counter-clockwise>
 },
 "final_state": "<short phrase for where the block ended up>",
 "primary_failure": null | "never_approached" | "nudged_away" | "never_grasped" | "dropped" |
                    "tipped_on_side" | "never_released" | "stuck" | "drifted_away" | "gate_blocked",
 "note": "<one or two sentences, lab-notebook style, in the voice of an operator>",
 "unlisted_observations": [
   {"label": "<snake_case name you propose for this behaviour>",
    "what": "<one sentence describing it>",
    "frames": "<which of the numbered frames show it>",
    "interesting_because": "<why an operator studying this policy would care>"}
 ]
}"""

PROMPT = """You are labelling one episode of a robot manipulation rollout. Frames in time order:
{paths}

Each image is the policy's camera views side by side, labelled in the top-left of each tile:
low_scene (front view), right_wrist (the gripper's own camera, looking down the jaws),
top_scene (overhead). Task: a single right arm picks up a small blue four-lobed flower
block from the wooden table and places it on the printed blue clover target on the white
sheet, then lets go.

Conventions, so the labels mean what the operator means:
- "too high" / "too low" are along the VERTICAL OF THE WRIST CAMERA (what you see in that
  tile), not world height. Too high = the jaws closed above the block's mid-line.
- Left/right are as seen in the wrist tile.
- Rotation is in degrees, positive counter-clockwise, and the block is four-fold
  symmetric, so only report between -45 and +45.
- "proceeded empty" = the arm carried on toward the target with nothing in the gripper.
- "twisted into itself" = the wrist or forearm rolled far past any useful orientation.
- Distances in centimetres; the block is about 3.8 cm across, which is a useful ruler.

Judge only what the frames actually show. Use null where a field does not apply or you
cannot tell.

The schema is not exhaustive and is not meant to be. If the arm does something the fields
above have no place for -- an unexpected strategy, a recovery, a way of using contact, a
tell that it is uncertain, anything a careful observer would write down -- put it in
"unlisted_observations" and propose a name for it. Recurring proposals become new fields,
so name the behaviour, not the episode. Leave the list empty if nothing stands out.

A scale bar is drawn into the bottom-left of the overhead tile: the white/black bar is
5 cm, ticks every 1 cm. Use it for any distance you report.

Reply with ONLY this JSON, no prose:
{schema}"""


def keyframes(rec, video):
    """Five frames keyed to what happened, not to the clock."""
    rows = rec["rows"]
    g = np.array([x.get("right_gripper_cmd", 0.0) for x in rows])
    closed = g < g.max() - 0.5
    idx = np.flatnonzero(closed)
    close = int(idx[0]) if len(idx) else len(rows) // 3
    end = len(rows) - 1
    want = {0: "1_start", max(0, close - 50): "2_pre_grasp", close: "3_grasp",
            min(end, close + 60): "4_lift", min(end, (close + end) // 2): "5_carry",
            max(0, end - 120): "6_pre_place", end: "7_end"}
    got = {}
    c = av.open(str(video))
    for i, f in enumerate(c.decode(video=0)):
        if i in want:
            got[want[i]] = f.to_ndarray(format="bgr24")
        if i > max(want):
            break
    c.close()
    return got


def scale_bar(img, cams, mm_per_px_x):
    """Burn a 5 cm ruler into the overhead tile.

    A physical ruler on the table would also work, but it would change the
    scene the policies are fed -- top_scene is a policy camera and feeds the
    object detector, so anything added to the workspace is a distribution
    shift for every future rollout.  Drawing the bar afterwards costs nothing,
    cannot leak into training data, and is exact: the scale comes from the
    same px->metres fit the arm is driven with.
    """
    if "top_scene" not in cams:
        return img
    k = cams.index("top_scene")
    tw = img.shape[1] // len(cams)
    x0 = k * tw + 12
    y0 = img.shape[0] - 18
    px_cm = 1.0 / (mm_per_px_x / 10.0)            # pixels per centimetre
    L = int(round(5 * px_cm))
    cv2.rectangle(img, (x0 - 3, y0 - 13), (x0 + L + 3, y0 + 6), (0, 0, 0), -1)
    for i in range(5):
        a = x0 + int(round(i * px_cm)); b = x0 + int(round((i + 1) * px_cm))
        cv2.rectangle(img, (a, y0 - 4), (b, y0 + 1), (255, 255, 255) if i % 2 == 0 else (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0 - 4), (x0 + L, y0 + 1), (255, 255, 255), 1)
    cv2.putText(img, "5cm", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def ask(paths):
    prompt = PROMPT.format(paths=" ".join(str(p) for p in paths), schema=SCHEMA)
    p = subprocess.run(["claude", "-p", "--model", "claude-opus-5", "--allowedTools", "Read"],
                       input=prompt, capture_output=True, text=True, timeout=300)
    txt = p.stdout.strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {txt[:200]}")
    return json.loads(m.group(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy"); ap.add_argument("--run"); ap.add_argument("--limit", type=int)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    scores = {}
    for line in open(HERE / "rollouts" / "rollout_scores.jsonl"):
        if line.strip():
            r = json.loads(line)
            sd = r.get("snapshot_dir")
            if sd:
                scores[(Path(sd).name.replace("snapshots_rollout_", ""), r["episode"])] = r

    done = set()
    if OUT.exists() and not args.redo:
        for line in open(OUT):
            if line.strip():
                d = json.loads(line)
                done.add((d["run"], d["episode"]))

    todo = []
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
            if (run, r["episode"]) in done:
                continue
            ## Prefer the H.264 copy; fall back to the recorder's own path.
            ## Path("") resolves to "." and passes exists(), so the fallback
            ## has to be tested for emptiness FIRST -- otherwise av.open()
            ## is handed a directory.
            vid = HERE / "rollouts" / "h264" / f"rollout_rollout_{run}_ep{r['episode']:02d}.mp4"
            if not vid.is_file():
                alt = r.get("video") or ""
                vid = Path(alt) if alt else None
            if vid is None or not vid.is_file():
                continue
            r["_run"], r["_vid"], r["_job"] = run, vid, job
            todo.append(r)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} episode(s) to label ({len(done)} already done)\n")

    fo = open(OUT, "a")
    labels = []
    for n, r in enumerate(todo, 1):
        try:
            frames = keyframes(r, r["_vid"])
            with tempfile.TemporaryDirectory() as td:
                paths = []
                mmx = _fit()[0] * 10.0        # cm/px -> mm/px
                for k in sorted(frames):
                    p = Path(td) / f"{k}.png"
                    cv2.imwrite(str(p), scale_bar(frames[k], r.get("cameras") or [], mmx))
                    paths.append(p)
                lab = ask(paths)
            row = {"run": r["_run"], "episode": r["episode"], "job": r["_job"],
                   "cell": scores.get((r["_run"], r["episode"]), {}).get("cell"), **lab}
            fo.write(json.dumps(row) + "\n"); fo.flush()
            labels.append(row)
            gq=(lab.get("grasp") or {}); pl=(lab.get("placement") or {}); ev=(lab.get("events") or {})
            print(f"[{n}/{len(todo)}] {r['_job'][:26]:26s} {str(row['cell']):>3} "
                  f"app={(lab.get('approach') or {}).get('offset','?'):12s} "
                  f"grasp={str(gq.get('where_on_block')):9s} held={str(gq.get('held')):5s} "
                  f"rel={str(pl.get('released')):5s} gate={str(ev.get('blocked_by_collision_gate')):5s} "
                  f"fail={lab.get('primary_failure')}")
        except Exception as exc:
            print(f"[{n}/{len(todo)}] FAILED ep{r['episode']}: {str(exc)[:110]}")
    fo.close()

    if args.validate and labels:
        import cv2 as _cv
        sys.path.insert(0, str(HERE))
        from analysis.placement_report import detect_placed
        ag = {"grasp": [0, 0], "released": [0, 0]}
        for row in labels:
            rec = scores.get((row["run"], row["episode"]))
            if not rec:
                continue
            st = rec.get("stages") or {}
            ag["grasp"][1] += 1
            ag["grasp"][0] += int(bool(st.get("grasp")) == bool((row.get("grasp") or {}).get("held")))
            sd = rec.get("snapshot_dir")
            p = f"{sd}/ep{row['episode']:02d}_top_scene_end.png" if sd else None
            if p and Path(p).exists():
                det = detect_placed(_cv.imread(p))[4]
                ag["released"][1] += 1
                ag["released"][0] += int(det == bool((row.get("placement") or {}).get("released")))
        print("\nagreement with the existing labels:")
        for k, (a, b) in ag.items():
            src = "operator stage flag" if k == "grasp" else "placement detector"
            print(f"   {k:9s} {a}/{b} ({a/max(b,1):.0%})  vs the {src}")


if __name__ == "__main__":
    main()
