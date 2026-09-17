"""Aggregate rollout_scores.jsonl into a per-condition comparison table.

    python summarize_rollouts.py                      # all conditions
    python summarize_rollouts.py --since 2026-09-06   # today's only

Reports a Wilson 95% interval beside every rate.  At n=10 a 7/10 and a 5/10
overlap heavily, and the whole point of the earlier finding -- two runs of an
IDENTICAL model scoring "missed by 7.5 cm" and "best in the study" -- is that
a bare success count invites conclusions the sample cannot support.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

## Stage columns come from the records themselves (rollout_policy.py writes a
## scene-specific stage set: flower = approach/grasp/transport/placement,
## shape_sorter = approach/grasp/transport/hole/inserted), so a table never
## shows a column no record in it could have scored.


def wilson(k, n, z=1.96):
    """95% CI for a proportion; sane at n=10 where normal approx is not."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def condition(rec):
    """job-dir name + checkpoint step, e.g. 'act_envstate_50ep @ 050000'."""
    parts = Path(rec["checkpoint"]).parts
    job = step = "?"
    if "train" in parts:
        i = parts.index("train")
        if i + 1 < len(parts):
            job = parts[i + 1]
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        if i + 1 < len(parts):
            step = parts[i + 1]
    ncam = len(rec.get("cameras") or [])
    tgt = f" target={rec['target']}" if rec.get("target") else ""
    return f"{job} @ {step} ({ncam}cam){tgt}"


def stages_of(recs):
    """Stage names in scoring order, as the first record lists them."""
    return list(recs[0]["stages"].keys())


def final_dist(rec):
    """Object->target distance in top_scene px: the generic key when the
    scene wrote one, else the flower-era key."""
    a = rec.get("auto") or {}
    for k in ("final_dist_target_px", "final_dist_flower_px"):
        if a.get(k) is not None:
            return a[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=None,
                    help="rollout_scores.jsonl (default: rollouts/ beside this script)")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD filter")
    ap.add_argument("--notes", action="store_true", help="print every note")
    args = ap.parse_args()

    path = Path(args.scores or Path(__file__).parent / "rollouts" /
                "rollout_scores.jsonl")
    if not path.exists():
        raise SystemExit(f"{path} not found -- roll out with --score first")

    import datetime as dt
    cutoff = None
    if args.since:
        cutoff = dt.datetime.strptime(args.since, "%Y-%m-%d").timestamp()

    groups = defaultdict(list)
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if cutoff and rec.get("wall_time", 0) < cutoff:
            continue
        groups[condition(rec)].append(rec)

    if not groups:
        raise SystemExit("no records matched")

    ## One table per stage set, so flower and shape-sorter rollouts in the
    ## same scores file do not share a header that fits neither.
    by_stages = defaultdict(dict)
    for cond, recs in groups.items():
        by_stages[tuple(stages_of(recs))][cond] = recs

    w = max(len(k) for k in groups) + 2
    for stages, conds in by_stages.items():
        has_wrong = any("wrong_piece" in r for rs in conds.values() for r in rs)
        head = f"{'condition':<{w}} {'n':>3}  " + "  ".join(
            f"{s[:5]:>13}" for s in stages) + f"  {'dist_px':>9}  {'reached':>7}" + (
            f"  {'wrong':>5}" if has_wrong else "")
        print(head)
        print("-" * len(head))
        for cond in sorted(conds):
            recs = conds[cond]
            n = len(recs)
            cells = []
            for stage in stages:
                k = sum(1 for r in recs if r["stages"].get(stage))
                lo, hi = wilson(k, n)
                cells.append(f"{k}/{n} {lo:.0%}-{hi:.0%}".rjust(13))
            d = [final_dist(r) for r in recs if final_dist(r) is not None]
            dtxt = f"{st.median(d):.0f}" if d else "-"
            reach = st.mean([r.get("furthest", 0) for r in recs])
            wrong = sum(1 for r in recs if r.get("wrong_piece"))
            print(f"{cond:<{w}} {n:>3}  " + "  ".join(cells) +
                  f"  {dtxt:>9}  {reach:>7.1f}" +
                  (f"  {wrong:>5}" if has_wrong else ""))
        print(f"reached = mean furthest stage, 0-{len(stages)}"
              + ("; wrong = episodes that grasped a non-target piece" if has_wrong else "")
              + "\n")

    print("dist_px = median final object->target distance (top_scene pixels)")

    if args.notes:
        for cond in sorted(groups):
            notes = [(r["episode"], r["note"]) for r in groups[cond]
                     if r.get("note")]
            if notes:
                print(f"\n{cond}")
                for ep, note in notes:
                    print(f"  ep{ep:>2}: {note}")


if __name__ == "__main__":
    main()
