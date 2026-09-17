"""Unblind a scene-paired A/B rollout and test it.

    python ab_report.py --since 2026-09-10
    python ab_report.py --since 2026-09-10 --stage placement

Reads rollouts/rollout_scores.jsonl, keeps the records that carry an
`ab_arm` (written by rollout_policy.py --checkpoint-b), pairs them by
`scene_index`, and reports per stage:

  * each arm's success count with a Wilson 95% interval
  * the 2x2 table of DISCORDANT scenes -- the only ones carrying information
  * an exact (binomial) McNemar p-value

WHY McNEMAR AND NOT TWO PROPORTIONS.  The scenes are not a random sample of
some population; they are a fixed set of placements, and both policies saw
each one. Comparing two independent intervals throws that away and needs far
more trials to see the same effect. McNemar asks the only question the design
supports: among the scenes where the two policies DISAGREED, did one win more
often than a coin would?

WHY THE PAIRING HAS TO BE WITHIN ONE SESSION.  This rig's own logs settle it:
the same checkpoint, on scenes replicated to a median 0.7 px, scored 8/17 on
2026-09-09 and 4/17 on 2026-09-10. A comparison against another day's numbers
measures the day. `scene_index` pairs are minutes apart, not days.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", default=None)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD")
    ap.add_argument("--stage", default=None,
                    help="only this stage (default: every stage present)")
    args = ap.parse_args()

    here = Path(__file__).parent
    path = Path(args.scores or here / "rollouts" / "rollout_scores.jsonl")
    cutoff = (dt.datetime.strptime(args.since, "%Y-%m-%d").timestamp()
              if args.since else None)

    recs = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("ab_arm") not in ("a", "b"):
            continue
        if cutoff and r.get("wall_time", 0) < cutoff:
            continue
        recs.append(r)
    if not recs:
        raise SystemExit("no A/B records found (run rollout_policy.py "
                         "--checkpoint-b, or widen --since)")

    ckpt = {}
    for r in recs:
        ckpt.setdefault(r["ab_arm"], r["checkpoint"])
    print(f"{len(recs)} A/B episodes"
          + (f" since {args.since}" if args.since else ""))
    for a in sorted(ckpt):
        print(f"  arm {a.upper()} = {ckpt[a]}")
    if any(r.get("ab_blinded") for r in recs):
        print("  (scored blind)")

    ## scene_index -> arm -> record.  A scene the session did not finish both
    ## halves of carries no comparison and is dropped, loudly.
    scenes = defaultdict(dict)
    for r in recs:
        scenes[r.get("scene_index")][r["ab_arm"]] = r
    complete = {k: v for k, v in scenes.items() if len(v) == 2}
    partial = sorted(k for k, v in scenes.items() if len(v) != 2)
    if partial:
        print(f"  dropped {len(partial)} scene(s) with only one arm: {partial}")
    if not complete:
        raise SystemExit("no scene has both arms")

    stages = ([args.stage] if args.stage else
              list(next(iter(complete.values()))["a"]["stages"].keys()))
    n = len(complete)
    print(f"\n{n} paired scenes\n")
    for st in stages:
        ka = sum(1 for v in complete.values() if v["a"]["stages"].get(st))
        kb = sum(1 for v in complete.values() if v["b"]["stages"].get(st))
        both = only_a = only_b = neither = 0
        flips = []
        for sc, v in sorted(complete.items()):
            x = bool(v["a"]["stages"].get(st))
            y = bool(v["b"]["stages"].get(st))
            both += x and y
            neither += (not x) and (not y)
            only_a += x and not y
            only_b += y and not x
            if x != y:
                flips.append(f"{sc}{'A' if x else 'B'}")
        p = mcnemar_exact(only_a, only_b)
        la, ha = wilson(ka, n)
        lb, hb = wilson(kb, n)
        print(f"  {st}")
        print(f"    A {ka:>2}/{n}  ({ka / n:.0%}, 95% CI {la:.0%}-{ha:.0%})")
        print(f"    B {kb:>2}/{n}  ({kb / n:.0%}, 95% CI {lb:.0%}-{hb:.0%})")
        print(f"    both {both}   neither {neither}   "
              f"only A {only_a}   only B {only_b}")
        print(f"    exact McNemar p = {p:.3f}"
              + ("   <- the discordant scenes are one-sided enough to matter"
                 if p < 0.05 else
                 "   (not separable at this trial count)"))
        if flips:
            print(f"    scenes that flipped (winner): {' '.join(flips)}")
        ## What this many scenes could ever have shown.
        need = 0
        while mcnemar_exact(need, 0) > 0.05 and need < 100:
            need += 1
        print(f"    note: with {only_a + only_b} discordant scene(s), even a "
              f"clean sweep needs >= {need} to reach p<0.05\n")


if __name__ == "__main__":
    main()
