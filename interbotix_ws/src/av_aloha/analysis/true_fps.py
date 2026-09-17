"""Measure a dataset's ACTUAL capture rate, instead of believing its metadata.

    python true_fps.py --root old_dataset/lerobot/transfer_flower_mask/20260601_000935
    python true_fps.py --scan old_dataset/lerobot          # every run under here

WHY
===
Datasets recorded before ~2026-08 wrote a CONSTANT into `fps` rather than a
measurement.  The error is not small and not uniform:

    block_square/20260524_224911    claims 30 Hz, actually 0.63 Hz   (48x)
    transfer_flower/20260601_000935 claims 50 Hz, actually 2.25 Hz   (22x)
    block_square/20260528_131838    claims 50 Hz, actually 2.77 Hz   (18x)

The `timestamp` column does not help -- it is synthesised as index/fps, so it
inherits the lie exactly.  What survives is `observation.timestamps.robot`, a
real monotonic clock sampled per frame.  Its absolute value is meaningless (it
is boot-relative, not epoch) but its DIFFERENCES are true seconds, which is all
a rate needs.

WHY IT MATTERS FOR ROLLOUT
==========================
An ACT policy has no notion of time.  It emits absolute joint waypoints; the
only place a clock enters is the rate you execute them at.  Roll out at 50 Hz a
chunk whose waypoints were demonstrated 444 ms apart and the arm is asked to
perform the motion ~22x too fast -- it cannot track, it lags, and you get the
oscillation and undershoot that look like a policy failure and are not.

So the correct rollout rate is a property of the DATA, and this module is how
you get it.  Nothing here retrains anything: the checkpoints were always fine.

CAVEAT.  A recovered rate tells you how fast the demonstrations were, not
whether the rest of the world still matches them.  Camera poses and scene
layout drift; this cannot see that.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLOCK_COL = "observation.timestamps.robot"
## Episodes shorter than this carry too little span to estimate a rate from.
MIN_FRAMES = 6


def _clock(df: pd.DataFrame) -> np.ndarray | None:
    if CLOCK_COL not in df.columns:
        return None
    v = df[CLOCK_COL]
    first = v.iloc[0]
    arr = np.stack(v.to_numpy()) if hasattr(first, "__len__") else v.to_numpy()
    ## Some runs store a per-arm vector; every element is the same tick, so
    ## column 0 is as good as any.
    return np.asarray(arr, dtype=float).reshape(len(df), -1)[:, 0]


def measure(root: Path):
    """(median Hz, per-episode Hz, claimed fps) for one dataset run."""
    claimed = None
    for m in ("meta/info.json", "meta.json"):
        if (root / m).exists():
            try:
                claimed = json.loads((root / m).read_text()).get("fps")
            except Exception:
                pass
            if claimed:
                break

    files = sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))
    if not files:
        return None, [], claimed

    rates = []
    for f in files:
        df = pd.read_parquet(f, columns=[CLOCK_COL, "episode_index"]) \
            if CLOCK_COL in pd.read_parquet(f).columns else None
        if df is None:
            return None, [], claimed
        ts = _clock(df)
        if ts is None:
            return None, [], claimed
        ep = df["episode_index"].to_numpy()
        for e in sorted(set(ep.tolist())):
            m = ep == e
            if m.sum() < MIN_FRAMES:
                continue
            t = ts[m]
            span = float(t.max() - t.min())
            ## A frame count spans (n-1) intervals, not n.  At these rates the
            ## difference is a percent or two, but the point of this module is
            ## to stop being casual about rates.
            if span > 0:
                rates.append((m.sum() - 1) / span)
    if not rates:
        return None, [], claimed
    return float(np.median(rates)), rates, claimed


def report(root: Path) -> None:
    hz, rates, claimed = measure(root)
    print(f"\n{root}")
    if hz is None:
        print(f"  claimed fps : {claimed}")
        print(f"  measured    : unavailable -- no '{CLOCK_COL}' column.")
        print( "                Newer runs record timing in meta/robustness.jsonl instead.")
        return
    a = np.asarray(rates)
    print(f"  claimed fps : {claimed}")
    print(f"  MEASURED    : {hz:.2f} Hz   (median over {len(rates)} episodes; "
          f"p10 {np.percentile(a,10):.2f}, p90 {np.percentile(a,90):.2f})")
    if claimed:
        print(f"  overstated  : {claimed / hz:.1f}x")
    print(f"  frame period: {1.0/hz*1000:.0f} ms between demonstrated waypoints")
    print(f"\n  Roll out with:  --hz {hz:.2f}")
    print(f"  An episode of N frames then takes N/{hz:.2f} s "
          f"(e.g. 100 frames -> {100/hz:.0f} s, not {100/(claimed or 50):.0f} s).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="one dataset run directory")
    ap.add_argument("--scan", help="a directory of runs; report each")
    args = ap.parse_args()
    if not args.root and not args.scan:
        ap.error("give --root or --scan")

    if args.root:
        report(Path(args.root).resolve())
    if args.scan:
        base = Path(args.scan).resolve()
        runs = sorted({Path(f).parents[2] for f in
                       glob.glob(str(base / "**" / "data" / "*" / "*.parquet"),
                                 recursive=True)})
        print(f"{'dataset':<54s} {'claimed':>8s} {'measured':>9s} {'ratio':>7s}")
        print("-" * 82)
        for r in runs:
            hz, _, claimed = measure(r)
            name = str(r.relative_to(base))[-54:]
            if hz is None:
                print(f"{name:<54s} {str(claimed):>8s} {'--':>9s} {'--':>7s}")
            else:
                ratio = f"{claimed/hz:.1f}x" if claimed else "--"
                print(f"{name:<54s} {str(claimed):>8s} {hz:9.2f} {ratio:>7s}")


if __name__ == "__main__":
    main()
