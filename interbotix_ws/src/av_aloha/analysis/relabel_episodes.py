"""Correct the task string on episodes the recorder mislabelled.

    python relabel_episodes.py --root <run_dir> --episodes 140-161 \
        --to cube                       # dry run: prints, changes nothing
    python relabel_episodes.py ... --apply

WHY THIS EXISTS.  data_collection.py steps `--target` through a round-robin
of the pieces after every save, and stamps each frame with the task string
for whatever the target currently is.  If the operator does not follow the
round -- inserting the cube every time while varying only which distractors
are on the table -- the frames carry a piece name that does not describe what
the arm did.  That happened to shape_sorter/20260907_204429 episodes 140-161
on 2026-09-10: every one is a cube insertion, but the recorder labelled 15 of
them triangle or flower.

WHAT IT BREAKS IF LEFT ALONE.  ACT never reads the task string, so an ACT run
is only affected through episode SELECTION (`train_real.py --pieces`), which
is why the cube fine-tune silently trained on 7 corrections instead of 21.
A language-conditioned policy -- SmolVLA, pi0 -- reads it every step, and
would be taught that "insert the red triangle" means pick up the cube.

WHAT IT CHANGES.  The `task_index` column of the data parquet for those
episodes, the `tasks` column of meta/episodes/*, and the `task` field of
episode_outcomes.jsonl.  Adds the target string to meta/tasks.parquet if it
is not already there.  Never touches observations or actions.  Writes
meta/relabel_<date>.json recording the previous label of every episode it
changed, so the edit can be undone.

Re-derive any canonicalized copy (canonicalize_waist.py) afterwards -- it
holds its own copy of meta/.
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import datetime as dt
import glob
import json
import sys
from pathlib import Path

import pandas as pd


def parse_episodes(spec):
    out = set()
    for part in spec.replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="dataset run directory")
    ap.add_argument("--episodes", required=True,
                    help="indices, e.g. '140-161' or '141,142,144'")
    ap.add_argument("--to", required=True,
                    help="piece name (cube/triangle/flower) or a full task string")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this is a dry run")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    eps = parse_episodes(args.episodes)

    sys.path.insert(0, str(Path(__file__).parent))
    import shape_sorter as ss
    task = ss.task_string(args.to) if args.to in ss.CLASSES else args.to

    tasks_path = root / "meta" / "tasks.parquet"
    tdf = pd.read_parquet(tasks_path).reset_index()
    known = {str(r.task): int(r.task_index) for r in tdf.itertuples()}
    if task not in known:
        raise SystemExit(f"task string {task!r} is not in {tasks_path}\n"
                         f"  known: {list(known)}")
    tidx = known[task]
    print(f"[root] {root}")
    print(f"[to]   {task!r}  (task_index {tidx})")
    print(f"[eps]  {len(eps)}: {eps}\n")

    ## What is there now, so the change is inspectable before it happens.
    outc_path = root / "episode_outcomes.jsonl"
    if not outc_path.exists():
        outc_path = root / "meta" / "episode_outcomes.jsonl"
    before = {}
    if outc_path.exists():
        for line in outc_path.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                before[int(r["episode_index"])] = r.get("task")
    changing = [e for e in eps if before.get(e) not in (None, task)]
    print(f"{'ep':>5}  {'current label':<48} -> new")
    for e in eps:
        cur = before.get(e, "(no outcome record)")
        mark = "  " if cur == task else "->"
        print(f"{e:>5}  {str(cur)[:48]:<48} {mark} "
              f"{'(unchanged)' if cur == task else task}")
    print(f"\n{len(changing)} episode(s) would change label.")
    if not args.apply:
        print("\nDRY RUN -- nothing written.  Re-run with --apply.")
        return

    undo = {"when": dt.datetime.now().isoformat(timespec="seconds"),
            "root": str(root), "to": task, "previous": {str(e): before.get(e)
                                                        for e in eps}}

    ## 1. frame-level task_index
    for f in sorted(glob.glob(str(root / "data" / "*" / "*.parquet"))):
        df = pd.read_parquet(f)
        m = df["episode_index"].isin(eps)
        if not m.any():
            continue
        df.loc[m, "task_index"] = tidx
        df.to_parquet(f, index=False)
        print(f"[data] {Path(f).name}: {int(m.sum())} frames -> task_index {tidx}")

    ## 2. per-episode meta
    for f in sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet"))):
        df = pd.read_parquet(f)
        if "tasks" not in df.columns:
            continue
        m = df["episode_index"].isin(eps)
        if not m.any():
            continue
        df.loc[m, "tasks"] = df.loc[m, "tasks"].apply(lambda _: [task])
        df.to_parquet(f, index=False)
        print(f"[meta] {Path(f).name}: {int(m.sum())} episode rows")

    ## 3. outcome records -- append, last record wins (dataset.py convention)
    if outc_path.exists():
        with outc_path.open() as fh:
            recs = [json.loads(l) for l in fh if l.strip()]
        last = {int(r["episode_index"]): r for r in recs}
        with outc_path.open("a") as fh:
            for e in eps:
                if e in last:
                    r = dict(last[e])
                    r["task"] = task
                    r["relabelled"] = True
                    fh.write(json.dumps(r) + "\n")
        print(f"[outcomes] appended {len(eps)} corrected record(s)")

    undo_path = root / "meta" / f"relabel_{dt.date.today():%Y%m%d}.json"
    undo_path.write_text(json.dumps(undo, indent=2))
    print(f"\n[undo] previous labels recorded in {undo_path}")
    print("Re-run canonicalize_waist.py if a canon copy of this run exists.")


if __name__ == "__main__":
    main()
