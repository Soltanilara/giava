"""Change an episode's success/failure label after it was recorded.

The label is written during the session by the ss / sf keys, when the operator
has half a second to judge and the arms are still live.  Reviewing the video
afterwards is when you actually find out -- the grasp that looked clean and
dropped the block two frames later, the "failure" that was really the operator
fumbling the controller.  So the label has to be changeable, and changing it
must not touch the recorded frames.

episode_outcomes.jsonl is append-only and LAST RECORD WINS (dataset.load_outcomes).
Relabelling appends; nothing is rewritten, nothing is lost, and the history of
what an episode was called stays readable.
"""

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

from pathlib import Path
import argparse
import json
import time

from dataset import load_outcomes


def episode_lengths(root):
    """episode_index -> length, from the dataset's own metadata (or {})."""
    try:
        import pandas as pd
        files = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            return {}
        df = pd.concat([pd.read_parquet(f) for f in files])
        return {int(r.episode_index): int(r.length) for r in df.itertuples()}
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=str, help="Dataset run folder")
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--outcome", choices=["success", "failure", "unknown"],
                   default=None)
    p.add_argument("--note", type=str, default=None,
                   help="Why it changed, kept with the record.")
    p.add_argument("--list", action="store_true",
                   help="Show current labels and exit.")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"no such dataset: {root}")

    outcomes = load_outcomes(root)
    lengths = episode_lengths(root)

    if args.list or args.episode is None:
        if not outcomes:
            raise SystemExit(f"{root} has no episode_outcomes.jsonl")
        n_ok = sum(1 for r in outcomes.values() if r["outcome"] == "success")
        print(f"{root}\n{len(outcomes)} labelled episode(s): "
              f"{n_ok} success, {len(outcomes) - n_ok} not\n")
        for idx in sorted(outcomes):
            r = outcomes[idx]
            mark = "*" if r.get("relabelled") else " "
            note = f"  [{r['note']}]" if r.get("note") else ""
            print(f" {mark} episode_{idx:04d}  {r['outcome']:8s} "
                  f"{r.get('num_frames', '?'):>6} frames{note}")
        if any(r.get("relabelled") for r in outcomes.values()):
            print("\n * = relabelled after the session")
        if args.episode is None and not args.list:
            print("\nPass --episode N --outcome success|failure to change one.")
        return

    if args.outcome is None:
        raise SystemExit("--episode needs --outcome")

    ## Refuse to label an episode the dataset does not have: a typo'd index
    ## writes a record that silently never matches anything, and the episode
    ## you meant to fix keeps its wrong label.
    if lengths and args.episode not in lengths:
        raise SystemExit(
            f"episode {args.episode} is not in this dataset "
            f"(it has {sorted(lengths)}). Nothing written.")

    prev = outcomes.get(args.episode)
    record = {
        "episode_index": args.episode,
        "outcome": args.outcome,
        "num_frames": (prev or {}).get("num_frames", lengths.get(args.episode)),
        "task": (prev or {}).get("task"),
        "wall_time": time.time(),
        "relabelled": True,
        "previous_outcome": (prev or {}).get("outcome"),
    }
    if args.note:
        record["note"] = args.note

    with open(root / "episode_outcomes.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    was = prev["outcome"] if prev else "unlabelled"
    print(f"episode_{args.episode:04d}: {was} -> {args.outcome}")


if __name__ == "__main__":
    main()
