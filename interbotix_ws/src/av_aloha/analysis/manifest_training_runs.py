"""Inventory every training run before its weights are deleted.

    python manifest_training_runs.py                     # summary to stdout
    python manifest_training_runs.py --csv runs.csv      # + a spreadsheet
    python manifest_training_runs.py --root ../../../../policy_training/outputs

WHAT THIS IS FOR
================
`outputs/` is ~266 GB of checkpoints trained on data whose recorded fps was
wrong (block_square claims 30 fps; the session's own timing logs measure ~1.7
Hz).  The weights are not worth keeping -- no amount of storage makes them mean
something they did not mean.  But the RECORD of what was tried is worth keeping
and is a few hundred kilobytes, so extract it first and delete second.

Everything here is read out of each run's own `config.json`, not guessed from
the directory name.  A run named `..._c10_a5` is only believed to be chunk 10 /
5 action steps if its config says so; where the name and the config disagree,
the manifest reports both and flags it, because that disagreement is exactly
the kind of thing that makes an old result unreproducible.

THE PAPER
=========
`271_Final_Report.pdf` (Amarsaikhan, Loddenkemper, Kincaid) reports six input
variants with their best training losses and rollout behaviour.  Those losses
live nowhere on disk -- these run directories carry no training log -- so they
are transcribed below and matched against runs by their input features.  That
matching is a HINT, not an identification: the paper does not record which
directory produced which row, so a match means "this run's inputs are
consistent with that variant", and the manifest says so rather than asserting
more than it knows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

## Table 2 of the report: representation -> best training loss.  Losses were
## tightly clustered in [0.064, 0.070] and, as the paper itself concludes, did
## NOT predict rollout behaviour -- runs with near-identical loss behaved very
## differently closed-loop.  Kept here so the manifest can carry the number
## without implying it means more than it does.
PAPER_VARIANTS = {
    1: ("RGB + blue mask",             0.0641),
    2: ("Masked RGB only",             0.0669),
    3: ("RGB + centroids",             0.0645),
    4: ("Centroids + vectors",         0.0640),
    5: ("RGB + blue mask + centroids", 0.0677),
    6: ("Masked RGB + centroids",      0.0694),
}

## Table 3's one-line verdicts, for the variants the paper singled out.
PAPER_ROLLOUT = {
    1: "high: missed block by ~7 cm; low: grasped, dropped before target",
    4: "high: grasped then stuck; low: STRONGEST -- placed within ~1-3 cm",
    5: "high: missed by ~4.5 cm (best high-start); low: released slightly early",
    6: "low: weakest -- failed grasp and stuck",
}

WEIGHT_SUFFIXES = (".pt", ".safetensors", ".bin", ".ckpt")


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}"
        x /= 1024


def classify(cfg: dict) -> str:
    """The representation variant, from the observation keys the run consumed.

    This is the one field the directory name is genuinely unreliable about --
    names were reused across reruns -- and it is the axis the whole study
    varied, so it is read from input_features every time."""
    feats = cfg.get("input_features") or {}
    keys = list(feats)
    imgs = [k for k in keys if k.startswith("observation.images.")]
    has_env = any(k.startswith("observation.environment_state") for k in keys)
    has_mask = any("mask" in k for k in imgs)
    bits = []
    if imgs:
        bits.append(f"{len(imgs)} cam" + ("s" if len(imgs) != 1 else ""))
    if has_mask:
        bits.append("mask")
    if has_env:
        n = feats.get("observation.environment_state", {}).get("shape")
        bits.append(f"env_state{tuple(n) if n else ''}")
    return " + ".join(bits) if bits else "?"


## The six-variant study's directory names ARE the variant, and they map 1:1
## onto Table 2.  The configs do not: every run reports the same two camera
## features, because the masks and centroids were packed into the image
## channels rather than declared as separate observation keys.  So for these
## runs the name is the only surviving record of which variant they are --
## which is precisely why this manifest has to exist before they are deleted.
_NAME_TO_ID = (
    ("masked_rgb_plus_centroids", 6),
    ("rgb_plus_blue_mask_plus_centroids", 5),
    ("centroids_plus_vectors", 4),
    ("rgb_plus_centroids", 3),
    ("masked_rgb_only", 2),
    ("rgb_plus_blue_mask", 1),
)


def paper_hint(cfg: dict, run: Path | None = None) -> str:
    """Which reported variant this run corresponds to.

    Name first (the six-variant dirs are unambiguous), then a weaker guess
    from the observation keys.  Returns "" rather than inventing a match."""
    if run is not None:
        stem = run.name.lower()
        ## Longest patterns first: "rgb_plus_blue_mask_plus_centroids" must not
        ## be swallowed by the "rgb_plus_blue_mask" prefix.
        for pat, i in sorted(_NAME_TO_ID, key=lambda kv: -len(kv[0])):
            if pat in stem:
                return f"ID {i} ({PAPER_VARIANTS[i][0]})"
    feats = cfg.get("input_features") or {}
    has_env = any(k.startswith("observation.environment_state") for k in feats)
    has_rgb = any(k.startswith("observation.images.") for k in feats)
    if has_env and not has_rgb:
        return "~ID 4 (centroids+vectors)"
    if has_env and has_rgb:
        return "~ID 3 (RGB+centroids)"
    return ""


def scan(root: Path):
    rows = []
    for cfg_path in sorted(root.rglob("config.json")):
        run = cfg_path.parent
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as exc:
            rows.append({"run": str(run.relative_to(root)), "error": str(exc)})
            continue
        weights = [p for p in run.rglob("*")
                   if p.is_file() and p.suffix in WEIGHT_SUFFIXES]
        total = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
        wbytes = sum(p.stat().st_size for p in weights)

        name = run.name
        chunk = cfg.get("chunk_size")
        ## Names encode `c<chunk>_a<n_action_steps>`; where that disagrees with
        ## the config, the config wins and the row is flagged.
        named_chunk = None
        for part in name.split("_"):
            if part.startswith("c") and part[1:].isdigit():
                named_chunk = int(part[1:])
        mismatch = (named_chunk is not None and chunk is not None
                    and named_chunk != chunk)

        rows.append({
            "run": str(run.relative_to(root)),
            "type": cfg.get("type"),
            "chunk_size": chunk,
            "n_action_steps": cfg.get("n_action_steps"),
            "kl_weight": cfg.get("kl_weight"),
            "use_vae": cfg.get("use_vae"),
            "lr": cfg.get("optimizer_lr"),
            "backbone": cfg.get("vision_backbone"),
            "representation": classify(cfg),
            "paper_hint": paper_hint(cfg, run),
            "action_dim": (cfg.get("output_features", {})
                           .get("action", {}).get("shape")),
            "total_bytes": total,
            "weight_bytes": wbytes,
            "n_weight_files": len(weights),
            "name_config_mismatch": mismatch,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).parent / "outputs"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    rows = [r for r in scan(root) if "error" not in r]
    if not rows:
        raise SystemExit(f"no config.json found under {root}")

    tot = sum(r["total_bytes"] for r in rows)
    wt = sum(r["weight_bytes"] for r in rows)
    print(f"\n{len(rows)} runs under {root}")
    print(f"total {human(tot)}, of which {human(wt)} is weights "
          f"({wt / max(1, tot) * 100:.0f}%)\n")

    hdr = (f"{'run':<48s} {'chunk':>5s} {'act':>4s} {'kl':>5s} {'vae':>5s} "
           f"{'representation':<26s} {'size':>8s}  paper")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["run"]):
        flag = " !" if r["name_config_mismatch"] else ""
        print(f"{r['run'][:48]:<48s} {str(r['chunk_size']):>5s} "
              f"{str(r['n_action_steps']):>4s} {str(r['kl_weight']):>5s} "
              f"{str(r['use_vae']):>5s} {r['representation'][:26]:<26s} "
              f"{human(r['total_bytes']):>8s}  {r['paper_hint']}{flag}")

    bad = [r for r in rows if r["name_config_mismatch"]]
    if bad:
        print(f"\n! {len(bad)} run(s) whose directory name disagrees with their "
              f"config's chunk_size -- the config is authoritative.")

    print("\nReported variants (271_Final_Report.pdf, Table 2 -- not on disk):")
    for i, (nm, loss) in sorted(PAPER_VARIANTS.items()):
        note = PAPER_ROLLOUT.get(i, "")
        print(f"  ID {i}  {nm:<30s} train loss {loss:.4f}"
              + (f"\n         {note}" if note else ""))
    print("\n  The paper's own conclusion: loss did not predict rollout "
          "behaviour.\n  Runs within 0.006 of each other behaved very "
          "differently closed-loop,\n  so these numbers are provenance, not a "
          "ranking.")

    if args.csv:
        out = Path(args.csv)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out}  ({len(rows)} rows) -- keep this, drop the weights")
    print(f"\nDeleting weight files alone would free {human(wt)}.")


if __name__ == "__main__":
    main()
