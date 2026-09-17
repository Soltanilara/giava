"""Migrate our ACT checkpoints to lerobot v0.6.0's external-normalization format.

Why this wrapper exists
-----------------------
Upstream's `lerobot/processor/migrate_policy_normalization.py` derives feature names
from the old state-dict buffer keys with a naive `.replace("_", ".")`:

    normalize_inputs.buffer_observation_ee_pose.mean  ->  observation.ee.pose

Any feature whose name contains an underscore is mangled. All of our checkpoints use
`observation.ee_pose`, `observation.images.right_wrist` and `observation.images.top_scene`,
so every one of them is affected. The stats values are correct, but they get stored under
names the policy will never look up -- so normalization would silently not be applied.

This script runs the upstream migration and then repairs the tensor keys in the saved
normalizer/unnormalizer safetensors, using the (correct) feature names from config.json
as the source of truth.

Usage:
    python migrate_checkpoints.py --dry-run
    python migrate_checkpoints.py
"""

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from safetensors.torch import load_file, save_file

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LEROBOT = Path("/home/devi/giava/lerobot")
MIGRATE = LEROBOT / "src/lerobot/processor/migrate_policy_normalization.py"
SUFFIX = "_lerobot_v06"


def correct_names(ckpt_dir: Path) -> set[str]:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    names = set(cfg.get("input_features", {})) | set(cfg.get("output_features", {}))
    names.add("action")
    return names


def repair_keys(out_dir: Path) -> dict:
    """Rename mangled tensor keys back to their real feature names."""
    good = correct_names(out_dir)
    # mangled form -> correct form
    lookup = {n.replace("_", "."): n for n in good}
    report = {"files": {}, "renamed": 0, "unresolved": []}

    for st_path in sorted(out_dir.glob("*normalizer_processor.safetensors")):
        tensors = load_file(str(st_path))
        new_tensors = {}
        renamed = []
        for key, val in tensors.items():
            feat, _, stat = key.rpartition(".")
            if feat in good:
                new_tensors[key] = val  # already correct
            elif feat in lookup:
                new_key = f"{lookup[feat]}.{stat}"
                new_tensors[new_key] = val
                renamed.append((key, new_key))
            else:
                new_tensors[key] = val
                report["unresolved"].append(f"{st_path.name}:{key}")
        if renamed:
            save_file(new_tensors, str(st_path))
        report["files"][st_path.name] = len(renamed)
        report["renamed"] += len(renamed)
    return report


def verify(out_dir: Path) -> dict:
    """Load the migrated policy + preprocessor and confirm stats keys match config."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import PolicyProcessorPipeline

    ACTPolicy.from_pretrained(out_dir)
    pre = PolicyProcessorPipeline.from_pretrained(
        out_dir, config_filename="policy_preprocessor.json"
    )
    good = correct_names(out_dir)
    seen: set[str] = set()
    for step in pre.steps:
        if hasattr(step, "stats"):
            seen |= set(step.stats.keys())
    bad = sorted(k for k in seen if k not in good)
    return {"stats_keys": sorted(seen), "unexpected": bad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ckpts = sorted(p.parent for p in OUTPUTS.rglob("model.safetensors")
                   if not p.parent.name.endswith(SUFFIX)
                   and SUFFIX not in str(p.parent))
    if args.limit:
        ckpts = ckpts[: args.limit]

    print(f"Found {len(ckpts)} checkpoints under {OUTPUTS}")
    if args.dry_run:
        for c in ckpts:
            print("  ", c.relative_to(OUTPUTS))
        return 0

    results = []
    for i, ckpt in enumerate(ckpts, 1):
        rel = ckpt.relative_to(OUTPUTS)
        out_dir = ckpt.parent / f"{ckpt.name}{SUFFIX}"
        print(f"\n[{i}/{len(ckpts)}] {rel}", flush=True)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        try:
            proc = subprocess.run(
                [sys.executable, str(MIGRATE),
                 "--pretrained-path", str(ckpt),
                 "--output-dir", str(out_dir)],
                cwd=str(LEROBOT), capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print("   MIGRATE FAILED")
                print(proc.stderr[-1500:])
                results.append({"ckpt": str(rel), "status": "migrate_failed",
                                "error": proc.stderr[-2000:]})
                continue

            rep = repair_keys(out_dir)
            ver = verify(out_dir)
            status = "ok" if not ver["unexpected"] else "BAD_KEYS"
            print(f"   {status}: renamed {rep['renamed']} keys; "
                  f"stats={ver['stats_keys']}")
            results.append({"ckpt": str(rel), "status": status,
                            "renamed": rep["renamed"], "verify": ver})
        except Exception:
            traceback.print_exc()
            results.append({"ckpt": str(rel), "status": "error",
                            "error": traceback.format_exc()[-2000:]})

    report = OUTPUTS / "checkpoint_migration_report.json"
    report.write_text(json.dumps(results, indent=2))

    print("\n===== SUMMARY =====")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")
    print(f"report: {report}")
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
