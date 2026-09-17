"""Package capture sessions for transfer to another machine.

Why not git: the sessions tree is ~150 MB of PNGs and grows every run.
Committing it to a shared repo doubles the clone size for everyone,
permanently -- git history is not prunable in practice.  These are
experiment OUTPUTS: transfer them, do not version them.

What this does: builds one self-describing tar.gz with a manifest, so the
receiving end knows what is inside without unpacking -- which sessions,
which cameras and serials, how many shots, when, and whether the poses were
recorded.

    python calibration/export_sessions.py --list
    python calibration/export_sessions.py --all
    python calibration/export_sessions.py --sessions multipose_20260818_2118
    python calibration/export_sessions.py --all --out /tmp/giava_sessions.tar.gz

Then move it however you like:

    scp <archive> user@host:/path/
    rsync -avP <archive> user@host:/path/

and unpack with:  tar xzf <archive>
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import DATA_ROOT, provenance  # noqa: E402

SESSIONS = DATA_ROOT / "sessions"


def describe(d: Path) -> Dict[str, Any]:
    """Summarise one session without unpacking anything."""
    imgs = list((d / "images").glob("*.png")) if (d / "images").is_dir() else []
    info: Dict[str, Any] = {
        "session": d.name,
        "n_images": len(imgs),
        "bytes": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()),
        "has_session_json": (d / "session.json").exists(),
        "has_summary": (d / "summary.txt").exists(),
    }
    sj = d / "session.json"
    if sj.exists():
        try:
            doc = json.loads(sj.read_text())
            info["n_shots"] = len(doc.get("shots", []))
            info["n_poses"] = doc.get("n_poses")
            info["recorded"] = doc.get("metadata", {}).get("date")
            ## metadata.cameras is a list of dicts from the CLI tool but a
            ## list of plain names from the world-view capture. Handle both
            ## -- assuming dicts raised AttributeError on the string form,
            ## which aborted the rest of this block and made every
            ## world-view session report 0 poses when the poses were there.
            cams = doc.get("metadata", {}).get("cameras", [])
            info["cameras"] = [
                {"name": c.get("name"), "serial": c.get("serial")}
                if isinstance(c, dict) else {"name": c}
                for c in cams
            ] or sorted(doc.get("intrinsics", {}))
            ## Whether the robot pose came along is the difference between
            ## "images" and "images you can compute extrinsics from".
            shots = doc.get("shots", [])
            info["shots_with_pose"] = sum(
                1 for s in shots if s.get("robot_pose"))
            info["has_intrinsics"] = bool(doc.get("intrinsics"))
        except Exception as exc:
            ## Never let a summariser bug masquerade as missing data.
            info["session_json_error"] = f"{type(exc).__name__}: {exc}"
            info["WARNING"] = (f"could not summarise session.json "
                               f"({type(exc).__name__}) -- the FILE may "
                               f"still be fine; this is a reader problem")
    else:
        ## A session with images but no session.json lost its timestamps and
        ## pose -- worth flagging loudly rather than shipping silently.
        info["WARNING"] = ("no session.json -- images only, no timestamps, "
                           "no pose, no intrinsics")
    return info


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="show what is available and exit")
    ap.add_argument("--all", action="store_true", help="export every session")
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="session directory names (prefixes are enough)")
    ap.add_argument("--out", default=None, help="output .tar.gz path")
    ap.add_argument("--dir", default=str(SESSIONS),
                    help="sessions root (default: calibration/data/sessions)")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"  no sessions directory at {root}")
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not dirs:
        raise SystemExit(f"  no sessions in {root}")

    infos = [describe(d) for d in dirs]
    print()
    print(f"  {len(dirs)} session(s) in {root}\n")
    print(f"  {'session':38s} {'shots':>6s} {'imgs':>6s} {'pose':>6s} "
          f"{'size':>9s}")
    print("  " + "-" * 72)
    for d, i in zip(dirs, infos):
        if not i.get("has_session_json"):
            pose = "--"
        elif "shots_with_pose" not in i:
            pose = "?"          # could not read, NOT "none present"
        else:
            pose = f"{i['shots_with_pose']}/{i.get('n_shots', 0)}"
        flag = "  <-- " + i["WARNING"] if "WARNING" in i else ""
        print(f"  {d.name:38s} {i.get('n_shots', 0):>6} {i['n_images']:>6} "
              f"{pose:>6s} {human(i['bytes']):>9s}{flag}")
    print("  " + "-" * 72)
    print(f"  total {human(sum(i['bytes'] for i in infos))}")

    if args.list or (not args.all and not args.sessions):
        if not args.list:
            print("\n  pick with --all or --sessions <name> [...]")
        print()
        return

    if args.all:
        chosen = dirs
    else:
        chosen = [d for d in dirs
                  if any(d.name.startswith(p) for p in args.sessions)]
        missing = [p for p in args.sessions
                   if not any(d.name.startswith(p) for d in dirs)]
        if missing:
            raise SystemExit(f"\n  no session matches {missing}")
    if not chosen:
        raise SystemExit("  nothing selected")

    out = Path(args.out) if args.out else (
        root.parent / f"giava_sessions_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz")
    manifest = {
        "metadata": provenance("session_export"),
        "n_sessions": len(chosen),
        "total_bytes": sum(i["bytes"] for d, i in zip(dirs, infos)
                           if d in chosen),
        "sessions": [i for d, i in zip(dirs, infos) if d in chosen],
        "note": ("Capture sessions: paired images with per-shot timestamps, "
                 "camera intrinsics, and the robot pose at each shot. "
                 "Camera extrinsics inside are derived from the mount "
                 "transform in effect when recorded -- check "
                 "mount_provenance before trusting them."),
    }

    print(f"\n  packing {len(chosen)} session(s) -> {out}")
    mpath = root.parent / "_export_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    try:
        with tarfile.open(out, "w:gz") as tar:
            tar.add(mpath, arcname="MANIFEST.json")
            for d in chosen:
                tar.add(d, arcname=f"sessions/{d.name}")
                print(f"    + {d.name}")
    finally:
        mpath.unlink(missing_ok=True)

    size = out.stat().st_size
    print(f"\n  {human(size)}  ->  {out}")
    print(f"""
  Move it:
      scp {out} user@host:/path/
      rsync -avP {out} user@host:/path/

  Unpack on the other end:
      tar xzf {out.name}          # gives sessions/ + MANIFEST.json

  MANIFEST.json lists what is inside without unpacking.
""")


if __name__ == "__main__":
    main()
