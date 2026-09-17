"""Where everything lives, resolved once, relative to the repo.

WHY THIS EXISTS
===============
Every path in this tree used to be spelled `/home/devi/giava/...`.  That is
fine until the repo is cloned anywhere else, moved, or run by a second person,
at which point the failures are not import errors but silent wrong behaviour:
`arm_config.URDF_PATH` pointing at a URDF that no longer exists gives a stack
trace, but pointing at a STALE one gives you a robot model that is quietly not
the robot in front of you.

FINDING THE ROOT.  Not by counting `parents[N]` -- that silently breaks the day
a file moves one directory, and it was already spelled two different ways in
this tree, at two different depths.
Instead walk up looking for a marker that only the repo root has.  Override
with GIAVA_ROOT when the layout is genuinely unusual (a worktree, a container
mount).
"""

from __future__ import annotations

import os
from pathlib import Path

## giava.urdf is the marker: it sits at the repo root, nothing else is named
## that, and it is the file most of these paths are ultimately about.
_MARKER = "giava.urdf"


def _find_root() -> Path:
    env = os.environ.get("GIAVA_ROOT")
    if env and str(env).strip():
        root = Path(env).expanduser().resolve()
        if not (root / _MARKER).exists():
            raise SystemExit(
                f"GIAVA_ROOT={root} does not contain {_MARKER} -- that is not "
                f"the giava repo root.")
        return root
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / _MARKER).exists():
            return cand
    raise SystemExit(
        f"could not find {_MARKER} above {here}. Set GIAVA_ROOT to the repo "
        f"root if this file has been moved out of the tree.")


REPO_ROOT = _find_root()

URDF_PATH = REPO_ROOT / _MARKER
## ONE assets location for the whole repo (moved out of
## data_collection_scripts/ on 2026-09-16).  `aloha_assets` at the repo
## root is a symlink to ASSETS_DIR/meshes, which is how giava.urdf's
## `filename="aloha_assets/*.stl"` resolves -- do not delete it.
ASSETS_DIR = REPO_ROOT / "assets"
SCRIPTS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPTS_DIR / "dataset" / "lerobot"
TUBE_MPC_CONFIG = REPO_ROOT / "tube_mpc" / "config.giava.yaml"
PYROKI_EXAMPLES = REPO_ROOT / "pyroki" / "examples"
## The catkin workspace's generated python packages -- added to sys.path by the
## entry points so `interbotix_xs_modules` imports without sourcing setup.bash.
ROS_DEVEL_SITE = (REPO_ROOT / "interbotix_ws" / "devel" / "lib" / "python3"
                  / "dist-packages")


def describe() -> str:
    src = "GIAVA_ROOT" if os.environ.get("GIAVA_ROOT") else f"found by {_MARKER}"
    return f"[paths] repo root {REPO_ROOT} ({src})"


if __name__ == "__main__":
    print(describe())
    for name in ("URDF_PATH", "SCRIPTS_DIR", "DATASET_ROOT", "TUBE_MPC_CONFIG",
                 "PYROKI_EXAMPLES", "ROS_DEVEL_SITE"):
        p = globals()[name]
        print(f"  {name:<18s} {'ok ' if Path(p).exists() else 'MISSING'} {p}")
