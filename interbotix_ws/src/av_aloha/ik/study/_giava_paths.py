"""Put the giava trees this folder borrows from on `sys.path`.

WHY THIS EXISTS
===============
The tools in this folder are run directly (`python3 <folder>/<tool>.py`), so
Python puts only *this* folder on `sys.path`.  Anything they import from
`data_collection_scripts/` -- arm_config, robot_control, camera_manager, and
friends -- is invisible without help.  Importing this module first supplies it.

Finding the repo root: walk up looking for `giava.urdf`, the same marker
`data_collection_scripts/paths.py` uses.  Not `parents[N]` -- counting
directories silently breaks the day a file moves one level, which is exactly
what happened to the copies of that idiom this tree used to carry.

Usage, as the first import after the docstring::

    import _giava_paths  # noqa: F401  (sys.path side effect)

    import arm_config
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MARKER = "giava.urdf"


def find_repo_root() -> Path:
    """The giava repo root.  GIAVA_ROOT wins when the layout is unusual."""
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
        f"could not find {_MARKER} above {here}.  Set GIAVA_ROOT to the repo "
        f"root if this folder has been moved out of the tree.")


REPO_ROOT = find_repo_root()
AV_ALOHA = REPO_ROOT / "interbotix_ws" / "src" / "av_aloha"
SCRIPTS_DIR = AV_ALOHA / "data_collection_scripts"
IK_DIR = AV_ALOHA / "ik"

## Order matters only in that this folder (already first, put there by Python)
## keeps priority: a local module shadows the shared tree, not the reverse.
for _p in (IK_DIR, SCRIPTS_DIR, AV_ALOHA):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.append(_s)
