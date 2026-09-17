"""Pull revolute-joint limits out of a URDF without extra dependencies.

URDFs carry position and velocity limits but not acceleration limits, so
a_max must be supplied (identify it from logged data or servo specs --
see the "what I need from you" list in README.md).
"""

import re

import numpy as np

from tube_mpc.model import JointLimits

_JOINT_RE = re.compile(
    r'<joint\s+name="([^"]+)"\s+type="revolute".*?'
    r'<limit[^>]*?lower="([^"]+)"[^>]*?upper="([^"]+)"[^>]*?velocity="([^"]+)"',
    re.DOTALL,
)


def parse_urdf_limits(path: str, prefix: str, a_max: float | np.ndarray) -> tuple[list[str], JointLimits]:
    """Limits for every revolute joint whose name starts with `prefix`.

    Returns (joint_names, JointLimits) in file order.
    """
    with open(path) as f:
        text = f.read()
    names, lo, hi, vel = [], [], [], []
    for m in _JOINT_RE.finditer(text):
        if not m.group(1).startswith(prefix):
            continue
        names.append(m.group(1))
        lo.append(float(m.group(2)))
        hi.append(float(m.group(3)))
        vel.append(float(m.group(4)))
    if not names:
        raise ValueError(f"no revolute joints with prefix {prefix!r} in {path}")
    n = len(names)
    a = np.full(n, float(a_max)) if np.isscalar(a_max) else np.asarray(a_max, dtype=float)
    return names, JointLimits(
        q_min=np.array(lo), q_max=np.array(hi), v_max=np.array(vel), a_max=a)
