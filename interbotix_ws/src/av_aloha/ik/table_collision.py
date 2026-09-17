"""World collision: a tabletop half-plane, for `pk.costs.world_collision_cost`.

WHY THIS IS SEPARATE FROM SELF-COLLISION
=========================================
Self-collision (collision_models.py, COLLISION_STUDY.md) checks robot
geometry against itself; this checks the same sphere geometry against one
static plane.  PyRoki ships this natively as `world_collision_residual` /
`pk.costs.world_collision_cost` -- no new distance math is needed, just the
plane and which geometries should be allowed to feel it.

SEMANTICS: CONTACT IS ALLOWED, PENETRATION IS NOT
==================================================
Unlike the inter-arm gate (which must PROVE separation), resting a hand or
the camera housing ON the table is a normal, wanted pose.  So this uses the
same soft-hinge shape as the validated self-collision cost (colldist_from_sdf:
zero above the margin, a quadratic ramp inside it) via `world_collision_cost`
-- NOT `world_collision_constraint` (the AL/constraint form).  Phase 9 of
COLLISION_STUDY.md already measured the AL form fighting boundary-riding
(81 LM iterations / 80% non-convergence holding a margin) and diverging
under infeasible commands for the self-collision case; there is no reason to
expect a plane constraint to behave better under a command that deliberately
presses down, and no sweep has tested it, so the soft form is the only one
in use here.

WHICH LINKS SEE THE PLANE
==========================
Every sphere in the deployed decomposition, MINUS the links that are
kinematically height-invariant relative to the table -- their own actuated
joint (waist yaw) cannot change how far they are from it, so a plane cost on
them is pure constant bias with no discriminative signal, exactly the C3
"structural" failure mode from the self-collision study.

Measured over the full frozen baseline corpus (results/baseline/*.npz,
`../../scratchpad` survey, 2026-08-23): every *_base_link and *_shoulder_link
reported EXACTLY ZERO height range (0.0 mm min-to-max) across every saved
trajectory -- they sit strictly between the base mount and the first
height-changing joint, so this is a kinematic fact, not a corpus-coverage
artifact:

    left_base_link, right_base_link, middle_base_link   (~20 mm rest height)
    left_shoulder_link, right_shoulder_link              (~96 mm)
    middle_shoulder_link                                 (~89 mm)

Excluded for that reason.  Everything else -- INCLUDING the camera housing
(middle_camera_body/cover ranged 8-384 mm over the corpus: it genuinely
swings close to the table during reach-down motions) and the finger links
(29-689 mm: real grasp-off-the-table proximity) -- keeps its plane check,
because those ARE the configurations a table cost needs to see.

STATUS: HARDWARE-VALIDATED 2026-09-09, NOT SWEPT
================================================
This mirrors the self-collision winner's margin/weight (20 mm / 100) as a
starting point ONLY -- no Phase-9-style margin x weight sweep, no
mesh-ground-truth check, and no hardware trial has been run for this term.
Treat GIAVA_IK_TABLE_* as an experiment, not a deployed default: it must be
explicitly enabled (see study_ik.py) and inspected in Viser
(view_table_collision.py) before it is trusted on hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp
import pyroki as pk
from pyroki.collision import CollGeom, HalfSpace, RobotCollision

## Kinematically height-invariant under their own joint (see module docstring).
## Excluding these is a structural fact about the kinematic chain, not a
## corpus-dependent choice, unlike everything else here.
TABLE_EXCLUDED_LINKS: Tuple[str, ...] = (
    "left_base_link", "right_base_link", "middle_base_link",
    "left_shoulder_link", "right_shoulder_link", "middle_shoulder_link",
)


def table_halfspace(table_z: float = 0.0) -> HalfSpace:
    """A world-frame half-space whose boundary is the tabletop, normal +z
    (up, away from the table into free space).  `table_z` is the same
    world-frame convention as `calibration/base_validation.py --table-z`
    (default 0.0: the URDF root sits at table height)."""
    return HalfSpace.from_point_and_normal(
        point=jnp.array([0.0, 0.0, table_z], dtype=jnp.float32),
        normal=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
    )


def table_robot_collision(
    urdf,
    results_dir: Path,
    excluded_links: Tuple[str, ...] = TABLE_EXCLUDED_LINKS,
) -> RobotCollision:
    """Sphere geometry for the table check: the same fitted decomposition
    self-collision uses, minus the structurally-invariant links above.
    A separate `RobotCollision` from the self-collision one on purpose --
    this one has no pairs, only geoms-vs-plane, so pair pruning doesn't
    apply and shouldn't be shared."""
    decomp = json.loads((results_dir / "sphere_decomposition.json").read_text())
    filtered = {k: v for k, v in decomp.items() if k not in excluded_links}
    return RobotCollision.from_sphere_decomposition(filtered, urdf)


def table_world_collision_cost(
    robot: pk.Robot,
    table_coll: RobotCollision,
    joint_var,
    table_geom: Optional[CollGeom] = None,
    table_z: float = 0.0,
    margin: float = 0.020,
    weight: float = 100.0,
):
    """`pk.costs.world_collision_cost` against the tabletop plane -- soft
    hinge, contact allowed, penetration penalized.  Pass a pre-built
    `table_geom` to avoid rebuilding the HalfSpace inside a jit'd solve."""
    geom = table_geom if table_geom is not None else table_halfspace(table_z)
    return pk.costs.world_collision_cost(
        robot=robot,
        robot_coll=table_coll,
        joint_var=joint_var,
        world_geom=geom,
        margin=margin,
        weight=weight,
    )
