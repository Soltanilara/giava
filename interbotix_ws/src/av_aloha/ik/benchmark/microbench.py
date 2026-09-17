"""Where does a control tick's time actually go?

Splits the per-step cost into the JIT-compiled solve versus the Python/JAX
dispatch and host-device transfer around it, on both backends.

  python microbench.py                 # current default backend
  JAX_PLATFORMS=cpu python microbench.py
"""

from __future__ import annotations

import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_benchmark import TARGET_LINKS, load_robot  # noqa: E402
from solver import ControllerConfig, ThreeArmIK, preset  # noqa: E402
import workloads as W  # noqa: E402

REPEATS = 200


def bench(variant: str) -> None:
    robot, robot_coll = load_robot("/home/devi/giava/giava.urdf", auto_ignore=True)
    structure, weights = preset(variant)
    ik = ThreeArmIK(robot, robot_coll, TARGET_LINKS, structure)
    controller = ControllerConfig()

    idx = np.asarray([robot.links.names.index(n) for n in TARGET_LINKS])
    q0 = np.asarray(robot.joint_var_cls(0).default_factory(), dtype=np.float32)
    base_pos, base_wxyz = W.home_poses(robot, q0, idx)
    wl = W.smooth_track(base_pos, base_wxyz, steps=REPEATS + 20, dt=controller.dt)

    ik.reset(q0)
    ik.warmup(weights, controller)
    ik.reset(q0)

    # Full step: solve + dispatch + clamp + numpy conversion, blocking.
    t0 = time.perf_counter()
    for t in range(REPEATS):
        ik.step(wl.positions[t], wl.wxyzs[t], weights, controller)
    full_ms = (time.perf_counter() - t0) / REPEATS * 1e3

    # Build the solve arguments ONCE, outside the timing loop.  Rebuilding them
    # per iteration would measure argument marshalling, not the solve.
    kwargs = dict(
        prev_q=jax.numpy.asarray(ik.q),
        rest_q=ik._rest_default,
        target_positions=jax.numpy.asarray(np.asarray(wl.positions[0], np.float32)),
        target_wxyzs=jax.numpy.asarray(np.asarray(wl.wxyzs[0], np.float32)),
        max_dq=jax.numpy.full(
            ik.num_joints, controller.nominal_velocity * controller.dt, dtype=jax.numpy.float32
        ),
        pos_w=jax.numpy.asarray(np.asarray(weights.position, np.float32)),
        ori_w=jax.numpy.asarray(np.asarray(weights.orientation, np.float32)),
        manip_w=jax.numpy.asarray(np.asarray(weights.manipulability, np.float32)),
        mask=jax.numpy.asarray(np.asarray(weights.active_mask, np.float32)),
        smooth_w=jax.numpy.float32(weights.smoothness),
        rest_w=jax.numpy.float32(weights.rest),
        self_coll_w=jax.numpy.float32(weights.self_collision),
        world_coll_w=jax.numpy.float32(weights.world_collision),
        barrier_w=jax.numpy.float32(weights.limit_barrier),
        coll_margin=jax.numpy.float32(weights.collision_margin),
        barrier_margin=jax.numpy.float32(weights.limit_barrier_margin),
        world_obstacle=ik._dummy_obstacle,
    )
    jax.block_until_ready(ik._solve_jax(**kwargs))  # warm

    # Solve only, blocking: device compute plus one dispatch.
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        jax.block_until_ready(ik._solve_jax(**kwargs))
    solve_ms = (time.perf_counter() - t0) / REPEATS * 1e3

    # Same call without waiting: how much of a tick is host-side dispatch that
    # an asynchronous control loop could overlap with device work.
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        out = ik._solve_jax(**kwargs)
    dispatch_ms = (time.perf_counter() - t0) / REPEATS * 1e3
    jax.block_until_ready(out)

    backend = jax.default_backend()
    overhead = full_ms - solve_ms
    print(
        f"{variant:16s} backend={backend:3s}  "
        f"tick={full_ms:8.3f}  solve={solve_ms:8.3f}  dispatch={dispatch_ms:7.3f}  "
        f"marshalling={overhead:7.3f} ms  "
        f"(solve is {solve_ms / full_ms * 100:5.1f}% of the tick)"
    )


if __name__ == "__main__":
    variants = sys.argv[1:] or ["pure_smooth", "collision", "full"]
    print(f"devices: {jax.devices()}")
    for v in variants:
        bench(v)
