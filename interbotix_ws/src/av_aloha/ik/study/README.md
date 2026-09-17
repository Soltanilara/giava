# GIAVA IK Study — controlled solver experiments + interactive playground

A controlled study of the three-arm PyRoki IK: clean baseline, frozen
27-trajectory benchmark, one-residual ablations, weight sweeps, a
manipulability deep-dive, and a geometry-first collision study.
Full narrative & numbers: `BASELINE.md`, `TRAJECTORIES.md`,
`RESULTS_{BASELINE,PHASE4,PHASE5,FINAL}.md`, `COLLISION_STUDY.md`.

**Recommended deployment configuration** (measured winner):

```
pose_cost_analytic_jac ×3      pos 50 / ori 10
limit_constraint               (pyroki augmented Lagrangian)
smoothing                      w = 0.05 (velocity-scaled prev-config)
centering                      w = 0.5  (range-normalized, arm joints)
sphere self-collision          180 spheres · 5,884 pruned pairs ·
                               soft margin hinge · margin 20 mm · w = 100
loop                           50 Hz, warm start = previous commanded q, CPU
```

Full-suite numbers: 0.60 mm / 0.24° clean-trajectory p95, every safety
boundary held (worst clearance −157 → +3 mm), rest fold permitted, bimanual
deviation ≤ 2.7 mm, 5.2 ms mean solve. Manipulability was measured
harmful in every weight and combination — rejected.

## Try it: the interactive playground

```bash
cd interbotix_ws/src/av_aloha/data_collection_scripts/ik_study

# GPU (pick a free device):
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python playground.py --port 8092
# or CPU (lower latency for this problem size):
JAX_PLATFORMS=cpu python playground.py --port 8092
```

Open http://localhost:8092 (port-forward if remote). Four solver modes —
each a separately compiled optimization problem (different cost lists, not
zero weights): tuned base / +collision / +manipulability / +both. Play any
benchmark trajectory at any speed, or drag the three end-effector gizmos and
watch the active solver track live, with solve-time, tracking-error and
clearance readouts.

Other viewers: `view_trajectories.py --port 8090` (benchmark suite preview),
`view_collision.py --port 8091` (collision geometry: pyroki capsules vs
corrected capsules vs the 180-sphere model, at inspection poses).

## Dependencies

The conda env needs: `jax` (CPU is fine; `jax[cuda12]` for GPU), `jaxls`,
`jaxlie`, `jax_dataclasses`, `pyroki` (repo submodule: `pip install -e
./pyroki`), `viser`, `yourdfpy`, `trimesh`, `numpy`, `matplotlib`.
`giava.urdf` is found at the repo root automatically (override:
`GIAVA_URDF=/path/to/giava.urdf`).

## Reproduce the benchmarks

```bash
JAX_PLATFORMS=cpu python run_study.py --variant baseline
JAX_PLATFORMS=cpu python run_study.py --variant smooth_center_collisionS --clearance-model sphere
python compare.py            # any set of variants vs baseline, hash-verified
```

The trajectory suite is frozen (hash `3815490046334af4`, verified per run).
Per-step rollout histories (`results/<variant>/*.npz`) are gitignored —
regenerate with the commands above. The committed data are the frozen model
inputs (`results/sphere_decomposition.json`, `sphere_pair_pruning.npz`) and
the analysis outputs (CSVs, `plots/`).
