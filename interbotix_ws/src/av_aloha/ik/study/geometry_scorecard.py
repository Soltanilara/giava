"""Collision-study Phases 2–4 — geometry accuracy scorecard.

Evaluates three collision representations against the mesh ground truth
(results/mesh_ground_truth.csv, 70 cases = 14 pairs × 5 configurations):

    pyroki   — original single-capsule (minimum-cylinder fit)
    tight    — corrected single-capsule (longest-axis fit)
    spheres  — multi-sphere model (results/sphere_decomposition.json)

Reports:
  A. regression accuracy: mean/median/max |error|, signed bias — overall and
     per category (fingers / fork / plates / arm links / inter-arm / self)
  B. safety classification at threshold 0 (collision) and the 25 mm
     near-collision band: FP (model says collision, mesh clear),
     FN (model clear, mesh collides), workspace falsely removed
  C. the three acceptance criteria:
     1. rest pose (self_fold_apex) must not read as severe collision
     2. bimanual (home, hand↔hand pairs): falsely-removed clearance
     3. real contacts (mesh < 0) must be detected (or near-detected)

Run: JAX_PLATFORMS=cpu python geometry_scorecard.py
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jaxlie
import jax.numpy as jnp

import robot_model as rm
from collision_models import tight_capsule_collision
from pyroki.collision import RobotCollision
from pyroki.collision._collision import collide

HERE = Path(__file__).resolve().parent
NEAR_BAND = 0.025  # m


def categorize(pair: str) -> str:
    a, b = pair.split("|")
    inter = (a[:4] != b[:4]) if not (a.startswith("middle") or b.startswith("middle")) else True
    side = "inter-arm" if (a.split("_")[0][:5] != b.split("_")[0][:5]) else "self"
    for tag, cat in (("finger", "fingers"), ("gripper", "fork"),
                     ("base_link", "plates"), ("camera_cover", "plates"),
                     ("camera", "camera"), ("forearm", "arm-links"),
                     ("upper_arm", "arm-links"), ("wrist", "arm-links")):
        if tag in a or tag in b:
            return f"{cat}/{side}"
    return f"other/{side}"


def main() -> None:
    robot, urdf = rm.load(with_urdf=True)
    rc_pyroki = RobotCollision.from_urdf(urdf)
    rc_tight = tight_capsule_collision(urdf)
    decomp = json.loads((HERE / "results" / "sphere_decomposition.json").read_text())
    name_to_idx = {n: i for i, n in enumerate(rc_pyroki.link_names)}

    rows = list(csv.DictReader(open(HERE / "results" / "mesh_ground_truth.csv")))
    configs_needed = sorted({r["config"] for r in rows})

    # reload the configs the ground-truth script used
    from mesh_ground_truth import CASES

    q0 = rm.home_config(robot)
    configs = {}
    for name, src in CASES.items():
        if name not in configs_needed:
            continue
        if src is None:
            configs[name] = q0
        else:
            q = np.load(HERE / "results" / "baseline" / src[0])["q"]
            configs[name] = q[min(src[1], len(q) - 1)]

    def capsule_dist(rc, q, a, b):
        fk = robot.forward_kinematics(np.asarray(q, dtype=np.float32))
        coll = rc.coll.transform(jaxlie.SE3(jnp.asarray(fk)[rc._geom_to_link_idx]))
        ca = jax.tree.map(lambda x: x[name_to_idx[a]], coll)
        cb = jax.tree.map(lambda x: x[name_to_idx[b]], coll)
        return float(collide(ca, cb))

    def sphere_dist(q, a, b):
        fk = np.asarray(robot.forward_kinematics(np.asarray(q, dtype=np.float32)))
        best = np.inf
        for la, lb in ((a, b),):
            Ta = jaxlie.SE3(jnp.asarray(fk[name_to_idx[la]]))
            Tb = jaxlie.SE3(jnp.asarray(fk[name_to_idx[lb]]))
            ca = np.asarray(decomp[la]["centers"]); ra = np.asarray(decomp[la]["radii"])
            cb = np.asarray(decomp[lb]["centers"]); rb = np.asarray(decomp[lb]["radii"])
            Ra = np.asarray(Ta.rotation().as_matrix()); pa = np.asarray(Ta.translation())
            Rb = np.asarray(Tb.rotation().as_matrix()); pb = np.asarray(Tb.translation())
            wa = ca @ Ra.T + pa
            wb = cb @ Rb.T + pb
            d = np.linalg.norm(wa[:, None, :] - wb[None, :, :], axis=-1) \
                - ra[:, None] - rb[None, :]
            best = min(best, float(d.min()))
        return best

    recs = []
    for r in rows:
        a, b = r["pair"].split("|")
        q = configs[r["config"]]
        recs.append(dict(
            config=r["config"], pair=r["pair"], cat=categorize(r["pair"]),
            mesh=float(r["mesh_mm"]) / 1e3,
            pyroki=capsule_dist(rc_pyroki, q, a, b),
            tight=capsule_dist(rc_tight, q, a, b),
            spheres=sphere_dist(q, a, b),
        ))

    models = ("pyroki", "tight", "spheres")

    # -- A: regression accuracy ----------------------------------------- #
    print("=== A. clearance accuracy vs mesh truth (mm) ===")
    print(f"{'scope':22s} {'model':8s} {'mean|e|':>8s} {'med|e|':>8s} "
          f"{'max|e|':>8s} {'bias':>8s}  n")
    scopes = {"ALL": recs}
    for rec in recs:
        scopes.setdefault(rec["cat"], []).append(rec)
    for scope, items in scopes.items():
        for m in models:
            e = np.array([it[m] - it["mesh"] for it in items]) * 1e3
            print(f"{scope:22s} {m:8s} {np.abs(e).mean():8.1f} "
                  f"{np.median(np.abs(e)):8.1f} {np.abs(e).max():8.1f} "
                  f"{e.mean():+8.1f}  {len(items)}")

    # -- B: safety classification --------------------------------------- #
    print("\n=== B. classification (mesh truth: collision <0 / "
          f"near <{NEAR_BAND*1e3:.0f}mm / clear) ===")
    print(f"{'model':8s} {'TP':>3s} {'FN':>3s} {'FP':>4s} {'TN':>4s}  "
          f"{'near-FP':>7s}  {'stolen mm (med/max)':>20s}")
    for m in models:
        truth_c = np.array([it["mesh"] < 0 for it in recs])
        model_c = np.array([it[m] < 0 for it in recs])
        tp = int((truth_c & model_c).sum())
        fn = int((truth_c & ~model_c).sum())
        fp = int((~truth_c & model_c).sum())
        tn = int((~truth_c & ~model_c).sum())
        near_fp = int(sum(1 for it in recs
                          if it["mesh"] >= NEAR_BAND and it[m] < NEAR_BAND))
        stolen = np.array([max(0.0, it["mesh"] - it[m]) for it in recs
                           if it["mesh"] >= 0]) * 1e3
        print(f"{m:8s} {tp:3d} {fn:3d} {fp:4d} {tn:4d}  {near_fp:7d}  "
              f"{np.median(stolen):9.1f}/{stolen.max():6.1f}")

    # -- C: acceptance criteria ----------------------------------------- #
    print("\n=== C. acceptance criteria ===")
    rest = [it for it in recs if it["config"] == "self_fold_apex"]
    print("C1 rest pose (self_fold_apex): worst model reading (mm), "
          "truth worst =", f"{min(it['mesh'] for it in rest)*1e3:.1f}")
    for m in models:
        worst = min(it[m] for it in rest)
        n_sev = sum(1 for it in rest if it[m] < -0.02 and it["mesh"] > 0)
        print(f"  {m:8s} worst {worst*1e3:8.1f}  severe-false-collisions "
              f"(< −20 mm while truly clear): {n_sev}")
    bi = [it for it in recs if it["config"] == "home"
          and ("gripper" in it["pair"] or "finger" in it["pair"])
          and it["cat"].endswith("inter-arm")]
    print("C2 bimanual (home, hand↔hand pairs): falsely-removed clearance (mm)")
    for m in models:
        stolen = [max(0.0, it["mesh"] - it[m]) * 1e3 for it in bi]
        fp = sum(1 for it in bi if it[m] < 0 <= it["mesh"])
        print(f"  {m:8s} stolen med {np.median(stolen):6.1f} max "
              f"{max(stolen):6.1f}  false-collisions: {fp}")
    contact = [it for it in recs if it["mesh"] < 0]
    print("C3 real contacts (mesh < 0): detection per model")
    for m in models:
        det = sum(1 for it in contact if it[m] < 0)
        near = sum(1 for it in contact if 0 <= it[m] < NEAR_BAND)
        print(f"  {m:8s} detected {det}/{len(contact)}  near-detected {near}")

    with open(HERE / "results" / "geometry_scorecard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nwrote results/geometry_scorecard.csv")


if __name__ == "__main__":
    main()
