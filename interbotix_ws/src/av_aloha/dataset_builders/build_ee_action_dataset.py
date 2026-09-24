#!/usr/bin/env python3
"""Rewrite a run's `action` as END-EFFECTOR pose + gripper, in a new root.

    python build_ee_action_dataset.py --src <run_dir> --dst <new_run_dir> [--arm right] [--verify]

WHY.  A joint-space action is six angles the network has to map from where
the object is; an end-effector action is nearly the object's own position.
The hypothesis (EE study, 2026-09-15) is that the second generalizes over
placement from fewer demonstrations.  Same episodes, same observations, only
the action column changes, so a joint-action and an EE-action policy trained
from the two roots differ in exactly one thing.

WHAT THE NEW ACTION IS.  The recorder already writes
observation.ee_pose.<arm> = FK(driver_to_urdf(q_cmd)) every tick -- the
forward kinematics of the COMMANDED joint vector, i.e. the commanded EE pose
at that tick -- as [qw, qx, qy, qz, x, y, z] in the URDF base frame (pyroki
wxyz_xyz).  --verify recomputes FK from the joint action on a sample of
frames and reports the discrepancy, so the equality is measured, not
assumed.  The new action is that pose followed by the arm's gripper command:

    action = [qw, qx, qy, qz, x, y, z, gripper]      (8-d)

QUATERNION SIGN.  q and -q are the same rotation; a regression target that
flips between them mid-episode teaches a discontinuity that is not there.
Each episode is made sign-continuous (dot with the previous frame >= 0) and
starts with qw >= 0.

STATS.  lerobot normalizes the action with meta/stats.json and the
per-episode stats in meta/episodes/*; both are recomputed for the new
column.  info.json declares the new shape and names.  Videos are symlinked,
everything else copied, as build_envstate_dataset.py does.

ROLLOUT MUST MATCH: a policy trained here emits EE targets; rollout_policy.py
needs the IK hook that turns them into joint commands with the same solver
teleop uses.  Without it the 8-d output means nothing to the servos.
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

POSE_NAMES = ["qw", "qx", "qy", "qz", "x", "y", "z"]
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def sign_continuous(q):
    """(N,4) quaternions made continuous in sign along axis 0, qw>=0 at start."""
    q = q.copy()
    if q[0, 0] < 0:
        q[0] = -q[0]
    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return q


def feature_stats(M):
    M = np.asarray(M, dtype=np.float64)
    return {
        "min": M.min(0).tolist(), "max": M.max(0).tolist(),
        "mean": M.mean(0).tolist(), "std": M.std(0).tolist(),
        "count": [int(M.shape[0])],
        "q01": np.percentile(M, 1, axis=0).tolist(),
        "q10": np.percentile(M, 10, axis=0).tolist(),
        "q50": np.percentile(M, 50, axis=0).tolist(),
        "q90": np.percentile(M, 90, axis=0).tolist(),
        "q99": np.percentile(M, 99, axis=0).tolist(),
    }


def verify_fk(src, arm, A, E, n=300, seed=0):
    """Recompute FK(action joints) with the same URDF and compare to the
    recorded ee_pose.  Returns (max position error [m], max quat angle [rad])."""
    from robot_kinematics import build_robot_model
    from arm_config import ARM_CONFIG
    robot, arm_data = build_robot_model(arm)
    idx = arm_data[arm]["joint_indices"]; ee = arm_data[arm]["ee_index"]
    nj = ARM_CONFIG[arm]["num_joints"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(A), size=min(n, len(A)), replace=False)
    n_act = len(robot.joints.actuated_names)
    perr, qerr = 0.0, 0.0
    for i in pick:
        q = np.zeros(n_act, dtype=np.float32); q[idx] = A[i, :nj]
        fk = np.asarray(robot.forward_kinematics(q))[ee]   # wxyz_xyz
        perr = max(perr, float(np.linalg.norm(fk[4:] - E[i, 4:])))
        d = abs(float(np.dot(fk[:4], E[i, :4])))
        qerr = max(qerr, float(2 * np.arccos(min(1.0, d))))
    return perr, qerr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--arm", default="right")
    ap.add_argument("--verify", action="store_true",
                    help="recompute FK from the joint action on 300 frames "
                         "and report the discrepancy against ee_pose")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- refusing to overwrite")
    key = f"observation.ee_pose.{args.arm}"

    meta = json.loads((src / "meta.json").read_text())
    if meta.get("active_arms") not in (None, [args.arm]) and meta.get("mode") != args.arm:
        raise SystemExit(f"{src} was recorded in mode {meta.get('mode')!r} with "
                         f"arms {meta.get('active_arms')}; this builder handles "
                         f"a single-arm run whose action is [{args.arm} joints, gripper].")
    info = json.loads((src / "meta" / "info.json").read_text())
    old_shape = info["features"]["action"]["shape"]
    old_names = info["features"]["action"].get("names")
    print(f"[src] {src}  action {old_shape} {old_names}")
    if key not in info["features"]:
        raise SystemExit(f"{src} has no {key} -- nothing to build the EE action from")

    files = sorted(glob.glob(str(src / "data" / "*" / "*.parquet")))
    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    (dst / "videos").mkdir()
    for cam in (src / "videos").iterdir():
        (dst / "videos" / cam.name).symlink_to(cam.resolve())

    per_ep, all_new, all_A, all_E = {}, [], [], []
    for f in files:
        df = pd.read_parquet(f)
        A = np.stack(df["action"].to_numpy()).astype(np.float32)
        E = np.stack(df[key].to_numpy()).astype(np.float32)
        if A.shape[1] != 7 or E.shape[1] != 7:
            raise SystemExit(f"unexpected shapes action {A.shape} ee_pose {E.shape}")
        new = np.empty((len(A), 8), dtype=np.float32)
        ep = df["episode_index"].to_numpy()
        for e in np.unique(ep):
            m = ep == e
            new[m, :4] = sign_continuous(E[m, :4])
            new[m, 4:7] = E[m, 4:7]
            new[m, 7] = A[m, 6]
            per_ep[int(e)] = new[m]
        df["action"] = list(new)
        out = dst / Path(f).relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        all_new.append(new); all_A.append(A); all_E.append(E)
        print(f"[data] {out.name}: action {A.shape} -> {new.shape}")
    all_new = np.concatenate(all_new); all_A = np.concatenate(all_A); all_E = np.concatenate(all_E)

    if args.verify:
        perr, qerr = verify_fk(src, args.arm, all_A, all_E)
        print(f"[verify] FK(action joints) vs recorded ee_pose on 300 frames: "
              f"max position error {perr * 1e3:.2f} mm, max rotation error "
              f"{np.degrees(qerr):.2f} deg")
        if perr > 5e-3 or qerr > np.radians(1.0):
            raise SystemExit("ee_pose is NOT the FK of the action column -- "
                             "do not train on this; find out what q_cmd is.")

    names = POSE_NAMES + ["gripper"]
    info["features"]["action"] = {"dtype": "float32", "shape": [8], "names": names}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = feature_stats(all_new)
    stats_path.write_text(json.dumps(stats, indent=4))
    for f in sorted(glob.glob(str(dst / "meta" / "episodes" / "*" / "*.parquet"))):
        ef = pd.read_parquet(f)
        cols = [c for c in ef.columns if c.startswith("stats/action/")]
        for c in cols:
            k = c.split("/")[-1]
            ef[c] = [np.asarray(feature_stats(per_ep[int(e)])[k], dtype=np.float32)
                     if int(e) in per_ep else ef.loc[i, c]
                     for i, e in zip(ef.index, ef["episode_index"])]
        ef.to_parquet(f, index=False)
        print(f"[meta] {Path(f).name}: per-episode action stats updated ({len(cols)} cols)")
    meta["action_space"] = f"ee_pose.{args.arm}+gripper"
    meta["action_names"] = names
    meta["action_source"] = str(src)
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    s = stats["action"]
    print("[stats] action mean", np.round(s["mean"], 3).tolist())
    print("[stats] action std ", np.round(s["std"], 3).tolist())
    print(f"[dst] {dst}")


if __name__ == "__main__":
    main()
