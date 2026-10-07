"""Dense surface point cloud of the three GIAVA arms, posed by joint angles.

Every surface point belongs to exactly one link, so it is sampled ONCE in that
link's frame and then carried by forward kinematics:

    p_world(q) = T_world_link(q) @ T_link_visual @ (scale * p_stl)

Sampling fixes the point's identity: point i is the same physical spot on the
same STL for every configuration, which is what makes a per-point 3D
trajectory meaningful.  World frame = the giava.urdf root, the same frame as
pyroki, the IK study and the recorded observation.ee_pose.

    python arm_surface.py --selftest
    python arm_surface.py --root <lerobot_run_dir> --episode 0 --out traj.npz --plot traj.png

No JAX: forward kinematics is yourdfpy on the same giava.urdf, plain numpy.

JOINT VALUES FROM A RECORDING.  The recorder stores DRIVER joint values.
For the two hand arms those equal the URDF values.  For the middle arm every
per-joint correction was folded into giava.urdf on 2026-08-21
(middle_joint_offsets.json), except the multiturn waist:

    urdf_middle_base = wrap(driver + pi - waist_driver_shift)

The gripper is recorded as the gripper MOTOR angle; both finger joints get the
same linear position, mapped exactly as calibration/kinematics.py does.
Joints a recording does not contain (the left arm in a right-only run) hold
`rest`, zeros by default -- they are posed, just not moving.

THE MESHES ARE THIN SHELLS.  The vx300s STLs are housings (12-23% solid), so
uniform surface sampling puts points on inner walls as well as the outside.
That is harmless for trajectories; use `outward_only` if only the visible skin
matters (keeps points whose normal faces away from the link's centroid --
cheap, and approximate on concave parts).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

## Gripper motor angle at the ends of travel and the finger joint range.
## Must match calibration/kinematics.py (GRIPPER_*_RAD, FINGER_*_M).
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_RAD = -1.5
FINGER_CLOSED_M = 0.010
FINGER_TRAVEL_M = 0.041
FINGER_JOINTS = {
    "right": ("right_left_finger", "right_right_finger"),
    "left": ("left_left_finger", "left_right_finger"),
}


def find_urdf() -> Path:
    for cand in Path(__file__).resolve().parents:
        if (cand / "giava.urdf").exists():
            return cand / "giava.urdf"
    raise SystemExit("could not find giava.urdf above this file")


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def finger_m_from_gripper(angle):
    """Gripper motor angle [rad] -> finger joint position [m]."""
    frac = (np.asarray(angle, float) - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    return np.clip(FINGER_CLOSED_M + frac * (FINGER_TRAVEL_M - FINGER_CLOSED_M),
                   FINGER_CLOSED_M, FINGER_TRAVEL_M)


def arm_of(link: str) -> str:
    for a in ("right", "left", "middle"):
        if link.startswith(a + "_"):
            return a
    return "other"


@dataclass
class SurfaceSample:
    """The fixed, configuration-independent part: where each point sits on its link."""
    p_link: np.ndarray       # (N, 3) float64, point in its link's frame [m]
    n_link: np.ndarray       # (N, 3) float64, outward face normal in link frame
    link_idx: np.ndarray     # (N,)   int32, index into `links`
    links: list              # link name per link_idx
    mesh: list               # STL path per link_idx (first visual on that link)
    arm: np.ndarray          # (N,)   'right' / 'left' / 'middle'
    spacing_m: float


class ArmSurface:
    """Sample once, pose many times.

        surf = ArmSurface(spacing_m=0.005)          # ~one point per 5 mm
        P = surf.points(q)                          # (N, 3) world, one config
        traj = surf.trajectory(Q)                   # (T, N, 3) world, a motion
    """

    def __init__(self, urdf_path: Optional[str] = None, spacing_m: float = 0.005,
                 seed: int = 0, arms: Iterable[str] = ("right", "left", "middle"),
                 outward_only: bool = False):
        import trimesh
        from yourdfpy import URDF

        self.urdf_path = Path(urdf_path) if urdf_path else find_urdf()
        self.urdf = URDF.load(str(self.urdf_path), load_meshes=False, build_scene_graph=True)
        self.joint_names = list(self.urdf.actuated_joint_names)
        self._jidx = {n: i for i, n in enumerate(self.joint_names)}
        self._lo = np.array([self.urdf.joint_map[n].limit.lower for n in self.joint_names], float)
        self._hi = np.array([self.urdf.joint_map[n].limit.upper for n in self.joint_names], float)
        arms = set(arms)
        rng = np.random.default_rng(seed)

        pts, nrm, lidx, links, meshes, arm_lab = [], [], [], [], [], []
        density = 1.0 / spacing_m ** 2                       # points per m^2
        for link_name, link in self.urdf.link_map.items():
            if arm_of(link_name) not in arms:
                continue
            for vis in link.visuals:
                g = vis.geometry
                if g.mesh is None:
                    continue
                path = (self.urdf_path.parent / g.mesh.filename).resolve()
                m = trimesh.load(str(path), force="mesh", process=False)
                if g.mesh.scale is not None:
                    m.apply_scale(np.asarray(g.mesh.scale, float))
                if vis.origin is not None:
                    m.apply_transform(vis.origin)            # now in the LINK frame
                count = max(1, int(round(m.area * density)))
                # trimesh's sampler takes a seed, so the sample (and every
                # point's identity) is reproducible run to run.
                p, face = trimesh.sample.sample_surface(m, count, seed=int(rng.integers(2**31)))
                n = m.face_normals[face]
                if outward_only:
                    keep = np.einsum("ij,ij->i", p - m.centroid, n) > 0
                    p, n = p[keep], n[keep]
                if link_name not in links:
                    links.append(link_name)
                    meshes.append(str(path))
                k = links.index(link_name)
                pts.append(p); nrm.append(n)
                lidx.append(np.full(len(p), k, np.int32))
                arm_lab.append(np.full(len(p), arm_of(link_name)))

        order = np.argsort(np.concatenate(lidx), kind="stable")  # contiguous per link
        self.sample = SurfaceSample(
            p_link=np.concatenate(pts)[order], n_link=np.concatenate(nrm)[order],
            link_idx=np.concatenate(lidx)[order], links=links, mesh=meshes,
            arm=np.concatenate(arm_lab)[order], spacing_m=spacing_m)
        li = self.sample.link_idx
        self._slices = [slice(int(np.searchsorted(li, k)), int(np.searchsorted(li, k, "right")))
                        for k in range(len(links))]

    # ------------------------------------------------------------------ #
    @property
    def n_points(self) -> int:
        return len(self.sample.p_link)

    def q_vector(self, values: Dict[str, float], rest: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Full actuated vector (URDF order) from {joint_name: value}, clipped to limits."""
        q = np.zeros(len(self.joint_names))
        for src in (rest or {}), values:
            for n, v in src.items():
                q[self._jidx[n]] = v
        return np.clip(q, self._lo, self._hi)

    def link_transforms(self, q: np.ndarray) -> np.ndarray:
        """(L, 4, 4) T_world_link for every sampled link."""
        self.urdf.update_cfg(np.asarray(q, float))
        return np.stack([self.urdf.get_transform(l) for l in self.sample.links])

    def points(self, q: np.ndarray, normals: bool = False):
        """World positions of every surface point at configuration q."""
        T = self.link_transforms(q)
        P = np.empty((self.n_points, 3))
        N = np.empty((self.n_points, 3)) if normals else None
        for k, s in enumerate(self._slices):
            R, t = T[k, :3, :3], T[k, :3, 3]
            P[s] = self.sample.p_link[s] @ R.T + t
            if normals:
                N[s] = self.sample.n_link[s] @ R.T
        return (P, N) if normals else P

    def trajectory(self, Q: np.ndarray, idx: Optional[np.ndarray] = None,
                   dtype=np.float32) -> np.ndarray:
        """(T, N, 3) world trajectories for a sequence of configurations.

        `idx` selects a subset of points.  Memory is T * N * 12 bytes: 100k
        points x 1,000 frames is 1.2 GB, so subset or chunk long episodes."""
        Q = np.atleast_2d(Q)
        sel = np.arange(self.n_points) if idx is None else np.asarray(idx)
        out = np.empty((len(Q), len(sel), 3), dtype)
        li = self.sample.link_idx[sel]
        pl = self.sample.p_link[sel]
        for t, q in enumerate(Q):
            T = self.link_transforms(q)
            out[t] = np.einsum("nij,nj->ni", T[li, :3, :3], pl) + T[li, :3, 3]
        return out

    # ------------------------------------------------------------------ #
    def q_from_recording(self, state: np.ndarray, names: Sequence[str],
                         rest: Optional[Dict[str, float]] = None,
                         waist_driver_shift: float = 0.0) -> np.ndarray:
        """(T, J) recorded driver values with their column names -> (T, n_actuated) URDF q."""
        state = np.atleast_2d(np.asarray(state, float))
        base = self.q_vector({}, rest)
        Q = np.repeat(base[None], len(state), axis=0)
        for c, name in enumerate(names):
            v = state[:, c]
            name = name.removesuffix("_cmd")     # action columns: right_waist_cmd, ...
            if name.endswith("_gripper"):
                arm = name[: -len("_gripper")]
                for fj in FINGER_JOINTS.get(arm, ()):
                    Q[:, self._jidx[fj]] = finger_m_from_gripper(v)
            elif name == "middle_base":
                Q[:, self._jidx[name]] = wrap_pi(v + np.pi - waist_driver_shift)
            elif name in self._jidx:
                Q[:, self._jidx[name]] = v
        return np.clip(Q, self._lo, self._hi)


    def waist_shift_from_recording(self, values: np.ndarray, names: Sequence[str],
                                   ee_middle: np.ndarray) -> float:
        """The middle-waist driver shift a recording implies, read off its own
        observation.ee_pose.middle (FK of the commanded joints at record time).

        Needed because trees differ: shape_sorter/... and shape_sorter_canon/...
        match at shift 0, shape_sorter_reclocked at shift pi (it is written in
        the driver frame re-clocked on 2026-09-11).  Solved on the first frame
        over a 0.5 deg grid; returns the best shift."""
        grid = np.linspace(-np.pi, np.pi, 721)
        errs = []
        for sh in grid:
            q = self.q_from_recording(values[:1], names, waist_driver_shift=sh)[0]
            self.urdf.update_cfg(q)
            errs.append(np.linalg.norm(self.urdf.get_transform("middle_camera_cover")[:3, 3]
                                       - ee_middle[0, 4:7]))
        return float(grid[int(np.argmin(errs))])


# ---------------------------------------------------------------------- #
def load_episode(root: Path, episode: int, key: str = "observation.state"):
    """(values (T, J), names, timestamps, frame table) for one LeRobot v3 episode."""
    import json
    import pandas as pd
    info = json.load(open(root / "meta" / "info.json"))
    names = info["features"][key]["names"]
    df = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").glob("chunk-*/file-*.parquet"))])
    df = df[df["episode_index"] == episode].sort_values("frame_index")
    if df.empty:
        raise SystemExit(f"episode {episode} not in {root}")
    return np.stack(df[key].to_numpy()), list(names), df["timestamp"].to_numpy(), df


def selftest(surf: ArmSurface, root: Optional[Path] = None, episode: int = 0) -> bool:
    """Checks against things the sampler cannot see. Returns True if all pass."""
    import trimesh
    ok = True
    rng = np.random.default_rng(1)

    # 1. Every posed point lies ON its posed mesh: catches scale, visual
    #    origin and wrong-link mistakes.  Independent path: the mesh is posed
    #    with yourdfpy's own transform, the points with ours.
    q = surf.q_vector({n: rng.uniform(lo, hi) for n, lo, hi in
                       zip(surf.joint_names, surf._lo, surf._hi)})
    P = surf.points(q)
    T = surf.link_transforms(q)
    worst = 0.0
    for k, link in enumerate(surf.sample.links):
        vis = [v for v in surf.urdf.link_map[link].visuals if v.geometry.mesh is not None]
        parts = []
        for v in vis:
            m = trimesh.load(str((surf.urdf_path.parent / v.geometry.mesh.filename).resolve()),
                             force="mesh", process=False)
            if v.geometry.mesh.scale is not None:
                m.apply_scale(np.asarray(v.geometry.mesh.scale, float))
            if v.origin is not None:
                m.apply_transform(v.origin)
            parts.append(m)
        m = trimesh.util.concatenate(parts)
        m.apply_transform(T[k])
        s = surf._slices[k]
        sub = P[s][:: max(1, (s.stop - s.start) // 200)]
        d = np.abs(trimesh.proximity.closest_point(m, sub)[1]).max()
        worst = max(worst, d)
    # Tolerance 0.5 mm, not ~0: trimesh's closest_point searches candidate
    # faces through nearest VERTICES and misses long thin triangles, so it
    # reports up to ~0.2 mm for points sampled exactly on the mesh, even with
    # no transform at all (measured on the gripper plates and camera body).
    # A wrong scale, visual origin or link is millimetres to metres.
    print(f"[1] posed points on posed meshes: worst {worst*1e3:.3f} mm   "
          f"{'PASS' if worst < 5e-4 else 'FAIL'}"); ok &= worst < 5e-4

    # 2. Rigidity: distances between points on one link never change.
    q2 = surf.q_vector({n: rng.uniform(lo, hi) for n, lo, hi in
                        zip(surf.joint_names, surf._lo, surf._hi)})
    P2 = surf.points(q2)
    worst = 0.0
    for s in surf._slices:
        a = np.arange(s.start, s.stop)[:: max(1, (s.stop - s.start) // 50)]
        D1 = np.linalg.norm(P[a, None] - P[None, a], axis=-1)
        D2 = np.linalg.norm(P2[a, None] - P2[None, a], axis=-1)
        worst = max(worst, np.abs(D1 - D2).max())
    print(f"[2] per-link rigidity across configs: worst {worst*1e9:.3f} nm   "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}"); ok &= worst < 1e-9

    # 3. trajectory() and points() agree.
    tr = surf.trajectory(np.stack([q, q2]), dtype=np.float64)
    d = max(np.abs(tr[0] - P).max(), np.abs(tr[1] - P2).max())
    print(f"[3] trajectory() == points(): {d:.2e} m   {'PASS' if d < 1e-12 else 'FAIL'}"); ok &= d < 1e-12

    # 4. Against a real recording: the recorder's ee_pose is FK of the
    #    COMMANDED joints (`action`) on <arm>_gripper_base, quaternion first.
    if root is not None:
        A, anames, _, df = load_episode(root, episode, "action")
        ee_keys = [c for c in df.columns if c.startswith("observation.ee_pose.")]
        shift = 0.0
        if "observation.ee_pose.middle" in df.columns:
            shift = surf.waist_shift_from_recording(
                A, anames, np.stack(df["observation.ee_pose.middle"].to_numpy()))
            print(f"    middle waist driver shift read from the recording: {shift:+.4f} rad")
        for key in ee_keys:
            arm = key.rsplit(".", 1)[-1]
            ee = np.stack(df[key].to_numpy())[:, 4:7]
            Q = surf.q_from_recording(A, anames, waist_driver_shift=shift)
            link = f"{arm}_gripper_base" if arm != "middle" else "middle_camera_cover"
            err = []
            for t in range(0, len(Q), max(1, len(Q) // 50)):
                surf.urdf.update_cfg(Q[t])
                err.append(np.linalg.norm(surf.urdf.get_transform(link)[:3, 3] - ee[t]))
            e = max(err)
            print(f"[4] FK(action) vs recorded ee_pose.{arm}: worst {e*1e3:.3f} mm over "
                  f"{len(err)} frames   {'PASS' if e < 1e-3 else 'FAIL'}"); ok &= e < 1e-3
    return bool(ok)


def plot(surf: ArmSurface, traj: np.ndarray, idx: np.ndarray, path: str, every: int = 1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    P0 = traj[0]
    colors = {"right": "#2E6FBF", "left": "#C2552C", "middle": "#2E7D53"}
    for arm, c in colors.items():
        m = surf.sample.arm[idx] == arm
        if m.any():
            ax.scatter(*P0[m][::every].T, s=0.3, c=c, alpha=0.25, label=f"{arm} arm, frame 0")
    track = np.linspace(0, len(idx) - 1, 12).astype(int)
    for i in track:
        ax.plot(*traj[:, i].T, lw=1.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_box_aspect(np.ptp(np.concatenate([P0, traj[:, track].reshape(-1, 3)]), axis=0))
    ax.legend(loc="upper left", markerscale=20)
    ax.set_title("Arm surface at frame 0, with 12 surface-point trajectories")
    fig.tight_layout(); fig.savefig(path, dpi=140)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spacing", type=float, default=0.005, help="target point spacing [m]")
    ap.add_argument("--arms", nargs="+", default=["right", "left", "middle"])
    ap.add_argument("--outward-only", action="store_true")
    ap.add_argument("--root", type=Path, help="LeRobot run dir (contains meta/ and data/)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--key", default="observation.state",
                    help="observation.state = measured (default), action = commanded")
    ap.add_argument("--waist-driver-shift", type=float, default=None,
                    help="middle waist shift [rad]; default: read from the recording's "
                         "ee_pose.middle, else 0 (pi for shape_sorter_reclocked)")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--max-points", type=int, default=None, help="random subset of points")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plot", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    surf = ArmSurface(spacing_m=a.spacing, arms=a.arms, outward_only=a.outward_only)
    counts = {arm: int((surf.sample.arm == arm).sum()) for arm in a.arms}
    print(f"{surf.n_points:,} surface points on {len(surf.sample.links)} links at "
          f"{a.spacing*1e3:.1f} mm spacing   {counts}")

    if a.selftest:
        sys.exit(0 if selftest(surf, a.root, a.episode) else 1)
    if a.root is None:
        return

    vals, names, ts, df = load_episode(a.root, a.episode, a.key)
    shift = a.waist_driver_shift
    if shift is None:
        shift = 0.0
        if "middle_base" in [n.removesuffix("_cmd") for n in names] \
                and "observation.ee_pose.middle" in df.columns:
            A, anames, _, _ = load_episode(a.root, a.episode, "action")
            shift = surf.waist_shift_from_recording(
                A, anames, np.stack(df["observation.ee_pose.middle"].to_numpy()))
            print(f"middle waist driver shift read from the recording: {shift:+.4f} rad")
    Q = surf.q_from_recording(vals[:: a.every], names, waist_driver_shift=shift)
    idx = np.arange(surf.n_points)
    if a.max_points and a.max_points < surf.n_points:
        idx = np.sort(np.random.default_rng(0).choice(surf.n_points, a.max_points, replace=False))
    traj = surf.trajectory(Q, idx)
    print(f"episode {a.episode}: {len(Q)} frames x {len(idx):,} points -> {traj.nbytes/1e6:.0f} MB "
          f"(joints from {a.key}: {names})")
    if a.out:
        np.savez_compressed(
            a.out, traj=traj, timestamps=ts[:: a.every], q=Q, joint_names=np.array(surf.joint_names),
            point_index=idx, p_link=surf.sample.p_link[idx], link_idx=surf.sample.link_idx[idx],
            links=np.array(surf.sample.links), arm=surf.sample.arm[idx], spacing_m=a.spacing)
        print(f"wrote {a.out}")
    if a.plot:
        plot(surf, traj, idx, str(a.plot), every=max(1, len(idx) // 40000))
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
