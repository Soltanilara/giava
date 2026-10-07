"""Side-by-side video: arm surface points reprojected into top_scene | their 3D trajectories.

    python render_surface_video.py --root <lerobot run dir> --episode 0 \
        --pose outputs/top_scene_pose_from_grasps.json --out ep0.mp4

LEFT   the recorded top_scene frame with every outward-facing surface point of
       the right arm projected through the camera model -- no detection, no
       matching, only FK + T_world_camera + intrinsics.  Back-facing points
       (normal pointing away from the camera) are culled; occlusion between
       links is not resolved.  The K tracked points are drawn larger, each in
       its own colour, with a short image-space trail.
RIGHT  the same K points in the URDF world frame: each trajectory drawn up to
       the current frame in that point's colour, over the faint current arm.

The pose is whatever --pose holds.  outputs/top_scene_pose_from_grasps.json is
PROVISIONAL (solved from recorded grasps, ~2.4 px median); swap in a real
scene_extrinsics result when one exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arm_surface import ArmSurface, load_episode  # noqa: E402
from camera import load_intrinsics  # noqa: E402


def farthest_points(P: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k indices spread evenly over the cloud P (greedy farthest-point sampling)."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(P)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(P - P[idx[-1]], axis=1))
    return np.array(idx)


def episode_frames(root: Path, episode: int, camera: str = "observation.images.top_scene"):
    """Decode this episode's frames from the LeRobot v3 video file, in order."""
    import av
    import pandas as pd
    info = json.loads((root / "meta" / "info.json").read_text())
    em = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").glob("*/*.parquet"))])
    row = em[em["episode_index"] == episode].iloc[0]
    ci, fi = int(row[f"videos/{camera}/chunk_index"]), int(row[f"videos/{camera}/file_index"])
    t0, t1 = float(row[f"videos/{camera}/from_timestamp"]), float(row[f"videos/{camera}/to_timestamp"])
    path = root / info["video_path"].format(video_key=camera, chunk_index=ci, file_index=fi)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, t0 - 0.5) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t < t0 - 1e-3:
                continue
            if t > t1 + 1e-3:
                break
            yield fr.to_ndarray(format="rgb24")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--pose", type=Path, default=HERE / "outputs" / "top_scene_pose_from_grasps.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=150, help="number of coloured tracked points")
    ap.add_argument("--spacing", type=float, default=0.005)
    ap.add_argument("--every", type=int, default=2, help="render every Nth recorded frame")
    ap.add_argument("--trail2d", type=int, default=25, help="image-space trail length [rendered frames]")
    ap.add_argument("--check", type=int, default=None, help="write only this frame as a PNG and stop")
    a = ap.parse_args()

    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # ---------------------------------------------------------------- data
    surf = ArmSurface(spacing_m=a.spacing, arms=("right",), outward_only=True)
    vals, names, ts, _ = load_episode(a.root, a.episode, "observation.state")
    info = json.loads((a.root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    Q = surf.q_from_recording(vals, names)
    print(f"{surf.n_points:,} right-arm outward points, {len(Q)} frames")

    traj = np.empty((len(Q), surf.n_points, 3))
    nrm = np.empty((len(Q), surf.n_points, 3))
    for t, q in enumerate(Q):
        traj[t], nrm[t] = surf.points(q, normals=True)
    track = farthest_points(traj[0], a.k)
    rng = np.random.default_rng(3)
    cmap = plt.get_cmap("turbo")
    col = cmap(rng.permutation(np.linspace(0.04, 0.96, a.k)))[:, :3]       # one colour per trajectory
    col_bgr = (col[:, ::-1] * 255).astype(np.uint8)

    cam = load_intrinsics("top_scene")
    pose = json.loads(a.pose.read_text())
    T_cw = np.linalg.inv(np.array(pose["matrix_4x4_row_major"]))
    C_world = np.array(pose["matrix_4x4_row_major"])[:3, 3]

    def project(P):
        pc = P @ T_cw[:3, :3].T + T_cw[:3, 3]
        uv = np.full((len(P), 2), np.nan)
        front = pc[:, 2] > 0.05
        uv[front] = cam.project(pc[front])
        return uv

    # ------------------------------------------------------------- 3D panel
    TR = traj[:, track]
    lo = np.minimum(traj.min((0, 1)), TR.min((0, 1))) - 0.03
    hi = np.maximum(traj.max((0, 1)), TR.max((0, 1))) + 0.03
    lo[2] = 0.0
    fig = plt.figure(figsize=(7.2, 5.4), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(0, 0, 1, 0.94)

    def render3d(t):
        ax.cla()
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=28, azim=-62)
        ax.set_xlabel("x [m]", labelpad=-6); ax.set_ylabel("y [m]", labelpad=-6); ax.set_zlabel("z [m]", labelpad=-6)
        ax.tick_params(labelsize=7, pad=-2)
        P = traj[t][::3]
        ax.scatter(*P.T, s=0.4, c="#9AA4B0", alpha=0.25, depthshade=False)
        segs = np.stack([TR[: t + 1, j] for j in range(a.k)])              # (k, t+1, 3)
        if t > 0:
            lc = Line3DCollection(segs, colors=col, linewidths=1.0, alpha=0.9)
            ax.add_collection3d(lc)
        ax.scatter(*TR[t].T, s=9, c=col, depthshade=False, edgecolors="none")
        ax.set_title(f"{a.k} surface-point trajectories, URDF world frame   t = {ts[t] - ts[0]:5.2f} s",
                     fontsize=9)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        return img

    # ------------------------------------------------------------- 2D panel
    hist2d = []

    def render2d(frame, t):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        s = 1.5
        P, N = traj[t], nrm[t]
        facing = np.einsum("ij,ij->i", N, C_world - P) > 0                  # back-face cull
        uv = project(P[facing]) * s
        ok = np.isfinite(uv).all(1)
        overlay = img.copy()
        for u, v in uv[ok].astype(int):
            cv2.circle(overlay, (u, v), 1, (235, 235, 235), -1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        uvk = project(TR[t]) * s
        hist2d.append(uvk)
        H = np.stack(hist2d[-a.trail2d:])
        for j in range(a.k):
            pts = H[:, j]
            pts = pts[np.isfinite(pts).all(1)].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, col_bgr[j].tolist(), 1, cv2.LINE_AA)
            if np.isfinite(uvk[j]).all():
                cv2.circle(img, tuple(uvk[j].astype(int)), 3, col_bgr[j].tolist(), -1, cv2.LINE_AA)
        cv2.putText(img, f"top_scene  ep {a.episode}  frame {t}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "FK + camera model only (provisional pose)", (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------ write
    frames = list(episode_frames(a.root, a.episode))
    n = min(len(frames), len(Q))
    if len(frames) != len(Q):
        print(f"note: {len(frames)} video frames vs {len(Q)} data frames; using {n}")
    sel = list(range(0, n, a.every))
    if a.check is not None:
        sel = [a.check]
    out = None
    stream = None
    for k, t in enumerate(sel):
        left = render2d(frames[t], t)
        right = render3d(t)
        h = left.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h), interpolation=cv2.INTER_AREA)
        both = np.concatenate([left, right], axis=1)
        both = both[: both.shape[0] // 2 * 2, : both.shape[1] // 2 * 2]
        if a.check is not None:
            cv2.imwrite(str(a.out.with_suffix(".png")), cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
            print(f"wrote {a.out.with_suffix('.png')}")
            return
        if out is None:
            out = av.open(str(a.out), "w")
            stream = out.add_stream("libx264", rate=int(round(fps / a.every)))
            stream.width, stream.height, stream.pix_fmt = both.shape[1], both.shape[0], "yuv420p"
            stream.options = {"crf": "20", "preset": "medium"}
        for pkt in stream.encode(av.VideoFrame.from_ndarray(both, format="rgb24")):
            out.mux(pkt)
        if k % 50 == 0:
            print(f"  {k}/{len(sel)}")
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()
    print(f"wrote {a.out}  ({len(sel)} frames at {fps / a.every:.0f} fps)")


if __name__ == "__main__":
    main()
