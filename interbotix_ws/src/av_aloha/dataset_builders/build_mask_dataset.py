"""Add an object-mask video stream to a LeRobot dataset, in a new root.

Renders the blue-object detector's segmentation as a greyscale video and
registers it as an extra camera, `observation.images.obj_mask`, so ACT can be
trained with it as a fourth image input.  The hypothesis being tested: given an
explicit mask of the object of interest, does the policy attend to it more
tightly than it does from raw RGB alone?

This is a richer form of the same idea as `observation.environment_state` --
a centroid says where the object is, a mask says where it is AND how it is
lying, at full spatial resolution, in a form a convolutional backbone can
consume directly rather than through a hand-scaled projection.

FILE LAYOUT -- why the mask mirrors the source camera exactly.

LeRobot v3 rolls videos over at ~200 MB, so each camera has its own number of
mp4 files (9, 8 and 13 for the three cameras in transfer_flower_merged), and
every episode carries a (chunk_index, file_index, from_timestamp,
to_timestamp) tuple PER CAMERA in meta/episodes/*.parquet.  Rather than
recompute that mapping, this writes exactly one mask file per source-camera
file and copies the source camera's four columns verbatim.  The mapping is
then correct by construction.

    python build_mask_dataset.py --src <run_dir> --dst <new_run_dir>
    python build_mask_dataset.py --src ... --dst ... --camera right_wrist
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import glob
import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from object_mask import MASK_KEY, mask_for  # the rule rollout_policy also uses


def write_mask_stats(dst: Path, sample_stride: int = 50):
    """Add a stats.json entry for the mask stream.

    LeRobot's dataset factory (datasets/factory.py) iterates every image
    feature and assigns ImageNet statistics into meta.stats[key] -- and it
    indexes the key, so a video stream declared in info.json but absent from
    stats.json is a KeyError at training start.  This was hit on the first
    launch of flower128_act_joint_mask.  Statistics are computed from the
    actual mask frames (strided) in the same [3,1,1] per-channel layout the
    other cameras use, values in [0, 1].
    """
    files = sorted(glob.glob(str(dst / "videos" / MASK_KEY / "*" / "*.mp4")))
    acc, acc2, n, mn, mx = np.zeros(3), np.zeros(3), 0, np.ones(3), np.zeros(3)
    for f in files:
        c = av.open(f)
        for i, fr in enumerate(c.decode(video=0)):
            if i % sample_stride:
                continue
            x = fr.to_ndarray(format="rgb24").astype(np.float64) / 255.0
            m = x.reshape(-1, 3)
            acc += m.mean(0); acc2 += (m ** 2).mean(0); n += 1
            mn = np.minimum(mn, m.min(0)); mx = np.maximum(mx, m.max(0))
        c.close()
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 0)) + 1e-8
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[MASK_KEY] = {
        "mean": mean.reshape(3, 1, 1).tolist(),
        "std": std.reshape(3, 1, 1).tolist(),
        "min": mn.reshape(3, 1, 1).tolist(),
        "max": mx.reshape(3, 1, 1).tolist(),
        "count": [int(n)],
    }
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[meta] stats.json: {MASK_KEY} mean={mean.round(4).tolist()} "
          f"std={std.round(4).tolist()} from {n} sampled frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--camera", default="top_scene",
                    choices=["top_scene", "right_wrist"],
                    help="which camera the mask is rendered from; the new "
                         "stream mirrors this camera's file layout")
    ap.add_argument("--dilate", type=int, default=0,
                    help="dilate the mask by N px (softens detector jitter)")
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- refusing to overwrite")

    src_key = f"observation.images.{args.camera}"
    src_files = sorted(glob.glob(str(src / "videos" / src_key / "*" / "*.mp4")))
    if not src_files:
        raise SystemExit(f"no {args.camera} video under {src}")
    print(f"[src] {src}  mask from {args.camera}  ({len(src_files)} video file(s))")

    ## meta + data copied, existing videos symlinked -- same convention as the
    ## other builders, so variants compose.
    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    shutil.copytree(src / "data", dst / "data")
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    (dst / "videos").mkdir()
    for cam_dir in (src / "videos").iterdir():
        (dst / "videos" / cam_dir.name).symlink_to(cam_dir.resolve())
    print(f"[dst] {dst}  (meta + data copied, existing videos symlinked)")

    info = json.loads((dst / "meta" / "info.json").read_text())
    fps = int(info.get("fps", 50))
    spec = info["features"][src_key]
    H, W = spec["shape"][0], spec["shape"][1]

    total = 0
    for sp in src_files:
        rel = Path(sp).relative_to(src / "videos" / src_key)
        out = dst / "videos" / MASK_KEY / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        cin = av.open(sp)
        cout = av.open(str(out), mode="w")
        stream = cout.add_stream("libx264", rate=fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(args.crf), "preset": "veryfast"}

        n = 0
        for frame in cin.decode(video=0):
            m = mask_for(frame.to_ndarray(format="rgb24"), args.camera, args.dilate)
            rgb = np.repeat(m[:, :, None], 3, axis=2)
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in stream.encode(vf):
                cout.mux(pkt)
            n += 1
        for pkt in stream.encode():
            cout.mux(pkt)
        cout.close()
        cin.close()
        total += n
        mb = out.stat().st_size / 1e6
        print(f"[video] {rel} : {n} frames -> {mb:.1f} MB")
    print(f"[video] {total} frames encoded to {MASK_KEY}")

    ## info.json: declare the stream, copying the source camera's spec and
    ## correcting the codec (libx264 here, not the AV1 the originals use).
    newspec = json.loads(json.dumps(spec))
    newspec["info"]["video.codec"] = "h264"
    newspec["info"]["video.crf"] = args.crf
    newspec["info"]["video.preset"] = "veryfast"
    info["features"][MASK_KEY] = newspec
    ## rollout_policy.py renders the same mask live and must use the same
    ## source camera and dilation, or the policy sees a different input
    ## distribution at test time than it trained on.
    info["obj_mask"] = {"source": args.camera, "dilate": int(args.dilate)}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    print(f"[meta] info.json: declared {MASK_KEY}")

    ## meta/episodes: the mask mirrors the source camera file-for-file, so its
    ## chunk/file/timestamp columns are copies of the source camera's.
    ep_files = sorted(glob.glob(str(dst / "meta" / "episodes" / "**" / "*.parquet"),
                                recursive=True))
    for f in ep_files:
        edf = pd.read_parquet(f)
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            sc = f"videos/{src_key}/{suffix}"
            if sc in edf.columns:
                edf[f"videos/{MASK_KEY}/{suffix}"] = edf[sc]
        edf.to_parquet(f, index=False)
        print(f"[meta] {Path(f).name}: added {MASK_KEY} video columns")

    write_mask_stats(dst)

    print(f"\nTrain with the mask as a fourth camera, e.g.\n"
          f"  --cameras right_wrist top_scene low_scene obj_mask")


if __name__ == "__main__":
    main()
