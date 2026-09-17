from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rgbd_episode_reader import RGBDEpisodeReader


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "opencv-python is required to build depth preview frames. "
            "Install it with `pip install opencv-python`."
        ) from exc
    return cv2


def require_av():
    try:
        import av
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyAV is required to write episode videos. "
            "Install it with `pip install av`."
        ) from exc
    return av


def make_depth_preview(
    depth_m: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    cv2 = require_cv2()
    valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    preview_u8 = np.zeros(depth_m.shape, dtype=np.uint8)

    if depth_max_m <= depth_min_m:
        return cv2.cvtColor(cv2.applyColorMap(preview_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    clipped = np.clip(depth_m, depth_min_m, depth_max_m)
    normalized = (clipped - depth_min_m) / (depth_max_m - depth_min_m)
    preview_u8[valid_mask] = np.round(normalized[valid_mask] * 255.0).astype(np.uint8)

    preview_bgr = cv2.applyColorMap(preview_u8, cv2.COLORMAP_JET)
    preview_bgr[~valid_mask] = 0
    return cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)


def compute_depth_visualization_range(depth_frames: list[np.ndarray]) -> tuple[float, float]:
    valid_values = []
    for depth in depth_frames:
        mask = np.isfinite(depth) & (depth > 0.0)
        if np.any(mask):
            valid_values.append(depth[mask])

    if not valid_values:
        return 0.0, 1.0

    all_valid = np.concatenate(valid_values)
    depth_min_m = float(np.percentile(all_valid, 1.0))
    depth_max_m = float(np.percentile(all_valid, 99.0))

    if depth_max_m <= depth_min_m:
        depth_min_m = float(np.min(all_valid))
        depth_max_m = float(np.max(all_valid))

    if depth_max_m <= depth_min_m:
        depth_max_m = depth_min_m + 1e-6

    return depth_min_m, depth_max_m


def build_default_output_dir(dataset_root: Path, camera: str, episode: int) -> Path:
    return (
        ROOT
        / "outputs"
        / dataset_root.parent.name
        / dataset_root.name
        / camera
        / f"episode_{episode:06d}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a full episode into one RGB video and one depth video.",
    )
    parser.add_argument("dataset_root", help="Path to a LeRobot dataset run root.")
    parser.add_argument("--camera", default="right_wrist", help="Camera name, e.g. right_wrist or top_scene.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to export.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: rgbd/outputs/<task>/<run>/<camera>/episode_xxxxxx",
    )
    parser.add_argument(
        "--max-ts-diff",
        type=float,
        default=None,
        help="Optional threshold in seconds for alignment validation.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print whole-episode alignment statistics.",
    )
    return parser.parse_args()


def _encoder_option_candidates(codec_name: str) -> list[dict[str, str]]:
    if codec_name == "av1":
        return [
            {"crf": "30", "preset": "medium"},
            {"crf": "30"},
            {},
        ]
    return [{}]


def open_video_stream(container, path: Path, codec_name: str, width: int, height: int, fps: int, pix_fmt: str):
    av = require_av()
    last_error = None

    for options in _encoder_option_candidates(codec_name):
        try:
            stream = container.add_stream(codec_name, rate=fps, options=options)
            stream.width = width
            stream.height = height
            stream.pix_fmt = pix_fmt
            return stream
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to create {codec_name} stream for {path}. "
        f"Last error: {last_error}"
    )


def encode_video(video_path: Path, frames_rgb: list[np.ndarray], codec_name: str, fps: int, pix_fmt: str) -> None:
    av = require_av()
    if not frames_rgb:
        raise ValueError(f"No frames available for {video_path}")

    height, width = frames_rgb[0].shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(video_path), mode="w") as container:
        stream = open_video_stream(
            container=container,
            path=video_path,
            codec_name=codec_name,
            width=width,
            height=height,
            fps=fps,
            pix_fmt=pix_fmt,
        )

        for frame_rgb in frames_rgb:
            frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def export_episode(reader: RGBDEpisodeReader, output_dir: Path, max_ts_diff: float | None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    info = reader.video_feature_info
    codec_name = info.get("video.codec", "av1")
    pix_fmt = info.get("video.pix_fmt", "yuv420p")
    fps = int(info.get("video.fps", reader._meta.get("fps", 30)))

    rgb_frames = []
    depth_raw_frames = []
    depth_frames = []

    depth_min = float("inf")
    depth_max = float("-inf")

    for frame in reader.iter_frames(rgb_order="rgb"):
        rgb_frames.append(frame.rgb)
        depth_raw_frames.append(frame.depth)
        depth_min = min(depth_min, float(np.nanmin(frame.depth)))
        depth_max = max(depth_max, float(np.nanmax(frame.depth)))

    vis_depth_min_m, vis_depth_max_m = compute_depth_visualization_range(depth_raw_frames)

    for depth in depth_raw_frames:
        depth_frames.append(
            make_depth_preview(
                depth_m=depth,
                depth_min_m=vis_depth_min_m,
                depth_max_m=vis_depth_max_m,
            )
        )

    rgb_video_path = output_dir / f"rgb_episode_{reader.episode_index:06d}.mp4"
    depth_video_path = output_dir / f"depth_episode_{reader.episode_index:06d}.mp4"

    encode_video(rgb_video_path, rgb_frames, codec_name=codec_name, fps=fps, pix_fmt=pix_fmt)
    encode_video(depth_video_path, depth_frames, codec_name=codec_name, fps=fps, pix_fmt=pix_fmt)

    summary = reader.validate_alignment(max_delta_s=max_ts_diff)
    summary.update(
        {
            "camera": reader.camera,
            "episode_index": reader.episode_index,
            "dataset_root": str(reader.dataset_root),
            "parquet_path": str(reader.parquet_path),
            "source_video_path": str(reader.video_path),
            "rgb_output_video": str(rgb_video_path),
            "depth_output_video": str(depth_video_path),
            "video_codec": codec_name,
            "video_pix_fmt": pix_fmt,
            "video_fps": fps,
            "depth_min_m": depth_min,
            "depth_max_m": depth_max,
            "depth_visualization_min_m": vis_depth_min_m,
            "depth_visualization_max_m": vis_depth_max_m,
            "depth_video_note": "Depth video is a colorized visualization, not raw metric depth.",
        }
    )

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else build_default_output_dir(dataset_root, args.camera, args.episode)
    )

    with RGBDEpisodeReader(
        dataset_root=dataset_root,
        camera=args.camera,
        episode_index=args.episode,
    ) as reader:
        summary = export_episode(
            reader=reader,
            output_dir=output_dir,
            max_ts_diff=args.max_ts_diff,
        )

    print(f"episode={summary['episode_index']} camera={summary['camera']}")
    print(f"dataset={summary['dataset_root']}")
    print(f"parquet={summary['parquet_path']}")
    print(f"source_rgb_video={summary['source_video_path']}")
    print(f"output_rgb_video={summary['rgb_output_video']}")
    print(f"output_depth_video={summary['depth_output_video']}")
    print(
        f"frames={summary['num_frames']} "
        f"fps={summary['video_fps']} "
        f"codec={summary['video_codec']} "
        f"pix_fmt={summary['video_pix_fmt']}"
    )
    print(
        "depth range meters="
        f"[{summary['depth_min_m']:.6f}, {summary['depth_max_m']:.6f}]"
    )
    print(
        f"timestamp pairs={summary['num_timestamp_pairs']} "
        f"max delta={summary['max_timestamp_delta_s']} "
        f"mean delta={summary['mean_timestamp_delta_s']}"
    )

    if args.print_summary:
        print(json.dumps(summary, indent=2))

    print(f"saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
