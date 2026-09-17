from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RGBDFrame:
    frame_index: int
    rgb: np.ndarray
    depth: np.ndarray
    rgb_timestamp: float | None
    depth_timestamp: float | None

    @property
    def timestamp_delta(self) -> float | None:
        if self.rgb_timestamp is None or self.depth_timestamp is None:
            return None
        return float(abs(self.rgb_timestamp - self.depth_timestamp))


class RGBDEpisodeReader:
    def __init__(self, dataset_root: str | Path, camera: str, episode_index: int):
        self.dataset_root = Path(dataset_root)
        self.camera = camera
        self.episode_index = episode_index
        self._meta = self._load_meta()
        self._video_capture = None
        self._video_frame_count: int | None = None

        self.parquet_path = self._build_parquet_path()
        self.video_path = self._build_video_path()
        self.depth_key = f"observation.depth.{camera}"
        self.rgb_ts_key = f"observation.timestamps.{camera}"
        self.depth_ts_key = f"observation.timestamps.{camera}_depth"

        self._table = None
        self._ensure_features_exist()
        self._prefer_pyav = self.video_feature_info.get("video.codec") == "av1"

    def _load_meta(self) -> dict[str, Any]:
        meta_path = self.dataset_root / "meta" / "info.json"
        with meta_path.open() as f:
            return json.load(f)

    def _require_cv2(self):
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "opencv-python is required to read RGB video. "
                "Install it with `pip install opencv-python`."
            ) from exc
        return cv2

    def _read_rgb_with_pyav(self, frame_index: int, rgb_order: str = "rgb") -> np.ndarray:
        try:
            import av
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OpenCV could not decode the video, and PyAV is not installed. "
                "Install it with `pip install av` to enable software decoding."
            ) from exc

        with av.open(str(self.video_path)) as container:
            video_stream = container.streams.video[0]

            for idx, frame in enumerate(container.decode(video_stream)):
                if idx != frame_index:
                    continue

                rgb = frame.to_ndarray(format="rgb24")
                if rgb_order.lower() == "bgr":
                    return rgb[..., ::-1]
                return rgb

        raise RuntimeError(f"Failed to read frame {frame_index} from {self.video_path} with PyAV")

    def _iter_rgb_with_pyav(self, rgb_order: str = "rgb"):
        try:
            import av
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyAV is required to decode this video stream. "
                "Install it with `pip install av`."
            ) from exc

        with av.open(str(self.video_path)) as container:
            video_stream = container.streams.video[0]
            for frame in container.decode(video_stream):
                rgb = frame.to_ndarray(format="rgb24")
                if rgb_order.lower() == "bgr":
                    yield rgb[..., ::-1]
                else:
                    yield rgb

    def _build_parquet_path(self) -> Path:
        episode_chunk = self.episode_index // int(self._meta["chunks_size"])
        return (
            self.dataset_root
            / "data"
            / f"chunk-{episode_chunk:03d}"
            / f"episode_{self.episode_index:06d}.parquet"
        )

    def _build_video_path(self) -> Path:
        episode_chunk = self.episode_index // int(self._meta["chunks_size"])
        return (
            self.dataset_root
            / "videos"
            / f"chunk-{episode_chunk:03d}"
            / f"observation.images.{self.camera}"
            / f"episode_{self.episode_index:06d}.mp4"
        )

    def _ensure_features_exist(self) -> None:
        features = self._meta.get("features", {})
        required = [
            f"observation.images.{self.camera}",
            self.depth_key,
        ]
        missing = [key for key in required if key not in features]
        if missing:
            raise KeyError(
                f"Dataset does not contain the required features for camera "
                f"{self.camera}: {missing}"
            )

    @property
    def video_feature_info(self) -> dict[str, Any]:
        features = self._meta.get("features", {})
        return features.get(f"observation.images.{self.camera}", {}).get("info", {})

    def _load_table(self):
        if self._table is not None:
            return self._table

        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyarrow is required to read depth from parquet. "
                "Install it with `pip install pyarrow`."
            ) from exc

        self._table = pq.read_table(self.parquet_path)
        return self._table

    def _open_video(self):
        cv2 = self._require_cv2()
        if self._video_capture is not None:
            return self._video_capture

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self._video_capture = capture
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self._video_frame_count = int(frame_count) if frame_count > 0 else None
        return capture

    def __len__(self) -> int:
        table = self._load_table()
        return table.num_rows

    def get_depth(self, frame_index: int) -> np.ndarray:
        table = self._load_table()
        return np.asarray(
            table[self.depth_key][frame_index].as_py(),
            dtype=np.float32,
        )

    def get_timestamp(self, key: str, frame_index: int) -> float | None:
        table = self._load_table()
        if key not in table.column_names:
            return None
        value = table[key][frame_index].as_py()
        if value is None:
            return None
        if isinstance(value, list):
            return float(value[0]) if value else None
        return float(value)

    def get_rgb(self, frame_index: int, rgb_order: str = "rgb") -> np.ndarray:
        if self._prefer_pyav:
            return self._read_rgb_with_pyav(frame_index, rgb_order=rgb_order)

        cv2 = self._require_cv2()
        capture = self._open_video()
        ok = capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if not ok:
            self._prefer_pyav = True
            return self._read_rgb_with_pyav(frame_index, rgb_order=rgb_order)

        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            self._prefer_pyav = True
            return self._read_rgb_with_pyav(frame_index, rgb_order=rgb_order)

        if rgb_order.lower() == "bgr":
            return frame_bgr
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def get_frame(self, frame_index: int, rgb_order: str = "rgb") -> RGBDFrame:
        if frame_index < 0 or frame_index >= len(self):
            raise IndexError(f"frame_index {frame_index} out of range for episode length {len(self)}")

        return RGBDFrame(
            frame_index=frame_index,
            rgb=self.get_rgb(frame_index, rgb_order=rgb_order),
            depth=self.get_depth(frame_index),
            rgb_timestamp=self.get_timestamp(self.rgb_ts_key, frame_index),
            depth_timestamp=self.get_timestamp(self.depth_ts_key, frame_index),
        )

    def iter_frames(self, rgb_order: str = "rgb"):
        if self._prefer_pyav:
            for frame_index, rgb in enumerate(self._iter_rgb_with_pyav(rgb_order=rgb_order)):
                yield RGBDFrame(
                    frame_index=frame_index,
                    rgb=rgb,
                    depth=self.get_depth(frame_index),
                    rgb_timestamp=self.get_timestamp(self.rgb_ts_key, frame_index),
                    depth_timestamp=self.get_timestamp(self.depth_ts_key, frame_index),
                )
            return

        for frame_index in range(len(self)):
            yield self.get_frame(frame_index, rgb_order=rgb_order)

    def validate_alignment(self, max_delta_s: float | None = None) -> dict[str, float | int | None]:
        deltas = []
        for frame_index in range(len(self)):
            rgb_ts = self.get_timestamp(self.rgb_ts_key, frame_index)
            depth_ts = self.get_timestamp(self.depth_ts_key, frame_index)
            if rgb_ts is None or depth_ts is None:
                continue
            deltas.append(abs(rgb_ts - depth_ts))

        result = {
            "num_frames": len(self),
            "num_timestamp_pairs": len(deltas),
            "max_timestamp_delta_s": max(deltas) if deltas else None,
            "mean_timestamp_delta_s": float(np.mean(deltas)) if deltas else None,
        }

        if max_delta_s is not None and deltas:
            result["num_exceeding_threshold"] = sum(delta > max_delta_s for delta in deltas)

        if self._video_frame_count is not None:
            result["video_frame_count"] = self._video_frame_count

        return result

    def close(self) -> None:
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

    def __enter__(self) -> "RGBDEpisodeReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
