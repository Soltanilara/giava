#!/usr/bin/env python

import os
import glob
import numpy as np
import pandas as pd
import cv2

from pathlib import Path
from lerobot.datasets import LeRobotDataset

# ========= CONFIG =========

ROOT = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square"
OUT_SUFFIX = "block_square_grasp_only"

# Video path pattern from metadata
DATA_PATTERN = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATTERN = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

VIDEO_KEYS = [
    "observation.images.right_wrist",
    "observation.images.top_scene",
]

# Trimming hyperparameters
DELTA_THRESH = 1e-2        # min ||Δ joints|| to consider "motion started"
GRIPPER_CLOSE_THRESH = 0.0 # gripper_state < this → "closed"
N_AFTER_CLOSE = 10         # keep this many steps after close


# ========= HELPERS =========

def extract_grasp_segment(df: pd.DataFrame,
                          delta_thresh: float = DELTA_THRESH,
                          gripper_close_thresh: float = GRIPPER_CLOSE_THRESH,
                          n_after_close: int = N_AFTER_CLOSE) -> pd.DataFrame:
    """
    Given one episode DataFrame with 'action' and 'observation.state',
    return a trimmed DataFrame for [reach + grab + small lift].
    """
    actions = np.stack(df["action"].to_numpy())            # [T, 7]
    joint_actions = actions[:, :6]
    states = np.stack(df["observation.state"].to_numpy())  # [T, 7]
    gripper_states = states[:, 6]

    T = len(df)
    if T < 3:
        return None, None, None

    # 1) motion start: first index where ||Δjoint_action|| > delta_thresh
    delta = np.linalg.norm(joint_actions[1:] - joint_actions[:-1], axis=1)
    motion_idx = np.where(delta > delta_thresh)[0]
    start_idx = int(motion_idx[0]) if len(motion_idx) > 0 else 0

    # 2) first gripper-close index
    close_idx_candidates = np.where(gripper_states < gripper_close_thresh)[0]
    if len(close_idx_candidates) > 0:
        t_close = int(close_idx_candidates[0])
    else:
        t_close = T - 1  # no close: keep until end

    # 3) end: a few steps after close
    end_idx = min(t_close + n_after_close, T - 1)

    df_trim = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)

    depth_cols = [
        "observation.depth.right_wrist",
        "observation.depth.top_scene",
        "observation.depth_intrinsics.right_wrist",
        "observation.depth_intrinsics.top_scene",
        "observation.timestamps.right_wrist_depth",
        "observation.timestamps.top_scene_depth",
    ]

    df_trim = df_trim.drop(columns=[c for c in depth_cols if c in df_trim.columns])
    return df_trim, start_idx, end_idx


def load_episode_indices(path: str):
    """
    Parse episode_chunk and episode_index from a path matching DATA_PATTERN.
    """
    # example path:
    # .../block_square/20260528_131838/data/chunk-000/episode_000000.parquet
    basename = os.path.basename(path)          # episode_000000.parquet
    episode_index = int(basename.split("_")[1].split(".")[0])
    # chunk-000
    chunk_name = os.path.basename(os.path.dirname(path))   # chunk-000
    episode_chunk = int(chunk_name.split("-")[1])
    return episode_chunk, episode_index


def trim_video_segment(video_path: str, start_idx: int, end_idx: int) -> tuple[list, float]:
    """
    Load a video and return frames[start_idx:end_idx+1] and fps.
    Assumes 1:1 mapping between parquet row index and frame index.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 50.0  # fallback to dataset fps

    frames = []
    for idx in range(start_idx, end_idx + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: failed to read frame {idx} from {video_path}")
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def write_video(path: str, frames: list, fps: float):
    """
    Write frames to an mp4 video at given fps.
    """
    if not frames:
        print(f"Warning: no frames to write for {path}")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"  saved video: {path}")

READ_ROOT = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_131838"
WRITE_ROOT = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square_grasp_only"

# ========= MAIN =========

def main():
    pattern = os.path.join(
        READ_ROOT,
        "data",
        "chunk-*",
        "episode_*.parquet",
    )
    episode_paths = sorted(glob.glob(pattern))
    print(f"Found {len(episode_paths)} episodes in {READ_ROOT}")

    if not episode_paths:
        return

    for ep_path in episode_paths:
        print(f"\nProcessing episode: {ep_path}")
        df = pd.read_parquet(ep_path)

        df_trim, start_idx, end_idx = extract_grasp_segment(df)
        if df_trim is None or len(df_trim) == 0:
            print("  skipping episode: no valid grasp segment found")
            continue

        print(f"  original T={len(df)}, trimmed T={len(df_trim)} (rows {start_idx}–{end_idx})")

        episode_chunk, episode_index = load_episode_indices(ep_path)

        # read videos from READ_ROOT
        trimmed_videos = {}
        for video_key in VIDEO_KEYS:
            rel_video = VIDEO_PATTERN.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
            )
            video_path = os.path.join(READ_ROOT, rel_video)
            frames, fps = trim_video_segment(video_path, start_idx, end_idx)
            trimmed_videos[video_key] = (frames, fps)

        # write parquet under WRITE_ROOT mirroring the same relative path
        rel = os.path.relpath(ep_path, READ_ROOT)  # e.g. data/chunk-000/episode_000000.parquet
        out_parquet_path = os.path.join(WRITE_ROOT, rel)
        os.makedirs(os.path.dirname(out_parquet_path), exist_ok=True)

        df_trim.to_parquet(out_parquet_path, index=False)
        print(f"  saved trimmed parquet: {out_parquet_path}")

        base_no_ext = os.path.splitext(out_parquet_path)[0]

        # write aligned videos under WRITE_ROOT next to parquet
        for video_key, (frames, fps) in trimmed_videos.items():
            rel_video = VIDEO_PATTERN.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
            )
            out_video_path = os.path.join(WRITE_ROOT, rel_video)
            os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
            write_video(out_video_path, frames, fps)

        

        # write actions CSV under WRITE_ROOT
        actions = np.stack(df_trim["action"].to_numpy())   # [T', 7]
        joints = actions[:, :6]
        gripper = actions[:, 6:7]
        arr = np.concatenate([joints, gripper], axis=1)
        cols = [f"joint{i}" for i in range(6)] + ["gripper"]
        csv_path = base_no_ext + "_actions.csv"
        pd.DataFrame(arr, columns=cols).to_csv(csv_path, index_label="t")
        print(f"  saved actions CSV: {csv_path}")

if __name__ == "__main__":
    main()