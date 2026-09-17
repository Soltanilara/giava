from pathlib import Path
import shutil
import pandas as pd
import numpy as np

DATASETS = [
    Path("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_131838"),
    Path("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_164157"),
]

REMOVE_COLUMNS = [
    "observation.depth.right_wrist",
    "observation.depth.top_scene",
    "observation.depth_intrinsics.right_wrist",
    "observation.depth_intrinsics.top_scene",
    "observation.timestamps.right_wrist_depth",
    "observation.timestamps.top_scene_depth",
]

IDLE_CFG = np.array([0.0, -1.27, 0.99, 0.0, 0.35, 0.0], dtype=np.float32)
GRIPPER_CLOSED = -1.5
RUN_LEN = 10  # need 10 consecutive closed steps


def trim_to_grasp_segment(df: pd.DataFrame):
    """
    Return trimmed df and original row indices kept, according to your rule:

    - remove timesteps whose joint action == IDLE_CFG
    - among remaining steps, keep all
    - once there is a run of RUN_LEN consecutive steps with gripper == GRIPPER_CLOSED,
      keep up to that RUN_LEN-th closed step and drop everything after
    """
    if "action" not in df.columns or "observation.state" not in df.columns:
        return None, None

    actions = np.stack(df["action"].to_numpy())             # [T, 7]
    states = np.stack(df["observation.state"].to_numpy())   # [T, 7]

    joint_actions = actions[:, :6]
    gripper_actions = actions[:, 6]

    T = len(df)
    if T < 3:
        return None, None

    # original row indices (these correspond to original frame indices)
    indices = df.index.to_numpy()

    # 1) drop idle config
    is_idle = np.all(np.isclose(joint_actions, IDLE_CFG, atol=1e-6), axis=1)
    keep = ~is_idle

    valid_idx = indices[keep]
    if len(valid_idx) == 0:
        return None, None

    valid_gripper = gripper_actions[keep]

    # 2) find first run of RUN_LEN consecutive closed-gripper steps (-1.5)
    is_closed = np.isclose(valid_gripper, GRIPPER_CLOSED, atol=1e-6)
    cut_after = None

    if len(valid_gripper) >= RUN_LEN:
        for start in range(0, len(valid_gripper) - RUN_LEN + 1):
            window = is_closed[start:start + RUN_LEN]
            if window.all():
                cut_after = start + RUN_LEN - 1
                break

    # 3) choose final indices
    if cut_after is None:
        final_indices = valid_idx
    else:
        final_indices = valid_idx[:cut_after + 1]

    if len(final_indices) == 0:
        return None, None

    final_indices = np.sort(final_indices)

    # IMPORTANT: keep original index to preserve alignment with videos
    df_trim = df.loc[final_indices]
    return df_trim, final_indices


def clean_dataset(source: Path):
    dest = source.parent / f"{source.name}_rgb_only_grasp"

    if dest.exists():
        raise RuntimeError(
            f"{dest} already exists. "
            "Delete it or choose another name."
        )

    print()
    print("=" * 80)
    print(f"Processing: {source}")
    print(f"Output:     {dest}")
    print("=" * 80)

    dest.mkdir(parents=True)

    # copy meta as-is (we'll update episode lengths later if needed)
    print("Copying meta...")
    shutil.copytree(source / "meta", dest / "meta")

    # copy videos as-is (no trimming, alignment preserved via indices)
    print("Copying videos...")
    shutil.copytree(source / "videos", dest / "videos")

    # rewrite parquet files
    src_data = source / "data" / "chunk-000"
    dst_data = dest / "data" / "chunk-000"
    dst_data.mkdir(parents=True)

    parquet_files = sorted(src_data.glob("episode_*.parquet"))

    for i, parquet_file in enumerate(parquet_files):
        episode_index = int(parquet_file.stem.split("_")[1])
        print(f"[{i+1}/{len(parquet_files)}] {parquet_file.name} (episode {episode_index})")

        df = pd.read_parquet(parquet_file)

        # apply grasp-only trimming
        df_trim, kept_indices = trim_to_grasp_segment(df)
        if df_trim is None or len(df_trim) == 0:
            print("  -> no valid grasp segment found, skipping episode")
            continue

        # drop depth-related columns
        cols_to_drop = [c for c in REMOVE_COLUMNS if c in df_trim.columns]
        df_trim = df_trim.drop(columns=cols_to_drop)

        # save trimmed parquet with original index preserved
        out_parquet = dst_data / parquet_file.name
        df_trim.to_parquet(out_parquet, index=True)
        print(f"  saved trimmed parquet: {out_parquet} "
              f"(original T={len(df)}, trimmed T={len(df_trim)})")

    print()
    print("Finished:", dest)


for dataset in DATASETS:
    clean_dataset(dataset)

print()
print("All datasets processed.")