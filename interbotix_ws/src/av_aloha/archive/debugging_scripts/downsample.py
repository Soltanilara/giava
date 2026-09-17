from pathlib import Path
import shutil
import pandas as pd

SOURCE = Path(
    "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp cube/20260528_235229"
)

DEST = Path(
    "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube_downsampled_20"
)

BAD_EPISODES = {3}
DOWNSAMPLE = 4

if DEST.exists():
    raise RuntimeError(f"{DEST} already exists")

DEST.mkdir(parents=True)

print("Copying meta...")
shutil.copytree(SOURCE / "meta", DEST / "meta")

print("Copying videos...")
shutil.copytree(SOURCE / "videos", DEST / "videos")

src_data = SOURCE / "data" / "chunk-000"
dst_data = DEST / "data" / "chunk-000"
dst_data.mkdir(parents=True)

parquet_files = sorted(src_data.glob("*.parquet"))

new_episode_idx = 0

for episode_idx, parquet_file in enumerate(parquet_files):

    if episode_idx in BAD_EPISODES:
        print(f"Skipping bad episode {episode_idx}")
        continue

    print(f"Processing episode {episode_idx}")

    df = pd.read_parquet(parquet_file)

    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)

    if "frame_index" in df.columns:
        df["frame_index"] = range(len(df))

    if "episode_index" in df.columns:
        df["episode_index"] = new_episode_idx

    output_file = (
        dst_data
        / f"episode_{new_episode_idx:06d}.parquet"
    )

    df.to_parquet(output_file, index=False)

    new_episode_idx += 1

print("Done.")
