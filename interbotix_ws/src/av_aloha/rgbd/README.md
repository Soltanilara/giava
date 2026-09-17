# Reading RGB and Depth

This folder contains utilities for:

- depth from episode `.parquet` files
- RGB from matching `.mp4` video files
- alignment information between the two
- full-episode export into one RGB video and one depth video
- interactive depth inspection by clicking on a pixel

## Files

- `rgbd_episode_reader.py`
  - reusable Python reader for one episode
- `extract_rgbd_episode.py`
  - command-line script for exporting a full episode
- `click_depth_viewer.py`
  - interactive tool for clicking a pixel and reading its depth in meters

## What these scripts assume

They are designed for dataset runs where:

- depth is stored in parquet fields such as:
  - `observation.depth.right_wrist`
  - `observation.depth.top_scene`
- RGB is stored as video under paths such as:
  - `videos/chunk-000/observation.images.right_wrist/episode_000000.mp4`
  - `videos/chunk-000/observation.images.top_scene/episode_000000.mp4`

They match RGB and depth primarily by `frame_index`.

They also read timestamps when available:

- `observation.timestamps.<camera>`
- `observation.timestamps.<camera>_depth`

## Dependencies

Install these first:

```bash
pip install numpy pyarrow opencv-python av
```

## Command-line usage

From the repo root:

```bash
cd /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts
```

Example:

```bash
python3 rgbd/extract_rgbd_episode.py \
  dataset/lerobot/block_square/20260528_164157 \
  --camera right_wrist \
  --episode 0 \
  --print-summary
```

What this does:

- reads the entire episode
- reads RGB from the matching source video
- reads depth from the parquet episode
- matches RGB and depth by frame index
- writes one RGB output video
- writes one depth output video
- writes a `summary.json`

By default, outputs are saved under:

```text
rgbd/outputs/<task>/<run>/<camera>/episode_xxxxxx/
```

For the example above, the folder will be:

```text
rgbd/outputs/block_square/20260528_164157/right_wrist/episode_000000/
```

Saved files include:

- `rgb_episode_000000.mp4`
- `depth_episode_000000.mp4`
- `summary.json`

Important note about `depth_episode.mp4`:

- it is a colorized visualization of depth, not raw metric depth
- it uses one fixed depth range for the whole episode
- this avoids flicker caused by frame-by-frame normalization
- invalid or zero-depth pixels are rendered as black

## Arguments

- `dataset_root`
  - path to a dataset run, for example:
  - `dataset/lerobot/block_square/20260528_164157`
- `--camera`
  - camera name such as `right_wrist` or `top_scene`
- `--episode`
  - episode index
- `--print-summary`
  - prints whole-episode alignment statistics
- `--output-dir`
  - directory where extracted files should be saved
- `--max-ts-diff`
  - optional threshold in seconds for timestamp alignment checks

The `summary.json` file also records:

- source parquet path
- source RGB video path
- output video paths
- codec / pixel format / fps
- episode-wide depth min/max
- visualization depth min/max used for the depth video

## Python usage

```python
from reading_RGB_depth.rgbd_episode_reader import RGBDEpisodeReader

with RGBDEpisodeReader(
    dataset_root="dataset/lerobot/block_square/20260528_164157",
    camera="right_wrist",
    episode_index=0,
) as reader:
    frame = reader.get_frame(0)

    rgb = frame.rgb
    depth = frame.depth

    print(rgb.shape, rgb.dtype)
    print(depth.shape, depth.dtype)
    print(frame.rgb_timestamp, frame.depth_timestamp, frame.timestamp_delta)
```

## Returned data

- `frame.rgb`
  - RGB image as a NumPy array
- `frame.depth`
  - depth map as a `float32` NumPy array in meters
- `frame.rgb_timestamp`
  - RGB timestamp if present
- `frame.depth_timestamp`
  - depth timestamp if present
- `frame.timestamp_delta`
  - absolute difference between RGB and depth timestamps

## Notes

- Depth is expected to come from parquet, not image files.
- RGB is expected to come from video, not parquet arrays.
- Some newer datasets in this repo contain RGB only and no depth fields.
- If the dataset does not contain the requested depth key, the reader raises a clear error.
- `depth_episode.mp4` is a colorized visualization of depth, not raw metric depth values.
- The export script tries to use the same video settings as the dataset metadata, such as `av1`, `yuv420p`, and the dataset FPS.
- AV1 source videos are decoded with PyAV when needed.

## Interactive depth inspection

Example:

```bash
python3 rgbd/click_depth_viewer.py \
  dataset/lerobot/block_square/20260528_164157 \
  --camera right_wrist \
  --episode 0 \
  --frame 0
```

What this does:

- opens one RGB frame and one depth visualization side by side
- lets you click on a pixel
- prints the depth value in meters
- marks the clicked pixel on both images

Usage notes:

- left half of the window: RGB
- right half of the window: depth visualization
- clicking either half queries the depth at that image coordinate
- press `q` or `Esc` to quit
