from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import json
import math

root = Path("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/20260601_045338")
video_root = root / "videos" / "chunk-000"

mask_dir = video_root / "observation.images.top_scene_mask"
preferred = mask_dir / "episode_000000_mask.mp4"

if preferred.is_file():
    mp4 = preferred
else:
    candidates = sorted(mask_dir.glob("*_mask.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No mask mp4 files found under {mask_dir}")
    print("Preferred mask file not found.")
    print("Using:", candidates[0])
    print("\nOther mask candidates:")
    for p in candidates[:20]:
        print(" ", p)
    mp4 = candidates[0]

print("Chosen file:", mp4)

out = Path("output/mp4_mask_verify")
out.mkdir(parents=True, exist_ok=True)

frames_dir = out / mp4.stem
frames_dir.mkdir(parents=True, exist_ok=True)

proc = subprocess.run(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(mp4),
        str(frames_dir / "frame_%06d.png"),
    ],
    capture_output=True,
    text=True,
)

if proc.returncode != 0:
    (out / "ffmpeg_error.txt").write_text(proc.stderr)
    raise RuntimeError(proc.stderr)

frames = sorted(frames_dir.glob("frame_*.png"))
if not frames:
    raise RuntimeError(f"No frames extracted from {mp4}")

probe = subprocess.run(
    [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames,width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        str(mp4),
    ],
    capture_output=True,
    text=True,
)

ffprobe_data = json.loads(probe.stdout) if probe.stdout.strip() else {}

info = {
    "mp4": str(mp4),
    "frame_count_extracted": len(frames),
    "first_frame": str(frames[0]),
    "last_frame": str(frames[-1]),
    "ffprobe": ffprobe_data,
}
(out / "info.json").write_text(json.dumps(info, indent=2))
(out / "frames.txt").write_text("\n".join(str(p) for p in frames))
(out / "ffprobe.json").write_text(json.dumps(ffprobe_data, indent=2))

cols = 4
thumb_w, thumb_h = 320, 180
n_preview = min(len(frames), 12)
rows = math.ceil(n_preview / cols)

sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h + 140), "white")
draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except:
    font = ImageFont.load_default()

draw.text((20, 15), f"MP4: {mp4.name}", fill="black", font=font)
draw.text((20, 45), f"Frame count: {len(frames)}", fill="black", font=font)
draw.text((20, 75), f"Source: {mp4.parent.name}", fill="black", font=font)
draw.text((20, 105), f"Frames dir: {frames_dir}", fill="black", font=font)

for i, p in enumerate(frames[:n_preview]):
    im = Image.open(p).convert("RGB").resize((thumb_w - 10, thumb_h - 40))
    x = (i % cols) * thumb_w + 5
    y = (i // cols) * thumb_h + 125
    sheet.paste(im, (x, y))
    draw.text((x, y + thumb_h - 32), p.name, fill="black", font=font)

sheet.save(out / "contact_sheet.png")

print("Wrote:", out / "info.json")
print("Wrote:", out / "frames.txt")
print("Wrote:", out / "ffprobe.json")
print("Wrote:", out / "contact_sheet.png")