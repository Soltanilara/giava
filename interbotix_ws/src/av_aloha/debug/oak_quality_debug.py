"""
Interactive OAK stereo streaming tuner

Keyboard controls happen in terminal.

Resolution:
    1 -> 320x240
    2 -> 640x480
    3 -> 800x600
    4 -> 1280x800

FPS:
    q -> 15
    w -> 20
    e -> 25
    r -> 30
    t -> 40
    y -> 50
    u -> 60
    i -> 75
    o -> 90
    p -> 120

Quit:
    x
"""

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import depthai as dai
import time
import threading
import sys
import termios
import tty

from interbotix_ws.src.av_aloha.data_collection_scripts.headset_link import make_headset
from headset_control import HeadsetFullControl as HeadsetControl
from headset_utils import HeadsetFeedback


# ============================================================
# KEYBOARD CONFIG
# ============================================================

RESOLUTIONS = {
    '1': (320, 240),
    '2': (640, 480),
    '3': (800, 600),
    '4': (1280, 800),
}

FPS_OPTIONS = {
    'q': 15,
    'w': 20,
    'e': 25,
    'r': 30,
    't': 40,
    'y': 50,
    'u': 60,
    'i': 75,
    'o': 90,
    'p': 120,
}

current_resolution = (640, 480)
current_fps = 30

running = True


# ============================================================
# HEADSET
# ============================================================

headset = make_headset()
headset.run_in_thread()

headset_control = HeadsetControl()
feedback = HeadsetFeedback()

headset_control.reset()

# ============================================================
# PIPELINE
# ============================================================

pipeline = None
q_left = None
q_right = None

def build_pipeline(resolution, fps):

    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B
    )

    cam_right = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C
    )

    left_out = cam_left.requestOutput(
        resolution,
        type=dai.ImgFrame.Type.GRAY8,
        fps=fps,
    )

    right_out = cam_right.requestOutput(
        resolution,
        type=dai.ImgFrame.Type.GRAY8,
        fps=fps,
    )

    q_left = left_out.createOutputQueue()
    q_right = right_out.createOutputQueue()

    return pipeline, q_left, q_right


def restart_pipeline():
    global pipeline
    global q_left
    global q_right

    try:
        if pipeline is not None:
            pipeline.stop()
            time.sleep(0.2)
    except:
        pass

    print("\n" + "=" * 60)
    print(f"Resolution: {current_resolution}")
    print(f"FPS: {current_fps}")
    print("=" * 60)

    pipeline, q_left, q_right = build_pipeline(
        current_resolution,
        current_fps,
    )

    pipeline.start()

    time.sleep(0.5)

# ============================================================
# TERMINAL KEYBOARD INPUT
# ============================================================

def keyboard_listener():

    global current_resolution
    global current_fps
    global running

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:

        tty.setcbreak(fd)

        while running:

            key = sys.stdin.read(1)

            # Quit
            if key == 'x':

                print("\nExiting...")

                running = False

                break

            # Resolution
            if key in RESOLUTIONS:

                current_resolution = RESOLUTIONS[key]

                restart_pipeline()

            # FPS
            if key in FPS_OPTIONS:

                current_fps = FPS_OPTIONS[key]

                restart_pipeline()

    finally:

        termios.tcsetattr(
            fd,
            termios.TCSADRAIN,
            old_settings,
        )


# ============================================================
# START
# ============================================================

print("\nControls:")
print("Resolution:")
print(" 1 -> 320x240")
print(" 2 -> 640x480")
print(" 3 -> 800x600")
print(" 4 -> 1280x800")
print("\nFPS:")
print(" q -> 15")
print(" w -> 20")
print(" e -> 25")
print(" r -> 30")
print(" t -> 40")
print(" y -> 50")
print(" u -> 60")
print(" i -> 75")
print(" o -> 90")
print(" p -> 120")
print("\nQuit:")
print(" x")
print()

restart_pipeline()

keyboard_thread = threading.Thread(
    target=keyboard_listener,
    daemon=True,
)

keyboard_thread.start()


# ============================================================
# MAIN LOOP
# ============================================================

frame_counter = 0
last_print_time = time.time()

while running:

    try:

        if not pipeline.isRunning():

            print("Pipeline stopped unexpectedly")

            restart_pipeline()

            continue

        left_img = q_left.get().getCvFrame()
        right_img = q_right.get().getCvFrame()

        headset.send_images(left_img, right_img)

        headset_data = headset.receive_data()

        frame_counter += 1

        now = time.time()

        if now - last_print_time >= 1.0:

            fps_actual = frame_counter / (now - last_print_time)

            print(
                f"\rStreaming | "
                f"{current_resolution} | "
                f"Target FPS: {current_fps} | "
                f"Actual Loop FPS: {fps_actual:.1f}",
                end=""
            )

            frame_counter = 0
            last_print_time = now

    except Exception as e:

        print(f"\nERROR: {e}")

        time.sleep(1)

        try:
            restart_pipeline()
        except Exception as restart_error:
            print(f"Restart failed: {restart_error}")


# ============================================================
# CLEANUP
# ============================================================

try:
    pipeline.stop()
except:
    pass

print("\nExited cleanly.")
