"""
Instrumented stereo WebRTC streaming pipeline for OAK cameras.

Features:
- FPS measurement
- CPU monitoring
- Frame age / latency measurement
- Dropped frame counting
- Visual frame overlays
- Side-by-side stereo option
- OpenCV local preview
- Queue instrumentation

Requires:
    pip install psutil opencv-python aiortc av depthai

Optional:
    pip install pynvml
"""

import asyncio
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import depthai as dai
import numpy as np
import psutil

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpSender,
    VideoStreamTrack,
)
from av import VideoFrame


# ============================================================
# CONFIG
# ============================================================

USE_SIDE_BY_SIDE = True

WIDTH = 1280
HEIGHT = 800

FPS = 60

SHOW_LOCAL_PREVIEW = True

QUEUE_SIZE = 1

IMAGE_FORMAT = "bgr24"

FORCED_CODEC = "video/H264"


# ============================================================
# UTILITIES
# ============================================================

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]

    codecs = RTCRtpSender.getCapabilities(kind).codecs

    transceiver = next(
        t for t in pc.getTransceivers()
        if t.sender == sender
    )

    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


@dataclass
class FramePacket:
    frame_id: int
    capture_timestamp: float
    image: np.ndarray


class FPSCounter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.last_time = time.time()

    def tick(self):
        self.count += 1

        now = time.time()

        if now - self.last_time >= 1.0:
            fps = self.count / (now - self.last_time)

            print(f"[{self.name}] FPS: {fps:.2f}")

            self.count = 0
            self.last_time = now


# ============================================================
# VIDEO TRACK
# ============================================================

class BufferVideoStreamTrack(VideoStreamTrack):

    def __init__(
        self,
        buffer_size=1,
        image_format="bgr24",
        max_fps=60,
    ):
        super().__init__()

        self.queue = queue.Queue(maxsize=buffer_size)

        self.image_format = image_format

        self.max_fps = max_fps

        self.last_send_time = time.time()

        self.last_packet = None

        self.dropped_frames = 0

        self.send_fps = FPSCounter("SEND")

    def add_frame(self, packet: FramePacket):

        try:
            self.queue.put_nowait(packet)

        except queue.Full:

            self.dropped_frames += 1

            try:
                self.queue.get_nowait()
                self.queue.put_nowait(packet)

            except Exception:
                pass

    async def get_packet(self):

        while True:

            try:
                packet = self.queue.get_nowait()

                self.last_packet = packet

                return packet

            except queue.Empty:

                if self.last_packet is not None:
                    return self.last_packet

                await asyncio.sleep(0)

    async def recv(self):

        pts, time_base = await self.next_timestamp()

        packet = await self.get_packet()

        frame_age_ms = (
            time.time() - packet.capture_timestamp
        ) * 1000.0

        frame = packet.image.copy()

        cv2.putText(
            frame,
            f"Frame: {packet.frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Age: {frame_age_ms:.1f} ms",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Dropped: {self.dropped_frames}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        video_frame = VideoFrame.from_ndarray(
            frame,
            format=self.image_format,
        )

        video_frame.pts = pts
        video_frame.time_base = time_base

        elapsed = time.time() - self.last_send_time

        await asyncio.sleep(
            max(1 / self.max_fps - elapsed, 0)
        )

        self.last_send_time = time.time()

        self.send_fps.tick()

        return video_frame


# ============================================================
# WEBRTC
# ============================================================

class WebRTCStreamer:

    def __init__(self):

        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer("stun:stun.l.google.com:19302")
            ])
        )

        self.track = BufferVideoStreamTrack(
            buffer_size=QUEUE_SIZE,
            image_format=IMAGE_FORMAT,
            max_fps=FPS,
        )

        sender = self.pc.addTrack(self.track)

        force_codec(self.pc, sender, FORCED_CODEC)

    def send_frame(self, packet: FramePacket):
        self.track.add_frame(packet)


# ============================================================
# OAK PIPELINE
# ============================================================

pipeline = dai.Pipeline()

cam_left = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_B
)

cam_right = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_C
)

left_out = cam_left.requestOutput(
    (WIDTH, HEIGHT),
    type=dai.ImgFrame.Type.GRAY8,
    fps=FPS,
)

right_out = cam_right.requestOutput(
    (WIDTH, HEIGHT),
    type=dai.ImgFrame.Type.GRAY8,
    fps=FPS,
)

q_left = left_out.createOutputQueue()
q_right = right_out.createOutputQueue()

pipeline.start()


# ============================================================
# STREAMER
# ============================================================

streamer = WebRTCStreamer()

capture_fps = FPSCounter("CAPTURE")

process = psutil.Process()

frame_id = 0

last_stats_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while pipeline.isRunning():

    left = q_left.get().getCvFrame()
    right = q_right.get().getCvFrame()

    capture_timestamp = time.time()

    # Convert mono -> BGR for visualization/WebRTC compatibility
    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    # Add eye labels
    cv2.putText(
        left,
        "LEFT",
        (20, HEIGHT - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4,
    )

    cv2.putText(
        right,
        "RIGHT",
        (20, HEIGHT - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),
        4,
    )

    # Side-by-side stereo
    if USE_SIDE_BY_SIDE:

        stereo = np.concatenate(
            [left, right],
            axis=1,
        )

    else:

        stereo = left

    packet = FramePacket(
        frame_id=frame_id,
        capture_timestamp=capture_timestamp,
        image=stereo,
    )

    streamer.send_frame(packet)

    # Local preview
    if SHOW_LOCAL_PREVIEW:

        preview = cv2.resize(
            stereo,
            (1600, 600),
        )

        cv2.imshow("Stereo Preview", preview)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    # FPS
    capture_fps.tick()

    # CPU stats
    if time.time() - last_stats_time >= 1.0:

        cpu = process.cpu_percent(interval=0)

        print(f"[CPU] {cpu:.1f}%")

        last_stats_time = time.time()

    frame_id += 1


cv2.destroyAllWindows()