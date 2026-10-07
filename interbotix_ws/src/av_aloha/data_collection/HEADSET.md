# The Unity headset link

How this robot talks to the Meta Quest app: what runs where, which files matter,
and what you need checked out before any of it works.

## The one thing to know first

**The code is split across two repositories, and this one is the smaller half.**

| repo | what lives there |
|---|---|
| `giava` (this one) | the robot-side *integration*: adapting our cameras, arms and teleop loop to the link |
| [`Soltanilara/av-aloha-unity`](https://github.com/Soltanilara/av-aloha-unity), branch **`v2`** | the Unity app itself **and `gvlink`, the wire protocol library** |

`gvlink` is the protocol. It is deliberately **not vendored** into giava: the Unity
viewer and the Python sender have to agree byte for byte, so there is exactly one
copy of it and it sits next to the C# that has to match it. We import it from disk:

```python
_DEFAULT_GVLINK = os.path.expanduser("~/av-aloha-unity/Guided-Vision/python")
_GVLINK_PATH = os.environ.get("GVLINK_PATH", _DEFAULT_GVLINK)
```

So: **clone `av-aloha-unity` (branch `v2`) next to this repo, or set `GVLINK_PATH`,
or nothing in this document runs.** A missing checkout surfaces as an `ImportError`
from `gvlink_headset.py` and nothing else.

> Watch out for `~/dev/av-aloha-unity` — that is a stale WebRTC-era clone that
> predates gvlink entirely. The live checkout is `~/av-aloha-unity`.

Read `av-aloha-unity/docs/HEADSET_API.md` for the protocol itself. This file only
covers our side of it.

## Transport

There is no cloud and no signalling handshake. The headset finds this process by
LAN broadcast beacon (or is handed its address), opens a TCP control channel, and
video starts flowing back to wherever that connection came from.

| port | direction | carries |
|---|---|---|
| 15550 | robot broadcasts | discovery beacon — robot name, camera list, ports |
| 15551 | headset → robot, TCP | control channel: session setup, camera geometry, viewer stats, our feedback/markers |
| 15552 | robot → headset, UDP | video: fragmented H.264 (MJPEG for the Unity Editor), one atlas per eye |
| 15553 | headset → robot, UDP | input uplink: head/hand/controller poses + gaze, fixed size, at display rate (90 Hz) |

Datagram payloads are capped at 1400 bytes on LAN and 1180 over a Tailscale tunnel,
both sized so a frame never gets IP-fragmented — one lost fragment would cost the
whole datagram.

## Files in this repo

### The link itself — read these in this order

| file | role |
|---|---|
| [headset_link.py](headset_link.py) | **start here.** `make_headset()` — picks the transport so no call site has to care, and injects the OAK's measured camera geometry |
| [gvlink_headset.py](gvlink_headset.py) | **the main event** (~840 lines). Robot-side end of the gvlink connection: beacon, session handling, encode thread, input decode, feedback |
| [headset_utils.py](headset_utils.py) | `HeadsetData` / `HeadsetFeedback` (the structs crossing the boundary) and the left↔right-handed coordinate conversion Unity requires |
| [webrtc_headset.py](webrtc_headset.py) | **legacy.** The old Firestore-signalled WebRTC transport, kept because it pairs with older builds of the app. `GIAVA_HEADSET_TRANSPORT=webrtc` |

`GvLinkHeadset` keeps `WebRTCHeadset`'s public surface exactly, which is why the two
are interchangeable:

```python
headset = make_headset()
headset.run_in_thread()
while not headset.data_channel_open: ...
headset.send_images(left_bgr, right_bgr)   # returns immediately; see below
data = headset.receive_data()              # HeadsetData, or None if stale
headset.send_feedback(feedback)
headset.close()
```

Two behaviours that differ from the WebRTC version and will surprise you:

- **`send_images()` does not encode on the calling thread.** It drops the pair into a
  one-slot mailbox and returns. An encoder thread takes the newest pair. If capture
  outruns the encoder, intermediate pairs are *discarded, not queued* — a frame from
  two captures ago is worth less than the current one.
- **The viewer is told the real camera geometry** (rectified intrinsics + baseline)
  over the control channel and places the images from it, instead of the operator
  guessing a field of view in headset settings.

### Consumers — where the link is actually used

| file | role |
|---|---|
| [data_collection.py](data_collection.py) | the real episode-recording loop; the primary consumer |
| [teleop.py](teleop.py) | standalone teleoperation without recording |
| [camera_manager.py](camera_manager.py) | OAK stereo capture. `oak_gvlink_camera_params()` builds the rectified intrinsics + baseline **in gvlink wire form**, `OAK_GVLINK_CAMERA` holds them, `_push_camera_params_to_headset()` sends them. This is the geometry the viewer places each eye from |
| [transform_utils.py](transform_utils.py) | the pose maths. `HEAD_LOCAL_FWD` is **app-specific**: `+z` for gvlink, and it was `+x` on the WebRTC build. The stale value sat 76 deg off and put a constant yaw error into every arm's mapping — re-measured 2026-08-26 with `frame_calibrate.py` |

`headset_control.py` looks like it belongs here but is **not live** — nothing in the
recording or teleop loop imports it, only `oak_quality_debug.py` and its own
commented-out `__main__`. Treat it as dead until something claims it.

### Debug tools — the fastest way to understand the link

None of these need robots or ROS. **`oak_to_headset.py` is the shortest path from a
cold start to pixels in the headset**, and `headset_viz_debug.py` is the best way to
see what the headset is actually sending.

| file | what it does |
|---|---|
| [oak_to_headset.py](oak_to_headset.py) | minimal end-to-end OAK → headset stream. No arms |
| [headset_viz_debug.py](headset_viz_debug.py) | live viser view of head/hand/controller frames + hand keypoints, through the real `gvlink_headset.py`. Sends a synthetic pattern so no camera is needed |
| [headset_frame_probe.py](headset_frame_probe.py) | measures the headset's actual coordinate conventions one guided motion at a time, and prints the correction. Written to settle handedness bugs by measurement instead of argument |
| [frame_calibrate.py](frame_calibrate.py), [frame_rigid_check.py](frame_rigid_check.py) | headset↔robot frame calibration and its verification |

Also useful on the Unity side, in `av-aloha-unity/Guided-Vision/python`:
`mock_robot.py` (stands in for this robot) and `bench_receiver.py` (stands in for the
headset) — you can exercise the whole protocol with neither device present.

## Environment variables

Transport and video:

| var | default | meaning |
|---|---|---|
| `GVLINK_PATH` | `~/av-aloha-unity/Guided-Vision/python` | where the protocol library is |
| `GIAVA_HEADSET_TRANSPORT` | `gvlink` | `gvlink` (UDP) or `webrtc` (legacy Firestore) |
| `GIAVA_ROBOT_NAME` | — | name shown in the headset's robot picker |
| `GIAVA_HEADSET_FPS` / `_BITRATE_KBPS` / `_MIN_KBPS` | — | stream rate and bitrate floor |
| `GIAVA_HEADSET_ADAPT` | — | adaptive bitrate on/off |
| `GIAVA_HEADSET_CODEC` | h264 | `mjpeg` lets the Unity Editor decode in C# |
| `GIAVA_TUNNEL` | — | shrink datagrams for a Tailscale/WireGuard path |
| `GIAVA_X264_PRESET`, `GIAVA_X264_VBV_MS` | `ultrafast`, 100 | quality/CPU trade. `ultrafast` disables deblocking and CABAC — it is why throwing bitrate at a grainy picture does not help as much as it should |

Geometry and foveation:

| var | meaning |
|---|---|
| `GIAVA_HEADSET_CALIB`, `_HFOV`, `_BASELINE` | camera geometry, if not taken from the OAK calibration |
| `GIAVA_FOVEATION`, `GIAVA_COARSE_SCALE`, `GIAVA_FOVEA_SCALE`, `GIAVA_SACCADE_ZOOM` | two-layer foveal atlas |
| `GIAVA_CANVAS_W` / `_H`, `GIAVA_SRC_W` / `_H` | atlas and source sizes |
| `GIAVA_EYE_INWARD`, `GIAVA_EYE_SCALE`, `GIAVA_OAK_SWAP_EYES` | stereo comfort; also tunable live from the headset menu |

Input and debug:

| var | default | meaning |
|---|---|---|
| `GIAVA_CONTROL_STALE_S` | 0.15 | how old the newest input packet may be before `receive_data()` calls it nothing. Deliberately much tighter than the 0.5 s the gaze path tolerates — at 90 Hz, 0.5 s is ~45 missed packets driven on stale poses |
| `GIAVA_HEADSET_HAND_TRACKING` | — | accept hand-tracking wrist poses as controller poses |
| `GIAVA_HEADSET_STATS`, `GIAVA_MEMDEBUG` | — | periodic link stats / memory tracing |

## Two traps worth knowing

Both are load-bearing and both were bought with debugging time:

- **A zero quaternion is a normal packet, not corruption.** Unity's
  `GvInputUplink.Update` sends `default(GvControllerState)` for both controllers
  whenever the runtime is hand-tracking, and C# zero-initialises that struct — so
  rotation arrives as `(0,0,0,0)`. scipy ≥1.16 rejects it outright, which used to
  kill data collection from inside `receive_data()` *before* the tracking guard could
  see the zeroed position and hold the arm. `pose2mat()` now substitutes identity and
  lets the position speak: the zeros stay in `l_pos`/`r_pos`, the guard reads them as
  LOST, and no packet content can kill teleop mid-episode.
- **An untracked controller defaults to the origin**, so a hand-tracking session used
  to report both hands at the same point — which read as a self-collision and got
  refused. `receive_data()` resolves controller-vs-hand per side.
