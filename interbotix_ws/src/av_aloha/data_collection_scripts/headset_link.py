"""
Chooses which headset transport to use, so call sites do not have to care.

The Unity viewer moved from WebRTC/Firestore to a direct UDP stack (av-aloha-unity
branch v2). Both implementations are kept because they pair with different builds of
the app, and the UDP one is still new:

    GIAVA_HEADSET_TRANSPORT=gvlink   (default) -- v2 viewer, direct UDP
    GIAVA_HEADSET_TRANSPORT=webrtc             -- older viewer, Firestore signalling

Usage is identical either way:

    from headset_link import make_headset
    headset = make_headset()
    headset.run_in_thread()
"""

from __future__ import annotations

import os


def transport_name() -> str:
    return os.environ.get("GIAVA_HEADSET_TRANSPORT", "gvlink").strip().lower()


def make_headset(**kwargs):
    """Build the configured headset link. Unknown kwargs go to the implementation."""
    name = transport_name()

    if name in ("webrtc", "rtc"):
        from webrtc_headset import WebRTCHeadset
        # The UDP-only options mean nothing to the WebRTC class.
        for k in ("camera_params", "src_size", "ports", "beacon", "fps",
                  "bitrate_kbps", "min_bitrate_kbps", "canvas", "codec",
                  "foveation", "hfov_deg", "baseline_m", "calib", "name",
                  "coarse_scale", "fovea_scale", "saccade_zoom", "tunnel",
                  "adapt_bitrate", "rectified"):
            kwargs.pop(k, None)
        print("[headset] transport: webrtc (Firestore signalling)")
        return WebRTCHeadset(**kwargs)

    from gvlink_headset import GvLinkHeadset

    # Describe the real camera when the OAK calibration has been read, so the viewer
    # places the images from measured geometry instead of a guessed field of view.
    if "camera_params" not in kwargs:
        try:
            if __package__:
                from . import camera_manager as _cm
            else:
                import camera_manager as _cm
            params = _cm.OAK_GVLINK_CAMERA.get("params")
            if params:
                kwargs["camera_params"] = params
        except Exception:
            pass

    print("[headset] transport: gvlink (direct UDP)")
    return GvLinkHeadset(**kwargs)
