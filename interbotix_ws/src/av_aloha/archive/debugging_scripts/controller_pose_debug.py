import time
import viser

from interbotix_ws.src.av_aloha.data_collection_scripts.webrtc_headset import WebRTCHeadset
from transform_utils import xyzw_to_wxyz

# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADSET
    # --------------------------------------------------------

    headset = WebRTCHeadset()
    headset.run_in_thread()

    # --------------------------------------------------------
    # VISER
    # --------------------------------------------------------

    server = viser.ViserServer()

    server.scene.add_grid(
        "/ground",
        width=2,
        height=2,
    )

    # --------------------------------------------------------
    # WORLD FRAME
    # --------------------------------------------------------

    server.scene.add_frame(
        "/world",
        axes_length=0.3,
        axes_radius=0.01,
    )

    # --------------------------------------------------------
    # CONTROLLER FRAME
    # --------------------------------------------------------

    controller_frame = server.scene.add_frame(
        "/controller",
        axes_length=0.2,
        axes_radius=0.01,
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        headset_data = headset.receive_data()

        if headset_data is None:
            continue

        # ----------------------------------------------------
        # RAW HEADSET DATA
        # ----------------------------------------------------

        p = headset_data.r_pos
        q_xyzw = headset_data.r_quat

        # ----------------------------------------------------
        # VISUALIZE RAW CONTROLLER POSE
        # ----------------------------------------------------

        controller_frame.position = p

        controller_frame.wxyz = xyzw_to_wxyz(
            q_xyzw
        )

        # ----------------------------------------------------
        # DEBUG PRINTS
        # ----------------------------------------------------

        print("------------------------------------------------")
        print("POSITION:", p)
        print("QUAT XYZW:", q_xyzw)

        time.sleep(0.01)

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()