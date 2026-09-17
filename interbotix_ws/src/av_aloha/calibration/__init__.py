"""GIAVA camera/robot calibration and validation pipeline.

Stage one of:

    world coordinates -> robot pose -> camera pose -> calibrated and
    synchronised images -> camera extrinsics -> 3D reconstruction

Everything here establishes and MEASURES the front half of that chain.  No
reconstruction code lives in this package.

Scripts (each is runnable standalone; see README.md):

    frame_report.py        phase 1  frame hierarchy, what the EE pose is
    move_validation.py     phase 2  commanded vs achieved Cartesian motion
    camera_mount.py        phase 3  T_ee_camera, and what is actually known
    rs_intrinsics.py       phase 4  RealSense factory intrinsics
    charuco_capture.py     phase 5  collect board images
    charuco_calibrate.py   phase 5  OpenCV intrinsic calibration
    compare_intrinsics.py  phase 6  factory vs OpenCV
    sync_capture.py        phase 7  two-camera timing
    viser_cameras.py       phase 8  live viewer + paired capture
    selftest.py            no-hardware validation of all of the above

Libraries:

    common.py       paths, provenance, never-overwrite IO, SE(3) helpers
    kinematics.py   URDF frame graph, FK, driver<->URDF joint bridge
    rs_camera.py    RealSense capture with full timestamp provenance
    board.py        ChArUco board definition and detection

The frame-naming convention used throughout is stated in common.py and
repeated in every artifact these tools write: ``T_a_b`` maps points from
frame b into frame a, and is the pose of b expressed in a.
"""
