import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.animation import FuncAnimation

from yourdfpy import URDF
import pyroki as pk

URDF_PATH = "/home/devi/giava/giava.urdf"

urdf = URDF.load(URDF_PATH)

robot = pk.Robot.from_urdf(urdf)

LEFT_ARM_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
]

left_arm_indices = [
    robot.joints.actuated_names.index(name)
    for name in LEFT_ARM_NAMES
]

data = np.load(
    "ros_ik_trajectory_logs/orientation_debug.npz",
    allow_pickle=True,
)

t = data["time"]

q = data["q"]

dq = data["dq"]

target_pos = data["target_position"]

actual_pos = data["actual_position"]

solve_time = data["solve_time"]

modes = data["trajectory_mode"]

mode_change_idxs = np.where(
    modes[:-1] != modes[1:]
)[0]

def add_mode_markers():

    for idx in mode_change_idxs:

        plt.axvline(
            t[idx],
            color="red",
            linestyle="--",
            alpha=0.5,
        )

        ymax = plt.ylim()[1]

        plt.text(
            t[idx],
            0.95 * ymax,
            modes[idx],
            rotation=90,
            fontsize=8,
        )

# =========================================================
# SAVE ALL PLOTS
# =========================================================

import os

PLOT_DIR = "ros_ik_trajectory_logs/orientation_plots_2"

os.makedirs(
    PLOT_DIR,
    exist_ok=True,
)

# =========================================================
# Joint Positions
# =========================================================

plt.figure(figsize=(12,6))

for idx in left_arm_indices:

    plt.plot(
        t,
        q[:, idx],
        label=f"joint_{idx}",
    )

plt.title("Joint Positions")

plt.xlabel("Time (s)")

plt.ylabel("Position (rad)")

plt.legend()

plt.grid(True)

add_mode_markers()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/joint_positions.png",
    dpi=300,
    bbox_inches="tight",
)

# =========================================================
# Joint Velocities
# =========================================================

plt.figure(figsize=(12,6))

for i in range(dq.shape[1]):

    plt.plot(
        t,
        dq[:, i],
        label=f"joint_{i}",
    )

plt.title("Joint Velocities")

plt.xlabel("Time (s)")

plt.ylabel("Velocity (rad/s)")

plt.legend()

plt.grid(True)

add_mode_markers()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/joint_velocities.png",
    dpi=300,
    bbox_inches="tight",
)

# =========================================================
# 3D EE Trajectory
# =========================================================

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(
    111,
    projection="3d",
)

ax.plot(
    target_pos[:,0],
    target_pos[:,1],
    target_pos[:,2],
    label="target",
)

ax.plot(
    actual_pos[:,0],
    actual_pos[:,1],
    actual_pos[:,2],
    label="actual",
)

ax.scatter(
    target_pos[0,0],
    target_pos[0,1],
    target_pos[0,2],
    label="start",
)

ax.scatter(
    target_pos[-1,0],
    target_pos[-1,1],
    target_pos[-1,2],
    label="end",
)

ax.set_title("EE Trajectory")

ax.set_xlabel("X")

ax.set_ylabel("Y")

ax.set_zlabel("Z")

ax.set_box_aspect([1,1,1])

ax.legend()

fig.tight_layout()

fig.savefig(
    f"{PLOT_DIR}/ee_trajectory_3d.png",
    dpi=300,
    bbox_inches="tight",
)

# =========================================================
# Position Error Norm
# =========================================================

position_error = np.linalg.norm(
    target_pos - actual_pos,
    axis=1,
)

plt.figure(figsize=(12,4))

plt.plot(
    t,
    position_error,
)

plt.title("EE Position Error")

plt.xlabel("Time (s)")

plt.ylabel("Error (m)")

plt.grid(True)

add_mode_markers()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/position_error.png",
    dpi=300,
    bbox_inches="tight",
)

# =========================================================
# Joint Velocity Norm
# =========================================================

dq_norm = np.linalg.norm(
    dq,
    axis=1,
)

plt.figure(figsize=(12,4))

plt.plot(
    t,
    dq_norm,
)

plt.title("Joint Velocity Norm")

plt.xlabel("Time (s)")

plt.ylabel("||dq||")

plt.grid(True)

add_mode_markers()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/joint_velocity_norm.png",
    dpi=300,
    bbox_inches="tight",
)

# =========================================================
# IK Solve Time
# =========================================================

plt.figure(figsize=(12,4))

plt.plot(
    t,
    solve_time * 1000.0,
)

plt.title("IK Solve Time")

plt.xlabel("Time (s)")

plt.ylabel("Milliseconds")

plt.grid(True)

add_mode_markers()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/ik_solve_time.png",
    dpi=300,
    bbox_inches="tight",
)

# =========================================================
# Position Error Components
# =========================================================

pos_error_xyz = (
    target_pos - actual_pos
)

plt.figure(figsize=(12,6))

labels = ["x", "y", "z"]

for i in range(3):

    plt.plot(
        t,
        pos_error_xyz[:, i],
        label=labels[i],
    )

plt.title("Position Error Components")

plt.xlabel("Time (s)")

plt.ylabel("Error (m)")

plt.legend()

plt.grid(True)

add_mode_markers()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/position_error_xyz.png",
    dpi=300,
    bbox_inches="tight",
)

# ==========================================
# Orientation Error
# ==========================================

orientation_error = data[
    "orientation_error"
]

plt.figure(figsize=(12,4))

plt.plot(
    t,
    orientation_error,
)

plt.title("EE Orientation Error")

plt.xlabel("Time (s)")

plt.ylabel("SO(3) Error (rad)")

plt.grid(True)

add_mode_markers()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/orientation_error.png",
    dpi=300,
    bbox_inches="tight",
)

# =========================================================
# SHOW EVERYTHING
# =========================================================

plt.show()