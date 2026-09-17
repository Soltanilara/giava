"""Write real (camera-less) LeRobot datasets for modes with no recordings yet."""

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys, json, time
from pathlib import Path
sys.path.insert(0, "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts")
sys.path.insert(0, "/opt/ros/noetic/lib/python3/dist-packages")
import numpy as np, torch
from lerobot.datasets import LeRobotDataset
from dataset import build_dataset_features, build_frame, save_dataset_metadata
from data_col_config import ARM_MODES
from arm_config import ARM_CONFIG

OUT = Path(sys.argv[1])
N_STEPS, FPS = 60, 50

for mode, write_meta in (("middle", True), ("av", True), ("av", False)):
    task = f"synth_{mode}" + ("" if write_meta else "_nometa")
    root = OUT / task / "20260821_000000"
    if root.exists():
        continue
    root.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create(repo_id=f"deviamar/{task}", root=str(root),
                               fps=FPS, features=build_dataset_features(mode, []))
    arms = ARM_MODES[mode]
    t0 = time.time()
    for i in range(N_STEPS):
        # arm index * 100 + joint index + a small ramp: makes mis-routing obvious
        states, actions, ee, ts = {}, {}, {}, {}
        for ai, arm in enumerate(arms):
            n = ARM_CONFIG[arm]["num_joints"]
            q = np.array([ai*100 + j + i*0.001 for j in range(n)], np.float32)
            states[arm] = {"joints": q, "gripper": float(ai) + i*0.001}
            actions[arm] = {"joints": q, "gripper": float(ai) + i*0.001}
            ee[arm] = np.zeros(7, np.float32)
            ts[arm] = t0 + i / FPS
        ds.add_frame(build_frame(mode, [], states, actions, ee, ts, {}), task)
    ds.save_episode()
    ds.finalize()
    if write_meta:
        save_dataset_metadata(root, task, mode, [], 1.0 / FPS)
    print(f"wrote {mode:7s} meta.json={write_meta}  {root}")
