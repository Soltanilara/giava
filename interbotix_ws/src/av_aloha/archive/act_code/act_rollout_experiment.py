from pathlib import Path
import argparse
import csv
import json
import re
import time
from datetime import datetime

""" python act_rollout_experiment.py \
  --dataset-root /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/20260601_000935 \
  --runs-root outputs/act_transfer_flower_6experiments \
  --priors-path /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/scene_priors.mat \
  --rollout-seconds 3 \
  --control-hz 50 """

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import torch
from scipy.io import loadmat

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies import make_policy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.configs.types import FeatureType


GRIPPER_CURRENT_LIMIT = 200

CAMERA_SERIALS = {
    "observation.images.right_wrist": "230322270105",
    "observation.images.top_scene": "230322270396",
}

RIGHT_RESET_Q = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float)
MAX_JOINT_STEP = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)

EXPERIMENTS = [
    "rgb_plus_blue_mask",
    "masked_rgb_only",
    "rgb_plus_centroids",
    "centroids_plus_vectors",
    "rgb_plus_blue_mask_plus_centroids",
    "masked_rgb_plus_centroids",
]


def resolve_migrated_checkpoint_dir(policy_dir):
    """Return the lerobot v0.6.0 checkpoint directory for `policy_dir`.

    v0.6.0 loads a policy from a directory (clean model.safetensors plus the
    policy_preprocessor / policy_postprocessor files), not from a raw .pt state
    dict. migrate_checkpoints.py writes those next to the original as
    "<original>_lerobot_v06"; if `policy_dir` is already such a directory it is
    used as-is.
    """
    p = Path(policy_dir)
    candidates = [p.parent / f"{p.name}_lerobot_v06", p]
    for c in candidates:
        if (c / "model.safetensors").exists() and (c / "policy_preprocessor.json").exists():
            return c
    raise FileNotFoundError(
        f"No migrated checkpoint for {policy_dir}. Expected "
        f"{p.parent / (p.name + '_lerobot_v06')} containing model.safetensors and "
        "policy_preprocessor.json. Run migrate_checkpoints.py first."
    )


def csv_list(arg: str):
    if not arg or not arg.strip():
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def int_csv_list(arg: str):
    if not arg or not arg.strip():
        return []
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def set_register(robot_name, motor_name, reg_name, value):
    service_name = f"/{robot_name}/set_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)

    req = RegisterValuesRequest()
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name
    req.value = value
    return srv(req)


def setup_camera(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
    pipeline.start(config)
    return pipeline


## ZOOM 1.8, NOT 1.6.  The June recorder applied digital_zoom(frame, zoom=1.8)
## before writing frames (data_collection.py@44895c6 line 421); this script
## shipped with 1.6, so every rollout fed the policy a ~12% different scale
## than it was trained on.  A policy cannot report that -- it just performs
## badly.  Verified against the recorded frames.
def digital_zoom(frame, zoom=1.8):
    h, w = frame.shape[:2]
    new_w = int(w / zoom)
    new_h = int(h / zoom)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def get_color_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("No color frame received")
    return np.asanyarray(color_frame.get_data())


def make_image_tensor(frame, device):
    return (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


def sanitize_value(x):
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return str(x).replace(".", "p")
    return str(x)


def run_name(variant, chunk_size, kl_weight):
    return f"{variant}__chunk_{sanitize_value(chunk_size)}__kl_{sanitize_value(kl_weight)}"


def parse_run_dir_name(run_dir: Path):
    m = re.fullmatch(
        r"(.+)__chunk_(\d+)__kl_([0-9]+(?:[p.][0-9]+)?)",
        run_dir.name
    )
    if not m:
        raise ValueError(f"Could not parse variant/chunk/kl from directory name: {run_dir.name}")

    variant = m.group(1)
    chunk_size = int(m.group(2))
    kl_weight = float(m.group(3).replace("p", "."))
    return variant, chunk_size, kl_weight


def should_skip_run(run_name: str, skip_runs: list[str], skip_chunks: list[int]):
    if run_name in skip_runs:
        return True

    try:
        _, chunk_size, _ = parse_run_dir_name(Path(run_name))
    except Exception:
        return False

    return chunk_size in skip_chunks


def find_checkpoint(run_dir: Path, checkpoint_mode="latest", checkpoint_step=None):
    final_ckpt = run_dir / "checkpoint.pt"

    numbered = []
    for p in run_dir.glob("checkpoint_*.pt"):
        m = re.fullmatch(r"checkpoint_(\d+)\.pt", p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    numbered.sort(key=lambda x: x[0])

    if checkpoint_mode == "final":
        if final_ckpt.exists():
            return final_ckpt
        raise FileNotFoundError(f"checkpoint.pt not found in {run_dir}")

    if checkpoint_mode == "latest":
        if final_ckpt.exists():
            return final_ckpt
        if numbered:
            return numbered[-1][1]
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    if checkpoint_mode == "step":
        if checkpoint_step is None:
            raise ValueError("checkpoint_step must be set when checkpoint_mode='step'")
        target = run_dir / f"checkpoint_{checkpoint_step}.pt"
        if target.exists():
            return target
        raise FileNotFoundError(f"{target} not found")

    raise ValueError(f"Unknown checkpoint_mode: {checkpoint_mode}")


def discover_runs(runs_root: Path, checkpoint_mode="latest", checkpoint_step=None):
    candidates = []
    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue
        try:
            variant, chunk_size, kl_weight = parse_run_dir_name(p)
            ckpt = find_checkpoint(
                p,
                checkpoint_mode=checkpoint_mode,
                checkpoint_step=checkpoint_step,
            )
            candidates.append({
                "run_dir": p,
                "variant": variant,
                "chunk_size": chunk_size,
                "kl_weight": kl_weight,
                "checkpoint": ckpt,
            })
        except Exception as e:
            print(f"[skip] {p}: {e}")
    return candidates


def build_features_for_variant(dataset_root, variant):
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="deviamar/transfer_flower",
        root=dataset_root,
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {
        k: v for k, v in features.items()
        if v.type is FeatureType.ACTION
    }

    excluded_features = {
        "observation.timestamps.robot",
        "observation.timestamps.right_wrist",
        "observation.timestamps.top_scene",
        "observation.depth.right_wrist",
        "observation.depth.top_scene",
        "observation.depth_intrinsics.right_wrist",
        "observation.depth_intrinsics.top_scene",
        "observation.timestamps.right_wrist_depth",
        "observation.timestamps.top_scene_depth",
    }

    input_features = {
        k: v for k, v in features.items()
        if k not in output_features and k not in excluded_features
    }

    def set_feature(name, feature_type, shape, dtype):
        input_features[name] = type(
            "DummyFeature",
            (),
            {
                "type": feature_type,
                "shape": shape,
                "dtype": dtype,
            },
        )()

    if variant in {
        "rgb_plus_centroids",
        "centroids_plus_vectors",
        "rgb_plus_blue_mask_plus_centroids",
        "masked_rgb_plus_centroids",
    }:
        set_feature("observation.object_centroid", FeatureType.STATE, (2,), "float32")
        set_feature("observation.flower_target_centroid", FeatureType.STATE, (2,), "float32")
        set_feature("observation.oval_target_centroid", FeatureType.STATE, (2,), "float32")

    if variant == "centroids_plus_vectors":
        set_feature("observation.scene_geometry", FeatureType.STATE, (14,), "float32")
        set_feature("observation.object_area", FeatureType.STATE, (1,), "float32")
        set_feature("observation.object_found", FeatureType.STATE, (1,), "float32")

    return dataset_metadata, input_features, output_features


def build_policy(dataset_root, policy_dir, device, checkpoint_mode="latest", checkpoint_step=None):
    variant, chunk_size, kl_weight = parse_run_dir_name(policy_dir)

    dataset_metadata, input_features, output_features = build_features_for_variant(
        dataset_root, variant
    )

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        use_vae=True,
        kl_weight=kl_weight,
        optimizer_lr=2e-5,
        optimizer_lr_backbone=1e-5,
    )

    ckpt_path = find_checkpoint(
        policy_dir,
        checkpoint_mode=checkpoint_mode,
        checkpoint_step=checkpoint_step,
    )
    migrated_dir = resolve_migrated_checkpoint_dir(policy_dir)

    # CHANGED (lerobot v0.6.0): load the policy together with its normalization
    # pipelines from a migrated checkpoint directory. Building a fresh policy and
    # loading a raw .pt state dict no longer works -- those state dicts carry the
    # old embedded `normalize_inputs.*` buffers, which the v0.6.0 model does not
    # define, so a strict load fails. `migrated_dir` is the directory produced
    # by migrate_checkpoints.py (named "<original>_lerobot_v06").
    policy = ACTPolicy.from_pretrained(migrated_dir)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        migrated_dir, config_filename="policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        migrated_dir, config_filename="policy_postprocessor.json"
    )

    policy.to(device)
    policy.eval()
    return policy, preprocessor, postprocessor, dataset_metadata, ckpt_path, variant, chunk_size, kl_weight


def extract_scene_features_from_top_image(rgb_img, priors):
    scene = priors["scenePriors"]

    flower_mask = scene.flowerMask.astype(np.uint8)
    oval_mask = scene.ovalMask.astype(np.uint8)
    flower_centroid = scene.flowerCentroidNorm.astype(np.float32)
    oval_centroid = scene.ovalCentroidNorm.astype(np.float32)

    hmin = float(scene.lightBlueHMin)
    hmax = float(scene.lightBlueHMax)
    smin = float(scene.lightBlueSMin)
    vmin = float(scene.lightBlueVMin)

    if rgb_img.max() <= 1.0:
        rgb_img = (rgb_img * 255).astype(np.uint8)
    else:
        rgb_img = rgb_img.astype(np.uint8)

    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0] / 179.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0

    object_mask = (
        (H >= hmin) &
        (H <= hmax) &
        (S >= smin) &
        (V >= vmin)
    ).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask)
    h, w = object_mask.shape

    if num_labels <= 1:
        object_mask = np.zeros((h, w), dtype=np.uint8)
        object_centroid = np.array([0.0, 0.0], dtype=np.float32)
        object_area = np.array([0.0], dtype=np.float32)
        found = np.array([0.0], dtype=np.float32)
    else:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        object_mask = (labels == largest_idx).astype(np.uint8)
        cx, cy = centroids[largest_idx]
        area = stats[largest_idx, cv2.CC_STAT_AREA] / float(h * w)

        object_centroid = np.array([cx / w, cy / h], dtype=np.float32)
        object_area = np.array([area], dtype=np.float32)
        found = np.array([1.0], dtype=np.float32)

    obj_to_flower = flower_centroid - object_centroid
    obj_to_oval = oval_centroid - object_centroid
    dist_flower = np.array([np.linalg.norm(obj_to_flower)], dtype=np.float32)
    dist_oval = np.array([np.linalg.norm(obj_to_oval)], dtype=np.float32)

    geom = np.concatenate(
        [
            object_centroid,
            flower_centroid,
            oval_centroid,
            obj_to_flower,
            obj_to_oval,
            dist_flower,
            dist_oval,
            object_area,
            found,
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "object_mask": object_mask,
        "object_centroid": object_centroid,
        "object_area": object_area,
        "object_found": found,
        "flower_centroid": flower_centroid.astype(np.float32),
        "oval_centroid": oval_centroid.astype(np.float32),
        "scene_geometry": geom,
        "flower_mask": flower_mask,
        "oval_mask": oval_mask,
    }


def make_observation(
    right_bot,
    pipelines,
    device,
    priors,
    variant,
    show_debug=True,
):
    joint_state_msg = right_bot.dxl.joint_states

    joint_positions = np.array(joint_state_msg.position[:6], dtype=np.float32)
    gripper_position = np.array([joint_state_msg.position[6]], dtype=np.float32)
    obs_state = np.concatenate([joint_positions, gripper_position], axis=0)

    obs = {
        "observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device),
        "observation.ee_pose": torch.zeros((1, 7), dtype=torch.float32, device=device),
    }

    top_frame_rgb = None

    for key, pipeline in pipelines.items():
        frame = get_color_frame(pipeline)
        if key == "observation.images.top_scene":
            frame = digital_zoom(frame)
            top_frame_rgb = frame.copy()

        obs[key] = make_image_tensor(frame, device)

    if top_frame_rgb is None:
        raise RuntimeError("Missing top_scene camera frame")

    feats = extract_scene_features_from_top_image(top_frame_rgb, priors)

    object_mask = feats["object_mask"]
    object_mask_3ch = np.repeat(object_mask[:, :, None], 3, axis=2).astype(np.float32)

    top_tensor = obs["observation.images.top_scene"]
    top_float = top_tensor.clone()
    if top_float.max() > 1.0:
        top_float = top_float / 255.0

    masked_rgb = top_float * (
        torch.from_numpy(object_mask_3ch)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    obs["observation.object_centroid"] = torch.from_numpy(
        feats["object_centroid"]
    ).unsqueeze(0).to(device)
    obs["observation.object_area"] = torch.from_numpy(
        feats["object_area"]
    ).unsqueeze(0).to(device)
    obs["observation.object_found"] = torch.from_numpy(
        feats["object_found"]
    ).unsqueeze(0).to(device)
    obs["observation.flower_target_centroid"] = torch.from_numpy(
        feats["flower_centroid"]
    ).unsqueeze(0).to(device)
    obs["observation.oval_target_centroid"] = torch.from_numpy(
        feats["oval_centroid"]
    ).unsqueeze(0).to(device)
    obs["observation.scene_geometry"] = torch.from_numpy(
        feats["scene_geometry"]
    ).unsqueeze(0).to(device)

    if variant in {
        "rgb_plus_blue_mask",
        "masked_rgb_only",
        "rgb_plus_blue_mask_plus_centroids",
        "masked_rgb_plus_centroids",
    }:
        obs["observation.images.top_scene"] = masked_rgb

    elif variant == "rgb_plus_centroids":
        pass

    elif variant == "centroids_plus_vectors":
        obs["observation.images.top_scene"] = torch.zeros_like(top_tensor)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    if show_debug:
        overlay = top_frame_rgb.copy()
        mask_bool = object_mask > 0
        if mask_bool.any():
            green = np.zeros_like(top_frame_rgb, dtype=np.uint8)
            green[:, :, 1] = 255
            overlay[mask_bool] = cv2.addWeighted(
                top_frame_rgb[mask_bool], 0.55, green[mask_bool], 0.45, 0
            )

        mask_vis = (object_mask * 255).astype(np.uint8)
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        panel = np.hstack([
            cv2.cvtColor(top_frame_rgb, cv2.COLOR_RGB2BGR),
            mask_vis,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        ])
        cv2.putText(panel, f"{variant}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("top_scene_debug", panel)
        cv2.waitKey(1)

    return obs


def send_action(right_bot, action):
    if torch.is_tensor(action):
        action = action.detach().float().cpu().numpy()
    action = np.asarray(action).reshape(-1)

    if action.shape[0] < 7:
        raise ValueError(f"Expected action dim >= 7, got {action.shape[0]}")

    arm_cmd = action[:6]
    gripper_cmd = float(action[6])

    current_q = np.array(right_bot.dxl.joint_states.position[:6], dtype=float)
    arm_cmd = np.clip(arm_cmd, current_q - MAX_JOINT_STEP, current_q + MAX_JOINT_STEP)

    right_bot.arm.set_joint_positions(
        arm_cmd.tolist(),
        moving_time=0.14,
        accel_time=0.04,
        blocking=False,
    )

    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = gripper_cmd
    right_bot.gripper.core.pub_single.publish(cmd)


def reset_robot(right_bot):
    right_bot.arm.set_joint_positions(
        RIGHT_RESET_Q.tolist(),
        moving_time=2.0,
        accel_time=0.5,
        blocking=True,
    )

    right_bot.dxl.robot_torque_enable("single", "gripper", False)
    set_register("puppet_right", "gripper", "Current_Limit", GRIPPER_CURRENT_LIMIT)
    right_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = 0.1
    right_bot.gripper.core.pub_single.publish(cmd)
    rospy.sleep(1.0)


def prompt_user_notes(run_name, ckpt_path):
    print("\n" + "=" * 80)
    print(f"Finished rollout for: {run_name}")
    print(f"Checkpoint used: {ckpt_path}")
    print("Enter your observations.")
    print("=" * 80)

    setup_notes = input("Pre-trial/setup notes (optional): ").strip()
    success = input("Success? [y/n/partial]: ").strip()
    smoothness = input("Smoothness [1-5]: ").strip()
    task_parts = input("Task parts completed: ").strip()
    notes = input("Freeform notes: ").strip()
    retry = input("Retry this checkpoint? [y/n]: ").strip().lower()

    return {
        "setup_notes": setup_notes,
        "success": success,
        "smoothness": smoothness,
        "task_parts": task_parts,
        "notes": notes,
        "retry": retry,
    }


def append_result(csv_path: Path, row: dict):
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "run_name",
        "variant",
        "checkpoint_path",
        "chunk_size",
        "kl_weight",
        "rollout_seconds",
        "control_hz",
        "setup_notes",
        "success",
        "smoothness",
        "task_parts",
        "notes",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def select_action_from_policy(policy, obs, preprocessor, postprocessor=None,
                              replan_each_step=True):
    # CHANGED (lerobot v0.6.0): the preprocessor is passed in rather than closed
    # over, and applied once here so both the select_action and the direct-call
    # branch below see normalized observations.

    ## REPLAN EVERY STEP vs EXECUTE THE CHUNK.
    ##
    ## reset() drops ACT's action queue, so with it the policy re-runs the
    ## network every tick and always returns chunk[0] -- receding horizon.
    ## Without it the queue drains over n_action_steps ticks, which for these
    ## runs is 75: open-loop execution of the whole predicted chunk.
    ##
    ## These are DIFFERENT EXPERIMENTS, and the difference is large here: over
    ## a 75-step chunk the demonstrated shoulder travels ~0.74 rad, while
    ## chunk[0] sits only ~0.01 rad from the current state.  If the policy
    ## predicts "stay roughly here" as its first action once it has grasped,
    ## replanning every step can sit still while the transport it planned
    ## never gets executed.
    ##
    ## DEFAULT IS True, i.e. the June behaviour -- reset() has been called
    ## per-step since commit de05030, so the paper's numbers are the
    ## receding-horizon ones and changing the default would break the
    ## comparison this rollout exists to make.  --open-loop flips it.
    if replan_each_step and hasattr(policy, "reset"):
        policy.reset()

    obs = preprocessor(obs)

    ## PUT THE BATCH BACK ON THE MODEL'S DEVICE.  make_observation builds every
    ## tensor on `device`, but the migrated v0.6.0 pipelines serialize their
    ## device_processor with whatever device was used at MIGRATION time -- cpu
    ## here -- so the preprocessor silently moves the batch off the GPU.  The
    ## model is on cuda (build_policy calls policy.to(device)), and ACT samples
    ## its inference latent onto batch["observation.state"].device, so the
    ## mismatch surfaces deep inside the encoder as
    ##   "mat1 is on cpu, different from other tensors on cuda:0".
    ## Taking the device from the model rather than a captured variable keeps
    ## this correct no matter what the pipeline decides to do.
    _dev = next(policy.parameters()).device
    obs = {k: (v.to(_dev) if torch.is_tensor(v) else v) for k, v in obs.items()}

    if hasattr(policy, "select_action"):
        return _unnormalize(policy.select_action(obs), postprocessor)

    out = policy(obs)
    if isinstance(out, dict):
        if "action" in out:
            return _unnormalize(out["action"], postprocessor)
        if "actions" in out:
            return _unnormalize(out["actions"][:, 0], postprocessor)
    if torch.is_tensor(out):
        if out.ndim == 3:
            return _unnormalize(out[:, 0], postprocessor)
        return _unnormalize(out, postprocessor)
    raise RuntimeError("Could not extract action from policy output")


def _unnormalize(action, postprocessor):
    """Map a policy action out of normalized space into joint radians.

    THIS STEP WAS MISSING AND IT IS NOT OPTIONAL.  Before lerobot v0.6.0 the
    normalization buffers lived INSIDE the model, so select_action() already
    returned radians and there was nothing to apply.  v0.6.0 moved them out
    into processor pipelines; build_policy was updated to LOAD
    policy_postprocessor.json but nothing ever ran it, so the raw normalized
    output -- roughly mean 0, std 1 -- was being commanded straight to the
    servos as joint angles.  That is why the arm twisted into poses unrelated
    to the task while the robot's own geometry was completely unchanged.

    Accepts a plain tensor or a {"action": tensor} batch, because the pipeline
    version decides which it wants."""
    if postprocessor is None:
        return action
    out = postprocessor({"action": action} if torch.is_tensor(action) else action)
    if isinstance(out, dict):
        return out.get("action", out.get("actions"))
    return out


def run_single_rollout(
    policy,
    preprocessor,
    postprocessor,
    dataset_metadata,
    right_bot,
    pipelines,
    device,
    priors,
    variant,
    rollout_seconds,
    control_hz,
    show_debug=True,
    replan_each_step=True,
):
    dt = 1.0 / control_hz
    num_steps = int(rollout_seconds * control_hz)

    print(f"Dataset fps={dataset_metadata.fps}")
    print(f"Variant={variant}")
    print(f"Running rollout for {num_steps} steps at {control_hz} Hz")

    if hasattr(policy, "reset"):
        policy.reset()

    aborted = False
    t0 = time.time()

    with torch.inference_mode():
        for step in range(num_steps):
            step_start = time.time()

            obs = make_observation(
                right_bot=right_bot,
                pipelines=pipelines,
                device=device,
                priors=priors,
                variant=variant,
                show_debug=show_debug,
            )

            action = select_action_from_policy(policy, obs, preprocessor,
                                               postprocessor,
                                               replan_each_step)
            send_action(right_bot, action)

            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            print(f"step {step+1}/{num_steps}  loop_dt={time.time()-step_start:.4f}s")

    rollout_elapsed = time.time() - t0
    return {
        "aborted": aborted,
        "elapsed": rollout_elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root used to build dataset metadata/features",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Root folder containing the 6 trained run directories",
    )
    parser.add_argument(
        "--priors-path",
        type=Path,
        required=True,
        help="Path to scene_priors.mat",
    )
    ## THE RATE IS A PROPERTY OF THE DATA, NOT A PREFERENCE.  These datasets
    ## record fps=50 and were CAPTURED at ~2.23 Hz (measured from
    ## observation.timestamps.robot; see true_fps.py).  ACT emits absolute
    ## joint waypoints, so executing them at 50 Hz replays a demonstration
    ## ~22x too fast: the servo cannot track, it lags, and you get oscillation
    ## and undershoot that look like policy failure and are not.
    ##
    ## Run `python ../true_fps.py --root <dataset>` and pass what it reports.
    ## The defaults below are that measurement for the transfer_flower runs.
    parser.add_argument("--control-hz", type=float, default=2.23,
                        help="MEASURED capture rate of the training data "
                             "(true_fps.py), not the declared fps")
    parser.add_argument("--rollout-seconds", type=float, default=50.0,
                        help="episodes were 94-116 frames at 2.23 Hz = 42-52 s")
    parser.add_argument("--checkpoint-mode", type=str, default="latest",
                        choices=["latest", "final", "step"])
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--skip-runs", type=str, default="")
    parser.add_argument("--skip-chunks", type=str, default="")
    parser.add_argument("--results-csv", type=Path,
                        default=Path("outputs/rollout_eval/results.csv"))
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument(
        "--open-loop", action="store_true",
        help="execute the whole predicted chunk (n_action_steps ticks) instead "
             "of replanning every tick. NOT the June behaviour -- label runs "
             "accordingly.")
    args = parser.parse_args()

    skip_runs = csv_list(args.skip_runs)
    skip_chunks = int_csv_list(args.skip_chunks)

    priors = loadmat(str(args.priors_path), squeeze_me=True, struct_as_record=False)

    # rospy.init_node("act_rollout_eval_6variants", anonymous=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    right_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_right",
        moving_time=0.2,
        accel_time=0.05,
    )

    pipelines = {
        key: setup_camera(serial)
        for key, serial in CAMERA_SERIALS.items()
    }

    runs = discover_runs(
        args.runs_root,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_step=args.checkpoint_step,
    )

    runs = [r for r in runs if r["variant"] in EXPERIMENTS]

    if not runs:
        raise RuntimeError(f"No valid runs found in {args.runs_root}")

    print("\nDiscovered runs:")
    for r in runs:
        print(
            f"  {r['run_dir'].name} | variant={r['variant']} "
            f"| chunk={r['chunk_size']} | kl={r['kl_weight']} "
            f"| ckpt={r['checkpoint']}"
        )

    try:
        for r in runs:
            run_dir = r["run_dir"]
            run_dir_name = run_dir.name

            if should_skip_run(run_dir_name, skip_runs, skip_chunks):
                print(f"[skip] {run_dir_name}")
                continue

            print("\n" + "#" * 100)
            print(f"Evaluating {run_dir_name}")
            print("#" * 100)

            # reset_robot(right_bot)
            input("Set up the scene, then press Enter to start rollout...")

            policy, preprocessor, postprocessor, dataset_metadata, ckpt_path, variant, chunk_size, kl_weight = build_policy(
                dataset_root=args.dataset_root,
                policy_dir=run_dir,
                device=device,
                checkpoint_mode=args.checkpoint_mode,
                checkpoint_step=args.checkpoint_step,
            )

            result = run_single_rollout(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                replan_each_step=not args.open_loop,
                dataset_metadata=dataset_metadata,
                right_bot=right_bot,
                pipelines=pipelines,
                device=device,
                priors=priors,
                variant=variant,
                rollout_seconds=args.rollout_seconds,
                control_hz=args.control_hz,
                show_debug=not args.no_debug,
            )

            # reset_robot(right_bot)

            notes = prompt_user_notes(run_dir_name, ckpt_path)
            append_result(
                args.results_csv,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_name": run_dir_name,
                    "variant": variant,
                    "checkpoint_path": str(ckpt_path),
                    "chunk_size": chunk_size,
                    "kl_weight": kl_weight,
                    "rollout_seconds": args.rollout_seconds,
                    "control_hz": args.control_hz,
                    "setup_notes": notes["setup_notes"],
                    "success": notes["success"],
                    "smoothness": notes["smoothness"],
                    "task_parts": notes["task_parts"],
                    "notes": notes["notes"],
                },
            )

            if notes["retry"] == "y":
                print("Retry requested. Re-running same checkpoint once.")
                # reset_robot(right_bot)
                input("Reset the scene, then press Enter to retry...")
                result = run_single_rollout(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    replan_each_step=not args.open_loop,
                    dataset_metadata=dataset_metadata,
                    right_bot=right_bot,
                    pipelines=pipelines,
                    device=device,
                    priors=priors,
                    variant=variant,
                    rollout_seconds=args.rollout_seconds,
                    control_hz=args.control_hz,
                    show_debug=not args.no_debug,
                )
                # reset_robot(right_bot)

    finally:
        for p in pipelines.values():
            try:
                p.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()