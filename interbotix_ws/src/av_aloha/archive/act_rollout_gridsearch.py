from email import parser
from pathlib import Path
import argparse
import csv
import re
import time
from datetime import datetime

import cv2
import numpy as np
from tomlkit import key
import pyrealsense2 as rs
import rospy
import torch

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

from ultralytics import YOLO


GRIPPER_CURRENT_LIMIT = 200

CAMERA_SERIALS = {
    "observation.images.right_wrist": "230322270105",
    "observation.images.top_scene": "230322270396",
}

RIGHT_RESET_Q = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float)
MAX_JOINT_STEP = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.12], dtype=float)

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


def digital_zoom(frame, zoom=1.6):
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


def get_combined_binary_mask(result, image_shape, conf_threshold=0.25):
    h, w = image_shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return combined

    if result.boxes is not None and result.boxes.conf is not None:
        confs = result.boxes.conf.detach().cpu().numpy()
    else:
        confs = None

    masks = result.masks.data.detach().cpu().numpy()

    for i, mask in enumerate(masks):
        if confs is not None and confs[i] < conf_threshold:
            continue

        mask_bin = (mask > 0.5).astype(np.uint8)

        if mask_bin.shape != (h, w):
            mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        combined = np.maximum(combined, mask_bin)

    combined = (combined * 255).astype(np.uint8)
    return combined


def make_mask_tensor(mask_img, device):
    mask_3ch = np.repeat(mask_img[:, :, None], 3, axis=2)
    mask_tensor = (
        torch.from_numpy(mask_3ch)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )
    return mask_tensor


def make_image_tensor(frame, device):
    return (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


def make_overlay(frame_rgb, mask_img, alpha=0.45):
    overlay = frame_rgb.copy()

    # If mask is empty, return the original frame unchanged
    if mask_img.max() == 0:
        return overlay

    green = np.zeros_like(frame_rgb, dtype=np.uint8)
    green[:, :, 1] = 255

    mask_bool = mask_img > 0

    # Ensure there are actually pixels to blend
    if not mask_bool.any():
        return overlay

    overlay[mask_bool] = cv2.addWeighted(
        frame_rgb[mask_bool], 1.0 - alpha, green[mask_bool], alpha, 0
    )

    return overlay


def stack_debug_views(frame_rgb, mask_img, overlay_rgb):
    mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    top = np.hstack([frame_bgr, mask_bgr, overlay_bgr])
    cv2.putText(top, "RGB", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(top, "MASK", (660, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(top, "OVERLAY", (1300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return top


def predict_mask(seg_model, frame_rgb, imgsz=640, conf=0.25, device=None):
    results = seg_model.predict(
        source=frame_rgb,
        verbose=False,
        imgsz=imgsz,
        conf=conf,
        retina_masks=True,
        device=device,
    )
    result = results[0]
    mask_img = get_combined_binary_mask(result, frame_rgb.shape, conf_threshold=conf)
    return result, mask_img


def make_observation(
    right_bot,
    pipelines,
    device,
    seg_model,
    show_masks=True,
    yolo_imgsz=640,
    yolo_conf=0.25,
):
    joint_state_msg = right_bot.dxl.joint_states

    joint_positions = np.array(joint_state_msg.position[:6], dtype=np.float32)
    gripper_position = np.array([joint_state_msg.position[6]], dtype=np.float32)
    obs_state = np.concatenate([joint_positions, gripper_position], axis=0)

    obs = {
        "observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device),
        "observation.ee_pose": torch.zeros((1, 7), dtype=torch.float32, device=device),
    }

    debug_panels = []

    for key, pipeline in pipelines.items():
        frame = get_color_frame(pipeline)

        if key == "observation.images.top_scene":
            frame = digital_zoom(frame)

        result, mask_img = predict_mask(
            seg_model=seg_model,
            frame_rgb=frame,
            imgsz=yolo_imgsz,
            conf=yolo_conf,
            device=0 if device.type == "cuda" else "cpu",
        )

        obs[key] = make_image_tensor(frame, device)

        mask_tensor = make_mask_tensor(mask_img, device)

        if key == "observation.images.right_wrist":
            obs["observation.images.right_wrist_mask"] = mask_tensor
        elif key == "observation.images.top_scene":
            obs["observation.images.top_scene_mask"] = mask_tensor

        if show_masks:
            overlay = make_overlay(frame, mask_img, alpha=0.45)
            panel = stack_debug_views(frame, mask_img, overlay)
            debug_panels.append((key, panel))

            n_det = 0 if result.boxes is None else len(result.boxes)
            mask_ratio = float((mask_img > 0).mean())
            print(f"{key}: detections={n_det}, mask_ratio={mask_ratio:.4f}")

    if show_masks and debug_panels:
        if len(debug_panels) == 1:
            cv2.imshow(debug_panels[0][0], debug_panels[0][1])
        else:
            for key, panel in debug_panels:
                cv2.imshow(key, panel)
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


def parse_run_dir_name(run_dir: Path):
    m = re.fullmatch(r"(?:clean_)?chunk_(\d+)__kl_([0-9]+(?:[p.][0-9]+)?)", run_dir.name)
    if not m:
        raise ValueError(f"Could not parse chunk/kl from directory name: {run_dir.name}")
    chunk_size = int(m.group(1))
    kl_str = m.group(2).replace("p", ".")
    kl_weight = float(kl_str)
    return chunk_size, kl_weight


def should_skip_run(run_name: str, skip_runs: list[str], skip_chunks: list[int]):
    if run_name in skip_runs:
        return True

    try:
        chunk_size, _ = parse_run_dir_name(Path(run_name))
    except Exception:
        return False

    if chunk_size in skip_chunks:
        return True

    return False


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
            chunk_size, kl_weight = parse_run_dir_name(p)
            ckpt = find_checkpoint(
                p,
                checkpoint_mode=checkpoint_mode,
                checkpoint_step=checkpoint_step,
            )
            candidates.append({
                "run_dir": p,
                "chunk_size": chunk_size,
                "kl_weight": kl_weight,
                "checkpoint": ckpt,
            })
        except Exception as e:
            print(f"[skip] {p}: {e}")
    return candidates


def build_policy(dataset_root, policy_dir, device, checkpoint_mode="latest", checkpoint_step=None):
    chunk_size, kl_weight = parse_run_dir_name(policy_dir)

    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="transfer_flower_mask",
        root=dataset_root,
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    excluded_features = {
        "observation.timestamps.robot",
        "observation.timestamps.right_wrist",
        "observation.timestamps.top_scene",
        # "observation.ee_pose",
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

    print("\nINPUT FEATURES")
    for k in input_features.keys():
        print(k)
    
    required_mask_keys = {
        "observation.images.right_wrist_mask",
        "observation.images.top_scene_mask",
    }
    missing_mask_keys = required_mask_keys - set(input_features.keys())
    if missing_mask_keys:
        raise RuntimeError(f"Missing required mask input features: {missing_mask_keys}")

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
    return policy, preprocessor, postprocessor, dataset_metadata, ckpt_path


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
    print("Examples: smooth, reached flower, unstable grasp, good transfer, hesitated")
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

def run_single_rollout(
    policy,
    preprocessor,
    dataset_metadata,
    right_bot,
    pipelines,
    device,
    seg_model,
    rollout_seconds,
    control_hz,
    show_masks=True,
    yolo_imgsz=640,
    yolo_conf=0.25,
):
    dt = 1.0 / control_hz
    num_steps = int(rollout_seconds * control_hz)

    print(f"Dataset fps={dataset_metadata.fps}")
    print(f"Running rollout for {num_steps} steps at {control_hz} Hz")

    aborted = False

    with torch.inference_mode():
        for step in range(num_steps):
            t0 = time.time()

            obs = make_observation(
                right_bot=right_bot,
                pipelines=pipelines,
                device=device,
                seg_model=seg_model,
                show_masks=show_masks,
                yolo_imgsz=yolo_imgsz,
                yolo_conf=yolo_conf,
            )

            output = policy.select_action(preprocessor(obs))
            action_np = output.squeeze().detach().cpu().numpy()

            print(
                f"step={step} "
                f"arm={np.round(action_np[:6], 3)} "
                f"gripper={action_np[6]:.4f}"
            )

            send_action(right_bot, action_np)

            if show_masks:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Rollout aborted by user.")
                    aborted = True
                    break

            elapsed = time.time() - t0
            sleep_time = max(0.0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return aborted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to LeRobot dataset root for transfer_flower",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/act_transfer_flower_gridsearch_mask"),
        help="Root directory containing chunk_*__kl_* subdirectories",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("outputs/act_transfer_flower_gridsearch_mask/rollout_notes.csv"),
        help="CSV file where rollout notes will be appended",
    )
    parser.add_argument("--rollout-seconds", type=float, default=3.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional substring filter, e.g. chunk_50 or kl_1",
    )
    parser.add_argument(
        "--checkpoint-mode",
        type=str,
        default="latest",
        choices=["final", "latest", "step"],
        help="Which checkpoint to load for each run",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step to load when --checkpoint-mode step, e.g. 500",
    )
    parser.add_argument(
        "--skip-runs",
        type=csv_list,
        default=[],
        help="Comma-separated run names to skip, e.g. chunk_35__kl_10,chunk_50__kl_3",
    )
    parser.add_argument(
        "--skip-chunks",
        type=int_csv_list,
        default=[],
        help="Comma-separated chunk sizes to skip, e.g. 35,75",
    )
    parser.add_argument("--show-masks", action="store_true")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.init_node("act_rollout_gridsearch", anonymous=True)

    seg_model = YOLO(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/best_weights_6_2.pt"
    )

    print(seg_model.task)
    

    runs = discover_runs(
        args.runs_root,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_step=args.checkpoint_step,
    )

    if args.only:
        runs = [r for r in runs if args.only in r["run_dir"].name]

    runs = [
        r for r in runs
        if not should_skip_run(
            r["run_dir"].name,
            skip_runs=args.skip_runs,
            skip_chunks=args.skip_chunks,
        )
    ]

    if not runs:
        print("No valid runs found.")
        return

    print("\nDiscovered runs:")
    for i, r in enumerate(runs):
        print(f"[{i}] {r['run_dir'].name}")
        print(f"     checkpoint: {r['checkpoint']}")

    right_bot = InterbotixManipulatorXS(
        robot_model="vx300s",
        group_name="arm",
        gripper_name="gripper",
        robot_name="puppet_right",
        moving_time=0.14,
        accel_time=0.04,
        init_node=False,
    )

    right_bot.dxl.robot_torque_enable("single", "gripper", False)
    set_register("puppet_right", "gripper", "Current_Limit", GRIPPER_CURRENT_LIMIT)
    right_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    right_bot.dxl.robot_torque_enable("single", "gripper", True)

    pipelines = {key: setup_camera(serial) for key, serial in CAMERA_SERIALS.items()}

    try:
        for idx, run in enumerate(runs):
            run_dir = run["run_dir"]
            run_name = run_dir.name

            while True:
                print("\n" + "#" * 100)
                print(f"[{idx + 1}/{len(runs)}] RUN: {run_name}")
                print(f"USING CHECKPOINT: {run['checkpoint']}")
                print(f"chunk_size={run['chunk_size']}  kl_weight={run['kl_weight']}")
                print("#" * 100 + "\n")

                input("Press Enter when ready to reset robot and start rollout...")

                reset_robot(right_bot)
                policy, preprocessor, postprocessor, dataset_metadata, ckpt_path = build_policy(
                    args.dataset_root,
                    run_dir,
                    device,
                    checkpoint_mode=args.checkpoint_mode,
                    checkpoint_step=args.checkpoint_step,
                )

                print(f"\nLoaded policy from: {run_dir}")
                print(f"Checkpoint path: {ckpt_path}")

                run_single_rollout(
                    policy=policy,
                    preprocessor=preprocessor,
                    dataset_metadata=dataset_metadata,
                    right_bot=right_bot,
                    pipelines=pipelines,
                    device=device,
                    seg_model=seg_model,
                    rollout_seconds=args.rollout_seconds,
                    control_hz=args.control_hz,
                    show_masks=args.show_masks,
                    yolo_imgsz=args.yolo_imgsz,
                    yolo_conf=args.yolo_conf,
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Rollout aborted by user.")
                    break

                aborted = run_single_rollout(
                    policy=policy,
                    preprocessor=preprocessor,
                    dataset_metadata=dataset_metadata,
                    right_bot=right_bot,
                    pipelines=pipelines,
                    device=device,
                    seg_model=seg_model,
                    rollout_seconds=args.rollout_seconds,
                    control_hz=args.control_hz,
                    show_masks=args.show_masks,
                    yolo_imgsz=args.yolo_imgsz,
                    yolo_conf=args.yolo_conf,
                )

                reset_robot(right_bot)

                if aborted:
                    break

                obs = prompt_user_notes(run_name, ckpt_path)
                append_result(
                    args.results_csv,
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "run_name": run_name,
                        "checkpoint_path": str(ckpt_path),
                        "chunk_size": run["chunk_size"],
                        "kl_weight": run["kl_weight"],
                        "rollout_seconds": args.rollout_seconds,
                        "control_hz": args.control_hz,
                        "setup_notes": obs["setup_notes"],
                        "success": obs["success"],
                        "smoothness": obs["smoothness"],
                        "task_parts": obs["task_parts"],
                        "notes": obs["notes"],
                    },
                )

                if obs["retry"] != "y":
                    break

    finally:
        for pipeline in pipelines.values():
            pipeline.stop()
        reset_robot(right_bot)


if __name__ == "__main__":
    main()