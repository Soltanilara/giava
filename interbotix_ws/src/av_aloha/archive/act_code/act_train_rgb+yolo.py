from pathlib import Path
import csv
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets import (LeRobotDataset, LeRobotDatasetMetadata)
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies import make_policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
import json
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np


REPO_A = "deviamar/transfer_flower_clean50"
REPO_B = "deviamar/transfer_flower_noisy1"
REPO_C = "deviamar/transfer_flower_noisy2"

def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]

def kept_frame_indices(dataset, bad_episodes):
    bad_episodes = set(bad_episodes)
    epi = dataset.episode_data_index
    kept = []
    n_eps = len(epi["from"])
    for ep_idx in range(n_eps):
        start = int(epi["from"][ep_idx])
        end = int(epi["to"][ep_idx])
        if ep_idx in bad_episodes:
            continue
        kept.extend(range(start, end))
    return kept

def main():
    print("In main")

    dataset_root_a = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/20260601_000935"
    )
    dataset_root_b = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/20260601_042518"
    )
    dataset_root_c = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/20260601_045338"
    )

    bad_episodes_a = {11, 15, 18, 32}
    bad_episodes_b = {7}
    bad_episodes_c = set()

    meta_a = LeRobotDatasetMetadata(
        repo_id=REPO_A,
        root=dataset_root_a,
    )

    meta_b = LeRobotDatasetMetadata(
        repo_id=REPO_B,
        root=dataset_root_b,
    )

    # meta_c = LeRobotDatasetMetadata(
    #     repo_id=REPO_C,
    #     root=dataset_root_c,
    # )

    meta_c = LeRobotDatasetMetadata(
        repo_id=REPO_C,
        root=dataset_root_c,
    )

    print(meta_a.info["total_episodes"])
    print(meta_b.info["total_episodes"])
    print(meta_c.info["total_episodes"])

    # sanity checks
    print("A features:", set(meta_a.features.keys()))
    print("B features:", set(meta_b.features.keys()))
    print("C features:", set(meta_c.features.keys()))

    assert set(meta_a.features.keys()) == set(meta_b.features.keys())
    assert set(meta_a.features.keys()) == set(meta_c.features.keys())

    print("A frames:", meta_a.info["total_frames"])
    print("B frames:", meta_b.info["total_frames"])
    print("C frames:", meta_c.info["total_frames"])

    schema_meta = meta_a

    print("episodes A:", meta_a.info["total_episodes"])
    print("episodes B:", meta_b.info["total_episodes"])
    print("episodes C:", meta_c.info["total_episodes"])

    output_directory = Path("outputs/flower_mask_test")
    output_directory.mkdir(parents=True, exist_ok=True)

    loss_log_path = output_directory / "loss.csv"

    with open(loss_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "recon_loss", "kl_loss"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = dataset_to_policy_features(schema_meta.features)

    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}
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

        # "observation.images.right_wrist",
        # "observation.images.top_scene",
    }

    input_features = {
        k: v
        for k, v in features.items()
        if (
            k not in output_features
            and k not in excluded_features
        )
    }

    best_loss = float("inf")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,

        # Temporal abstraction
        chunk_size=50,          # or 100
        n_action_steps=50,      # keep equal for now to keep behavior simple

        use_vae=True,
        kl_weight=1.0,         # standard ACT CVAE regularization

        optimizer_lr=2e-5,      # safer LR for ACT
        optimizer_lr_backbone=1e-5,
    )

    # cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)

    # print("Config")
    print(cfg)
    # policy = ACTPolicy(cfg)
    # print("action delta indices")
    print(cfg.action_delta_indices)
    # print("observation delta indices")
    print(cfg.observation_delta_indices)
    # print("chunk size")
    print(cfg.chunk_size)
    # print("action steps")
    print(cfg.n_action_steps)

    policy = make_policy(cfg, ds_meta=schema_meta)

    # CHANGED (lerobot v0.6.0): policies no longer carry normalization layers in
    # their weights. Normalization lives in an external processor pipeline built
    # from the dataset stats, and must be applied to every batch before
    # policy.forward(). See ACT_MODIFICATIONS.md.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=schema_meta.stats,
    )

    policy.train()
    policy.to(device)

    # dataset = LeRobotDataset(dataset_root, delta_timestamps=delta_timestamps)
    # dataset = LeRobotDataset(dataset_root, delta_timestamps=delta_timestamps, video_backend="pyav")

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, schema_meta.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, schema_meta.fps)
        for k in cfg.image_features
    }

    dataset_a = LeRobotDataset(
        repo_id=REPO_A,
        root=dataset_root_a,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    dataset_b = LeRobotDataset(
        repo_id=REPO_B,
        root=dataset_root_b,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    dataset_c = LeRobotDataset(
        repo_id=REPO_C,
        root=dataset_root_c,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    print("A repo:", dataset_a.repo_id)
    print("B repo:", dataset_b.repo_id)
    print("C repo:", dataset_c.repo_id)

    print("A root:", dataset_root_a)
    print("B root:", dataset_root_b)
    print("C root:", dataset_root_c)

    indices_a = kept_frame_indices(dataset_a, bad_episodes_a)
    indices_b = kept_frame_indices(dataset_b, bad_episodes_b)
    indices_c = kept_frame_indices(dataset_c, bad_episodes_c)

    print(f"dataset_a kept frames: {len(indices_a)}")
    print(f"dataset_b kept frames: {len(indices_b)}")
    print(f"dataset_c kept frames: {len(indices_c)}")

    print("Episodes A:", len(dataset_a.episode_data_index["from"]))
    print("Episodes B:", len(dataset_b.episode_data_index["from"]))
    print("Episodes C:", len(dataset_c.episode_data_index["from"]))

    train_dataset = ConcatDataset([
        Subset(dataset_a, indices_a),
        Subset(dataset_b, indices_b),
        Subset(dataset_c, indices_c),
    ])

    print(f"total kept frames: {len(train_dataset)}")

    batch_size = 8
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    training_steps = 100
    log_freq = 10

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {
                k: v.to(device, non_blocking=True)
                if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # CHANGED (lerobot v0.6.0): images arrive as uint8 and the normalizer
            # expects float in [0, 1], then the preprocessor applies the mean/std
            # normalization that used to live inside the policy.
            for cam_key in cfg.image_features:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

            batch = preprocessor(batch)

            loss, loss_dict = policy.forward(batch)

            # print("loss dict\n")
            # print(loss_dict.keys())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % log_freq == 0:
                print(f"step={step} loss={loss.item():.6f}")

            # log loss
            with open(loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    float(loss.item()),
                    float(loss_dict.get("l1_loss", float("nan"))),
                    float(loss_dict.get("kld_loss", float("nan"))),
                ])

            # save best model
            if loss.item() < best_loss:
                best_loss = loss.item()

                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "loss": best_loss,
                        # "config": cfg_dict,
                    },
                    output_directory / "best_checkpoint.pt",
                )

            # save periodic checkpoints
            if step > 0 and step % 500 == 0:
                ckpt_path = output_directory / f"checkpoint_{step}.pt"

                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        # "config": cfg_dict,
                    },
                    ckpt_path,
                )

                print(f"Saved {ckpt_path}")

            step += 1

            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)

    # CHANGED (lerobot v0.6.0): normalization stats now live in the processor
    # pipelines rather than the model weights, so they must be saved next to the
    # policy. Without these the checkpoint cannot be run at rollout time.
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    ckpt_path = output_directory / "checkpoint.pt"

    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            # "config": cfg_dict,
        },
        ckpt_path,
    )

    print(f"Final checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()