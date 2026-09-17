from pathlib import Path
import csv
import json
import math
import traceback

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from lerobot.configs.types import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies import make_policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig

from lerobot.datasets.compute_stats import compute_episode_stats

import cv2
import numpy as np

from scipy.io import loadmat
import numpy as np

from huggingface_hub import HfApi

import json

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

def sanitize_value(x):
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return str(x).replace(".", "p")
    return str(x)

def run_name(chunk_size, kl_weight):
    return f"chunk_{sanitize_value(chunk_size)}__kl_{sanitize_value(kl_weight)}"

def build_features(dataset_root):
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="lerobot/transfer_flower_mask",
        root=dataset_root,
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}

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
        k: v
        for k, v in features.items()
        if k not in output_features and k not in excluded_features
    }

    return dataset_metadata, input_features, output_features


def build_train_dataset(cfg, dataset_root_a, dataset_root_b, dataset_root_c,
                        bad_episodes_a, bad_episodes_b, bad_episodes_c,
                        dataset_metadata):
    print("------------------ENTERED build_train_dataset----------------------")
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # try:
    #     dataset_a = LeRobotDataset(
    #         repo_id="deviamar/transfer_flower_clean50",
    #         root=dataset_root_a,
    #         video_backend="pyav",
    #     )
    #     print("A loaded")
    # except Exception as e:
    #     print("A FAILED")
    #     raise
    
    dataset_a = LeRobotDataset(
        repo_id="deviamar/transfer_flower_clean50",
        root=dataset_root_a,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    print("Printing dataset_a features:")
    print(dataset_a.features.keys())
    print(dataset_a.meta.info["features"].keys())
    print(dataset_a.delta_timestamps)
    print(dataset_a.tolerance_s)
    print(dataset_a.meta.fps)

    sample = dataset_a[0]

    for k in sample.keys():
        print(k)
    
    print(dataset_a.meta.info.keys())

    # try:
    #     dataset_b = LeRobotDataset(
    #         repo_id="deviamar/transfer_flower_noisy1",
    #         root=dataset_root_b,
    #         delta_timestamps=delta_timestamps,
    #         video_backend="pyav",
    #     )
    #     print("B loaded")
    # except Exception as e:
    #     print("B FAILED")
    #     raise

    dataset_b = LeRobotDataset(
        repo_id="deviamar/transfer_flower_noisy1",
        root=dataset_root_b,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    print("Printing dataset_b features:")
    print(dataset_b.features.keys())
    print(dataset_b.meta.info["features"].keys())

    # try:
    #     dataset_c = LeRobotDataset(
    #         repo_id="deviamar/transfer_flower_noisy2",
    #         root=dataset_root_c,
    #         delta_timestamps=delta_timestamps,
    #         video_backend="pyav",
    #     )
    #     print("C loaded")
    # except Exception as e:
    #     print("C FAILED")
    #     raise

    dataset_c = LeRobotDataset(
        repo_id="deviamar/transfer_flower_noisy2",
        root=dataset_root_c,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    print("Printing dataset_c features:")
    print(dataset_c.features.keys())
    print(dataset_c.meta.info["features"].keys())

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
    return train_dataset

def train_one_run(
    chunk_size,
    kl_weight,
    dataset_root_a,
    dataset_root_b,
    dataset_root_c,
    bad_episodes_a,
    bad_episodes_b,
    bad_episodes_c,
    output_root,
    device,
    training_steps=2000,
    batch_size=8,
    log_freq=100,
    checkpoint_freq=500,
    optimizer_lr=2e-5,
    optimizer_lr_backbone=1e-5,
    seed=0,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_metadata, input_features, output_features = build_features(dataset_root_a)

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        use_vae=True,
        kl_weight=kl_weight,
        optimizer_lr=optimizer_lr,
        optimizer_lr_backbone=optimizer_lr_backbone,
    )
    print(cfg.input_features.keys())

    print("\n" + "=" * 80)
    print(f"Starting run: chunk_size={chunk_size}, kl_weight={kl_weight}")
    print(cfg)
    print("action delta indices:", cfg.action_delta_indices)
    print("observation delta indices:", cfg.observation_delta_indices)
    print("chunk size:", cfg.chunk_size)
    print("action steps:", cfg.n_action_steps)
    print("=" * 80)

    run_dir = output_root / run_name(chunk_size, kl_weight)
    run_dir.mkdir(parents=True, exist_ok=True)

    loss_log_path = run_dir / "loss.csv"
    with open(loss_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "recon_loss", "kl_loss"])

    cfg_json_path = run_dir / "run_config.json"
    with open(cfg_json_path, "w") as f:
        json.dump(
            {
                "chunk_size": chunk_size,
                "n_action_steps": chunk_size,
                "kl_weight": kl_weight,
                "optimizer_lr": optimizer_lr,
                "optimizer_lr_backbone": optimizer_lr_backbone,
                "training_steps": training_steps,
                "batch_size": batch_size,
                "seed": seed,
                "dataset_roots": [
                    str(dataset_root_a),
                    str(dataset_root_b),
                    str(dataset_root_c),
                ],
                "bad_episodes_a": sorted(list(bad_episodes_a)),
                "bad_episodes_b": sorted(list(bad_episodes_b)),
                "bad_episodes_c": sorted(list(bad_episodes_c)),
            },
            f,
            indent=2,
        )

    train_dataset = build_train_dataset(
        cfg,
        dataset_root_a, dataset_root_b, dataset_root_c,
        bad_episodes_a, bad_episodes_b, bad_episodes_c,
        dataset_metadata,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    policy = make_policy(cfg, ds_meta=dataset_metadata)

    # CHANGED (lerobot v0.6.0): policies no longer carry normalization layers in
    # their weights. Normalization lives in an external processor pipeline built
    # from the dataset stats, and must be applied to every batch before
    # policy.forward(). See ACT_MODIFICATIONS.md.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=dataset_metadata.stats,
    )
    policy.train()
    policy.to(device)

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    best_loss = float("inf")
    best_step = -1
    step = 0
    done = False

    while not done:
        for batch in dataloader:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            loss_value = float(loss.item())
            recon_loss = float(loss_dict.get("l1_loss", float("nan")))
            kl_loss_val = float(loss_dict.get("kld_loss", float("nan")))

            if step % log_freq == 0:
                print(
                    f"[{run_name(chunk_size, kl_weight)}] "
                    f"step={step} loss={loss_value:.6f} "
                    f"recon={recon_loss:.6f} kl={kl_loss_val:.6f}"
                )

            with open(loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, loss_value, recon_loss, kl_loss_val])

            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "loss": best_loss,
                        "chunk_size": chunk_size,
                        "kl_weight": kl_weight,
                    },
                    run_dir / "best_checkpoint.pt",
                )

            if step > 0 and step % checkpoint_freq == 0:
                ckpt_path = run_dir / f"checkpoint_{step}.pt"
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "loss": loss_value,
                        "chunk_size": chunk_size,
                        "kl_weight": kl_weight,
                    },
                    ckpt_path,
                )
                print(f"Saved {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(run_dir)

    # CHANGED (lerobot v0.6.0): normalization stats now live in the processor
    # pipelines rather than the model weights, so they must be saved next to the
    # policy. Without these the checkpoint cannot be run at rollout time.
    preprocessor.save_pretrained(run_dir)
    postprocessor.save_pretrained(run_dir)

    final_ckpt_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss_value,
            "best_loss": best_loss,
            "best_step": best_step,
            "chunk_size": chunk_size,
            "kl_weight": kl_weight,
        },
        final_ckpt_path,
    )

    print(f"Final checkpoint saved to {final_ckpt_path}")

    return {
        "run_name": run_name(chunk_size, kl_weight),
        "chunk_size": chunk_size,
        "kl_weight": kl_weight,
        "training_steps": step,
        "best_loss": best_loss,
        "best_step": best_step,
        "final_loss": loss_value,
        "run_dir": str(run_dir),
        "status": "ok",
    }

def main():
    print("------------------RUNNING NEW VERSION-----------------------")
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

    output_root = Path("outputs/act_transfer_flower_gridsearch_mask")
    output_root.mkdir(parents=True, exist_ok=True)

    summary_csv = output_root / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name",
            "chunk_size",
            "kl_weight",
            "training_steps",
            "best_loss",
            "best_step",
            "final_loss",
            "run_dir",
            "status",
            "error",
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    chunk_sizes = [30, 50, 75]
    kl_weights = [1.0, 3.0]

    training_steps = 2000
    batch_size = 8
    log_freq = 100
    checkpoint_freq = 500
    optimizer_lr = 2e-5
    optimizer_lr_backbone = 1e-5

    for chunk_size in chunk_sizes:
        for kl_weight in kl_weights:
            try:
                result = train_one_run(
                    chunk_size=chunk_size,
                    kl_weight=kl_weight,
                    dataset_root_a=dataset_root_a,
                    dataset_root_b=dataset_root_b,
                    dataset_root_c=dataset_root_c,
                    bad_episodes_a=bad_episodes_a,
                    bad_episodes_b=bad_episodes_b,
                    bad_episodes_c=bad_episodes_c,
                    output_root=output_root,
                    device=device,
                    training_steps=training_steps,
                    batch_size=batch_size,
                    log_freq=log_freq,
                    checkpoint_freq=checkpoint_freq,
                    optimizer_lr=optimizer_lr,
                    optimizer_lr_backbone=optimizer_lr_backbone,
                    seed=0,
                )

                with open(summary_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result["run_name"],
                        result["chunk_size"],
                        result["kl_weight"],
                        result["training_steps"],
                        result["best_loss"],
                        result["best_step"],
                        result["final_loss"],
                        result["run_dir"],
                        result["status"],
                        "",
                    ])

            except Exception as e:
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                failed_run_name = run_name(chunk_size, kl_weight)
                print(f"\nFAILED: {failed_run_name}")
                print(err)

                with open(summary_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        failed_run_name,
                        chunk_size,
                        kl_weight,
                        "",
                        "",
                        "",
                        "",
                        str(output_root / failed_run_name),
                        "failed",
                        err,
                    ])

    print(f"\nGrid search complete. Summary written to {summary_csv}")

if __name__ == "__main__":
    main()