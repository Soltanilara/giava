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

EXPERIMENTS = [
    "rgb_plus_blue_mask",
    "masked_rgb_only",
    "rgb_plus_centroids",
    "centroids_plus_vectors",
    "rgb_plus_blue_mask_plus_centroids",
    "masked_rgb_plus_centroids",
]

CHUNK_SIZE = 75
KL_WEIGHT = 1.0
TRAINING_STEPS = 5000
BATCH_SIZE = 8
LOG_FREQ = 100
CHECKPOINT_FREQ = 500
OPTIMIZER_LR = 2e-5
OPTIMIZER_LR_BACKBONE = 1e-5
SEED = 0

info_path = Path("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/20260601_000935/meta/info.json")
info = json.loads(info_path.read_text())
codebase_version = info["codebase_version"]

priors = loadmat("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/scene_priors.mat", squeeze_me=True, struct_as_record=False)

scene = priors["scenePriors"]

flower_mask = scene.flowerMask.astype(np.uint8)
oval_mask = scene.ovalMask.astype(np.uint8)

flower_centroid = scene.flowerCentroidNorm.astype(np.float32)
oval_centroid = scene.ovalCentroidNorm.astype(np.float32)

hmin = float(scene.lightBlueHMin)
hmax = float(scene.lightBlueHMax)
smin = float(scene.lightBlueSMin)
vmin = float(scene.lightBlueVMin)

print(flower_centroid)
print(oval_centroid)

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


def run_name(variant, chunk_size, kl_weight):
    return f"{variant}__chunk_{sanitize_value(chunk_size)}__kl_{sanitize_value(kl_weight)}"


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

    top_scene_feat = input_features["observation.images.top_scene"]
    top_scene_shape = getattr(top_scene_feat, "shape", (3, 480, 640))
    _, H, W = top_scene_shape

    mask_variants = {
        "rgb_plus_blue_mask",
        "rgb_plus_blue_mask_plus_centroids",
    }

    centroid_variants = {
        "rgb_plus_centroids",
        "centroids_plus_vectors",
        "rgb_plus_blue_mask_plus_centroids",
        "masked_rgb_plus_centroids",
    }

    if variant in mask_variants:
        set_feature(
            "observation.object_mask",
            FeatureType.VISUAL,
            (3, H, W),
            "float32",
        )

    if variant in centroid_variants:
        set_feature("observation.object_centroid", FeatureType.STATE, (2,), "float32")
        set_feature("observation.flower_target_centroid", FeatureType.STATE, (2,), "float32")
        set_feature("observation.oval_target_centroid", FeatureType.STATE, (2,), "float32")

    if variant in {
        "centroids_plus_vectors",
        "rgb_plus_centroids",
        "rgb_plus_blue_mask_plus_centroids",
        "masked_rgb_plus_centroids",
    }:
        set_feature("observation.scene_geometry", FeatureType.STATE, (14,), "float32")
        set_feature("observation.object_area", FeatureType.STATE, (1,), "float32")
        set_feature("observation.object_found", FeatureType.STATE, (1,), "float32")

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
        if k in dataset_metadata.features
    }

    dataset_a = LeRobotDataset(
        repo_id="deviamar/transfer_flower",
        root=dataset_root_a,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    print("Printing dataset_a features:")
    print(dataset_a.features.keys())
    print(dataset_a.meta.info["features"].keys())

    dataset_b = LeRobotDataset(
        repo_id="deviamar/transfer_flower",
        root=dataset_root_b,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    print("Printing dataset_b features:")
    print(dataset_b.features.keys())
    print(dataset_b.meta.info["features"].keys())


    dataset_c = LeRobotDataset(
        repo_id="deviamar/transfer_flower",
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

def preprocess_batch_for_variant(batch, variant, priors):
    scene = priors["scenePriors"]

    flower_mask_np = scene.flowerMask.astype(np.uint8)
    oval_mask_np = scene.ovalMask.astype(np.uint8)
    flower_centroid_np = scene.flowerCentroidNorm.astype(np.float32)
    oval_centroid_np = scene.ovalCentroidNorm.astype(np.float32)

    hmin = float(scene.lightBlueHMin)
    hmax = float(scene.lightBlueHMax)
    smin = float(scene.lightBlueSMin)
    vmin = float(scene.lightBlueVMin)

    mask_variants = {
        "rgb_plus_blue_mask",
        "rgb_plus_blue_mask_plus_centroids",
    }

    centroid_variants = {
        "rgb_plus_centroids",
        "centroids_plus_vectors",
        "rgb_plus_blue_mask_plus_centroids",
        "masked_rgb_plus_centroids",
    }

    geometry_variants = {
        "rgb_plus_centroids",
        "centroids_plus_vectors",
        "rgb_plus_blue_mask_plus_centroids",
        "masked_rgb_plus_centroids",
    }

    def to_float_rgb(imgs):
        if imgs.dtype != torch.float32:
            imgs = imgs.float()
        else:
            imgs = imgs.clone()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        return imgs

    def stack_np(items, device, dtype=torch.float32):
        return torch.from_numpy(np.stack(items)).to(device=device, dtype=dtype)

    def extract_blue_object_features(rgb_img):
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
            return {
                "object_mask": np.zeros((h, w), dtype=np.uint8),
                "object_centroid": np.array([0.0, 0.0], dtype=np.float32),
                "object_area": np.array([0.0], dtype=np.float32),
                "found": np.array([0.0], dtype=np.float32),
            }

        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = (labels == largest_idx).astype(np.uint8)
        cx, cy = centroids[largest_idx]
        area = stats[largest_idx, cv2.CC_STAT_AREA] / float(h * w)

        return {
            "object_mask": largest_mask,
            "object_centroid": np.array([cx / w, cy / h], dtype=np.float32),
            "object_area": np.array([area], dtype=np.float32),
            "found": np.array([1.0], dtype=np.float32),
        }

    device = batch["observation.images.top_scene"].device
    top_imgs = batch["observation.images.top_scene"]
    rgb = to_float_rgb(top_imgs)

    B, C, H, W = rgb.shape

    object_masks_np = []
    object_centroids_np = []
    object_areas_np = []
    found_flags_np = []

    for img in rgb:
        img_np = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        feats = extract_blue_object_features(img_np)
        object_masks_np.append(feats["object_mask"])
        object_centroids_np.append(feats["object_centroid"])
        object_areas_np.append(feats["object_area"])
        found_flags_np.append(feats["found"])

    object_masks = stack_np(object_masks_np, device).unsqueeze(1)
    object_centroids = stack_np(object_centroids_np, device)
    object_areas = stack_np(object_areas_np, device).view(B, 1)
    found_flags = stack_np(found_flags_np, device).view(B, 1)

    flower_mask = torch.from_numpy(flower_mask_np).to(device=device, dtype=torch.float32)
    oval_mask = torch.from_numpy(oval_mask_np).to(device=device, dtype=torch.float32)

    flower_masks = flower_mask.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1)
    oval_masks = oval_mask.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1)

    flower_centroids = torch.from_numpy(flower_centroid_np).to(device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    oval_centroids = torch.from_numpy(oval_centroid_np).to(device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1)

    combined_scene_mask = torch.clamp(object_masks + flower_masks + oval_masks, 0.0, 1.0)

    masked_rgb = rgb * combined_scene_mask

    mask_scene = combined_scene_mask.repeat(1, 3, 1, 1)

    obj_to_flower = flower_centroids - object_centroids
    obj_to_oval = oval_centroids - object_centroids

    dist_flower = torch.norm(obj_to_flower, dim=1, keepdim=True)
    dist_oval = torch.norm(obj_to_oval, dim=1, keepdim=True)

    geom = torch.cat(
        [
            object_centroids,
            flower_centroids,
            oval_centroids,
            obj_to_flower,
            obj_to_oval,
            dist_flower,
            dist_oval,
            object_areas,
            found_flags,
        ],
        dim=1,
    )

    if variant in centroid_variants:
        batch["observation.object_centroid"] = object_centroids
        batch["observation.flower_target_centroid"] = flower_centroids
        batch["observation.oval_target_centroid"] = oval_centroids

    if variant in geometry_variants:
        batch["observation.object_area"] = object_areas
        batch["observation.object_found"] = found_flags
        batch["observation.scene_geometry"] = geom

    if variant in mask_variants:
        batch["observation.object_mask"] = mask_scene

    if variant == "rgb_plus_blue_mask":
        batch["observation.images.top_scene"] = rgb

    elif variant == "masked_rgb_only":
        batch["observation.images.top_scene"] = masked_rgb

    elif variant == "rgb_plus_centroids":
        batch["observation.images.top_scene"] = rgb

    elif variant == "rgb_plus_blue_mask_plus_centroids":
        batch["observation.images.top_scene"] = rgb

    elif variant == "masked_rgb_plus_centroids":
        batch["observation.images.top_scene"] = masked_rgb

    elif variant == "centroids_plus_vectors":
        batch["observation.images.top_scene"] = rgb

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return batch

def print_input_features(variant, input_features, output_features=None):
    print("\n" + "=" * 100)
    print(f"INPUT FEATURES FOR VARIANT: {variant}")
    print("-" * 100)

    for key in sorted(input_features.keys()):
        feat = input_features[key]
        feat_type = getattr(feat, "type", "UNKNOWN")
        feat_shape = getattr(feat, "shape", "UNKNOWN")
        feat_dtype = getattr(feat, "dtype", "UNKNOWN")
        print(f"{key}")
        print(f"  type : {feat_type}")
        print(f"  shape: {feat_shape}")
        print(f"  dtype: {feat_dtype}")

    if output_features is not None:
        print("-" * 100)
        print("OUTPUT FEATURES")
        for key in sorted(output_features.keys()):
            feat = output_features[key]
            feat_type = getattr(feat, "type", "UNKNOWN")
            feat_shape = getattr(feat, "shape", "UNKNOWN")
            feat_dtype = getattr(feat, "dtype", "UNKNOWN")
            print(f"{key}")
            print(f"  type : {feat_type}")
            print(f"  shape: {feat_shape}")
            print(f"  dtype: {feat_dtype}")

    print("=" * 100 + "\n")

def tensor_chw_to_bgr_uint8(img_t):
    img = img_t.detach().cpu()
    if img.dtype != torch.float32:
        img = img.float()
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



def save_debug_views(
    run_dir,
    variant,
    batch_before,
    batch_after,
    declared_input_keys,
    sample_idx=0,
):
    debug_dir = run_dir / "debug_first_batch"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if "observation.images.top_scene" in declared_input_keys:
        rgb_before = batch_before["observation.images.top_scene"][sample_idx]
        rgb_after = batch_after["observation.images.top_scene"][sample_idx]

        rgb_before_bgr = tensor_chw_to_bgr_uint8(rgb_before)
        rgb_after_bgr = tensor_chw_to_bgr_uint8(rgb_after)

        cv2.imwrite(str(debug_dir / f"{variant}_00_rgb_before.png"), rgb_before_bgr)
        cv2.imwrite(str(debug_dir / f"{variant}_01_rgb_after.png"), rgb_after_bgr)

    if "observation.object_mask" in declared_input_keys and "observation.object_mask" in batch_after:
        mask_scene_bgr = tensor_chw_to_bgr_uint8(batch_after["observation.object_mask"][sample_idx])
        cv2.imwrite(str(debug_dir / f"{variant}_02_mask_scene.png"), mask_scene_bgr)

    summary_path = debug_dir / f"{variant}_batch_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"variant: {variant}\n")
        f.write("Declared ACT input features:\n")
        for k in sorted(declared_input_keys):
            f.write(f"  {k}\n")
        f.write("\nBatch tensors actually present after preprocess:\n")

        for k, v in sorted(batch_after.items()):
            if torch.is_tensor(v):
                f.write(
                    f"{k}: shape={tuple(v.shape)} dtype={v.dtype} "
                    f"min={float(v.min().item()):.6f} max={float(v.max().item()):.6f}"
                )
                if k not in declared_input_keys:
                    f.write("   [NOT DECLARED AS ACT INPUT]")
                f.write("\n")
            else:
                f.write(f"{k}: type={type(v)}")
                if k not in declared_input_keys:
                    f.write("   [NOT DECLARED AS ACT INPUT]")
                f.write("\n")

        f.write("\nDeclared state inputs only:\n")
        for k in sorted(declared_input_keys):
            if k.startswith("observation.") and not k.startswith("observation.images.") and k in batch_after:
                v = batch_after[k]
                if torch.is_tensor(v):
                    f.write(
                        f"{k}: shape={tuple(v.shape)} dtype={v.dtype} "
                        f"min={float(v.min().item()):.6f} max={float(v.max().item()):.6f}\n"
                    )

        f.write("\nUnexpected extra observation keys:\n")
        extras = [k for k in batch_after.keys() if k.startswith("observation.") and k not in declared_input_keys]
        if extras:
            for k in sorted(extras):
                f.write(f"  {k}\n")
        else:
            f.write("  none\n")

# def save_topcam_prior_alignment(run_dir, batch_after, sample_idx=0):
#     debug_dir = run_dir / "debug_first_batch"
#     debug_dir.mkdir(parents=True, exist_ok=True)

#     rgb = tensor_chw_to_bgr_uint8(batch_after["observation.images.top_scene"][sample_idx])

#     if "observation.object_mask" in batch_after:
#         mask_scene = tensor_chw_to_bgr_uint8(batch_after["observation.object_mask"][sample_idx])
#         cv2.imwrite(str(debug_dir / "prior_alignment_mask_scene.png"), mask_scene)

#     if all(k in batch_after for k in [
#         "observation.flower_target_centroid",
#         "observation.oval_target_centroid",
#     ]):
#         vis = rgb.copy()
#         vis = draw_centroid_on_bgr(
#             vis,
#             batch_after["observation.flower_target_centroid"][sample_idx].detach().cpu().numpy(),
#             color=(255, 0, 255),
#             label="flower_target",
#         )
#         vis = draw_centroid_on_bgr(
#             vis,
#             batch_after["observation.oval_target_centroid"][sample_idx].detach().cpu().numpy(),
#             color=(255, 255, 0),
#             label="oval_target",
#         )
#         cv2.imwrite(str(debug_dir / "prior_alignment_centroids.png"), vis)

def train_one_run(
    variant,
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
    priors,
    training_steps=5000,
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

    dataset_metadata, input_features, output_features = build_features_for_variant(
        dataset_root_a, variant
    )

    declared_input_keys = set(input_features.keys())

    print_input_features(
        variant=variant,
        input_features=input_features,
        output_features=output_features,
    )

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
    print(f"Starting run: variant={variant}, chunk_size={chunk_size}, kl_weight={kl_weight}")
    print(cfg)
    print("action delta indices:", cfg.action_delta_indices)
    print("observation delta indices:", cfg.observation_delta_indices)
    print("chunk size:", cfg.chunk_size)
    print("action steps:", cfg.n_action_steps)
    print("=" * 80)

    run_dir = output_root / run_name(variant, chunk_size, kl_weight)
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_saved = False

    loss_log_path = run_dir / "loss.csv"
    with open(loss_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "recon_loss", "kl_loss"])

    cfg_json_path = run_dir / "run_config.json"
    with open(cfg_json_path, "w") as f:
        json.dump(
            {
                "variant": variant,
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

            batch_before = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            batch = preprocess_batch_for_variant(batch, variant, priors)

            if not debug_saved:
                print(f"\n[DEBUG] Verifying first batch for variant={variant}")
                print("Batch keys after preprocess:")
                for k in sorted(batch.keys()):
                    v = batch[k]
                    if torch.is_tensor(v):
                        print(
                            f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} "
                            f"min={float(v.min().item()):.6f} max={float(v.max().item()):.6f}"
                        )
                    else:
                        print(f"  {k}: type={type(v)}")

                save_debug_views(run_dir, variant, batch_before, batch, declared_input_keys, sample_idx=0)
                print(f"[DEBUG] Wrote debug files to {run_dir / 'debug_first_batch'}")
                debug_saved = True

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
                    f"[{run_name(variant, chunk_size, kl_weight)}] "
                    f"step={step} loss={loss_value:.6f} "
                    f"recon={recon_loss:.6f} kl={kl_loss_val:.6f}"
                )
                # done = True
                # break # Remove this break to run through the entire dataset

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
        "run_name": run_name(variant, chunk_size, kl_weight),
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
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/20260601_000935"
    )
    dataset_root_b = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/20260601_042518"
    )
    dataset_root_c = Path(
        "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/20260601_045338"
    )

    bad_episodes_a = {11, 15, 18, 32}
    bad_episodes_b = {7}
    bad_episodes_c = set()

    output_root = Path("outputs/act_transfer_flower_6experiments")
    output_root.mkdir(parents=True, exist_ok=True)

    summary_csv = output_root / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant",
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

    training_steps = 5000
    batch_size = 8
    log_freq = 100
    checkpoint_freq = 500
    optimizer_lr = 2e-5
    optimizer_lr_backbone = 1e-5

    for variant in EXPERIMENTS:
        try:
            result = train_one_run(
                variant=variant,
                chunk_size=CHUNK_SIZE,
                kl_weight=KL_WEIGHT,
                dataset_root_a=dataset_root_a,
                dataset_root_b=dataset_root_b,
                dataset_root_c=dataset_root_c,
                bad_episodes_a=bad_episodes_a,
                bad_episodes_b=bad_episodes_b,
                bad_episodes_c=bad_episodes_c,
                output_root=output_root,
                device=device,
                priors=priors,
                training_steps=training_steps,
                batch_size=batch_size,
                log_freq=log_freq,
                checkpoint_freq=checkpoint_freq,
                optimizer_lr=optimizer_lr,
                optimizer_lr_backbone=optimizer_lr_backbone,
                seed=SEED,
            )

            with open(summary_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    variant,
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
            print(f"\nFAILED: {variant}")
            print(err)

            with open(summary_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    variant,
                    CHUNK_SIZE,
                    KL_WEIGHT,
                    "",
                    "",
                    "",
                    "",
                    str(output_root / variant),
                    "failed",
                    err,
                ])

    print(f"\nAll 6 experiments complete. Summary written to {summary_csv}")

if __name__ == "__main__":
    main()