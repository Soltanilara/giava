import os
from lerobot.datasets import LeRobotDataset

def upload_to_hub(repo_id: str, dataset_root: str):
    dataset_path="/home/jinyu/GitHub/gaze-av-aloha/gaze_av_aloha/robot_scripts/outputs_jy/iantc105/av_aloha_place_tube"
    #dataset_path = os.path.join(dataset_root, repo_id)
    print(f"Loading dataset from: {dataset_path}")

    # 注意：把本地路径当成 repo_id 传入即可
    dataset = LeRobotDataset(repo_id, root=dataset_path)

    print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
    dataset.push_to_hub(repo_id)        # 如果 push_to_hub 需要 repo_id，可显式传
    print("Upload complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Upload an existing LeRobot dataset to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="iantc104/av_aloha_place_tube_v3",
        help="Hugging Face dataset repo ID, e.g., 'your-username/your-dataset-name'",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="outputs_jy",
        help="Root directory where the dataset is stored locally.",
    )

    args = parser.parse_args()
    upload_to_hub(args.repo_id, args.root)