# import os


# from huggingface_hub import login, upload_folder

# DATASET_ROOT = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/20260601_000935"
# task_name = "transfer_flower_mask"

# # login()

# print("\nUPLOADING DATASET...")

# upload_folder(
#     folder_path=os.path.join(
#         DATASET_ROOT,
#         task_name,
#     ),
#     repo_id=f"deviamar/transfer_flower_clean50",
#     repo_type="dataset",
# )

# print("\nUPLOAD COMPLETE")


from huggingface_hub import upload_folder

# upload_folder(
#     repo_id="deviamar/transfer_flower_clean50",
#     repo_type="dataset",
#     folder_path="/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/20260601_000935",
# )

# upload_folder(
#     repo_id="deviamar/transfer_flower_noisy1",
#     repo_type="dataset",
#     folder_path="/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/20260601_042518",
# )

upload_folder(
    repo_id="deviamar/transfer_flower_noisy2",
    repo_type="dataset",
    folder_path="/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower_mask/20260601_045338",
)