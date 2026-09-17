from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="grasp_cube",
    root="/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube/20260529_162433",
)

print(type(dataset.hf_dataset))
print(dataset.hf_dataset.column_names)
print(dataset.meta)

row0 = dataset.hf_dataset[0]

for k in row0:
    print(k, type(row0[k]))
    