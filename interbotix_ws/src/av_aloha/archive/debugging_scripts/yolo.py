import torch
import ultralytics

from ultralytics import YOLO

model = YOLO("/home/devi/giava/lerobot/lerobot/common/policies/act/best_10epoch.pt")

print(model.model)

policy = ACTPolicy(cfg).cuda()

batch = next(iter(train_loader))

with torch.no_grad():
    actions, _ = policy.model(batch)

print(actions.shape)