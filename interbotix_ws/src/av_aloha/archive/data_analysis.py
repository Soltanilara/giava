import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parquet_path = "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/transfer_flower/20260601_000935/data/chunk-000/episode_000036.parquet"
#"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube/20260529_162433/data/chunk-000/episode_000000.parquet"
#"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_131838/data/chunk-000/episode_000013.parquet"
#"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube/20260529_162433/data/chunk-000/episode_000000.parquet"
#"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube/20260529_162433/data/chunk-000/episode_000000.parquet"
#"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_131838_rgb_only_grasp/data/chunk-000/episode_000000.parquet"
#"/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/block_square/20260528_131838/data/chunk-000/episode_000002.parquet"

# 1. Load parquet into a DataFrame
df = pd.read_parquet(parquet_path)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# 2. Extract state and action as numpy arrays
# Adjust these selectors if your schema differs.
state_cols = [c for c in df.columns if c.startswith("observation.state")]
action_cols = [c for c in df.columns if c.startswith("action")]

states = df[state_cols].to_numpy()   # shape [T, 7]
actions = df[action_cols].to_numpy() # shape [T, 7]

T = states.shape[0]
print(f"Timesteps T = {T}\n")

print(df.dtypes["action"])
print(df["action"].iloc[0], type(df["action"].iloc[0]))

for i, action in enumerate(df["action"]):
    a = np.array(action)
    joints = a[:6]
    gripper = a[6]
    print(f"t={i:03d}: joints={joints}, gripper={gripper:.3f}")

""" # 3. Compute per-step changes in joint positions (state)
#   - ignore gripper for now (first 6 dims)
joint_states = states[:, :6]
dq = np.linalg.norm(joint_states[1:] - joint_states[:-1], axis=1)

print("State Δq stats (per step, L2 over 6 joints):")
print("  mean:", dq.mean())
print("  median:", np.median(dq))
print("  max:", dq.max())

# 4. Compute per-step changes in commanded actions
joint_actions = actions[:, :6]
da = np.linalg.norm(joint_actions[1:] - joint_actions[:-1], axis=1)

print("Action Δa stats (per step, L2 over 6 joints):")
print("  mean:", da.mean())
print("  median:", np.median(da))
print("  max:", da.max())

# 5. Rough phase check: where does the gripper close?
gripper_state = states[:, 6]
gripper_action = actions[:, 6]

# Example: threshold crossing (you may adjust based on your open/close positions)
close_threshold = (gripper_state.min() + gripper_state.max()) / 2.0
close_indices = np.where(gripper_state < close_threshold)[0]
if len(close_indices) > 0:
    t_close = close_indices[0]
    print(f"First gripper-close (state) at step {t_close} (≈ t_close / fps seconds).")
else:
    print("No clear close event detected in gripper_state.")

# 6. Optional: visualize joint trajectories over time
time = np.arange(T)  # if fps=15, seconds = time / 15.0

plt.figure(figsize=(10, 6))
for j in range(6):
    plt.plot(time, joint_states[:, j], label=f"joint{j}")
plt.xlabel("timestep")
plt.ylabel("joint angle (rad)")
plt.title("Joint trajectories")
plt.legend()
plt.tight_layout()
plt.show()

# 7. Optional: visualize per-step Δq
plt.figure(figsize=(8, 4))
plt.plot(time[1:], dq)
plt.xlabel("timestep")
plt.ylabel("||Δq||")
plt.title("Per-step joint change magnitude")
plt.tight_layout()
plt.show() """