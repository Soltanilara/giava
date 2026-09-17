import numpy as np
from gaze_av_aloha.robot.config import REAL_DT, FPS
from gym_av_aloha.vr.headset import WebRTCHeadset
from gym_av_aloha.vr.headset_control_no_left import HeadsetControl
from gym_av_aloha.vr.headset_utils import HeadsetFeedback
from gaze_av_aloha.robot.env_no_left import RealEnv
from lerobot.datasets import LeRobotDataset
import time
import os
from gym_av_aloha.utils.dataset_utils import interpolate_data
import torch

def reset_env(env, headset):
    """Resets the robot environment and sends feedback to the headset."""
    print("Resetting the environment...")
    feedback = HeadsetFeedback()
    feedback.info = "Resetting the environment..."
    headset.send_feedback(feedback)
    env.reset()

def waiting_zone(env: RealEnv, headset: WebRTCHeadset):
    """
    A waiting area where the user can freely move the robot arms without recording.
    The user can exit this zone to start an episode.
    """
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset()
    action = None
        
    print("Waiting zone...")
    while True:
        start_time = time.time()

        # Create a copy of the action to modify, removing the left arm
        action_for_env = {}
        if headset_control.is_running() and action is not None:
            action_for_env = action.copy()
            if 'left_arm_pose' in action_for_env:
                del action_for_env['left_arm_pose']
            _, info = env.step_pose(**action_for_env)
        else:
            info = env.get_info()        

        headset_data = headset.receive_data()
        if headset_data is not None:
            # Get the action and feedback from the headset control, excluding the left arm
            action, feedback = headset_control.run(
                headset_data, 
                right_arm_pose=info['right_arm_pose'],
                middle_arm_pose=info['middle_arm_pose'],
            )

            # Start moving arms if the user holds the right button, without checking left arm sync
            if not headset_control.is_running() and \
                headset_data.r_button_one == True and feedback.head_out_of_sync == False and \
                feedback.right_out_of_sync == False: # Removed left_out_of_sync check
                headset_control.start(
                    headset_data, 
                    info['middle_arm_pose'],
                )

            # Stop moving arms if the user lets go of the right button
            if headset_control.is_running() and headset_data.r_button_one == False:
                headset_control.reset()

            # Break if the user holds the left button
            if headset_data.l_button_one == True:
                break
        
        feedback.info = f"Align and hold A to move. Press X to continue."
        headset.send_feedback(feedback)
        time_until_next_step = REAL_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step)) 


def combine_data(dataset: LeRobotDataset, eye_data):
    """Interpolates and adds eye tracking data to the dataset buffer."""
    aligned_left_eye = interpolate_data(
        data = np.array([frame['left_eye'] for frame in eye_data]),
        valid_indices = np.array([frame['left_eye_frame_id'] for frame in eye_data]),
        total_len = dataset.episode_buffer['size'],
    )
    aligned_right_eye = interpolate_data(
        data = np.array([frame['right_eye'] for frame in eye_data]),
        valid_indices = np.array([frame['right_eye_frame_id'] for frame in eye_data]),
        total_len = dataset.episode_buffer['size'],
    )
    dataset.episode_buffer['left_eye'] = [torch.tensor(x.copy()) for x in aligned_left_eye]
    dataset.episode_buffer['right_eye'] = [torch.tensor(x.copy()) for x in aligned_right_eye]


def run_episode(dataset: LeRobotDataset, env: RealEnv, headset: WebRTCHeadset, episode_idx: int, task: str):
    """
    Runs a single data recording episode.
    """
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset()
    action = None

    env.stereo_cam.set_count(0)

    # Wait for user to start the episode
    print("Waiting for user to start the episode...")
    while True:
        start_time = time.time()
        
        info = env.get_info()

        headset_data = headset.receive_data()
        if headset_data is not None:
            # Get action and feedback, excluding left arm
            action, feedback = headset_control.run(
                headset_data, 
                right_arm_pose=info['right_arm_pose'],
                middle_arm_pose=info['middle_arm_pose'],
            )
            # Break if the user holds the right button, without checking left arm sync
            if headset_data.r_button_one == True and feedback.head_out_of_sync == False and \
                feedback.right_out_of_sync == False: # Removed left_out_of_sync check
                headset_control.start(
                    headset_data, 
                    info['middle_arm_pose'],
                )
                break

        feedback.info = f"Align and hold A to start the episode {episode_idx}."
        headset.send_feedback(feedback)

        time_until_next_step = REAL_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step))

    # Run the episode
    print(f"Starting episode {episode_idx}...")

    eye_data = []
    step_idx = 0
    obs = env.get_obs()
    info = env.get_info()
    while True:
        step_start = time.time()

        env.stereo_cam.set_count(step_idx)

        # Assuming the 21-element vector is [left(7), right(7), middle(7)]
        # We combine the right and middle arm data to create a 14-element vector.
        

        frame = {
            'action': torch.tensor(obs['control'], dtype=torch.float32),
            'observation.state': torch.tensor(obs['joints'], dtype=torch.float32),
            'observation.images.left_eye_cam': obs['images']['left_eye_cam'],
            'observation.images.right_eye_cam': obs['images']['right_eye_cam'],
            # 'observation.images.wrist_cam_left' is removed
            'observation.images.wrist_cam_right': obs['images']['wrist_cam_right'],
            'observation.images.overhead_cam': obs['images']['overhead_cam'],
            'observation.images.worms_eye_cam': obs['images']['worms_eye_cam'],
            'left_eye': torch.zeros(2, dtype=torch.float32),
            'right_eye': torch.zeros(2, dtype=torch.float32),
            # 'left_arm_pose' is removed
            'right_arm_pose': torch.tensor(info['right_arm_pose'].reshape(-1), dtype=torch.float32),
            'middle_arm_pose': torch.tensor(info['middle_arm_pose'].reshape(-1), dtype=torch.float32),
        }
        dataset.add_frame(frame, task=task)

        # Create a copy of the action to modify for the environment step
        action_for_env = action.copy()
        if 'left_arm_pose' in action_for_env:
            del action_for_env['left_arm_pose']

        # Take a step in the environment using the action without the left arm
        obs, info = env.step_pose(**action_for_env)

        # Receive data from the headset
        headset_data = headset.receive_data()
        if headset_data is not None:
            action, feedback = headset_control.run(
                headset_data, 
                right_arm_pose=info['right_arm_pose'],
                middle_arm_pose=info['middle_arm_pose'],
            )
            if headset_data.r_button_one == False:
                print("Episode finished by user.")
                break
            
            # Save the eye data
            eye_frame = {}
            eye_frame['left_eye'] = headset_data.l_eye.copy()
            eye_frame['right_eye'] = headset_data.r_eye.copy()
            eye_frame['left_eye_frame_id'] = headset_data.l_eye_frame_id
            eye_frame['right_eye_frame_id'] = headset_data.r_eye_frame_id
            l_h = obs['images']['left_eye_cam'].shape[0]#480
            l_w = obs['images']['left_eye_cam'].shape[1]#640
            r_h = obs['images']['right_eye_cam'].shape[0]
            r_w = obs['images']['right_eye_cam'].shape[1]
            eye_frame['left_eye'][0] = (eye_frame['left_eye'][0] / l_w) * 2 - 1
            eye_frame['left_eye'][1] = (eye_frame['left_eye'][1] / l_h) * 2 - 1
            eye_frame['right_eye'][0] = (eye_frame['right_eye'][0] / r_w) * 2 - 1
            eye_frame['right_eye'][1] = (eye_frame['right_eye'][1] / r_h) * 2 - 1
            eye_data.append(eye_frame)

        # feedback.info = f"Episode {episode_idx}, Timestep: {str(step_idx).zfill(4)}\n{info}"
        feedback.info = f"Episode {episode_idx}, Timestep: {str(step_idx).zfill(4)}"
        headset.send_feedback(feedback) 

        time_until_next_step = REAL_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step)) 

        step_idx += 1

    combine_data(dataset, eye_data)      
    
    return True

def confirm_episode(headset: WebRTCHeadset, episode_idx):
    """Waits for user confirmation to save or discard the episode."""
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset()

    print("Waiting for user to redo or do next episode...")
    while True:
        start_time = time.time()

        headset_data = headset.receive_data()
        if headset_data is not None:
            if headset_data.l_button_one == True:
                return True
            elif headset_data.l_button_two == True:
                return False      
              
        feedback.info = f"Episode {episode_idx} completed. Press X to start next episode or Y to redo."
        headset.send_feedback(feedback)
        
        time_until_next_step = REAL_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step))
    
def main(cfg):
    print(f"Starting the dataset recording script, using fps: {FPS}")
    print(cfg)
    current_episode = 0
    # Reduced camera count from 6 to 5
    num_cameras = 5 
    dataset = LeRobotDataset.create(
        repo_id=cfg['repo_id'],
        root=os.path.join(cfg['root'], cfg['repo_id']),
        fps=FPS,
        features={
            "observation.images.left_eye_cam": {
                "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"],
            },
            "observation.images.right_eye_cam": {
                "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"],
            },
            # Removed "observation.images.wrist_cam_left"
            "observation.images.wrist_cam_right": {
                "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"],
            },
            "observation.images.overhead_cam": {
                "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"],
            },
            "observation.images.worms_eye_cam": {
                "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"],
            },
            # Changed shape from (21,) to (14,)
            "observation.state": {
                "dtype": "float32", "shape": (14,), "names": None,
            },
            # Changed shape from (21,) to (14,)
            "action": {
                "dtype": "float32", "shape": (14,), "names": None,
            },
            "left_eye": {
                "dtype": "float32", "shape": (2,), "names": None,
            },
            "right_eye": {
                "dtype": "float32", "shape": (2,), "names": None,
            },
            # Removed "left_arm_pose"
            "right_arm_pose": {
                "dtype": "float32", "shape": (16,), "names": None,
            },
            "middle_arm_pose": {
                "dtype": "float32", "shape": (16,), "names": None,
            },
        },
        image_writer_threads=num_cameras,
        image_writer_processes=4 * num_cameras,
    )

    if dataset.num_episodes < cfg['num_episodes']:
        headset = WebRTCHeadset()
        headset.run_in_thread()

        env = RealEnv(init_node=True, headset=headset)
        reset_env(env, headset)
        while True:
            if dataset.num_episodes >= cfg['num_episodes']:
                break

            episode_idx = current_episode
            waiting_zone(env, headset)
            
            reset_env(env, headset)

            ok = run_episode(dataset, env, headset, episode_idx, cfg['task'])

            if not ok:
                dataset.clear_episode_buffer()
                continue

            ok = confirm_episode(headset, episode_idx)

            if not ok:
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            current_episode += 1
            #save_progress(progress_file, {"current_episode": current_episode})

            # Upload the dataset to Hugging Face after each batch
            if current_episode % cfg['batch_size'] == 0:
                dataset.push_to_hub()
                print(f"Uploaded episodes {current_episode - cfg['batch_size']} to {current_episode - 1} to Hugging Face.")

        waiting_zone(env, headset)
        reset_env(env, headset)
        env.sleep_no_left()
    #     while True:
    #         if dataset.num_episodes >= cfg['num_episodes']:
    #             break

    #         episode_idx = dataset.num_episodes
    #         waiting_zone(env, headset)
            
    #         reset_env(env, headset)

    #         ok = run_episode(dataset, env, headset, episode_idx, cfg['task'])

    #         if not ok:
    #             dataset.clear_episode_buffer()
    #             continue

    #         ok = confirm_episode(headset, episode_idx)

    #         if not ok:
    #             dataset.clear_episode_buffer()
    #             continue

    #         dataset.save_episode()

    #     waiting_zone(env, headset)
    #     reset_env(env, headset)
    #     env.sleep_no_left()

    # dataset.push_to_hub()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record simulation episodes for AV Aloha.")
    parser.add_argument("--num-episodes", type=int, default=80, help="Number of episodes to record.")
    parser.add_argument("--repo-id", type=str, default="Jinyu220/vedio_55", help="Repository ID for the dataset.")
    parser.add_argument("--root", type=str, default="vedio_55", help="Root directory for the dataset.")
    parser.add_argument("--task", type=str, default="vedio_55", help="Task name for the dataset.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of episodes to record before uploading to Hugging Face.")
    #parser.add_argument("--process", type=str, default="/home/jinyu/GitHub/gaze-av-aloha/gaze_av_aloha/robot_scripts/outputs_single_put_coinv3/progress.json", help="Number of episodes to record before uploading to Hugging Face.")
    args = parser.parse_args()
    
    args_dict = vars(args)

    import traceback
    import rospy
    def shutdown():
        print("Shutting down...")
        os._exit(42)
    rospy.on_shutdown(shutdown)
    try:
        main(args_dict)
    except Exception as e:
        print(f"An error occured: {e}")
        traceback.print_exc()
    finally:
        print("Shutting down...")
        os._exit(42)
