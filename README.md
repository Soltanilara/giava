# ALOHA Real Arm Setup and Operation

This repository contains the software used to operate the Aloha bimanipulation robotics platform for teleoperation, robot control, data collection, replay, and dataset uploading. This repository provides everything needed to set up the software stack on a fresh Ubuntu installation and operate the robot for data collection. 

This repository builds upon the original ALOHA software stack developed by Tony Zhao et al. If you are setting up a machine from scratch, first install ROS and complete the required Interbotix/ALOHA system configuration. Then follow the GIAVA-specific installation and setup instructions below.

## Prerequisites

Before installing this repository, complete the following tutorials:

- ROS Noetic Installation:
  https://wiki.ros.org/noetic/Installation/Ubuntu

- Original ALOHA repository:
  https://github.com/tonyzhaozh/aloha

- Interbotix ROS setup:
  https://docs.trossenrobotics.com/interbotix_xsarms_docs/

Complete the following once on a fresh machine.

✓ Install ROS Noetic

✓ Install Interbotix

✓ Create ~/interbotix_ws

✓ Configure udev rules

✓ Configure robot USB ports

✓ Configure camera USB ports

✓ Verify all devices appear under /dev

✓ Build the workspace

Remember to include all 3 in the Robot Configuration: /dev/ttyDXL_puppet_left, /dev/ttyDXL_puppet_right, /dev/ttyDXL_puppet_middle

## Software Setup
1 Install ROS

2 Follow instructions from Tony to setup everything

2 Install Miniconda

3 Clone repository

4 Initialize submodules

5 Create environment

6 Install Python packages

7 Configure USB devices

8 Build workspace

9 Verify installation

### System Requirements

Operating System: Ubuntu 20.04 LTS

Required Software:
ROS Noetic
Python 3.10
Conda (Miniconda or Anaconda)
Git

Installation
1. Install ROS Noetic

Install ROS Noetic by following the official installation guide.

After installation, verify:

roscore
2. Clone the Repository real-v2-spr26 branch
git clone [<repository_url>](https://github.com/Soltanilara/giava/tree/real-v2-spr26/interbotix_ws/src)

cd giava

git submodule update --init --recursive

This repository uses several Git submodules that are required for robot operation.

Current submodules include:

Interbotix
Pyroki
LeRobot
Gym AV
ALOHA
3. Create the Conda Environment
conda env create -f environment.yml

conda activate gym_av

If creating the environment manually:

conda create -n gym_av python=3.12

conda activate gym_av
4. Install Python Dependencies
pip install -r requirements.txt

Additional packages requiring manual installation:

LeRobot
pip install lerobot==0.6.0
aiortc

The VR teleoperation stack requires a patched fork of aiortc.

pip install git+https://github.com/ian-chuang/aiortc.git@91cdb627b2510dba80786f9236277f103617c87a

Do not install the default PyPI version.

5. Build the ROS Workspace
cd interbotix_ws

catkin_make

source devel/setup.bash
Robot Configuration
USB Device Naming

The robot arms use persistent symbolic device names created through udev rules.

Expected devices:

/dev/ttyDXL_puppet_left
/dev/ttyDXL_puppet_right
/dev/ttyDXL_puppet_middle

These symbolic links ensure each robot always appears under the same device name regardless of USB enumeration order.

(Include instructions here for installing the udev rules.)

Verify Connected Devices
ls /dev/ttyDXL*

Verify that all expected devices are present before launching the system.

Repository Structure
interbotix_ws/
    src/
        av_aloha/
        gym_av_aloha/
        interbotix_ros_*
        pyroki/
        lerobot/

datasets/

outputs/

checkpoints/
Running the System

Activate the environment

conda activate gym_av

Source ROS

source /opt/ros/noetic/setup.bash

Source the workspace

source ~/interbotix_ws/devel/setup.bash

Launch the robot

roslaunch ...
Common Workflows
Reset Robot
python reset_arm.py

Moves the robot to its home configuration.

Move Robot
python move_arm.py

Executes a specified joint or Cartesian motion.

Teleoperation
python teleop.py

Launches the VR teleoperation interface.

Collect Demonstrations
python record_episodes.py

Records synchronized robot observations and actions.

Replay Demonstrations
python replay_episode.py

Replays a recorded demonstration on the robot.

Upload Dataset
python upload_dataset.py

Converts and uploads demonstrations to the LeRobot dataset format.

Important Scripts
Script	Description
reset_arm.py	Home the robot
move_arm.py	Move the robot
record_episodes.py	Record demonstrations
replay_episode.py	Replay demonstrations
camera_manager.py	Camera interface
robot_factory.py	Create robot objects
arm_controller.py	Robot motion control
config.py	Global configuration
teleop_utils.py	Coordinate transforms and utilities
Data Organization

Document where demonstrations are saved.

Example:

datasets/

    task_name/

        episode_0000/

        episode_0001/

        episode_0002/

Include the naming convention and directory structure for images, robot states, actions, and metadata.

source ros and interbotix
roslaunch

new terminal, conda activate gym_av
python _
