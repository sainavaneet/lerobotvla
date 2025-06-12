#!/bin/bash
source /opt/ros/${ROS_DISTRO}/setup.bash

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5


eval "$(conda shell.bash hook)"

# Run the Python script with passed arguments
conda activate env_isaaclab
python dataset/dataset_creator.py "$@"