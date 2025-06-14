import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.motors.feetech import TorqueMode
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)


class SO100Robot:
    def __init__(self, port="/dev/ttyACM0"):
        self.config = SO101FollowerConfig(port=port)
        self.config.cameras = {}
        self.config.leader_arms = {}
        self.robot = make_robot_from_config(self.config)

    def go_home(self):
        print("Moving to home pose...")
        self.robot.send_action(self.robot.initial_pose)
        time.sleep(2)

def disable_torque(robot):
    robot.robot.bus.disable_torque()


#################################################################################

if __name__ == "__main__":
    robot = SO100Robot()
    robot.robot.connect()
    robot.go_home()
    time.sleep(3)
    robot.robot.disconnect()