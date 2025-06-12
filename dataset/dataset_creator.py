from __future__ import annotations
"""Launch Isaac Sim Simulator first."""
import argparse
import time
import os
import threading
import sys
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--custom_env", type=str, default="office", help="Setup the environment")
parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
parser.add_argument("--terrain", type=str, default="flat", help="Setup the robot")
parser.add_argument("--robot_amount", type=int, default=1, help="Setup the robot amount")
parser.add_argument("--action_log_dir", type=str, required=True, help="Directory to store action logs")
parser.add_argument("--task_index", type=int, default=1, help="Task index")
parser.add_argument("--is_last", type=bool, default=False, help="Is it last element of the episode")
parser.add_argument("--create_dataset", type=bool, default=False, help="Create a dataset")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs import cli_args

cli_args.add_rsl_rl_args(parser)

from isaaclab.app import AppLauncher # type: ignore
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
ext_manager.set_extension_enabled_immediate("omni.isaac.core_nodes", True)
while not ext_manager.is_extension_enabled("omni.isaac.core_nodes"):
    time.sleep(0.1)

import gymnasium as gym
import torch
import carb # type: ignore

from isaaclab_tasks.utils import get_checkpoint_path # type: ignore
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg # type: ignore

import isaaclab.sim as sim_utils # type: ignore
import omni.appwindow # type: ignore
from rsl_rl.runners import OnPolicyRunner

import rclpy

from envs.ros import RobotBaseNode, add_camera, pub_robo_data_ros2
from geometry_msgs.msg import Twist

from envs.utils.agent import unitree_go2_agent_cfg, unitree_g1_agent_cfg
from envs.env import UnitreeGo2CustomEnvCfg
import envs.env as env

from envs.utils.omni_graph import create_front_cam_omnigraph # type: ignore
import omni.timeline # type: ignore


from rclpy.node import Node

from envs.action_logger import ActionLogger

from key_input import KeyboardInputHandler  # <--- NEW IMPORT
import numpy as np
class Simulator(Node):
    def __init__(self):
        super().__init__('simulator_node')
        self.args = args_cli
        self._input = None
        self._appwindow = None
        self._keyboard = None
        self._sub_keyboard = None
        self._timeline = None
        self.env = None
        self.ppo_runner = None
        self.policy = None
        self.base_node = None
        self.cameras = None
        self.start_time = 0.0

        self.create_dataset = self.args.create_dataset

        if self.create_dataset:      
            self.action_logger = ActionLogger(log_dir=self.args.action_log_dir)
        self.task_index = self.args.task_index  
        self.keyboard_handler = KeyboardInputHandler()  # <--- USE HANDLER
        self.velocities = np.array([0.0, 0.0, 0.0])

        print("╔════════════════════════════════════════╗")
        print("║           Task Index Received         ║")
        print("╠════════════════════════════════════════╣")
        print(f"║              {self.task_index:^20}              ║")
        print("╚════════════════════════════════════════╝")

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        result = self.keyboard_handler.keyboard_event(event, env.base_command, self.args.robot_amount)
        if result == 'reset':
            self.env.unwrapped.command_manager.reset()
            self.start_time = time.time()
            return True
        return True

    def _setup_custom_env(self):
        print("--------------------------------Setting up custom environment--------------------------------")
        try:
            env_path = "/home/navaneet/go2_new/velocity_control/envs"
            if not os.path.exists(env_path):
                print(f"\n[ERROR] Environment directory not found at {os.path.abspath(env_path)}")
                return
            if (self.args.custom_env == "warehouse" and self.args.terrain == 'flat'):
                usd_path = 'envs/assets/new.usd'
                if not os.path.exists(usd_path):
                    print(f"\n[ERROR] Warehouse environment file not found at {usd_path}")
                    return
                cfg_scene = sim_utils.UsdFileCfg(usd_path=usd_path)
                cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))
            if (self.args.custom_env == "office" and self.args.terrain == 'flat'):
                usd_path = 'envs/assets/office.usd'
                if not os.path.exists(usd_path):
                    print(f"\n[ERROR] Office environment file not found at {usd_path}")
                    return
                cfg_scene = sim_utils.UsdFileCfg(usd_path=usd_path)
                cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))
        except Exception as e:
            print(f"\n[ERROR] Failed to load custom environment: {str(e)}")
 


    def _cmd_vel_cb(self, msg, num_robot):
        x = msg.linear.x
        y = msg.linear.y
        z = msg.angular.z
        env.base_command[str(num_robot)] = [x, y, z]

    def _add_cmd_sub(self, num_envs):
        node_test = rclpy.create_node('position_velocity_publisher')
        for i in range(num_envs):
            node_test.create_subscription(Twist, f'robot{i}/cmd_vel', lambda msg, i=i: self._cmd_vel_cb(msg, str(i)), 10)
        thread = threading.Thread(target=rclpy.spin, args=(node_test,), daemon=True)
        thread.start()

    def _specify_cmd_for_robots(self, num_envs):
        for i in range(num_envs):
            env.base_command[str(i)] = [0, 0, 0]
    def _log_actions_thread(self, obs_manager, actions, env_idx, task_index, action_logger):
        action_logger.log_actions(obs_manager, actions, env_idx=env_idx, task_index=task_index)

    def run(self):
        # acquire input interface
        self._input = carb.input.acquire_input_interface()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

        self._timeline = omni.timeline.get_timeline_interface()

        env_cfg = UnitreeGo2CustomEnvCfg()
        env_cfg.scene.num_envs = self.args.robot_amount

        for i in range(env_cfg.scene.num_envs):
            create_front_cam_omnigraph(i)

        self._specify_cmd_for_robots(env_cfg.scene.num_envs)

        agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg
        if self.args.robot == "g1":
            agent_cfg: RslRlOnPolicyRunnerCfg = unitree_g1_agent_cfg

        self.env = gym.make(self.args.task, cfg=env_cfg)
        self.env = RslRlVecEnvWrapper(self.env)


        resume_path = 'envs/utils/rl_policy/model_final.pt'
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        self.ppo_runner = OnPolicyRunner(self.env, agent_cfg, log_dir=None, device=agent_cfg["device"])
        self.ppo_runner.load(resume_path)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        self.policy = self.ppo_runner.get_inference_policy(device=self.env.unwrapped.device)
        obs, _ = self.env.get_observations()

        self.base_node = RobotBaseNode(env_cfg.scene.num_envs)
        self._add_cmd_sub(env_cfg.scene.num_envs)
        self.cameras = add_camera(env_cfg.scene.num_envs, self.args.robot)
        time.sleep(2)
        self._setup_custom_env()

        self.start_time = time.time()
        dt = self.env.unwrapped.step_dt

        while simulation_app.is_running():
            self.keyboard_handler.update_velocities(env.base_command)
            # Get the actual velocity values for robot 0
            vx, vy, vz = env.base_command["0"]
            # self.velocities = np.array([vx, vy, vz])

            with torch.inference_mode():
                actions = self.policy(obs)
                obs, _, _, _ = self.env.step(actions)
                obs_manager = self.env.unwrapped.observation_manager
                if self.create_dataset:
                    log_thread = threading.Thread(
                        target=self._log_actions_thread,
                        args=(obs_manager, actions, 0, self.task_index, self.action_logger)
                    )
                    log_thread.start()
                pub_robo_data_ros2(args_cli.robot, env_cfg.scene.num_envs, self.base_node, self.env, self.cameras, self.start_time)



if __name__ == "__main__":
    rclpy.init()
    try:
        simulator = Simulator()
        simulator.run()
    except KeyboardInterrupt:
        simulator.destroy_node()
        rclpy.shutdown()