import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import argparse
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig

class SO100Robot:
    def __init__(self, port="/dev/tty_follower_arm"):
        self.config = SO101FollowerConfig(port=port)
        self.robot_is_connected = False
        self.robot = make_robot_from_config(self.config)

    def connect(self):
        self.robot.connect()
        print("================> SO101 Robot is connected =================")

    def disconnect(self):
        try:
            # Add a small delay before disconnecting to allow motors to settle
            time.sleep(0.5)
            self.robot.disconnect()
            print("================> SO101 Robot disconnected")
        except RuntimeError as e:
            print(f"Warning: Error during disconnect: {e}")
            print("Attempting to force disconnect...")
            try:
                # Try one more time with a longer delay
                time.sleep(1.0)
                self.robot.disconnect()
                print("================> SO101 Robot force disconnected")
            except Exception as e:
                print(f"Error: Could not disconnect robot: {e}")

    def get_current_state(self):
        obs = self.robot.get_observation()
        state = np.array([
            obs['shoulder_pan.pos'],
            obs['shoulder_lift.pos'],
            obs['elbow_flex.pos'],
            obs['wrist_flex.pos'],
            obs['wrist_roll.pos'],
            obs['gripper.pos']
        ])
        return state

    def set_target_state(self, target_state):
        state_dict = {
            "shoulder_pan.pos": target_state[0],
            "shoulder_lift.pos": target_state[1],  
            "elbow_flex.pos": target_state[2],  
            "wrist_flex.pos": target_state[3],
            "wrist_roll.pos": target_state[4],
            "gripper.pos": target_state[5]
        }
        self.robot.send_action(state_dict)

def load_trajectory(parquet_file):
    """
    Load trajectory data from a parquet file.
    Expected columns: observation.state (as a list/array)
    """
    df = pd.read_parquet(parquet_file)
    return df

def execute_trajectory(robot, trajectory_df, delay=0.02):
    """
    Execute the trajectory on the robot by setting states.
    
    Args:
        robot: SO100Robot instance
        trajectory_df: DataFrame containing trajectory data
        delay: Time delay between actions in seconds
    """
    print("Starting trajectory execution...")
    
    try:
        # Execute trajectory
        for idx, row in tqdm(trajectory_df.iterrows(), total=len(trajectory_df), desc="Executing trajectory"):
            # Get state from the observation.state column
            state = np.array(row['observation.state'])
            
            # Set target state for the robot
            robot.set_target_state(state)
            time.sleep(delay)
            
        # Add a final delay to allow motors to settle
        time.sleep(1.0)
    except Exception as e:
        print(f"Error during trajectory execution: {e}")
        raise

def publish_trajectory():
    # Hardcoded values
    trajectory_file = "/home/navaneet/lerobotvla/datasets/traj/data/chunk-000/episode_000000.parquet"
    delay = 0.02
    port = "/dev/tty_follower_arm"
    
    # Initialize robot
    robot = SO100Robot(port=port)
    
    try:
        # Load trajectory
        print(f"Loading trajectory from {trajectory_file}")
        trajectory_df = load_trajectory(trajectory_file)
        
        # Execute trajectory
        robot.connect()
        execute_trajectory(robot, trajectory_df, delay)
            
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    publish_trajectory() 