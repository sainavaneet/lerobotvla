import time
from contextlib import contextmanager
import threading
import queue
import collections

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.motors.feetech.feetech import TorqueMode
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError, InvalidActionError

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from gr00t.eval.service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################


class SO100Robot:
    def __init__(self, img_width, img_height, calibrate=False, enable_camera=False, cam_main_idx=6, port="/dev/tty_follower_arm"):
        self.config = SO101FollowerConfig(port=port)
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.cam_main_idx = cam_main_idx
        self.robot_is_connected = False
        
        # Add smoothing buffer for temporal ensembling
        self.action_buffer = collections.deque(maxlen=5)  # Store last 5 actions
        self.smoothing_weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # Weights for weighted average
        
        # Initialize cameras directly with OpenCV
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.main_camera = cv2.VideoCapture(cam_main_idx)
            # Set camera properties
            self.main_camera.set(cv2.CAP_PROP_FPS, 30)
            self.main_camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
            self.main_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
            
        self.config.leader_arms = {}
        print("Camera initialized:", self.enable_camera)

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)

    @contextmanager
    def activate(self):
        try:
            self.connect()
            yield
        finally:
            self.disconnect()

    def connect(self):
        self.robot.connect()

        if self.enable_camera:
            if not self.main_camera.isOpened():
                raise DeviceNotConnectedError("Failed to open camera")
            print("================> SO101 Robot is fully connected with camera =================")
        else:
            print("================> SO101 Robot is fully connected =================")

    # def go_home(self):
    #     # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
    #     print("-------------------------------- moving to home pose")

    #     # state [-0.11618123 -1.72656227  1.73625868  0.34490323  0.05362618  0.0602237 ]
    #     home_state_radians = np.array([-0.11618123, -1.72656227, 1.73625868, 0.34490323, 0.05362618, 0.0602237])
    #     home_state = torch.tensor(np.degrees(home_state_radians))
    #     print("home_state", home_state)
    #     self.set_target_state(home_state)
    #     time.sleep(2)

    def get_observation(self):
        obs = self.robot.get_observation()
        print("All observations:", obs)
        return obs

    def get_current_state(self):
        obs = self.get_observation()
        # Extract all joint positions in order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        state = np.array([
            obs['shoulder_pan.pos'],
            obs['shoulder_lift.pos'],
            obs['elbow_flex.pos'],
            obs['wrist_flex.pos'],
            obs['wrist_roll.pos'],
            obs['gripper.pos']
        ])

        print("state", state)
        return state

    def get_current_img(self):
        if not self.enable_camera:
            return None
            
        ret, img = self.main_camera.read()
        cv2.imwrite(f"eval_images/img_{image_count}.png", img)
        
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
            
        return img

    def set_target_state(self, target_state: torch.Tensor):
        print("setting target state", target_state)
        print("target_state type", type(target_state))
        new_state = torch.tensor(target_state)
        print("new_state type", type(new_state))

        # Add current action to buffer
        self.action_buffer.append(new_state.numpy())
        
        # If buffer is not full, use current action directly
        if len(self.action_buffer) < self.action_buffer.maxlen:
            smoothed_state = new_state
        else:
            # Calculate weighted moving average
            smoothed_state = np.average(self.action_buffer, weights=self.smoothing_weights, axis=0)
            smoothed_state = torch.tensor(smoothed_state)

        state_dict = {
            "shoulder_pan.pos": smoothed_state[0],
            "shoulder_lift.pos": smoothed_state[1],  
            "elbow_flex.pos": smoothed_state[2],  
            "wrist_flex.pos": smoothed_state[3],
            "wrist_roll.pos": smoothed_state[4],
            "gripper.pos": smoothed_state[5]
        }
        self.robot.send_action(state_dict)

    def disconnect(self):
        if self.enable_camera:
            self.main_camera.release()
        
        self.robot.disconnect()

        print("================> SO100 Robot disconnected")

    # def __del__(self):
    #     self.disconnect()


#################################################################################


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="192.168.0.145",
        port=5555,
        language_instruction="take eraser",
        img_width=224,
        img_height=224,
    ):
        self.language_instruction = language_instruction
        self.img_size = (img_height, img_width)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.action_queue = queue.Queue()
        self.inference_thread = None
        self.stop_thread = threading.Event()

    def _inference_worker(self, img, state):
        try:
            obs_dict = {
                "video.front": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[np.newaxis, :, :, :],
                "state.joints": state[np.newaxis, :].astype(np.float64),
                "annotation.task_index": [self.language_instruction],
            }
            res = self.policy.get_action(obs_dict)
            self.action_queue.put(res)
        except Exception as e:
            print(f"Error in inference thread: {e}")
            self.action_queue.put(None)

    def get_action(self, img, state):
        # Stop any existing thread
        if self.inference_thread and self.inference_thread.is_alive():
            self.stop_thread.set()
            self.inference_thread.join()
            self.stop_thread.clear()

        # Start new inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            args=(img, state)
        )
        self.inference_thread.start()
        
        # Wait for result
        action = self.action_queue.get()
        if action is None:
            raise RuntimeError("Action inference failed")
        return action

    def cleanup(self):
        if self.inference_thread and self.inference_thread.is_alive():
            self.stop_thread.set()
            self.inference_thread.join()

    def sample_action(self):
        obs_dict = {
            "video.front": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.joints": np.zeros((1, 5)),
            "annotation.task_index": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction


def convert_to_degrees(normalized_action):
    """
    Convert normalized actions to degrees.
    The normalized actions are typically in range [-1, 1] and need to be scaled to the robot's joint ranges.
    """
    # Joint limits in degrees for SO101 robot
    return np.degrees(normalized_action)


#################################################################################


def view_img(img):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    """
    plt.clf()  # Clear the current figure

    # Remove batch dimension if present
    if len(img.shape) == 4:
        img = img.squeeze(0)

    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    plt.imshow(img)
    plt.title('Main Camera')
    plt.axis("off")
    plt.pause(0.01)  # Non-blocking show


#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    first_time = True

    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="192.168.0.145")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--actions_to_execute", type=int, default=500)
    parser.add_argument("--cam_main_idx", type=int, default=6)
    parser.add_argument(
        "--lang_instruction", type=str, default="take eraser"
    )
    parser.add_argument("--record_imgs", action="store_true")
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--img_height", type=int, default=480)
    args = parser.parse_args()

    # Create figure for visualization
    plt.figure(figsize=(8, 4))
    
    # print lang_instruction
    print("lang_instruction: ", args.lang_instruction)

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["joints"]
    if USE_POLICY:
        client = Gr00tRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction=args.lang_instruction,
        )

        try:
            if args.record_imgs:
                os.makedirs("eval_images", exist_ok=True)
                for file in os.listdir("eval_images"):
                    os.remove(os.path.join("eval_images", file))
        
            robot = SO100Robot(img_width=args.img_width, img_height=args.img_height, calibrate=False, enable_camera=True, cam_main_idx=args.cam_main_idx)
            image_count = 0
            with robot.activate():
                for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                    if first_time:
                        for j in tqdm(range(50), desc="Initializing camera"):
                            img = robot.get_current_img()
                        first_time = False
                    img = robot.get_current_img()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # view_img(img)

                    state = robot.get_current_state()
                    
                    action = client.get_action(img, state)
                    
                    start_time = time.time()
                    for i in range(ACTION_HORIZON):
                        concat_action = np.concatenate(
                            [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                            axis=0,
                        )
                        assert concat_action.shape == (6,), concat_action.shape

                        robot.set_target_state(concat_action) 
                        time.sleep(0.02)
                        print("executing action", i, "time taken", time.time() - start_time)
                    print("Action chunk execution time taken", time.time() - start_time)
        finally:
            client.cleanup()
    else:
        # Test Dataset Source https://huggingface.co/datasets/youliangtan/so100_strawberry_grape
        dataset = LeRobotDataset(
            repo_id="",
            root=args.dataset_path,
        )

        robot = SO100Robot(calibrate=False, enable_camera=True, cam_main_idx=args.cam_main_idx)


        with robot.activate():
            print("Run replay of the dataset")
            actions = []
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):
                action = dataset[i]["action"]
                img = dataset[i]["observation.images.main"].data.numpy()
                # original shape (3, 480, 640) for image data
                realtime_img = robot.get_current_img()

                img = img.transpose(1, 2, 0)
                view_img(img)
                actions.append(action)
                robot.set_target_state(action)
                time.sleep(0.05)

            # plot the actions
            plt.plot(actions)
            plt.show()

            print("Done all actions")
            # robot.go_home()
            print("Done home")
