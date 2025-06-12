import time
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame

if __name__ == "__main__":
    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument("--record_imgs", action="store_true")
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute

    # Test Dataset Source https://huggingface.co/datasets/youliangtan/so100_strawberry_grape
    dataset = LeRobotDataset(
        repo_id="",
        root=args.dataset_path,
    )

    if args.record_imgs:
        # create a folder to save the images and delete all the images in the folder
        os.makedirs("eval_images", exist_ok=True)
        for file in os.listdir("eval_images"):
            os.remove(os.path.join("eval_images", file))

    print("Running dataset visualization")
    actions = []
    image_count = 0
    
    for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Processing dataset"):
        action = dataset[i]["action"]
        img = dataset[i]["observation.images.main"].data.numpy()
        img2 = dataset[i]["observation.images.secondary_0"].data.numpy()
       
        # Convert images from (3, 480, 640) to (480, 640, 3)
        img = img.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)
        view_img(img, img2)
        
        if args.record_imgs:
            # resize the images to 320x240
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (640, 480))
            img2_resized = cv2.resize(img2_bgr, (640, 480))
            cv2.imwrite(f"eval_images/img_{image_count}.jpg", img_resized)
            cv2.imwrite(f"eval_images/img2_{image_count}.jpg", img2_resized)
            image_count += 1
            
        actions.append(action)
        time.sleep(0.05)  # Small delay to visualize

    # Plot the actions
    plt.figure(figsize=(12, 6))
    actions_array = np.array(actions)
    for i in range(actions_array.shape[1]):
        plt.plot(actions_array[:, i], label=f'Joint {i+1}')
    plt.legend()
    plt.title('Joint Trajectories from Dataset')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Position')
    plt.show()

    print("Done processing dataset") 