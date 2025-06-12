import pandas as pd
import cv2
import os
from tqdm import tqdm
import numpy as np
import shutil
from tkinter import messagebox
import time
import sys


from dataset.velocity_meta_creator import create_meta_files

def generate_gr00t_structure(PROJECT_ROOT):
    # ------------- CONFIGURATION -------------
    PROJECT_ROOT = os.path.join('dataset', 'tasks' , PROJECT_ROOT)
    ACTION_LOGS_DIR = os.path.join(PROJECT_ROOT, "raw_data")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data/chunk-000")
    VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos/chunk-000/observation.images.ego_view")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "meta")
    FPS = 20

    # Delete existing directories if they exist
    for dir_path in [DATA_DIR, VIDEOS_DIR, METADATA_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted existing directory: {dir_path}")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    # ------------- PROCESS EACH EPISODE -------------
    episode_folders = sorted(os.listdir(ACTION_LOGS_DIR))
    episode_index = 0

    for episode_folder in episode_folders:
        time.sleep(1)
        episode_path = os.path.join(ACTION_LOGS_DIR, episode_folder)
        if not os.path.isdir(episode_path):
            continue  # skip files

        # Find the .parquet file
        parquet_files = [f for f in os.listdir(episode_path) if f.endswith('.parquet')]
        if not parquet_files:
            print(f"No parquet file found in {episode_path}")
            continue
        parquet_path = os.path.join(episode_path, parquet_files[0])

        # Images directory
        images_dir = os.path.join(episode_path, "images")

        # Output paths
        output_parquet = os.path.join(DATA_DIR, f"{episode_folder}.parquet")
        output_video = os.path.join(VIDEOS_DIR, f"{episode_folder}.mp4")

        print(f"\nProcessing {episode_folder}...")

        try:
            # Read parquet
            df = pd.read_parquet(parquet_path)
            df = df.iloc[15:]

            new_records = []
            frame_paths = []

            for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
                action = row['actions']
                velocities = row['velocity_commands']
                task_index = row['task_index']
                image_rel_path = row['camera_img_path']
                image_path = os.path.join(images_dir, os.path.basename(image_rel_path))
                is_last = (i == len(df) - 1)

                timestamp = float(np.float32(i) / np.float32(FPS))

                record = {
                    "observation.state": velocities,
                    "action": velocities,
                    "timestamp": timestamp,
                    "annotation.human.action.task_description": 0,
                    "task_index": task_index,
                    "annotation.human.validity": 1,
                    "episode_index": episode_index,
                    "index": i,
                    "next.reward": 1 if is_last else 0,
                    "next.done": is_last
                }
                new_records.append(record)
                frame_paths.append(image_path)

            # Save new parquet
            new_df = pd.DataFrame(new_records)
            new_df.to_parquet(output_parquet)
            print(f"Saved new dataset: {output_parquet}")

            # Create video
            first_img = cv2.imread(frame_paths[0])
            if first_img is None:
                print(f"First image not found: {frame_paths[0]}")
                continue
            height, width, layers = first_img.shape
            size = (width, height)
            out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), FPS, size)

            for frame_path in tqdm(frame_paths, desc="Writing video frames"):
                img = cv2.imread(frame_path)
                if img is not None:
                    out.write(img)
                else:
                    print(f"Warning: Missing image {frame_path}")
            out.release()
            print(f"Saved video: {output_video}")

            episode_index += 1

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate structure:\n{e}")
            print(f"Error processing {episode_folder}: {e}")
            # Remove the problematic episode directory
            # shutil.rmtree(episode_path)
            # print(f"Removed episode directory: {episode_path}")

    # # Update the order of the episodes in the raw data directory
    # episode_folders = sorted(os.listdir(ACTION_LOGS_DIR))
    # for index, episode_folder in enumerate(tqdm(episode_folders, desc="Updating episode indices")):
    #     episode_path = os.path.join(ACTION_LOGS_DIR, episode_folder)
    #     if not os.path.isdir(episode_path):
    #         continue

    #     parquet_files = [f for f in os.listdir(episode_path) if f.endswith('.parquet')]
    #     if not parquet_files:
    #         continue
    #     parquet_path = os.path.join(episode_path, parquet_files[0])

    #     try:
    #         df = pd.read_parquet(parquet_path)
    #         df['episode_index'] = index
    #         df.to_parquet(parquet_path)
    #         # print(f"Updated episode_index for {episode_folder}")

    #         # Rename the episode folder to reflect the new order
    #         new_episode_folder = f"episode_{index}"
    #         new_episode_path = os.path.join(ACTION_LOGS_DIR, new_episode_folder)
    #         if episode_folder != new_episode_folder:
    #             shutil.move(episode_path, new_episode_path)
    #             # print(f"Renamed {episode_folder} to {new_episode_folder}")

    #     except Exception as e:
    #         print(f"Error updating episode_index for {episode_folder}: {e}")

    # Create meta files
    create_meta_files(PROJECT_ROOT)
