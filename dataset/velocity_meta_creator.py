import os
import json
import glob
import cv2
import numpy as np
import pandas as pd

def calculate_stats(data):
    """Calculate statistics for a given data array."""
    if isinstance(data, (list, np.ndarray)):
        if len(data) == 0:
            return {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            }
        try:
            data_array = np.array(data)
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)
            
            # Handle boolean data
            if data_array.dtype == bool:
                data_array = data_array.astype(float)
            
            # Handle non-numeric data
            if not np.issubdtype(data_array.dtype, np.number):
                return {
                    "mean": [0.0],
                    "std": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "q01": [0.0],
                    "q99": [1.0]
                }
            
            # Calculate statistics
            stats = {
                "mean": np.mean(data_array, axis=0).tolist(),
                "std": np.std(data_array, axis=0).tolist(),
                "min": np.min(data_array, axis=0).tolist(),
                "max": np.max(data_array, axis=0).tolist(),
                "q01": np.percentile(data_array, 1, axis=0).tolist(),
                "q99": np.percentile(data_array, 99, axis=0).tolist()
            }
            
            # Replace any NaN or inf values with defaults
            for key in stats:
                stats[key] = [0.0 if not np.isfinite(x) else x for x in stats[key]]
            
            return stats
        except Exception as e:
            print(f"Error calculating stats: {e}")
            return {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            }
    else:
        return {
            "mean": [0.0],
            "std": [1.0],
            "min": [0.0],
            "max": [1.0],
            "q01": [0.0],
            "q99": [1.0]
        }

def create_stats_file(dataset_dir):
    """Create stats.json file with statistics for all features."""
    # Initialize empty stats dictionary
    stats = {}
    
    # Load all parquet files
    parquet_files = glob.glob(os.path.join(dataset_dir, 'data/chunk-000/*.parquet'))
    if not parquet_files:
        print("No parquet files found. Creating empty stats.json")
        stats = {
            "observation.state": {
                "mean": [0.0] * 3,
                "std": [1.0] * 3,
                "min": [-1.0] * 3,
                "max": [1.0] * 3,
                "q01": [-1.0] * 3,
                "q99": [1.0] * 3
            },
            "action": {
                "mean": [0.0] * 3,
                "std": [1.0] * 3,
                "min": [-1.0] * 3,
                "max": [1.0] * 3,
                "q01": [-1.0] * 3,
                "q99": [1.0] * 3
            },
            "timestamp": {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            },
            "annotation.human.action.task_description": {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            },
            "task_index": {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            },
            "annotation.human.validity": {
                "mean": [1.0],
                "std": [0.0],
                "min": [1.0],
                "max": [1.0],
                "q01": [1.0],
                "q99": [1.0]
            },
            "episode_index": {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            },
            "index": {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            },
            "next.reward": {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            },
            "next.done": {
                "mean": [0.0],
                "std": [1.0],
                "min": [0.0],
                "max": [1.0],
                "q01": [0.0],
                "q99": [1.0]
            }
        }
    else:
        try:
            # Read all parquet files
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
            
            for column in df.columns:
                try:
                    # Convert boolean columns to float
                    if df[column].dtype == bool:
                        df[column] = df[column].astype(float)

                    col_data = df[column].values
                    # If column is object, try to stack into a 2D array
                    if df[column].dtype == object:
                        try:
                            stacked = np.stack(col_data)
                            stats[column] = calculate_stats(stacked)
                            continue
                        except Exception as e:
                            print(f"Could not stack column {column}: {e}")
                            print(f"Skipping non-numeric column: {column}")
                            continue

                    if not np.issubdtype(df[column].dtype, np.number):
                        print(f"Skipping non-numeric column: {column}")
                        continue

                    stats[column] = calculate_stats(col_data)
                except Exception as e:
                    print(f"Error processing column {column}: {e}")
                    stats[column] = {
                        "mean": [0.0],
                        "std": [1.0],
                        "min": [0.0],
                        "max": [1.0],
                        "q01": [0.0],
                        "q99": [1.0]
                    }

        except Exception as e:
            print(f"Error reading parquet files: {e}")
            # Create default stats if there's an error
            num_features = 3
            stats = {
                "observation.state": {
                    "mean": [0.0] * num_features,
                    "std": [1.0] * num_features,
                    "min": [-1.0] * num_features,
                    "max": [1.0] * num_features,
                    "q01": [-1.0] * num_features,
                    "q99": [1.0] * num_features
                },
                "action": {
                    "mean": [0.0] * num_features,
                    "std": [1.0] * num_features,
                    "min": [-1.0] * num_features,
                    "max": [1.0] * num_features,
                    "q01": [-1.0] * num_features,
                    "q99": [1.0] * num_features
                },
                "timestamp": {
                    "mean": [0.0],
                    "std": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "q01": [0.0],
                    "q99": [1.0]
                },
                "annotation.human.action.task_description": {
                    "mean": [0.0],
                    "std": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "q01": [0.0],
                    "q99": [1.0]
                },
                "task_index": {
                    "mean": [0.0],
                    "std": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "q01": [0.0],
                    "q99": [1.0]
                },
                "annotation.human.validity": {
                    "mean": [1.0],
                    "std": [0.0],
                    "min": [1.0],
                    "max": [1.0],
                    "q01": [1.0],
                    "q99": [1.0]
                },
                "episode_index": {
                    "mean": [0.0],
                    "std": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "q01": [0.0],
                    "q99": [1.0]
                },
                "index": {
                    "mean": [0.0],
                    "std": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "q01": [0.0],
                    "q99": [1.0]
                },
                "next.reward": {
                    "mean": [0.0],
                    "std": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "q01": [0.0],
                    "q99": [1.0]
                },
                "next.done": {
                    "mean": [0.0],
                    "std": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "q01": [0.0],
                    "q99": [1.0]
                }
            }
    
    # Save stats to file
    with open(os.path.join(dataset_dir, 'meta/stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

def create_episodes_file(dataset_dir):
    """Create episodes.jsonl file with episode information."""
    episodes_data = []
    
    # Load all parquet files
    parquet_files = glob.glob(os.path.join(dataset_dir, 'data/chunk-000/*.parquet'))
    
    if parquet_files:
        try:
            # Read all parquet files
            dfs = [pd.read_parquet(f) for f in parquet_files]
            
            # Process each episode
            for df in dfs:
                if 'episode_index' in df.columns and 'task_index' in df.columns:
                    episode_index = df['episode_index'].iloc[0]
                    task_index = df['task_index'].iloc[0]
                    
                    # Map task_index to task description
                    if task_index == 0:
                        tasks = ['go to blue cube', 'valid']
                    elif task_index == 2:
                        tasks = ['go to red cube', 'valid']
                    elif task_index == 3:
                        tasks = ['go to chair', 'valid']
                    elif task_index == 4:
                        tasks = ['go to drawer', 'valid']
                    elif task_index == 5:
                        tasks = ['go to globe bar', 'valid']
                    elif task_index == 6:
                        tasks = ['go to sofa', 'valid']
                    else:
                        raise ValueError(f"Invalid task index: {task_index}")
                    
                    episode_data = {
                        "episode_index": int(episode_index),
                        "tasks": tasks,
                        "length": len(df)
                    }
                    episodes_data.append(episode_data)
            
            # Sort episodes by episode_index
            episodes_data.sort(key=lambda x: x['episode_index'])
            
            # Write to episodes.jsonl
            with open(os.path.join(dataset_dir, 'meta/episodes.jsonl'), 'w') as f:
                for episode in episodes_data:
                    f.write(json.dumps(episode) + '\n')
                    
        except Exception as e:
            print(f"Error creating episodes.jsonl: {e}")
            # Create empty file if there's an error
            open(os.path.join(dataset_dir, 'meta/episodes.jsonl'), 'w').close()

def create_meta_files(dataset_dir):
    # Create necessary directories
    os.makedirs(os.path.join( dataset_dir, 'meta'), exist_ok=True)
    
    # Create tasks.jsonl
    tasks = [
        {"task_index": 0, "task": "go to blue cube"},
        {"task_index": 1, "task": "valid"},
        {"task_index": 2, "task": "go to red cube"},
        {"task_index": 3, "task": "go to chair"},
        {"task_index": 4, "task": "go to drawer"},
        {"task_index": 5, "task": "go to globe bar"},
        {"task_index": 6, "task": "go to sofa"}
    ]


    with open(os.path.join(dataset_dir, 'meta/tasks.jsonl'), 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
    
    # Create modality.json
    modality_data = {
        "state": {
            "velocities": {
                "start": 0,
                "end": 3
            }
        },
        "action": {
            "velocities": {
                "start": 0,
                "end": 3
            }
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view"
            }
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {}
        }
    }
    with open(os.path.join(dataset_dir, 'meta/modality.json'), 'w') as f:
        json.dump(modality_data, f, indent=2)
    
    # Create info.json
    episodes = glob.glob(os.path.join(dataset_dir, 'videos/chunk-000/observation.images.ego_view/episode_*.mp4'))
    num_episodes = len(episodes)
    num_frames = 0
    for ep in episodes:
        cap = cv2.VideoCapture(ep)
        num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    info_data = {
        "codebase_version": "v1.0",
        "robot_type": "Go2",
        "total_episodes": num_episodes,
        "total_frames": num_frames,
        "total_tasks": 4,
        "total_videos": num_episodes,
        "total_chunks": 1,
        "chunks_size": num_episodes,
        "fps": 20.0,
        "splits": {
            "train": "0:100"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float64",
                "shape": [
                    3
                ],
                "names": [
                    "vel_x",
                    "vel_y",
                    "vel_z"
                ]
            },
            "action": {
                "dtype": "float64",
                "shape": [
                    3
                ],
                "names": [
                    "vel_x",
                    "vel_y",
                    "vel_z"
                ]
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [
                    1
                ]
            },
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [
                    1
                ]
            },
            "task_index": {
                "dtype": "int64",
                "shape": [
                    1
                ]
            },
            "annotation.human.validity": {
                "dtype": "int64",
                "shape": [
                    1
                ]
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [
                    1
                ]
            },
            "index": {
                "dtype": "int64",
                "shape": [
                    1
                ]
            },
            "next.reward": {
                "dtype": "float64",
                "shape": [
                    1
                ]
            },
            "next.done": {
                "dtype": "bool",
                "shape": [
                    1
                ]
            },
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [
                    480,
                    640,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "channel"
                ],
            "video_info": {
                "video.fps": 20.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                }
            }
        }
    }
    with open(os.path.join(dataset_dir, 'meta/info.json'), 'w') as f:
        json.dump(info_data, f, indent=2)
    
    # Create stats.json
    create_stats_file(dataset_dir)
    
    # Create episodes.jsonl
    create_episodes_file(dataset_dir) 