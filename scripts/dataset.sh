# /home/navaneet/miniconda3/envs/lerobot/bin/python -m lerobot.record \
#     --dataset.fps=30 \
#     --dataset.num_image_writer_processes=4 \
#     --robot.type=so101_follower \
#     --robot.port=/dev/tty_follower_arm \
#     --robot.cameras="{left_cam: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, right_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, down_cam: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}"  \
#     --teleop.type=so101_leader \
#     --teleop.port=/dev/tty_leader_arm \
#     --dataset.single_task="place" \
#     --dataset.repo_id="lerobot/example_dataset" \
#     --dataset.push_to_hub=false \
#     --dataset.episode_time_s=10 \
#     --dataset.root=/home/navaneet/lerobotvla/datasets/test2 \
    


python -m lerobot.record \
    --dataset.fps=30 \
    --dataset.num_image_writer_processes=4 \
    --robot.type=so101_follower \
    --robot.port=/dev/tty_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty_leader_arm \
    --dataset.push_to_hub=false \
    --dataset.repo_id="lerobot/example_dataset" \
    --dataset.single_task="place" \
    --dataset.root=/home/navaneet/lerobotvla/datasets/traj