# python -m lerobot.record \
#     --dataset.fps=30 \
#     --dataset.num_image_writer_processes=4 \
#     --robot.type=so101_follower \
#     --robot.port=/dev/tty_follower_arm \
#     --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
#     --teleop.type=so101_leader \
#     --teleop.port=/dev/tty_leader_arm \
#     --dataset.single_task="pick" \
#     --dataset.repo_id="lerobot/example_dataset" \
#     --dataset.push_to_hub=false \
#     --dataset.root=/home/navaneet/lerobotvla/datasets/test







python -m lerobot.record \
    --dataset.fps=30 \
    --dataset.num_image_writer_processes=4 \
    --robot.type=so101_follower \
    --robot.port=/dev/tty_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty_leader_arm \
    --dataset.single_task="plce" \
    --dataset.repo_id="lerobot/example_dataset" \
    --dataset.push_to_hub=false \
    --dataset.root=/home/navaneet/lerobotvla/datasets/test
