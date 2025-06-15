python -m lerobot.replay \
  --robot.type=so101_follower \
  --robot.port=/dev/tty_follower_arm \
  --dataset.root=/home/navaneet/lerobotvla/datasets/traj  \
  --dataset.repo_id=lerobot/local_dataset \
  --dataset.episode=0 \
  --dataset.fps=30
