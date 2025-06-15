# Motor Ports 


- Motor 1: Gripper : /dev/ttyACM0


follower port : 58FA082683


leader port : 58FA083023


right cam : 200901010001


left cam : 200901010001




SUBSYSTEM=="video4linux", ATTRS{serial}=="<serial number here>", ATTR{index}=="0", ATTRS{idProduct}=="085c", ATTR{device/latency_timer}="1", SYMLINK+="CAM_RIGHT_WRIST"


SUBSYSTEM=="tty", ATTRS{serial}=="<serial number here>", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_master_right"



udevadm info --name=/dev/ttyACM0 --attribute-walk | grep serial


sudo udevadm control --reload && sudo udevadm trigger



camera right = 2


camera left = 0

camera3 - real = 8


sudo chmod 666 /dev/tty_follower_arm /dev/tty_leader_arm


mint - yellow


yellow - mint

red - mint






