#!/bin/bash
# Gazebo（crane_plus_with_table.launch.py）を起動した後に実行する。
source /opt/ros/$(ls /opt/ros | head -n1)/setup.bash
ros2 run ros_gz_sim create -file /home/ubuntu/physai_141948/models/trash_bin/model.sdf -name trash_bin -x 0.11 -y 0.0 -z 1.015
ros2 run ros_gz_sim create -file /home/ubuntu/physai_141948/models/cube/red_cube/model.sdf -name red_cube -x 0.08 -y -0.12 -z 1.05
ros2 run ros_gz_sim create -file /home/ubuntu/physai_141948/models/cube/blue_cube/model.sdf -name blue_cube -x 0.10 -y 0.13 -z 1.05
ros2 run ros_gz_sim create -file /home/ubuntu/physai_141948/models/cube/green_cube/model.sdf -name green_cube -x 0.02 -y 0.12 -z 1.05
ros2 run ros_gz_sim create -file /home/ubuntu/physai_141948/models/overhead_camera/model.sdf -name overhead_camera -x 0.10 -y 0.0 -z 1.465 -P 1.5708
