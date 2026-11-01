# physai_141948

# 使っているもの
- ROS2
- Gazebo
- CRANE+ V2
- YOLO-World

# 準備
- CLIPを`src/CLIP`にcloneしておく
- `python3 models/generate_meshes.py`でSTLを作成しておく

# 実行
## gazebo
```console
ros2 launch crane_plus_gazebo crane_plus_with_table.launch.py
```

## set initial positions
```console
./spawn_blocks.sh
```

## camera
```console
ros2 run ros_gz_bridge parameter_bridge /overhead_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image
```

## run
```console
python3 drop_to_bin.py red blue green
```