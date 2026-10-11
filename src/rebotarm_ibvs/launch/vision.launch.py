import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define the path to the standard ROS 2 config file
    config_path = os.path.expanduser('~/rebotarm_ros2/config/ibvs_pipeline.yaml')

    # 1. The Camera Node (v4l2_camera)
    camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='camera_node',
        # Do not use namespace='camera' if you want it to match the yaml 'camera_node' exactly,
        # or we could map the namespace inside the YAML. We'll drop namespace for simplicity.
        parameters=[config_path],
        remappings=[
            ('/image_raw', '/camera/image_raw') # Match our yolo subscriber
        ]
    )

    # 2. The YOLO Vision Node
    yolo_node = Node(
        package='rebotarm_ibvs',
        executable='yolo_vision_node',
        name='yolo_vision_node',
        parameters=[config_path],
        output='screen'
    )

    return LaunchDescription([
        camera_node,
        yolo_node
    ])
