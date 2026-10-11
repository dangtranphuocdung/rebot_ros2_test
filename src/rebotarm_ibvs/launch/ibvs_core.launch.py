import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 1. Load MoveIt Configs (mirroring demo.launch.py)
    moveit_config = (
        MoveItConfigsBuilder("rebotarm", package_name="rebotarm_moveit_config")
        .robot_description(file_path="config/rebotarm.urdf.xacro")
        .robot_description_semantic(file_path="config/rebotarm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # 2. Load Servo config
    servo_yaml = os.path.join(
        get_package_share_directory('rebotarm_ibvs'),
        'config',
        'servo.yaml'
    )
    with open(servo_yaml, 'r') as file:
        servo_params = yaml.safe_load(file)

    # 3. Servo Node
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        parameters=[
            {"moveit_servo": servo_params},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
        output='screen'
    )

    # 4. Hand-Eye Calibration TF Publisher
    # Values derived from config/calibration/hand_eye.npz
    hand_eye_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='hand_eye_tf_publisher',
        arguments=[
            '-0.12848879', '0.00707998', '0.09285028',  # XYZ
            '0.60757216', '-0.58596113', '0.40408812', '-0.35244632',  # Qx Qy Qz Qw
            'link6', 'camera_link'
        ]
    )

    # 5. IBVS Controller Node
    # Load parameters from ibvs_pipeline.yaml
    config_path = os.path.expanduser('~/rebotarm_ros2/config/ibvs_pipeline.yaml')
    controller_node = Node(
        package='rebotarm_ibvs',
        executable='ibvs_controller_node',
        name='ibvs_controller_node',
        parameters=[config_path],
        output='screen'
    )

    # 6. Include Vision Launch
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rebotarm_ibvs'), 'launch', 'vision.launch.py')
        )
    )

    return LaunchDescription([
        servo_node,
        hand_eye_tf_node,
        controller_node,
        vision_launch
    ])

