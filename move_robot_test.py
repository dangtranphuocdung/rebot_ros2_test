import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class Mover(Node):
    def __init__(self):
        super().__init__('mover')
        self.pub = self.create_publisher(JointTrajectory, '/rebotarm_controller/joint_trajectory', 10)
        self.timer = self.create_timer(1.0, self.move)
    
    def move(self):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        point = JointTrajectoryPoint()
        point.positions = [0.0, -0.5, 0.5, 0.0, -1.0, 0.0]
        point.time_from_start.sec = 2
        msg.points.append(point)
        self.pub.publish(msg)
        self.get_logger().info('Published trajectory!')

rclpy.init()
Mover()
rclpy.spin_once(Mover(), timeout_sec=2.0)
