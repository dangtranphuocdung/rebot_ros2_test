#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import TwistStamped
import time

class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__('ibvs_controller_node')
        
        # Parameters
        self.declare_parameter('target_pixel_x', 320)
        self.declare_parameter('target_pixel_y', 240)
        self.declare_parameter('target_area_px', 40000)
        
        self.declare_parameter('gain_x', 0.001)
        self.declare_parameter('gain_y', 0.001)
        self.declare_parameter('gain_z', 0.00001)
        self.declare_parameter('max_translation_speed_m_s', 0.05)
        
        self.declare_parameter('camera_frame_id', 'camera_link')
        
        # Extract values
        self.target_pixel_x = self.get_parameter('target_pixel_x').value
        self.target_pixel_y = self.get_parameter('target_pixel_y').value
        self.target_area_px = self.get_parameter('target_area_px').value
        
        self.gain_x = self.get_parameter('gain_x').value
        self.gain_y = self.get_parameter('gain_y').value
        self.gain_z = self.get_parameter('gain_z').value
        self.max_velocity = self.get_parameter('max_translation_speed_m_s').value
        self.frame_id = self.get_parameter('camera_frame_id').value
        
        # State
        self.is_active = False
        self.current_vel_x = 0.0
        self.current_vel_y = 0.0
        self.current_vel_z = 0.0
        self.last_detection_time = 0.0
        
        # Subscribers
        self.create_subscription(Float32MultiArray, '/yolo/target_bounding_box', self.bbox_callback, 10)
        self.create_subscription(Bool, '/ibvs/enable', self.enable_callback, 10)
        
        # Publishers
        self.twist_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        # 50 Hz control loop to keep MoveIt Servo continuously fed
        self.timer = self.create_timer(0.02, self.control_loop_publish)
        
        self.get_logger().info(f"IBVS Controller initialized. Frame: {self.frame_id}. Max speed: {self.max_velocity} m/s")

    def enable_callback(self, msg):
        self.is_active = msg.data
        if self.is_active:
            self.get_logger().info("IBVS Control ENABLED")
        else:
            self.get_logger().info("IBVS Control DISABLED")
            self.current_vel_x = 0.0
            self.current_vel_y = 0.0
            self.current_vel_z = 0.0
            self.publish_zero_twist()

    def bbox_callback(self, msg):
        if not self.is_active:
            return
            
        if len(msg.data) < 4:
            return
            
        cx, cy, width, height = msg.data
        current_area = width * height
        
        error_x = self.target_pixel_x - cx
        error_y = self.target_pixel_y - cy
        error_area = self.target_area_px - current_area
        
        # Compute velocities
        vel_y = error_x * self.gain_x  # image X error -> move camera Y
        vel_z = error_y * self.gain_y  # image Y error -> move camera Z
        vel_x = error_area * self.gain_z # image area error -> move camera X
        
        self.current_vel_x = self.clamp(vel_x, -self.max_velocity, self.max_velocity)
        self.current_vel_y = self.clamp(vel_y, -self.max_velocity, self.max_velocity)
        self.current_vel_z = self.clamp(vel_z, -self.max_velocity, self.max_velocity)
        self.last_detection_time = time.time()

    def control_loop_publish(self):
        if not self.is_active:
            return
            
        now = time.time()
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = self.frame_id
        
        if now - self.last_detection_time < 0.5:
            twist_msg.twist.linear.x = float(self.current_vel_x)
            twist_msg.twist.linear.y = float(self.current_vel_y)
            twist_msg.twist.linear.z = float(self.current_vel_z)
            self.get_logger().info(
                f"[Publishing Twist]: Vx={self.current_vel_x:.3f}, Vy={self.current_vel_y:.3f}, Vz={self.current_vel_z:.3f} ({self.frame_id})",
                throttle_duration_sec=1.0
            )
        else:
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.linear.z = 0.0
            
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = 0.0
        self.twist_pub.publish(twist_msg)

    def publish_zero_twist(self):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = self.frame_id
        self.twist_pub.publish(twist_msg)
        
    def clamp(self, value, min_val, max_val):
        return max(min_val, min(value, max_val))

def main(args=None):
    rclpy.init(args=args)
    node = IBVSControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
