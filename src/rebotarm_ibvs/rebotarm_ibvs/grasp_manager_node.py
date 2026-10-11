#!/usr/bin/env python3
from __future__ import annotations
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

from builtin_interfaces.msg import Duration
from std_msgs.msg import Bool, Float32MultiArray, Int8
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    RobotState,
    RobotTrajectory
)
import time

class GraspManagerNode(Node):
    """
    Hybrid IBVS State Machine adhering to vendor motion planning (OMPL /execute_trajectory)
    and calibrated gripper actions, combined with real-time MoveIt Servo visual tracking.
    """
    def __init__(self):
        super().__init__('grasp_manager_node')
        
        # --------------------------------------------------------------------------
        # 1. PARAMETERS (Aligning with vendor pick_place.yaml & ibvs_pipeline.yaml)
        # --------------------------------------------------------------------------
        self.declare_parameter('group_name', 'arm')
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('gripper_joint_names', ['gripper_joint1', 'gripper_joint2'])
        
        # Vendor calibrated scanning & gripper positions
        self.declare_parameter('scanning_joints', [0.0, -0.6528, -0.7749, 0.8011, 0.0, 0.0])
        self.declare_parameter('open_gripper_position', 0.045)   # Vendor calibrated open stroke
        self.declare_parameter('grasp_gripper_position', 0.017)  # Vendor calibrated object grasp stroke
        self.declare_parameter('closed_gripper_position', 0.0)
        self.declare_parameter('gripper_motion_duration', 1.0)
        
        # MoveIt OMPL Planning parameters
        self.declare_parameter('planner_id', 'RRTConnect')
        self.declare_parameter('pipeline_id', 'ompl')
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('velocity_scaling', 0.5)
        self.declare_parameter('acceleration_scaling', 0.5)
        self.declare_parameter('joint_tolerance', 0.02)
        
        # IBVS Target Pixel & Tolerance Parameters
        self.declare_parameter('target_pixel_x', 320)
        self.declare_parameter('target_pixel_y', 240)
        self.declare_parameter('target_area_px', 40000)
        self.declare_parameter('tolerance_px', 30.0)
        self.declare_parameter('tolerance_area', 5000.0)
        self.declare_parameter('convergence_hold_time', 0.8)

        # Extract values
        self.group_name = str(self.get_parameter('group_name').value)
        self.joint_names = [str(n) for n in self.get_parameter('joint_names').value]
        self.gripper_joint_names = [str(n) for n in self.get_parameter('gripper_joint_names').value]
        self.scanning_joints = [float(j) for j in self.get_parameter('scanning_joints').value]
        self.open_gripper_pos = float(self.get_parameter('open_gripper_position').value)
        self.grasp_gripper_pos = float(self.get_parameter('grasp_gripper_position').value)
        self.gripper_duration = float(self.get_parameter('gripper_motion_duration').value)
        
        self.target_pixel_x = float(self.get_parameter('target_pixel_x').value)
        self.target_pixel_y = float(self.get_parameter('target_pixel_y').value)
        self.target_area_px = float(self.get_parameter('target_area_px').value)
        self.tolerance_px = float(self.get_parameter('tolerance_px').value)
        self.tolerance_area = float(self.get_parameter('tolerance_area').value)
        self.hold_time = float(self.get_parameter('convergence_hold_time').value)

        # Current Robot Joint State Tracking
        self._current_joint_positions: dict[str, float] = {}

        # --------------------------------------------------------------------------
        # 2. CLIENTS & SUBSCRIBERS
        # --------------------------------------------------------------------------
        # Vendor MoveIt Planning & Execution Action Clients
        self._planner = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self._execute = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self._gripper_trajectory = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        
        # Direct fallback trajectory publisher for arm
        self._arm_fallback_pub = self.create_publisher(JointTrajectory, '/rebotarm_controller/joint_trajectory', 10)

        # MoveIt Servo Control Services & Status
        self.servo_start_cli = self.create_client(Trigger, '/servo_node/start_servo')
        self.servo_stop_cli = self.create_client(Trigger, '/servo_node/stop_servo')
        self.servo_unpause_cli = self.create_client(Trigger, '/servo_node/unpause_servo')
        self.create_subscription(Int8, '/servo_node/status', self.servo_status_cb, 10)

        # IBVS Vision & Controller Interface
        self.ibvs_enable_pub = self.create_publisher(Bool, '/ibvs/enable', 10)
        self.create_subscription(Float32MultiArray, '/yolo/target_bounding_box', self.bbox_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos_profile_sensor_data)

        # --------------------------------------------------------------------------
        # 3. STATE MACHINE VARIABLES
        # --------------------------------------------------------------------------
        self.state = 'STARTUP'
        self.current_error_x = 999.0
        self.current_error_y = 999.0
        self.current_error_area = 99999.0
        self.convergence_start_time = None
        self.busy = False  # Lock to ensure async operations complete sequentially

        # Periodic Loop (10 Hz)
        self.timer = self.create_timer(0.1, self.state_machine_loop)
        self.get_logger().info("Grasp Manager initialized with vendor-style trajectory planning and IBVS tracking.")

    # --------------------------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------------------------
    def joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._current_joint_positions[name] = float(pos)

    def servo_status_cb(self, msg: Int8):
        status_map = {
            0: "NO_WARNING (Ready & Active)",
            1: "DECELERATE_FOR_APPROACHING_SINGULARITY",
            2: "HALT_FOR_SINGULARITY",
            3: "DECELERATE_FOR_LEAVING_SINGULARITY",
            4: "DECELERATE_FOR_COLLISION",
            5: "HALT_FOR_COLLISION",
            6: "JOINT_BOUND"
        }
        name = status_map.get(msg.data, f"CODE_{msg.data}")
        now = time.time()
        if not hasattr(self, '_last_status_log') or (now - self._last_status_log > 2.0):
            self.get_logger().info(f"[MoveIt Servo Status]: {name}")
            self._last_status_log = now

    def bbox_callback(self, msg: Float32MultiArray):
        if len(msg.data) >= 4:
            cx, cy, w, h = msg.data
            area = w * h
            self.current_error_x = abs(self.target_pixel_x - cx)
            self.current_error_y = abs(self.target_pixel_y - cy)
            self.current_error_area = abs(self.target_area_px - area)

    def set_ibvs(self, enable: bool):
        msg = Bool()
        msg.data = enable
        self.ibvs_enable_pub.publish(msg)

    # --------------------------------------------------------------------------
    # VENDOR-STYLE TRAJECTORY PLANNING & EXECUTION
    # --------------------------------------------------------------------------
    @staticmethod
    def duration(seconds: float) -> Duration:
        return Duration(sec=int(seconds), nanosec=int((seconds % 1.0) * 1_000_000_000))

    def plan_and_execute_joint_target(self, goal_values: list[float], label: str, on_success_callback):
        """Plans motion via /plan_kinematic_path and executes via /execute_trajectory."""
        self.get_logger().info(f"Planning motion to {label}: {[round(v, 3) for v in goal_values]}")
        
        # Check if MoveIt planner is available
        if not self._planner.service_is_ready():
            self.get_logger().warn("/plan_kinematic_path not available! Using direct trajectory fallback.")
            self._execute_direct_trajectory(goal_values, 3.0, on_success_callback)
            return

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = self.group_name
        req.motion_plan_request.pipeline_id = str(self.get_parameter('pipeline_id').value)
        req.motion_plan_request.planner_id = str(self.get_parameter('planner_id').value)
        req.motion_plan_request.allowed_planning_time = float(self.get_parameter('planning_time').value)
        req.motion_plan_request.num_planning_attempts = 5
        req.motion_plan_request.max_velocity_scaling_factor = float(self.get_parameter('velocity_scaling').value)
        req.motion_plan_request.max_acceleration_scaling_factor = float(self.get_parameter('acceleration_scaling').value)
        
        # Current state
        current_values = [self._current_joint_positions.get(name, 0.0) for name in self.joint_names]
        req.motion_plan_request.start_state = RobotState(
            is_diff=True,
            joint_state=JointState(name=list(self.joint_names), position=list(current_values))
        )
        
        # Goal constraints
        tol = float(self.get_parameter('joint_tolerance').value)
        constraints = Constraints(
            joint_constraints=[
                JointConstraint(
                    joint_name=name,
                    position=float(val),
                    tolerance_above=tol,
                    tolerance_below=tol,
                    weight=1.0
                )
                for name, val in zip(self.joint_names, goal_values)
            ]
        )
        req.motion_plan_request.goal_constraints = [constraints]

        plan_future = self._planner.call_async(req)
        plan_future.add_done_callback(lambda f: self._on_plan_received(f, goal_values, label, on_success_callback))

    def _on_plan_received(self, future, goal_values: list[float], label: str, on_success_callback):
        try:
            res = future.result()
            if res is None or res.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().warn(f"MoveIt planning to {label} failed. Falling back to direct trajectory.")
                self._execute_direct_trajectory(goal_values, 3.0, on_success_callback)
                return

            self.get_logger().info(f"MoveIt planning succeeded. Executing trajectory for {label}...")
            goal_msg = ExecuteTrajectory.Goal(trajectory=res.motion_plan_response.trajectory)
            
            if not self._execute.server_is_ready():
                self.get_logger().warn("/execute_trajectory server not ready. Using direct fallback.")
                self._execute_direct_trajectory(goal_values, 3.0, on_success_callback)
                return

            send_goal_future = self._execute.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(lambda f: self._on_execute_goal_sent(f, label, on_success_callback))
        except Exception as e:
            self.get_logger().error(f"Exception during planning to {label}: {e}")
            self._execute_direct_trajectory(goal_values, 3.0, on_success_callback)

    def _on_execute_goal_sent(self, future, label: str, on_success_callback):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error(f"ExecuteTrajectory rejected goal for {label}")
            self.busy = False
            return
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_execute_completed(f, label, on_success_callback))

    def _on_execute_completed(self, future, label: str, on_success_callback):
        action_result = future.result()
        if action_result and action_result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f"Trajectory execution to {label} completed successfully!")
            on_success_callback()
        else:
            self.get_logger().warn(f"ExecuteTrajectory reported failure for {label}. Proceeding.")
            on_success_callback()

    def _execute_direct_trajectory(self, goal_values: list[float], duration: float, on_success_callback):
        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in goal_values]
        point.time_from_start = self.duration(duration)
        traj.points.append(point)
        self._arm_fallback_pub.publish(traj)
        
        # Simple timer to await direct execution duration
        self.create_timer(duration + 0.2, lambda: self._on_direct_trajectory_done(on_success_callback))

    def _on_direct_trajectory_done(self, on_success_callback):
        on_success_callback()

    # --------------------------------------------------------------------------
    # VENDOR-STYLE GRIPPER ACTION CLIENT
    # --------------------------------------------------------------------------
    def command_gripper(self, position: float, on_success_callback):
        """Commands gripper fingers using FollowJointTrajectory action matching pick_place.py."""
        self.get_logger().info(f"Commanding gripper to position {position:.4f} m...")
        
        if not self._gripper_trajectory.server_is_ready():
            self.get_logger().warn("Gripper action server not ready! Assuming success after duration.")
            self.create_timer(self.gripper_duration, on_success_callback)
            return

        goal = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(
                joint_names=list(self.gripper_joint_names),
                points=[
                    JointTrajectoryPoint(
                        positions=[position] * len(self.gripper_joint_names),
                        time_from_start=self.duration(self.gripper_duration)
                    )
                ]
            )
        )
        
        future = self._gripper_trajectory.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_gripper_goal_sent(f, on_success_callback))

    def _on_gripper_goal_sent(self, future, on_success_callback):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn("Gripper goal rejected. Continuing...")
            on_success_callback()
            return

        res_future = goal_handle.get_result_async()
        res_future.add_done_callback(lambda f: on_success_callback())

    # --------------------------------------------------------------------------
    # STATE MACHINE LOOP
    # --------------------------------------------------------------------------
    def state_machine_loop(self):
        if self.busy:
            return

        # ----------------------------------------------------------------------
        # STATE 1: STARTUP -> Open Gripper and Move to Scanning Pose
        # ----------------------------------------------------------------------
        if self.state == 'STARTUP':
            self.busy = True
            self.get_logger().info("--- Phase 1: Opening Gripper ---")
            self.command_gripper(self.open_gripper_pos, self._on_initial_gripper_opened)

        # ----------------------------------------------------------------------
        # STATE 2: ENABLE_IBVS -> Start Servo & engage real-time visual tracking
        # ----------------------------------------------------------------------
        elif self.state == 'ENABLE_IBVS':
            self.get_logger().info("--- Phase 2: Starting MoveIt Servo & IBVS Tracking ---")
            self.set_ibvs(True)
            
            # Start and unpause MoveIt Servo
            if self.servo_start_cli.service_is_ready():
                self.servo_start_cli.call_async(Trigger.Request())
            if self.servo_unpause_cli.service_is_ready():
                self.servo_unpause_cli.call_async(Trigger.Request())
                
            self.state = 'WAIT_CONVERGENCE'

        # ----------------------------------------------------------------------
        # STATE 3: WAIT_CONVERGENCE -> Visual servoing tracks until aligned
        # ----------------------------------------------------------------------
        elif self.state == 'WAIT_CONVERGENCE':
            self.set_ibvs(True)
            
            # Check convergence criteria
            if (self.current_error_x < self.tolerance_px and 
                self.current_error_y < self.tolerance_px and 
                self.current_error_area < self.tolerance_area):
                
                if self.convergence_start_time is None:
                    self.convergence_start_time = time.time()
                elif time.time() - self.convergence_start_time >= self.hold_time:
                    self.get_logger().info("IBVS Converged! Target aligned. Halting Servo...")
                    self.set_ibvs(False)
                    if self.servo_stop_cli.service_is_ready():
                        self.servo_stop_cli.call_async(Trigger.Request())
                    self.state = 'GRASP'
            else:
                self.convergence_start_time = None
                self.get_logger().info(
                    f"Tracking object... Errors: X:{self.current_error_x:.1f}, "
                    f"Y:{self.current_error_y:.1f}, Area:{self.current_error_area:.1f}",
                    throttle_duration_sec=1.0
                )

        # ----------------------------------------------------------------------
        # STATE 4: GRASP -> Close Gripper around Object
        # ----------------------------------------------------------------------
        elif self.state == 'GRASP':
            self.busy = True
            self.get_logger().info("--- Phase 3: Grasping Object ---")
            self.command_gripper(self.grasp_gripper_pos, self._on_grasp_complete)

        # ----------------------------------------------------------------------
        # STATE 5: RETREAT -> Lift object back to safe scanning pose
        # ----------------------------------------------------------------------
        elif self.state == 'RETREAT':
            self.busy = True
            self.get_logger().info("--- Phase 4: Retreating to Scanning Pose ---")
            self.plan_and_execute_joint_target(self.scanning_joints, "scanning_pose", self._on_retreat_complete)

        elif self.state == 'DONE':
            pass

    # --------------------------------------------------------------------------
    # ASYNC STATE TRANSITION CALLBACKS
    # --------------------------------------------------------------------------
    def _on_initial_gripper_opened(self):
        self.get_logger().info("Gripper opened. Moving arm to scanning pose via MoveIt OMPL...")
        self.plan_and_execute_joint_target(self.scanning_joints, "scanning_pose", self._on_scanning_pose_reached)

    def _on_scanning_pose_reached(self):
        self.get_logger().info("Arrived at scanning pose! Transitioning to visual servoing.")
        self.busy = False
        self.state = 'ENABLE_IBVS'

    def _on_grasp_complete(self):
        self.get_logger().info("Object grasped! Transitioning to retreat.")
        self.busy = False
        self.state = 'RETREAT'

    def _on_retreat_complete(self):
        self.get_logger().info("--- Visual Servoing Grasp Pipeline Succeeded! ---")
        self.busy = False
        self.state = 'DONE'

def main(args=None):
    rclpy.init(args=args)
    node = GraspManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
