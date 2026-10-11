#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os
import yaml
from ament_index_python.packages import get_package_share_directory

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        
        # ROS Parameter for the config file path
        default_config = os.path.expanduser('~/rebotarm_ros2/config/ibvs_pipeline.yaml')
        self.declare_parameter('config_path', default_config)
        config_path = self.get_parameter('config_path').value
        # Declare and get parameters (natively loaded from the ROS 2 parameter file in launch)
        self.declare_parameter('model_name', 'yolo11n.pt')
        self.declare_parameter('target_class', 'bottle')
        self.declare_parameter('custom_classes', ['bottle'])
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('device', 'cpu')
        
        # Load the YAML config
        self.get_logger().info(f"Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            vision_cfg = config.get('vision', {})
        except Exception as e:
            self.get_logger().error(f"Failed to load config file {config_path}: {e}")
            vision_cfg = {}
        self.declare_parameter('target_pixel_x', 640)
        self.declare_parameter('target_pixel_y', 480)
        self.declare_parameter('target_area_px', 40000)
        
        # Extract parameters from the 'vision' block (with defaults just in case)
        model_name = vision_cfg.get('model_name', 'yolo11n.pt')
        self.target_class_name = vision_cfg.get('target_class', 'bottle')
        self.custom_classes = vision_cfg.get('custom_classes', [self.target_class_name])
        self.conf_threshold = vision_cfg.get('conf_threshold', 0.5)
        self.device = vision_cfg.get('device', 'cpu')
        # Extract values
        model_name = self.get_parameter('model_name').value
        self.target_class_name = self.get_parameter('target_class').value
        self.custom_classes = self.get_parameter('custom_classes').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.device = self.get_parameter('device').value
        
        # Extract target parameters for IBVS
        self.target_pixel_x = vision_cfg.get('target_pixel_x', 640)
        self.target_pixel_y = vision_cfg.get('target_pixel_y', 480)
        self.target_area_px = vision_cfg.get('target_area_px', 40000)
        self.target_pixel_x = self.get_parameter('target_pixel_x').value
        self.target_pixel_y = self.get_parameter('target_pixel_y').value
        self.target_area_px = self.get_parameter('target_area_px').value
        
        # Resolve path to the models directory
        package_share_directory = get_package_share_directory('rebotarm_ibvs')
        model_path = os.path.join(package_share_directory, 'models', model_name)
        
        self.get_logger().info(f"Loading YOLO model from: {model_path} on device {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # If using an open-vocab model like YOLO-World or YoloE, set the custom classes
        if "world" in model_name.lower() or "yoloe" in model_name.lower():
            self.get_logger().info(f"Open-vocabulary model detected. Setting classes to: {self.custom_classes}")
            self.model.set_classes(self.custom_classes)
        
        # Get class ID for target class if it exists in the model names
        self.target_class_id = None
        for class_id, class_name in self.model.names.items():
            if class_name.lower() == self.target_class_name.lower():
                self.target_class_id = class_id
                break
                
        if self.target_class_id is None:
            self.get_logger().warn(f"Target class '{self.target_class_name}' not found in model classes. Will default to tracking the first detection.")
        
        self.bridge = CvBridge()
        
        # FPS tracking
        self.fps_start_time = self.get_clock().now().nanoseconds / 1e9
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publishers
        self.bbox_pub = self.create_publisher(Float32MultiArray, '/yolo/target_bounding_box', 10)
        self.annotated_img_pub = self.create_publisher(Image, '/yolo/annotated_image', 10)
        
        self.get_logger().info("YOLO Vision Node initialized and waiting for images...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # FPS Calculation
            self.fps_frame_count += 1
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_frame_count / (now - self.fps_start_time)
                self.fps_frame_count = 0
                self.fps_start_time = now
            
            # Run YOLO inference
            results = self.model(cv_image, conf=self.conf_threshold, verbose=False)
            
            best_det = None
            largest_area = 0
            
            # Parse results
            best_mask_polygon = None
            
            for result in results:
                boxes = result.boxes
                masks = result.masks if hasattr(result, 'masks') else None
                
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    
                    # If we have a specific target class, filter for it
                    if self.target_class_id is not None and cls_id != self.target_class_id:
                        continue
                        
                    # Get box coordinates [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Track the largest detection of the target class
                    if area > largest_area:
                        largest_area = area
                        cx = x1 + width / 2
                        cy = y1 + height / 2
                        best_det = [cx, cy, width, height]
                        
                        # Extract the segmentation mask polygon if it exists
                        if masks is not None and masks.xy is not None and len(masks.xy) > i:
                            best_mask_polygon = masks.xy[i]
            
            # Publish Bounding Box if found
            if best_det is not None:
                bbox_msg = Float32MultiArray()
                bbox_msg.data = [float(val) for val in best_det]
                self.bbox_pub.publish(bbox_msg)
            
            # Annotate and publish image (turn off built-in masks to keep it clean)
            annotated_frame = results[0].plot(masks=False)
            
            # If we found our target, draw the custom overlays
            if best_det is not None:
                import numpy as np
                cx, cy, w, h = best_det
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                
                # 1. Draw AABB (Green)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Determine OBB points
                if best_mask_polygon is not None and len(best_mask_polygon) > 0:
                    contour = np.array(best_mask_polygon, dtype=np.float32)
                    rect = cv2.minAreaRect(contour)
                    box_points = cv2.boxPoints(rect)
                else:
                    # Fallback to AABB if no mask is available
                    box_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                
                # 2. Draw OBB (Cyan/Yellowish)
                box_points_int = np.int32(box_points)
                cv2.polylines(annotated_frame, [box_points_int], True, (255, 200, 0), 2, cv2.LINE_AA)
                
                # 3. Find and draw the short edge line
                best_vec = box_points[1] - box_points[0]
                best_len = float(np.linalg.norm(best_vec))
                for i in range(4):
                    p0 = box_points[i]
                    p1 = box_points[(i + 1) % 4]
                    vec = p1 - p0
                    length = float(np.linalg.norm(vec))
                    if length < best_len:
                        best_vec = vec
                        best_len = length
                        
                center = np.array([cx, cy], dtype=np.float32)
                p0 = np.round(center - 0.5 * best_vec).astype(np.int32)
                p1 = np.round(center + 0.5 * best_vec).astype(np.int32)
                cv2.line(annotated_frame, tuple(p0), tuple(p1), (255, 255, 255), 3, cv2.LINE_AA)
                
                # 4. Draw the center red point
                cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
                # 5. Print X, Y, Z (Area) below the bounding box
                info_text = f"X:{int(cx)} Y:{int(cy)} Z:{int(largest_area)}"
                cv2.putText(annotated_frame, info_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add FPS and crosshair text
            cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Target Class: {self.target_class_name}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
            # Show the target crosshair based on config
            cv2.drawMarker(annotated_frame, (self.target_pixel_x, self.target_pixel_y), 
                           (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            
            # Display image in a pop-up window
            cv2.imshow("YOLO IBVS Vision", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # 27 is ESC key
                self.get_logger().info("Quit requested by user.")
                cv2.destroyAllWindows()
                import sys
                sys.exit(0)
            
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.annotated_img_pub.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

