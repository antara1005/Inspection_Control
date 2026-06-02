import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
import numpy as np  
import cv2
from PIL import Image as PILImage
import io

from cv_bridge import CvBridge, CvBridgeError
from inspection_control import focus_metrics
from viewpoint_generation_interfaces.msg import FocusValue
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage, Image

class FocusMonitorNode(Node):
    def __init__(self):
        super().__init__('focus_monitor_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('metric', 'sobel'),
                ('image_topic', '/camera/image_raw/compressed'),
                ('roi_width', 100),
                ('roi_height', 100),
                ('roi_x', 0.5),
                ('roi_y', 0.5),
            ]
        )
        self.metric = self.get_parameter('metric').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.roi_width = self.get_parameter('roi_width').get_parameter_value().integer_value
        self.roi_height = self.get_parameter('roi_height').get_parameter_value().integer_value
        self.roi_x = self.get_parameter('roi_x').get_parameter_value().double_value
        self.roi_y = self.get_parameter('roi_y').get_parameter_value().double_value

        self.bridge = CvBridge()

        self.subscriber = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(FocusValue, f'{self.image_topic}/focus_value', 10)
        self.float_publisher = self.create_publisher(Float64, f'{self.image_topic}/focus_value_float', 10)

        self.add_on_set_parameters_callback(self.parameter_callback)

    def image_callback(self, msg):
        # Process the incoming image message
        self.get_logger().info('Received compressed image message')

        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg).astype(np.uint8)

        height, width, _ = cv_image.shape
        x_center = int(self.roi_x * width)
        y_center = int(self.roi_y * height)
        x_start = max(0, x_center - self.roi_width // 2)
        y_start = max(0, y_center - self.roi_height // 2)
        x_end = min(width, x_center + self.roi_width // 2)
        y_end = min(height, y_center + self.roi_height // 2)

        roi = cv_image[y_start:y_end, x_start:x_end]

        if roi.size == 0:
            self.get_logger().warn("ROI is empty, skipping focus calculation")
            return

        try:
            if self.metric == 'laplacian':
                self.get_logger().info('Calculating Sobel-Laplacian focus metric')
                focus_value, _ = focus_metrics.sobel_laplacian(roi)
            elif self.metric == 'sobel':
                self.get_logger().info('Calculating Sobel focus metric')
                focus_value, _ = focus_metrics.sobel(roi)
            elif self.metric == 'squared_gradient':
                self.get_logger().info('Calculating Squared Gradient focus metric')
                focus_value, _ = focus_metrics.squared_gradient(roi)
            elif self.metric == 'fswm':
                self.get_logger().info('Calculating FSWM focus metric')
                focus_value, _ = focus_metrics.fswm(roi)
            else:
                self.get_logger().error(f'Unknown focus metric: {self.metric}')
                return

            focus_value_msg = FocusValue()
            focus_value_msg.header.stamp = self.get_clock().now().to_msg()
            focus_value_msg.header.frame_id = "eoat_camera_link"
            focus_value_msg.metric = self.metric
            focus_value_msg.data = float(focus_value)
            self.publisher.publish(focus_value_msg)
            self.float_publisher.publish(Float64(data=float(focus_value)))

        except Exception as e:
            self.get_logger().error(f"Error calculating focus metric: {e}")

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'metric':
                self.metric = param.value
            elif param.name == 'image_topic':
                self.image_topic = param.value
                self.subscriber.destroy()
                self.subscriber = self.create_subscription(
                    CompressedImage,
                    self.image_topic,
                    self.image_callback,
                    10
                )
            elif param.name == 'roi_width':
                self.roi_width = param.value
            elif param.name == 'roi_height':
                self.roi_height = param.value
            elif param.name == 'roi_x':
                self.roi_x = param.value
            elif param.name == 'roi_y':
                self.roi_y = param.value

        result = SetParametersResult()
        result.successful = True

        return result
        
def main():
    rclpy.init()
    node = FocusMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    def image_callback(self, msg):
        # Process the incoming image message
        self.get_logger().info('Received compressed image message')
        
        try:
            # Workaround: Use PIL to decode JPEG, then convert to OpenCV format
            # Convert array.array to bytes
            if hasattr(msg.data, 'tobytes'):
                jpeg_bytes = msg.data.tobytes()
            else:
                jpeg_bytes = bytes(msg.data)
            
            # Use PIL to decode JPEG
            pil_image = PILImage.open(io.BytesIO(jpeg_bytes))
            
            # Convert PIL image to numpy array
            cv_image = np.array(pil_image)
            
            # PIL uses RGB, OpenCV uses BGR - convert if needed
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            self.get_logger().info(f"Successfully converted image shape: {cv_image.shape}")
            
        except Exception as e:
            self.get_logger().error(f"Error decoding compressed image: {e}")
            return

        if cv_image.size == 0:
            self.get_logger().warn("Received empty image")
            return
            
        self.get_logger().info(f"Successfully converted image shape: {cv_image.shape}")

        height, width, _ = cv_image.shape
        x_center = int(self.roi_x * width)
        y_center = int(self.roi_y * height)
        x_start = max(0, x_center - self.roi_width // 2)
        y_start = max(0, y_center - self.roi_height // 2)
        x_end = min(width, x_center + self.roi_width // 2)
        y_end = min(height, y_center + self.roi_height // 2)

        roi = cv_image[y_start:y_end, x_start:x_end]

        if roi.size == 0:
            self.get_logger().warn("ROI is empty, skipping focus calculation")
            return

        try:
            if self.metric == 'laplacian':
                self.get_logger().info('Calculating Sobel-Laplacian focus metric')
                focus_value, _ = focus_metrics.sobel_laplacian(roi)
            elif self.metric == 'sobel':
                self.get_logger().info('Calculating Sobel focus metric')
                focus_value, _ = focus_metrics.sobel(roi)
            elif self.metric == 'squared_gradient':
                self.get_logger().info('Calculating Squared Gradient focus metric')
                focus_value, _ = focus_metrics.squared_gradient(roi)
            elif self.metric == 'fswm':
                self.get_logger().info('Calculating FSWM focus metric')
                focus_value, _ = focus_metrics.fswm(roi)
            else:
                self.get_logger().error(f'Unknown focus metric: {self.metric}')
                return

            focus_value_msg = FocusValue()
            focus_value_msg.header.stamp = self.get_clock().now().to_msg()
            focus_value_msg.header.frame_id = "eoat_camera_link"
            focus_value_msg.metric = self.metric
            focus_value_msg.value = float(focus_value)
            self.publisher.publish(focus_value_msg)
            
            self.get_logger().info(f"Published focus value: {focus_value}")

        except Exception as e:
            self.get_logger().error(f"Error calculating focus metric: {e}")

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'metric':
                self.metric = param.value
            elif param.name == 'image_topic':
                self.image_topic = param.value
                self.subscriber.destroy()
                self.subscriber = self.create_subscription(
                    CompressedImage,
                    self.image_topic,
                    self.image_callback,
                    10
                )
            elif param.name == 'roi_width':
                self.roi_width = param.value
            elif param.name == 'roi_height':
                self.roi_height = param.value
            elif param.name == 'roi_x':
                self.roi_x = param.value
            elif param.name == 'roi_y':
                self.roi_y = param.value

        result = SetParametersResult()
        result.successful = True

        return result
        
def main():
    rclpy.init()
    node = FocusMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()