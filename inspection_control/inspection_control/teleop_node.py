#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Joy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
from rclpy.time import Time
import math


class TeleopNode(Node):
    """
    Node that converts Joy messages to Wrench messages with fixed-rate publishing.
    """

    def __init__(self):
        super().__init__('teleop')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 50.0),  # Hz
                ('force_scale', 1.0),  # Max linear velocity (m/s)
                ('torque_scale', 1.0),  # Max angular velocity (rad/s)
                ('x_axis', 0),  # Pan Camera Left/Right
                ('y_axis', 1),  # Pan Camera Up/Down
                ('z_axis', -1),  # Pan Camera Up/Down
                ('z_axis_push', 2),  # Push Camera
                ('z_axis_pull', 6),  # Pull camera
                ('roll_axis_positive', 3),  # Roll Camera About Z-Axis
                ('roll_axis_negative', 7),  # Roll Camera
                ('pitch_axis', 4),  # Pitch Camera About X-Axis
                ('yaw_axis', 5),   # Yaw Camera About Y-Axis
                ('invert_x', False),  # Invert X-axis for left/right movement
                ('invert_y', False),  # Invert Y-axis for up/down movement
                ('invert_z', False),  # Invert Z-axis for push/pull movement
                ('invert_roll', False),  # Invert X-axis for left/right movement
                ('invert_pitch', False),  # Invert X-axis for left/right movement
                ('invert_yaw', False),  # Invert X-axis for left/right movement
                ('enable_button', 0),  # Safety button to enable/disable control
                ('deadzone', 0.05),  # Deadzone to prevent drift
                ('frame_id', 'base_link'),  # Frame ID for TwistStamped
                ('joy_topic', 'joy'),  # Topic for incoming Joy messages
                # Topic for outgoing TwistStamped messages
                ('twist_topic', '/teleop/wrench_cmds')
            ]
        )

        # Get parameters
        self.publish_rate = self.get_parameter(
            'publish_rate').get_parameter_value().double_value
        self.force_scale = self.get_parameter(
            'force_scale').get_parameter_value().double_value
        self.torque_scale = self.get_parameter(
            'torque_scale').get_parameter_value().double_value
        self.x_axis = self.get_parameter(
            'x_axis').get_parameter_value().integer_value
        self.y_axis = self.get_parameter(
            'y_axis').get_parameter_value().integer_value
        self.z_axis = self.get_parameter(
            'z_axis').get_parameter_value().integer_value
        self.z_axis_push = self.get_parameter(
            'z_axis_push').get_parameter_value().integer_value
        self.z_axis_pull = self.get_parameter(
            'z_axis_pull').get_parameter_value().integer_value
        self.roll_axis_positive = self.get_parameter(
            'roll_axis_positive').get_parameter_value().integer_value
        self.roll_axis_negative = self.get_parameter(
            'roll_axis_negative').get_parameter_value().integer_value
        self.pitch_axis = self.get_parameter(
            'pitch_axis').get_parameter_value().integer_value
        self.yaw_axis = self.get_parameter(
            'yaw_axis').get_parameter_value().integer_value
        self.invert_x = self.get_parameter(
            'invert_x').get_parameter_value().bool_value
        self.invert_y = self.get_parameter(
            'invert_y').get_parameter_value().bool_value
        self.invert_z = self.get_parameter(
            'invert_z').get_parameter_value().bool_value
        self.invert_roll = self.get_parameter(
            'invert_roll').get_parameter_value().bool_value
        self.invert_pitch = self.get_parameter(
            'invert_pitch').get_parameter_value().bool_value
        self.invert_yaw = self.get_parameter(
            'invert_yaw').get_parameter_value().bool_value
        self.enable_button = self.get_parameter(
            'enable_button').get_parameter_value().integer_value
        self.deadzone = self.get_parameter(
            'deadzone').get_parameter_value().double_value
        self.frame_id = self.get_parameter(
            'frame_id').get_parameter_value().string_value
        joy_topic = self.get_parameter(
            'joy_topic').get_parameter_value().string_value
        twist_topic = self.get_parameter(
            'twist_topic').get_parameter_value().string_value

        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publishers and subscribers
        self.wrench_pub = self.create_publisher(
            WrenchStamped,
            twist_topic,
            qos_profile
        )

        self.joy_sub = self.create_subscription(
            Joy,
            joy_topic,
            self.joy_callback,
            qos_profile
        )

        # Internal state
        self.last_joy_msg = None
        self.current_wrench = WrenchStamped()
        self.current_wrench.header.frame_id = self.frame_id

        # Create timer for fixed-rate publishing
        timer_period = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(timer_period, self.publish_wrench)

        # Log startup info
        self.get_logger().info(f'Teleop node started')
        self.get_logger().info(f'Publishing at {self.publish_rate} Hz')
        self.get_logger().info(f'Enable button: {self.enable_button}')

        # ---------------- Parameter update callback ----------------
        self.add_on_set_parameters_callback(self._on_param_update)

    def joy_callback(self, msg):
        """
        Process incoming Joy messages and update wrench command.
        """
        self.last_joy_msg = msg

        # Check if enable button is pressed (safety feature) or if button is not configured
        if not msg.buttons[self.enable_button] and self.enable_button >= 0:
            # Enable button not pressed - stop the robot
            self.current_wrench.wrench.force.x = 0.0
            self.current_wrench.wrench.force.y = 0.0
            self.current_wrench.wrench.force.z = 0.0
            self.current_wrench.wrench.torque.x = 0.0
            self.current_wrench.wrench.torque.y = 0.0
            self.current_wrench.wrench.torque.z = 0.0
            return

        # Get raw axis values
        fx_raw = msg.axes[self.x_axis] if not self.invert_x else - \
            msg.axes[self.x_axis]
        fy_raw = msg.axes[self.y_axis] if not self.invert_y else - \
            msg.axes[self.y_axis]

        if self.z_axis >= 0:
            fz_raw = msg.axes[self.z_axis] if not self.invert_z else - \
                msg.axes[self.z_axis]
        else:
            fz_raw_up = (1-msg.axes[self.z_axis_push])/2.0
            fz_raw_down = (1-msg.axes[self.z_axis_pull])/2.0
            fz_raw = fz_raw_up - fz_raw_down if not self.invert_z else fz_raw_down - \
                fz_raw_up  # range from 1 to -1 like x and y axes

        tr_raw_positive = msg.buttons[self.roll_axis_positive]
        tr_raw_negative = msg.buttons[self.roll_axis_negative]
        tr_raw = tr_raw_positive - tr_raw_negative if not self.invert_roll else tr_raw_negative - \
            tr_raw_positive  # range from 1 to -1 like x and y axes
        tp_raw = msg.axes[self.pitch_axis] if not self.invert_pitch else - \
            msg.axes[self.pitch_axis]
        ty_raw = msg.axes[self.yaw_axis] if not self.invert_yaw else - \
            msg.axes[self.yaw_axis]

        # Apply deadzone
        fx_filtered = self.apply_deadzone(fx_raw)
       # fx_filtered =self.apply_normalize(fx_filtered_deadzone)
        fy_filtered = self.apply_deadzone(fy_raw)
      #  fy_filtered = self.apply_normalize(fy_filtered_deadzone)
        fz_filtered = self.apply_deadzone(fz_raw)
       # fz_filtered = self.apply_normalize(fz_filtered_deadzone)
        # vz_filtered = self.apply_deadzone(vz_raw)
        tr_filtered = self.apply_deadzone(tr_raw)
       # tr_filtered = self.apply_normalize(tr_filtered_deadzone)
        tp_filtered = self.apply_deadzone(tp_raw)
      #  tp_filtered = self.apply_normalize(tp_filtered_deadzone)
        ty_filtered = self.apply_deadzone(ty_raw)
      #  ty_filtered = self.apply_normalize(ty_filtered_deadzone)

        # Calculate scaled forces and torques
        fx = fx_filtered * self.force_scale
        fy = fy_filtered * self.force_scale
        fz = fz_filtered * self.force_scale
        # fz_push = (1.0 - fz_raw_push) / 2.0 *  self.force_scale
        # fz_pull = (1.0 - fz_raw_pull) / 2.0 *  self.force_scale
        # tr_positive = tr_raw_positive * self.torque_scale
        # tr_negative= tr_raw_negative* self.torque_scale
        tr = tr_filtered * self.torque_scale
        tp = tp_filtered * self.torque_scale
        ty = ty_filtered * self.torque_scale

        # Update twist message
        self.current_wrench.wrench.force.x = fx
        self.current_wrench.wrench.force.y = fy
        self.current_wrench.wrench.force.z = fz
        # self.current_wrench.force.z_pull=fz_pull
        # self.current_wrench.torque.z_negative= tr_negative
        # self.current_wrench.force.z = vz if not self.invert_z else -vz
        self.current_wrench.wrench.torque.z = tr
        self.current_wrench.wrench.torque.x = tp
        self.current_wrench.wrench.torque.y = ty

        self.current_wrench.header.stamp = msg.header.stamp

        # self.current_twist.twist.angular.z = wr if not self.invert_z else -wr

    def apply_deadzone(self, value):
        """
        Apply deadzone to joystick input to eliminate drift.
        """
        if abs(value) < self.deadzone:
            return 0.0
        else:
            # return value
            # def apply_normalize(self,value):
            # Scale the remaining range to 0-1
           # def apply_normalize(self, value):
            if value > 0:
                return (value - self.deadzone) / (1.0 - self.deadzone)
            else:
                return (value + self.deadzone) / (1.0 - self.deadzone)

    def publish_wrench(self):
        """
        Publish TwistStamped message at fixed rate.
        """
        # Update timestamp
       # self.current_wrench.header.stamp = self.get_clock().now().to_msg()

        # Publish the message
        self.wrench_pub.publish(self.current_wrench)

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'force_scale' and p.type_ == p.Type.DOUBLE:
                self.force_scale = p.value
            elif p.name == 'torque_scale' and p.type_ == p.Type.DOUBLE:
                self.torque_scale = p.value
                
        result = SetParametersResult()
        result.successful = True

        return result

def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = TeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
