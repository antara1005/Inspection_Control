#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Joy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
from std_srvs.srv import Trigger
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
                # Fine-mode XY nudge on the D-pad (digital hat axes). Added on
                # top of the analog-stick force with its own small scale.
                # D-pad R (axis=-1)->+x, L (+1)->-x; Up (+1)->-y, Down (-1)->+y.
                # Set an axis to -1 to disable that fine direction.
                ('fine_x_axis', 6),
                ('fine_y_axis', 7),
                ('fine_force_scale', 0.1),
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
                ('twist_topic', '/teleop/wrench_cmds'),
                # Buttons that trigger orientation_control_node actions via
                # its Trigger services (this node owns all Joy parsing so
                # button/axis bounds only need to be validated in one place).
                ('orientation_control_enable_button', 0),
                ('visualize_normal_button', 9),
                ('save_data_button', 8),
                # Autofocus button actions, driven over the autofocus node's
                # Trigger services (this node owns all Joy parsing).
                ('autofocus_record_button', 2),   # toggle peak recording
                ('autofocus_drive_button', 3),    # toggle PD drive-to-peak
                # "All stop" button (B): disables orientation + autofocus
                # controllers unconditionally.
                ('disable_controllers_button', 1),
                # Node names used to build the Trigger service names
                # (e.g. /<name>/toggle_orientation_control).
                ('orientation_control_node_name', 'orientation_controller'),
                ('autofocus_node_name', 'autofocus'),
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
        self.fine_x_axis = self.get_parameter(
            'fine_x_axis').get_parameter_value().integer_value
        self.fine_y_axis = self.get_parameter(
            'fine_y_axis').get_parameter_value().integer_value
        self.fine_force_scale = self.get_parameter(
            'fine_force_scale').get_parameter_value().double_value
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
        self.orientation_control_enable_button = self.get_parameter(
            'orientation_control_enable_button').get_parameter_value().integer_value
        self.visualize_normal_button = self.get_parameter(
            'visualize_normal_button').get_parameter_value().integer_value
        self.save_data_button = self.get_parameter(
            'save_data_button').get_parameter_value().integer_value
        self.autofocus_record_button = self.get_parameter(
            'autofocus_record_button').get_parameter_value().integer_value
        self.autofocus_drive_button = self.get_parameter(
            'autofocus_drive_button').get_parameter_value().integer_value
        self.disable_controllers_button = self.get_parameter(
            'disable_controllers_button').get_parameter_value().integer_value
        orientation_control_node_name = self.get_parameter(
            'orientation_control_node_name').get_parameter_value().string_value
        autofocus_node_name = self.get_parameter(
            'autofocus_node_name').get_parameter_value().string_value

        # Every axis/button index this node reads from a Joy message. Indices
        # of -1 mean "unused" (matches the existing enable_button/z_axis
        # convention below) and are excluded from the bounds check.
        self._joy_button_indices = [i for i in (
            self.enable_button, self.roll_axis_positive, self.roll_axis_negative,
            self.orientation_control_enable_button, self.visualize_normal_button,
            self.save_data_button, self.autofocus_record_button,
            self.autofocus_drive_button, self.disable_controllers_button,
        ) if i >= 0]
        axis_indices = [self.x_axis, self.y_axis, self.pitch_axis, self.yaw_axis,
                        self.fine_x_axis, self.fine_y_axis]
        if self.z_axis >= 0:
            axis_indices.append(self.z_axis)
        else:
            axis_indices.extend([self.z_axis_push, self.z_axis_pull])
        self._joy_axis_indices = [i for i in axis_indices if i >= 0]
        self._max_button_idx = max(self._joy_button_indices, default=-1)
        self._max_axis_idx = max(self._joy_axis_indices, default=-1)

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

        # Trigger service clients for orientation_control_node's button-driven
        # actions. Non-blocking: if the node isn't up yet the call is skipped
        # with a warning rather than stalling the joy callback.
        self.toggle_orientation_control_client = self.create_client(
            Trigger, f'/{orientation_control_node_name}/toggle_orientation_control')
        self.toggle_normal_estimation_viz_client = self.create_client(
            Trigger, f'/{orientation_control_node_name}/toggle_normal_estimation_viz')
        self.toggle_save_data_client = self.create_client(
            Trigger, f'/{orientation_control_node_name}/toggle_save_data')

        # Autofocus Trigger clients (same non-blocking pattern as above).
        self.toggle_autofocus_recording_client = self.create_client(
            Trigger, f'/{autofocus_node_name}/toggle_recording')
        self.toggle_autofocus_driving_client = self.create_client(
            Trigger, f'/{autofocus_node_name}/toggle_driving')

        # "All stop" clients: disable both controllers unconditionally.
        self.disable_orientation_control_client = self.create_client(
            Trigger, f'/{orientation_control_node_name}/disable_orientation_control')
        self.disable_autofocus_client = self.create_client(
            Trigger, f'/{autofocus_node_name}/disable_autofocus')

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

    def _zero_wrench(self):
        self.current_wrench.wrench.force.x = 0.0
        self.current_wrench.wrench.force.y = 0.0
        self.current_wrench.wrench.force.z = 0.0
        self.current_wrench.wrench.torque.x = 0.0
        self.current_wrench.wrench.torque.y = 0.0
        self.current_wrench.wrench.torque.z = 0.0

    def _validate_joy_msg(self, msg: Joy) -> bool:
        """Confirm this Joy message has enough buttons/axes for the indices
        configured via parameters, so a joystick with fewer buttons/axes than
        expected (or a malformed message) can't cause an IndexError."""
        if len(msg.buttons) <= self._max_button_idx or len(msg.axes) <= self._max_axis_idx:
            self.get_logger().error(
                f'Joy message has {len(msg.buttons)} buttons / {len(msg.axes)} axes, '
                f'but configured *_button/*_axis parameters need at least '
                f'{self._max_button_idx + 1} buttons and {self._max_axis_idx + 1} axes. '
                'Check the joystick model against this node\'s parameters. '
                'Ignoring this message and stopping output for safety.',
                throttle_duration_sec=5.0)
            return False
        return True

    @staticmethod
    def _rising_edge(msg: Joy, last_msg, idx: int) -> bool:
        if idx < 0 or last_msg is None:
            return False
        if idx >= len(msg.buttons) or idx >= len(last_msg.buttons):
            return False
        return bool(msg.buttons[idx]) and not bool(last_msg.buttons[idx])

    def _call_trigger(self, client, action_name: str):
        if not client.service_is_ready():
            self.get_logger().warn(
                f'{action_name} service not available (orientation control node not up?); '
                'ignoring button press.', throttle_duration_sec=5.0)
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self._on_trigger_response(f, action_name))

    def _on_trigger_response(self, future, action_name: str):
        try:
            result = future.result()
            if not result.success:
                self.get_logger().warn(f'{action_name} failed: {result.message}')
        except Exception as e:
            self.get_logger().error(f'{action_name} service call raised: {e}')

    def joy_callback(self, msg):
        """
        Process incoming Joy messages and update wrench command.
        """
        if not self._validate_joy_msg(msg):
            # Fail safe: stop the robot rather than keep publishing a stale
            # command, and don't adopt this message as the edge-detection
            # baseline.
            self._zero_wrench()
            return

        prev_msg = self.last_joy_msg
        self.last_joy_msg = msg

        if self._rising_edge(msg, prev_msg, self.orientation_control_enable_button):
            self._call_trigger(self.toggle_orientation_control_client, 'toggle_orientation_control')
        if self._rising_edge(msg, prev_msg, self.visualize_normal_button):
            self._call_trigger(self.toggle_normal_estimation_viz_client, 'toggle_normal_estimation_viz')
        if self._rising_edge(msg, prev_msg, self.save_data_button):
            self._call_trigger(self.toggle_save_data_client, 'toggle_save_data')
        if self._rising_edge(msg, prev_msg, self.autofocus_record_button):
            self._call_trigger(self.toggle_autofocus_recording_client, 'toggle_recording')
        if self._rising_edge(msg, prev_msg, self.autofocus_drive_button):
            self._call_trigger(self.toggle_autofocus_driving_client, 'toggle_driving')
        # "All stop" (B button): disable both controllers at once.
        if self._rising_edge(msg, prev_msg, self.disable_controllers_button):
            self._call_trigger(self.disable_orientation_control_client, 'disable_orientation_control')
            self._call_trigger(self.disable_autofocus_client, 'disable_autofocus')

        # Check if enable button is pressed (safety feature) or if button is not configured
        if not msg.buttons[self.enable_button] and self.enable_button >= 0:
            # Enable button not pressed - stop the robot
            self._zero_wrench()
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

        # Fine-mode XY nudge from the D-pad (digital hat axes), added on top of
        # the analog-stick force. Axes are -1/0/+1 so no deadzone is applied.
        # D-pad R (axis=-1)->+x, L (+1)->-x; Up (+1)->-y, Down (-1)->+y.
        if self.fine_x_axis >= 0:
            fx += -msg.axes[self.fine_x_axis] * self.fine_force_scale
        if self.fine_y_axis >= 0:
            fy += -msg.axes[self.fine_y_axis] * self.fine_force_scale
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
            elif p.name == 'fine_force_scale' and p.type_ == p.Type.DOUBLE:
                self.fine_force_scale = p.value
                
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
