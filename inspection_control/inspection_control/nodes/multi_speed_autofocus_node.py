#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, TwistStamped, Quaternion
import geometry_msgs.msg
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import tf2_geometry_msgs
from std_srvs.srv import Trigger
from rclpy.clock import ROSClock
from viewpoint_generation_interfaces.srv import MoveToPoseStamped
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import rosbag2_py

import numpy as np
from std_msgs.msg import Float64, Header
from viewpoint_generation_interfaces.msg import FocusValue
import math
import csv
import os
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from datetime import datetime

MOVING_DISTANCE = 0.15
kv = 0.8  # Example value for kv, adjust as needed

IDLE = 0
FEEDBACK = 1
SLEEP_INIT = 2


class PoseStampedCreator(Node):

    def __init__(self):
        super().__init__('pose_stamped_creator')
        self.timestamp = 0

        self.state = SLEEP_INIT  # IDLE
        self.counter = 0
        self.initial_timestamp = None
        self.speed = 0.25

        # Create a client to the service
        self.pose_client = self.create_client(
            MoveToPoseStamped, 'viewpoint_traversal/move_to_pose_stamped')
        while not self.pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info('Hello world.')

        # Publisher to see where the robot is going
        self.pose_publisher = self.create_publisher(
            PoseStamped, 'published_pose', 10)

        # TF stuff
        self.tf_buffer = Buffer()  # store and manipulate transformations
        self.listener = TransformListener(self.tf_buffer, self)

        # srv request (target_pose)
        self.req = MoveToPoseStamped.Request()

        # Create the twist client
        self.twist_publisher = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.get_logger().info('Twist says hello.')
        self.twist_req = Trigger.Request()
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.twist_callback)

        # Dynamic reconfigure twist_started
        # dynamic_typing=True makes the parameter dynamically reconfigurable
        self.declare_parameter(
            'x_twist_speed', 0.0, descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('y_twist_speed', self.speed,
                               descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('z_twist_speed', 0.0, descriptor=ParameterDescriptor(
            dynamic_typing=True))  # declare twist_speed as a parameter
        self.declare_parameter(
            'x_twist_angular', 0.0, descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter(
            'y_twist_angular', 0.0, descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('z_twist_angular', 0.0, descriptor=ParameterDescriptor(
            dynamic_typing=True))  # declare z_twist_angular as a parameter

        self.x_twist_speed = self.get_parameter('x_twist_speed').value
        self.y_twist_speed = self.get_parameter('y_twist_speed').value
        self.z_twist_speed = self.get_parameter('z_twist_speed').value
        self.x_twist_angular = self.get_parameter('x_twist_angular').value
        self.y_twist_angular = self.get_parameter('y_twist_angular').value
        self.z_twist_angular = self.get_parameter('z_twist_angular').value
        # call callback when parameter changes
        self.add_on_set_parameters_callback(self.on_parameter_change)
        ###########################

        # Sobel initialize
        self.focus_pose_dict = {}
        self.new_focus_value_sub = self.create_subscription(
            FocusValue, '/camera/image_raw/compressed/focus_value', self.new_focus_value_callback, 10)
        self.ema_focus_value = 0
        self.ema_focus_value2 = 0
        self.dema_focus_value = 0
        self.previous_dema_focus_value = 0
        self.ratio = 0
        self.dFV = 0  # computed using dema_focus_value
        self.smooth_ddFV = 0  # computed using dema_focus_value
        self.previous_dFV = 0
        self.keys = None
        self.max_pose = Pose()
        ###########################

        self.timer_callback_group = ReentrantCallbackGroup()
        self.tf_wt = TransformStamped()
        self.tf_timer = self.create_timer(
            0.1, self.on_timer, callback_group=self.timer_callback_group)

        # Create client to run multiple times
        self.start_callback_group = ReentrantCallbackGroup()
        self.start_service = self.create_service(
            Trigger, '/inspection/count_callback', self.start_callback, callback_group=self.start_callback_group)
        self.get_logger().info('Start gathering data.')

    # Move to the pose we received

    def on_timer(self):
        # print(self.state)
        if self.state == FEEDBACK:
            self.simple_feedback()

        elif self.state == SLEEP_INIT:
            try:
                self.tf_wt_start = self.tf_buffer.lookup_transform(
                    'object_frame', 'eoat_camera_link', rclpy.time.Time(), rclpy.time.Duration(seconds=1.0))
                print('Got the transform')
            except TransformException as e:
                self.get_logger().error(f"Transform lookup failed: {e}")
            self.start_pose = PoseStamped()
            self.start_pose.header.frame_id = 'object_frame'
            self.start_pose.pose.position.x = self.tf_wt_start.transform.translation.x
            self.start_pose.pose.position.y = self.tf_wt_start.transform.translation.y
            self.start_pose.pose.position.z = self.tf_wt_start.transform.translation.z
            self.start_pose.pose.orientation = self.tf_wt_start.transform.rotation
            self.state = IDLE

    def start_callback(self, request, response):
        self.counter = 0
        if self.state == IDLE:
            self.initial_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.send_request_world(self.start_pose.pose)
            # Wait for the robot to move to the start pose. Necessary, otherwise sometimes the robot doesn't move to the start pose
            time.sleep(2)
            self.focus_pose_dict = {}
            self.ema_focus_value = 0
            self.ema_focus_value2 = 0
            self.dema_focus_value = 0
            self.previous_dema_focus_value = 0
            self.ratio = 0
            self.previous_dFV = 0
            self.dFV = 0  # computed using dema_focus_value
            self.smooth_ddFV = 0  # computed using dema_focus_value
            self.keys = None
            self.max_pose = Pose()
            self.state = FEEDBACK

            while self.counter < 1:
                if self.state == IDLE:
                    self.send_request_world(self.start_pose.pose)
                    # Wait for the robot to move to the start pose
                    time.sleep(2)
                    self.state = FEEDBACK
                time.sleep(1)

            self.send_request_world(self.max_pose.pose)
            time.sleep(1)

            # Get the current time and pose after moving to max_pose
            curr_time = rclpy.clock.Clock().now()
            # Create a new timestamp for the maximum focus entry
            final_timestamp = curr_time.nanoseconds / 1e9  # Ensure a unique timestamp

            # Save the maximum focus value and its associated pose
            # Do I need this??
            self.focus_pose_dict[final_timestamp] = {
                'focus_value': self.curr_focus_value,
                'ema': self.ema_focus_value,
                'dema': self.dema_focus_value,
                'dFV': self.dFV,
                'smooth_ddFV': self.smooth_ddFV,
                'ratio': self.ratio,
                'pose': self.max_pose,
                'metric': self.metric
            }

            self.save_focus_pose_dict_to_csv()
            self.focus_pose_dict = {}

            return response

    def send_request_world(self, pose_map):
        curr_time = rclpy.time.Time()
        tf_wt = self.tf_buffer.lookup_transform(
            'object_frame', 'eoat_camera_link', curr_time)

        # Send the transformed pose as the request
        target_pose_map = PoseStamped()
        target_pose_map.header.frame_id = 'object_frame'
        # Ensure pose_map is a Pose object
        if not isinstance(pose_map, Pose):
            pose_map = Pose(
                position=pose_map.position,
                orientation=pose_map.orientation
            )
        target_pose_map.pose = pose_map
        self.req.pose_goal = target_pose_map

        # publish destination pose to topic 'published_pose'
        self.pose_publisher.publish(target_pose_map)

        # Call the service
        req = MoveToPoseStamped.Request()
        req.pose_goal = target_pose_map

        future = self.pose_client.call_async(req)
        future.add_done_callback(self.response_callback_world)
        self.state = IDLE  # State change needs to occur after moving to pose with max focus
        return

    def response_callback_world(self, future):
        try:
            response = future.result()
            if response.done:
                self.get_logger().info('Final Service completed successfully.')
                return
            else:
                self.get_logger().error('Final Service failed.')
        except Exception as e:
            self.get_logger().error('Final Service call failed %r' % (e,))

    # on_parameter_change is a new method that gets called whenever a parameter changes. It updates self.twist_started with the new value of the twist_started parameter
    def on_parameter_change(self, parameters):
        for parameter in parameters:
            if parameter.name == 'x_twist_speed':
                self.x_twist_speed = parameter.value
            elif parameter.name == 'y_twist_speed':
                self.y_twist_speed = parameter.value
            elif parameter.name == 'z_twist_speed':
                self.z_twist_speed = parameter.value
            elif parameter.name == 'x_twist_angular':
                self.x_twist_angular = parameter.value
            elif parameter.name == 'y_twist_angular':
                self.y_twist_angular = parameter.value
            elif parameter.name == 'z_twist_angular':
                self.z_twist_angular = parameter.value
        return SetParametersResult(successful=True)

    def twist_callback(self):
        msg = TwistStamped()
        msg.header.stamp = ROSClock().now().to_msg()
        msg.header.frame_id = 'eoat_camera_link'
        msg.twist.linear.x = self.x_twist_speed
        msg.twist.linear.y = self.y_twist_speed
        msg.twist.linear.z = self.z_twist_speed
        msg.twist.angular.x = self.x_twist_angular
        msg.twist.angular.y = self.y_twist_angular
        msg.twist.angular.z = self.z_twist_angular
        self.twist_publisher.publish(msg)

    def new_focus_value_callback(self, msg):
        self.new_time = msg.header.stamp
        self.curr_focus_value = msg.data  # Raw data. Filtering would now be using EMA
        self.metric = msg.metric

        try:
            # Attempt to get the transform at the exact requested time
            tf_wt = self.tf_buffer.lookup_transform(
                'object_frame', 'eoat_camera_link', self.new_time)
        except tf2_ros.TransformException as ex:
            # self.get_logger().info(f'Could not fine tune. Using latest known transform instead') # {ex}')
            # Fallback to the latest available transform within a 1-second duration
            tf_wt = self.tf_buffer.lookup_transform(
                'object_frame', 'eoat_camera_link', rclpy.time.Time(), rclpy.time.Duration(seconds=1.0))

        pose_world = PoseStamped()
        pose_world.header.frame_id = 'object_frame'
        pose_world.pose.position.x = tf_wt.transform.translation.x
        pose_world.pose.position.y = tf_wt.transform.translation.y
        pose_world.pose.position.z = tf_wt.transform.translation.z
        pose_world.pose.orientation = tf_wt.transform.rotation

        # Convert Time msg to float value
        self.timestamp = self.new_time.sec + self.new_time.nanosec / 1e9

        # Calculate the EMA
        N_ema = 20  # Doesn't work consistently with 15
        if self.ema_focus_value == 0:
            self.ema_focus_value = self.curr_focus_value
            self.ema_focus_value2 = self.curr_focus_value  # For DEMA calculation
            self.dema_focus_value = self.curr_focus_value
        K = 2 / (N_ema + 1)  # EMA smoothing factor for the last 15 periods

        self.previous_dema_focus_value = self.dema_focus_value
        self.ema_focus_value = (
            K * (self.curr_focus_value - self.ema_focus_value)) + self.ema_focus_value
        self.ema_focus_value2 = (
            K * (self.ema_focus_value - self.ema_focus_value2)) + self.ema_focus_value2
        self.dema_focus_value = 2 * self.ema_focus_value - self.ema_focus_value2

        if self.previous_dema_focus_value != 0:
            self.ratio = self.dema_focus_value / self.previous_dema_focus_value

        # Compute dFV and ddFV
        if len(self.focus_pose_dict) == 0:
            self.dFV = 0
            ddFV = 0
        elif len(self.focus_pose_dict) == 1:
            self.dFV = self.dema_focus_value - self.previous_dema_focus_value
            ddFV = 0
        else:
            self.previous_dFV = self.dFV
            self.dFV = self.dema_focus_value - self.previous_dema_focus_value
            ddFV = self.dFV - self.previous_dFV
        # Smoothing ddFV
            K_smooth = 2 / (3 + 1)
            self.smooth_ddFV = (
                K_smooth * (ddFV - self.smooth_ddFV)) + self.smooth_ddFV

        # Save the current focus value and its associated pose
        self.focus_pose_dict[self.timestamp] = {
            'focus_value': self.curr_focus_value,
            'ema': self.ema_focus_value,
            'dema': self.dema_focus_value,
            'dFV': self.dFV,
            'smooth_ddFV': self.smooth_ddFV,
            'ratio': self.ratio,
            'pose': pose_world,
            'metric': self.metric
        }

    def simple_feedback(self):
        # Current and Offset pose to move forward
        curr_time_fb = rclpy.time.Time()
        tf_wt_fb = self.tf_buffer.lookup_transform(
            'object_frame', 'eoat_camera_link', curr_time_fb)

        current_pose = PoseStamped()
        current_pose.header.frame_id = 'object_frame'
        current_pose.pose.position.x = tf_wt_fb.transform.translation.x
        current_pose.pose.position.y = tf_wt_fb.transform.translation.y
        current_pose.pose.position.z = tf_wt_fb.transform.translation.z
        current_pose.pose.orientation = tf_wt_fb.transform.rotation

        if self.focus_pose_dict:
            if self.smooth_ddFV < -0.1 and self.dFV > 0:  # account for noise
                self.speed = kv*(self.ratio-0.5)  # kv*(self.ratio-kv)
                self.set_parameters([rclpy.parameter.Parameter(
                    'y_twist_speed', rclpy.parameter.Parameter.Type.DOUBLE, self.speed)])
                self.y_twist_speed = self.get_parameter('y_twist_speed').value
                print('dFV, ddFV, kv*r', self.dFV,
                      self.smooth_ddFV, self.speed)
            # elif self.smooth_ddFV > 0.1 and self.dFV < 0: # account for noise
            #     self.speed = -kv*(self.ratio-0.5) # kv*(self.ratio-kv)
            #     self.set_parameters([rclpy.parameter.Parameter('y_twist_speed', rclpy.parameter.Parameter.Type.DOUBLE, self.speed)])
            #     self.y_twist_speed = self.get_parameter('y_twist_speed').value
            #     print('dFV, ddFV, -kv*r',self.dFV,self.smooth_ddFV,self.speed,self.y_twist_speed)
            # dFV approximately 0 and ddFV negative (account for noise)
            elif self.previous_dFV > 2 and self.dFV < 2 and self.smooth_ddFV < -0.1:
                # self.set_parameters([rclpy.parameter.Parameter(
                #     'twist_started', rclpy.parameter.Parameter.Type.BOOL, False)])
                # self.twist_started = self.get_parameter('twist_started').value
                print('Completed autofocus!')
                print('dFV, ddFV', self.dFV, self.smooth_ddFV)

                # Find the pose with the maximum focus value
                max_focus_entry = max(self.focus_pose_dict.items(
                ), key=lambda item: item[1]['focus_value'])
                self.max_pose = max_focus_entry[1]['pose']

                # self.save_focus_pose_dict_to_csv()
                # self.focus_pose_dict = {}
                self.counter += 1
                print('counter =', self.counter)
                self.state = IDLE
            else:
                self.speed = kv/self.ratio
                self.set_parameters([rclpy.parameter.Parameter(
                    'y_twist_speed', rclpy.parameter.Parameter.Type.DOUBLE, self.speed)])
                self.y_twist_speed = self.get_parameter('y_twist_speed').value
                print('dFV, ddFV, kv/r', self.dFV,
                      self.smooth_ddFV, self.speed)
        return

    def save_focus_pose_dict_to_csv(self):
        self.focus_value_count = -1
        # Define the directory and file name
        directory = '/data/'
        # Generate the initial timestamp if it doesn't exist

        # Define the file name with the initial timestamp
        file_name = f'focus_pose_dict_{self.initial_timestamp}.csv'
        file_path = os.path.join(directory, file_name)

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Check if the file exists and is not empty
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='') as csvfile:
            # Extract the keys for the header
            fieldnames = ['timestamp', 'focus_value', 'ema', 'dema',
                          'dFV', 'smooth_ddFV', 'ratio', 'x', 'y', 'z', 'metric']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header only if the file is empty or doesn't exist
            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writeheader()

            t0 = list(self.focus_pose_dict.keys())[0]
            for timestamp, data in self.focus_pose_dict.items():
                pose = data['pose'].pose
                positionx = pose.position.x
                positiony = pose.position.y
                positionz = pose.position.z
                metric = data['metric']
                row = {
                    'timestamp': timestamp - t0,
                    'focus_value': data['focus_value'],
                    'ema': data['ema'],
                    'dema': data['dema'],
                    'dFV': data['dFV'],
                    'smooth_ddFV': data['smooth_ddFV'],
                    'ratio': data['ratio'],
                    'x': positionx,
                    'y': positiony,
                    'z': positionz,
                    'metric': metric
                }
                writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    pose_stamped_client = PoseStampedCreator()
    executor = MultiThreadedExecutor()
    executor.add_node(pose_stamped_client)
    try:
        # rclpy.spin(pose_stamped_client)
        executor.spin()
    except KeyboardInterrupt:
        # pass
        pose_stamped_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
