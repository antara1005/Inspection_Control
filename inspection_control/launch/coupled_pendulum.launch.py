#!/usr/bin/env python3
"""Launch the Tier 3 coupled-pendulum orientation/admittance pair.

Mirrors admittance_control.launch.py but swaps in the coupled-pendulum variants:

  orientation_control_coupled_pendulum  (name: orientation_controller_coupled_pendulum)
      -> emits a PURE swing torque on /orientation_controller_coupled_pendulum/wrench_cmds
      -> publishes the pivot lever arm r on /orientation_controller_coupled_pendulum/pivot_r

  admittance_control_coupled_pendulum   (name: admittance_control_coupled_pendulum)
      -> integrates the coupled pivot-referenced pendulum and drives the servo twist.

The node names are chosen so the plant's default wrench/pivot topics line up with what
the orientation node publishes (topics are derived from the node name) -- no remaps
needed. Do NOT run this together with admittance_control.launch.py; both drive
/servo_node/delta_twist_cmds.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    orientation_config_file = DeclareLaunchArgument(
        'orientation_config_file',
        default_value='orientation_controller_coupled_pendulum.yaml',
        description='Orientation (coupled-pendulum) configuration file'
    )
    admittance_config_file = DeclareLaunchArgument(
        'admittance_config_file',
        default_value='admittance_control_coupled_pendulum.yaml',
        description='Admittance (coupled-pendulum plant) configuration file'
    )
    autofocus_config_file = DeclareLaunchArgument(
        'autofocus_config_file',
        default_value='autofocus.yaml',
        description='Autofocus configuration file'
    )
    controller_config_file = DeclareLaunchArgument(
        'controller_config_file',
        default_value='xbox_controller.yaml',
        description='Teleop controller configuration file'
    )
    turntable_config_file = DeclareLaunchArgument(
        'turntable_config_file',
        default_value='turntable_joy.yaml',
        description='Turntable joy configuration file'
    )
    tsdf_config_file = DeclareLaunchArgument(
        'tsdf_config_file',
        default_value='tsdf_pose.yaml',
        description='TSDF pose-estimation configuration file'
    )

    moveit_config = (
        MoveItConfigsBuilder("inspection_cell")
        .moveit_cpp(file_path="config/motion_planning.yaml")
        .planning_scene_monitor(
            publish_planning_scene=False,
            publish_geometry_updates=False,
            publish_state_updates=False,
            publish_transforms_updates=False,
        )
        .to_moveit_configs()
    )

    orientation_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'), 'config',
        LaunchConfiguration('orientation_config_file')
    ])
    admittance_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'), 'config',
        LaunchConfiguration('admittance_config_file')
    ])
    autofocus_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'), 'config',
        LaunchConfiguration('autofocus_config_file')
    ])
    teleop_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'), 'config',
        LaunchConfiguration('controller_config_file')
    ])
    turntable_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'), 'config',
        LaunchConfiguration('turntable_config_file')
    ])
    tsdf_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'), 'config',
        LaunchConfiguration('tsdf_config_file')
    ])

    orientation_control_node = Node(
        package='inspection_control',
        executable='orientation_control_coupled_pendulum',
        name='orientation_controller_coupled_pendulum',
        parameters=[orientation_config, moveit_config.to_dict()],
        output='screen',
        emulate_tty=True
    )
    admittance_control_node = Node(
        package='inspection_control',
        executable='admittance_control_coupled_pendulum',
        name='admittance_control_coupled_pendulum',
        parameters=[admittance_config],
        output='screen',
        emulate_tty=True
    )
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy'
    )
    autofocus_node = Node(
        package='inspection_control',
        executable='autofocus_node',
        name='autofocus',
        parameters=[autofocus_config],
        output='screen',
        emulate_tty=True
    )
    teleop_node = Node(
        package='inspection_control',
        executable='teleop',
        name='teleop',
        parameters=[teleop_config],
        output='screen',
        emulate_tty=True
    )
    servo_logger_node = Node(
        package='inspection_control',
        executable='servo_logger',
        name='servo_logger',
        output='screen',
        emulate_tty=True
    )
    turntable_joy_node = Node(
        package='inspection_control',
        executable='turntable_joy_node',
        name='turntable_joy_node',
        parameters=[turntable_config],
        output='screen',
        emulate_tty=True
    )
    tsdf_pose_node = Node(
        package='inspection_control',
        executable='tsdf_pose_node',
        name='tsdf_pose',
        parameters=[tsdf_config],
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        orientation_config_file,
        admittance_config_file,
        autofocus_config_file,
        controller_config_file,
        turntable_config_file,
        tsdf_config_file,
        orientation_control_node,
        admittance_control_node,
        joy_node,
        autofocus_node,
        teleop_node,
        servo_logger_node,
        turntable_joy_node,
        tsdf_pose_node,
    ])
