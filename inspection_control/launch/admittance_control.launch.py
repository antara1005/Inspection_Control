#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """Generate launch description for teleop_twist_stamped_joy node."""

    # Launch arguments
    particle_filter_config_file = DeclareLaunchArgument(
        'particle_filter_config_file',
        default_value='particle_filter.yaml',
        description='Name of particle filter configuration file'
    )
    orientation_config_file = DeclareLaunchArgument(
        'orientation_config_file',
        default_value='orientation_controller.yaml',
        description='Name of controller configuration file'
    )
    autofocus_config_file = DeclareLaunchArgument(
        'autofocus_config_file',
        default_value='autofocus.yaml',
        description='Name of controller configuration file'
    )
    teleop_config_file = DeclareLaunchArgument(
        'teleop_config_file',
        default_value='xbox_controller.yaml',
        description='Name of controller configuration file'
    )
    admittance_config_file = DeclareLaunchArgument(
        'admittance_config_file',
        default_value='admittance_control.yaml',
        description='Name of controller configuration file'
    )
    turntable_config_file = DeclareLaunchArgument(
        'turntable_config_file',
        default_value='turntable_joy.yaml',
        description='Name of turntable joy configuration file'
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

    particle_filter_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'),
        'config',
        LaunchConfiguration('particle_filter_config_file')
    ])
    orientation_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'),
        'config',
        LaunchConfiguration('orientation_config_file')
    ])
    autofocus_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'),
        'config',
        LaunchConfiguration('autofocus_config_file')
    ])
    teleop_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'),
        'config',
        LaunchConfiguration('teleop_config_file')
    ])
   # admittance_control_config = PathJoinSubstitution([
      #  FindPackageShare('inspection_control'),
      #  'config',
      #  LaunchConfiguration('admittance_config_file')
   # ])
    admittance_control_combine_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'),
        'config',
        LaunchConfiguration('admittance_config_file')
    ])
    turntable_config = PathJoinSubstitution([
        FindPackageShare('inspection_control'),
        'config',
        LaunchConfiguration('turntable_config_file')
    ])
    
    particle_filter_node = Node(
        package='inspection_control',
        executable='particle_filter_node',
        name='particle_filter',
        parameters=[particle_filter_config],
        output='screen',
        emulate_tty=True
    )
    orientation_control_node = Node(
        package='inspection_control',
        executable='orientation_control_node',
        name='orientation_controller',
        parameters=[orientation_config,
                    moveit_config.to_dict()],
        output='screen',
        emulate_tty=True
    )
    joy_node = Node(
        package='joy',
        executable="joy_node",
        name='joy'
    )

    # autofocus_node = Node(
    #     package="inspection_control",
    #     executable="autofocus_node",
    #     name="autofocus",
    #     parameters=[autofocus_config],
    #     output="screen",
    #     emulate_tty=True
    # )

    # Teleop node
    teleop_node = Node(
        package='inspection_control',
        executable='teleop',
        name='teleop',
        parameters=[teleop_config],
        output='screen',
        emulate_tty=True
    )
    #admittance_control_node = Node(
      #  package='inspection_control',
      #  executable='admittance_control',
      #  name='admittance_control',
      #  parameters=[admittance_control_config],
      #  output='screen',
      #  emulate_tty=True
   # )
    admittance_control_combine_node = Node(
        package='inspection_control',
        executable='admittance_control_combine',
        name='admittance_control',
        parameters=[admittance_control_combine_config],
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
    return LaunchDescription([
        particle_filter_config_file,
        particle_filter_node,
        orientation_config_file,
        orientation_control_node,
        autofocus_config_file,
        # autofocus_node,
        joy_node,
        teleop_config_file,
        admittance_config_file,
        turntable_config_file,
        teleop_node,
       # admittance_control_node,
        admittance_control_combine_node,
        turntable_joy_node,
    ])
