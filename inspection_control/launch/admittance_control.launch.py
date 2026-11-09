#!/usr/bin/env python3
"""Launch the inspection cell's control stack.

The orientation/admittance pair uses the Tier 3 coupled-pendulum dynamics:

  orientation_control_node  (name: orientation_controller)
      -> emits a PURE swing torque on /orientation_controller/wrench_cmds
      -> publishes the pivot lever arm r on /orientation_controller/pivot_r

  admittance_control_node   (name: admittance_control)
      -> integrates the coupled pivot-referenced pendulum and drives the servo twist.

The node names are chosen so the plant's default wrench/pivot topics line up with what
the orientation node publishes (topics are derived from the node name) -- no remaps
needed. Alongside them: autofocus, focus_volume, teleop (behind remote_teleop_gate
with remote_teleop:=true), servo_logger and turntable_joy; with dual_arm:=true, a
second admittance plant for the Zivid arm.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    orientation_config_file = DeclareLaunchArgument(
        'orientation_config_file',
        default_value='orientation_controller.yaml',
        description='Orientation controller configuration file'
    )
    admittance_config_file = DeclareLaunchArgument(
        'admittance_config_file',
        default_value='admittance_control.yaml',
        description='Admittance control (coupled-pendulum plant) configuration file'
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
    dual_arm = DeclareLaunchArgument(
        'dual_arm',
        default_value='false',
        description='Add admittance control for the second (Zivid) arm, and let '
                    'arm_switch_button hand the joystick between the arms.'
    )
    remote_teleop = DeclareLaunchArgument(
        'remote_teleop',
        default_value='false',
        description='Let a remote operator drive through the Remote Teleop Foxglove panel: '
                    'starts remote_teleop_gate between the joystick (joy_node, moved to '
                    'joy_local) and teleop. The local joystick keeps working and always '
                    'takes over. See the README, "Remote teleop".'
    )
    focus_volume = DeclareLaunchArgument(
        'focus_volume',
        default_value='true',
        description='Start focus_volume_node: AKAZE + KLT features on the macro camera, '
                    'each with its own focus measure, for the hybrid autofocus; with a '
                    'Foxglove overlay. See the README, "Focus volume". false leaves it out.'
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

    orientation_control_node = Node(
        package='inspection_control',
        executable='orientation_control_node',
        name='orientation_controller',
        parameters=[orientation_config, moveit_config.to_dict()],
        output='screen',
        emulate_tty=True
    )
    admittance_control_node = Node(
        package='inspection_control',
        executable='admittance_control_node',
        name='admittance_control',
        parameters=[admittance_config],
        output='screen',
        emulate_tty=True
    )
    # joy_node repeats its last message at autorepeat_rate while a stick is
    # held, which the teleop and turntable joystick timeouts rely on: a silent
    # joystick means a lost one. With remote_teleop, it publishes on joy_local
    # and remote_teleop_gate decides what reaches `joy`.
    #
    # SDL_JOYSTICK_HIDAPI=0 forces SDL onto the kernel evdev joystick driver.
    # Where the user can open /dev/hidraw* (uaccess ACL, e.g. this desktop),
    # SDL otherwise uses its HIDAPI driver, which reports the GameController
    # layout (6 axes, D-pad as buttons 11-14) instead of the 8-axis layout
    # (triggers on 2/5, D-pad hat on 6/7) that xbox_controller.yaml expects.
    # Pinning it also keeps a controller's layout from changing when a newer
    # SDL gains a HIDAPI driver for it.
    remote = LaunchConfiguration('remote_teleop')
    # The controller profile (controller_config_file) has a `joy` section:
    # which device to open, and its deadzone.
    joy_params = [{'autorepeat_rate': 20.0}, teleop_config]
    joy_env = {'SDL_JOYSTICK_HIDAPI': '0'}
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy',
        parameters=joy_params,
        additional_env=joy_env,
        condition=UnlessCondition(remote)
    )
    joy_node_behind_gate = Node(
        package='joy',
        executable='joy_node',
        name='joy',
        parameters=joy_params,
        additional_env=joy_env,
        remappings=[('joy', 'joy_local')],
        condition=IfCondition(remote)
    )
    autofocus_node = Node(
        package='inspection_control',
        executable='autofocus_node',
        name='autofocus',
        parameters=[autofocus_config],
        output='screen',
        emulate_tty=True
    )
    # Feature front end of the hybrid autofocus (several cores at 30 fps; it
    # pins itself off the real-time cores, see focus_volume.yaml).
    focus_volume_node = Node(
        package='inspection_control',
        executable='focus_volume_node',
        name='focus_volume',
        parameters=[PathJoinSubstitution([
            FindPackageShare('inspection_control'), 'config', 'focus_volume.yaml'])],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('focus_volume'))
    )
    # Second arm: admittance plant + the teleop target it listens on. Only
    # admittance and teleop; the orientation/autofocus stack is for the main
    # arm's macro camera.
    is_dual = PythonExpression(["'", LaunchConfiguration('dual_arm'), "' == 'true'"])
    zivid_admittance_control_node = Node(
        package='inspection_control',
        executable='admittance_control_node',
        name='admittance_control',
        namespace='zivid',
        parameters=[PathJoinSubstitution([
            FindPackageShare('inspection_control'), 'config',
            'admittance_control_zivid.yaml'])],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(is_dual)
    )
    teleop_node = Node(
        package='inspection_control',
        executable='teleop',
        name='teleop',
        parameters=[teleop_config, {
            # Empty unless dual_arm: arm_switch_button then does nothing.
            'secondary_twist_topic': PythonExpression([
                "'/zivid/teleop/wrench_cmds' if ", is_dual, " else ''"]),
            'secondary_frame_id': 'zivid_camera_optical_frame',
            'secondary_deadman_topic': PythonExpression([
                "'/zivid/teleop/deadman' if ", is_dual, " else ''"]),
            'secondary_servo_node': PythonExpression([
                "'/zivid_servo_node' if ", is_dual, " else ''"]),
            # The Zivid is mounted upside down relative to the main arm's
            # camera, so x/y translation and rotation about x/y (pitch/yaw)
            # are flipped for that arm.
            'secondary_invert_x': True,
            'secondary_invert_y': True,
            'secondary_invert_pitch': True,
            'secondary_invert_yaw': True,
        }],
        output='screen',
        emulate_tty=True
    )
    # The only path from a remote operator to the arm; see remote_teleop_gate.py.
    # Its speed limits apply to every admittance controller that is running.
    def remote_teleop_gate(admittance_nodes, condition):
        return Node(
            package='inspection_control',
            executable='remote_teleop_gate',
            name='remote_teleop_gate',
            # Layout and deadman from the controller profile's
            # remote_teleop_gate section.
            parameters=[teleop_config, {'admittance_nodes': admittance_nodes}],
            output='screen',
            emulate_tty=True,
            condition=condition
        )
    gate_single = remote_teleop_gate(['/admittance_control'], IfCondition(PythonExpression([
        "'", remote, "' == 'true' and not ", is_dual])))
    gate_dual = remote_teleop_gate(
        ['/admittance_control', '/zivid/admittance_control'], IfCondition(PythonExpression([
            "'", remote, "' == 'true' and ", is_dual])))

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
        # The profile's turntable_joy_node section (jog buttons, deadman)
        # overrides turntable_joy.yaml.
        parameters=[turntable_config, teleop_config],
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        orientation_config_file,
        admittance_config_file,
        autofocus_config_file,
        controller_config_file,
        turntable_config_file,
        dual_arm,
        remote_teleop,
        focus_volume,
        orientation_control_node,
        admittance_control_node,
        zivid_admittance_control_node,
        joy_node,
        joy_node_behind_gate,
        gate_single,
        gate_dual,
        autofocus_node,
        focus_volume_node,
        teleop_node,
        servo_logger_node,
        turntable_joy_node,
    ])
