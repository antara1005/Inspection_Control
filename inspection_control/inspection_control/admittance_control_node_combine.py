#!/usr/bin/env python3
# coding: utf-8

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import WrenchStamped, TwistStamped

class AdmittanceControlNode(Node):
    """
    Admittance controller that integrates commanded forces/torques into
    linear/angular velocities and publishes TwistStamped at a fixed rate.

    There are two wrench sources:
      1) teleop (e.g., user inputs)
      2) orientation controller (e.g., proportional torque from vision)
    The callbacks only STORE the latest forces/torques; all dynamics
    integration is done at a fixed rate in publish_twist().
    """

    def __init__(self):
        super().__init__('admittance_control')

        # ---------------- Parameters ----------------
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 50.0),             # Hz
                ('sphere_mass', 5.0),                      # kg
                ('sphere_radius', 0.1),                    # m (for inertia and drag calc)
                ('fluid_viscosity', 0.0010016),                  # Pa·s (for drag calc)
                ('frame_id', 'eoat_camera_link'),
                ('wrench_topic', '/teleop/wrench_cmds'),
                ('wrench_teleop_topic', '/teleop/wrench_cmds'),
                ('wrench_orientation_topic', '/orientation_controller/wrench_cmds'),
                ('twist_topic', '/servo_node/delta_twist_cmds'),

                # Optional safety/comfort limits (set to 0 to disable)
                ('max_linear_speed', 0.0),          # m/s (0 = no clamp)
                ('max_angular_speed', 0.0),         # rad/s (0 = no clamp)
            ]
        )

        # Read parameters
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.mass = float(self.get_parameter('sphere_mass').value)
        # Calculate inertia for solid sphere: I = 2/5 m r²
        self.sphere_radius = float(self.get_parameter('sphere_radius').value)
        self.fluid_viscosity = float(self.get_parameter('fluid_viscosity').value)
        self.update_inertia_and_drag(self.mass, self.sphere_radius, self.fluid_viscosity)

        self.frame_id = str(self.get_parameter('frame_id').value)
        wrench_teleop_topic = str(self.get_parameter('wrench_teleop_topic').value)
        wrench_orientation_topic = str(self.get_parameter('wrench_orientation_topic').value)
        twist_topic = str(self.get_parameter('twist_topic').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        # ---------------- QoS ----------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ---------------- Pub/Sub ----------------
        self.twist_pub = self.create_publisher(TwistStamped, twist_topic, qos_profile)

        self.teleop_sub = self.create_subscription(
            WrenchStamped, wrench_teleop_topic, self.wrench_callback_teleop, qos_profile
        )
        self.orient_sub = self.create_subscription(
            WrenchStamped, wrench_orientation_topic, self.wrench_callback_orientation, qos_profile
        )

        # ---------------- State ----------------
        # velocities (integrated state)
        self.linear_vel  = [0.0, 0.0, 0.0]   # m/s
        self.angular_vel = [0.0, 0.0, 0.0]   # rad/s

        # latest wrench commands (buffers)
        self.teleop_F = [0.0, 0.0, 0.0]      # N
        self.teleop_T = [0.0, 0.0, 0.0]      # N·m
        self.orient_F = [0.0, 0.0, 0.0]      # N
        self.orient_T = [0.0, 0.0, 0.0]      # N·m

        self.have_any_wrench = False

        # Twist message we reuse
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id

        # timer / dt tracking
        self.last_time = None
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_twist)

        # ---------------- Parameter update callback ----------------
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(
            f'Admittance control node started — publishing {self.publish_rate:.1f} Hz in frame "{self.frame_id}"'
        )

    def update_inertia_and_drag(self, mass: float, radius: float, fluid_viscosity: float):
        """Update inertia and drag coefficients based on mass, radius, and fluid viscosity."""
        # Inertia for solid sphere: I = 2/5 m r²
        self.inertia = (2.0 / 5.0) * mass * (radius ** 2)

        # Linear drag coefficients: D = 6πμr
        self.linear_drag = 6.0 * 3.141592653589793 * fluid_viscosity * radius

        # Rotational drag coefficients: D_rot = 2.4πμr³ (breaks Stoke's Law but preserves v = rω)
        self.angular_drag = 2.4 * 3.141592653589793 * fluid_viscosity * (radius ** 3)

        self.get_logger().info(
            f'Updated inertia to {self.inertia} kg·m² and drag to {self.linear_drag} N·s/m, {self.angular_drag} N·m·s/rad'
        )

    # --------------- Callbacks: store only ---------------
    def wrench_callback_teleop(self, msg: WrenchStamped):
        """Store teleop wrench (forces & torques)."""
        self.teleop_F[0], self.teleop_F[1], self.teleop_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.teleop_T[0], self.teleop_T[1], self.teleop_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    def wrench_callback_orientation(self, msg: WrenchStamped):
        """Store orientation wrench (forces & torques)."""
        self.orient_F[0], self.orient_F[1], self.orient_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.orient_T[0], self.orient_T[1], self.orient_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    # --------------- Timer: integrate & publish ---------------
    def publish_twist(self):
        """Integrate admittance dynamics and publish TwistStamped at fixed rate."""
        if not self.have_any_wrench:
            return

        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_time = now

        # Net commands = teleop + orientation - damping
        F_cmd = [
            self.teleop_F[0] + self.orient_F[0] - self.linear_drag * self.linear_vel[0],
            self.teleop_F[1] + self.orient_F[1] - self.linear_drag * self.linear_vel[1],
            self.teleop_F[2] + self.orient_F[2] - self.linear_drag * self.linear_vel[2],
        ]
        T_cmd = [
            self.teleop_T[0] + self.orient_T[0] - self.angular_drag * self.angular_vel[0],
            self.teleop_T[1] + self.orient_T[1] - self.angular_drag * self.angular_vel[1],
            self.teleop_T[2] + self.orient_T[2] - self.angular_drag * self.angular_vel[2],
        ]

        # Integrate translational dynamics: v = v + (F/m)*dt
        inv_m = 1.0 / max(self.mass, 1e-9)
        for i in range(3):
            a = F_cmd[i] * inv_m
            self.linear_vel[i] += a * dt

        # Integrate rotational dynamics: w = w + (τ/I)*dt
        for i in range(3):
            inv_I = 1.0 / max(self.inertia, 1e-9)
            alpha = T_cmd[i] * inv_I
            self.angular_vel[i] += alpha * dt

        # Optional speed clamps
        if self.max_linear_speed > 0.0:
            for i in range(3):
                if self.linear_vel[i] >  self.max_linear_speed: self.linear_vel[i] =  self.max_linear_speed
                if self.linear_vel[i] < -self.max_linear_speed: self.linear_vel[i] = -self.max_linear_speed
        if self.max_angular_speed > 0.0:
            for i in range(3):
                if self.angular_vel[i] >  self.max_angular_speed: self.angular_vel[i] =  self.max_angular_speed
                if self.angular_vel[i] < -self.max_angular_speed: self.angular_vel[i] = -self.max_angular_speed

        # If speed is nan, reset to zero
        for i in range(3):
            if not isinstance(self.linear_vel[i], float) or self.linear_vel[i] != self.linear_vel[i]:
                self.linear_vel[i] = 0.0
            if not isinstance(self.angular_vel[i], float) or self.angular_vel[i] != self.angular_vel[i]:
                self.angular_vel[i] = 0.0

        # Fill TwistStamped
        tw = self.current_twist
        tw.header.stamp = now.to_msg()
        tw.header.frame_id = self.frame_id

        tw.twist.linear.x  = round(self.linear_vel[0], 3)
        tw.twist.linear.y  = round(self.linear_vel[1], 3)
        tw.twist.linear.z  = round(self.linear_vel[2], 3)
        tw.twist.angular.x = round(self.angular_vel[0], 3)
        tw.twist.angular.y = round(self.angular_vel[1], 3)
        tw.twist.angular.z = round(self.angular_vel[2], 3)

        # Publish (skip if everything zero)
        if not any([tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z,
                    tw.twist.angular.x, tw.twist.angular.y, tw.twist.angular.z]):
            return
        self.twist_pub.publish(tw)

        # Param updates
    def _on_param_update(self, params):
        for p in params:
            if p.name == 'publish_rate' and p.type_ == p.Type.DOUBLE:
                self.publish_rate = float(p.value)
                self.get_logger().info(f'Updated publish_rate to {self.publish_rate:.1f} Hz')
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_twist)
            elif p.name == 'sphere_mass' and p.type_ == p.Type.DOUBLE:
                self.mass = float(p.value)
                self.get_logger().info(f'Updated sphere_mass to {self.mass:.3f} kg')
                self.update_inertia_and_drag(self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'sphere_radius' and p.type_ == p.Type.DOUBLE:
                self.sphere_radius = float(p.value)
                self.get_logger().info(f'Updated sphere_radius to {self.sphere_radius:.3f} m')
                self.update_inertia_and_drag(self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'fluid_viscosity' and p.type_ == p.Type.DOUBLE:
                self.fluid_viscosity = float(p.value)
                self.get_logger().info(f'Updated fluid_viscosity to {self.fluid_viscosity:.6f} Pa·s')
                self.update_inertia_and_drag(self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'max_linear_speed' and p.type_ == p.Type.DOUBLE:
                self.max_linear_speed = float(p.value)
                self.get_logger().info(f'Updated max_linear_speed to {self.max_linear_speed:.3f} m/s')
            elif p.name == 'max_angular_speed' and p.type_ == p.Type.DOUBLE:
                self.max_angular_speed = float(p.value)
                self.get_logger().info(f'Updated max_angular_speed to {self.max_angular_speed:.3f} rad/s')
        
        result = SetParametersResult()
        result.successful = True

        return result


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AdmittanceControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'Unhandled exception: {e}')
        else:
            print(f'Unhandled exception before node init: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
