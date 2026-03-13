#!/usr/bin/env python3
# coding: utf-8

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import WrenchStamped, TwistStamped, AccelStamped  # <-- ADD

class AdmittanceControlNode(Node):
    def __init__(self):
        super().__init__('admittance_control')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 50.0),
                ('sphere_mass', 5.0),
                ('sphere_radius', 0.1),
                ('fluid_viscosity', 0.0010016),
                ('frame_id', 'eoat_camera_link'),
                ('wrench_topic', '/teleop/wrench_cmds'),
                ('wrench_teleop_topic', '/teleop/wrench_cmds'),
                ('wrench_orientation_topic', '/orientation_controller/wrench_cmds'),
                ('twist_topic', '/servo_node/delta_twist_cmds'),

                # NEW: accel topic
                ('accel_topic', '/admittance/accel_cmds'),

                ('max_linear_speed', 0.0),
                ('max_angular_speed', 0.0),
            ]
        )

        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.mass = float(self.get_parameter('sphere_mass').value)
        self.sphere_radius = float(self.get_parameter('sphere_radius').value)
        self.fluid_viscosity = float(self.get_parameter('fluid_viscosity').value)
        self.update_inertia_and_drag(self.mass, self.sphere_radius, self.fluid_viscosity)

        self.frame_id = str(self.get_parameter('frame_id').value)
        wrench_teleop_topic = str(self.get_parameter('wrench_teleop_topic').value)
        wrench_orientation_topic = str(self.get_parameter('wrench_orientation_topic').value)
        twist_topic = str(self.get_parameter('twist_topic').value)

        # NEW: accel topic
        accel_topic = str(self.get_parameter('accel_topic').value)

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.twist_pub = self.create_publisher(TwistStamped, twist_topic, qos_profile)

        # NEW: accel publisher
        self.accel_pub = self.create_publisher(AccelStamped, accel_topic, qos_profile)

        self.teleop_sub = self.create_subscription(
            WrenchStamped, wrench_teleop_topic, self.wrench_callback_teleop, qos_profile
        )
        self.orient_sub = self.create_subscription(
            WrenchStamped, wrench_orientation_topic, self.wrench_callback_orientation, qos_profile
        )

        self.linear_vel  = [0.0, 0.0, 0.0]
        self.angular_vel = [0.0, 0.0, 0.0]

        # NEW: store last computed accelerations (optional, but convenient)
        self.linear_accel  = [0.0, 0.0, 0.0]   # m/s^2
        self.angular_accel = [0.0, 0.0, 0.0]   # rad/s^2

        self.teleop_F = [0.0, 0.0, 0.0]
        self.teleop_T = [0.0, 0.0, 0.0]
        self.orient_F = [0.0, 0.0, 0.0]
        self.orient_T = [0.0, 0.0, 0.0]

        self.have_any_wrench = False

        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id

        # NEW: reuse accel msg too
        self.current_accel = AccelStamped()
        self.current_accel.header.frame_id = self.frame_id

        self.last_time = None
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_twist)

        self.add_on_set_parameters_callback(self._on_param_update)

    def update_inertia_and_drag(self, mass: float, radius: float, fluid_viscosity: float):
        self.inertia = (2.0 / 5.0) * mass * (radius ** 2)
        self.linear_drag = 6.0 * 3.141592653589793 * fluid_viscosity * radius
        self.angular_drag = 2.4 * 3.141592653589793 * fluid_viscosity * (radius ** 3)

    def wrench_callback_teleop(self, msg: WrenchStamped):
        self.teleop_F[0], self.teleop_F[1], self.teleop_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.teleop_T[0], self.teleop_T[1], self.teleop_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    def wrench_callback_orientation(self, msg: WrenchStamped):
        self.orient_F[0], self.orient_F[1], self.orient_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.orient_T[0], self.orient_T[1], self.orient_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    def publish_twist(self):
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

        inv_m = 1.0 / max(self.mass, 1e-9)
        inv_I = 1.0 / max(self.inertia, 1e-9)

        # --- COMPUTE + STORE ACCELS (this is what you asked for) ---
        for i in range(3):
            self.linear_accel[i] = F_cmd[i] * inv_m          # a = F/m
            self.angular_accel[i] = T_cmd[i] * inv_I         # alpha = tau/I

        # --- integrate velocities ---
        for i in range(3):
            self.linear_vel[i] += self.linear_accel[i] * dt
            self.angular_vel[i] += self.angular_accel[i] * dt

        # Optional speed clamps (same as before)
        if self.max_linear_speed > 0.0:
            for i in range(3):
                if self.linear_vel[i] >  self.max_linear_speed: self.linear_vel[i] =  self.max_linear_speed
                if self.linear_vel[i] < -self.max_linear_speed: self.linear_vel[i] = -self.max_linear_speed
        if self.max_angular_speed > 0.0:
            for i in range(3):
                if self.angular_vel[i] >  self.max_angular_speed: self.angular_vel[i] =  self.max_angular_speed
                if self.angular_vel[i] < -self.max_angular_speed: self.angular_vel[i] = -self.max_angular_speed

        # Publish TwistStamped
        tw = self.current_twist
        tw.header.stamp = now.to_msg()
        tw.header.frame_id = self.frame_id
        tw.twist.linear.x  = round(self.linear_vel[0], 3)
        tw.twist.linear.y  = round(self.linear_vel[1], 3)
        tw.twist.linear.z  = round(self.linear_vel[2], 3)
        tw.twist.angular.x = round(self.angular_vel[0], 3)
        tw.twist.angular.y = round(self.angular_vel[1], 3)
        tw.twist.angular.z = round(self.angular_vel[2], 3)

        if any([tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z,
                tw.twist.angular.x, tw.twist.angular.y, tw.twist.angular.z]):
            self.twist_pub.publish(tw)

        # --- NEW: Publish AccelStamped (linear accel + angular accel) ---
        ac = self.current_accel
        ac.header.stamp = now.to_msg()
        ac.header.frame_id = self.frame_id
        ac.accel.linear.x  = float(self.linear_accel[0])
        ac.accel.linear.y  = float(self.linear_accel[1])
        ac.accel.linear.z  = float(self.linear_accel[2])
        ac.accel.angular.x = float(self.angular_accel[0])
        ac.accel.angular.y = float(self.angular_accel[1])
        ac.accel.angular.z = float(self.angular_accel[2])

        # You can keep/remove this "skip if all zero"
        if any([ac.accel.linear.x, ac.accel.linear.y, ac.accel.linear.z,
                ac.accel.angular.x, ac.accel.angular.y, ac.accel.angular.z]):
            self.accel_pub.publish(ac)

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'publish_rate' and p.type_ == p.Type.DOUBLE:
                self.publish_rate = float(p.value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_twist)
            elif p.name == 'sphere_mass' and p.type_ == p.Type.DOUBLE:
                self.mass = float(p.value)
                self.update_inertia_and_drag(self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'sphere_radius' and p.type_ == p.Type.DOUBLE:
                self.sphere_radius = float(p.value)
                self.update_inertia_and_drag(self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'fluid_viscosity' and p.type_ == p.Type.DOUBLE:
                self.fluid_viscosity = float(p.value)
                self.update_inertia_and_drag(self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'max_linear_speed' and p.type_ == p.Type.DOUBLE:
                self.max_linear_speed = float(p.value)
            elif p.name == 'max_angular_speed' and p.type_ == p.Type.DOUBLE:
                self.max_angular_speed = float(p.value)

        result = SetParametersResult()
        result.successful = True
        return result


def main(args=None):
    rclpy.init(args=args)
    node = AdmittanceControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()