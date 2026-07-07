#!/usr/bin/env python3
# coding: utf-8

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import WrenchStamped, TwistStamped, AccelStamped, PointStamped


class AdmittanceControlCoupledPendulumNode(Node):
    def __init__(self):
        super().__init__('admittance_control')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 10.0),
                ('sphere_mass', 2.5),
                ('sphere_radius', 0.65),
                ('fluid_viscosity', 1.0),
                ('frame_id', 'eoat_camera_link'),
                ('teleop_wrench_topic', '/teleop/wrench_cmds'),
                # Tier 3: subscribe to the coupled-pendulum orientation node (pure torque wrench).
                ('orientation_wrench_topic', '/orientation_controller/wrench_cmds'),
                ('autofocus_wrench_topic', '/autofocus/wrench_cmds'),
                ('servo_twist_topic', '/servo_node/delta_twist_cmds'),

                # NEW: accel topic
                ('accel_topic', '/admittance/accel_cmds'),

                # Tier 3 coupled pendulum: pivot lever arm r (= camera - A) + staleness timeout.
                ('pivot_topic', '/orientation_controller/pivot_r'),
                ('pivot_timeout', 0.2),

                ('max_linear_speed', 0.0),
                ('max_angular_speed', 0.0),
            ]
        )

        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.mass = float(self.get_parameter('sphere_mass').value)
        self.sphere_radius = float(self.get_parameter('sphere_radius').value)
        self.fluid_viscosity = float(
            self.get_parameter('fluid_viscosity').value)
        self.update_inertia_and_drag(
            self.mass, self.sphere_radius, self.fluid_viscosity)

        self.frame_id = str(self.get_parameter('frame_id').value)
        teleop_wrench_topic = str(
            self.get_parameter('teleop_wrench_topic').value)
        orientation_wrench_topic = str(
            self.get_parameter('orientation_wrench_topic').value)
        autofocus_wrench_topic = str(
            self.get_parameter('autofocus_wrench_topic').value)
        servo_twist_topic = str(self.get_parameter('servo_twist_topic').value)

        # NEW: accel topic
        accel_topic = str(self.get_parameter('accel_topic').value)

        # Tier 3 pivot frame
        pivot_topic = str(self.get_parameter('pivot_topic').value)
        self.pivot_timeout = float(self.get_parameter('pivot_timeout').value)

        self.max_linear_speed = float(
            self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(
            self.get_parameter('max_angular_speed').value)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.twist_pub = self.create_publisher(
            TwistStamped, servo_twist_topic, qos_profile)

        # NEW: accel publisher
        self.accel_pub = self.create_publisher(
            AccelStamped, accel_topic, qos_profile)

        self.teleop_sub = self.create_subscription(
            WrenchStamped, teleop_wrench_topic, self.wrench_callback_teleop, qos_profile
        )
        self.orient_sub = self.create_subscription(
            WrenchStamped, orientation_wrench_topic, self.wrench_callback_orientation, qos_profile
        )
        self.autofocus_sub = self.create_subscription(
            WrenchStamped, autofocus_wrench_topic, self.wrench_callback_autofocus, qos_profile
        )

        # Tier 3: pivot lever arm r (= camera - A, |r| = d) from the coupled-pendulum
        # orientation node. None / stale => free point-mass fallback.
        self.r_pivot = None
        self._pivot_t = 0.0
        self.pivot_sub = self.create_subscription(
            PointStamped, pivot_topic, self.on_pivot, qos_profile)

        self.linear_vel = [0.0, 0.0, 0.0]
        self.angular_vel = [0.0, 0.0, 0.0]

        # NEW: store last computed accelerations (optional, but convenient)
        self.linear_accel = [0.0, 0.0, 0.0]   # m/s^2
        self.angular_accel = [0.0, 0.0, 0.0]   # rad/s^2

        self.teleop_F = [0.0, 0.0, 0.0]
        self.teleop_T = [0.0, 0.0, 0.0]
        self.orient_F = [0.0, 0.0, 0.0]
        self.orient_T = [0.0, 0.0, 0.0]
        self.autofocus_F = [0.0, 0.0, 0.0]
        self.autofocus_T = [0.0, 0.0, 0.0]

        self.have_any_wrench = False

        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id

        # NEW: reuse accel msg too
        self.current_accel = AccelStamped()
        self.current_accel.header.frame_id = self.frame_id

        self.last_time = None
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.publish_twist)

        self.add_on_set_parameters_callback(self._on_param_update)

    def update_inertia_and_drag(self, mass: float, radius: float, fluid_viscosity: float):
        self.inertia = (2.0 / 5.0) * mass * (radius ** 2)
        self.linear_drag = 6.0 * 3.141592653589793 * fluid_viscosity * radius
        self.angular_drag = 2.4 * 3.141592653589793 * \
            fluid_viscosity * (radius ** 3)

    def wrench_callback_teleop(self, msg: WrenchStamped):
        self.teleop_F[0], self.teleop_F[1], self.teleop_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.teleop_T[0], self.teleop_T[1], self.teleop_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    def wrench_callback_orientation(self, msg: WrenchStamped):
        self.orient_F[0], self.orient_F[1], self.orient_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.orient_T[0], self.orient_T[1], self.orient_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    def wrench_callback_autofocus(self, msg: WrenchStamped):
        self.autofocus_F[0], self.autofocus_F[1], self.autofocus_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.autofocus_T[0], self.autofocus_T[1], self.autofocus_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self.have_any_wrench = True

    def on_pivot(self, msg: PointStamped):
        # r = camera - A, expressed in the camera frame; the pendulum math assumes this
        # matches the integration frame (frame_id).
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            self.get_logger().warn(
                f'pivot_r frame "{msg.header.frame_id}" != integration frame '
                f'"{self.frame_id}"; coupled-pendulum math assumes they match.',
                throttle_duration_sec=5.0)
        self.r_pivot = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        self._pivot_t = self.get_clock().now().nanoseconds * 1e-9

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

        # ---- Summed RAW efforts (drag now lives inside the pendulum EOM) ----
        F = np.array([
            self.teleop_F[0] + self.orient_F[0] + self.autofocus_F[0],
            self.teleop_F[1] + self.orient_F[1] + self.autofocus_F[1],
            self.teleop_F[2] + self.orient_F[2] + self.autofocus_F[2],
        ], dtype=float)
        T = np.array([
            self.teleop_T[0] + self.orient_T[0] + self.autofocus_T[0],
            self.teleop_T[1] + self.orient_T[1] + self.autofocus_T[1],
            self.teleop_T[2] + self.orient_T[2] + self.autofocus_T[2],
        ], dtype=float)

        v = np.array(self.linear_vel, dtype=float)
        w = np.array(self.angular_vel, dtype=float)
        m, c, I_B, c_ang = self.mass, self.linear_drag, self.inertia, self.angular_drag

        now_s = now.nanoseconds * 1e-9
        pivot_fresh = (self.r_pivot is not None
                       and (now_s - self._pivot_t) <= self.pivot_timeout
                       and np.linalg.norm(self.r_pivot) > 1e-4)

        if pivot_fresh:
            # ===== Coupled pivot-referenced pendulum (Tier 3) =====
            # Decompose the camera twist onto the local surface frame, integrate the
            # decoupled, pivot-relative EOM, recompose. Drag + centrifugal/Coriolis
            # coupling fall out dissipatively (no feedforward, no algebraic loop).
            r = self.r_pivot
            d = float(np.linalg.norm(r))
            u = r / d                                   # line of sight A->camera (+u grows standoff)

            w_roll = float(w @ u) * u                   # roll part (about optical axis)
            w_tilt = w - w_roll                         # tilt/swing part (perp to u)

            Q_d = float(F @ u)                          # standoff force (along u)
            Q_a = F - Q_d * u                           # tangential slide force (perp u)
            tau_roll = float(T @ u) * u                 # roll torque (along u)
            tau_tilt = T - tau_roll                     # swing torque (perp u)

            ddot = float(v @ u)                         # standoff rate
            v_sw = np.cross(w, r)                       # swing-induced velocity (omega x r)
            v_a = v - ddot * u - v_sw                   # tangential slide velocity

            I_A = I_B + m * d * d                        # parallel-axis swing inertia

            a_a    = (Q_a - c * v_a) / max(m, 1e-9)
            d_ddot = (Q_d - c * ddot + m * d * float(w_tilt @ w_tilt)) / max(m, 1e-9)
            a_tilt = (tau_tilt - (c * d * d) * w_tilt
                      - 2.0 * m * d * ddot * w_tilt) / max(I_A, 1e-9)
            a_roll = (tau_roll - c_ang * w_roll) / max(I_B, 1e-9)

            v_a    = v_a    + a_a    * dt
            ddot   = ddot   + d_ddot * dt
            w_tilt = w_tilt + a_tilt * dt
            w_roll = w_roll + a_roll * dt

            v_new = v_a + ddot * u + np.cross(w_tilt, r)
            w_new = w_tilt + w_roll
        else:
            # ===== Fallback: free point mass (original behaviour, drag re-added) =====
            F_cmd = F - c * v
            T_cmd = T - c_ang * w
            v_new = v + (F_cmd / max(m, 1e-9)) * dt
            w_new = w + (T_cmd / max(I_B, 1e-9)) * dt

        # store accel (for AccelStamped / KF) then commit the integrated velocity
        self.linear_accel = list((v_new - v) / dt)
        self.angular_accel = list((w_new - w) / dt)
        self.linear_vel = list(v_new)
        self.angular_vel = list(w_new)

        # Optional speed clamps (same as before)
        if self.max_linear_speed > 0.0:
            for i in range(3):
                if self.linear_vel[i] > self.max_linear_speed:
                    self.linear_vel[i] = self.max_linear_speed
                if self.linear_vel[i] < -self.max_linear_speed:
                    self.linear_vel[i] = -self.max_linear_speed
        if self.max_angular_speed > 0.0:
            for i in range(3):
                if self.angular_vel[i] > self.max_angular_speed:
                    self.angular_vel[i] = self.max_angular_speed
                if self.angular_vel[i] < -self.max_angular_speed:
                    self.angular_vel[i] = -self.max_angular_speed

        # Publish TwistStamped
        tw = self.current_twist
        tw.header.stamp = now.to_msg()
        tw.header.frame_id = self.frame_id
        tw.twist.linear.x = round(self.linear_vel[0], 3)
        tw.twist.linear.y = round(self.linear_vel[1], 3)
        tw.twist.linear.z = round(self.linear_vel[2], 3)
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
        ac.accel.linear.x = float(self.linear_accel[0])
        ac.accel.linear.y = float(self.linear_accel[1])
        ac.accel.linear.z = float(self.linear_accel[2])
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
                self.timer = self.create_timer(
                    1.0 / self.publish_rate, self.publish_twist)
            elif p.name == 'sphere_mass' and p.type_ == p.Type.DOUBLE:
                self.mass = float(p.value)
                self.update_inertia_and_drag(
                    self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'sphere_radius' and p.type_ == p.Type.DOUBLE:
                self.sphere_radius = float(p.value)
                self.update_inertia_and_drag(
                    self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'fluid_viscosity' and p.type_ == p.Type.DOUBLE:
                self.fluid_viscosity = float(p.value)
                self.update_inertia_and_drag(
                    self.mass, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'max_linear_speed' and p.type_ == p.Type.DOUBLE:
                self.max_linear_speed = float(p.value)
            elif p.name == 'max_angular_speed' and p.type_ == p.Type.DOUBLE:
                self.max_angular_speed = float(p.value)

        result = SetParametersResult()
        result.successful = True
        return result


def main(args=None):
    rclpy.init(args=args)
    node = AdmittanceControlCoupledPendulumNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
