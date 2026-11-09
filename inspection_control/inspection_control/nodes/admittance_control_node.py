#!/usr/bin/env python3
# coding: utf-8

import json
import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import SetParametersAtomically
from geometry_msgs.msg import WrenchStamped, TwistStamped, AccelStamped, PointStamped

from inspection_control.deadman import DeadmanMonitor
from inspection_control.virtual_sphere import (
    MISSING_MESSAGE, PARAMETERS as SPHERE_PARAMETERS, VirtualSphere, latched_qos, message_type)
from std_msgs.msg import String

# What ~/set_tuning may set: the parameters the lab's Foxglove bridge can reach
# (it admits ~/set_tuning, never ~/set_parameters). Only the virtual sphere: the
# speed clamps are safety limits that remote_teleop_gate lowers while a remote
# operator drives (and restores on disarm), and the deadman and teleop timeouts
# are interlocks, so those stay with the YAML, the CLI and the gate.
TUNABLE_PARAMETERS = frozenset(SPHERE_PARAMETERS)

# Values the node can't run on, refused before anything is applied (any setter).
# 0 still turns a clamp or a timeout off, as documented in the YAML.
NONNEGATIVE = ('max_linear_speed', 'max_angular_speed', 'teleop_timeout',
               'deadman_timeout', 'pivot_timeout')


def _finite_or_none(x, digits=5):
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None


class AdmittanceControlNode(Node):
    def __init__(self):
        super().__init__('admittance_control')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 10.0),
                # The virtual sphere (the plant). This node owns these: the
                # orientation controller and autofocus follow ~/virtual_sphere,
                # so set them here only. All must be > 0.
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

                # A teleop wrench older than this (s) counts as zero, so the arm
                # can't keep moving on the last command if teleop stops
                # publishing (crash, lost link). 0 disables the check.
                ('teleop_timeout', 0.5),

                # Deadman (std_msgs/Bool from teleop): no wrench applies at all
                # (teleop, orientation, autofocus) unless it is true and newer
                # than deadman_timeout (s). Empty topic = no deadman.
                ('deadman_topic', '/teleop/deadman'),
                ('deadman_timeout', 0.5),
            ]
        )

        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.sphere = VirtualSphere(
            *(float(self.get_parameter(n).value) for n in SPHERE_PARAMETERS))
        problem = self.sphere.problem()
        if problem is not None:
            raise ValueError(f'Invalid virtual sphere: {problem}')
        # Latched, so a controller that starts later still gets it at once.
        sphere_msg_type = message_type()
        if sphere_msg_type is None:
            self.get_logger().error(
                f'{MISSING_MESSAGE}. The orientation controller and autofocus get no '
                'plant and hold zero output until then; teleop is unaffected.')
            self.sphere_pub = None
        else:
            self.sphere_pub = self.create_publisher(
                sphere_msg_type, '~/virtual_sphere', latched_qos())
        self._publish_sphere()

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
        self.teleop_timeout = float(self.get_parameter('teleop_timeout').value)
        self._teleop_t = None  # receive time (s) of the last teleop wrench
        self._orient_t = None
        self._autofocus_t = None
        # Last integration step, for ~/status.
        self._cycle = {'mode': 'idle', 'pivot': None,
                       'clamped_linear': False, 'clamped_angular': False}
        self._teleop_stale = False

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
        self.deadman = DeadmanMonitor(
            self, str(self.get_parameter('deadman_topic').value),
            float(self.get_parameter('deadman_timeout').value), qos_profile)

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

        # set_parameters_atomically restricted to TUNABLE_PARAMETERS.
        self.create_service(SetParametersAtomically, '~/set_tuning', self._on_set_tuning)
        # ~/status (std_msgs/String, JSON, latched, 5 Hz): what the plant is doing.
        self.status_pub = self.create_publisher(String, '~/status', QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1))
        self.create_timer(0.2, self._publish_status)

    def _on_set_tuning(self, request, response):
        refused = sorted({p.name for p in request.parameters} - TUNABLE_PARAMETERS)
        if refused:
            response.result = SetParametersResult(
                successful=False,
                reason=f'not settable through ~/set_tuning: {", ".join(refused)}')
            return response
        try:
            response.result = self.set_parameters_atomically(
                [Parameter.from_parameter_msg(p) for p in request.parameters])
        except Exception as e:
            response.result = SetParametersResult(successful=False, reason=str(e))
        return response

    def _publish_status(self):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        engaged, deadman_age = self.deadman.state(now_s)

        def source(t, force, torque, timeout=0.0):
            age = None if t is None else now_s - t
            return {
                'age_s': _finite_or_none(age, 3),
                'stale': age is None or (timeout > 0.0 and age > timeout),
                'force': _finite_or_none(float(np.linalg.norm(force))),
                'torque': _finite_or_none(float(np.linalg.norm(torque))),
            }

        s, c = self.sphere, self._cycle
        status = {
            'engaged': bool(engaged),
            'deadman_topic': self.deadman.topic,
            'deadman_age_s': _finite_or_none(deadman_age, 3),
            'deadman_timeout': self.deadman.timeout,
            'mode': c['mode'],
            'pivot_m': _finite_or_none(c['pivot'], 4),
            'linear': [_finite_or_none(v) for v in self.linear_vel],
            'angular': [_finite_or_none(v) for v in self.angular_vel],
            'speed': _finite_or_none(float(np.linalg.norm(self.linear_vel))),
            'angular_speed': _finite_or_none(float(np.linalg.norm(self.angular_vel))),
            'clamped_linear': bool(c['clamped_linear']),
            'clamped_angular': bool(c['clamped_angular']),
            'inputs': {
                'teleop': source(
                    self._teleop_t, self.teleop_F, self.teleop_T, self.teleop_timeout),
                'orientation': source(self._orient_t, self.orient_F, self.orient_T),
                'autofocus': source(self._autofocus_t, self.autofocus_F, self.autofocus_T),
            },
            'max_linear_speed': self.max_linear_speed,
            'max_angular_speed': self.max_angular_speed,
            'teleop_timeout': self.teleop_timeout,
            'pivot_timeout': self.pivot_timeout,
            'publish_rate': self.publish_rate,
            'sphere': {
                'mass': s.mass, 'radius': s.radius, 'viscosity': s.viscosity,
                'inertia': _finite_or_none(s.inertia),
                'linear_drag': _finite_or_none(s.linear_drag),
                'angular_drag': _finite_or_none(s.angular_drag),
                'coast_s': _finite_or_none(s.coast_time(), 4),
                'published': self.sphere_pub is not None,
            },
            'swing_coast_s': (_finite_or_none(s.swing_coast_time(c['pivot']), 4)
                              if c['pivot'] else None),
        }
        self.status_pub.publish(String(data=json.dumps(status)))

    def _publish_sphere(self):
        if self.sphere_pub is not None:
            self.sphere_pub.publish(self.sphere.to_msg(self.get_clock().now().to_msg()))
        self.get_logger().info(f'Virtual sphere: {self.sphere.describe()}')

    def wrench_callback_teleop(self, msg: WrenchStamped):
        self.teleop_F[0], self.teleop_F[1], self.teleop_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.teleop_T[0], self.teleop_T[1], self.teleop_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self._teleop_t = self.get_clock().now().nanoseconds * 1e-9
        self.have_any_wrench = True

    def wrench_callback_orientation(self, msg: WrenchStamped):
        self.orient_F[0], self.orient_F[1], self.orient_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.orient_T[0], self.orient_T[1], self.orient_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self._orient_t = self.get_clock().now().nanoseconds * 1e-9
        self.have_any_wrench = True

    def wrench_callback_autofocus(self, msg: WrenchStamped):
        self.autofocus_F[0], self.autofocus_F[1], self.autofocus_F[2] = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self.autofocus_T[0], self.autofocus_T[1], self.autofocus_T[2] = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        self._autofocus_t = self.get_clock().now().nanoseconds * 1e-9
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

        # A stale teleop wrench counts as zero (teleop stopped publishing).
        now_s = now.nanoseconds * 1e-9
        teleop_stale = (self.teleop_timeout > 0.0 and self._teleop_t is not None
                        and (now_s - self._teleop_t) > self.teleop_timeout)
        if teleop_stale:
            self.teleop_F = [0.0, 0.0, 0.0]
            self.teleop_T = [0.0, 0.0, 0.0]
            if not self._teleop_stale:
                self.get_logger().warn(
                    f'No teleop wrench for {now_s - self._teleop_t:.2f} s: ignoring teleop until it is back')
        elif self._teleop_stale:
            self.get_logger().info('Teleop wrench is back')
        self._teleop_stale = teleop_stale

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

        # Deadman released (or teleop silent): no effort at all; the plant
        # coasts to rest on its drag.
        if not self.deadman.engaged(now_s):
            F[:] = 0.0
            T[:] = 0.0

        v = np.array(self.linear_vel, dtype=float)
        w = np.array(self.angular_vel, dtype=float)
        s = self.sphere
        m, c, I_B, c_ang = s.mass, s.linear_drag, s.inertia, s.angular_drag

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

            I_A = s.swing_inertia(d)                     # parallel-axis swing inertia

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
        unclamped = (list(self.linear_vel), list(self.angular_vel))
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

        self._cycle = {
            'mode': 'pendulum' if pivot_fresh else 'point_mass',
            'pivot': d if pivot_fresh else None,
            'clamped_linear': self.linear_vel != unclamped[0],
            'clamped_angular': self.angular_vel != unclamped[1],
        }

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
        # The sphere first, whole: refuse the update before applying anything
        # if it would make the plant invalid (the controllers' pole placement
        # takes square roots of these).
        sphere_values = dict(zip(
            SPHERE_PARAMETERS, (self.sphere.mass, self.sphere.radius, self.sphere.viscosity)))
        for p in params:
            if p.name in sphere_values:
                sphere_values[p.name] = float(p.value)
        sphere = VirtualSphere(*(sphere_values[n] for n in SPHERE_PARAMETERS))
        problem = sphere.problem()
        if problem is None:
            for p in params:
                if p.name == 'publish_rate' and not (math.isfinite(p.value) and p.value > 0.0):
                    problem = f'publish_rate must be > 0 (got {p.value})'
                elif p.name in NONNEGATIVE and not (math.isfinite(p.value) and p.value >= 0.0):
                    problem = f'{p.name} must be >= 0 (0 turns it off; got {p.value})'
        if problem is not None:
            return SetParametersResult(successful=False, reason=problem)

        for p in params:
            if p.name == 'publish_rate' and p.type_ == p.Type.DOUBLE:
                self.publish_rate = float(p.value)
                self.timer.cancel()
                self.timer = self.create_timer(
                    1.0 / self.publish_rate, self.publish_twist)
            elif p.name == 'max_linear_speed' and p.type_ == p.Type.DOUBLE:
                self.max_linear_speed = float(p.value)
            elif p.name == 'max_angular_speed' and p.type_ == p.Type.DOUBLE:
                self.max_angular_speed = float(p.value)
            elif p.name == 'teleop_timeout' and p.type_ == p.Type.DOUBLE:
                self.teleop_timeout = float(p.value)
            elif p.name == 'deadman_timeout' and p.type_ == p.Type.DOUBLE:
                self.deadman.timeout = float(p.value)
            elif p.name == 'pivot_timeout' and p.type_ == p.Type.DOUBLE:
                self.pivot_timeout = float(p.value)

        if sphere != self.sphere:
            self.sphere = sphere
            self._publish_sphere()

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
