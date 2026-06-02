#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import json
import os
from collections import deque
from datetime import datetime
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import WrenchStamped, TwistStamped
from moveit_msgs.msg import ServoStatus
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

try:
    import PyKDL
    from urdf_parser_py.urdf import URDF as _URDF
    _KDL_AVAILABLE = True
    _KDL_IMPORT_ERROR = None
except ImportError as e:
    PyKDL = None
    _URDF = None
    _KDL_AVAILABLE = False
    _KDL_IMPORT_ERROR = str(e)


def _kdl_frame_from_origin(origin):
    if origin is None:
        return PyKDL.Frame()
    xyz = list(origin.xyz) if origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = list(origin.rpy) if origin.rpy is not None else [0.0, 0.0, 0.0]
    return PyKDL.Frame(PyKDL.Rotation.RPY(*rpy), PyKDL.Vector(*xyz))


def _kdl_joint_from_urdf(jnt):
    f = _kdl_frame_from_origin(jnt.origin)
    if jnt.joint_type == 'fixed':
        return PyKDL.Joint(jnt.name, PyKDL.Joint.Fixed)
    axis_arr = list(jnt.axis) if jnt.axis is not None else [1.0, 0.0, 0.0]
    axis = f.M * PyKDL.Vector(*axis_arr)
    type_map = {
        'revolute':   PyKDL.Joint.RotAxis,
        'continuous': PyKDL.Joint.RotAxis,
        'prismatic':  PyKDL.Joint.TransAxis,
    }
    if jnt.joint_type in type_map:
        return PyKDL.Joint(jnt.name, f.p, axis, type_map[jnt.joint_type])
    return PyKDL.Joint(jnt.name, PyKDL.Joint.Fixed)


def _kdl_tree_from_urdf_string(xml_string):
    """Inline replacement for kdl_parser_py.urdf.treeFromString.

    Returns (ok, tree, link_names) — link_names is the full list from the
    URDF so callers can produce a useful error message when getChain fails.
    """
    model = _URDF.from_xml_string(xml_string)
    link_names = sorted(model.link_map.keys())
    tree = PyKDL.Tree(model.get_root())

    def add_children(parent_name):
        for joint_name, child_name in model.child_map.get(parent_name, []):
            urdf_joint = model.joint_map[joint_name]
            kdl_joint = _kdl_joint_from_urdf(urdf_joint)
            f_parent_jnt = _kdl_frame_from_origin(urdf_joint.origin)
            kdl_segment = PyKDL.Segment(
                child_name, kdl_joint, f_parent_jnt, PyKDL.RigidBodyInertia())
            if not tree.addSegment(kdl_segment, parent_name):
                return False
            if not add_children(child_name):
                return False
        return True
    ok = add_children(model.get_root())
    return ok, tree, link_names


SERVO_STATUS_NAMES = {
    0: 'NO_WARNING',
    1: 'DECELERATE_APPROACHING_SINGULARITY',
    2: 'HALT_FOR_SINGULARITY',
    3: 'DECELERATE_LEAVING_SINGULARITY',
    4: 'DECELERATE_FOR_COLLISION',
    5: 'HALT_FOR_COLLISION',
    6: 'JOINT_BOUND',
    7: 'HALT_FOR_VELOCITY_LIMIT',
}

matplotlib.use('Agg')


class ServoLoggerNode(Node):
    def __init__(self):
        super().__init__('servo_logger')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('teleop_wrench_topic', '/teleop/wrench_cmds'),
                ('orient_wrench_topic', '/orientation_controller/wrench_cmds'),
                ('servo_twist_topic', '/servo_node/delta_twist_cmds'),
                ('joint_vel_topic', '/ur5e_forward_velocity_controller/commands'),
                ('joint_states_topic', '/joint_states'),
                ('servo_status_topic', '/servo_node/status'),
                ('robot_description_topic', '/robot_description'),
                ('base_frame', 'object_frame'),
                ('tip_frame', 'eoat_camera_link'),
                ('buffer_seconds', 10.0),
                ('output_dir', '/data'),
                ('trigger_service', '/servo_logger/save_plot'),
            ]
        )

        self.buffer_seconds = float(self.get_parameter('buffer_seconds').value)
        self.output_dir = str(self.get_parameter('output_dir').value)

        teleop_topic = str(self.get_parameter('teleop_wrench_topic').value)
        orient_topic = str(self.get_parameter('orient_wrench_topic').value)
        twist_topic = str(self.get_parameter('servo_twist_topic').value)
        jvel_topic = str(self.get_parameter('joint_vel_topic').value)
        jstate_topic = str(self.get_parameter('joint_states_topic').value)
        status_topic = str(self.get_parameter('servo_status_topic').value)
        urdf_topic = str(self.get_parameter('robot_description_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.tip_frame = str(self.get_parameter('tip_frame').value)
        trigger_name = str(self.get_parameter('trigger_service').value)

        self.lock = Lock()
        self.teleop_buf = deque()
        self.orient_buf = deque()
        self.twist_buf = deque()
        self.jvel_buf = deque()
        self.jstate_buf = deque()
        self.jstate_names = []
        self.status_buf = deque()
        self.actual_twist_buf = deque()
        self.jac_buf = deque()

        self.kdl_chain = None
        self.jac_solver = None
        self.fk_solver = None
        self.chain_joint_names = []

        # BEST_EFFORT matches both RELIABLE and BEST_EFFORT publishers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=50,
        )

        self.create_subscription(
            WrenchStamped, teleop_topic, self._on_teleop_wrench, qos)
        self.create_subscription(
            WrenchStamped, orient_topic, self._on_orient_wrench, qos)
        self.create_subscription(
            TwistStamped, twist_topic, self._on_twist, qos)
        self.create_subscription(
            Float64MultiArray, jvel_topic, self._on_joint_vel, qos)
        self.create_subscription(
            JointState, jstate_topic, self._on_joint_state, qos)
        self.create_subscription(
            ServoStatus, status_topic, self._on_servo_status, qos)

        if _KDL_AVAILABLE:
            # /robot_description is latched: TRANSIENT_LOCAL to grab it once.
            urdf_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            )
            self.create_subscription(
                String, urdf_topic, self._on_robot_description, urdf_qos)
        else:
            self.get_logger().warn(
                f"actual-twist (Jacobian) computation disabled: {_KDL_IMPORT_ERROR}. "
                "Install with `apt install ros-jazzy-urdfdom-py python3-pykdl` "
                "to enable.")

        self.save_srv = self.create_service(
            Trigger, trigger_name, self._on_save_trigger)

        self.get_logger().info(
            f"servo_logger up — buffer={self.buffer_seconds:.1f}s, "
            f"output_dir={self.output_dir}, trigger={trigger_name}")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _trim(self, buf: deque):
        cutoff = self._now() - self.buffer_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _store_wrench(self, buf: deque, msg: WrenchStamped):
        t = self._now()
        with self.lock:
            buf.append((
                t,
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            ))
            self._trim(buf)

    def _on_teleop_wrench(self, msg: WrenchStamped):
        self._store_wrench(self.teleop_buf, msg)

    def _on_orient_wrench(self, msg: WrenchStamped):
        self._store_wrench(self.orient_buf, msg)

    def _on_twist(self, msg: TwistStamped):
        t = self._now()
        with self.lock:
            self.twist_buf.append((
                t,
                msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z,
            ))
            self._trim(self.twist_buf)

    def _on_joint_vel(self, msg: Float64MultiArray):
        t = self._now()
        with self.lock:
            self.jvel_buf.append((t, list(msg.data)))
            self._trim(self.jvel_buf)

    def _on_robot_description(self, msg: String):
        if self.kdl_chain is not None:
            return
        try:
            ok, tree, link_names = _kdl_tree_from_urdf_string(msg.data)
        except Exception as e:
            self.get_logger().error(f'URDF→KDL conversion failed: {e}')
            return
        if not ok:
            self.get_logger().error('failed to build KDL tree from /robot_description')
            return
        self._urdf_link_names = link_names
        try:
            chain = tree.getChain(self.base_frame, self.tip_frame)
        except Exception as e:
            self.get_logger().error(
                f'getChain({self.base_frame} -> {self.tip_frame}) failed: {e}')
            return

        names = []
        for i in range(chain.getNrOfSegments()):
            joint = chain.getSegment(i).getJoint()
            if joint.getType() != PyKDL.Joint.Fixed:
                names.append(joint.getName())

        if not names:
            # Chain came back empty — most likely base_frame or tip_frame
            # isn't a link in the URDF, or there's no actuated path between
            # them. Dump the available link names so the user can fix the
            # parameter without re-reading the URDF by hand.
            self.get_logger().error(
                f'KDL chain {self.base_frame} -> {self.tip_frame} has 0 joints. '
                f'Check that both link names exist in the URDF and there is an '
                f'actuated path between them. '
                f'Available links: {self._urdf_link_names}')
            return

        self.kdl_chain = chain
        self.jac_solver = PyKDL.ChainJntToJacSolver(chain)
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
        self.chain_joint_names = names
        self.get_logger().info(
            f'KDL chain {self.base_frame} -> {self.tip_frame}, joints={names}')

    def _compute_actual_twist(self, name_to_pos, name_to_vel):
        """Returns (twist_in_tip_frame_6, singular_values_sorted_desc) or None."""
        if self.kdl_chain is None:
            return None
        n = len(self.chain_joint_names)
        if n == 0:
            return None
        q = PyKDL.JntArray(n)
        qdot = np.zeros(n)
        for i, jn in enumerate(self.chain_joint_names):
            if jn not in name_to_pos:
                return None
            q[i] = float(name_to_pos[jn])
            qdot[i] = float(name_to_vel.get(jn, 0.0))

        jac = PyKDL.Jacobian(n)
        self.jac_solver.JntToJac(q, jac)
        J = np.empty((6, n))
        for r in range(6):
            for c in range(n):
                J[r, c] = jac[r, c]

        # KDL gives twist of tip with reference point at tip, expressed in
        # the chain base frame. Rotate into the tip frame so it's directly
        # comparable to admittance output (frame_id = eoat_camera_link).
        F_tip = PyKDL.Frame()
        self.fk_solver.JntToCart(q, F_tip)
        R = np.array([[F_tip.M[i, j] for j in range(3)] for i in range(3)])
        twist_base = J @ qdot
        v_tip = R.T @ twist_base[0:3]
        w_tip = R.T @ twist_base[3:6]

        # Singular values of J (sorted descending) — diagnostic for
        # rank deficiency and ill-conditioning that servo's IK has to fight.
        sv = np.linalg.svd(J, compute_uv=False)
        return np.concatenate([v_tip, w_tip]), sv

    def _on_joint_state(self, msg: JointState):
        t = self._now()
        positions = list(msg.position)
        velocities = list(msg.velocity)
        # Build name->value maps before acquiring the lock so we don't hold it
        # over the (small but non-trivial) KDL call.
        name_to_pos = dict(zip(msg.name, positions))
        name_to_vel = dict(zip(msg.name, velocities))
        result = self._compute_actual_twist(name_to_pos, name_to_vel)
        with self.lock:
            if msg.name and not self.jstate_names:
                self.jstate_names = list(msg.name)
            self.jstate_buf.append((t, positions, velocities))
            self._trim(self.jstate_buf)
            if result is not None:
                actual, sv = result
                self.actual_twist_buf.append((
                    t,
                    float(actual[0]), float(actual[1]), float(actual[2]),
                    float(actual[3]), float(actual[4]), float(actual[5]),
                ))
                self._trim(self.actual_twist_buf)
                self.jac_buf.append((t, [float(s) for s in sv]))
                self._trim(self.jac_buf)

    def _on_servo_status(self, msg: ServoStatus):
        t = self._now()
        with self.lock:
            self.status_buf.append((t, int(msg.code), str(msg.message)))
            self._trim(self.status_buf)

    def _on_save_trigger(self, request, response):
        try:
            png_path, json_path = self._save_snapshot()
            response.success = True
            response.message = f"saved {png_path} and {json_path}"
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"save failed: {e}"
            self.get_logger().error(response.message)
        return response

    def _save_snapshot(self) -> tuple:
        os.makedirs(self.output_dir, exist_ok=True)
        with self.lock:
            tel = list(self.teleop_buf)
            ori = list(self.orient_buf)
            tw = list(self.twist_buf)
            jv = list(self.jvel_buf)
            js = list(self.jstate_buf)
            js_names = list(self.jstate_names)
            st = list(self.status_buf)
            at = list(self.actual_twist_buf)
            jb = list(self.jac_buf)

        starts = []
        for buf in (tel, ori, tw, jv, js, st, at, jb):
            if buf:
                starts.append(buf[0][0])
        if not starts:
            raise RuntimeError("all buffers are empty — nothing to save")
        t0 = min(starts)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = os.path.join(self.output_dir, f'servo_log_{ts}.png')
        json_path = os.path.join(self.output_dir, f'servo_log_{ts}.json')

        self._write_json(json_path, ts, t0, tel, ori,
                         tw, jv, js, js_names, st, at, jb)
        self._write_plot(png_path, ts, t0, tel, ori,
                         tw, jv, js, js_names, st, at, jb)
        return png_path, json_path

    def _write_json(self, path, ts, t0, tel, ori, tw, jv, js, js_names, st, at, jb):
        def wrench_cols(buf):
            return {
                't':  [r[0] - t0 for r in buf],
                'fx': [r[1] for r in buf],
                'fy': [r[2] for r in buf],
                'fz': [r[3] for r in buf],
                'tx': [r[4] for r in buf],
                'ty': [r[5] for r in buf],
                'tz': [r[6] for r in buf],
            }

        twist_cols = {
            't':  [r[0] - t0 for r in tw],
            'vx': [r[1] for r in tw],
            'vy': [r[2] for r in tw],
            'vz': [r[3] for r in tw],
            'wx': [r[4] for r in tw],
            'wy': [r[5] for r in tw],
            'wz': [r[6] for r in tw],
        }

        n_cmd = max((len(r[1]) for r in jv), default=0)
        jvel_cmd_cols = {'t': [r[0] - t0 for r in jv]}
        for j in range(n_cmd):
            jvel_cmd_cols[f'j{j}'] = [
                r[1][j] if len(r[1]) > j else None for r in jv
            ]

        n_pos = max((len(r[1]) for r in js), default=0)
        n_vel = max((len(r[2]) for r in js), default=0)
        jstate_cols = {
            't': [r[0] - t0 for r in js],
            'names': js_names,
            'position': {},
            'velocity': {},
        }
        for j in range(n_pos):
            label = js_names[j] if j < len(js_names) else f'j{j}'
            jstate_cols['position'][label] = [
                r[1][j] if len(r[1]) > j else None for r in js
            ]
        for j in range(n_vel):
            label = js_names[j] if j < len(js_names) else f'j{j}'
            jstate_cols['velocity'][label] = [
                r[2][j] if len(r[2]) > j else None for r in js
            ]

        status_cols = {
            't':       [r[0] - t0 for r in st],
            'code':    [r[1] for r in st],
            'name':    [SERVO_STATUS_NAMES.get(r[1], f'code={r[1]}') for r in st],
            'message': [r[2] for r in st],
        }

        actual_twist_cols = {
            'frame_id': self.tip_frame,
            't':  [r[0] - t0 for r in at],
            'vx': [r[1] for r in at],
            'vy': [r[2] for r in at],
            'vz': [r[3] for r in at],
            'wx': [r[4] for r in at],
            'wy': [r[5] for r in at],
            'wz': [r[6] for r in at],
        }

        # Jacobian singular values + condition number.
        # cond = sigma_max / sigma_min; if sigma_min == 0 we report None.
        n_sv = max((len(r[1]) for r in jb), default=0)
        jac_cols = {
            't': [r[0] - t0 for r in jb],
            'sigma': {f's{i}': [r[1][i] if len(r[1]) > i else None
                                for r in jb] for i in range(n_sv)},
            'sigma_min': [(r[1][-1] if r[1] else None) for r in jb],
            'sigma_max': [(r[1][0] if r[1] else None) for r in jb],
            'cond': [
                (r[1][0] / r[1][-1]) if (r[1] and r[1][-1] > 1e-12) else None
                for r in jb
            ],
        }

        payload = {
            'saved_at': ts,
            'buffer_seconds': self.buffer_seconds,
            't0_epoch_sec': t0,
            'time_units': 'seconds relative to t0',
            'teleop_wrench': wrench_cols(tel),
            'orient_wrench': wrench_cols(ori),
            'servo_twist': twist_cols,
            'joint_vel_cmd': jvel_cmd_cols,
            'joint_states': jstate_cols,
            'servo_status': status_cols,
            'actual_twist': actual_twist_cols,
            'jacobian': jac_cols,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    def _write_plot(self, png_path, ts, t0, tel, ori, tw, jv, js, js_names, st, at, jb):
        fig, axes = plt.subplots(7, 1, figsize=(14, 19), sharex=True)
        (ax_force, ax_torque, ax_twist, ax_jvel,
         ax_jvel_act, ax_jpos, ax_jac) = axes

        # 1. Wrench forces
        if tel:
            t = [r[0] - t0 for r in tel]
            ax_force.plot(t, [r[1] for r in tel], label='teleop Fx')
            ax_force.plot(t, [r[2] for r in tel], label='teleop Fy')
            ax_force.plot(t, [r[3] for r in tel], label='teleop Fz')
        if ori:
            t = [r[0] - t0 for r in ori]
            ax_force.plot(t, [r[1]
                          for r in ori], label='orient Fx', linestyle='--')
            ax_force.plot(t, [r[2]
                          for r in ori], label='orient Fy', linestyle='--')
            ax_force.plot(t, [r[3]
                          for r in ori], label='orient Fz', linestyle='--')
        ax_force.set_ylabel('force (N)')
        ax_force.set_title('wrench inputs')
        ax_force.legend(loc='upper right', fontsize=8, ncol=2)
        ax_force.grid(True, alpha=0.3)

        # 2. Wrench torques
        if tel:
            t = [r[0] - t0 for r in tel]
            ax_torque.plot(t, [r[4] for r in tel], label='teleop Tx')
            ax_torque.plot(t, [r[5] for r in tel], label='teleop Ty')
            ax_torque.plot(t, [r[6] for r in tel], label='teleop Tz')
        if ori:
            t = [r[0] - t0 for r in ori]
            ax_torque.plot(t, [r[4] for r in ori],
                           label='orient Tx', linestyle='--')
            ax_torque.plot(t, [r[5] for r in ori],
                           label='orient Ty', linestyle='--')
            ax_torque.plot(t, [r[6] for r in ori],
                           label='orient Tz', linestyle='--')
        ax_torque.set_ylabel('torque (N·m)')
        ax_torque.legend(loc='upper right', fontsize=8, ncol=2)
        ax_torque.grid(True, alpha=0.3)

        # 3. Admittance output twist (commanded, solid/dashed)
        #    vs actual EE twist from J(q)·q_dot in tip frame (dotted).
        colors = ['tab:blue', 'tab:orange', 'tab:green',
                  'tab:red', 'tab:purple', 'tab:brown']
        labels = ['Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz']
        if tw:
            t = [r[0] - t0 for r in tw]
            for i, lbl in enumerate(labels):
                style = '-' if i < 3 else '--'
                ax_twist.plot(t, [r[1 + i] for r in tw],
                              label=f'{lbl} cmd', color=colors[i], linestyle=style)
        if at:
            t = [r[0] - t0 for r in at]
            for i, lbl in enumerate(labels):
                ax_twist.plot(t, [r[1 + i] for r in at],
                              label=f'{lbl} act', color=colors[i],
                              linestyle=':', linewidth=1.2, alpha=0.9)
        ax_twist.set_ylabel('twist (m/s, rad/s)')
        ax_twist.set_title(
            f'admittance output (cmd) vs actual EE twist (act) — '
            f'frame: {self.tip_frame}')
        ax_twist.legend(loc='upper right', fontsize=7, ncol=4)
        ax_twist.grid(True, alpha=0.3)

        # 4. Joint velocity commands (servo output)
        if jv:
            t = [r[0] - t0 for r in jv]
            n_joints = max(len(r[1]) for r in jv)
            for j in range(n_joints):
                ax_jvel.plot(
                    t,
                    [r[1][j] if len(r[1]) > j else 0.0 for r in jv],
                    label=f'j{j}',
                )
        ax_jvel.set_ylabel('cmd vel (rad/s)')
        ax_jvel.set_title('servo output → controller (commanded)')
        ax_jvel.legend(loc='upper right', fontsize=8, ncol=3)
        ax_jvel.grid(True, alpha=0.3)

        # 5. Actual joint velocities from /joint_states
        if js:
            t = [r[0] - t0 for r in js]
            n_joints = max(len(r[2]) for r in js) if any(r[2]
                                                         for r in js) else 0
            for j in range(n_joints):
                label = js_names[j] if j < len(js_names) else f'j{j}'
                ax_jvel_act.plot(
                    t,
                    [r[2][j] if len(r[2]) > j else 0.0 for r in js],
                    label=label,
                )
        ax_jvel_act.set_ylabel('actual vel (rad/s)')
        ax_jvel_act.set_title('joint_states velocity (measured)')
        ax_jvel_act.legend(loc='upper right', fontsize=8, ncol=3)
        ax_jvel_act.grid(True, alpha=0.3)

        # 6. Joint positions from /joint_states
        if js:
            t = [r[0] - t0 for r in js]
            n_joints = max(len(r[1]) for r in js) if any(r[1]
                                                         for r in js) else 0
            for j in range(n_joints):
                label = js_names[j] if j < len(js_names) else f'j{j}'
                ax_jpos.plot(
                    t,
                    [r[1][j] if len(r[1]) > j else 0.0 for r in js],
                    label=label,
                )
        ax_jpos.set_ylabel('position (rad)')
        ax_jpos.set_title('joint_states position (measured)')
        ax_jpos.legend(loc='upper right', fontsize=8, ncol=3)
        ax_jpos.grid(True, alpha=0.3)

        # 7. Jacobian singular values (log scale) + condition number.
        # A collapsing smallest singular value or a spiking condition number
        # at a moment when actual ≠ commanded is the smoking gun for
        # rank-deficient IK at that configuration.
        if jb:
            t = [r[0] - t0 for r in jb]
            n_sv = max(len(r[1]) for r in jb)
            for i in range(n_sv):
                ax_jac.plot(
                    t,
                    [r[1][i] if len(r[1]) > i else float('nan') for r in jb],
                    label=f'σ{i}',
                )
            cond = [
                (r[1][0] / r[1][-1]) if (r[1] and r[1][-1] > 1e-12)
                else float('nan')
                for r in jb
            ]
            ax_jac.plot(t, cond, label='cond(J)', color='black',
                        linewidth=1.3, linestyle='--')
        ax_jac.set_yscale('log')
        ax_jac.set_ylabel('singular values / cond')
        ax_jac.set_xlabel('time (s)')
        ax_jac.set_title('Jacobian SVD (σ0..σ5 desc; cond=σ_max/σ_min)')
        ax_jac.legend(loc='upper right', fontsize=8, ncol=4)
        ax_jac.grid(True, alpha=0.3, which='both')

        # Servo status transitions: mark every change into a non-zero code.
        # Vline spans all panels (axvline per axis); annotation only on top panel.
        prev_code = 0
        for t_ev, code, msg in st:
            if code != prev_code and code != 0:
                x = t_ev - t0
                for ax in axes:
                    ax.axvline(x, color='red', alpha=0.4, linewidth=1.0)
                name = SERVO_STATUS_NAMES.get(code, f'code={code}')
                ax_force.annotate(
                    name,
                    xy=(x, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(2, -2), textcoords='offset points',
                    rotation=90, va='top', ha='left',
                    fontsize=7, color='red',
                )
            prev_code = code

        fig.suptitle(f'servo log — {ts}  (last {self.buffer_seconds:.0f}s)')
        fig.tight_layout()
        fig.savefig(png_path, dpi=110)
        plt.close(fig)


def main(args=None):
    rclpy.init(args=args)
    node = ServoLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
