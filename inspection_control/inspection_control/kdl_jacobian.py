#!/usr/bin/env python3
"""Shared KDL kinematics helpers.

Builds a serial chain from a URDF string (typically the latched
`/robot_description`) and computes the geometric Jacobian + forward-kinematics
rotation for a base->tip link pair, without depending on MoveIt.

Used by both servo_logger_node and orientation_control_node so the two stay in
sync. Import guard mirrors the optional nature of the PyKDL / urdf_parser_py
system packages.
"""

import numpy as np

try:
    import PyKDL
    from urdf_parser_py.urdf import URDF as _URDF
    KDL_AVAILABLE = True
    KDL_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - depends on system packages
    PyKDL = None
    _URDF = None
    KDL_AVAILABLE = False
    KDL_IMPORT_ERROR = str(e)


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


def kdl_tree_from_urdf_string(xml_string):
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


class KdlChain:
    """A base->tip serial chain with Jacobian + FK solvers.

    Raises RuntimeError if the URDF cannot be parsed or the requested chain has
    no actuated joints (usually a bad base/tip frame name).
    """

    def __init__(self, urdf_xml: str, base_frame: str, tip_frame: str):
        if not KDL_AVAILABLE:
            raise RuntimeError(f"PyKDL/urdf_parser_py unavailable: {KDL_IMPORT_ERROR}")

        ok, tree, link_names = kdl_tree_from_urdf_string(urdf_xml)
        if not ok:
            raise RuntimeError("failed to build KDL tree from URDF")
        self.link_names = link_names

        chain = tree.getChain(base_frame, tip_frame)
        names = []
        for i in range(chain.getNrOfSegments()):
            joint = chain.getSegment(i).getJoint()
            if joint.getType() != PyKDL.Joint.Fixed:
                names.append(joint.getName())
        if not names:
            raise RuntimeError(
                f"KDL chain {base_frame} -> {tip_frame} has 0 joints. "
                f"Check both link names exist and there is an actuated path "
                f"between them. Available links: {link_names}")

        self.base_frame = base_frame
        self.tip_frame = tip_frame
        self.chain = chain
        self.joint_names = names
        self.jac_solver = PyKDL.ChainJntToJacSolver(chain)
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)

    def compute(self, name_to_pos: dict, name_to_vel: dict):
        """Return (J, R_base_tip, qdot) or None if a chain joint is missing.

        J          : (6, N) geometric Jacobian expressed in the base frame,
                     reference point at the tip origin (KDL convention).
        R_base_tip : (3, 3) rotation of the tip frame in the base frame (FK).
        qdot       : (N,) joint velocities ordered as self.joint_names.
        """
        n = len(self.joint_names)
        q = PyKDL.JntArray(n)
        qdot = np.zeros(n)
        for i, jn in enumerate(self.joint_names):
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

        F_tip = PyKDL.Frame()
        self.fk_solver.JntToCart(q, F_tip)
        R = np.array([[F_tip.M[i, j] for j in range(3)] for i in range(3)])
        return J, R, qdot
