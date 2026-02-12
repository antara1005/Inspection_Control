#!/usr/bin/env python3
import math
import numpy as np
import cv2
import open3d as o3d
import numpy.linalg as LA
import copy
import os
import threading

from scipy.linalg import expm

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import WrenchStamped, TwistStamped

from sensor_msgs.msg import Joy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, Float64
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from viewpoint_generation_interfaces.msg import OrientationControlData
from geometry_msgs.msg import PointStamped, Point, Vector3

from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import serialize_message

from tf2_ros import Buffer, TransformListener, TransformException, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from inspection_control.debag_orientation_data import debag


class AngleKalmanFilter:
    """
    2D Kalman filter with true linear plant dynamics:

      x = [angle, dangle]^T
      x_{k+1} = Ad x_k + Bd u_k + w_k
      z_k = H x_k + v_k,   H = [1 0]

    Continuous-time model used to build (Ad, Bd):
      angle_dot  = dangle
      dangle_dot = -(b/I_A) dangle + (1/I_A) u
    => A = [[0, 1],
            [0, -b/I_A]]
       B = [[0],
            [1/I_A]]
    """

    def __init__(self, R: float, Q_angle: float, Q_dangle: float):
        """
        Initialize Kalman filter with noise parameters.

        Args:
            R: Measurement noise variance (rad²)
            Q_angle: Process noise variance for angle (rad²)
            Q_dangle: Process noise variance for dangle (rad²/s²)
        """
        self.x = np.zeros(2, dtype=np.float64)
        self.P = np.diag([(50 * math.pi / 180) ** 2, (50.0 * math.pi / 180) ** 2]).astype(np.float64)

        self.R = float(R)
        self.Q = np.diag([float(Q_angle), float(Q_dangle)]).astype(np.float64)

        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        self._initialized = False

        # Cached discretization to avoid expm every time when dt & model unchanged
        self._last_dt = None
        self._last_I_A = None
        self._last_b = None
        self._Ad = np.eye(2, dtype=np.float64)
        self._Bd = np.zeros((2, 1), dtype=np.float64)

    def reset(self, angle_init: float = 0.0, dangle_init: float = 0.0):
        """Reset filter state."""
        self.x = np.array([angle_init, dangle_init], dtype=np.float64)
        self.P = np.diag([(0.5 * math.pi / 180) ** 2, (50.0 * math.pi / 180) ** 2]).astype(np.float64)
        self._initialized = True
        self._last_dt = None  # Force refresh discretization next predict

    def set_noise(self, R: float = None, Q_angle: float = None, Q_dangle: float = None):
        """Update noise parameters."""
        if R is not None:
            self.R = float(R)
        if Q_angle is not None or Q_dangle is not None:
            qa = self.Q[0, 0] if Q_angle is None else float(Q_angle)
            qd = self.Q[1, 1] if Q_dangle is None else float(Q_dangle)
            self.Q = np.diag([qa, qd]).astype(np.float64)

    def _discretize(self, dt: float, I_A: float, b: float):
        """
        Exact discretization using augmented expm:
            Md = expm([A B; 0 0]*dt)
            Ad = Md(0:2,0:2), Bd = Md(0:2,2)
        """
        A = np.array([[0.0, 1.0],
                      [0.0, -b / max(1e-12, I_A)]], dtype=np.float64)
        B = np.array([[0.0],
                      [1.0 / max(1e-12, I_A)]], dtype=np.float64)

        M = np.zeros((3, 3), dtype=np.float64)
        M[0:2, 0:2] = A
        M[0:2, 2:3] = B

        Md = expm(M * dt)
        self._Ad = Md[0:2, 0:2]
        self._Bd = Md[0:2, 2:3]

        self._last_dt = dt
        self._last_I_A = I_A
        self._last_b = b

    def predict(self, dt: float, u_prev: float, I_A: float, b: float):
        """
        Prediction step: propagate state forward using plant dynamics.
        """
        if dt <= 0:
            return

        if (self._last_dt != dt) or (self._last_I_A != I_A) or (self._last_b != b):
            self._discretize(dt, I_A, b)

        u = float(u_prev)

        self.x = self._Ad @ self.x + (self._Bd.flatten() * u)
        self.P = self._Ad @ self.P @ self._Ad.T + self.Q

    def update(self, z_angle: float):
        """
        Update step: incorporate measurement.
        """
        if not self._initialized:
            self.reset(angle_init=float(z_angle), dangle_init=0.0)
            return self.x[0], self.x[1]

        z = float(z_angle)

        y = z - (self.H @ self.x)[0]
        S = (self.H @ self.P @ self.H.T)[0, 0] + self.R
        K = (self.P @ self.H.T) / max(1e-12, S)

        self.x = self.x + (K[:, 0] * y)

        I = np.eye(2, dtype=np.float64)
        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K * self.R * K.T

        return self.x[0], self.x[1]

    def predict_and_update(self, z_angle: float, dt: float, u_prev: float, I_A: float, b: float):
        self.predict(dt, u_prev=u_prev, I_A=I_A, b=b)
        x0, x1 = self.update(z_angle)
        return x0, x1

    @property
    def angle(self) -> float:
        return float(self.x[0])

    @property
    def dangle(self) -> float:
        return float(self.x[1])


def _quat_to_R_xyzw(x, y, z, w):
    """Return 3x3 rotation matrix from xyzw quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ], dtype=np.float32)


def _R_to_quat_xyzw(R: np.ndarray) -> Quaternion:
    """Return xyzw quaternion from 3x3 rotation matrix R."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return Quaternion(x=qx, y=qy, z=qz, w=qw)


def make_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    points_xyz: (N,3) float32 array in meters.
    """
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = int(points_xyz.shape[0])
    msg.is_bigendian = False
    msg.is_dense = True
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points_xyz.astype(np.float32).tobytes()
    return msg


def _visualize_normal_estimation(pts_np: np.ndarray, centroid: np.ndarray, normal: np.ndarray,
                                 inlier_mask: np.ndarray = None, title: str = 'Normal Estimation'):
    geometries = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)

    if inlier_mask is not None:
        colors = np.zeros((len(pts_np), 3))
        colors[inlier_mask] = [0.0, 0.8, 0.0]
        colors[~inlier_mask] = [0.8, 0.0, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.0, 0.8, 0.0])

    geometries.append(pcd)

    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    centroid_sphere.translate(centroid)
    centroid_sphere.paint_uniform_color([0.0, 0.0, 1.0])
    geometries.append(centroid_sphere)

    arrow_length = 0.05
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.002,
        cone_radius=0.004,
        cylinder_height=arrow_length * 0.8,
        cone_height=arrow_length * 0.2
    )
    z_axis = np.array([0.0, 0.0, 1.0])
    rotation_axis = np.cross(z_axis, normal)
    rotation_axis_norm = LA.norm(rotation_axis)
    if rotation_axis_norm > 1e-6:
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(centroid)
    arrow.paint_uniform_color([0.0, 0.8, 0.8])
    geometries.append(arrow)

    plane_size = 0.08
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.001)
    plane_mesh.translate([-plane_size / 2, -plane_size / 2, -0.0005])
    if rotation_axis_norm > 1e-6:
        plane_mesh.rotate(R, center=[0, 0, 0])
    plane_mesh.translate(centroid)
    plane_mesh.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(plane_mesh)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
    geometries.append(coord_frame)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=800,
        height=600,
        point_show_normal=False
    )


def _pca_plane_normal(pts_np: np.ndarray, visualize: bool = False):
    """Return (centroid, unit normal) for best-fit plane to pts_np (N,3)."""
    r_xy = np.sqrt(pts_np[:, 0] ** 2 + pts_np[:, 1] ** 2)
    centroid_idx = np.argmin(r_xy)
    c = pts_np[centroid_idx]
    X = pts_np - c

    C = (X.T @ X) / max(len(X) - 1, 1)
    w, v = LA.eigh(C)
    n = v[:, 0]
    if n[2] < 0:
        n = -n
    n /= (LA.norm(n) + 1e-12)

    if visualize:
        _visualize_normal_estimation(
            pts_np, c, n,
            inlier_mask=None,
            title='PCA Normal Estimation'
        )

    return c, n


def _ransac_plane_normal(pts_np: np.ndarray, n_iterations: int = 100, distance_threshold: float = 0.005,
                         visualize: bool = False):
    n_points = pts_np.shape[0]
    if n_points < 3:
        return np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)

    best_inlier_count = 0
    best_normal = None
    best_inlier_mask = None

    for _ in range(n_iterations):
        idx = np.random.choice(n_points, size=3, replace=False)
        p1, p2, p3 = pts_np[idx[0]], pts_np[idx[1]], pts_np[idx[2]]

        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_mag = LA.norm(n)
        if norm_mag < 1e-12:
            continue
        n = n / norm_mag

        distances = np.abs((pts_np - p1) @ n)
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = n
            best_inlier_mask = inlier_mask

    if best_normal is None or best_inlier_count < 3:
        return _pca_plane_normal(pts_np, visualize=visualize)

    inlier_pts = pts_np[best_inlier_mask]

    r_xy = np.sqrt(inlier_pts[:, 0] ** 2 + inlier_pts[:, 1] ** 2)
    c = inlier_pts[np.argmin(r_xy)]

    X = inlier_pts - c
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, v = LA.eigh(C)
    n = v[:, 0]

    if n[2] < 0:
        n = -n
    n = n / (LA.norm(n) + 1e-12)

    if visualize:
        _visualize_normal_estimation(
            pts_np, c, n,
            inlier_mask=best_inlier_mask,
            title='RANSAC Normal Estimation'
        )

    return c.astype(np.float32), n.astype(np.float32)


def _quaternion_from_z(normal: np.ndarray) -> Quaternion:
    """Return quaternion with +Z aligned with normal."""
    z = normal / LA.norm(normal)
    up = np.array([0, 0, 1])

    if np.array_equal(z, up):
        x = np.array([1, 0, 0])
    elif np.array_equal(z, -up):
        x = np.array([-1, 0, 0])
    else:
        x = np.cross(z, up)
        x /= LA.norm(x)

    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)

    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return Quaternion(x=qx, y=qy, z=qz, w=qw)


def _z_axis_rotvec_error(z_goal: np.ndarray) -> np.ndarray:
    """Return rotation-vector ω that rotates zc=[0,0,1] onto z_goal."""
    zc = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    zg = z_goal.astype(np.float32)
    zg /= (LA.norm(zg) + 1e-12)

    c = float(np.clip(np.dot(zc, zg), -1.0, 1.0))
    theta = math.acos(c)
    print(f'Debug: theta={math.degrees(theta):.2f}°')

    axis = np.cross(zg, zc)
    n = LA.norm(axis)

    if n < 1e-9:
        if c > 0.0:
            return np.float32(0.0), np.zeros(3, dtype=np.float32)
        else:
            return np.float32(theta), np.array([1.0, 0.0, 0.0], dtype=np.float32)

    axis /= n
    return np.float32(theta), axis.astype(np.float32)


def _roll_error(x_current_world) -> float:
    """Angle between camera x-axis and world XY plane."""
    xc = x_current_world / (LA.norm(x_current_world) + 1e-12)
    theta = math.asin(float(np.clip(xc[2], -1.0, 1.0)))
    return theta


class OrientationControlNode(Node):
    def __init__(self):
        super().__init__('orientation_controller')

        sub_cb_group = ReentrantCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.declare_parameters(
            namespace='',
            parameters=[
                ('depth_topic', '/camera/d405_camera/depth/image_rect_raw/compressed'),
                ('camera_info_topic', '/camera/d405_camera/depth/camera_info'),
                ('bounding_box_topic', '/viewpoint_generation/bounding_box_marker'),
                ('joy_topic', 'joy'),
                ('enable_button', 0),
                ('dmap_filter_min', 0.07),
                ('dmap_filter_max', 0.50),
                ('viz_enable', True),
                ('pcd_downsampling_stride', 4),
                ('target_frame', 'object_frame'),
                ('main_camera_frame', 'eoat_camera_link'),
                ('crop_radius', 0.05),
                ('crop_z_min', 0.05),
                ('crop_z_max', 0.40),
                ('standoff_m', 0.10),
                ('standoff_mode', 'euclidean'),
                ('ema_enable', True),
                ('ema_tau', 0.25),
                ('normal_estimation_method', 'PCA'),
                ('visualize_normal_estimation', False),
                ('no_target_timeout_s', 0.25),
                ('orientation_control_enabled', False),
                ('autofocus_enabled', False),
                ('save_data', False),
                ('data_path', '/tmp'),
                ('object', ''),
                ('sphere_mass', 5.0),
                ('sphere_radius', 0.1),
                ('fluid_viscosity', 0.0010016),
                ('d_min', 0.15),
                ('d_max', 0.30),
                ('v_max', 0.40),
                ('theta_max_deg', 30.0),
                ('controller_type', 'PD'),
                ('Kp', 200.0),
                ('Ki', 5.0),
                ('Kd', 5.0),
                ('anti_windup_enabled', True),
                ('integral_alpha', 5.0),
                ('surface_target_frame', 'surface_target'),
                # Kalman filter parameters
                ('kalman_enabled', True),
                ('kalman_R', 1e-01),
                ('kalman_Q_angle', 4e-03),
                ('kalman_Q_dangle', 1.245061e-01),
                ('zeta', 1.0),
                ('ie_clamp', 8.0),

                # =========================
                # Simple adaptive Q (NEW)
                # =========================
                ('adaptive_q_enabled', True),
                ('adaptive_q_tau_thresh', 0.5),      # N·m
                ('adaptive_q_w_thresh', 0.15),       # rad/s
                ('adaptive_q_scale_moving', 50.0),   # 10–100x typical
            ]
        )

        # Get parameters
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.bounding_box_topic = self.get_parameter('bounding_box_topic').get_parameter_value().string_value
        joy_topic = self.get_parameter('joy_topic').get_parameter_value().string_value
        self.enable_button = int(self.get_parameter('enable_button').value)
        self.dmap_filter_min = float(self.get_parameter('dmap_filter_min').value)
        self.dmap_filter_max = float(self.get_parameter('dmap_filter_max').value)
        self.viz_enable = bool(self.get_parameter('viz_enable').value)
        self.pcd_downsampling_stride = int(self.get_parameter('pcd_downsampling_stride').value)
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        self.crop_radius = float(self.get_parameter('crop_radius').value)
        self.crop_z_min = float(self.get_parameter('crop_z_min').value)
        self.crop_z_max = float(self.get_parameter('crop_z_max').value)
        self.standoff_m = float(self.get_parameter('standoff_m').value)
        self.standoff_mode = str(self.get_parameter('standoff_mode').value).lower()
        self.ema_enable = bool(self.get_parameter('ema_enable').value)
        self.ema_tau = float(self.get_parameter('ema_tau').value)

        self._ema_normal = None
        self._ema_centroid = None
        self._ema_last_t = None

        # Kalman filters
        self.kalman_enabled = bool(self.get_parameter('kalman_enabled').value)
        kalman_R = float(self.get_parameter('kalman_R').value)
        kalman_Q_angle = float(self.get_parameter('kalman_Q_angle').value)
        kalman_Q_dangle = float(self.get_parameter('kalman_Q_dangle').value)

        self.pitch_kalman = AngleKalmanFilter(kalman_R, kalman_Q_angle, kalman_Q_dangle)
        self.yaw_kalman = AngleKalmanFilter(kalman_R, kalman_Q_angle, kalman_Q_dangle)
        self._kalman_last_t = None

        # =========================
        # Adaptive Q params + baseline storage (NEW)
        # =========================
        self.adaptive_q_enabled = bool(self.get_parameter('adaptive_q_enabled').value)
        self.adaptive_q_tau_thresh = float(self.get_parameter('adaptive_q_tau_thresh').value)
        self.adaptive_q_w_thresh = float(self.get_parameter('adaptive_q_w_thresh').value)
        self.adaptive_q_scale_moving = float(self.get_parameter('adaptive_q_scale_moving').value)

        self._kalman_Q_angle_base = kalman_Q_angle
        self._kalman_Q_dangle_base = kalman_Q_dangle

        # ---- Initialize bbox fields ----
        self.bbox_min = np.array([-float('inf'), -float('inf'), -float('inf')], dtype=float)
        self.bbox_max = np.array([float('inf'), float('inf'), float('inf')], dtype=float)
        self.main_camera_frame = self.get_parameter('main_camera_frame').get_parameter_value().string_value

        self.normal_estimation_method = self.get_parameter('normal_estimation_method').get_parameter_value().string_value.upper()
        self.visualize_normal_estimation = bool(self.get_parameter('visualize_normal_estimation').value)
        self._orientation_control_before_viz = False

        self.d_min = float(self.get_parameter('d_min').value)
        self.d_max = float(self.get_parameter('d_max').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.theta_max_deg = float(self.get_parameter('theta_max_deg').value)

        self.Kp = float(self.get_parameter('Kp').value)
        self.Ki = float(self.get_parameter('Ki').value)
        self.Kd = float(self.get_parameter('Kd').value)
        self.no_target_timeout_s = float(self.get_parameter('no_target_timeout_s').value)
        self.controller_type = str(self.get_parameter('controller_type').value).upper()
        self.integral_alpha = float(self.get_parameter('integral_alpha').value)
        self.zeta = float(self.get_parameter('zeta').value)
        self.ie_clamp = float(self.get_parameter('ie_clamp').value)

        self._had_target_last_cycle = False
        self._last_target_time_s = None
        self.orientation_control_enabled = bool(self.get_parameter('orientation_control_enabled').value)
        self.autofocus_enabled = bool(self.get_parameter('autofocus_enabled').value)

        # save data parameters
        self.save_data = bool(self.get_parameter('save_data').value)
        self.data_path = self.get_parameter('data_path').get_parameter_value().string_value
        self.object = self.get_parameter('object').get_parameter_value().string_value
        self.storage_options = StorageOptions(uri=self.data_path, storage_id='sqlite3')
        self.converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        self.writer = SequentialWriter()
        self.topic_info = TopicMetadata(
            name='orientation_control_data',
            type='viewpoint_generation_interfaces/msg/OrientationControlData',
            serialization_format='cdr'
        )

        self.mass_B = float(self.get_parameter('sphere_mass').value)
        self.sphere_radius = float(self.get_parameter('sphere_radius').value)
        self.fluid_viscosity = float(self.get_parameter('fluid_viscosity').value)
        self.update_inertia_and_drag(self.mass_B, self.sphere_radius, self.fluid_viscosity)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.bridge = CvBridge()
        self.K = None
        self.depth_frame_id = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._tf_ready = False

        self.ocd = OrientationControlData()

        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, qos)
        self.sub_depth = self.create_subscription(CompressedImage, self.depth_topic, self.on_depth, 1,
                                                  callback_group=sub_cb_group)
        self.bbox = self.create_subscription(Marker, self.bounding_box_topic, self.on_bbox, qos)
        self.create_subscription(TwistStamped, f'/servo_node/delta_twist_cmds', self.on_delta_twist, qos)
        self.create_subscription(WrenchStamped, f'/teleop/wrench_cmds', self.on_wrench_cmd, qos)

        self.last_joy_msg = None
        self.joy_sub = self.create_subscription(Joy, joy_topic, self.joy_callback, qos)
        self.depth_msg = None

        self._timer_cb_group = timer_cb_group
        self._process_timer = None
        self._startup_timer = self.create_timer(0.5, self._check_tf_ready, callback_group=timer_cb_group)

        self.point_cloud_publisher = self.create_publisher(
            PointCloud2, f'/camera/d405_camera/depth/fov_points_{self.main_camera_frame}_bbox', 10
        )
        self.normal_estimate_pub = self.create_publisher(PoseStamped, f'/{self.get_name()}/crop_normal', 10)
        self.pub_eoat_pose_crop = self.create_publisher(
            PoseStamped, f'/{self.get_name()}/eoat_desired_pose_in_{self.main_camera_frame}', 10
        )

        self.surface_target_frame = self.get_parameter('surface_target_frame').get_parameter_value().string_value
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_z_rotvec_err = self.create_publisher(
            Vector3Stamped, f'/{self.get_name()}/z_axis_rotvec_error_in_{self.main_camera_frame}', 10
        )
        self.pub_wrench_cmd = self.create_publisher(WrenchStamped, f'/{self.get_name()}/wrench_cmds', 10)
        self.distance_pub = self.create_publisher(Float64, f'/{self.get_name()}/focal_distance_m', 10)

        self.force_B = np.zeros(3, dtype=np.float32)
        self.lin_vel_cam = np.zeros(3, dtype=np.float32)
        self.rot_vel_cam = np.zeros(3, dtype=np.float32)
        self.tau_B = np.zeros(3, dtype=np.float32)
        self.force_teleop = np.zeros(3, dtype=np.float32)
        self.torque_teleop = np.zeros(3, dtype=np.float32)
        self._last_tau = np.zeros(3, dtype=np.float32)
        self._last_force = np.zeros(3, dtype=np.float32)

        self._last_err_t = None

        # Pitch state
        self.last_pitch = 0.0
        self.int_pitch = 0.0
        self.pitch_err = 0.0
        self.dpitch = 0.0
        self._prev_tau_pitch = 0.0

        # Yaw state
        self.last_yaw = 0.0
        self.int_yaw = 0.0
        self.yaw_err = 0.0
        self.dyaw = 0.0
        self._prev_tau_yaw = 0.0

        # Focus distance PID state
        self.focal_distance = 0.2810768097639084

        self._measurement_lock = threading.Lock()
        self._latest_measurement = {
            'valid': False,
            'surface_points': None,
            'centroid': np.zeros(3, dtype=np.float32),
            'normal': np.zeros(3, dtype=np.float32),
            'timestamp': 0.0,
            'stamp': None,
            'rot_vec_error': np.zeros(3, dtype=np.float32),
            'pitch_err': 0.0,
            'yaw_err': 0.0,
            'd': 0.0,
            'roll_error': 0.0,
            'r': np.zeros(3, dtype=np.float32),
            'goal_pose': None,
            'camera_transform': None,
        }

        self.get_logger().info(
            'Background remover running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  camera_info_topic={self.camera_info_topic}\n'
            f'  bounding_box_topic={self.bounding_box_topic}\n'
            f'  dmap_filter_min={self.dmap_filter_min:.3f}, dmap_filter_max={self.dmap_filter_max:.3f}\n'
            f'  pcd_downsampling_stride={self.pcd_downsampling_stride}\n'
            f'  target_frame={self.target_frame}'
            f'  main_camera_frame={self.main_camera_frame}'
            f'  crop_radius={self.crop_radius:.3f}, crop_z=[{self.crop_z_min:.3f},{self.crop_z_max:.3f}]'
            f'  standoff_mode={self.standoff_mode}, standoff_m={self.standoff_m:.3f}'
        )

        self.add_on_set_parameters_callback(self._on_param_update)

    def update_inertia_and_drag(self, mass: float, radius: float, fluid_viscosity: float):
        I = (2.0 / 5.0) * mass * (radius ** 2)
        self.inertia_B = I

        self.linear_drag = 6.0 * math.pi * fluid_viscosity * radius
        self.angular_drag = 2.4 * math.pi * fluid_viscosity * (radius ** 3)

        self.get_logger().info(
            f'Updated inertia to {self.inertia_B} kg·m² and drag to {self.linear_drag} N·s/m, {self.angular_drag} N·m·s/rad'
        )

    def _check_tf_ready(self):
        if self._tf_ready:
            return

        if self.depth_frame_id is None:
            self.get_logger().info('Waiting for camera info to determine depth frame...', throttle_duration_sec=2.0)
            return

        tf_checks = [
            (self.target_frame, self.main_camera_frame),
            (self.target_frame, self.depth_frame_id),
        ]

        all_ready = True
        for target, source in tf_checks:
            try:
                self.tf_buffer.lookup_transform(
                    target, source,
                    Time(sec=0, nanosec=0),
                    timeout=Duration(seconds=0.1)
                )
            except TransformException:
                self.get_logger().info(f'Waiting for TF: {target} <- {source}', throttle_duration_sec=2.0)
                all_ready = False
                break

        if all_ready:
            self._tf_ready = True
            self.get_logger().info('TF tree ready. Starting orientation control processing.')
            self._startup_timer.cancel()
            self._process_timer = self.create_timer(0.1, self.process_controller, callback_group=self._timer_cb_group)

    def joy_callback(self, msg):
        if not self.last_joy_msg:
            self.last_joy_msg = msg
            return

        if not self.last_joy_msg.buttons[self.enable_button]:
            if msg.buttons[self.enable_button] and not self.orientation_control_enabled:
                self.enable_orientation_control()
            elif msg.buttons[self.enable_button] and self.orientation_control_enabled:
                self.disable_orientation_control()

        if not self.last_joy_msg.buttons[1]:
            if msg.buttons[1] and not self.autofocus_enabled:
                self.enable_autofocus()
            elif msg.buttons[1] and self.autofocus_enabled:
                self.disable_autofocus()

        if not self.last_joy_msg.buttons[9]:
            if msg.buttons[9] and not self.visualize_normal_estimation:
                self.enable_normal_estimation_viz()
            elif msg.buttons[9] and self.visualize_normal_estimation:
                self.disable_normal_estimation_viz()

        if not self.last_joy_msg.buttons[8]:
            if msg.buttons[8] and not self.save_data:
                self.enable_save_data()
            elif msg.buttons[8] and self.save_data:
                self.disable_save_data()

        self.last_joy_msg = msg

    def on_delta_twist(self, msg: TwistStamped):
        self.lin_vel_cam[0] = msg.twist.linear.x
        self.lin_vel_cam[1] = msg.twist.linear.y
        self.lin_vel_cam[2] = msg.twist.linear.z
        self.rot_vel_cam[0] = msg.twist.angular.x
        self.rot_vel_cam[1] = msg.twist.angular.y
        self.rot_vel_cam[2] = msg.twist.angular.z

    def on_wrench_cmd(self, msg: WrenchStamped):
        self.force_teleop[0] = msg.wrench.force.x
        self.force_teleop[1] = msg.wrench.force.y
        self.torque_teleop[0] = msg.wrench.torque.x
        self.torque_teleop[1] = msg.wrench.torque.y
        self.torque_teleop[2] = msg.wrench.torque.z

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.depth_frame_id = msg.header.frame_id

    def on_bbox(self, msg: Marker):
        if msg.type != Marker.CUBE:
            self.get_logger().warn(f'Ignoring non-CUBE bounding box marker of type {msg.type}')
            return
        if msg.scale.x <= 0.0 or msg.scale.y <= 0.0 or msg.scale.z <= 0.0:
            self.get_logger().warn(f'Ignoring invalid bounding box with non-positive scale {msg.scale.x}, {msg.scale.y}, {msg.scale.z}')
            return
        if abs(msg.pose.orientation.x) > 1e-3 or abs(msg.pose.orientation.y) > 1e-3 or abs(msg.pose.orientation.z) > 1e-3:
            self.get_logger().warn(f'Ignoring non-axis-aligned bounding box with orientation {msg.pose.orientation}')
            return

        cx, cy, cz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        sx, sy, sz = msg.scale.x, msg.scale.y, msg.scale.z

        half_sizes = np.array([sx, sy, sz], dtype=float) * 0.5
        center = np.array([cx, cy, cz], dtype=float)

        self.bbox_min = center - half_sizes
        self.bbox_max = center + half_sizes

    def _ema_update(self, prev, x, alpha):
        if prev is None:
            return x.copy()
        return (1.0 - alpha) * prev + alpha * x

    def on_depth(self, msg: CompressedImage):
        self.depth_msg = msg
        self.process_depth(msg)

    def process_depth(self, msg: CompressedImage):
        if self.K is None:
            self.get_logger().warn('No camera intrinsics yet, cannot process depth image')
            return

        stamp = msg.header.stamp
        now_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        measurement_ok = False

        centroid_out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        normal_out = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        goal_pose_out = None
        rotvec_err_out = np.zeros(3, dtype=np.float32)
        camera_transform_out = None

        np_arr = np.frombuffer(msg.data, np.uint8)
        if np_arr.size <= 12:
            self.get_logger().warn("Empty or invalid compressedDepth data, skipping frame")
            return
        png_data = np_arr[12:]
        depth = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)
        if depth is None:
            self.get_logger().warn("Failed to decode depth image, skipping frame")
            return

        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 1e-3
        else:
            depth_m = depth.astype(np.float32)

        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        mask = valid & (depth_m >= self.dmap_filter_min) & (depth_m <= self.dmap_filter_max)

        depth_filtered = depth_m.copy()
        depth_filtered[~mask] = np.nan

        points1 = np.zeros((0, 3), dtype=np.float32)
        points2 = np.zeros((0, 3), dtype=np.float32)

        pitch_err = 0.0
        yaw_err = 0.0
        d = 0.0
        roll_error = 0.0
        r = np.zeros(3, dtype=np.float32)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        stride = max(1, self.pcd_downsampling_stride)
        ys, xs = np.where(mask)

        if stride > 1:
            ys = ys[::stride]
            xs = xs[::stride]

        if xs.shape[0] > 0:
            z = depth_m[ys, xs]
            x = (xs.astype(np.float32) - cx) * z / fx
            y = (ys.astype(np.float32) - cy) * z / fy
            points1 = np.stack([x, y, z], axis=1)
            src_frame = msg.header.frame_id
            pts_bbox = np.zeros((0, 3), dtype=np.float32)
            try:
                T = self.tf_buffer.lookup_transform(
                    self.target_frame, src_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                q = T.transform.rotation
                R = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
                t = T.transform.translation
                t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                pts_tgt = (R @ points1.T).T + t_vec

            except TransformException as e:
                self.get_logger().warn(f'No TF {self.target_frame} <- {src_frame} at stamp: {e}')
                pts_tgt = np.zeros((0, 3), dtype=np.float32)
            else:
                sel = np.all((pts_tgt >= self.bbox_min) & (pts_tgt <= self.bbox_max), axis=1)
                pts_bbox = np.ascontiguousarray(pts_tgt[sel])

            try:
                T_out = self.tf_buffer.lookup_transform(
                    self.main_camera_frame, self.target_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                q_out = T_out.transform.rotation
                R_out = _quat_to_R_xyzw(q_out.x, q_out.y, q_out.z, q_out.w)
                t_out = T_out.transform.translation
                t_vec_out = np.array([t_out.x, t_out.y, t_out.z], dtype=np.float32)
                pts_bbox_out = (R_out @ pts_bbox.T).T + t_vec_out

            except TransformException as e:
                self.get_logger().warn(f'No TF {self.main_camera_frame} <- {self.target_frame}: {e}')
                pts_bbox_out = np.zeros((0, 3), dtype=np.float32)

            points1 = pts_bbox_out

            if pts_bbox_out.shape[0] > 0:
                Xc, Yc, Zc = pts_bbox_out[:, 0], pts_bbox_out[:, 1], pts_bbox_out[:, 2]
                r2 = Xc * Xc + Yc * Yc
                sel_crop = (
                    (Zc >= self.crop_z_min) & (Zc <= self.crop_z_max) &
                    (r2 <= (self.crop_radius * self.crop_radius))
                )

                pts_crop = np.ascontiguousarray(pts_bbox_out[sel_crop])
                points2 = pts_crop

                if pts_crop.shape[0] >= 10:
                    if self.normal_estimation_method == 'PCA':
                        centroid, normal = _pca_plane_normal(pts_crop, visualize=self.visualize_normal_estimation)
                    elif self.normal_estimation_method == 'RANSAC':
                        centroid, normal = _ransac_plane_normal(pts_crop, visualize=self.visualize_normal_estimation)
                    else:
                        centroid, normal = _pca_plane_normal(pts_crop, visualize=self.visualize_normal_estimation)

                    if self.visualize_normal_estimation:
                        self.visualize_normal_estimation = False
                        if self._orientation_control_before_viz:
                            self.orientation_control_enabled = True
                            self.get_logger().info('Re-enabling orientation_control_enabled after visualization closed')
                        self._orientation_control_before_viz = False

                    if not self.ema_enable:
                        cen_s = centroid
                        nrm_s = normal
                        self._ema_centroid = centroid
                        self._ema_normal = normal
                        self._ema_last_t = now_s
                    else:
                        if self._ema_last_t is None:
                            self._ema_centroid = centroid.copy()
                            self._ema_normal = normal.copy()
                            self._ema_last_t = now_s

                        dt = max(0.0, now_s - (self._ema_last_t if self._ema_last_t is not None else now_s))
                        alpha = 1.0 - math.exp(-dt / max(1e-3, self.ema_tau))
                        alpha = min(1.0, max(0.0, alpha))

                        self._ema_centroid = self._ema_update(self._ema_centroid, centroid, alpha)
                        self._ema_normal = self._ema_update(self._ema_normal, normal, alpha)

                        n = LA.norm(self._ema_normal) + 1e-12
                        self._ema_normal /= n

                        cen_s = self._ema_centroid
                        nrm_s = self._ema_normal
                        self._ema_last_t = now_s

                    r = -cen_s

                    T = self.tf_buffer.lookup_transform(
                        'object_frame', self.main_camera_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                    camera_transform_out = T
                    q = T.transform.rotation
                    R_tf = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
                    x_cf, y_cf, z_cf = R_tf[:, 0], R_tf[:, 1], R_tf[:, 2]
                    t = T.transform.translation
                    t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                    surface_position_of = (R_tf @ cen_s) + t_vec
                    surface_normal_of = R_tf @ nrm_s

                    surface_target_pose = PoseStamped()
                    surface_target_pose.header = msg.header
                    surface_target_pose.header.frame_id = 'object_frame'
                    surface_target_pose.pose.position.x = float(surface_position_of[0])
                    surface_target_pose.pose.position.y = float(surface_position_of[1])
                    surface_target_pose.pose.position.z = float(surface_position_of[2])
                    surface_target_pose.pose.orientation = _quaternion_from_z(surface_normal_of)
                    self.normal_estimate_pub.publish(surface_target_pose)

                    tf_msg = TransformStamped()
                    tf_msg.header = surface_target_pose.header
                    tf_msg.child_frame_id = self.surface_target_frame
                    tf_msg.transform.translation.x = surface_target_pose.pose.position.x
                    tf_msg.transform.translation.y = surface_target_pose.pose.position.y
                    tf_msg.transform.translation.z = surface_target_pose.pose.position.z
                    tf_msg.transform.rotation = surface_target_pose.pose.orientation
                    self.tf_broadcaster.sendTransform(tf_msg)

                    if self.standoff_mode == 'euclidean':
                        d = float(LA.norm(cen_s))
                    elif self.standoff_mode == 'along_normal':
                        d = float(max(0.0, float(np.dot(nrm_s, cen_s))))
                    else:
                        d = self.standoff_m

                    goal_position_cf = cen_s - d * nrm_s
                    goal_orientation_of = _quaternion_from_z(surface_normal_of)
                    R_goal_of = _quat_to_R_xyzw(goal_orientation_of.x, goal_orientation_of.y,
                                                goal_orientation_of.z, goal_orientation_of.w)
                    R_goal_cf = R_tf.T @ R_goal_of
                    goal_orientation_cf = _R_to_quat_xyzw(R_goal_cf)

                    goal_pose = PoseStamped()
                    goal_pose.header = msg.header
                    goal_pose.header.frame_id = self.main_camera_frame
                    goal_pose.pose.position.x = float(goal_position_cf[0])
                    goal_pose.pose.position.y = float(goal_position_cf[1])
                    goal_pose.pose.position.z = float(goal_position_cf[2])
                    goal_pose.pose.orientation = goal_orientation_cf
                    self.pub_eoat_pose_crop.publish(goal_pose)

                    nx, ny, nz = nrm_s[0], nrm_s[1], nrm_s[2]
                    pitch_err = math.atan2(float(ny), float(nz))
                    yaw_err = math.atan2(float(nx), float(nz))

                    roll_error = _roll_error(x_cf)

                    rot_vec_error = np.array([pitch_err, yaw_err, roll_error], dtype=np.float32)

                    centroid_out = cen_s.astype(np.float32)
                    normal_out = nrm_s.astype(np.float32)
                    rotvec_err_out = rot_vec_error.astype(np.float32)

                    err_msg = Vector3Stamped()
                    err_msg.header = msg.header
                    err_msg.header.frame_id = self.main_camera_frame
                    err_msg.vector.x, err_msg.vector.y, err_msg.vector.z = map(float, rot_vec_error)
                    self.pub_z_rotvec_err.publish(err_msg)

                    measurement_ok = True
                    self._had_target_last_cycle = True
                    self._last_target_time_s = now_s

        pcd_msg = make_pointcloud2(points2, frame_id=self.main_camera_frame, stamp=msg.header.stamp)
        self.point_cloud_publisher.publish(pcd_msg)

        goal_pose_out = None
        if measurement_ok:
            goal_pose_out = goal_pose

        with self._measurement_lock:
            self._latest_measurement['valid'] = measurement_ok
            self._latest_measurement['centroid'] = centroid_out.copy()
            self._latest_measurement['normal'] = normal_out.copy()
            self._latest_measurement['timestamp'] = now_s
            self._latest_measurement['stamp'] = msg.header.stamp
            self._latest_measurement['rot_vec_error'] = rotvec_err_out.copy()
            self._latest_measurement['pitch_err'] = float(pitch_err)
            self._latest_measurement['yaw_err'] = float(yaw_err)
            self._latest_measurement['surface_points'] = points2.copy() if points2 is not None else None
            self._latest_measurement['d'] = float(d)
            self._latest_measurement['roll_error'] = float(roll_error)
            self._latest_measurement['r'] = r.copy()
            self._latest_measurement['goal_pose'] = goal_pose_out
            self._latest_measurement['camera_transform'] = camera_transform_out

        if not measurement_ok:
            if (self._last_target_time_s is None) or ((now_s - self._last_target_time_s) > self.no_target_timeout_s):
                if self._had_target_last_cycle:
                    self.get_logger().warn('No valid target: lost')
                self._ema_normal = None
                self._ema_centroid = None
                self._ema_last_t = None
                self._had_target_last_cycle = False

    def _handle_no_measurement(self):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        print("No valid measurement available.")

        if self._had_target_last_cycle:
            self._last_target_time_s = now_s
            self._had_target_last_cycle = False

        time_since_target = 0.0 if self._last_target_time_s is None else (now_s - self._last_target_time_s)

        if time_since_target > self.no_target_timeout_s:
            self._last_err_t = None
            self.int_pitch = 0.0
            self.int_yaw = 0.0
            self.last_pitch = 0.0
            self.last_yaw = 0.0
            self._prev_tau_pitch = 0.0
            self._prev_tau_yaw = 0.0

        w = WrenchStamped()
        w.header.stamp = self.get_clock().now().to_msg()
        w.header.frame_id = self.main_camera_frame
        self.pub_wrench_cmd.publish(w)

    def process_controller(self):
        with self._measurement_lock:
            if not self._latest_measurement['valid']:
                self._handle_no_measurement()
                return

            cen_s = self._latest_measurement['centroid'].copy()
            nrm_s = self._latest_measurement['normal'].copy()
            rot_vec_error = self._latest_measurement['rot_vec_error'].copy()
            pitch_err_meas = self._latest_measurement['pitch_err']
            yaw_err_meas = self._latest_measurement['yaw_err']
            meas_stamp = self._latest_measurement['stamp']
            d = self._latest_measurement['d']
            roll_error = self._latest_measurement['roll_error']
            r = self._latest_measurement['r'].copy()
            surface_points = self._latest_measurement['surface_points']
            goal_pose = self._latest_measurement['goal_pose']
            camera_transform = self._latest_measurement['camera_transform']

        now_s = self.get_clock().now().nanoseconds * 1e-9

        tau_out = np.zeros(3, dtype=np.float32)
        self.anti_windup_enabled = True

        dt_ctrl = 0.0
        if self._last_err_t is not None:
            dt_ctrl = max(1e-6, now_s - self._last_err_t)
        self._last_err_t = now_s

        # Raw errors
        pitch_err_raw = -pitch_err_meas
        yaw_err_raw = yaw_err_meas

        # ===== NEW: measured raw rates BEFORE KF (for adaptive-Q gating) =====
        dpitch_meas = 0.0
        dyaw_meas = 0.0
        if dt_ctrl > 0.0:
            dpitch_meas = (pitch_err_raw - float(self.last_pitch)) / dt_ctrl
            dyaw_meas = (yaw_err_raw - float(self.last_yaw)) / dt_ctrl

        inertia_A = self.inertia_B + self.mass_B * d ** 2
        b_damping = self.linear_drag * d * d

        # ===== NEW: Simple adaptive Q (screenshot logic) =====
        if self.kalman_enabled and self.adaptive_q_enabled and dt_ctrl > 0.0:
            pitch_moving = (abs(self._prev_tau_pitch) > self.adaptive_q_tau_thresh)
            pitch_moving = pitch_moving or (abs(dpitch_meas) > self.adaptive_q_w_thresh)

            yaw_moving = (abs(self._prev_tau_yaw) > self.adaptive_q_tau_thresh)
            yaw_moving = yaw_moving or (abs(dyaw_meas) > self.adaptive_q_w_thresh)

            pitch_scale = self.adaptive_q_scale_moving if pitch_moving else 1.0
            yaw_scale = self.adaptive_q_scale_moving if yaw_moving else 1.0

            self.pitch_kalman.set_noise(
                Q_angle=self._kalman_Q_angle_base * pitch_scale,
                Q_dangle=self._kalman_Q_dangle_base * pitch_scale
            )
            self.yaw_kalman.set_noise(
                Q_angle=self._kalman_Q_angle_base * yaw_scale,
                Q_dangle=self._kalman_Q_dangle_base * yaw_scale
            )

        # Apply Kalman filtering if enabled
        if self.kalman_enabled and dt_ctrl > 0:
            pitch_filtered, dpitch_filtered = self.pitch_kalman.predict_and_update(
                pitch_err_raw, dt_ctrl,
                u_prev=-self._prev_tau_pitch,
                I_A=inertia_A,
                b=b_damping
            )
            self.pitch_err = pitch_filtered
            self.dpitch = dpitch_filtered

            yaw_filtered, dyaw_filtered = self.yaw_kalman.predict_and_update(
                yaw_err_raw, dt_ctrl,
                u_prev=-self._prev_tau_yaw,
                I_A=inertia_A,
                b=b_damping
            )
            self.yaw_err = yaw_filtered
            self.dyaw = dyaw_filtered

            self.pitch_err_raw = pitch_err_raw
            self.yaw_err_raw = yaw_err_raw
            self.dpitch_raw = (pitch_err_raw - self.last_pitch) / dt_ctrl if dt_ctrl > 0 else 0.0
            self.dyaw_raw = (yaw_err_raw - self.last_yaw) / dt_ctrl if dt_ctrl > 0 else 0.0
        else:
            self.pitch_err = pitch_err_raw
            self.yaw_err = yaw_err_raw
            if dt_ctrl > 0:
                self.dpitch = (self.pitch_err - self.last_pitch) / dt_ctrl
                self.dyaw = (self.yaw_err - self.last_yaw) / dt_ctrl
            else:
                self.dpitch = 0.0
                self.dyaw = 0.0
            self.pitch_err_raw = self.pitch_err
            self.yaw_err_raw = self.yaw_err
            self.dpitch_raw = self.dpitch
            self.dyaw_raw = self.dyaw

        self.last_pitch = float(pitch_err_raw)
        self.last_yaw = float(yaw_err_raw)

        tau_max = self.linear_drag * d * self.v_max
        omega_n_max = math.sqrt(tau_max / (inertia_A * self.theta_max_deg * math.pi / 180.0))
        omega_n = omega_n_max

        p_1 = -self.zeta * omega_n + omega_n * math.sqrt(self.zeta ** 2 - 1)
        p_2 = -self.zeta * omega_n - omega_n * math.sqrt(self.zeta ** 2 - 1)
        p_3 = 5 * p_2

        if self.controller_type == 'PD':
            self.Kp = inertia_A * (p_1 * p_2)
            self.Kd = -inertia_A * (p_1 + p_2) - self.linear_drag * d * d
            self.Ki = 0.0
        elif self.controller_type == 'PID':
            self.Kp = inertia_A * (p_1 * p_2 + p_1 * p_3 + p_2 * p_3)
            self.Ki = self.integral_alpha * -inertia_A * (p_1 * p_2 * p_3)
            self.Kd = -inertia_A * (p_1 + p_2 + p_3) - self.linear_drag * d * d
        else:
            self.get_logger().warn(f'Unknown controller_type {self.controller_type}, defaulting to PD')
            self.Kp = inertia_A * (p_1 * p_2)
            self.Kd = -inertia_A * (p_1 + p_2) - self.linear_drag * d * d
            self.Ki = 0.0

        tau_pitch_raw = self.Kp * self.pitch_err + self.Ki * self.int_pitch + self.Kd * self.dpitch
        tau_yaw_raw = self.Kp * self.yaw_err + self.Ki * self.int_yaw + self.Kd * self.dyaw

        tau_magnitude = math.sqrt(tau_pitch_raw ** 2 + tau_yaw_raw ** 2)
        if tau_magnitude > tau_max and tau_magnitude > 1e-9:
            scale = tau_max / tau_magnitude
            tau_pitch = tau_pitch_raw * scale
            tau_yaw = tau_yaw_raw * scale
            is_saturated = True
        else:
            tau_pitch = tau_pitch_raw
            tau_yaw = tau_yaw_raw
            is_saturated = False

        if self.anti_windup_enabled and self.Ki > 1e-9 and dt_ctrl > 0.0:
            if is_saturated:
                if (tau_pitch_raw > 0 and self.pitch_err < 0) or (tau_pitch_raw < 0 and self.pitch_err > 0):
                    self.int_pitch += self.pitch_err * dt_ctrl
                if (tau_yaw_raw > 0 and self.yaw_err < 0) or (tau_yaw_raw < 0 and self.yaw_err > 0):
                    self.int_yaw += self.yaw_err * dt_ctrl
            else:
                self.int_pitch += self.pitch_err * dt_ctrl
                self.int_yaw += self.yaw_err * dt_ctrl
        elif dt_ctrl > 0.0:
            self.int_pitch += self.pitch_err * dt_ctrl
            self.int_yaw += self.yaw_err * dt_ctrl

        int_lim = float(self.ie_clamp) * (math.pi / 180)
        self.int_pitch = np.clip(self.int_pitch, -int_lim, int_lim)
        self.int_yaw = np.clip(self.int_yaw, -int_lim, int_lim)

        if not self.orientation_control_enabled:
            tau_pitch = 0.0
            tau_yaw = 0.0

        self._prev_tau_pitch = float(tau_pitch)
        self._prev_tau_yaw = float(tau_yaw)

        tau_theta_vec = np.array([tau_pitch, tau_yaw, 0.0], dtype=np.float32)
        moment_tele_A = np.cross(r, self.force_teleop)

        tau_roll = roll_error * self.Kp
        F_z = -1 * self.Kp * (self.focal_distance - d)

        force_drag_B = -self.linear_drag * self.lin_vel_cam
        moment_drag_B = -self.angular_drag * self.rot_vel_cam
        moment_dragB_A = np.cross(r, force_drag_B)
        self.ang_acc = (moment_dragB_A + tau_theta_vec + moment_tele_A) / inertia_A

        tau_out = tau_theta_vec.copy()
        self._last_tau = tau_theta_vec.copy()
        self.force_B = self.mass_B * np.cross(self.ang_acc, r) - force_drag_B - self.force_teleop
        self.force_B[2] = F_z
        self.tau_B = self.inertia_B * self.ang_acc - moment_drag_B - self.torque_teleop
        self.tau_B[2] = tau_roll

        self.distance_pub.publish(Float64(data=float(LA.norm(cen_s))))

        w = WrenchStamped()
        w.header.stamp = self.get_clock().now().to_msg()
        w.header.frame_id = self.main_camera_frame
        w.wrench.torque.z = float(tau_roll)

        if self.orientation_control_enabled:
            w.wrench.force.x = float(self.force_B[0])
            w.wrench.force.y = float(self.force_B[1])
            w.wrench.force.z = float(self.force_B[2])
            w.wrench.torque.x = float(self.tau_B[0])
            w.wrench.torque.y = float(self.tau_B[1])
        else:
            tau_out[0] = 0.0
            tau_out[1] = 0.0
            self._last_tau = tau_out.copy()
            self._last_force[0] = 0.0
            self._last_force[1] = 0.0

        if self.autofocus_enabled:
            w.wrench.force.z = float(F_z)
            print(f"Applying autofocus force F_z: {F_z:.3f} N based on focal distance error {self.focal_distance - d:.3f} m")

        self.pub_wrench_cmd.publish(w)

        # Populate OrientationControlData
        if meas_stamp is not None:
            self.ocd.header.stamp = meas_stamp
        else:
            self.ocd.header.stamp = self.get_clock().now().to_msg()
        self.ocd.header.frame_id = self.main_camera_frame

        if surface_points is not None and len(surface_points) > 0:
            self.ocd.point_cloud = make_pointcloud2(surface_points, frame_id=self.main_camera_frame, stamp=self.ocd.header.stamp)
        else:
            self.ocd.point_cloud = make_pointcloud2(np.zeros((0, 3), dtype=np.float32), frame_id=self.main_camera_frame, stamp=self.ocd.header.stamp)

        self.ocd.normal_method = self.normal_estimation_method

        self.ocd.surface_centroid = Point(x=float(cen_s[0]), y=float(cen_s[1]), z=float(cen_s[2]))
        self.ocd.surface_normal = Vector3(x=float(nrm_s[0]), y=float(nrm_s[1]), z=float(nrm_s[2]))

        if goal_pose is not None:
            self.ocd.goal_pose = goal_pose
        else:
            self.ocd.goal_pose = PoseStamped()
            self.ocd.goal_pose.header = self.ocd.header

        self.ocd.pitch_error = float(self.pitch_err)
        self.ocd.dpitch_error = float(self.dpitch)
        self.ocd.ipitch_error = float(self.int_pitch)
        self.ocd.pitch_error_raw = float(self.pitch_err_raw)
        self.ocd.dpitch_error_raw = float(self.dpitch_raw)

        self.ocd.yaw_error = float(self.yaw_err)
        self.ocd.dyaw_error = float(self.dyaw)
        self.ocd.iyaw_error = float(self.int_yaw)
        self.ocd.yaw_error_raw = float(self.yaw_err_raw)
        self.ocd.dyaw_error_raw = float(self.dyaw_raw)

        self.ocd.roll_error = float(roll_error)
        self.ocd.droll_error = 0.0
        self.ocd.iroll_error = 0.0

        self.ocd.pitch_torque_command = float(tau_pitch)
        self.ocd.yaw_torque_command = float(tau_yaw)
        self.ocd.roll_torque_command = float(tau_roll)

        self.ocd.k_p = float(self.Kp)
        self.ocd.k_d = float(self.Kd)
        self.ocd.k_i = float(self.Ki)

        try:
            tf_pose = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.main_camera_frame,
                Time(sec=0, nanosec=0),
                timeout=Duration(seconds=0.01)
            )
            self.ocd.current_pose.header.stamp = self.ocd.header.stamp
            self.ocd.current_pose.header.frame_id = self.target_frame
            self.ocd.current_pose.pose.position.x = tf_pose.transform.translation.x
            self.ocd.current_pose.pose.position.y = tf_pose.transform.translation.y
            self.ocd.current_pose.pose.position.z = tf_pose.transform.translation.z
            self.ocd.current_pose.pose.orientation = tf_pose.transform.rotation
        except TransformException:
            self.ocd.current_pose.header.stamp = self.ocd.header.stamp

        self.ocd.current_twist.header.stamp = self.ocd.header.stamp
        self.ocd.current_twist.header.frame_id = self.main_camera_frame
        self.ocd.current_twist.twist.linear.x = float(self.lin_vel_cam[0])
        self.ocd.current_twist.twist.linear.y = float(self.lin_vel_cam[1])
        self.ocd.current_twist.twist.linear.z = float(self.lin_vel_cam[2])
        self.ocd.current_twist.twist.angular.x = float(self.rot_vel_cam[0])
        self.ocd.current_twist.twist.angular.y = float(self.rot_vel_cam[1])
        self.ocd.current_twist.twist.angular.z = float(self.rot_vel_cam[2])

        if camera_transform is not None:
            self.ocd.camera_transform = camera_transform
        else:
            self.ocd.camera_transform.header.stamp = self.ocd.header.stamp

        if self.save_data:
            self.bag_orientation_control_data()

    def bag_orientation_control_data(self):
        self.get_logger().warn('Writing orientation control data to bag...')
        self.writer.write(
            'orientation_control_data',
            serialize_message(self.ocd),
            self.get_clock().now().nanoseconds
        )

    def enable_save_data(self):
        self.get_logger().info('Data saving ENABLED.')
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        uri = f'{self.data_path}/{self.object}_{self.get_clock().now().to_msg().sec}.bag'
        self.storage_options = StorageOptions(uri=uri, storage_id='sqlite3')
        self.get_logger().info(f'Opening bag file at: {uri}')
        self.writer.open(self.storage_options, self.converter_options)
        self.writer.create_topic(self.topic_info)
        self.save_data = True

    def disable_save_data(self):
        self.save_data = False
        self.get_logger().info('Data saving DISABLED.')
        self.writer.close()
        self.get_logger().info(f'Closed bag file.')
        debag(
            self.storage_options.uri,
            sphere_mass=self.mass_B,
            sphere_radius=self.sphere_radius,
            fluid_viscosity=self.fluid_viscosity
        )

    def enable_normal_estimation_viz(self):
        if not self.visualize_normal_estimation:
            self.visualize_normal_estimation = True
            self._orientation_control_before_viz = self.orientation_control_enabled
            w = WrenchStamped()
            w.header.stamp = self.get_clock().now().to_msg()
            w.header.frame_id = self.main_camera_frame
            self.pub_wrench_cmd.publish(w)
            if self.orientation_control_enabled:
                self.orientation_control_enabled = False
                self.get_logger().info('Disabling orientation_control_enabled for normal estimation visualization')

    def disable_normal_estimation_viz(self):
        if self.visualize_normal_estimation:
            self.visualize_normal_estimation = False
            if self._orientation_control_before_viz:
                self.orientation_control_enabled = True
                self.get_logger().info('Re-enabling orientation_control_enabled after normal estimation visualization closed')
            self._orientation_control_before_viz = False

    def enable_orientation_control(self):
        self._last_err_t = None
        self.int_pitch = 0.0
        self.int_yaw = 0.0
        self.orientation_control_enabled = True
        self.get_logger().info('Orientation control ENABLED.')

    def disable_orientation_control(self):
        self.orientation_control_enabled = False
        self.get_logger().info('Orientation control DISABLED.')

    def enable_autofocus(self):
        self.autofocus_enabled = True
        self.get_logger().info('Autofocus ENABLED.')

    def disable_autofocus(self):
        self.autofocus_enabled = False
        self.get_logger().info('Autofocus DISABLED.')

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'dmap_filter_min':
                self.dmap_filter_min = float(p.value); self.get_logger().info(f'dmap_filter_min -> {self.dmap_filter_min:.3f} m')
            elif p.name == 'dmap_filter_max':
                self.dmap_filter_max = float(p.value); self.get_logger().info(f'dmap_filter_max  -> {self.dmap_filter_max:.3f} m')
            elif p.name == 'pcd_downsampling_stride':
                self.pcd_downsampling_stride = max(1, int(p.value))
            elif p.name == 'target_frame':
                self.target_frame = str(p.value)
            elif p.name == 'main_camera_frame':
                new_frame = str(p.value)
                if new_frame != self.main_camera_frame:
                    self.main_camera_frame = new_frame
            elif p.name == 'crop_radius':
                self.crop_radius = float(p.value)
            elif p.name == 'crop_z_min':
                self.crop_z_min = float(p.value)
            elif p.name == 'crop_z_max':
                self.crop_z_max = float(p.value)
            elif p.name == 'standoff_m':
                self.standoff_m = float(p.value)
            elif p.name == 'standoff_mode':
                self.standoff_mode = str(p.value).lower()
            elif p.name == 'ema_enable':
                self.ema_enable = bool(p.value)
            elif p.name == 'ema_tau':
                self.ema_tau = max(1e-3, float(p.value))
            elif p.name == 'Kp':
                self.Kp = float(p.value)
            elif p.name == 'Ki':
                self.Ki = float(p.value)
            elif p.name == 'Kd':
                self.Kd = float(p.value)
            elif p.name == 'no_target_timeout_s':
                self.no_target_timeout_s = float(p.value)

            elif p.name == 'object':
                self.object = p.value
            elif p.name == 'save_data':
                self.save_data = p.value
            elif p.name == 'data_path':
                self.data_path = p.value

            elif p.name == 'orientation_control_enabled':
                if not self.orientation_control_enabled and p.value:
                    self.enable_orientation_control()
                elif self.orientation_control_enabled and not p.value:
                    self.disable_orientation_control()

            elif p.name == 'autofocus_enabled':
                if not self.autofocus_enabled and p.value:
                    self.enable_autofocus()
                elif self.autofocus_enabled and not p.value:
                    self.disable_autofocus()

            elif p.name == 'sphere_mass' and p.type_ == p.Type.DOUBLE:
                self.mass_B = float(p.value)
                self.get_logger().info(f'Updated sphere_mass to {self.mass_B:.3f} kg')
                self.update_inertia_and_drag(self.mass_B, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'sphere_radius' and p.type_ == p.Type.DOUBLE:
                self.sphere_radius = float(p.value)
                self.get_logger().info(f'Updated sphere_radius to {self.sphere_radius:.3f} m')
                self.update_inertia_and_drag(self.mass_B, self.sphere_radius, self.fluid_viscosity)
            elif p.name == 'fluid_viscosity' and p.type_ == p.Type.DOUBLE:
                self.fluid_viscosity = float(p.value)
                self.get_logger().info(f'Updated fluid_viscosity to {self.fluid_viscosity:.6f} Pa·s')
                self.update_inertia_and_drag(self.mass_B, self.sphere_radius, self.fluid_viscosity)

            elif p.name == 'v_max' and p.type_ == p.Type.DOUBLE:
                self.v_max = float(p.value)
                self.get_logger().info(f'Updated v_max to {self.v_max:.3f} m/s')
            elif p.name == 'theta_max_deg' and p.type_ == p.Type.DOUBLE:
                self.theta_max_deg = float(p.value)
                self.get_logger().info(f'Updated theta_max_deg to {self.theta_max_deg:.3f} deg')
            elif p.name == 'controller_type':
                self.controller_type = str(p.value).upper()
                self.get_logger().info(f'Updated controller_type to {self.controller_type}')
            elif p.name == 'anti_windup_enabled':
                self.anti_windup_enabled = bool(p.value)
                self.get_logger().info(f'Updated anti_windup_enabled to {self.anti_windup_enabled}')
            elif p.name == 'integral_alpha':
                self.integral_alpha = float(p.value)
                self.get_logger().info(f'Updated integral_alpha to {self.integral_alpha}')
            elif p.name == 'normal_estimation_method':
                self.normal_estimation_method = str(p.value).upper()
                self.get_logger().info(f'Updated normal_estimation_method to {self.normal_estimation_method}')
            elif p.name == 'visualize_normal_estimation':
                self.visualize_normal_estimation = bool(p.value)
                if self.visualize_normal_estimation:
                    self.enable_normal_estimation_viz()
                else:
                    self.disable_normal_estimation_viz()
                self.get_logger().info(f'Updated visualize_normal_estimation to {self.visualize_normal_estimation}')

            # Kalman params
            elif p.name == 'kalman_enabled':
                self.kalman_enabled = bool(p.value)
                if self.kalman_enabled:
                    self.pitch_kalman.reset()
                    self.yaw_kalman.reset()
                self.get_logger().info(f'Updated kalman_enabled to {self.kalman_enabled}')
            elif p.name == 'kalman_R':
                self.pitch_kalman.set_noise(R=float(p.value))
                self.yaw_kalman.set_noise(R=float(p.value))
                self.get_logger().info(f'Updated kalman_R to {float(p.value):.2e}')
            elif p.name == 'kalman_Q_angle':
                self._kalman_Q_angle_base = float(p.value)
                self.pitch_kalman.set_noise(Q_angle=self._kalman_Q_angle_base)
                self.yaw_kalman.set_noise(Q_angle=self._kalman_Q_angle_base)
                self.get_logger().info(f'Updated kalman_Q_angle(base) to {self._kalman_Q_angle_base:.2e}')
            elif p.name == 'kalman_Q_dangle':
                self._kalman_Q_dangle_base = float(p.value)
                self.pitch_kalman.set_noise(Q_dangle=self._kalman_Q_dangle_base)
                self.yaw_kalman.set_noise(Q_dangle=self._kalman_Q_dangle_base)
                self.get_logger().info(f'Updated kalman_Q_dangle(base) to {self._kalman_Q_dangle_base:.2e}')
            elif p.name == 'zeta':
                self.zeta = float(p.value)
            elif p.name == 'ie_clamp':
                self.ie_clamp = float(p.value)

            # ===== NEW: adaptive-Q params =====
            elif p.name == 'adaptive_q_enabled':
                self.adaptive_q_enabled = bool(p.value)
                self.get_logger().info(f'Updated adaptive_q_enabled to {self.adaptive_q_enabled}')
            elif p.name == 'adaptive_q_tau_thresh':
                self.adaptive_q_tau_thresh = float(p.value)
                self.get_logger().info(f'Updated adaptive_q_tau_thresh to {self.adaptive_q_tau_thresh:.3f}')
            elif p.name == 'adaptive_q_w_thresh':
                self.adaptive_q_w_thresh = float(p.value)
                self.get_logger().info(f'Updated adaptive_q_w_thresh to {self.adaptive_q_w_thresh:.3f}')
            elif p.name == 'adaptive_q_scale_moving':
                self.adaptive_q_scale_moving = float(p.value)
                self.get_logger().info(f'Updated adaptive_q_scale_moving to {self.adaptive_q_scale_moving:.1f}')

        result = SetParametersResult()
        result.successful = True
        return result


def main():
    rclpy.init()
    node = OrientationControlNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
