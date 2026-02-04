#!/usr/bin/env python3
import math
import numpy as np
import cv2
import open3d as o3d
import numpy.linalg as LA
import copy
import os
import threading

from sympy import zeta

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


def _quat_to_R_xyzw(x, y, z, w):
    """Return 3x3 rotation matrix from xyzw quaternion."""
    n = math.sqrt(x*x + y*y + z*z + w*w) + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ], dtype=np.float32)

def _R_to_quat_xyzw(R: np.ndarray) -> Quaternion:
    """Return xyzw quaternion from 3x3 rotation matrix R."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
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
    msg.is_dense = True  # no NaNs because we filtered them out
    msg.point_step = 12  # 3 * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points_xyz.astype(np.float32).tobytes()
    return msg


def _visualize_normal_estimation(pts_np: np.ndarray, centroid: np.ndarray, normal: np.ndarray,
                                  inlier_mask: np.ndarray = None, title: str = 'Normal Estimation'):
    """
    Visualize the normal estimation results using Open3D.

    Args:
        pts_np: (N, 3) array of all input points
        centroid: (3,) array of the selected centroid point
        normal: (3,) unit normal vector
        inlier_mask: Optional boolean mask for inliers (None = all points are inliers)
        title: Window title
    """
    geometries = []

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)

    # Color points: green for inliers, red for outliers
    if inlier_mask is not None:
        colors = np.zeros((len(pts_np), 3))
        colors[inlier_mask] = [0.0, 0.8, 0.0]  # Green for inliers
        colors[~inlier_mask] = [0.8, 0.0, 0.0]  # Red for outliers
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # All points are inliers (PCA case)
        pcd.paint_uniform_color([0.0, 0.8, 0.0])

    geometries.append(pcd)

    # Create centroid sphere (blue)
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    centroid_sphere.translate(centroid)
    centroid_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    geometries.append(centroid_sphere)

    # Create normal arrow (cyan)
    arrow_length = 0.05
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.002,
        cone_radius=0.004,
        cylinder_height=arrow_length * 0.8,
        cone_height=arrow_length * 0.2
    )
    # Rotate arrow to align with normal (default arrow points in +Z)
    z_axis = np.array([0.0, 0.0, 1.0])
    rotation_axis = np.cross(z_axis, normal)
    rotation_axis_norm = LA.norm(rotation_axis)
    if rotation_axis_norm > 1e-6:
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(centroid)
    arrow.paint_uniform_color([0.0, 0.8, 0.8])  # Cyan
    geometries.append(arrow)

    # Create fitted plane (semi-transparent gray)
    plane_size = 0.08
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.001)
    plane_mesh.translate([-plane_size / 2, -plane_size / 2, -0.0005])
    # Rotate plane to align with normal
    if rotation_axis_norm > 1e-6:
        plane_mesh.rotate(R, center=[0, 0, 0])
    plane_mesh.translate(centroid)
    plane_mesh.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(plane_mesh)

    # Create coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=800,
        height=600,
        point_show_normal=False
    )


def _pca_plane_normal(pts_np: np.ndarray, visualize: bool = False):
    """Return (centroid, unit normal) for best-fit plane to pts_np (N,3)."""
    # Pick centroid as the point closest to the z-axis (min radial distance in XY)
    r_xy = np.sqrt(pts_np[:, 0]**2 + pts_np[:, 1]**2)
    centroid_idx = np.argmin(r_xy)
    c = pts_np[centroid_idx]
    X = pts_np - c

    # 3x3 covariance; smallest eigenvalue's eigenvector is the plane normal
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, v = LA.eigh(C)
    n = v[:, 0]
    # Make direction consistent (toward camera -Z in depth cam frame)
    if n[2] < 0:
        n = -n
    n /= (LA.norm(n) + 1e-12)

    if visualize:
        _visualize_normal_estimation(
            pts_np, c, n,
            inlier_mask=None,  # PCA uses all points
            title='PCA Normal Estimation'
        )

    return c, n


def _ransac_plane_normal(pts_np: np.ndarray, n_iterations: int = 100, distance_threshold: float = 0.005, visualize: bool = False):
    """
    Return (centroid, unit normal) for RANSAC-fit plane to pts_np (N,3).

    RANSAC is more robust to outliers than PCA by iteratively fitting planes
    to random point samples and selecting the one with most inliers.

    Args:
        pts_np: (N, 3) array of 3D points
        n_iterations: Number of RANSAC iterations
        distance_threshold: Max distance from plane for a point to be an inlier (meters)
        visualize: If True, show Open3D visualization of the results

    Returns:
        (centroid, normal): centroid is the inlier point closest to z-axis,
                           normal is the unit normal vector pointing toward camera
    """
    n_points = pts_np.shape[0]
    if n_points < 3:
        # Not enough points, return default
        return np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)

    best_inlier_count = 0
    best_normal = None
    best_inlier_mask = None

    for _ in range(n_iterations):
        # Randomly sample 3 points to define a plane
        idx = np.random.choice(n_points, size=3, replace=False)
        p1, p2, p3 = pts_np[idx[0]], pts_np[idx[1]], pts_np[idx[2]]

        # Compute plane normal from two edge vectors
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_mag = LA.norm(n)
        if norm_mag < 1e-12:
            # Degenerate case (collinear points), skip
            continue
        n = n / norm_mag

        # Compute signed distance of all points to the plane
        # Plane equation: n · (x - p1) = 0
        distances = np.abs((pts_np - p1) @ n)

        # Count inliers
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = n
            best_inlier_mask = inlier_mask

    if best_normal is None or best_inlier_count < 3:
        # RANSAC failed, fall back to PCA on all points
        return _pca_plane_normal(pts_np, visualize=visualize)

    # Refine normal using PCA on inliers only
    inlier_pts = pts_np[best_inlier_mask]

    # Pick centroid as the inlier point closest to the z-axis (min radial distance in XY)
    r_xy = np.sqrt(inlier_pts[:, 0]**2 + inlier_pts[:, 1]**2)
    c = inlier_pts[np.argmin(r_xy)]

    # Refine plane normal using PCA on inliers
    X = inlier_pts - c
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, v = LA.eigh(C)
    n = v[:, 0]  # Eigenvector with smallest eigenvalue

    # Make direction consistent (toward camera, +Z in depth cam frame)
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
    """Return rotation-vector ω that rotates zc=[0,0,1] onto z_goal.
    Both vectors must be expressed in the same frame."""
    zc = np.array([0.0, 0.0, 1.0], dtype=np.float32)      # camera's current Z
    zg = z_goal.astype(np.float32)
    zg /= (LA.norm(zg) + 1e-12)                           # normalize

    c = float(np.clip(np.dot(zc, zg), -1.0, 1.0))         # cos(theta)
    theta = math.acos(c)
    print(f'Debug: theta={math.degrees(theta):.2f}°')

    axis = np.cross(zg, zc)
    n = LA.norm(axis)

    if n < 1e-9:
        # zc and zg are parallel
        if c > 0.0:
            #return np.zeros(3, dtype=np.float32)  # aligned
            return np.float32(0.0), np.zeros(3, dtype=np.float32)
        else:
            # 180°: pick x-axis as arbitrary rotation axis
            #return np.array([theta, 0.0, 0.0], dtype=np.float32)
            return np.float32(theta), np.array([1.0, 0.0, 0.0], dtype=np.float32)

    axis /= n
    return np.float32(theta), axis.astype(np.float32)

def _roll_error(x_current_world) -> float:
    """Return roll error (radians): angle between camera x-axis and world XY plane.

    Positive when x-axis points above XY plane, negative when below.
    """
    xc = x_current_world / (LA.norm(x_current_world) + 1e-12)
    # Angle to XY plane is arcsin of z-component for a unit vector
    theta = math.asin(float(np.clip(xc[2], -1.0, 1.0)))
    return theta

class OrientationControlNode(Node):
    # Node that gives desired EOAT pose based on depth image, bounding box, and cropping
    def __init__(self):
        super().__init__('pd_controller')

        sub_cb_group = ReentrantCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- Parameters ----
        self.declare_parameters(
            namespace='',
            parameters=[
                ('depth_topic', '/camera/d405_camera/depth/image_rect_raw/compressed'),
                ('camera_info_topic', '/camera/d405_camera/depth/camera_info'),
                ('bounding_box_topic', '/viewpoint_generation/bounding_box_marker'),
                ('joy_topic', 'joy'),  # Topic for incoming Joy messages
                ('enable_button', 0),  # Button index to enable orientation control
                ('dmap_filter_min', 0.07),
                ('dmap_filter_max', 0.50),
                ('viz_enable', True),
                ('publish_pointcloud', True),
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
                ('save_data', False),
                ('data_path', '/tmp'),
                ('object', ''),
                ('sphere_mass', 5.0),                      # kg
                ('sphere_radius', 0.1),                    # m (for inertia and drag calc)
                ('fluid_viscosity', 0.0010016),                  # Pa·s (for drag calc)
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
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)
        self.pcd_downsampling_stride = int(self.get_parameter('pcd_downsampling_stride').value)
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        
        # Crop reads
        self.crop_radius   = float(self.get_parameter('crop_radius').value)
        self.crop_z_min    = float(self.get_parameter('crop_z_min').value)
        self.crop_z_max    = float(self.get_parameter('crop_z_max').value)
        self.standoff_m    = float(self.get_parameter('standoff_m').value)
        self.standoff_mode = str(self.get_parameter('standoff_mode').value).lower()
        self.ema_enable = bool(self.get_parameter('ema_enable').value)
        self.ema_tau    = float(self.get_parameter('ema_tau').value)

        # EMA state (persist across frames)
        self._ema_normal   = None      # np.ndarray (3,)
        self._ema_centroid = None      # np.ndarray (3,)
        self._ema_last_t   = None      # float seconds

        # ---- Initialize bbox fields so they're always present ----
        # Use infinities so "no box yet" behaves like "pass-through".
        self.bbox_min = np.array([-float('inf'), -float('inf'), -float('inf')], dtype=float)  # [xmin, ymin, zmin]
        self.bbox_max = np.array([ float('inf'),  float('inf'),  float('inf')], dtype=float)  # [xmax, ymax, zmax]
        self.main_camera_frame = self.get_parameter('main_camera_frame').get_parameter_value().string_value

        self.normal_estimation_method = self.get_parameter('normal_estimation_method').get_parameter_value().string_value.upper()
        self.visualize_normal_estimation = bool(self.get_parameter('visualize_normal_estimation').value)
        self._orientation_control_before_viz = False  # Stores orientation_control state before visualization

        self.d_min = float(self.get_parameter('d_min').value) # Minimum focal distance (meters)
        self.d_max = float(self.get_parameter('d_max').value) # Maximum focal distance (meters)
        self.v_max = float(self.get_parameter('v_max').value) # Maximum linear velocity (meters/second)
        self.theta_max_deg = float(self.get_parameter('theta_max_deg').value) # Maximum angular displacement (degrees)

        self.Kp = float(self.get_parameter('Kp').value)
        self.Ki = float(self.get_parameter('Ki').value)    # integral gain about camera Z  # <<< NEW
        self.Kd = float(self.get_parameter('Kd').value)  # <<< NEW
        self.no_target_timeout_s = float(self.get_parameter('no_target_timeout_s').value)
        self.controller_type = str(self.get_parameter('controller_type').value).upper()
        self.integral_alpha = float(self.get_parameter('integral_alpha').value)
        
        # Loss tracking
        self._had_target_last_cycle = False
        self._last_target_time_s = None  # float seconds of last valid crop/pose
        self.orientation_control_enabled = bool(self.get_parameter('orientation_control_enabled').value)
        
        # save data parameters
        self.save_data = bool(self.get_parameter('save_data').value)
        self.data_path = self.get_parameter('data_path').get_parameter_value().string_value
        self.object = self.get_parameter('object').get_parameter_value().string_value
        self.storage_options = StorageOptions(
            uri=self.data_path, storage_id='sqlite3')
        self.converter_options = ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr')
        self.writer = SequentialWriter()
        self.topic_info = TopicMetadata(
            name='orientation_control_data',
            type='viewpoint_generation_interfaces/msg/OrientationControlData',
            serialization_format='cdr')
        
        
        self.mass_B = float(self.get_parameter('sphere_mass').value)  # mass of the object in kg for force control   # <<< NEW
        self.sphere_radius = float(self.get_parameter('sphere_radius').value)
        self.fluid_viscosity = float(self.get_parameter('fluid_viscosity').value)
        self.update_inertia_and_drag(self.mass_B, self.sphere_radius, self.fluid_viscosity)

        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.bridge = CvBridge()
        self.K = None
        self.depth_frame_id = None



        # TF2 for transforms to target_frame
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))  # Increased cache for startup
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._tf_ready = False  # Flag to track TF availability

        # Orientation control data msg
        self.ocd = OrientationControlData()

        # Subs
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, qos)
        self.sub_depth = self.create_subscription(CompressedImage, self.depth_topic, self.on_depth, 1, callback_group=sub_cb_group)
        self.bbox = self.create_subscription(Marker, self.bounding_box_topic, self.on_bbox, qos)
        self.create_subscription(TwistStamped, f'/servo_node/delta_twist_cmds', self.on_delta_twist, qos)
        self.create_subscription(WrenchStamped, f'/teleop/wrench_cmds', self.on_wrench_cmd, qos)
        # Joy sub
        self.last_joy_msg = None
        self.joy_sub = self.create_subscription(
            Joy,
            joy_topic,
            self.joy_callback,
            qos,
        )
        self.depth_msg = None

        # Store callback group for later timer creation
        self._timer_cb_group = timer_cb_group
        self._process_timer = None

        # Start with a TF readiness check timer instead of processing timer
        self._startup_timer = self.create_timer(0.5, self._check_tf_ready, callback_group=timer_cb_group)

        # Pubs
        self.eoat_pointcloud_publisher = self.create_publisher(
            PointCloud2, f'/camera/d405_camera/depth/eoat_points_{self.main_camera_frame}_bbox', 10
        ) if self.publish_pointcloud else None

        self.fov_pointcloud_publisher = self.create_publisher(
            PointCloud2, f'/camera/d405_camera/depth/fov_points_{self.main_camera_frame}_bbox', 10
        ) if self.publish_pointcloud else None
        self.normal_estimate_pub = self.create_publisher(
            PoseStamped, f'/{self.get_name()}/crop_normal', 10
        )
        self.pub_eoat_pose_crop = self.create_publisher(
            PoseStamped, f'/{self.get_name()}/eoat_desired_pose_in_{self.main_camera_frame}', 10
        )

        # TF broadcaster for surface target pose
        self.surface_target_frame = self.get_parameter('surface_target_frame').get_parameter_value().string_value
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_z_rotvec_err = self.create_publisher(
            Vector3Stamped, f'/{self.get_name()}/z_axis_rotvec_error_in_{self.main_camera_frame}', 10
        )
        self.pub_wrench_cmd = self.create_publisher(
            WrenchStamped, f'/{self.get_name()}/wrench_cmds', 10
        )
        self.distance_pub = self.create_publisher(
            Float64, f'/{self.get_name()}/focal_distance_m', 10
        )

        self.force_B = np.zeros(3, dtype=np.float32)  # <<< NEW
        self.lin_vel_cam = np.zeros(3, dtype=np.float32)  # <<< NEW
        self.rot_vel_cam = np.zeros(3, dtype=np.float32)  # <<< NEW
        self.tau_B = np.zeros(3, dtype=np.float32)  # <<< NEW
        self.force_teleop = np.zeros(3, dtype=np.float32)  # <<< NEW
        self.torque_teleop = np.zeros(3, dtype=np.float32)  # <<< NEW
        # keep a “last torque” handy so we can report it in the bundle
        self._last_tau = np.zeros(3, dtype=np.float32)
        self._last_force = np.zeros(3, dtype=np.float32)   # <<< NEW
        # PD state: last rotation-vector error and timestamp                # <<< NEW
        self._last_rotvec_err = None                                       # <<< NEW
        self._last_err_t = None  
        self._int_rotvec_err = np.zeros(3, dtype=np.float32)   
        self.last_e = 0.0                                        # <<< NEW
        self.int_e =0.0

        # PID state for force (position) control                         # <<< NEW FOR FORCE PID >>>
        #self._last_pos_err = None                                       # <<< NEW FOR FORCE PID >>>
        #self._last_pos_t = None                                         # <<< NEW FOR FORCE PID >>>
        #self._int_pos_err = np.zeros(3, dtype=np.float32)               # <<< NEW FOR FORCE PID >>>

        # Thread-safe state for depth→timer communication
        self._measurement_lock = threading.Lock()
        self._latest_measurement = {
            'valid': False,
            'centroid': np.zeros(3, dtype=np.float32),  # Smoothed, in main_camera_frame
            'normal': np.zeros(3, dtype=np.float32),    # Smoothed, in main_camera_frame
            'timestamp': 0.0,
            'pts_crop': None,       # For visualization/debug
            'pts_bbox': None,       # For visualization/debug
            'rot_vec_error': np.zeros(3, dtype=np.float32),
            'theta_err': 0.0,
            'axis_err': np.zeros(3, dtype=np.float32),
            'd': 0.0,               # Focal distance for controller
            'roll_error': 0.0,      # Roll error about Z-axis
            'r': np.zeros(3, dtype=np.float32),  # Vector from centroid to origin
        }

        self.get_logger().info(
            'Background remover running:\n'
            f'  depth_topic={self.depth_topic}\n'
            f'  camera_info_topic={self.camera_info_topic}\n'
            f'  bounding_box_topic={self.bounding_box_topic}\n'
            f'  dmap_filter_min={self.dmap_filter_min:.3f}, dmap_filter_max={self.dmap_filter_max:.3f}\n'
            f'  viz_enable={self.viz_enable}, publish_pointcloud={self.publish_pointcloud}\n'
            f'  pcd_downsampling_stride={self.pcd_downsampling_stride}\n'
            f'  target_frame={self.target_frame}'
            f'  main_camera_frame={self.main_camera_frame}'
            f'  crop_radius={self.crop_radius:.3f}, crop_z=[{self.crop_z_min:.3f},{self.crop_z_max:.3f}]'
            f'  standoff_mode={self.standoff_mode}, standoff_m={self.standoff_m:.3f}'
        )

        # ---------------- Parameter update callback ----------------
        self.add_on_set_parameters_callback(self._on_param_update)


    def update_inertia_and_drag(self, mass: float, radius: float, fluid_viscosity: float):
        """Update inertia and drag coefficients based on mass, radius, and fluid viscosity."""
        # Inertia for solid sphere: I = 2/5 m r²
        I = (2.0 / 5.0) * mass * (radius ** 2)
        self.inertia_B = I

        # Linear drag coefficients: D = 6πμr
        self.linear_drag = 6.0 * 3.141592653589793 * fluid_viscosity * radius

        # Rotational drag coefficients: D_rot = 2.4πμr³ (breaks Stoke's Law but preserves v = rω)
        self.angular_drag = 2.4 * 3.141592653589793 * fluid_viscosity * (radius ** 3)

        self.get_logger().info(
            f'Updated inertia to {self.inertia_B} kg·m² and drag to {self.linear_drag} N·s/m, {self.angular_drag} N·m·s/rad'
        )

    def _check_tf_ready(self):
        """Check if required TF frames are available before starting main processing."""
        if self._tf_ready:
            return

        # Check if we have camera info (needed to know the depth frame)
        if self.depth_frame_id is None:
            self.get_logger().info(
                'Waiting for camera info to determine depth frame...',
                throttle_duration_sec=2.0
            )
            return

        # Required transform pairs to check
        tf_checks = [
            (self.target_frame, self.main_camera_frame),
            (self.target_frame, self.depth_frame_id),
        ]

        all_ready = True
        for target, source in tf_checks:
            try:
                # Use a longer timeout for startup checks
                self.tf_buffer.lookup_transform(
                    target, source,
                    Time(sec=0, nanosec=0),
                    timeout=Duration(seconds=0.1)
                )
            except TransformException as e:
                self.get_logger().info(
                    f'Waiting for TF: {target} <- {source}',
                    throttle_duration_sec=2.0
                )
                all_ready = False
                break

        if all_ready:
            self._tf_ready = True
            self.get_logger().info(
                'TF tree ready. Starting orientation control processing.'
            )
            # Cancel startup timer and start the main processing timer
            self._startup_timer.cancel()
            self._process_timer = self.create_timer(
                0.1, self.process_controller, callback_group=self._timer_cb_group
            )

    def joy_callback(self, msg):
        """
        Process incoming Joy messages and turn orientation control on/off.
        """
        if not self.last_joy_msg:
            self.last_joy_msg = msg
            return
        
        # Turn orientation control on/off based on button presses
        if not self.last_joy_msg.buttons[self.enable_button]:
            if msg.buttons[self.enable_button] and not self.orientation_control_enabled:
                self.enable_orientation_control()
            elif msg.buttons[self.enable_button] and self.orientation_control_enabled:
                self.disable_orientation_control()
        # Turn on normal_estimation_visualization on button 9 press
        if not self.last_joy_msg.buttons[9]:
            if msg.buttons[9] and not self.visualize_normal_estimation:
                self.enable_normal_estimation_viz()
            elif msg.buttons[9] and self.visualize_normal_estimation:
                self.disable_normal_estimation_viz()

        self.last_joy_msg = msg


    def on_delta_twist(self, msg: TwistStamped):
        # Receive delta twist commands from servo node
        # We only care about torque (angular) components for orientation control
        self.lin_vel_cam[0] = msg.twist.linear.x
        self.lin_vel_cam[1] = msg.twist.linear.y
        self.lin_vel_cam[2] = msg.twist.linear.z
        self.rot_vel_cam[0] = msg.twist.angular.x
        self.rot_vel_cam[1] = msg.twist.angular.y
        self.rot_vel_cam[2] = msg.twist.angular.z

    def on_wrench_cmd(self, msg: WrenchStamped):
        # Receive wrench commands from teleop node
        self.force_teleop[0] = msg.wrench.force.x
        self.force_teleop[1] = msg.wrench.force.y
        # self.force_teleop[2] = msg.wrench.force.z
        self.torque_teleop[0] = msg.wrench.torque.x
        self.torque_teleop[1] = msg.wrench.torque.y
        self.torque_teleop[2] = msg.wrench.torque.z 

    def enable_orientation_control(self):
        # Open the bag file for writing
        if self.save_data:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            uri = f'{self.data_path}/{self.object}_{self.get_clock().now().to_msg().sec}.bag'
            self.storage_options = StorageOptions(
                uri=uri, storage_id='sqlite3')
            self.get_logger().info(f'Opening bag file at: {uri}')
            self.writer.open(self.storage_options, self.converter_options)
            self.writer.create_topic(self.topic_info)

        # Reset controller internal state so first dt_ctrl is not "time since last enable"
        self._last_err_t = None
        self.last_e = 0.0
        self.de = 0.0
        #self._int_rotvec_err = np.zeros(3, dtype=np.float32)
        self.int_e = 0.0
        self.orientation_control_enabled = True
        self.get_logger().info('Orientation control ENABLED.')

    def disable_orientation_control(self):
        self.orientation_control_enabled = False
        self.get_logger().info('Orientation control DISABLED.')
        # Close the bag file
        if self.save_data:
            self.writer.close()
            self.get_logger().info(f'Closed bag file.')
            debag(self.storage_options.uri)
            # Update parameters or state as needed

    # Camera intrinsics
    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.depth_frame_id = msg.header.frame_id

    # Bounding box updates
    def on_bbox(self, msg: Marker):
        if msg.type != Marker.CUBE:
            self.get_logger().warn(f'Ignoring non-CUBE bounding box marker of type {msg.type}')
            return
        if msg.scale.x <= 0.0 or msg.scale.y <= 0.0 or msg.scale.z <= 0.0:
            self.get_logger().warn(f'Ignoring invalid bounding box with non-positive scale {msg.scale.x}, {msg.scale.y}, {msg.scale.z}')
            return
        # Axis-aligned box only (no rotation)
        if abs(msg.pose.orientation.x) > 1e-3 or abs(msg.pose.orientation.y) > 1e-3 or abs(msg.pose.orientation.z) > 1e-3:
            self.get_logger().warn(f'Ignoring non-axis-aligned bounding box with orientation {msg.pose.orientation}')
            return

        cx, cy, cz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        sx, sy, sz = msg.scale.x, msg.scale.y, msg.scale.z

        # Half dimensions
        half_sizes = np.array([sx, sy, sz], dtype=float) * 0.5
        center     = np.array([cx, cy, cz], dtype=float)

        # Now update bbox as arrays
        self.bbox_min = center - half_sizes
        self.bbox_max = center + half_sizes

    def _ema_update(self, prev, x, alpha):
        """One EMA step for vectors (broadcast-safe)."""
        if prev is None:
            return x.copy()
        return (1.0 - alpha) * prev + alpha * x

    def on_depth(self, msg: CompressedImage):
        """Depth image callback - process pointcloud and estimate normal."""
        self.ocd.header = msg.header
        self.depth_msg = msg  # Keep for backwards compatibility
        self.process_depth(msg)

    def process_depth(self, msg: CompressedImage):
        """Process depth image: decompress, build pointcloud, estimate normal.

        This runs at depth camera rate (~25Hz) and stores results for the controller.
        """
        # --- Time bookkeeping for watchdog ---
        stamp = msg.header.stamp
        now_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        measurement_ok = False

        centroid_out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        normal_out = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        goal_pose_out = None
        rotvec_err_out = np.zeros(3, dtype=np.float32)

        # Decompress compressedDepth format -> numpy
        np_arr = np.frombuffer(msg.data, np.uint8)
        if np_arr.size <= 12:
            self.get_logger().warn("Empty or invalid compressedDepth data, skipping frame")
            return
        png_data = np_arr[12:]
        depth = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)
        if depth is None:
            self.get_logger().warn("Failed to decode depth image, skipping frame")
            return

        # Store decompressed image for ocd message
        if depth.dtype == np.uint16:
            self.ocd.depth_image = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        else:
            self.ocd.depth_image = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        self.ocd.depth_image.header = msg.header

        # Normalize to meters
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 1e-3
        else:
            depth_m = depth.astype(np.float32)

        # Build mask
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        mask = valid & (depth_m >= self.dmap_filter_min) & (depth_m <= self.dmap_filter_max)

        # Build filtered depth
        depth_filtered = depth_m.copy()
        depth_filtered[~mask] = np.nan

        self.ocd.depth_filtered_image = self.bridge.cv2_to_imgmsg(depth_filtered.astype(np.float32), encoding='32FC1')
        self.ocd.dmap_filter_min = self.dmap_filter_min
        self.ocd.dmap_filter_max = self.dmap_filter_max

        points1 = np.zeros((0, 3), dtype=np.float32)
        points2 = np.zeros((0, 3), dtype=np.float32)

        # Local variables for measurement storage
        theta_err = 0.0
        axis_err = np.zeros(3, dtype=np.float32)
        d = 0.0
        roll_error = 0.0
        r = np.zeros(3, dtype=np.float32)

        # Point cloud (if enabled and we have intrinsics)
        if self.K is not None:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            stride = max(1, self.pcd_downsampling_stride)
            ys, xs = np.where(mask)

            if stride > 1:
                ys = ys[::stride]; xs = xs[::stride]

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
                    r2 = Xc*Xc + Yc*Yc
                    sel_crop = (
                        (Zc >= self.crop_z_min) & (Zc <= self.crop_z_max) &
                        (r2 <= (self.crop_radius * self.crop_radius))
                    )

                    pts_crop = np.ascontiguousarray(pts_bbox_out[sel_crop])
                    points2 = pts_crop

                    if pts_crop.shape[0] >= 10:
                        # ---------------- NORMAL ESTIMATION --------------- #
                        if self.normal_estimation_method == 'PCA':
                            centroid, normal = _pca_plane_normal(pts_crop, visualize=self.visualize_normal_estimation)
                        elif self.normal_estimation_method == 'RANSAC':
                            centroid, normal = _ransac_plane_normal(pts_crop, visualize=self.visualize_normal_estimation)

                        # Restore state after visualization window is closed
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

                        # Transform centroid and normal to object_frame
                        T = self.tf_buffer.lookup_transform(
                            'object_frame', self.main_camera_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                        q = T.transform.rotation
                        R_tf = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
                        x_cf, y_cf, z_cf = R_tf[:, 0], R_tf[:, 1], R_tf[:, 2]
                        t = T.transform.translation
                        t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                        cen_s_obj = (R_tf @ cen_s) + t_vec
                        nrm_s_obj = R_tf @ nrm_s

                        # Publish surface target pose (at depth rate)
                        surface_target_pose = PoseStamped()
                        surface_target_pose.header = msg.header
                        surface_target_pose.header.frame_id = 'object_frame'
                        surface_target_pose.pose.position.x = float(cen_s_obj[0])
                        surface_target_pose.pose.position.y = float(cen_s_obj[1])
                        surface_target_pose.pose.position.z = float(cen_s_obj[2])
                        surface_target_pose.pose.orientation = _quaternion_from_z(nrm_s_obj)
                        self.normal_estimate_pub.publish(surface_target_pose)

                        # Broadcast surface target transform (at depth rate)
                        tf_msg = TransformStamped()
                        tf_msg.header = surface_target_pose.header
                        tf_msg.child_frame_id = self.surface_target_frame
                        tf_msg.transform.translation.x = surface_target_pose.pose.position.x
                        tf_msg.transform.translation.y = surface_target_pose.pose.position.y
                        tf_msg.transform.translation.z = surface_target_pose.pose.position.z
                        tf_msg.transform.rotation = surface_target_pose.pose.orientation
                        self.tf_broadcaster.sendTransform(tf_msg)

                        # Compute standoff
                        if self.standoff_mode == 'euclidean':
                            d = float(LA.norm(cen_s))
                        elif self.standoff_mode == 'along_normal':
                            d = float(max(0.0, float(np.dot(nrm_s, cen_s))))
                        else:
                            d = self.standoff_m

                        # Desired EOAT pose
                        p_des_cf = cen_s - d*nrm_s
                        q_des_cf = _quaternion_from_z(nrm_s_obj)
                        R_des_cf = _quat_to_R_xyzw(q_des_cf.x, q_des_cf.y, q_des_cf.z, q_des_cf.w)
                        R_cam = R_tf.T @ R_des_cf
                        q_des_cf = _R_to_quat_xyzw(R_cam)

                        # Publish desired EOAT pose (at depth rate)
                        eoat_cf = PoseStamped()
                        eoat_cf.header = msg.header
                        eoat_cf.header.frame_id = self.main_camera_frame
                        eoat_cf.pose.position.x = float(p_des_cf[0])
                        eoat_cf.pose.position.y = float(p_des_cf[1])
                        eoat_cf.pose.position.z = float(p_des_cf[2])
                        eoat_cf.pose.orientation = q_des_cf
                        self.pub_eoat_pose_crop.publish(eoat_cf)

                        # Z-axis alignment error
                        theta_err, axis_err = _z_axis_rotvec_error(nrm_s.astype(np.float32))
                        rot_vec_error = axis_err * theta_err

                        # Roll error
                        roll_error = _roll_error(x_cf)
                        rot_vec_error[2] = roll_error

                        centroid_out = cen_s.astype(np.float32)
                        normal_out = nrm_s.astype(np.float32)
                        rotvec_err_out = rot_vec_error.astype(np.float32)
                        goal_pose_out = eoat_cf

                        # Publish rotation error (at depth rate)
                        err_msg = Vector3Stamped()
                        err_msg.header = msg.header
                        err_msg.header.frame_id = self.main_camera_frame
                        err_msg.vector.x, err_msg.vector.y, err_msg.vector.z = map(float, rot_vec_error)
                        self.pub_z_rotvec_err.publish(err_msg)

                        measurement_ok = True
                        self._had_target_last_cycle = True
                        self._last_target_time_s = now_s

        # Publish pointclouds (at depth rate)
        pcd2_msg = make_pointcloud2(points1, frame_id=self.main_camera_frame, stamp=msg.header.stamp)
        self.eoat_pointcloud_publisher.publish(pcd2_msg)

        pcd2_msg_final = make_pointcloud2(points2, frame_id=self.main_camera_frame, stamp=msg.header.stamp)
        self.fov_pointcloud_publisher.publish(pcd2_msg_final)

        # Store in ocd
        self.ocd.cloud_bbox = pcd2_msg
        self.ocd.cloud_crop = pcd2_msg_final
        self.ocd.centroid = Point(x=float(centroid_out[0]), y=float(centroid_out[1]), z=float(centroid_out[2]))
        self.ocd.normal = Vector3(x=float(normal_out[0]), y=float(normal_out[1]), z=float(normal_out[2]))

        if goal_pose_out is not None:
            self.ocd.goal_pose = goal_pose_out
        else:
            empty_pose = PoseStamped()
            empty_pose.header = self.ocd.header
            self.ocd.goal_pose = empty_pose

        self.ocd.rotvec_error = Vector3(x=float(rotvec_err_out[0]), y=float(rotvec_err_out[1]), z=float(rotvec_err_out[2]))

        # Store results thread-safely for controller timer
        with self._measurement_lock:
            self._latest_measurement['valid'] = measurement_ok
            self._latest_measurement['centroid'] = centroid_out.copy()
            self._latest_measurement['normal'] = normal_out.copy()
            self._latest_measurement['timestamp'] = now_s
            self._latest_measurement['rot_vec_error'] = rotvec_err_out.copy()
            self._latest_measurement['theta_err'] = float(theta_err)
            self._latest_measurement['axis_err'] = axis_err.copy()
            self._latest_measurement['pts_crop'] = points2
            self._latest_measurement['pts_bbox'] = points1
            self._latest_measurement['d'] = float(d)
            self._latest_measurement['roll_error'] = float(roll_error)
            self._latest_measurement['r'] = r.copy()

        # Handle watchdog when no valid target
        if not measurement_ok:
            if (self._last_target_time_s is None) or ((now_s - self._last_target_time_s) > self.no_target_timeout_s):
                if self._had_target_last_cycle:
                    self.get_logger().warn('No valid target: lost')
                self._ema_normal = None
                self._ema_centroid = None
                self._ema_last_t = None
                self._had_target_last_cycle = False

    def _handle_no_measurement(self):
        """Handle case when no valid measurement available - publish zero wrench."""
        now_s = self.get_clock().now().nanoseconds * 1e-9
        print("No valid measurement available.")

        if self._had_target_last_cycle:
            self._last_target_time_s = now_s
            self._had_target_last_cycle = False

        time_since_target = 0.0 if self._last_target_time_s is None else (now_s - self._last_target_time_s)

        if time_since_target > self.no_target_timeout_s:
            self._last_rotvec_err = None
            self._last_err_t = None
            self._int_rotvec_err = np.zeros(3, dtype=np.float32)
            self.int_e = 0.0
            self.last_e = 0.0

        # Publish zero wrench
        w = WrenchStamped()
        w.header.stamp = self.get_clock().now().to_msg()
        w.header.frame_id = self.main_camera_frame
        self.pub_wrench_cmd.publish(w)

    def process_controller(self):
        """Timer callback - run controller at fixed rate (~10Hz)."""
        # Read latest measurement thread-safely
        with self._measurement_lock:
            if not self._latest_measurement['valid']:
                self._handle_no_measurement()
                return

            cen_s = self._latest_measurement['centroid'].copy()
            nrm_s = self._latest_measurement['normal'].copy()
            rot_vec_error = self._latest_measurement['rot_vec_error'].copy()
            theta_err = self._latest_measurement['theta_err']
            axis_err = self._latest_measurement['axis_err'].copy()
            meas_time = self._latest_measurement['timestamp']
            d = self._latest_measurement['d']
            roll_error = self._latest_measurement['roll_error']
            r = self._latest_measurement['r'].copy()

        # Check staleness
        now_s = self.get_clock().now().nanoseconds * 1e-9
        # if (now_s - meas_time) > self.no_target_timeout_s:
        #     self._handle_no_measurement()
        #     return

        tau_out = np.zeros(3, dtype=np.float32)

        if self.orientation_control_enabled:
            # --- PID control ---
            self.anti_windup_enabled = True
            self.aw_Tt = 0.05
            self.int_limit = 1e3
            self.e = -theta_err
            dt_ctrl = 0.0
            if self._last_err_t is not None:
                dt_ctrl = max(1e-6, now_s - self._last_err_t)
                print(f"dt_ctrl: {dt_ctrl}")
                #dt_ctrl=0.001
                self.de = (self.e - self.last_e) / dt_ctrl
            else:
                dt_ctrl = 0.0
                self.de = 0.0

            self._last_err_t = now_s
            self.last_e = float(self.e)
            self._last_rotvec_err = rot_vec_error.copy()

            inertia_A = self.inertia_B + self.mass_B * d**2
            tau_max = self.linear_drag * d * self.v_max
            omega_n_max = math.sqrt(tau_max / (inertia_A * self.theta_max_deg * 3.141592653589793 / 180.0))
            omega_n = omega_n_max

            p_1 = -omega_n
            p_2 = -omega_n
            p_3 = -self.integral_alpha * omega_n

            if self.controller_type == 'PD':
                self.Kp = inertia_A * (p_1 * p_2)
                self.Kd = -inertia_A * (p_1 + p_2) - self.linear_drag * d * d
                self.Ki = 0.0
            elif self.controller_type == 'PID':
                self.Kp = inertia_A * (p_1 * p_2 + p_1 * p_3 + p_2 * p_3)
                self.Ki = -inertia_A * (p_1 * p_2 * p_3)
                self.Kd = -inertia_A * (p_1 + p_2 + p_3) - self.linear_drag * d * d
            else:
                self.get_logger().warn(f'Unknown controller_type {self.controller_type}, defaulting to PD')
                self.Kp = inertia_A * (p_1 * p_2)
                self.Kd = -inertia_A * (p_1 + p_2) - self.linear_drag * d * d
                self.Ki = 0.0

            print(f"e: {self.e}, de: {self.de}, int_e: {self.int_e}")
            print(f"Kp: {self.Kp}, Ki: {self.Ki}, Kd: {self.Kd}")
            print(f'd: {d}')
            print(f"Kp*e: {self.Kp * self.e}, Ki*int_e: {self.Ki * self.int_e}, Kd*de: {self.Kd * self.de}")
            tau_A = (self.Kp * self.e + self.Ki * self.int_e + self.Kd * self.de)
            tau_sat = np.clip(tau_A, -tau_max, tau_max)
            print(f"tau_A: {tau_A}, tau_sat: {tau_sat}, tau_max: {tau_max}")

            # Anti-windup
            if (getattr(self, "anti_windup_enabled", True) and (self.Ki > 1e-9) and (dt_ctrl > 0.0)):
                Tt = float(getattr(self, "aw_Tt", 0.05))
                print(f"(tau_sat - tau_A): {(tau_sat - tau_A)}")
                print(f"Ki * Tt: {self.Ki * Tt}")
                i_dot = self.e + (tau_sat - tau_A) / (self.Ki * max(1e-6, Tt))
                print(f"i_dot: {i_dot}")
                self.int_e += i_dot * dt_ctrl
                print(f"Updated int_e (with anti-windup): {self.int_e}")
            else:
                if dt_ctrl > 0.0:
                    self.int_e += self.e * dt_ctrl

            int_lim = float(getattr(self, "int_limit", 1e3))
            self.int_e = np.clip(self.int_e, -int_lim, int_lim)
            tau_A = tau_sat
            tau_A_vec = tau_A * axis_err
            moment_tele_A = np.cross(r, self.force_teleop)
            force_drag_B = -self.linear_drag * self.lin_vel_cam
            moment_drag_B = -self.angular_drag * self.rot_vel_cam
            moment_dragB_A = np.cross(r, force_drag_B)
            self.ang_acc = (moment_dragB_A + tau_A_vec + moment_tele_A) / inertia_A

            tau_out = tau_A_vec.copy()
            self._last_tau = tau_A_vec.copy()
            self.force_B = self.mass_B * np.cross(self.ang_acc, r) - force_drag_B - self.force_teleop
            self.tau_B = self.inertia_B * self.ang_acc - moment_drag_B - self.torque_teleop
            self.tau_B[2] = roll_error * self.Kp

            self.distance_pub.publish(Float64(data=float(LA.norm(cen_s))))

            w = WrenchStamped()
            w.header.stamp = self.get_clock().now().to_msg()
            w.header.frame_id = self.main_camera_frame
            w.wrench.force.x = float(self.force_B[0])
            w.wrench.force.y = float(self.force_B[1])
            w.wrench.force.z = float(self.force_B[2])
            w.wrench.torque.x = float(self.tau_B[0])
            w.wrench.torque.y = float(self.tau_B[1])
            w.wrench.torque.z = float(self.tau_B[2])
            self.pub_wrench_cmd.publish(w)
        else:
            # Control disabled - publish zero wrench
            tau_out = np.zeros(3, dtype=np.float32)
            self._last_tau = tau_out.copy()
            self._last_force = np.zeros(3, dtype=np.float32)

            w = WrenchStamped()
            w.header.stamp = self.get_clock().now().to_msg()
            w.header.frame_id = self.main_camera_frame
            self.pub_wrench_cmd.publish(w)

        # Update OCD with controller outputs
        self.ocd.torque_cmd = Vector3(x=float(tau_out[0]), y=float(tau_out[1]), z=float(tau_out[2]))
        self.ocd.cam_force_cmd = Vector3(x=float(self.force_B[0]), y=float(self.force_B[1]), z=float(self.force_B[2]))
        self.ocd.cam_torque_cmd = Vector3(x=float(self.tau_B[0]), y=float(self.tau_B[1]), z=float(self.tau_B[2]))

        self.ocd.k_p = float(self.Kp)
        self.ocd.k_d = float(self.Kd)
        if hasattr(self.ocd, 'k_i'):
            self.ocd.k_i = float(self.Ki)

        self.ocd.main_camera_frame = str(self.main_camera_frame)
        self.ocd.target_frame = str(self.target_frame)
        self.ocd.ema_enable = bool(self.ema_enable)
        self.ocd.ema_tau_s = float(self.ema_tau)
        self.ocd.no_target_timeout_s = float(self.no_target_timeout_s)

        if self.save_data:
            self.bag_orientation_control_data()


    def bag_orientation_control_data(self):
        if self.orientation_control_enabled:
            self.writer.write(
                'orientation_control_data',
                serialize_message(self.ocd),
                self.get_clock().now().nanoseconds
            )

    def enable_normal_estimation_viz(self):
        if not self.visualize_normal_estimation:
            self.visualize_normal_estimation = True
            self._orientation_control_before_viz = self.orientation_control_enabled
            # Publish zero wrench to stop motion during visualization
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

    # Param updates
    def _on_param_update(self, params):
        for p in params:
            if p.name == 'dmap_filter_min':
                self.dmap_filter_min = float(p.value); self.get_logger().info(f'dmap_filter_min -> {self.dmap_filter_min:.3f} m')
            elif p.name == 'dmap_filter_max':
                self.dmap_filter_max = float(p.value); self.get_logger().info(f'dmap_filter_max  -> {self.dmap_filter_max:.3f} m')
            elif p.name == 'publish_pointcloud':
                self.publish_pointcloud = bool(p.value)
                if self.publish_pointcloud:
                    if self.fov_pointcloud_publisher is None:
                        self.fov_pointcloud_publisher = self.create_publisher(
                            PointCloud2, f'/camera/d405_camera/depth/fov_points_{self.main_camera_frame}_bbox', 10
                        )
                if not self.publish_pointcloud:
                    self.fov_pointcloud_publisher = None
            elif p.name == 'pcd_downsampling_stride':
                self.pcd_downsampling_stride = max(1, int(p.value))
            elif p.name == 'target_frame':
                self.target_frame = str(p.value)
            elif p.name == 'main_camera_frame':
                new_frame = str(p.value)
                if new_frame != self.main_camera_frame:
                    self.main_camera_frame = new_frame
                    if self.publish_pointcloud:
                        # Recreate out publisher with new name
                        self.fov_pointcloud_publisher = self.create_publisher(
                            PointCloud2, f'/camera/d405_camera/depth/fov_points_{self.main_camera_frame}_bbox', 10)
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
            # elif p.name == 'torque_limit':
            #     self.torque_limit = float(p.value)
            elif p.name == 'no_target_timeout_s':
                self.no_target_timeout_s = float(p.value)
            # Save Data Parameters
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

        result = SetParametersResult()
        result.successful = True

        return result


def main():
    rclpy.init()
    node = OrientationControlNode()

    # Use MultiThreadedExecutor with at least 2 threads
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
