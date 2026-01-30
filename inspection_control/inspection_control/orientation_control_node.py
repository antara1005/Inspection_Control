#!/usr/bin/env python3
import math
import numpy as np
import cv2
import open3d as o3d
import numpy.linalg as LA
import copy
import os

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

from tf2_ros import Buffer, TransformListener, TransformException
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


def _pca_plane_normal(pts_np: np.ndarray):
    """Return (centroid, unit normal) for best-fit plane to pts_np (N,3)."""
    # Pick centroid as the point closest to the z-axis (min radial distance in XY)
    r_xy = np.sqrt(pts_np[:, 0]**2 + pts_np[:, 1]**2)
    c = pts_np[np.argmin(r_xy)]
    print(f"Debug: PCA centroid at {c}")
    X = pts_np - c
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(X)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    # cluster = pcd.cluster_dbscan(eps=0.02, min_points=10)
    # # Paint clusters
    # for i in range(len(cluster)):
    #     if cluster[i] > 0:
    #         pcd.colors.append([1, 0, 0])  # Red for inliers
    #     else:
    #         pcd.colors.append([0.5, 0.5, 0.5])  # Gray for outliers
    # # Get index of point closeset to origin
    # centroid_idx = np.argmin(np.linalg.norm(X, axis=1))
    # # Get normal at centroid
    # n = np.asarray(pcd.normals)[centroid_idx]
    # # Visualize PCA points
    # o3d.visualization.draw_geometries([pcd], window_name='PCA Points', width=800, height=600)
    # Only use inliers for plane fitting
    # inlier_indices = [i for i in range(len(cluster)) if cluster[i] > 0]
    # X = X[inlier_indices]
    # 3x3 covariance; smallest eigenvalue's eigenvector is the plane normal
    C = (X.T @ X) / max(len(X) - 1, 1)
    w, v = LA.eigh(C)
    n = v[:, 0]
    # Make direction consistent (toward camera -Z in depth cam frame)
    if n[2] < 0:
        n = -n
    n /= (LA.norm(n) + 1e-12)
    
    return c, n


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

    axis = np.cross(zc, zg)
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
def _x_axis_rotvec_error(x_goal: np.ndarray) -> np.ndarray:
    """Return rotation-vector ω that rotates xc=[1,0,0] onto x_goal.
    Both vectors must be expressed in the same frame."""
    xc = np.array([1.0, 0.0, 0.0], dtype=np.float32)      # camera's current X
    xg = x_goal.astype(np.float32)
    xg /= (LA.norm(xg) + 1e-12)                           # normalize

    c = float(np.clip(np.dot(xc, xg), -1.0, 1.0))         # cos(theta)
    theta = math.acos(c)
    print(f'Debug: theta_roll={math.degrees(theta):.2f}°')

    axis = np.cross(xc, xg)
    n = LA.norm(axis)

    if n < 1e-9:
        # xc and xg are parallel
        if c > 0.0:
            return np.zeros(3, dtype=np.float32)  # aligned
        else:
            # 180°: pick y-axis as arbitrary rotation axis
            return np.array([0.0, theta, 0.0], dtype=np.float32)

    axis /= n
    return (theta * axis).astype(np.float32)

def _roll_error(x_current_world) -> float:
    """Return roll error (radians): angle between camera x-axis and world XY plane.

    Positive when x-axis points above XY plane, negative when below.
    """
    xc = x_current_world / (LA.norm(x_current_world) + 1e-12)
    # Angle to XY plane is arcsin of z-component for a unit vector
    theta = math.asin(float(np.clip(xc[2], -1.0, 1.0)))
    print(f"Debug: roll error={math.degrees(theta):.2f}°")
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
                ('no_target_timeout_s', 0.25),
                ('publish_zero_when_lost', True),
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

        self.d_min = float(self.get_parameter('d_min').value) # Minimum focal distance (meters)
        self.d_max = float(self.get_parameter('d_max').value) # Maximum focal distance (meters)
        self.v_max = float(self.get_parameter('v_max').value) # Maximum linear velocity (meters/second)
        self.theta_max_deg = float(self.get_parameter('theta_max_deg').value) # Maximum angular displacement (degrees)

        self.Kp = float(self.get_parameter('Kp').value)
        self.Ki = float(self.get_parameter('Ki').value)    # integral gain about camera Z  # <<< NEW
        self.Kd = float(self.get_parameter('Kd').value)  # <<< NEW
        self.no_target_timeout_s = float(self.get_parameter('no_target_timeout_s').value)
        self.publish_zero_when_lost = bool(self.get_parameter('publish_zero_when_lost').value)
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
        self.pub_z_rotvec_err = self.create_publisher(
            Vector3Stamped, f'/{self.get_name()}/z_axis_rotvec_error_in_{self.main_camera_frame}', 10
        )
        self.pub_wrench_cmd = self.create_publisher(
            WrenchStamped, f'/{self.get_name()}/wrench_cmds', 10
        )
        self.distance_pub = self.create_publisher(Float64, f'/{self.get_name()}/focal_distance_m', 10)

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
                0.1, self.process_dmap, callback_group=self._timer_cb_group
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

    def _publish_zero_wrench(self):
        w = WrenchStamped()
        # use the latest header if available; otherwise make a minimal header
        if self.depth_msg is not None:
            w.header = self.depth_msg.header
            w.header.frame_id = self.main_camera_frame
        else:
            w.header.frame_id = self.main_camera_frame
        w.wrench.force.x = 0.0; w.wrench.force.y = 0.0; w.wrench.force.z = 0.0
        w.wrench.torque.x = 0.0; w.wrench.torque.y = 0.0; w.wrench.torque.z = 0.0
        self.pub_wrench_cmd.publish(w)

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
        # Store compressed depth message for processing
        self.ocd.header = msg.header
        self.depth_msg = msg

    def process_dmap(self):
        if not self.depth_msg:
            return
        # --- Time bookkeeping for watchdog ---
        stamp = self.depth_msg.header.stamp
        now_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        measurement_ok = False

        centroid_out = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        normal_out   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        goal_pose_out = None  # will become a PoseStamped if we have one
        rotvec_err_out = np.zeros(3, dtype=np.float32)
        tau_out = np.zeros(3, dtype=np.float32)
        force_out = np.zeros(3, dtype=np.float32)
        # Decompress compressedDepth format -> numpy
        # compressedDepth has a 12-byte header: uint32 format + float32 depthQuantA + float32 depthQuantB
        np_arr = np.frombuffer(self.depth_msg.data, np.uint8)
        if np_arr.size <= 12:
            self.get_logger().warn("Empty or invalid compressedDepth data, skipping frame")
            return
        # Skip the 12-byte header to get to the PNG data
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
        self.ocd.depth_image.header = self.depth_msg.header

        # Normalize to meters (compressed depth is typically 16UC1 PNG)
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 1e-3
        else:
            depth_m = depth.astype(np.float32)

        # Build mask
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        mask = valid & (depth_m >= self.dmap_filter_min) & (depth_m <= self.dmap_filter_max)

        # Build filtered depth (32FC1) - invalid pixels set to NaN
        depth_filtered = depth_m.copy()
        depth_filtered[~mask] = np.nan

        # Convert to ROS
        self.ocd.depth_filtered_image = self.bridge.cv2_to_imgmsg(depth_filtered.astype(np.float32), encoding='32FC1')
        self.ocd.dmap_filter_min = self.dmap_filter_min
        self.ocd.dmap_filter_max = self.dmap_filter_max
        # Defaults for pointcloud outputs
        points1 = np.zeros((0, 3), dtype=np.float32)
        points2 = np.zeros((0, 3), dtype=np.float32)

        # Point cloud (if enabled and we have intrinsics)
        if self.K is not None:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            h, w = depth_m.shape
            stride = max(1, self.pcd_downsampling_stride)

            ys, xs = np.where(mask)

            if stride > 1:
                ys = ys[::stride]; xs = xs[::stride]

            if xs.shape[0] > 0:
                z = depth_m[ys, xs]
                x = (xs.astype(np.float32) - cx) * z / fx
                y = (ys.astype(np.float32) - cy) * z / fy
                points1 = np.stack([x, y, z], axis=1)
                src_frame = self.depth_msg.header.frame_id
                pts_bbox = np.zeros((0, 3), dtype=np.float32)
                try:
                    T = self.tf_buffer.lookup_transform(
                        self.target_frame, src_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                    q = T.transform.rotation
                    R = _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
                    t = T.transform.translation
                    t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                    pts_tgt = (R @ points1.T).T + t_vec  # (N,3)

                except TransformException as e:
                    self.get_logger().warn(f'No TF {self.target_frame} <- {src_frame} at stamp: {e}')
                    pts_tgt = np.zeros((0, 3), dtype=np.float32)
                else:
                    # --- Axis-aligned bounding-box filter in target_frame ---
                    sel = np.all((pts_tgt >= self.bbox_min) & (pts_tgt <= self.bbox_max), axis=1)
                    pts_bbox = np.ascontiguousarray(pts_tgt[sel])

                try:
                    T_out = self.tf_buffer.lookup_transform(
                        self.main_camera_frame, self.target_frame, Time(sec=0, nanosec=0), timeout=Duration(seconds=0.001))
                    q_out = T_out.transform.rotation
                    R_out = _quat_to_R_xyzw(q_out.x, q_out.y, q_out.z, q_out.w)
                    t_out = T_out.transform.translation
                    t_vec_out = np.array([t_out.x, t_out.y, t_out.z], dtype=np.float32)
                    pts_bbox_out = (R_out @ pts_bbox.T).T + t_vec_out  # (N,3)

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

                    # Compute PCA normal
                    if pts_crop.shape[0] >= 10:
                        centroid, normal = _pca_plane_normal(pts_crop)

                        if not self.ema_enable:
                            cen_s = centroid
                            nrm_s = normal
                            self._ema_centroid = centroid
                            self._ema_normal   = normal
                            self._ema_last_t   = now_s
                        else:
                            if self._ema_last_t is None:
                                # First sample initializes the EMA
                                self._ema_centroid = centroid.copy()
                                self._ema_normal   = normal.copy()
                                self._ema_last_t   = now_s

                            # Keep a consistent normal hemisphere to avoid ± flips
                            # if self._ema_normal is not None and np.dot(normal, self._ema_normal) < 0.0:
                            #     normal = -normal

                        dt = max(0.0, now_s - (self._ema_last_t if self._ema_last_t is not None else now_s))
                        # α from time-constant τ (handles variable frame rate)
                        alpha = 1.0 - math.exp(-dt / max(1e-3, self.ema_tau))
                        alpha = min(1.0, max(0.0, alpha))

                        # EMA updates
                        self._ema_centroid = self._ema_update(self._ema_centroid, centroid, alpha)
                        self._ema_normal   = self._ema_update(self._ema_normal,   normal,   alpha)

                        # Renormalize the smoothed normal
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
                        x_cf, y_cf, z_cf = R_tf[:, 0], R_tf[:, 1], R_tf[:, 2]   # each is a length-3 unit vector
                        t = T.transform.translation
                        t_vec = np.array([t.x, t.y, t.z], dtype=np.float32)
                        cen_s_obj = (R_tf @ cen_s) + t_vec
                        nrm_s_obj = R_tf @ nrm_s

                        surface_target_pose = PoseStamped()
                        surface_target_pose.header = self.depth_msg.header
                        surface_target_pose.header.frame_id = 'object_frame'
                        surface_target_pose.pose.position.x = float(cen_s_obj[0])
                        surface_target_pose.pose.position.y = float(cen_s_obj[1])
                        surface_target_pose.pose.position.z = float(cen_s_obj[2])
                        surface_target_pose.pose.orientation = _quaternion_from_z(nrm_s_obj)
                        self.normal_estimate_pub.publish(surface_target_pose)

                        # ---- Compute standoff based on mode ----
                        if self.standoff_mode == 'euclidean':
                            d = float(LA.norm(cen_s))  # ||c||
                        elif self.standoff_mode == 'along_normal':
                            d = float(max(0.0, float(np.dot(nrm_s, cen_s))))
                        else:  # 'fixed' or anything else
                            d = self.standoff_m

                        # Desired EOAT pose: back off along normal by d; +Z aligned with normal
                        p_des_cf = cen_s - d*nrm_s
                        q_des_cf = _quaternion_from_z(nrm_s_obj)
                        # Transform q_des_cf back to main_camera_frame
                        R_des_cf = _quat_to_R_xyzw(q_des_cf.x, q_des_cf.y, q_des_cf.z, q_des_cf.w)
                        R_cam = R_tf.T @ R_des_cf
                        q_des_cf = _R_to_quat_xyzw(R_cam)

                        eoat_cf = PoseStamped()
                        eoat_cf.header = self.depth_msg.header
                        eoat_cf.header.frame_id = self.main_camera_frame
                        eoat_cf.pose.position.x = float(p_des_cf[0])
                        eoat_cf.pose.position.y = float(p_des_cf[1])
                        eoat_cf.pose.position.z = float(p_des_cf[2])
                        eoat_cf.pose.orientation = q_des_cf
                        self.pub_eoat_pose_crop.publish(eoat_cf)

                        R_goal = _quat_to_R_xyzw(q_des_cf.x, q_des_cf.y, q_des_cf.z, q_des_cf.w)
                        xg, yg, zg = R_goal[:, 0], R_goal[:, 1], R_goal[:, 2]   # each is a length-3 unit vector

                        # --- Z-axis alignment error in main_camera_frame ---
                        theta_err, axis_err = _z_axis_rotvec_error(nrm_s.astype(np.float32))
                        rot_vec_error = axis_err * theta_err  # 3-vector [ωx, ωy, ωz]

                        # Compute rotational error w.r.t. object_frame
                        roll_error = _roll_error(x_cf)
                        rot_vec_error[2] = roll_error

                        centroid_out = cen_s.astype(np.float32)
                        normal_out   = nrm_s.astype(np.float32)
                        rotvec_err_out = rot_vec_error.astype(np.float32)
                        goal_pose_out = eoat_cf  # PoseStamped already filled

                        err_msg = Vector3Stamped()
                        err_msg.header = self.depth_msg.header
                        err_msg.header.frame_id = self.main_camera_frame
                        err_msg.vector.x, err_msg.vector.y, err_msg.vector.z = map(float, rot_vec_error)
                        self.pub_z_rotvec_err.publish(err_msg)

                        if self.orientation_control_enabled:
                            # --- CONTROLLER ---
                            # Compute control torques based on rot_vec_error


                            # ================== FORCE PID (translational) ====================  # <<< NEW FOR FORCE PID >>>
                            # Use p_des_cf as position error vector in main_camera_frame
                            #pos_err = p_des_cf.astype(np.float32)

                            #if self._last_pos_t is not None and self._last_pos_err is not None:
                              #  dt_pos = max(1e-6, now_s - self._last_pos_t)
                              #  dpos = (pos_err - self._last_pos_err) / dt_pos
                            #else:
                             #   dt_pos = 0.0
                             #   dpos = np.zeros(3, dtype=np.float32)

                           # self._last_pos_t = now_s
                            #self._last_pos_err = pos_err.copy()

                            # Integrate position error
                           # if dt_pos > 0.0:
                           #     self._int_pos_err += pos_err * dt_pos

                           # Kp_lin = np.array([self.Kp, self.Kp, self.Kp], dtype=np.float32)
                           # Kd_lin = np.array([self.Kd, self.Kd, self.Kd], dtype=np.float32)
                           # Ki_lin = np.array([self.Ki, self.Ki, self.Ki], dtype=np.float32)

                           # force = (Kp_lin * pos_err +
                                   #  Kd_lin * dpos +
                                 #    Ki_lin * self._int_pos_err).astype(np.float32)
                           # force_out = force.copy()
                           # self._last_force = force.copy()                                            # <<< NEW FOR FORCE PID >>>
                            # --- PID control on rotation-vector error rot_vec_error---   # <<< NEW
                            # Compute error derivative (finite difference)         # <<< NEW
                            self.anti_windup_enabled = True
                            self.aw_Tt = 0.05
                            self.int_limit = 1e3
                            self.e = theta_err  # store for debugging
                            dt_ctrl = 0.0
                            if self._last_err_t is not None:  # <<< NEW
                                dt_ctrl = max(1e-6, now_s - self._last_err_t)                      # <<< NEW
                                self.de = (self.e - self.last_e) / dt_ctrl                 # <<< NEW
                            else:  
                                dt_ctrl = 0.0                                                                 # <<< NEW
                                self.de = 0.0                             # <<< NEW

                            self._last_err_t = now_s   
                            self.last_e = float(self.e)                                          # <<< NEW
                            self._last_rotvec_err = rot_vec_error.copy()                                   # <<< NEW

                             # Integral of error
                            #if dt_ctrl > 0.0:
                             #   self._int_rotvec_err += rot_vec_error* dt_ctrl
                                # Optional: simple anti-windup clamp
                              #  int_limit = 1e3   # tune as needed
                               # self._int_rotvec_err = np.clip(
                               #     self._int_rotvec_err, -int_limit, int_limit
                               # )
                            
                            inertia_A = self.inertia_B + self.mass_B * d**2 # rotational inertia for torque control
                            tau_max = self.linear_drag*d*self.v_max
                            omega_n_max = math.sqrt(tau_max/(inertia_A*self.theta_max_deg*3.141592653589793/180.0))  # max angular 
                            omega_n = omega_n_max; # natural frequency for desired max torque

                            p_1 = -omega_n
                            p_2 = -omega_n
                            p_3 =-self.integral_alpha*omega_n

                            if self.controller_type == 'PD':
                                # PD controller
                                self.Kp = inertia_A*(p_1*p_2)
                                self.Kd = -inertia_A*(p_1 + p_2) - self.linear_drag*d*d
                                self.Ki = 0.0
                            elif self.controller_type == 'PID':
                                # PID controller
                                self.Kp = inertia_A*(p_1*p_2+p_1*p_3+p_2*p_3)
                                self.Ki = -inertia_A*(p_1*p_2*p_3)
                                self.Kd = -inertia_A*(p_1 + p_2 + p_3) - self.linear_drag*d*d
                            else:
                                self.get_logger().warn(f'Unknown controller_type {self.controller_type}, defaulting to PD')
                                # PD controller
                                self.Kp = inertia_A*(p_1*p_2)
                                self.Kd = -inertia_A*(p_1 + p_2) - self.linear_drag*d*d
                                self.Ki = 0.0

                            # self.Kp = inertia_A*(p_1*p_2)
                            # self.Kd = -inertia_A*(p_1 + p_2) - self.linear_drag*d*d
                            # self.Ki = 0.0
                            # print(f"Updated gains: Kp={self.Kp}, Kd={self.Kd}" f"Ki={self.Ki}")
                            # Elementwise PD: τ = Kp*ω + Kd*ω̇                                      # <<< NEW
                           # Kp_vec = np.array([self.Kp,  self.Kp,  self.Kp],  dtype=np.float32)  # <<< NEW
                          #  Ki_vec = np.array([self.Ki, self.Ki, self.Ki], dtype=np.float32)   # <<< NEW
                          #  Kd_vec = np.array([self.Kd, self.Kd, self.Kd], dtype=np.float32)  # <<< NEW
                            #tau_A = (Kp_vec * rot_vec_error+ Ki_vec * self._int_rotvec_err - Kd_vec * drot_vec_error).astype(np.float32)                # <<< NEW
                            tau_A = (self.Kp * self.e + self.Ki * self.int_e + self.Kd * self.de)               # <<< NEW
                            print(f"Pre-sat tau: {tau_A}")
                            # Saturation
                            tau_sat = np.clip(tau_A, -tau_max, tau_max)
                          #  mag = np.linalg.norm(tau_A)
                         #   if mag > tau_max:
                           #     tau_sat = tau_A * (tau_max / mag)
                           # else:
                            #    tau_sat = tau_A


                            print(f"Sat tau: {tau_sat}")
                            # --------- Anti-windup (back-calculation) ----------
                               # i_dot = e + (u_sat - u)/(Ki*Tt)
                            if (getattr(self, "anti_windup_enabled", True) and (self.Ki > 1e-9) and (dt_ctrl > 0.0)):
                                Tt = float(getattr(self, "aw_Tt", 0.05))
                                print(f"theta_err: {self.e}")
                                i_dot = self.e + (tau_sat - tau_A) / (self.Ki * max(1e-6, Tt))
                                print(f"i_dot: {i_dot}")
                                self.int_e += i_dot * dt_ctrl
                                print(self.int_e)
                            else:
                                if dt_ctrl > 0.0:
                                  self.int_e += self.e * dt_ctrl

                            # Integrator clamp
                            int_lim = float(getattr(self, "int_limit", 1e3))
                            self.int_e = np.clip(self.int_e, -int_lim, int_lim)
                            tau_A=tau_sat 
                            tau_A_vec = tau_A * axis_err  # scale by error axis to get full 3-vector
                            moment_tele_A = np.cross(r, self.force_teleop)
                            force_drag_B = -self.linear_drag * self.lin_vel_cam
                            moment_drag_B = -self.angular_drag * self.rot_vel_cam
                            moment_dragB_A = np.cross(r, force_drag_B)
                            self.ang_acc = (moment_dragB_A + tau_A_vec + moment_tele_A)/ inertia_A  # <<< NEW

                            tau_out = tau_A_vec.copy()                                                     # <<< NEW
                            self._last_tau = tau_A_vec.copy()                                              # <<< NEW
                            self.force_B = self.mass_B * np.cross(self.ang_acc,r) - force_drag_B - self.force_teleop  # simple proportional model for force command based on torque command
                            # Centripetal force command
                            # self.force[2] = self.mass * np.linalg.norm(self.lin_vel_cam)**2 / np.linalg.norm(cen_s)
                            self.tau_B = self.inertia_B * self.ang_acc - moment_drag_B - self.torque_teleop  # torque about camera origin
                            self.tau_B[2] = roll_error * self.Kp  # add roll correction about Z-axis only

                            self.distance_pub.publish(Float64(data=float(LA.norm(cen_s))))
                            # Saturation (optional)
                            # lim = self.torque_limit
                            # tau = np.clip(tau, -lim, lim)

                            # Publish as Wrench (torque only; set forces to 0 or add your own position control)
                            w = WrenchStamped()
                            w.header = self.depth_msg.header
                            w.header.frame_id = self.main_camera_frame
                            w.wrench.force.x = float(self.force_B[0])
                            w.wrench.force.y = float(self.force_B[1])
                            w.wrench.force.z = float(self.force_B[2]) # Centripetal force command
                            w.wrench.torque.x = float(self.tau_B[0])
                            w.wrench.torque.y = float(self.tau_B[1])
                            w.wrench.torque.z = float(self.tau_B[2])  # will be ~0 for Z-only error
                            self.pub_wrench_cmd.publish(w)
                        else:
                            tau_out = np.zeros(3, dtype=np.float32)
                            force_out = np.zeros(3, dtype=np.float32)   # <<< NEW
                            self._last_tau = tau_out.copy()
                            self._last_force = force_out.copy()
                            self._publish_zero_wrench()

                        # Mark this cycle as valid
                        measurement_ok = True
                        self._had_target_last_cycle = True
                        self._last_target_time_s = now_s

        # Publish pointclouds
        pcd2_msg = make_pointcloud2(
            points1, frame_id=self.main_camera_frame, stamp=self.depth_msg.header.stamp
        )
        self.eoat_pointcloud_publisher.publish(pcd2_msg)

        pcd2_msg_final = make_pointcloud2(
            points2, frame_id=self.main_camera_frame, stamp=self.depth_msg.header.stamp
        )
        self.fov_pointcloud_publisher.publish(pcd2_msg_final)

        # cloud after bbox (in main_camera_frame) — reuse the message you just published
        cloud_bbox_msg = make_pointcloud2(points1, frame_id=self.main_camera_frame, stamp=self.depth_msg.header.stamp)
        self.ocd.cloud_bbox = cloud_bbox_msg

        # cloud after cylindrical crop (in main_camera_frame)
        cloud_crop_msg = make_pointcloud2(points2, frame_id=self.main_camera_frame, stamp=self.depth_msg.header.stamp)
        self.ocd.cloud_crop = cloud_crop_msg

        # Also publish those clouds to your existing topics (keep your current behavior)
        self.eoat_pointcloud_publisher.publish(cloud_bbox_msg)
        self.fov_pointcloud_publisher.publish(cloud_crop_msg)

        # Geometry and pose data (smoothed centroid & normal, and EOAT goal pose if available)
        self.ocd.centroid = Point(x=float(centroid_out[0]), y=float(centroid_out[1]), z=float(centroid_out[2]))
        self.ocd.normal   = Vector3(x=float(normal_out[0]),   y=float(normal_out[1]),   z=float(normal_out[2]))

        if goal_pose_out is not None:
            self.ocd.goal_pose = goal_pose_out
        else:
            # Fill a PoseStamped with just the header so field is valid even when not tracking
            empty_pose = PoseStamped()
            empty_pose.header = self.ocd.header
            self.ocd.goal_pose = empty_pose

        # Control data (rotvec error, torque command, gains)
        self.ocd.rotvec_error = Vector3(x=float(rotvec_err_out[0]), y=float(rotvec_err_out[1]), z=float(rotvec_err_out[2]))

        # Use the torque we computed this cycle; if none, fall back to last commanded (zeros otherwise)
        tau_for_msg = tau_out if measurement_ok else self._last_tau
        self.ocd.torque_cmd = Vector3(x=float(tau_for_msg[0]), y=float(tau_for_msg[1]), z=float(tau_for_msg[2]))
        self.ocd.cam_force_cmd = Vector3(x=float(self.force_B[0]), y=float(self.force_B[1]), z=float(self.force_B[2]))  # <<< NEW
        self.ocd.cam_torque_cmd = Vector3(x=float(self.tau_B[0]), y=float(self.tau_B[1]), z=float(self.tau_B[2]))  # <<< NEW

       # self.ocd.force_cmd = Vector3(x=float(force[0]),y=float(force[1]),z=float(force[2]),)  # <<< NEW
       # self.ocd.torque_cmd_cam = Vector3(x=float(tau_cam[0]), y=float(tau_cam[1]), z=float(tau_cam[2]))  # <<< NEW

        
      #  force_for_msg = force_out if measurement_ok else self._last_force    # <<< NEW
      #  self.ocd.force_cmd = Vector3(x=float(force_for_msg[0]), y=float(force_for_msg[1]), z=float(force_for_msg[2]))  # <<< NEW

        self.ocd.k_p = float(self.Kp)
        self.ocd.k_d = float(self.Kd)
        if hasattr(self.ocd, 'k_i'):                             # <<< NEW
            self.ocd.k_i = float(self.Ki)


        # Frames
        self.ocd.main_camera_frame = str(self.main_camera_frame)
        self.ocd.target_frame      = str(self.target_frame)

        # EMA & safety config
        self.ocd.ema_enable            = bool(self.ema_enable)
        self.ocd.ema_tau_s             = float(self.ema_tau)                # note: msg field is *_tau_s
        self.ocd.no_target_timeout_s   = float(self.no_target_timeout_s)
        self.ocd.publish_zero_when_lost = bool(self.publish_zero_when_lost)

        if self.save_data:
            self.bag_orientation_control_data()

        # ---- Watchdog / fallback when no valid target this cycle ----
        if not measurement_ok:
            if self.publish_zero_when_lost:
                self._publish_zero_wrench()
                self._last_tau = np.zeros(3, dtype=np.float32)
                self._last_force = np.zeros(3, dtype=np.float32)   # <<< NEW
            if (self._last_target_time_s is None) or ((now_s - self._last_target_time_s) > self.no_target_timeout_s):
                if self._had_target_last_cycle:
                    self.get_logger().warn('No valid target: lost')
                self._ema_normal = None
                self._ema_centroid = None
                self._ema_last_t = None
                self._had_target_last_cycle = False
                # Reset PD error history when we lose the target         # <<< NEW
                self._last_rotvec_err = None                              # <<< NEW
                self._last_err_t = None                                   # <<< NEW
                self._int_rotvec_err = np.zeros(3, dtype=np.float32)     # <<< NEW
               # self._last_pos_err = None                        # <<< NEW FOR FORCE PID >>>
               # self._last_pos_t = None                          # <<< NEW FOR FORCE PID >>>
              #  self._int_pos_err = np.zeros(3, dtype=np.float32)  # <<< NEW FOR FORCE PID >>>


    def bag_orientation_control_data(self):
        if self.orientation_control_enabled:
            self.writer.write(
                'orientation_control_data',
                serialize_message(self.ocd),
                self.get_clock().now().nanoseconds
            )

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
            elif p.name == 'publish_zero_when_lost':
                self.publish_zero_when_lost = bool(p.value)
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
