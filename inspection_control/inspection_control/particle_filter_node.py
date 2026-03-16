#!/usr/bin/env python3
"""
Particle filter node for estimating the pose of a known object using depth data
from a RealSense D405 camera. Estimates 3-DOF pose (tx, ty, theta_z) and publishes
a synthetic depth image rendered from the known mesh at the estimated pose.
"""

import threading
import numpy as np
import cv2
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.srv import GetParameters

import struct

from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

import tf2_ros
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UNIT_TO_METERS = {
    "m": 1.0, "meters": 1.0,
    "mm": 0.001, "millimeters": 0.001,
    "cm": 0.01, "centimeters": 0.01,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
}


def unit_scale(unit_str: str) -> float:
    unit_str = unit_str.strip().lower()
    if unit_str in UNIT_TO_METERS:
        return UNIT_TO_METERS[unit_str]
    raise ValueError(f"Unknown unit '{unit_str}'. Supported: {list(UNIT_TO_METERS.keys())}")


def pose_matrix(tx: float, ty: float, theta_z: float) -> np.ndarray:
    """Build a 4x4 homogeneous transform from 3-DOF (tx, ty, theta_z)."""
    c, s = np.cos(theta_z), np.sin(theta_z)
    T = np.eye(4)
    T[0, 0] = c;  T[0, 1] = -s
    T[1, 0] = s;  T[1, 1] = c
    T[0, 3] = tx; T[1, 3] = ty
    return T


def msg_to_matrix(transform_msg: TransformStamped) -> np.ndarray:
    """Convert a geometry_msgs TransformStamped to a 4x4 numpy matrix."""
    t = transform_msg.transform.translation
    q = transform_msg.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[0, 3] = t.x; T[1, 3] = t.y; T[2, 3] = t.z
    return T


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    """Low-variance systematic resampling. Returns array of selected indices."""
    N = len(weights)
    positions = (np.arange(N) + np.random.uniform()) / N
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return indices


def circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean for angles."""
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return np.arctan2(s, c)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__("particle_filter")

        # -- Declare all particle filter parameters -------------------------
        self.declare_parameters("", [
            ("num_particles",              500),
            ("sigma_translation",          0.005),   # m – prediction noise
            ("sigma_rotation",             0.087),   # rad (~5°) – prediction noise
            ("sigma_observation",          0.005),   # m – observation likelihood σ
            ("inlier_threshold",           0.005),   # m – inlier distance
            ("use_inlier_scoring",         True),
            ("resample_threshold_ratio",   0.5),
            ("reference_voxel_size",       0.002),   # m – downsample reference for scoring
            ("observation_voxel_size",     0.003),   # m – downsample observed cloud
            ("depth_min",                  0.07),    # m – D405 min usable range
            ("depth_max",                  0.50),    # m – D405 max usable range
            ("position_bound",             0.17),    # m – radius of uniform init region
            ("roi_margin",                 0.05),    # m – extra margin around position_bound for cropping
            ("synthetic_depth_topic",      "particle_filter/depth/synthetic"),
            ("pose_topic",                 "particle_filter/pose"),
            ("viewpoint_generation_node",  "viewpoint_generation"),
        ])

        # Read them into attributes
        self.N                  = self.get_parameter("num_particles").value
        self.sigma_t            = self.get_parameter("sigma_translation").value
        self.sigma_r            = self.get_parameter("sigma_rotation").value
        self.sigma_obs          = self.get_parameter("sigma_observation").value
        self.inlier_thresh      = self.get_parameter("inlier_threshold").value
        self.use_inlier         = self.get_parameter("use_inlier_scoring").value
        self.resample_ratio     = self.get_parameter("resample_threshold_ratio").value
        self.ref_voxel          = self.get_parameter("reference_voxel_size").value
        self.obs_voxel          = self.get_parameter("observation_voxel_size").value
        self.depth_min          = self.get_parameter("depth_min").value
        self.depth_max          = self.get_parameter("depth_max").value
        self.pos_bound          = self.get_parameter("position_bound").value
        self.roi_margin         = self.get_parameter("roi_margin").value
        self.vg_node_name       = self.get_parameter("viewpoint_generation_node").value

        # -- State ----------------------------------------------------------
        self.intrinsics = None      # dict with fx, fy, cx, cy, width, height
        self.camera_frame = None
        self.mesh = None            # o3d.geometry.TriangleMesh  (meters)
        self.ref_cloud = None       # np.ndarray (M, 3) – downsampled reference cloud for scoring
        self.particles = None       # np.ndarray (N, 3) – [tx, ty, theta_z]
        self.weights = None         # np.ndarray (N,)
        self.processing = False
        self.lock = threading.Lock()
        self.model_ready = False
        self._T_model_in_obj = None   # latest particle filter estimate (set by _process_frame)

        # -- TF -------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -- Publishers -----------------------------------------------------
        synth_topic = self.get_parameter("synthetic_depth_topic").value
        pose_topic  = self.get_parameter("pose_topic").value
        self.depth_pub          = self.create_publisher(Image,           synth_topic,                               10)
        self.depth_compressed_pub = self.create_publisher(CompressedImage, synth_topic + "/compressedDepth",         10)
        self.pose_pub      = self.create_publisher(PoseStamped, pose_topic,                         10)
        self.particles_pub  = self.create_publisher(PoseArray,    "particle_filter/particles",        10)
        self.markers_pub    = self.create_publisher(MarkerArray,  "particle_filter/particle_markers", 10)
        self.obs_cloud_pub  = self.create_publisher(PointCloud2,  "particle_filter/observed_cloud",   10)
        self.camera_info_pub = self.create_publisher(CameraInfo,  "particle_filter/depth/camera_info", 10)

        # -- Subscribers (start before model is loaded so we buffer info) ---
        # The depth callback can block for seconds (particle update step).
        # Give it a ReentrantCallbackGroup so it does not starve the polling
        # timer and service-response callbacks that live in the default
        # MutuallyExclusiveCallbackGroup.  The self.processing lock already
        # prevents overlapping depth-frame processing.
        self._depth_cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            CameraInfo,
            "/camera/d405_camera/depth/camera_info",
            self._camera_info_cb, 10)

        self.create_subscription(
            CompressedImage,
            "/camera/d405_camera/depth/image_rect_raw/compressedDepth",
            self._depth_cb, 5,
            callback_group=self._depth_cb_group)

        # -- Fetch model paths from viewpoint_generation node ---------------
        self._param_client = self.create_client(
            GetParameters,
            f"/{self.vg_node_name}/get_parameters")

        # Track last known paths and units so we can detect changes and restart
        self._current_mesh_file  = None
        self._current_mesh_units = None
        self._current_pc_file    = None
        self._current_pc_units   = None

        self.get_logger().info(
            "Polling viewpoint_generation for model parameters every 5 s …")
        self.create_timer(5.0, self._poll_model_params)

    # -----------------------------------------------------------------------
    # Initialization helpers
    # -----------------------------------------------------------------------

    def _poll_model_params(self):
        """Timer callback (every 5 s): fetch model paths from viewpoint_generation."""
        if not self._param_client.service_is_ready():
            self.get_logger().info(
                "viewpoint_generation parameter service not ready yet, retrying …",
                throttle_duration_sec=10.0)
            return

        request = GetParameters.Request()
        request.names = [
            "model.mesh.file", "model.mesh.units",
            "model.point_cloud.file", "model.point_cloud.units",
        ]
        future = self._param_client.call_async(request)
        future.add_done_callback(self._on_model_params)

    def _on_model_params(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"Failed to get parameters: {e}")
            return

        values = result.values
        mesh_file  = values[0].string_value
        mesh_units = values[1].string_value
        pc_file    = values[2].string_value
        pc_units   = values[3].string_value

        # Wait until both paths are populated
        if not mesh_file or not pc_file:
            self.get_logger().info(
                "Model file paths not yet set in viewpoint_generation, waiting …",
                throttle_duration_sec=10.0)
            return

        # Check whether paths or units have changed since the last load
        params_changed = (mesh_file  != self._current_mesh_file  or
                          mesh_units != self._current_mesh_units  or
                          pc_file    != self._current_pc_file     or
                          pc_units   != self._current_pc_units)
        if not params_changed:
            return

        if self._current_mesh_file is not None:
            self.get_logger().info(
                "Model parameters changed – restarting particle filter …")

        self.get_logger().info(f"Mesh : {mesh_file} ({mesh_units})")
        self.get_logger().info(f"Cloud: {pc_file} ({pc_units})")

        try:
            self.model_ready = False
            self._load_models(mesh_file, mesh_units, pc_file, pc_units)
            self._init_particles()
            self._current_mesh_file  = mesh_file
            self._current_mesh_units = mesh_units
            self._current_pc_file    = pc_file
            self._current_pc_units   = pc_units
            self.model_ready = True
            self.get_logger().info("Model loaded and particles initialised.")
        except Exception as e:
            self.get_logger().error(f"Failed to load models: {e}")

    def _load_models(self, mesh_path, mesh_units, pc_path, pc_units):
        """Load STL mesh and PLY cloud, scale both to metres."""
        # Mesh (for rendering)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh from {mesh_path}")
        s = unit_scale(mesh_units)
        mesh.scale(s, center=(0, 0, 0))
        mesh.compute_vertex_normals()
        self.mesh = mesh

        # Reference point cloud (for scoring)
        pcd = o3d.io.read_point_cloud(pc_path)
        if pcd.is_empty():
            raise RuntimeError(f"Failed to load point cloud from {pc_path}")
        s = unit_scale(pc_units)
        pcd.scale(s, center=(0, 0, 0))
        pcd_ds = pcd.voxel_down_sample(self.ref_voxel)
        self.ref_cloud = np.asarray(pcd_ds.points)
        self.get_logger().info(
            f"Reference cloud: {len(self.ref_cloud)} points "
            f"(voxel {self.ref_voxel*1e3:.1f} mm)")

        # Set up raycasting scene
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.ray_scene = o3d.t.geometry.RaycastingScene()
        self.ray_scene.add_triangles(mesh_t)

    def _init_particles(self):
        """Uniformly sample initial particles in the bounded workspace."""
        tx = np.random.uniform(-self.pos_bound, self.pos_bound, self.N)
        ty = np.random.uniform(-self.pos_bound, self.pos_bound, self.N)
        theta = np.random.uniform(-np.pi, np.pi, self.N)
        self.particles = np.column_stack([tx, ty, theta])
        self.weights = np.ones(self.N) / self.N

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _camera_info_cb(self, msg: CameraInfo):
        K = msg.k
        self.intrinsics = {
            "fx": K[0], "fy": K[4],
            "cx": K[2], "cy": K[5],
            "width": msg.width, "height": msg.height,
        }
        self.camera_frame = msg.header.frame_id
        self.camera_info_pub.publish(msg)

    def _depth_cb(self, msg: CompressedImage):
        if not self.model_ready:
            self.get_logger().info("depth_cb: model not ready yet, skipping",
                                   throttle_duration_sec=5.0)
            return
        if self.intrinsics is None:
            self.get_logger().info("depth_cb: no camera intrinsics yet, skipping",
                                   throttle_duration_sec=5.0)
            return

        # Run particle filter update if not already processing
        run_pf = False
        with self.lock:
            if not self.processing:
                self.processing = True
                run_pf = True
        if run_pf:
            try:
                self._process_frame(msg)
            except Exception as e:
                self.get_logger().error(f"Processing error: {e}", throttle_duration_sec=2.0)
            finally:
                with self.lock:
                    self.processing = False

        # Always render at full frame rate using the latest T_model_in_obj
        if self._T_model_in_obj is not None:
            try:
                tf_cam_to_obj = self.tf_buffer.lookup_transform(
                    "object_frame", self.camera_frame,
                    rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.01))
                T_obj_to_cam = np.linalg.inv(msg_to_matrix(tf_cam_to_obj))

                result = self._decode_compressed_depth(msg)
                if result is not None:
                    _, depth_scale = result
                    self._render_and_publish(T_obj_to_cam, self._T_model_in_obj,
                                             msg.header.stamp, depth_scale)
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f"Render TF lookup failed: {e}",
                                       throttle_duration_sec=2.0)

    # -----------------------------------------------------------------------
    # Main processing pipeline
    # -----------------------------------------------------------------------

    def _process_frame(self, msg: CompressedImage):
        stamp = msg.header.stamp
        self.get_logger().info("process_frame: start", throttle_duration_sec=2.0)

        # 1. Decode compressed depth image
        result = self._decode_compressed_depth(msg)
        if result is None:
            return
        depth_m, _ = result
        self.get_logger().info("process_frame: depth decoded", throttle_duration_sec=2.0)

        # 2. Look up transform: camera_frame -> object_frame
        try:
            tf_cam_to_obj = self.tf_buffer.lookup_transform(
                "object_frame", self.camera_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return
        T_cam_to_obj = msg_to_matrix(tf_cam_to_obj)
        self.get_logger().info("process_frame: TF lookup OK", throttle_duration_sec=2.0)

        # 3. Deproject depth to 3-D points in camera frame, then to object_frame
        obs_points = self._deproject_and_transform(depth_m, T_cam_to_obj)
        if obs_points is None or len(obs_points) < 50:
            self.get_logger().warn(
                f"Too few observed points after filtering: "
                f"{len(obs_points) if obs_points is not None else 0}",
                throttle_duration_sec=2.0)
            return
        self.get_logger().info(
            f"process_frame: deprojected {len(obs_points)} pts", throttle_duration_sec=2.0)

        # 4. Crop to region of interest around origin (position_bound + margin)
        roi = self.pos_bound + self.roi_margin
        mask =  (np.abs(obs_points[:, 0]) < roi) &  \
                (np.abs(obs_points[:, 1]) < roi) & \
                (obs_points[:, 2] > 0.01)  # keep only points above the object_frame
        obs_points = obs_points[mask]
        if len(obs_points) < 50:
            self.get_logger().warn(
                f"Too few points after ROI crop: {len(obs_points)}", throttle_duration_sec=2.0)
            return
        self.get_logger().info(
            f"process_frame: {len(obs_points)} pts after ROI crop", throttle_duration_sec=2.0)

        # 5. Voxel downsample observed cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd = pcd.voxel_down_sample(self.obs_voxel)
        obs_points = np.asarray(pcd.points)
        self.get_logger().info(
            f"process_frame: {len(obs_points)} pts after voxel downsample", throttle_duration_sec=2.0)
        self._publish_obs_cloud(obs_points, stamp)

        # 6. Build KD-tree on observed cloud (queried once per particle)
        obs_tree = o3d.geometry.KDTreeFlann(pcd)
        self.get_logger().info("process_frame: KD-tree built", throttle_duration_sec=2.0)

        # 7. Particle filter ---
        self.get_logger().info("process_frame: starting predict", throttle_duration_sec=2.0)
        self._predict()
        self.get_logger().info(
            f"process_frame: starting update ({self.N} particles, "
            f"{len(self.ref_cloud)} ref pts, {len(obs_points)} obs pts)",
            throttle_duration_sec=2.0)
        self._update(obs_points, obs_tree)
        self.get_logger().info("process_frame: update done, estimating", throttle_duration_sec=2.0)
        best_tx, best_ty, best_theta = self._estimate()
        self._resample()

        # 8. Publish estimated pose and particle visualisation
        self._publish_pose(best_tx, best_ty, best_theta, stamp)
        self._publish_particles(stamp)

        # 9. Save T_model_in_obj for rendering (done in _depth_cb at full frame rate)
        self._T_model_in_obj = pose_matrix(best_tx, best_ty, best_theta)
        self.get_logger().info(
            f"process_frame: complete  tx={best_tx:.4f} ty={best_ty:.4f} "
            f"theta={np.degrees(best_theta):.1f}°",
            throttle_duration_sec=2.0)

    # -----------------------------------------------------------------------
    # Depth decoding
    # -----------------------------------------------------------------------

    def _decode_compressed_depth(self, msg: CompressedImage) -> tuple[np.ndarray, float] | None:
        """Decode a compressed or compressedDepth image.

        compressedDepth images have a 12-byte ConfigHeader prepended before
        the PNG payload; regular compressed images (format "png"/"jpeg") do not.

        Returns (depth_metres, depth_scale) or None on failure.
        depth_scale is the raw→metres factor used so the renderer can
        convert back to the same encoding.
        """
        np_arr = np.frombuffer(msg.data, np.uint8)
        if np_arr.size == 0:
            self.get_logger().warn("Empty compressedDepth data, skipping frame",
                                   throttle_duration_sec=2.0)
            return None

        # compressedDepth transport prepends a 12-byte ConfigHeader before the PNG.
        # Regular compressed transport (png/jpeg) has no such header.
        if "compressedDepth" in msg.format:
            if np_arr.size <= 12:
                self.get_logger().warn("compressedDepth payload too short, skipping frame",
                                       throttle_duration_sec=2.0)
                return None
            png_data = np_arr[12:]
        else:
            png_data = np_arr

        depth = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)
        if depth is None:
            self.get_logger().warn("Failed to decode depth image, skipping frame",
                                   throttle_duration_sec=2.0)
            return None

        if depth.dtype == np.uint16:
            depth_scale = 1e-3          # 16UC1 in millimetres
            depth_m = depth.astype(np.float32) * depth_scale
        else:
            depth_scale = 1.0           # already in metres
            depth_m = depth.astype(np.float32)

        return depth_m, depth_scale

    # -----------------------------------------------------------------------
    # Point cloud helpers
    # -----------------------------------------------------------------------

    def _deproject_and_transform(self, depth_m: np.ndarray,
                                  T_cam_to_obj: np.ndarray) -> np.ndarray | None:
        """Deproject depth image (already in metres) to 3-D points in *object_frame*."""
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        # Build pixel coordinate grids
        v, u = np.where(depth_m > 0)
        z = depth_m[v, u].astype(np.float64)

        # Filter by usable range
        valid = (z >= self.depth_min) & (z <= self.depth_max)
        u, v, z = u[valid], v[valid], z[valid]
        if len(z) == 0:
            return None

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.column_stack([x, y, z])                     # (K, 3)

        # Transform to object_frame
        pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])  # (K, 4)
        pts_obj = (T_cam_to_obj @ pts_h.T).T[:, :3]
        return pts_obj

    # -----------------------------------------------------------------------
    # Particle filter steps
    # -----------------------------------------------------------------------

    def _predict(self):
        noise_t = np.random.normal(0, self.sigma_t, (self.N, 2))
        noise_r = np.random.normal(0, self.sigma_r, self.N)
        self.particles[:, :2] += noise_t
        self.particles[:, 2]  += noise_r
        # Wrap angle to [-π, π]
        self.particles[:, 2] = np.arctan2(
            np.sin(self.particles[:, 2]),
            np.cos(self.particles[:, 2]))

    def _update(self, obs_points: np.ndarray,
                obs_tree: o3d.geometry.KDTreeFlann):
        """Score each particle against the observed point cloud."""
        ref = self.ref_cloud  # (M, 3)

        for i in range(self.N):
            tx, ty, theta = self.particles[i]
            c, s = np.cos(theta), np.sin(theta)
            # Rotate + translate reference cloud  (only xy rotation, z unchanged)
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], dtype=np.float64)
            hyp = (R @ ref.T).T + np.array([tx, ty, 0.0])

            # Query closest observed point for each hypothesis point
            dists = np.empty(len(hyp))
            for j, pt in enumerate(hyp):
                _, _, d2 = obs_tree.search_knn_vector_3d(pt, 1)
                dists[j] = d2[0]
            dists = np.sqrt(dists)  # KDTree returns squared distances

            if self.use_inlier:
                inlier_ratio = np.mean(dists < self.inlier_thresh)
                self.weights[i] = inlier_ratio + 1e-10
            else:
                mean_d = np.mean(dists)
                self.weights[i] = np.exp(-mean_d**2 / (2 * self.sigma_obs**2))

        # Normalise
        w_sum = self.weights.sum()
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights[:] = 1.0 / self.N

    def _estimate(self) -> tuple[float, float, float]:
        """Weighted mean estimate (circular mean for angle)."""
        tx = np.average(self.particles[:, 0], weights=self.weights)
        ty = np.average(self.particles[:, 1], weights=self.weights)
        theta = circular_mean(self.particles[:, 2], self.weights)
        return tx, ty, theta

    def _resample(self):
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.N * self.resample_ratio:
            idx = systematic_resample(self.weights)
            self.particles = self.particles[idx].copy()
            self.weights[:] = 1.0 / self.N

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def _publish_obs_cloud(self, points: np.ndarray, stamp):
        """Publish observed points (in object_frame) as a PointCloud2 message."""
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = "object_frame"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12          # 3 × float32
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        self.obs_cloud_pub.publish(msg)

    def _publish_particles(self, stamp):
        """Publish all particles as PoseArray and a weight-coloured MarkerArray."""
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = "object_frame"

        ma = MarkerArray()
        w_max = self.weights.max()
        w_norm = self.weights / w_max if w_max > 0 else self.weights

        for i, (tx, ty, theta) in enumerate(self.particles):
            q = Rotation.from_euler("z", theta).as_quat()  # [x, y, z, w]

            pose = Pose()
            pose.position.x = float(tx)
            pose.position.y = float(ty)
            pose.position.z = 0.0
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pa.poses.append(pose)

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "object_frame"
            m.ns = "particles"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = pose
            m.scale.x = 0.015   # arrow length
            m.scale.y = 0.003   # shaft diameter
            m.scale.z = 0.003
            w = float(w_norm[i])
            m.color.r = w
            m.color.g = 0.0
            m.color.b = 1.0 - w
            m.color.a = 0.7
            m.lifetime.sec = 1
            m.lifetime.nanosec = 0
            ma.markers.append(m)

        self.particles_pub.publish(pa)
        self.markers_pub.publish(ma)

    def _publish_pose(self, tx, ty, theta, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "object_frame"
        msg.pose.position.x = tx
        msg.pose.position.y = ty
        msg.pose.position.z = 0.0
        q = Rotation.from_euler("z", theta).as_quat()  # [x, y, z, w]
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pose_pub.publish(msg)

    def _render_and_publish(self, T_obj_to_cam: np.ndarray,
                            T_model_in_obj: np.ndarray,
                            stamp, depth_scale: float):
        """Render the mesh from the camera viewpoint and publish a depth image."""
        # Extrinsic: model-frame → camera-frame
        T_model_to_cam = T_obj_to_cam @ T_model_in_obj

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]
        w  = self.intrinsics["width"]
        h  = self.intrinsics["height"]

        intrinsic_matrix = o3d.core.Tensor(
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1 ]], dtype=o3d.core.float64)

        extrinsic_matrix = o3d.core.Tensor(T_model_to_cam, dtype=o3d.core.float64)

        rays = self.ray_scene.create_rays_pinhole(
            intrinsic_matrix, extrinsic_matrix, w, h)

        result = self.ray_scene.cast_rays(rays)
        t_hit = result["t_hit"].numpy()                 # (H, W) – Euclidean distance

        # Convert Euclidean ray distance to z-depth
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        dx = (u_grid - cx) / fx
        dy = (v_grid - cy) / fy
        cos_factor = 1.0 / np.sqrt(dx**2 + dy**2 + 1.0)
        z_depth = t_hit * cos_factor                    # metres

        # Convert to 16UC1 (same scale as raw depth)
        z_depth[~np.isfinite(z_depth)] = 0.0
        depth_uint16 = (z_depth / depth_scale).astype(np.uint16)

        # Build and publish raw Image message
        img_msg = Image()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.camera_frame
        img_msg.height = h
        img_msg.width = w
        img_msg.encoding = "16UC1"
        img_msg.is_bigendian = False
        img_msg.step = w * 2
        img_msg.data = depth_uint16.tobytes()
        self.depth_pub.publish(img_msg)

        # Build and publish compressedDepth message (12-byte ConfigHeader + PNG)
        ok, png_buf = cv2.imencode(".png", depth_uint16)
        if ok:
            # ConfigHeader: int32 format=0, float32 depthParam[2]={0,0}
            config_header = struct.pack("<iff", 0, 0.0, 0.0)
            comp_msg = CompressedImage()
            comp_msg.header = img_msg.header
            comp_msg.format = "16UC1; compressedDepth png"
            comp_msg.data = config_header + png_buf.tobytes()
            self.depth_compressed_pub.publish(comp_msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
