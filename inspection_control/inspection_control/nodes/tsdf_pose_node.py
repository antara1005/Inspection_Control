#!/usr/bin/env python3
"""
TSDF-based pose estimation node.

Fuses incoming depth frames from the RealSense D405 into an Open3D
VoxelBlockGrid TSDF expressed in a fixed `fusion_frame` (default
`object_frame`), extracts a fused point cloud every N frames, and
registers it against a reference point cloud sampled from the CAD mesh
(FPFH + RANSAC for initialization, point-to-plane ICP for refinement).
Publishes the resulting 6-DOF object pose in `fusion_frame` and broadcasts
the `fusion_frame`->`model_frame` TF (the authoritative placement consumed
by viewpoint_generation and the GUI), plus a synthetic depth image rendered
from the registered mesh.

The reference cloud is sampled from the same CAD mesh used for the
synthetic-depth render; the node only tracks `model.mesh.*` from the
viewpoint_generation node and no longer depends on a separately-sampled
whole-part point cloud.
"""

import struct
import threading
import numpy as np
import cv2
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger

import tf2_ros
from tf2_ros import TransformBroadcaster
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


def msg_to_matrix(transform_msg: TransformStamped) -> np.ndarray:
    """Convert geometry_msgs/TransformStamped to a 4x4 numpy matrix.

    The returned matrix maps points expressed in `child_frame_id` to points
    expressed in `header.frame_id` (i.e. the pose of the child frame
    expressed in the parent frame).
    """
    t = transform_msg.transform.translation
    q = transform_msg.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[0, 3] = t.x; T[1, 3] = t.y; T[2, 3] = t.z
    return T


def matrix_to_pose(T: np.ndarray) -> Pose:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
    p = Pose()
    p.position.x = float(T[0, 3])
    p.position.y = float(T[1, 3])
    p.position.z = float(T[2, 3])
    p.orientation.x = float(q[0])
    p.orientation.y = float(q[1])
    p.orientation.z = float(q[2])
    p.orientation.w = float(q[3])
    return p


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TSDFPoseNode(Node):
    def __init__(self):
        super().__init__("tsdf_pose")

        # -- Parameters -----------------------------------------------------
        self.declare_parameters("", [
            # Frames
            ("fusion_frame",                "object_frame"),
            ("world_frame",                 "world"),
            ("model_frame",                 "model_frame"),
            ("publish_tf",                  True),

            # TSDF integration (VoxelBlockGrid)
            ("voxel_size",                  0.001),    # m
            ("sdf_trunc_mult",              4.0),      # sdf_trunc = mult * voxel_size
            ("block_resolution",            16),
            ("block_count",                 10000),
            ("integration_device",          "CUDA:0"),  # "CUDA:0" if available
            ("integrate_every_n_frames",    1),

            # Depth sensor
            ("depth_min",                   0.07),     # m
            ("depth_max",                   0.50),     # m
            ("depth_max_integration",       0.50),     # m – truncate depth before integrating

            # Registration cadence
            ("registration_enabled",        True),     # master switch for pose registration
            ("register_every_n_frames",     30),
            ("min_integrations_before_register", 10),

            # Surface-cloud publish cadence (decoupled from registration)
            ("surface_publish_every_n_frames", 10),

            # Point cloud extraction from VBG
            ("extract_weight_threshold",    3.0),      # min TSDF weight per voxel

            # Fused-cloud crop (axis-aligned box in fusion_frame, metres)
            ("crop_enabled",                True),
            ("crop_min",                    [-0.5, -0.5, 0.01]),
            ("crop_max",                    [0.5,  0.5,  1.0]),

            # Turntable-plane removal (RANSAC in fusion_frame). The turntable top
            # is a static, ~horizontal plane in fusion_frame, so the fitted plane
            # is cached and only re-fit every `plane_refit_every` extractions.
            ("remove_turntable",            True),
            ("plane_distance_mult",         2.0),    # RANSAC inlier dist = mult * voxel_size
            ("plane_ransac_iterations",     200),
            ("plane_min_verticality",       0.9),    # |n_z| after normalising; reject tilted planes
            ("plane_min_inliers",           200),    # min inliers to accept a plane as the table
            ("plane_margin",                0.002),  # m – also drop points within this margin above it
            ("plane_refit_every",           5),      # re-run RANSAC every N extractions (cache between)

            # Reference cloud sampled from the CAD mesh (Poisson-disk)
            ("reference_sample_points",     50000),    # points sampled from the mesh surface

            # Registration: downsampling
            ("fused_voxel_size",            0.003),    # m
            ("reference_voxel_size",        0.003),    # m

            # Registration: features
            ("normal_radius_mult",          2.0),      # * voxel_size
            ("fpfh_radius_mult",            5.0),      # * voxel_size

            # Registration: RANSAC
            ("ransac_distance_mult",        1.5),      # * voxel_size
            ("ransac_max_iterations",       100000),
            ("ransac_confidence",           0.999),

            # Registration: ICP
            ("icp_distance_mult",           1.5),      # * voxel_size
            ("icp_max_iterations",          50),
            ("icp_fitness_min",             0.3),

            # External
            ("viewpoint_generation_node",   "viewpoint_generation"),

            # Topics
            ("pose_topic",                  "~/pose"),
            ("fused_cloud_topic",           "~/fused_cloud"),
            ("surface_cloud_topic",         "/perception/surface_cloud"),
            ("synthetic_depth_topic",       "~/depth/synthetic"),
            ("model_marker_topic",          "~/model_marker"),
        ])

        # Convert to attributes
        gp = self.get_parameter
        self.fusion_frame    = gp("fusion_frame").value
        self.world_frame     = gp("world_frame").value
        self.model_frame     = gp("model_frame").value
        self.publish_tf      = gp("publish_tf").value

        self.voxel_size      = float(gp("voxel_size").value)
        self.sdf_trunc_mult  = float(gp("sdf_trunc_mult").value)
        self.sdf_trunc       = self.sdf_trunc_mult * self.voxel_size
        self.block_resolution = int(gp("block_resolution").value)
        self.block_count     = int(gp("block_count").value)
        self.device_str      = gp("integration_device").value
        self.integrate_every = int(gp("integrate_every_n_frames").value)

        self.depth_min       = float(gp("depth_min").value)
        self.depth_max       = float(gp("depth_max").value)
        self.depth_max_int   = float(gp("depth_max_integration").value)

        self.registration_enabled = bool(gp("registration_enabled").value)
        self.register_every  = int(gp("register_every_n_frames").value)
        self.min_int_before_reg = int(gp("min_integrations_before_register").value)
        self.surface_publish_every = int(gp("surface_publish_every_n_frames").value)

        self.extract_weight_threshold = float(gp("extract_weight_threshold").value)

        self.crop_enabled = bool(gp("crop_enabled").value)
        self.crop_min = np.asarray(gp("crop_min").value, dtype=np.float64)
        self.crop_max = np.asarray(gp("crop_max").value, dtype=np.float64)

        self.remove_turntable = bool(gp("remove_turntable").value)
        self.plane_dist_mult = float(gp("plane_distance_mult").value)
        self.plane_ransac_iters = int(gp("plane_ransac_iterations").value)
        self.plane_min_verticality = float(gp("plane_min_verticality").value)
        self.plane_min_inliers = int(gp("plane_min_inliers").value)
        self.plane_margin = float(gp("plane_margin").value)
        self.plane_refit_every = int(gp("plane_refit_every").value)

        self.reference_sample_points = int(gp("reference_sample_points").value)

        self.fused_voxel     = float(gp("fused_voxel_size").value)
        self.ref_voxel       = float(gp("reference_voxel_size").value)

        self.normal_r_mult   = float(gp("normal_radius_mult").value)
        self.fpfh_r_mult     = float(gp("fpfh_radius_mult").value)

        self.ransac_dist_mult = float(gp("ransac_distance_mult").value)
        self.ransac_max_iter = int(gp("ransac_max_iterations").value)
        self.ransac_conf     = float(gp("ransac_confidence").value)

        self.icp_dist_mult   = float(gp("icp_distance_mult").value)
        self.icp_max_iter    = int(gp("icp_max_iterations").value)
        self.icp_fitness_min = float(gp("icp_fitness_min").value)

        self.vg_node_name    = gp("viewpoint_generation_node").value

        # -- State ----------------------------------------------------------
        self.intrinsics = None       # dict with fx, fy, cx, cy, width, height
        self.camera_frame = None
        self.mesh = None             # o3d.geometry.TriangleMesh (metres, model frame)
        self._model_marker = None    # cached TRIANGLE_LIST marker (geometry built once)
        self.ray_scene = None        # o3d.t.geometry.RaycastingScene for synth depth
        self.ref_cloud = None        # o3d.geometry.PointCloud (downsampled, with normals)
        self.ref_fpfh = None         # FPFH feature for global registration
        self._ref_pcd_raw = None     # reference cloud sampled from the mesh (for live rebuild)

        self.device = o3d.core.Device(self.device_str)
        self.vbg = None              # o3d.t.geometry.VoxelBlockGrid
        self.frame_count = 0
        self.integration_count = 0
        self.have_global_init = False
        self.T_model_in_fusion = None   # latest pose estimate (4x4)
        self.last_pose_stamp = None

        # Cached turntable plane [a,b,c,d] (normal normalised, pointing +z toward
        # the object). The plane is static in fusion_frame, so it is only re-fit
        # every plane_refit_every extractions; None forces a fit on next use.
        self._cached_plane = None
        self._plane_fit_counter = 0

        # Independent readiness flags (decoupled from CAD availability):
        #   tsdf_ready      -> VBG exists; integration + surface publishing allowed
        #   reference_ready -> reference cloud sampled from the mesh; registration / ICP allowed
        #   mesh_ready      -> CAD mesh loaded; synthetic-depth render allowed
        self.tsdf_ready = False
        self.reference_ready = False
        self.mesh_ready = False
        self.integ_lock = threading.Lock()
        self.reg_lock = threading.Lock()
        self.processing_reg = False

        # -- TF -------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # -- QoS Profiles ------------------------------------------------------

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # -- Publishers -----------------------------------------------------
        synth_topic = gp("synthetic_depth_topic").value
        pose_topic  = gp("pose_topic").value
        cloud_topic = gp("fused_cloud_topic").value
        surface_topic = gp("surface_cloud_topic").value
        self.pose_pub  = self.create_publisher(PoseStamped,    pose_topic,                  10)
        self.cloud_pub = self.create_publisher(PointCloud2,    cloud_topic,                 10)
        # Source-agnostic surface contract: latched (TRANSIENT_LOCAL) so a late-joining
        # consumer (orientation control / future Zivid swap) immediately gets the most
        # recent fused surface. Carries per-point normals.
        surface_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.surface_pub = self.create_publisher(PointCloud2, surface_topic, surface_qos)
        # Model mesh as a latched marker (geometry published once; animated by the
        # model_frame TF). TRANSIENT_LOCAL so it renders as soon as the TF appears
        # and for any late-joining viewer (e.g. Foxglove).
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        marker_topic = gp("model_marker_topic").value
        self.model_marker_pub = self.create_publisher(Marker, marker_topic, marker_qos)
        self.depth_pub = self.create_publisher(Image,          synth_topic,                 10)
        self.depth_compressed_pub = self.create_publisher(
            CompressedImage, synth_topic + "/compressedDepth", 10)
        # CameraInfo must sit alongside its image (ROS convention: same parent
        # namespace, named "camera_info") so subscribers pair them automatically.
        # Derive it from synth_topic so it tracks any override of that parameter.
        info_topic = synth_topic.rsplit("/", 1)[0] + "/camera_info"
        self.camera_info_pub = self.create_publisher(
            CameraInfo, info_topic, 10)

        # -- Subscribers ----------------------------------------------------
        self._depth_cb_group = ReentrantCallbackGroup()
        self.create_subscription(
            CameraInfo,
            "/camera/d405_camera/depth/camera_info",
            self._camera_info_cb, qos)
        self.create_subscription(
            CompressedImage,
            "/camera/d405_camera/depth/image_rect_raw/compressedDepth",
            self._depth_cb, qos,
            callback_group=self._depth_cb_group)

        # -- Services -------------------------------------------------------
        self.create_service(Trigger, "~/reset", self._on_reset)
        self.create_service(Trigger, "~/register_now", self._on_register_now)

        # -- Fetch model paths from viewpoint_generation node ---------------
        self._param_client = self.create_client(
            GetParameters, f"/{self.vg_node_name}/get_parameters")
        self._current_mesh_file  = None
        self._current_mesh_units = None

        self.get_logger().info(
            "Polling viewpoint_generation for model parameters every 5 s …")
        self.create_timer(5.0, self._poll_model_params)

        # React to live parameter updates (e.g. `ros2 param set ...`).
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Build the voxel grid up front so fusion can run before — or entirely
        # without — CAD. Only registration (ICP) and the CAD synthetic-depth render
        # depend on model data; integration does not.
        self._reset_tsdf()
        self.tsdf_ready = True

    # -----------------------------------------------------------------------
    # Live parameter updates
    # -----------------------------------------------------------------------

    def _on_set_parameters(self, params):
        """Apply on-the-fly parameter changes.

        Scalar tuning values take effect immediately. Changes to the reference
        cloud sampling trigger a reference rebuild; changes to the VBG geometry
        rebuild the voxel grid (which clears accumulated fusion).
        """
        pending = {}             # attr name -> new value (applied only if all valid)
        rebuild_reference = False
        resample_reference = False
        rebuild_tsdf = False

        def reject(reason):
            return SetParametersResult(successful=False, reason=reason)

        for p in params:
            n, v = p.name, p.value

            # --- live scalars ------------------------------------------
            if n == "publish_tf":
                pending["publish_tf"] = bool(v)
            elif n == "registration_enabled":
                pending["registration_enabled"] = bool(v)
            elif n == "integrate_every_n_frames":
                if int(v) < 1:
                    return reject("integrate_every_n_frames must be >= 1")
                pending["integrate_every"] = int(v)
            elif n == "depth_min":
                pending["depth_min"] = float(v)
            elif n == "depth_max":
                pending["depth_max"] = float(v)
            elif n == "depth_max_integration":
                pending["depth_max_int"] = float(v)
            elif n == "register_every_n_frames":
                if int(v) < 1:
                    return reject("register_every_n_frames must be >= 1")
                pending["register_every"] = int(v)
            elif n == "surface_publish_every_n_frames":
                if int(v) < 1:
                    return reject("surface_publish_every_n_frames must be >= 1")
                pending["surface_publish_every"] = int(v)
            elif n == "min_integrations_before_register":
                pending["min_int_before_reg"] = int(v)
            elif n == "extract_weight_threshold":
                if float(v) < 0:
                    return reject("extract_weight_threshold must be >= 0")
                pending["extract_weight_threshold"] = float(v)
            elif n == "crop_enabled":
                pending["crop_enabled"] = bool(v)
            elif n == "crop_min":
                if len(v) != 3:
                    return reject("crop_min must have 3 elements")
                pending["crop_min"] = np.asarray(v, dtype=np.float64)
            elif n == "crop_max":
                if len(v) != 3:
                    return reject("crop_max must have 3 elements")
                pending["crop_max"] = np.asarray(v, dtype=np.float64)
            elif n == "remove_turntable":
                pending["remove_turntable"] = bool(v)
            elif n == "plane_distance_mult":
                if float(v) <= 0:
                    return reject("plane_distance_mult must be > 0")
                pending["plane_dist_mult"] = float(v)
            elif n == "plane_ransac_iterations":
                if int(v) < 1:
                    return reject("plane_ransac_iterations must be >= 1")
                pending["plane_ransac_iters"] = int(v)
            elif n == "plane_min_verticality":
                pending["plane_min_verticality"] = float(v)
            elif n == "plane_min_inliers":
                if int(v) < 0:
                    return reject("plane_min_inliers must be >= 0")
                pending["plane_min_inliers"] = int(v)
            elif n == "plane_margin":
                pending["plane_margin"] = float(v)
            elif n == "plane_refit_every":
                if int(v) < 1:
                    return reject("plane_refit_every must be >= 1")
                pending["plane_refit_every"] = int(v)
            elif n == "fused_voxel_size":
                if float(v) <= 0:
                    return reject("fused_voxel_size must be > 0")
                pending["fused_voxel"] = float(v)
            elif n == "ransac_distance_mult":
                pending["ransac_dist_mult"] = float(v)
            elif n == "ransac_max_iterations":
                pending["ransac_max_iter"] = int(v)
            elif n == "ransac_confidence":
                pending["ransac_conf"] = float(v)
            elif n == "icp_distance_mult":
                pending["icp_dist_mult"] = float(v)
            elif n == "icp_max_iterations":
                pending["icp_max_iter"] = int(v)
            elif n == "icp_fitness_min":
                pending["icp_fitness_min"] = float(v)

            # --- frames ------------------------------------------------
            elif n == "fusion_frame":
                pending["fusion_frame"] = str(v)
            elif n == "world_frame":
                pending["world_frame"] = str(v)
            elif n == "model_frame":
                pending["model_frame"] = str(v)

            # --- reference-cloud resample (from the mesh) --------------
            elif n == "reference_sample_points":
                if int(v) < 1:
                    return reject("reference_sample_points must be >= 1")
                pending["reference_sample_points"] = int(v)
                resample_reference = True

            # --- reference-cloud rebuild -------------------------------
            elif n == "reference_voxel_size":
                if float(v) <= 0:
                    return reject("reference_voxel_size must be > 0")
                pending["ref_voxel"] = float(v)
                rebuild_reference = True
            elif n == "normal_radius_mult":
                pending["normal_r_mult"] = float(v)
                rebuild_reference = True
            elif n == "fpfh_radius_mult":
                pending["fpfh_r_mult"] = float(v)
                rebuild_reference = True

            # --- VBG structural rebuild --------------------------------
            elif n == "voxel_size":
                if float(v) <= 0:
                    return reject("voxel_size must be > 0")
                pending["voxel_size"] = float(v)
                rebuild_tsdf = True
            elif n == "sdf_trunc_mult":
                # sdf_trunc is derived but not consumed by the VBG in this
                # node, so updating it does not require clearing fusion.
                pending["sdf_trunc_mult"] = float(v)
            elif n == "block_resolution":
                if int(v) < 1:
                    return reject("block_resolution must be >= 1")
                pending["block_resolution"] = int(v)
                rebuild_tsdf = True
            elif n == "block_count":
                if int(v) < 1:
                    return reject("block_count must be >= 1")
                pending["block_count"] = int(v)
                rebuild_tsdf = True
            elif n == "integration_device":
                pending["device_str"] = str(v)
                rebuild_tsdf = True
            # else: unknown / static parameter — accept without side effects.

        # All validated — commit cached attributes.
        for attr, value in pending.items():
            setattr(self, attr, value)

        # Recompute derived values.
        if "voxel_size" in pending or "sdf_trunc_mult" in pending:
            self.sdf_trunc = self.sdf_trunc_mult * self.voxel_size

        # Plane-fit tuning (or voxel_size, which scales the inlier distance)
        # changed -> drop the cached plane so it re-fits on the next extraction.
        if any(k in pending for k in
               ("plane_dist_mult", "plane_ransac_iters", "plane_min_verticality",
                "plane_min_inliers", "voxel_size")):
            self._cached_plane = None

        # toggle TF broadcaster on/off if publish_tf changed
        if "publish_tf" in pending:
            if self.publish_tf and self.tf_broadcaster is None:
                self.tf_broadcaster = TransformBroadcaster(self)
            elif not self.publish_tf:
                self.tf_broadcaster = None

        # Re-sampling supersedes a plain rebuild (it rebuilds internally).
        if resample_reference and self.mesh is not None:
            try:
                self._sample_reference_from_mesh()
                self.get_logger().info(
                    "Reference cloud re-sampled from mesh after parameter update.")
            except Exception as e:
                return reject(f"Reference resample failed: {e}")
        elif rebuild_reference:
            try:
                self._build_reference()
                self.get_logger().info("Reference cloud rebuilt from parameter update.")
            except Exception as e:
                return reject(f"Reference rebuild failed: {e}")

        if rebuild_tsdf:
            try:
                if "device_str" in pending:
                    self.device = o3d.core.Device(self.device_str)
                with self.integ_lock:
                    self._reset_tsdf()
                self.get_logger().warn(
                    "VBG rebuilt from parameter update; accumulated fusion cleared.")
            except Exception as e:
                return reject(f"VBG rebuild failed: {e}")

        return SetParametersResult(successful=True)

    # -----------------------------------------------------------------------
    # Model loading (mirrors particle_filter_node)
    # -----------------------------------------------------------------------

    def _poll_model_params(self):
        if not self._param_client.service_is_ready():
            self.get_logger().info(
                "viewpoint_generation parameter service not ready yet, retrying …",
                throttle_duration_sec=10.0)
            return
        request = GetParameters.Request()
        request.names = ["model.mesh.file", "model.mesh.units"]
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

        # Reset accumulated fusion ONLY when the object identity changes, i.e. a
        # previously-loaded non-empty mesh switches to a *different* non-empty mesh.
        # The empty->set transition (we fused CAD-free, then CAD is provided for the
        # same object) must keep the fused surface and merely enable the CAD-dependent
        # capabilities (registration / synthetic depth).
        def _identity_change(old, new):
            return bool(old) and bool(new) and old != new

        reset_needed = _identity_change(self._current_mesh_file, mesh_file)

        # --- CAD mesh -> synthetic-depth render (mesh_ready) AND the reference
        # cloud used for registration/ICP (reference_ready), sampled from the mesh ---
        if mesh_file and (mesh_file != self._current_mesh_file or
                          mesh_units != self._current_mesh_units):
            try:
                self.mesh_ready = False
                self.reference_ready = False
                self._load_mesh(mesh_file, mesh_units)
                self._current_mesh_file  = mesh_file
                self._current_mesh_units = mesh_units
                self.get_logger().info(f"Mesh loaded: {mesh_file} ({mesh_units})")
            except Exception as e:
                self.get_logger().error(f"Failed to load mesh: {e}")

        if not mesh_file:
            self.get_logger().info(
                "No CAD mesh set in viewpoint_generation; fusing CAD-free "
                "(registration and synthetic depth disabled).",
                throttle_duration_sec=30.0)

        if reset_needed:
            with self.integ_lock:
                self._reset_tsdf()
            self.get_logger().warn(
                "Object identity changed; cleared accumulated TSDF fusion.")

    def _load_mesh(self, mesh_path, mesh_units):
        """Load the CAD mesh for raycast / synthetic depth (sets mesh_ready) and
        sample a reference point cloud from it for registration (sets
        reference_ready)."""
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh from {mesh_path}")
        mesh.scale(unit_scale(mesh_units), center=(0, 0, 0))
        mesh.compute_vertex_normals()
        self.mesh = mesh

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.ray_scene = o3d.t.geometry.RaycastingScene()
        self.ray_scene.add_triangles(mesh_t)
        self.mesh_ready = True

        # Publish the mesh geometry once (latched) so Foxglove can overlay it on
        # the fused surface; the model_frame TF moves it on each registration.
        self._publish_model_marker()

        # Sample the reference cloud straight from this mesh: the viewpoint
        # generation node no longer emits a separately-sampled whole-part cloud.
        self._sample_reference_from_mesh()

    def _sample_reference_from_mesh(self):
        """Sample a reference point cloud from the currently loaded mesh and
        (re)build the downsampled reference + FPFH (sets reference_ready).

        Uses Poisson-disk sampling for even surface coverage, matching how the
        whole-part reference cloud was previously generated upstream.
        """
        if self.mesh is None:
            return
        n = max(1, int(self.reference_sample_points))
        pcd = self.mesh.sample_points_poisson_disk(
            number_of_points=n, init_factor=5, use_triangle_normal=True)
        if pcd.is_empty():
            raise RuntimeError("Sampling reference cloud from mesh produced no points")
        # Keep the sampled cloud so the reference can be rebuilt on the fly when
        # ref_voxel / feature-radius parameters change (see _build_reference).
        self._ref_pcd_raw = pcd
        self._build_reference()
        self.reference_ready = True
        self.get_logger().info(
            f"Reference cloud sampled from mesh: {len(pcd.points)} points")

    def _build_reference(self):
        """(Re)build the downsampled reference cloud + FPFH from the raw cloud.

        Uses the current ref_voxel / normal_radius / fpfh_radius parameters, so
        it can be re-run after a live parameter update.
        """
        if self._ref_pcd_raw is None:
            return
        ref_down = self._ref_pcd_raw.voxel_down_sample(self.ref_voxel)
        ref_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.ref_voxel * self.normal_r_mult, max_nn=30))
        self.ref_cloud = ref_down
        self.ref_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            ref_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.ref_voxel * self.fpfh_r_mult, max_nn=100))
        self.get_logger().info(
            f"Reference cloud: {len(ref_down.points)} points "
            f"(voxel {self.ref_voxel*1e3:.1f} mm), FPFH ready")

    def _reset_tsdf(self):
        """Clear the voxel grid and reset pose / counters."""
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'),
            attr_dtypes=(o3d.core.float32, o3d.core.float32),
            attr_channels=((1), (1)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=self.block_count,
            device=self.device,
        )
        self.integration_count = 0
        self.frame_count = 0
        self.have_global_init = False
        self.T_model_in_fusion = None

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _camera_info_cb(self, msg: CameraInfo):
        K = msg.k
        self.intrinsics = {
            "fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5],
            "width": msg.width, "height": msg.height,
        }
        self.camera_frame = msg.header.frame_id
        self.camera_info_pub.publish(msg)

    def _depth_cb(self, msg: CompressedImage):
        # Integration needs only the VBG + intrinsics + TF — NOT CAD.
        if not self.tsdf_ready or self.intrinsics is None:
            return

        self.frame_count += 1
        do_integrate = (self.frame_count % max(1, self.integrate_every) == 0)
        if do_integrate:
            try:
                self._integrate_frame(msg)
            except Exception as e:
                self.get_logger().error(f"Integration error: {e}",
                                        throttle_duration_sec=2.0)

        # Publish the fused surface on its own cadence, independent of CAD /
        # registration, so it flows as soon as geometry accumulates.
        if (self.integration_count > 0 and
                self.frame_count % max(1, self.surface_publish_every) == 0):
            try:
                self._publish_surface_cloud(msg.header.stamp)
            except Exception as e:
                self.get_logger().error(f"Surface publish error: {e}",
                                        throttle_duration_sec=2.0)

        # Registration (non-blocking guard) — requires the CAD reference cloud.
        if (self.registration_enabled and
                self.reference_ready and
                self.integration_count >= self.min_int_before_reg and
                self.integration_count % max(1, self.register_every) == 0):
            with self.reg_lock:
                if not self.processing_reg:
                    self.processing_reg = True
                    try:
                        self._run_registration(msg.header.stamp)
                    except Exception as e:
                        self.get_logger().error(f"Registration error: {e}",
                                                throttle_duration_sec=2.0)
                    finally:
                        self.processing_reg = False

        # Render synthetic depth from the CAD mesh at the latest known pose.
        if self.mesh_ready and self.T_model_in_fusion is not None:
            try:
                self._render_synthetic_depth(msg)
            except Exception as e:
                self.get_logger().warn(f"Render failed: {e}",
                                       throttle_duration_sec=2.0)

    def _on_reset(self, _request, response):
        with self.integ_lock:
            self._reset_tsdf()
        response.success = True
        response.message = "TSDF reset."
        self.get_logger().info("TSDF reset via service.")
        return response

    def _on_register_now(self, _request, response):
        if not self.registration_enabled:
            response.success = False
            response.message = "Registration is disabled (registration_enabled=False)."
            return response
        if self.integration_count < self.min_int_before_reg:
            response.success = False
            response.message = (
                f"Need at least {self.min_int_before_reg} integrated frames; "
                f"have {self.integration_count}.")
            return response
        with self.reg_lock:
            if self.processing_reg:
                response.success = False
                response.message = "Registration already in progress."
                return response
            self.processing_reg = True
        try:
            ok = self._run_registration(self.get_clock().now().to_msg(),
                                        force_global=True)
            response.success = bool(ok)
            response.message = "Registration complete." if ok else "Registration failed."
        finally:
            with self.reg_lock:
                self.processing_reg = False
        return response

    # -----------------------------------------------------------------------
    # Integration
    # -----------------------------------------------------------------------

    def _integrate_frame(self, msg: CompressedImage):
        depth_raw, depth_scale = self._decode_compressed_depth(msg)
        if depth_raw is None:
            return

        # TF: pose of camera expressed in fusion frame, then invert for extrinsic
        try:
            tf_cam_in_fusion = self.tf_buffer.lookup_transform(
                self.fusion_frame, self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup ({self.fusion_frame} <- {self.camera_frame}) failed: {e}",
                                   throttle_duration_sec=2.0)
            return

        T_cam_in_fusion = msg_to_matrix(tf_cam_in_fusion)
        # VoxelBlockGrid integrate expects extrinsic = world->camera, i.e. fusion->cam
        extrinsic_np = np.linalg.inv(T_cam_in_fusion)

        # Clamp depth to integration range (zero out invalid pixels)
        depth_clipped = depth_raw.copy()
        depth_m = depth_clipped * depth_scale
        invalid = (depth_m < self.depth_min) | (depth_m > self.depth_max_int)
        depth_clipped[invalid] = 0

        fx = self.intrinsics["fx"]; fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]; cy = self.intrinsics["cy"]
        intrinsic = o3d.core.Tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3d.core.float64)
        extrinsic = o3d.core.Tensor(extrinsic_np, dtype=o3d.core.float64)

        # Depth tensor MUST be on the same device as the VBG
        depth_o3d = o3d.t.geometry.Image(
            o3d.core.Tensor(depth_clipped, device=self.device))

        # depth_scale here converts raw values -> metres (Open3D's convention)
        # For uint16 depth at 1mm/unit, depth_scale=1000.0
        ds_for_o3d = 1.0 / depth_scale  # convert "metres per raw unit" -> "raw units per metre"

        with self.integ_lock:
            frustum = self.vbg.compute_unique_block_coordinates(
                depth_o3d, intrinsic, extrinsic, ds_for_o3d, self.depth_max_int)
            self.vbg.integrate(
                frustum, depth_o3d, intrinsic, extrinsic, ds_for_o3d, self.depth_max_int)
        self.integration_count += 1

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def _extract_fused_cloud(self):
        with self.integ_lock:
            pcd_t = self.vbg.extract_point_cloud(
                weight_threshold=self.extract_weight_threshold)
        # extract_point_cloud may return zero points if no surface intersected
        pcd_legacy = pcd_t.to_legacy()
        if len(pcd_legacy.points) == 0:
            return None
        pcd_down = pcd_legacy.voxel_down_sample(self.fused_voxel)
        pcd_down = self._crop_fused_cloud(pcd_down)
        pcd_down = self._remove_turntable_plane(pcd_down)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.fused_voxel * self.normal_r_mult, max_nn=30))
        return pcd_down
    
    def _crop_fused_cloud(self, cloud: o3d.geometry.PointCloud):
        if not self.crop_enabled:
            return cloud
        cloud_bbox = o3d.geometry.AxisAlignedBoundingBox(self.crop_min, self.crop_max)
        return cloud.crop(bounding_box=cloud_bbox)

    def _remove_turntable_plane(self, cloud: o3d.geometry.PointCloud):
        """Remove the turntable top from the fused cloud.

        The turntable surface is a static, ~horizontal plane in fusion_frame, so
        RANSAC is only run every `plane_refit_every` extractions and the plane is
        cached in between — the steady-state per-extraction cost is a single
        vectorised half-space test. Every point below the plane, and every point
        within `plane_margin` above it, is dropped, so the object base is trimmed
        cleanly off the table. The fitted plane is only accepted when its normal
        is near-vertical and it has enough inliers, so a large flat face of the
        object is never mistaken for the table.
        """
        if not self.remove_turntable or len(cloud.points) < 3:
            return cloud

        need_fit = (self._cached_plane is None or
                    self._plane_fit_counter % max(1, self.plane_refit_every) == 0)
        self._plane_fit_counter += 1

        if need_fit:
            plane, inliers = cloud.segment_plane(
                distance_threshold=self.plane_dist_mult * self.voxel_size,
                ransac_n=3,
                num_iterations=max(1, self.plane_ransac_iters))
            coeffs = np.asarray(plane, dtype=np.float64)
            n_norm = np.linalg.norm(coeffs[:3])
            if n_norm < 1e-9:
                return cloud
            # Normalise, then orient the normal to point up (+z), i.e. toward the
            # object sitting on the table.
            coeffs = coeffs / n_norm
            if coeffs[2] < 0.0:
                coeffs = -coeffs
            # Guard: reject tilted planes (object faces) and weak fits. Drop the
            # stale cache so the next extraction re-fits from scratch.
            if (abs(coeffs[2]) < self.plane_min_verticality or
                    len(inliers) < self.plane_min_inliers):
                self._cached_plane = None
                return cloud
            self._cached_plane = coeffs

        if self._cached_plane is None:
            return cloud

        pts = np.asarray(cloud.points, dtype=np.float64)
        # Signed height above the plane (normal points up). Keep only points
        # clearly above it; anything below, or within the margin, is the table.
        signed = pts @ self._cached_plane[:3] + self._cached_plane[3]
        keep = np.nonzero(signed > self.plane_margin)[0]
        return cloud.select_by_index(keep)

    def _run_registration(self, stamp, force_global: bool = False) -> bool:
        fused = self._extract_fused_cloud()
        if fused is None or len(fused.points) < 100:
            self.get_logger().warn(
                f"Fused cloud has too few points "
                f"({0 if fused is None else len(fused.points)}); skipping registration.",
                throttle_duration_sec=2.0)
            return False

        do_global = force_global or (not self.have_global_init)
        if do_global:
            self.get_logger().info("Running global FPFH+RANSAC initialisation …")
            init_T = self._global_register(fused)
            if init_T is None:
                self.get_logger().warn("Global registration failed.")
                return False
        else:
            init_T = self.T_model_in_fusion

        # ICP refinement (point-to-plane). Source = reference, target = fused.
        icp_dist = self.voxel_size * self.icp_dist_mult
        icp_result = o3d.pipelines.registration.registration_icp(
            self.ref_cloud, fused, icp_dist, init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iter))

        if icp_result.fitness < self.icp_fitness_min:
            self.get_logger().warn(
                f"ICP fitness {icp_result.fitness:.3f} below threshold "
                f"{self.icp_fitness_min:.3f}; rejecting result.",
                throttle_duration_sec=2.0)
            return False

        self.T_model_in_fusion = np.array(icp_result.transformation)
        self.have_global_init = True
        self.last_pose_stamp = stamp

        self._publish_pose(self.T_model_in_fusion, stamp)
        self.get_logger().info(
            f"Registered pose: fitness={icp_result.fitness:.3f} "
            f"inlier_rmse={icp_result.inlier_rmse*1e3:.2f} mm "
            f"t=({self.T_model_in_fusion[0,3]:.3f},"
            f"{self.T_model_in_fusion[1,3]:.3f},"
            f"{self.T_model_in_fusion[2,3]:.3f})")
        return True

    def _global_register(self, fused: o3d.geometry.PointCloud):
        # FPFH for the fused cloud
        fused_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            fused,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.fused_voxel * self.fpfh_r_mult, max_nn=100))

        dist = self.voxel_size * self.ransac_dist_mult
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.ref_cloud, fused, self.ref_fpfh, fused_fpfh,
            mutual_filter=True,
            max_correspondence_distance=dist,
            estimation_method=o3d.pipelines.registration
                .TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration
                    .CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration
                    .CorrespondenceCheckerBasedOnDistance(dist),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.ransac_max_iter, self.ransac_conf))

        if result.fitness == 0.0:
            return None
        return np.array(result.transformation)

    # -----------------------------------------------------------------------
    # Depth decoding
    # -----------------------------------------------------------------------

    def _decode_compressed_depth(self, msg: CompressedImage):
        """Decode compressed/compressedDepth image -> (raw_array, metres_per_unit).

        Returns (raw_uint_array, scale_to_metres) or (None, None) on failure.
        For 16UC1 depth (D405 default), raw is uint16 with scale 1e-3.
        """
        np_arr = np.frombuffer(msg.data, np.uint8)
        if np_arr.size == 0:
            return None, None

        if "compressedDepth" in msg.format:
            if np_arr.size <= 12:
                return None, None
            png_data = np_arr[12:]
        else:
            png_data = np_arr

        depth = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None, None

        if depth.dtype == np.uint16:
            return depth, 1e-3
        # Float depth (already in metres) — promote to uint16 mm so the
        # tensor pipeline can use a single common code path.
        depth_u16 = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        return depth_u16, 1e-3

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def _publish_cloud(self, pcd: o3d.geometry.PointCloud, stamp):
        pts = np.asarray(pcd.points, dtype=np.float32)
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.fusion_frame
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = pts.tobytes()
        self.cloud_pub.publish(msg)

    def _publish_surface_cloud(self, stamp):
        """Extract the fused surface (one VBG extraction) and publish it on the
        source-agnostic surface contract (with normals) plus the legacy xyz-only
        fused_cloud. Independent of CAD and registration so it flows as soon as
        geometry accumulates."""
        fused = self._extract_fused_cloud()
        if fused is None or len(fused.points) == 0:
            return
        self._publish_surface_cloud_with_normals(fused, stamp)
        self._publish_cloud(fused, stamp)              # legacy xyz-only fused_cloud

    def _publish_surface_cloud_with_normals(self, pcd: o3d.geometry.PointCloud, stamp):
        """Publish a fused cloud carrying per-point normals on the surface contract
        (sensor_msgs/PointCloud2 with x,y,z + normal_x,normal_y,normal_z)."""
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] == 0:
            return
        if pcd.has_normals():
            nrm = np.asarray(pcd.normals, dtype=np.float32).copy()
            # Best-effort outward orientation (away from the cloud centroid). The
            # consumer is the authority on final sign (it re-orients toward the
            # camera); this is a convenience for RViz / other subscribers.
            center = pts.mean(axis=0)
            flip = np.einsum('ij,ij->i', nrm, pts - center) < 0.0
            nrm[flip] = -nrm[flip]
        else:
            nrm = np.zeros_like(pts)

        data = np.empty((pts.shape[0], 6), dtype=np.float32)
        data[:, 0:3] = pts
        data[:, 3:6] = nrm

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.fusion_frame
        msg.height = 1
        msg.width = pts.shape[0]
        msg.fields = [
            PointField(name="x",        offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",        offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",        offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_x", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_y", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_z", offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = data.tobytes()
        self.surface_pub.publish(msg)

    def _publish_pose(self, T_model_in_fusion: np.ndarray, stamp):
        """Publish the model placement as fusion_frame (object_frame) <- model
        frame: a PoseStamped in fusion_frame plus the fusion_frame->model_frame
        TF. This is the authoritative placement consumers use (viewpoint
        generation's planning scene / occlusion, and the GUI) to position the
        origin-frame mesh and viewpoints. Expressing it in fusion_frame
        (registered directly, no world composition) keeps the placement correct
        as the turntable moves."""
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.fusion_frame
        ps.pose = matrix_to_pose(T_model_in_fusion)
        self.pose_pub.publish(ps)

        if self.publish_tf and self.tf_broadcaster is not None:
            ts = TransformStamped()
            ts.header.stamp = stamp
            ts.header.frame_id = self.fusion_frame
            ts.child_frame_id = self.model_frame
            ts.transform.translation.x = float(T_model_in_fusion[0, 3])
            ts.transform.translation.y = float(T_model_in_fusion[1, 3])
            ts.transform.translation.z = float(T_model_in_fusion[2, 3])
            q = Rotation.from_matrix(T_model_in_fusion[:3, :3]).as_quat()
            ts.transform.rotation.x = float(q[0])
            ts.transform.rotation.y = float(q[1])
            ts.transform.rotation.z = float(q[2])
            ts.transform.rotation.w = float(q[3])
            self.tf_broadcaster.sendTransform(ts)

        # Re-publish the cached mesh marker stamped to match the model_frame TF
        # just broadcast. Renderers transform a marker using the TF at its header
        # stamp, so restamping is what makes the mesh track each pose update
        # instead of staying frozen at its first placement. Geometry is reused
        # from the cache (built once), so this only re-sends, never rebuilds.
        if self._model_marker is not None:
            self._model_marker.header.stamp = stamp
            self.model_marker_pub.publish(self._model_marker)

    def _publish_model_marker(self):
        """Build (once) and publish the CAD mesh as a latched TRIANGLE_LIST marker
        in model_frame.

        The geometry is expensive to assemble, so it is built once and cached in
        self._model_marker. _publish_pose then re-publishes that cached marker
        with each new pose's stamp: a renderer transforms a marker using the TF at
        its header stamp, so a single stale-stamped publish would freeze the mesh
        at its first placement while model_frame keeps moving. Latched
        (TRANSIENT_LOCAL) so it renders as soon as model_frame appears and for any
        late-joining viewer.
        """
        if self.mesh is None:
            return
        verts = np.asarray(self.mesh.vertices, dtype=np.float64)
        tris = np.asarray(self.mesh.triangles)
        if len(tris) == 0:
            return
        # One marker point per triangle vertex (TRIANGLE_LIST expects 3N points).
        tri_verts = verts[tris.reshape(-1)]

        marker = Marker()
        marker.header.frame_id = self.model_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "model_mesh"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        # Semi-transparent so the fused surface stays visible underneath.
        marker.color.r = 0.1
        marker.color.g = 0.6
        marker.color.b = 1.0
        marker.color.a = 0.5
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                         for p in tri_verts]
        self._model_marker = marker
        self.model_marker_pub.publish(marker)
        self.get_logger().info(
            f"Published model marker: {len(tris)} triangles in '{self.model_frame}'.")

    # -----------------------------------------------------------------------
    # Synthetic depth render (for visualisation / sanity check)
    # -----------------------------------------------------------------------

    def _render_synthetic_depth(self, msg: CompressedImage):
        try:
            tf_cam_in_fusion = self.tf_buffer.lookup_transform(
                self.fusion_frame, self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return

        T_cam_in_fusion = msg_to_matrix(tf_cam_in_fusion)
        T_fusion_in_cam = np.linalg.inv(T_cam_in_fusion)
        # Render the mesh (in its model frame) seen from the camera:
        # extrinsic = world(model)->camera = fusion->cam * model->fusion
        T_model_to_cam = T_fusion_in_cam @ self.T_model_in_fusion

        fx = self.intrinsics["fx"]; fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]; cy = self.intrinsics["cy"]
        w  = self.intrinsics["width"]; h = self.intrinsics["height"]

        intrinsic_t = o3d.core.Tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3d.core.float64)
        extrinsic_t = o3d.core.Tensor(T_model_to_cam, dtype=o3d.core.float64)

        rays = self.ray_scene.create_rays_pinhole(intrinsic_t, extrinsic_t, w, h)
        result = self.ray_scene.cast_rays(rays)
        t_hit = result["t_hit"].numpy()

        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        dx = (u_grid - cx) / fx
        dy = (v_grid - cy) / fy
        cos_factor = 1.0 / np.sqrt(dx**2 + dy**2 + 1.0)
        z_depth = t_hit * cos_factor

        z_depth[~np.isfinite(z_depth)] = 0.0
        depth_uint16 = (z_depth * 1000.0).astype(np.uint16)

        img = Image()
        img.header.stamp = msg.header.stamp
        img.header.frame_id = self.camera_frame
        img.height = h; img.width = w
        img.encoding = "16UC1"
        img.is_bigendian = False
        img.step = w * 2
        img.data = depth_uint16.tobytes()
        self.depth_pub.publish(img)

        ok, png_buf = cv2.imencode(".png", depth_uint16)
        if ok:
            config_header = struct.pack("<iff", 0, 0.0, 0.0)
            comp = CompressedImage()
            comp.header = img.header
            comp.format = "16UC1; compressedDepth png"
            comp.data = config_header + png_buf.tobytes()
            self.depth_compressed_pub.publish(comp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = TSDFPoseNode()
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
