import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import os
import shutil
import csv
import rosbag2_py
from rclpy.serialization import deserialize_message
import pathlib
from viewpoint_generation_interfaces.msg import OrientationControlData
from sensor_msgs_py import point_cloud2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def quat_to_rotation_matrix(q):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    # Normalize
    n = np.sqrt(x*x + y*y + z*z + w*w) + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ], dtype=np.float64)


def transform_points(points, transform_msg):
    """Transform points using a TransformStamped message."""
    if len(points) == 0:
        return points

    R = quat_to_rotation_matrix(transform_msg.transform.rotation)
    t = np.array([
        transform_msg.transform.translation.x,
        transform_msg.transform.translation.y,
        transform_msg.transform.translation.z
    ], dtype=np.float64)

    # Transform: p' = R @ p + t
    return (R @ points.T).T + t


def transform_vector(vec, transform_msg):
    """Transform a vector (rotation only, no translation) using a TransformStamped message."""
    R = quat_to_rotation_matrix(transform_msg.transform.rotation)
    return R @ vec


def generate_theta_control_plots(csv_filepath):
    """
    Generate plots showing theta state and control:
    - Top: theta error with filtered error on same plot
    - Middle: derivative and integral
    - Bottom: theta_torque command
    """
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)

    # Convert timestamps to seconds relative to start
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Top plot: theta error and filtered error
    axes[0].plot(timestamps_sec, np.degrees(df['theta_error']),
                 label='Theta Error', color='blue', alpha=0.7)
    axes[0].plot(timestamps_sec, np.degrees(df['theta_error_filtered']),
                 label='Filtered Error', color='red', linewidth=2)
    axes[0].set_ylabel('Error (degrees)')
    axes[0].set_ylim(-90, 90)
    axes[0].set_title('Theta Error Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middle plot: derivative and integral
    ax_deriv = axes[1]
    ax_integ = ax_deriv.twinx()

    line1, = ax_deriv.plot(timestamps_sec, np.degrees(df['dtheta_error']),
                           label='Derivative (dθ/dt)', color='green')
    line2, = ax_integ.plot(timestamps_sec, np.degrees(df['itheta_error']),
                           label='Integral (∫θ dt)', color='orange')

    ax_deriv.set_ylabel('Derivative (deg/s)', color='green')
    ax_integ.set_ylabel('Integral (deg·s)', color='orange')
    ax_deriv.tick_params(axis='y', labelcolor='green')
    ax_integ.tick_params(axis='y', labelcolor='orange')
    ax_deriv.set_title('Derivative and Integral of Theta Error')
    ax_deriv.legend(handles=[line1, line2], loc='upper right')
    ax_deriv.grid(True, alpha=0.3)

    # Bottom plot: theta torque command
    axes[2].plot(timestamps_sec, df['theta_torque_command'],
                 label='Theta Torque', color='purple', linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Torque (Nm)')
    axes[2].set_title('Theta Torque Command')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = csv_path.parent / f'{csv_path.stem}_theta_control.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Theta control plot saved to {plot_filename}")


def generate_roll_control_plots(csv_filepath):
    """
    Generate plots showing roll state and control (similar structure to theta).
    """
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)

    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Top: roll error
    axes[0].plot(timestamps_sec, np.degrees(df['roll_error']),
                 label='Roll Error', color='blue', linewidth=1.5)
    axes[0].set_ylabel('Error (degrees)')
    axes[0].set_title('Roll Error Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middle: derivative and integral
    ax_deriv = axes[1]
    ax_integ = ax_deriv.twinx()

    line1, = ax_deriv.plot(timestamps_sec, np.degrees(df['droll_error']),
                           label='Derivative', color='green')
    line2, = ax_integ.plot(timestamps_sec, np.degrees(df['iroll_error']),
                           label='Integral', color='orange')

    ax_deriv.set_ylabel('Derivative (deg/s)', color='green')
    ax_integ.set_ylabel('Integral (deg·s)', color='orange')
    ax_deriv.tick_params(axis='y', labelcolor='green')
    ax_integ.tick_params(axis='y', labelcolor='orange')
    ax_deriv.set_title('Derivative and Integral of Roll Error')
    ax_deriv.legend(handles=[line1, line2], loc='upper right')
    ax_deriv.grid(True, alpha=0.3)

    # Bottom: roll torque command
    axes[2].plot(timestamps_sec, df['roll_torque_command'],
                 label='Roll Torque', color='purple', linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Torque (Nm)')
    axes[2].set_title('Roll Torque Command')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = csv_path.parent / f'{csv_path.stem}_roll_control.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Roll control plot saved to {plot_filename}")


def estimate_kalman_noise_parameters(csv_filepath):
    """
    Estimate measurement noise (R) and process noise (Q) for Kalman filter tuning.

    Measurement Noise (R): Variance of theta_error during near-stationary periods
    Process Noise (Q): Variance of prediction errors (actual - predicted state change)

    Returns dict with noise estimates and saves analysis plot.
    """
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)

    # Convert timestamps to seconds
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
    dt = np.diff(timestamps_sec)

    theta = df['theta_error'].values
    dtheta = df['dtheta_error'].values

    # === Measurement Noise (R) ===
    # Detect if dataset is mostly stationary (controller off) vs has real motion
    # Compare velocity std to a motion threshold
    dtheta_std = np.std(dtheta)
    motion_threshold = 0.1  # rad/s - below this, consider "no real motion"
    is_stationary_dataset = dtheta_std < motion_threshold

    if is_stationary_dataset:
        # Dataset is stationary - use ALL samples for R estimation
        # This is ideal for measurement noise estimation
        stationary_mask = np.ones(len(theta), dtype=bool)
        velocity_threshold = np.inf
    else:
        # Dataset has motion - find stationary periods
        velocity_threshold = np.percentile(np.abs(dtheta), 25)
        stationary_mask = np.abs(dtheta) < velocity_threshold

    if np.sum(stationary_mask) > 10:
        theta_stationary = theta[stationary_mask]
        # Use high-pass filter to remove slow drift, keep only noise
        theta_detrended = theta_stationary - pd.Series(theta_stationary).rolling(
            window=min(10, len(theta_stationary)//2), center=True, min_periods=1).mean().values
        R_theta = np.var(theta_detrended)
    else:
        # Fallback: use variance of consecutive differences
        R_theta = np.var(np.diff(theta)) / 2  # Divide by 2 for difference variance

    # === Process Noise (Q) ===
    # Process model: theta(k+1) = theta(k) + dtheta(k) * dt
    # Prediction error = actual_change - predicted_change
    theta_predicted = theta[:-1] + dtheta[:-1] * dt
    theta_actual = theta[1:]
    prediction_errors = theta_actual - theta_predicted

    Q_theta = np.var(prediction_errors)

    # Also estimate derivative noise
    # For dtheta, the process model might be: dtheta(k+1) = dtheta(k) (constant velocity)
    dtheta_prediction_errors = np.diff(dtheta)
    Q_dtheta = np.var(dtheta_prediction_errors)

    # === Create analysis plot ===
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Theta error with stationary periods highlighted
    axes[0].plot(timestamps_sec, np.degrees(theta), 'b-', alpha=0.7, label='Theta Error')
    stationary_times = timestamps_sec[stationary_mask]
    stationary_theta = theta[stationary_mask]
    axes[0].scatter(stationary_times, np.degrees(stationary_theta),
                    c='green', s=10, alpha=0.5, label=f'Stationary (|dθ| < {np.degrees(velocity_threshold):.2f}°/s)')
    axes[0].set_ylabel('Theta Error (deg)')
    axes[0].set_title(f'Measurement Noise Estimation: R = {R_theta:.2e} rad² ({np.degrees(np.sqrt(R_theta)):.3f}° std)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prediction errors
    axes[1].plot(timestamps_sec[1:], np.degrees(prediction_errors), 'r-', alpha=0.7)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].axhline(y=np.degrees(np.sqrt(Q_theta)), color='g', linestyle='--', alpha=0.5, label=f'±1σ = {np.degrees(np.sqrt(Q_theta)):.3f}°')
    axes[1].axhline(y=-np.degrees(np.sqrt(Q_theta)), color='g', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Prediction Error (deg)')
    axes[1].set_title(f'Process Noise Estimation: Q_theta = {Q_theta:.2e} rad² ({np.degrees(np.sqrt(Q_theta)):.3f}° std)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Derivative prediction errors (for Q_dtheta)
    axes[2].plot(timestamps_sec[1:], np.degrees(dtheta_prediction_errors), 'purple', alpha=0.7)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('dTheta Change (deg/s)')
    axes[2].set_title(f'Derivative Process Noise: Q_dtheta = {Q_dtheta:.2e} (rad/s)² ({np.degrees(np.sqrt(Q_dtheta)):.3f}°/s std)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = csv_path.parent / f'{csv_path.stem}_kalman_noise.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()

    # Write results to text file
    txt_filename = csv_path.parent / f'{csv_path.stem}_kalman_noise.txt'
    with open(txt_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("KALMAN FILTER NOISE PARAMETER ESTIMATES\n")
        f.write("="*60 + "\n")
        if is_stationary_dataset:
            f.write("Dataset type: STATIONARY (controller off)\n")
            f.write("  -> R estimate is VALID (ideal conditions)\n")
            f.write("  -> Q estimates are NOT VALID (no real dynamics)\n")
        else:
            f.write("Dataset type: MOTION (controller on)\n")
            f.write("  -> Both R and Q estimates are valid\n")
        f.write(f"\nMeasurement Noise (R):\n")
        f.write(f"  R_theta = {R_theta:.6e} rad²  ({np.degrees(np.sqrt(R_theta)):.4f}° std)\n")
        f.write(f"\nProcess Noise (Q):" + (" [MAY BE INVALID]\n" if is_stationary_dataset else "\n"))
        f.write(f"  Q_theta  = {Q_theta:.6e} rad²  ({np.degrees(np.sqrt(Q_theta)):.4f}° std)\n")
        f.write(f"  Q_dtheta = {Q_dtheta:.6e} (rad/s)²  ({np.degrees(np.sqrt(Q_dtheta)):.4f}°/s std)\n")
        f.write(f"\nFor 2D state [theta, dtheta], suggested Q matrix:\n")
        f.write(f"  Q = diag([{Q_theta:.6e}, {Q_dtheta:.6e}])\n")
        f.write("="*60 + "\n")

    print(f"Kalman noise parameters saved to {txt_filename}")

    return {
        'R_theta': R_theta,
        'Q_theta': Q_theta,
        'Q_dtheta': Q_dtheta,
        'is_stationary_dataset': is_stationary_dataset,
        'velocity_threshold': velocity_threshold,
        'n_stationary_samples': np.sum(stationary_mask),
    }


def pointcloud2_to_array(cloud_msg):
    """
    Convert a ROS2 PointCloud2 message to numpy arrays of points and colors.
    Returns (points, colors) where colors may be None if not present.
    """
    # Read points from the cloud
    points_list = []
    colors_list = []

    # Get field names
    field_names = [f.name for f in cloud_msg.fields]
    has_rgb = 'rgb' in field_names or 'rgba' in field_names

    for point in point_cloud2.read_points(cloud_msg, skip_nans=True):
        points_list.append([point['x'], point['y'], point['z']])

        if has_rgb:
            # RGB is packed as a float, need to unpack
            if 'rgb' in field_names:
                rgb = point['rgb']
            else:
                rgb = point['rgba']

            # Unpack RGB from float
            if isinstance(rgb, float):
                rgb_int = int(rgb)
            else:
                rgb_int = rgb
            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF
            colors_list.append([r / 255.0, g / 255.0, b / 255.0])

    points = np.array(points_list, dtype=np.float64)
    colors = np.array(colors_list, dtype=np.float64) if colors_list else None

    return points, colors


def create_time_gradient_color(t_normalized, cmap_name='viridis'):
    """
    Create a color based on normalized time [0, 1] using a colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    return np.array(cmap(t_normalized)[:3])  # RGB, drop alpha


def visualize_pointclouds_over_time(point_clouds, normals, centroids, timestamps, output_path):
    """
    Create an Open3D visualization with all point clouds colored by time gradient.
    Also shows normals at each centroid.

    Args:
        point_clouds: List of numpy arrays (N, 3) for each timestep
        normals: List of numpy arrays (3,) surface normals
        centroids: List of numpy arrays (3,) surface centroids
        timestamps: List of timestamps
        output_path: Path to save the visualization
    """
    if not point_clouds:
        print("No point clouds to visualize")
        return

    # Normalize timestamps to [0, 1]
    t_min = min(timestamps)
    t_max = max(timestamps)
    t_range = t_max - t_min if t_max > t_min else 1.0

    # Compute average centroid for camera positioning
    valid_centroids = [c for c in centroids if c is not None]
    if valid_centroids:
        avg_centroid = np.mean(valid_centroids, axis=0)
    else:
        avg_centroid = np.zeros(3)

    # Combined geometry for visualization
    combined_pcd = o3d.geometry.PointCloud()
    geometries = []

    # Process each point cloud
    for points, normal, centroid, t in zip(point_clouds, normals, centroids, timestamps):
        if len(points) == 0:
            continue

        t_normalized = (t - t_min) / t_range
        color = create_time_gradient_color(t_normalized)

        # Create point cloud with time-based color
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))

        # Add to combined
        combined_pcd += pcd

        # Create normal arrow at centroid (flip normal sign)
        if centroid is not None and normal is not None:
            flipped_normal = -normal  # Flip the normal vector
            arrow_length = 0.05  # 5cm arrow
            line_points = [centroid, centroid + flipped_normal * arrow_length]
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(line_points)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([color])
            geometries.append(line)

    # Clean up the combined point cloud
    if combined_pcd.has_points():
        print(f"Raw combined cloud: {len(combined_pcd.points)} points")

        # Voxel downsample to create uniform point density
        voxel_size = 0.002  # 2mm voxel size
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
        print(f"After voxel downsampling ({voxel_size*1000:.1f}mm): {len(combined_pcd.points)} points")

        # Remove statistical outliers
        if len(combined_pcd.points) > 20:
            combined_pcd, inlier_indices = combined_pcd.remove_statistical_outlier(
                nb_neighbors=20,  # Number of neighbors to consider
                std_ratio=2.0     # Standard deviation threshold
            )
            print(f"After statistical outlier removal: {len(combined_pcd.points)} points")

    geometries.insert(0, combined_pcd)

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(coord_frame)

    # Save to file
    ply_path = output_path.parent / f'{output_path.stem}_pointclouds.ply'
    o3d.io.write_point_cloud(str(ply_path), combined_pcd)
    print(f"Combined point cloud saved to {ply_path}")

    # Create visualization and save screenshot
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)

    for geom in geometries:
        vis.add_geometry(geom)

    # Set camera view: position at (-d/2, -d/2, -d/2) looking at average centroid
    ctr = vis.get_view_control()

    # Compute d as average distance from origin to centroids
    if valid_centroids:
        distances = [np.linalg.norm(c) for c in valid_centroids]
        d = np.mean(distances)
    else:
        d = 1.0

    # Camera position offset from average centroid
    camera_offset = np.array([-d / 2, -d / 2, -d / 2])
    camera_pos = avg_centroid + camera_offset

    # Direction from camera to target (average centroid)
    front = avg_centroid - camera_pos
    front = front / (np.linalg.norm(front) + 1e-12)

    # Set view parameters
    ctr.set_lookat(avg_centroid)
    ctr.set_front(-front)  # Open3D front is opposite of look direction
    ctr.set_up([0, 0, -1])  # Z-up, but inverted for this view
    ctr.set_zoom(0.5)

    vis.poll_events()
    vis.update_renderer()

    # Save screenshot
    screenshot_path = output_path.parent / f'{output_path.stem}_pointclouds.png'
    vis.capture_screen_image(str(screenshot_path))
    vis.destroy_window()
    print(f"Point cloud visualization saved to {screenshot_path}")

    return combined_pcd


def debag(bag_file):
    bag_file = pathlib.Path(bag_file)
    output_dir = bag_file.with_suffix('')
    output_dir.mkdir(exist_ok=True)

    # Output file base path (in output directory, same stem as bag)
    output_base = output_dir / bag_file.stem

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_file)),
                rosbag2_py.ConverterOptions())

    if not reader.has_next():
        print(f"No messages in bag file {bag_file}")
        return

    # Storage for point cloud visualization
    all_point_clouds = []
    all_normals = []
    all_centroids = []
    all_timestamps = []

    # Initialize CSV writer
    csv_filename = output_base.with_suffix('.csv')
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # CSV Header matching new message structure
    csv_writer.writerow([
        'timestamp',
        'normal_method',
        # Surface geometry
        'centroid_x',
        'centroid_y',
        'centroid_z',
        'normal_x',
        'normal_y',
        'normal_z',
        # Current pose
        'current_pose_x',
        'current_pose_y',
        'current_pose_z',
        'current_pose_qx',
        'current_pose_qy',
        'current_pose_qz',
        'current_pose_qw',
        # Current twist
        'current_twist_linear_x',
        'current_twist_linear_y',
        'current_twist_linear_z',
        'current_twist_angular_x',
        'current_twist_angular_y',
        'current_twist_angular_z',
        # Goal pose
        'goal_pose_x',
        'goal_pose_y',
        'goal_pose_z',
        'goal_pose_qx',
        'goal_pose_qy',
        'goal_pose_qz',
        'goal_pose_qw',
        # Theta control
        'theta_error',
        'theta_error_filtered',
        'dtheta_error',
        'itheta_error',
        'theta_axis_x',
        'theta_axis_y',
        'theta_axis_z',
        'theta_torque_command',
        # Roll control
        'roll_error',
        'droll_error',
        'iroll_error',
        'roll_torque_command',
        # PID gains
        'k_p_theta',
        'k_d_theta',
        'k_i_theta',
        'k_p_roll',
        'k_d_roll',
        'k_i_roll',
    ])

    msg_count = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg = deserialize_message(data, OrientationControlData)
        msg_count += 1

        # Extract point cloud for visualization (transform to object_frame)
        try:
            points, _ = pointcloud2_to_array(msg.point_cloud)
            if len(points) > 0:
                # Transform point cloud from camera_frame to object_frame
                points_obj = transform_points(points, msg.camera_transform)
                all_point_clouds.append(points_obj)

                # Transform normal (rotation only)
                normal_cam = np.array([
                    msg.surface_normal.x,
                    msg.surface_normal.y,
                    msg.surface_normal.z
                ])
                normal_obj = transform_vector(normal_cam, msg.camera_transform)
                all_normals.append(normal_obj)

                # Transform centroid
                centroid_cam = np.array([
                    msg.surface_centroid.x,
                    msg.surface_centroid.y,
                    msg.surface_centroid.z
                ])
                centroid_obj = transform_points(centroid_cam.reshape(1, 3), msg.camera_transform).flatten()
                all_centroids.append(centroid_obj)

                all_timestamps.append(t)
        except Exception as e:
            print(f"Error extracting point cloud at msg {msg_count}: {e}")

        # Write data to CSV
        csv_writer.writerow([
            t,
            msg.normal_method,
            # Surface geometry
            msg.surface_centroid.x,
            msg.surface_centroid.y,
            msg.surface_centroid.z,
            msg.surface_normal.x,
            msg.surface_normal.y,
            msg.surface_normal.z,
            # Current pose
            msg.current_pose.pose.position.x,
            msg.current_pose.pose.position.y,
            msg.current_pose.pose.position.z,
            msg.current_pose.pose.orientation.x,
            msg.current_pose.pose.orientation.y,
            msg.current_pose.pose.orientation.z,
            msg.current_pose.pose.orientation.w,
            # Current twist
            msg.current_twist.twist.linear.x,
            msg.current_twist.twist.linear.y,
            msg.current_twist.twist.linear.z,
            msg.current_twist.twist.angular.x,
            msg.current_twist.twist.angular.y,
            msg.current_twist.twist.angular.z,
            # Goal pose
            msg.goal_pose.pose.position.x,
            msg.goal_pose.pose.position.y,
            msg.goal_pose.pose.position.z,
            msg.goal_pose.pose.orientation.x,
            msg.goal_pose.pose.orientation.y,
            msg.goal_pose.pose.orientation.z,
            msg.goal_pose.pose.orientation.w,
            # Theta control
            msg.theta_error,
            msg.theta_error_filtered,
            msg.dtheta_error,
            msg.itheta_error,
            msg.theta_axis.x,
            msg.theta_axis.y,
            msg.theta_axis.z,
            msg.theta_torque_command,
            # Roll control
            msg.roll_error,
            msg.droll_error,
            msg.iroll_error,
            msg.roll_torque_command,
            # PID gains
            msg.k_p_theta,
            msg.k_d_theta,
            msg.k_i_theta,
            msg.k_p_roll,
            msg.k_d_roll,
            msg.k_i_roll,
        ])

    csv_file.close()
    print(f"CSV saved to {csv_filename} ({msg_count} messages)")

    # Generate the theta and roll control plots
    generate_theta_control_plots(csv_filename)
    generate_roll_control_plots(csv_filename)

    # Estimate Kalman filter noise parameters
    estimate_kalman_noise_parameters(csv_filename)

    # Generate Open3D point cloud visualization
    print(f"Creating point cloud visualization from {len(all_point_clouds)} clouds...")
    visualize_pointclouds_over_time(
        all_point_clouds,
        all_normals,
        all_centroids,
        all_timestamps,
        output_base
    )

    # Delete the original bag file (directory) to save space
    if bag_file.exists():
        shutil.rmtree(bag_file)
        print(f"Deleted bag file: {bag_file}")
