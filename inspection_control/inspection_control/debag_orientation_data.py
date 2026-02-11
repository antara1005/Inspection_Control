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
from scipy.linalg import expm
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


# =========================
#  Geometry / TF Helpers
# =========================

def quat_to_rotation_matrix(q):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
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

    return (R @ points.T).T + t


def transform_vector(vec, transform_msg):
    """Transform a vector (rotation only, no translation) using a TransformStamped message."""
    R = quat_to_rotation_matrix(transform_msg.transform.rotation)
    return R @ vec


# =========================
#  PointCloud Helpers
# =========================

def pointcloud2_to_array(cloud_msg):
    """
    Convert a ROS2 PointCloud2 message to numpy arrays of points and colors.
    Returns (points, colors) where colors may be None if not present.
    """
    points_list = []
    colors_list = []

    field_names = [f.name for f in cloud_msg.fields]
    has_rgb = 'rgb' in field_names or 'rgba' in field_names

    for point in point_cloud2.read_points(cloud_msg, skip_nans=True):
        points_list.append([point['x'], point['y'], point['z']])

        if has_rgb:
            rgb = point['rgb'] if 'rgb' in field_names else point['rgba']
            rgb_int = int(rgb) if isinstance(rgb, float) else rgb
            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF
            colors_list.append([r / 255.0, g / 255.0, b / 255.0])

    points = np.array(points_list, dtype=np.float64)
    colors = np.array(colors_list, dtype=np.float64) if colors_list else None
    return points, colors


def create_time_gradient_color(t_normalized, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    return np.array(cmap(t_normalized)[:3])


def visualize_pointclouds_over_time(point_clouds, normals, centroids, torque_vectors, timestamps, output_path):
    """
    Create an Open3D visualization with all point clouds colored by time gradient.
    Also shows normals and torque vectors at each centroid.
    """
    if not point_clouds:
        print("No point clouds to visualize")
        return

    t_min = min(timestamps)
    t_max = max(timestamps)
    t_range = t_max - t_min if t_max > t_min else 1.0

    valid_centroids = [c for c in centroids if c is not None]
    avg_centroid = np.mean(valid_centroids, axis=0) if valid_centroids else np.zeros(3)

    combined_pcd = o3d.geometry.PointCloud()
    geometries = []
    torque_lines = []

    for points, normal, centroid, torque_vec, t in zip(point_clouds, normals, centroids, torque_vectors, timestamps):
        if len(points) == 0:
            continue

        t_normalized = (t - t_min) / t_range
        color = create_time_gradient_color(t_normalized)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
        combined_pcd += pcd

        if centroid is not None and normal is not None:
            flipped_normal = -normal
            arrow_length = 0.05
            line_points = [centroid, centroid + flipped_normal * arrow_length]
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(line_points)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([color])
            geometries.append(line)

        if centroid is not None and torque_vec is not None:
            torque_norm = np.linalg.norm(torque_vec)
            if torque_norm > 1e-6:
                torque_dir = torque_vec / torque_norm
                arrow_length = 0.03
                line_points = [centroid, centroid + torque_dir * arrow_length]
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector(line_points)
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0]])
                torque_lines.append(line)

    if combined_pcd.has_points():
        print(f"Raw combined cloud: {len(combined_pcd.points)} points")
        voxel_size = 0.002
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
        print(f"After voxel downsampling ({voxel_size*1000:.1f}mm): {len(combined_pcd.points)} points")

        if len(combined_pcd.points) > 20:
            combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            print(f"After statistical outlier removal: {len(combined_pcd.points)} points")

    geometries.insert(0, combined_pcd)
    geometries.extend(torque_lines)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(coord_frame)

    ply_path = output_path.parent / f'{output_path.stem}_pointclouds.ply'
    o3d.io.write_point_cloud(str(ply_path), combined_pcd)
    print(f"Combined point cloud saved to {ply_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    for geom in geometries:
        vis.add_geometry(geom)

    ctr = vis.get_view_control()

    if valid_centroids:
        distances = [np.linalg.norm(c) for c in valid_centroids]
        d = np.mean(distances)
    else:
        d = 1.0

    camera_offset = np.array([-d / 2, -d / 2, -d / 2])
    camera_pos = avg_centroid + camera_offset
    front = avg_centroid - camera_pos
    front = front / (np.linalg.norm(front) + 1e-12)

    ctr.set_lookat(avg_centroid)
    ctr.set_front(-front)
    ctr.set_up([0, 0, -1])
    ctr.set_zoom(0.5)

    vis.poll_events()
    vis.update_renderer()

    screenshot_path = output_path.parent / f'{output_path.stem}_pointclouds.png'
    vis.capture_screen_image(str(screenshot_path))
    vis.destroy_window()
    print(f"Point cloud visualization saved to {screenshot_path}")

    return combined_pcd


# =========================
#  Plotting Helpers
# =========================

def generate_pitch_yaw_control_plots(csv_filepath):
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    #has_raw = 'pitch_error_raw' in df.columns

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    axes[0].plot(timestamps_sec, np.degrees(df['pitch_error']), label='Pitch Filtered', linewidth=1.5, c='b')
    #if has_raw:
    axes[0].plot(timestamps_sec, np.degrees(df['pitch_error_raw']), label='Pitch Raw', linewidth=1.0, linestyle='--', alpha=0.5)

    axes[0].plot(timestamps_sec, np.degrees(df['yaw_error']), label='Yaw Filtered', linewidth=1.5, c='g')
    #if has_raw:
    axes[0].plot(timestamps_sec, np.degrees(df['yaw_error_raw']), label='Yaw Raw', linewidth=1.0, linestyle='--', alpha=0.5)

    axes[0].set_ylabel('Error (degrees)')
    axes[0].set_title('Pitch and Yaw Errors: Filtered (solid) vs Raw (dashed)')
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax_deriv = axes[1]
    ax_integ = ax_deriv.twinx()
    line1, = ax_deriv.plot(timestamps_sec, np.degrees(df['dpitch_error']), label='d(pitch)/dt Filtered', linewidth=1.5, c='b')
    #if has_raw:
    line1_raw, = ax_deriv.plot(timestamps_sec, np.degrees(df['dpitch_error_raw']), label='d(pitch)/dt Raw', linewidth=1.0, linestyle='--', alpha=0.5)
    line2, = ax_integ.plot(timestamps_sec, np.degrees(df['ipitch_error']), label='∫pitch dt', linewidth=1.5, c='m')

    ax_deriv.set_ylabel('Derivative (deg/s)')
    ax_integ.set_ylabel('Integral (deg·s)')
    ax_deriv.set_title('Pitch: Derivative (Filtered vs Raw) and Integral')
    ax_deriv.legend(handles=[line1, line1_raw, line2])# if has_raw else [line1, line2], loc='upper right')
    ax_deriv.grid(True, alpha=0.3)

    ax_deriv2 = axes[2]
    ax_integ2 = ax_deriv2.twinx()
    line3, = ax_deriv2.plot(timestamps_sec, np.degrees(df['dyaw_error']), label='d(yaw)/dt Filtered', linewidth=1.5, c='g')
    #if has_raw:
    line3_raw, = ax_deriv2.plot(timestamps_sec, np.degrees(df['dyaw_error_raw']), label='d(yaw)/dt Raw', linewidth=1.0, linestyle='--', alpha=0.5)
    line4, = ax_integ2.plot(timestamps_sec, np.degrees(df['iyaw_error']), label='∫yaw dt', linewidth=1.5, c='y')

    ax_deriv2.set_ylabel('Derivative (deg/s)')
    ax_integ2.set_ylabel('Integral (deg·s)')
    ax_deriv2.set_title('Yaw: Derivative (Filtered vs Raw) and Integral')
    ax_deriv2.legend(handles=[line3, line3_raw, line4])# if has_raw else [line3, line4], loc='upper right')
    ax_deriv2.grid(True, alpha=0.3)

    axes[3].plot(timestamps_sec, df['pitch_torque_command'], label='Pitch Torque (X)', linewidth=1.5)
    axes[3].plot(timestamps_sec, df['yaw_torque_command'], label='Yaw Torque (Y)', linewidth=1.5)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Torque (Nm)')
    axes[3].set_title('Pitch and Yaw Torque Commands')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plot_filename = csv_path.parent / f'{csv_path.stem}_pitch_yaw_control.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Pitch/Yaw control plot saved to {plot_filename}")


def generate_roll_control_plots(csv_filepath):
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(timestamps_sec, np.degrees(df['roll_error']), label='Roll Error', linewidth=1.5)
    axes[0].set_ylabel('Error (degrees)')
    axes[0].set_title('Roll Error Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    ax_deriv = axes[1]
    ax_integ = ax_deriv.twinx()
    line1, = ax_deriv.plot(timestamps_sec, np.degrees(df['droll_error']), label='Derivative')
    line2, = ax_integ.plot(timestamps_sec, np.degrees(df['iroll_error']), label='Integral')

    ax_deriv.set_ylabel('Derivative (deg/s)')
    ax_integ.set_ylabel('Integral (deg·s)')
    ax_deriv.set_title('Derivative and Integral of Roll Error')
    ax_deriv.legend(handles=[line1, line2], loc='upper right')
    ax_deriv.grid(True, alpha=0.3)

    axes[2].plot(timestamps_sec, df['roll_torque_command'], label='Roll Torque', linewidth=1.5)
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


# =========================
#  Plant Discretization
# =========================

def discretize_plant(dt: float, I_A: float, b: float):
    A = np.array([[0.0, 1.0],
                  [0.0, -b / max(1e-12, I_A)]], dtype=np.float64)
    B = np.array([[0.0],
                  [1.0 / max(1e-12, I_A)]], dtype=np.float64)

    M = np.zeros((3, 3), dtype=np.float64)
    M[0:2, 0:2] = A
    M[0:2, 2:3] = B

    Md = expm(M * float(dt))
    Ad = Md[0:2, 0:2]
    Bd = Md[0:2, 2:3]
    return Ad, Bd


# =========================
#  Option B (Smoothed derivative)
# =========================

def _odd_at_least(x: int, lo: int = 5) -> int:
    x = max(lo, int(x))
    return x if (x % 2 == 1) else (x + 1)

def _sgolay_derivative(y: np.ndarray, t: np.ndarray,
                       window_s: float = 0.35,
                       polyorder: int = 3) -> np.ndarray:
    """
    Option B:
      - use RAW angle
      - compute d(angle)/dt via Savitzky–Golay derivative (smoothed differentiator)
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        delta = 1.0 / 30.0
        fs = 30.0
    else:
        delta = float(np.median(dt))
        fs = 1.0 / delta

    win = _odd_at_least(int(round(window_s * fs)), lo=5)
    win = min(win, len(y) if len(y) % 2 == 1 else max(5, len(y) - 1))
    if win <= polyorder:
        polyorder = max(2, win - 2)

    dy = savgol_filter(y, window_length=win, polyorder=polyorder, deriv=1, delta=delta, mode="interp")
    return dy.astype(np.float64)


# =========================
#  Kalman Noise Estimation (UPDATED with Option B)
# =========================

def estimate_kalman_noise_parameters(csv_filepath,
                                     sphere_mass: float = 2.5,
                                     sphere_radius: float = 0.65,
                                     fluid_viscosity: float = 1.0,
                                     optionB_window_s: float = 0.35,
                                     optionB_polyorder: int = 3):
    """
    UPDATED:
      - R uses RAW angle (pitch_error_raw / yaw_error_raw) when available
      - Q uses plant prediction errors with state:
            x_k = [angle_raw, dangle_smooth]^T
        where dangle_smooth is Savitzky–Golay derivative of the RAW angle (Option B)

    NOTE:
      We do NOT use dpitch_error_raw/dyaw_error_raw even if logged, because those are typically
      noisy finite differences and inflate Q_dangle. Option B fixes that.
    """
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)

    # Time base
    timestamps_sec = (df['timestamp'].astype(np.float64) - df['timestamp'].iloc[0]) / 1e9
    timestamps_sec = np.asarray(timestamps_sec, dtype=np.float64)

    # Use RAW angles if present; fallback to filtered
   # has_raw = ('pitch_error_raw' in df.columns) and ('yaw_error_raw' in df.columns)
    #if has_raw:
    pitch = df['pitch_error_raw'].to_numpy(dtype=np.float64)
    yaw   = df['yaw_error_raw'].to_numpy(dtype=np.float64)
    #else:
      #  print("[WARN] Raw columns not found. Falling back to filtered pitch_error/yaw_error.")
      #  pitch = df['pitch_error'].to_numpy(dtype=np.float64)
      #  yaw   = df['yaw_error'].to_numpy(dtype=np.float64)

    # Option B smoothed derivatives from the chosen angle signals
    dpitch = _sgolay_derivative(pitch, timestamps_sec, window_s=optionB_window_s, polyorder=optionB_polyorder)
    dyaw   = _sgolay_derivative(yaw,   timestamps_sec, window_s=optionB_window_s, polyorder=optionB_polyorder)

    # Torques
    tau_pitch = df['pitch_torque_command'].to_numpy(dtype=np.float64)
    tau_yaw   = df['yaw_torque_command'].to_numpy(dtype=np.float64)

    # Focal distance d = ||centroid||
    centroid_x = df['centroid_x'].to_numpy(dtype=np.float64)
    centroid_y = df['centroid_y'].to_numpy(dtype=np.float64)
    centroid_z = df['centroid_z'].to_numpy(dtype=np.float64)
    d_array = np.sqrt(centroid_x**2 + centroid_y**2 + centroid_z**2).astype(np.float64)

    # Physical params
    inertia_B   = (2.0 / 5.0) * sphere_mass * (sphere_radius ** 2)
    linear_drag = 6.0 * np.pi * fluid_viscosity * sphere_radius

    def estimate_axis_noise_plant(angle, dangle, tau, t_sec, d_arr, axis_name):
        n = len(angle)

        # Stationary heuristic using smoothed derivative (more reliable)
        dangle_std = float(np.std(dangle))
        motion_threshold = 5  # rad/s
        is_stationary = dangle_std < motion_threshold

        if is_stationary:
            stationary_mask = np.ones(n, dtype=bool)
            velocity_threshold = np.inf
        else:
            velocity_threshold = float(np.percentile(np.abs(dangle), 25))
            stationary_mask = np.abs(dangle) < velocity_threshold

        # R from detrended angle during stationary
        if np.sum(stationary_mask) > 10:
            angle_stationary = angle[stationary_mask]
            angle_detrended = angle_stationary - pd.Series(angle_stationary).rolling(
                window=min(10, len(angle_stationary)//2), center=True, min_periods=1
            ).mean().values
            R = float(np.var(angle_detrended))
        else:
            R = float(np.var(np.diff(angle)) / 2.0)

        # Process noise via plant prediction errors
        angle_pred_errors = []
        dangle_pred_errors = []
        t = np.asarray(t_sec, dtype=np.float64)

        for k in range(n - 1):
            dt = float(t[k+1] - t[k])
            if not np.isfinite(dt) or dt <= 0:
                continue

            d = float(d_arr[k])
            I_A = float(inertia_B + sphere_mass * d**2)
            b = float(linear_drag * d * d)

            Ad, Bd = discretize_plant(dt, I_A, b)

            x_k = np.array([angle[k], dangle[k]], dtype=np.float64)
            u_k = float(tau[k])

            x_pred = Ad @ x_k + Bd.flatten() * u_k
            x_actual = np.array([angle[k+1], dangle[k+1]], dtype=np.float64)

            angle_pred_errors.append(x_actual[0] - x_pred[0])
            dangle_pred_errors.append(x_actual[1] - x_pred[1])

        angle_pred_errors = np.asarray(angle_pred_errors, dtype=np.float64)
        dangle_pred_errors = np.asarray(dangle_pred_errors, dtype=np.float64)

        Q_angle  = float(np.var(angle_pred_errors))  if angle_pred_errors.size  > 5 else 0.0
        Q_dangle = float(np.var(dangle_pred_errors)) if dangle_pred_errors.size > 5 else 0.0

        return {
            'R': R,
            'Q_angle': Q_angle,
            'Q_dangle': Q_dangle,
            'is_stationary': is_stationary,
            'stationary_mask': stationary_mask,
            'velocity_threshold': velocity_threshold,
            'prediction_errors': angle_pred_errors,
            'dangle_prediction_errors': dangle_pred_errors,
        }

    pitch_est = estimate_axis_noise_plant(pitch, dpitch, tau_pitch, timestamps_sec, d_array, 'pitch')
    yaw_est   = estimate_axis_noise_plant(yaw,   dyaw,   tau_yaw,   timestamps_sec, d_array, 'yaw')

    R_combined        = max(pitch_est['R'], yaw_est['R'])
    Q_angle_combined  = max(pitch_est['Q_angle'], yaw_est['Q_angle'])
    Q_dangle_combined = max(pitch_est['Q_dangle'], yaw_est['Q_dangle'])

    R_avg        = 0.5 * (pitch_est['R'] + yaw_est['R'])
    Q_angle_avg  = 0.5 * (pitch_est['Q_angle'] + yaw_est['Q_angle'])
    Q_dangle_avg = 0.5 * (pitch_est['Q_dangle'] + yaw_est['Q_dangle'])

    is_stationary_dataset = bool(pitch_est['is_stationary'] and yaw_est['is_stationary'])

    d_mean = float(np.mean(d_array))
    I_A_mean = float(inertia_B + sphere_mass * d_mean**2)
    b_mean = float(linear_drag * d_mean * d_mean)

    # Plot
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    for col, (angle, est, name) in enumerate([
        (pitch, pitch_est, 'Pitch'),
        (yaw,   yaw_est,   'Yaw')
    ]):
        axes[0, col].plot(timestamps_sec, np.degrees(angle), alpha=0.7, label=f'{name} Error (raw if avail)')
        stationary_times = timestamps_sec[est['stationary_mask']]
        stationary_angle = angle[est['stationary_mask']]
        axes[0, col].scatter(stationary_times, np.degrees(stationary_angle),
                             c='green', s=10, alpha=0.5, label='Stationary')
        axes[0, col].set_ylabel(f'{name} Error (deg)')
        axes[0, col].set_title(f'{name}: R = {est["R"]:.2e} rad² ({np.degrees(np.sqrt(est["R"])):.3f}° std)')
        axes[0, col].legend(loc='upper right')
        axes[0, col].grid(True, alpha=0.3)

    for col, (est, name) in enumerate([
        (pitch_est, 'Pitch'),
        (yaw_est,   'Yaw')
    ]):
        axes[1, col].plot(timestamps_sec[1:1+len(est['prediction_errors'])], np.degrees(est['prediction_errors']), alpha=0.7)
        axes[1, col].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        std_deg = np.degrees(np.sqrt(est['Q_angle'])) if est['Q_angle'] > 0 else 0.0
        axes[1, col].set_ylabel('Prediction Error (deg)')
        axes[1, col].set_title(f'{name}: Q_angle = {est["Q_angle"]:.2e} rad² ({std_deg:.3f}° std)')
        axes[1, col].grid(True, alpha=0.3)

    for col, (est, name) in enumerate([
        (pitch_est, 'Pitch'),
        (yaw_est,   'Yaw')
    ]):
        axes[2, col].plot(timestamps_sec[1:1+len(est['dangle_prediction_errors'])], np.degrees(est['dangle_prediction_errors']), alpha=0.7)
        axes[2, col].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        std_deg_s = np.degrees(np.sqrt(est['Q_dangle'])) if est['Q_dangle'] > 0 else 0.0
        axes[2, col].set_ylabel('d(angle)/dt Pred Error (deg/s)')
        axes[2, col].set_title(f'{name}: Q_dangle = {est["Q_dangle"]:.2e} ({std_deg_s:.3f}°/s std)')
        axes[2, col].grid(True, alpha=0.3)

    axes[3, 0].text(0.5, 0.5,
        f"OPTION B (RAW angle + SG smoothed derivative)\n\n"
        f"Measurement Noise:\n"
        f"  R = {R_combined:.6e} rad²\n"
        f"    ({np.degrees(np.sqrt(R_combined)):.4f}° std)\n\n"
        f"Process Noise:\n"
        f"  Q_angle  = {Q_angle_combined:.6e} rad²\n"
        f"  Q_dangle = {Q_dangle_combined:.6e} (rad/s)²\n\n"
        f"For state [θ, θ̇]:\n"
        f"  Q = diag([{Q_angle_combined:.2e}, {Q_dangle_combined:.2e}])",
        transform=axes[3, 0].transAxes, fontsize=10, va='center', ha='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[3, 0].axis('off')
    axes[3, 0].set_title('Combined Parameters (Recommended)')

    axes[3, 1].text(0.5, 0.5,
        f"SYSTEM PARAMETERS USED\n\n"
        f"Physical:\n"
        f"  mass = {sphere_mass:.2f} kg\n"
        f"  radius = {sphere_radius:.3f} m\n"
        f"  viscosity = {fluid_viscosity:.6f} Pa·s\n\n"
        f"Derived (avg d={d_mean:.3f}m):\n"
        f"  I_B = {inertia_B:.4f} kg·m²\n"
        f"  I_A = {I_A_mean:.4f} kg·m²\n"
        f"  b   = {b_mean:.6f} N·m·s/rad\n"
        f"  linear_drag = {linear_drag:.6f} N·s/m\n\n"
        f"Individual R estimates:\n"
        f"  Pitch: {pitch_est['R']:.2e}\n"
        f"  Yaw:   {yaw_est['R']:.2e}",
        transform=axes[3, 1].transAxes, fontsize=10, va='center', ha='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[3, 1].axis('off')
    axes[3, 1].set_title('System Parameters')

    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
    plt.tight_layout()

    plot_filename = csv_path.parent / f'{csv_path.stem}_kalman_noise.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()

    # Text output
    txt_filename = csv_path.parent / f'{csv_path.stem}_kalman_noise.txt'
    with open(txt_filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("KALMAN FILTER NOISE ESTIMATES (Plant Dynamics Model)\n")
        f.write("UPDATED: Option B (RAW angle + Savitzky–Golay smoothed derivative)\n")
        f.write("="*70 + "\n\n")

        f.write("Plant Model:\n")
        f.write("  θ̈ = -(b/I_A)·θ̇ + (1/I_A)·τ\n")
        f.write("  x_{k+1} = Ad·x_k + Bd·u_k + w_k\n")
        f.write("  z_k = H·x_k + v_k,  H = [1, 0]\n\n")

        f.write("-"*70 + "\n")
        f.write("OPTION B SETTINGS\n")
        f.write("-"*70 + "\n")
        f.write(f"  SG window_s  = {optionB_window_s:.3f} s\n")
        f.write(f"  SG polyorder = {optionB_polyorder}\n")
        #f.write(f"  Using RAW angle columns? {has_raw}\n\n")

        f.write("-"*70 + "\n")
        f.write("SYSTEM PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"  sphere_mass     = {sphere_mass:.4f} kg\n")
        f.write(f"  sphere_radius   = {sphere_radius:.4f} m\n")
        f.write(f"  fluid_viscosity = {fluid_viscosity:.8f} Pa·s\n")
        f.write(f"  inertia_B       = {inertia_B:.6f} kg·m²\n")
        f.write(f"  linear_drag     = {linear_drag:.8f} N·s/m\n")
        f.write(f"  avg focal dist  = {d_mean:.4f} m\n")
        f.write(f"  avg I_A         = {I_A_mean:.6f} kg·m²\n")
        f.write(f"  avg b           = {b_mean:.8f} N·m·s/rad\n\n")

        f.write("-"*70 + "\n")
        f.write("RECOMMENDED PARAMETERS (max of pitch/yaw for robustness)\n")
        f.write("-"*70 + "\n")
        f.write(f"Measurement Noise (R):\n")
        f.write(f"  R = {R_combined:.6e} rad²  ({np.degrees(np.sqrt(R_combined)):.4f}° std)\n\n")
        f.write(f"Process Noise (Q):\n")
        f.write(f"  Q_angle  = {Q_angle_combined:.6e} rad²  ({np.degrees(np.sqrt(Q_angle_combined)):.4f}° std)\n")
        f.write(f"  Q_dangle = {Q_dangle_combined:.6e} (rad/s)²  ({np.degrees(np.sqrt(Q_dangle_combined)):.4f}°/s std)\n\n")
        f.write(f"For 2D Kalman state [angle, dangle]:\n")
        f.write(f"  Q = diag([{Q_angle_combined:.6e}, {Q_dangle_combined:.6e}])\n\n")

        f.write("-"*70 + "\n")
        f.write("INDIVIDUAL AXIS ESTIMATES\n")
        f.write("-"*70 + "\n")
        f.write(f"Pitch:\n")
        f.write(f"  R = {pitch_est['R']:.6e} rad²\n")
        f.write(f"  Q_angle  = {pitch_est['Q_angle']:.6e} rad²\n")
        f.write(f"  Q_dangle = {pitch_est['Q_dangle']:.6e} (rad/s)²\n")
        f.write(f"Yaw:\n")
        f.write(f"  R = {yaw_est['R']:.6e} rad²\n")
        f.write(f"  Q_angle  = {yaw_est['Q_angle']:.6e} rad²\n")
        f.write(f"  Q_dangle = {yaw_est['Q_dangle']:.6e} (rad/s)²\n")
        f.write("="*70 + "\n")

    print(f"Kalman noise parameters saved to {txt_filename}")

    return {
        'R': R_combined,
        'Q_angle': Q_angle_combined,
        'Q_dangle': Q_dangle_combined,
        'R_avg': R_avg,
        'Q_angle_avg': Q_angle_avg,
        'Q_dangle_avg': Q_dangle_avg,
        'pitch_estimates': pitch_est,
        'yaw_estimates': yaw_est,
        'is_stationary_dataset': is_stationary_dataset,
        'system_params': {
            'sphere_mass': sphere_mass,
            'sphere_radius': sphere_radius,
            'fluid_viscosity': fluid_viscosity,
            'inertia_B': inertia_B,
            'linear_drag': linear_drag,
            'd_mean': d_mean,
            'I_A_mean': I_A_mean,
            'b_mean': b_mean,
        }
    }


# =========================
#  Debag (unchanged except calls the updated estimator)
# =========================

def debag(bag_file,
          sphere_mass: float = 2.5,
          sphere_radius: float = 0.65,
          fluid_viscosity: float = 1.0):
    bag_file = pathlib.Path(bag_file)
    output_dir = bag_file.with_suffix('')
    output_dir.mkdir(exist_ok=True)

    output_base = output_dir / bag_file.stem

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_file)),
                rosbag2_py.ConverterOptions())

    if not reader.has_next():
        print(f"No messages in bag file {bag_file}")
        return

    all_point_clouds = []
    all_normals = []
    all_centroids = []
    all_torque_vectors = []
    all_timestamps = []

    csv_filename = output_base.with_suffix('.csv')
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        'timestamp',
        'normal_method',
        'centroid_x', 'centroid_y', 'centroid_z',
        'normal_x', 'normal_y', 'normal_z',
        'current_pose_x', 'current_pose_y', 'current_pose_z',
        'current_pose_qx', 'current_pose_qy', 'current_pose_qz', 'current_pose_qw',
        'current_twist_linear_x', 'current_twist_linear_y', 'current_twist_linear_z',
        'current_twist_angular_x', 'current_twist_angular_y', 'current_twist_angular_z',
        'goal_pose_x', 'goal_pose_y', 'goal_pose_z',
        'goal_pose_qx', 'goal_pose_qy', 'goal_pose_qz', 'goal_pose_qw',
        'pitch_error', 'dpitch_error', 'ipitch_error',
        'pitch_error_raw', 'dpitch_error_raw', 'pitch_torque_command',
        'yaw_error', 'dyaw_error', 'iyaw_error',
        'yaw_error_raw', 'dyaw_error_raw', 'yaw_torque_command',
        'roll_error', 'droll_error', 'iroll_error', 'roll_torque_command',
        'k_p', 'k_d', 'k_i',
    ])

    msg_count = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg = deserialize_message(data, OrientationControlData)
        msg_count += 1

        # Visualization extraction
        try:
            points, _ = pointcloud2_to_array(msg.point_cloud)
            if len(points) > 0:
                points_obj = transform_points(points, msg.camera_transform)
                all_point_clouds.append(points_obj)

                normal_cam = np.array([msg.surface_normal.x, msg.surface_normal.y, msg.surface_normal.z], dtype=np.float64)
                normal_obj = transform_vector(normal_cam, msg.camera_transform)
                all_normals.append(normal_obj)

                centroid_cam = np.array([msg.surface_centroid.x, msg.surface_centroid.y, msg.surface_centroid.z], dtype=np.float64)
                centroid_obj = transform_points(centroid_cam.reshape(1, 3), msg.camera_transform).flatten()
                all_centroids.append(centroid_obj)

                torque_cam = np.array([msg.pitch_torque_command, msg.yaw_torque_command, msg.roll_torque_command], dtype=np.float64)
                torque_obj = transform_vector(torque_cam, msg.camera_transform)
                all_torque_vectors.append(torque_obj)

                all_timestamps.append(t)
        except Exception as e:
            print(f"Error extracting point cloud at msg {msg_count}: {e}")

        # CSV write
        csv_writer.writerow([
            t,
            msg.normal_method,
            msg.surface_centroid.x, msg.surface_centroid.y, msg.surface_centroid.z,
            msg.surface_normal.x, msg.surface_normal.y, msg.surface_normal.z,
            msg.current_pose.pose.position.x, msg.current_pose.pose.position.y, msg.current_pose.pose.position.z,
            msg.current_pose.pose.orientation.x, msg.current_pose.pose.orientation.y,
            msg.current_pose.pose.orientation.z, msg.current_pose.pose.orientation.w,
            msg.current_twist.twist.linear.x, msg.current_twist.twist.linear.y, msg.current_twist.twist.linear.z,
            msg.current_twist.twist.angular.x, msg.current_twist.twist.angular.y, msg.current_twist.twist.angular.z,
            msg.goal_pose.pose.position.x, msg.goal_pose.pose.position.y, msg.goal_pose.pose.position.z,
            msg.goal_pose.pose.orientation.x, msg.goal_pose.pose.orientation.y,
            msg.goal_pose.pose.orientation.z, msg.goal_pose.pose.orientation.w,
            msg.pitch_error, msg.dpitch_error, msg.ipitch_error,
            msg.pitch_error_raw, msg.dpitch_error_raw, msg.pitch_torque_command,
            msg.yaw_error, msg.dyaw_error, msg.iyaw_error,
            msg.yaw_error_raw, msg.dyaw_error_raw, msg.yaw_torque_command,
            msg.roll_error, msg.droll_error, msg.iroll_error, msg.roll_torque_command,
            msg.k_p, msg.k_d, msg.k_i,
        ])

    csv_file.close()
    print(f"CSV saved to {csv_filename} ({msg_count} messages)")

    generate_pitch_yaw_control_plots(csv_filename)
    generate_roll_control_plots(csv_filename)

    # UPDATED call: Option B inside estimator
    estimate_kalman_noise_parameters(
        csv_filename,
        sphere_mass=sphere_mass,
        sphere_radius=sphere_radius,
        fluid_viscosity=fluid_viscosity,
        optionB_window_s=0.35,
        optionB_polyorder=3
    )

    print(f"Creating point cloud visualization from {len(all_point_clouds)} clouds...")
    visualize_pointclouds_over_time(
        all_point_clouds,
        all_normals,
        all_centroids,
        all_torque_vectors,
        all_timestamps,
        output_base
    )

    if bag_file.exists():
        shutil.rmtree(bag_file)
        print(f"Deleted bag file: {bag_file}") 
