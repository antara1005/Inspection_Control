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


def generate_pitch_yaw_control_plots(csv_filepath):
    """
    Generate plots showing pitch and yaw state and control:
    - Row 1: pitch and yaw errors (filtered vs raw)
    - Row 2: pitch derivative (filtered vs raw) and integral
    - Row 3: yaw derivative (filtered vs raw) and integral
    - Row 4: pitch and yaw torque commands
    """
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)

    # Convert timestamps to seconds relative to start
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    # Check if raw columns exist (for backwards compatibility)
    has_raw = 'pitch_error_raw' in df.columns

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # Row 1: pitch and yaw errors (filtered solid, raw dashed)
    # Pitch errors
    axes[0].plot(timestamps_sec, np.degrees(df['pitch_error']),
                 label='Pitch Filtered', color='blue', linewidth=1.5)
    if has_raw:
        axes[0].plot(timestamps_sec, np.degrees(df['pitch_error_raw']),
                     label='Pitch Raw', color='blue', linewidth=1.0, linestyle='--', alpha=0.5)
    # Yaw errors
    axes[0].plot(timestamps_sec, np.degrees(df['yaw_error']),
                 label='Yaw Filtered', color='red', linewidth=1.5)
    if has_raw:
        axes[0].plot(timestamps_sec, np.degrees(df['yaw_error_raw']),
                     label='Yaw Raw', color='red', linewidth=1.0, linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Error (degrees)')
    axes[0].set_title('Pitch and Yaw Errors: Filtered (solid) vs Raw (dashed)')
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Row 2: pitch derivative (filtered vs raw) and integral
    ax_deriv = axes[1]
    ax_integ = ax_deriv.twinx()

    line1, = ax_deriv.plot(timestamps_sec, np.degrees(df['dpitch_error']),
                           label='d(pitch)/dt Filtered', color='blue', linewidth=1.5)
    if has_raw:
        line1_raw, = ax_deriv.plot(timestamps_sec, np.degrees(df['dpitch_error_raw']),
                                   label='d(pitch)/dt Raw', color='blue', linewidth=1.0,
                                   linestyle='--', alpha=0.5)
    line2, = ax_integ.plot(timestamps_sec, np.degrees(df['ipitch_error']),
                           label='∫pitch dt', color='cyan', linewidth=1.5)

    ax_deriv.set_ylabel('Derivative (deg/s)', color='blue')
    ax_integ.set_ylabel('Integral (deg·s)', color='cyan')
    ax_deriv.tick_params(axis='y', labelcolor='blue')
    ax_integ.tick_params(axis='y', labelcolor='cyan')
    ax_deriv.set_title('Pitch: Derivative (Filtered vs Raw) and Integral')
    if has_raw:
        ax_deriv.legend(handles=[line1, line1_raw, line2], loc='upper right')
    else:
        ax_deriv.legend(handles=[line1, line2], loc='upper right')
    ax_deriv.grid(True, alpha=0.3)

    # Row 3: yaw derivative (filtered vs raw) and integral
    ax_deriv2 = axes[2]
    ax_integ2 = ax_deriv2.twinx()

    line3, = ax_deriv2.plot(timestamps_sec, np.degrees(df['dyaw_error']),
                            label='d(yaw)/dt Filtered', color='red', linewidth=1.5)
    if has_raw:
        line3_raw, = ax_deriv2.plot(timestamps_sec, np.degrees(df['dyaw_error_raw']),
                                    label='d(yaw)/dt Raw', color='red', linewidth=1.0,
                                    linestyle='--', alpha=0.5)
    line4, = ax_integ2.plot(timestamps_sec, np.degrees(df['iyaw_error']),
                            label='∫yaw dt', color='orange', linewidth=1.5)

    ax_deriv2.set_ylabel('Derivative (deg/s)', color='red')
    ax_integ2.set_ylabel('Integral (deg·s)', color='orange')
    ax_deriv2.tick_params(axis='y', labelcolor='red')
    ax_integ2.tick_params(axis='y', labelcolor='orange')
    ax_deriv2.set_title('Yaw: Derivative (Filtered vs Raw) and Integral')
    if has_raw:
        ax_deriv2.legend(handles=[line3, line3_raw, line4], loc='upper right')
    else:
        ax_deriv2.legend(handles=[line3, line4], loc='upper right')
    ax_deriv2.grid(True, alpha=0.3)

    # Row 4: pitch and yaw torque commands
    axes[3].plot(timestamps_sec, df['pitch_torque_command'],
                 label='Pitch Torque (X)', color='blue', linewidth=1.5)
    axes[3].plot(timestamps_sec, df['yaw_torque_command'],
                 label='Yaw Torque (Y)', color='red', linewidth=1.5)
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


def discretize_plant(dt: float, I_A: float, b: float):
    """
    Exact discretization of the plant dynamics using matrix exponential.

    Continuous-time model:
        angle_dot  = dangle
        dangle_dot = -(b/I_A) * dangle + (1/I_A) * u

    Returns (Ad, Bd) discrete-time matrices.
    """
    A = np.array([[0.0, 1.0],
                  [0.0, -b / max(1e-12, I_A)]], dtype=np.float64)
    B = np.array([[0.0],
                  [1.0 / max(1e-12, I_A)]], dtype=np.float64)

    # Augmented matrix for exact discretization
    M = np.zeros((3, 3), dtype=np.float64)
    M[0:2, 0:2] = A
    M[0:2, 2:3] = B

    Md = expm(M * dt)
    Ad = Md[0:2, 0:2]
    Bd = Md[0:2, 2:3]

    return Ad, Bd


def estimate_kalman_noise_parameters(csv_filepath,
                                      sphere_mass: float = 2.5,
                                      sphere_radius: float = 0.65,
                                      fluid_viscosity: float = 1.0):
    """
    Estimate measurement noise (R) and process noise (Q) for Kalman filter tuning.

    Uses the true plant dynamics model:
        x_{k+1} = Ad * x_k + Bd * u_k + w_k
        z_k = H * x_k + v_k

    where x = [angle, dangle]^T, u = torque command

    Args:
        csv_filepath: Path to the CSV file with orientation control data
        sphere_mass: Mass of the equivalent sphere (kg) - default 5.0
        sphere_radius: Radius of the equivalent sphere (m) - default 0.1
        fluid_viscosity: Fluid viscosity (Pa·s) - default 0.0010016 (water at 20°C)

    Returns dict with noise estimates and saves analysis plot.
    """
    df = pd.read_csv(csv_filepath)
    csv_path = pathlib.Path(csv_filepath)

    # Convert timestamps to seconds
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
    dt_array = np.diff(timestamps_sec)

    # Extract pitch and yaw data
    pitch = df['pitch_error'].values
    dpitch = df['dpitch_error'].values
    yaw = df['yaw_error'].values
    dyaw = df['dyaw_error'].values

    # Extract torque commands
    tau_pitch = df['pitch_torque_command'].values
    tau_yaw = df['yaw_torque_command'].values

    # Compute focal distance from centroid (d = ||centroid||)
    centroid_x = df['centroid_x'].values
    centroid_y = df['centroid_y'].values
    centroid_z = df['centroid_z'].values
    d_array = np.sqrt(centroid_x**2 + centroid_y**2 + centroid_z**2)

    # Compute physical parameters
    inertia_B = (2.0 / 5.0) * sphere_mass * (sphere_radius ** 2)
    linear_drag = 6.0 * np.pi * fluid_viscosity * sphere_radius

    def estimate_axis_noise_plant(angle, dangle, tau, dt_array, d_array, axis_name):
        """Estimate noise parameters for a single axis using plant dynamics model."""
        n = len(angle)

        # Detect if dataset is mostly stationary
        dangle_std = np.std(dangle)
        motion_threshold = 0.1  # rad/s
        is_stationary = dangle_std < motion_threshold

        if is_stationary:
            stationary_mask = np.ones(n, dtype=bool)
            velocity_threshold = np.inf
        else:
            velocity_threshold = np.percentile(np.abs(dangle), 25)
            stationary_mask = np.abs(dangle) < velocity_threshold

        # Measurement noise (R) - variance during stationary periods
        if np.sum(stationary_mask) > 10:
            angle_stationary = angle[stationary_mask]
            angle_detrended = angle_stationary - pd.Series(angle_stationary).rolling(
                window=min(10, len(angle_stationary)//2), center=True, min_periods=1).mean().values
            R = np.var(angle_detrended)
        else:
            R = np.var(np.diff(angle)) / 2

        # Process noise using plant dynamics model
        # x_{k+1} = Ad * x_k + Bd * u_k
        # Prediction error = x_{k+1,actual} - x_{k+1,predicted}
        angle_pred_errors = []
        dangle_pred_errors = []

        for k in range(n - 1):
            dt = dt_array[k]
            d = d_array[k]

            # Compute I_A and b for this timestep
            I_A = inertia_B + sphere_mass * d**2
            b = linear_drag * d * d

            # Get discretized matrices
            Ad, Bd = discretize_plant(dt, I_A, b)

            # Current state
            x_k = np.array([angle[k], dangle[k]])

            # Previous torque (u_{k-1} affects prediction to k)
            # For first step, use current torque as approximation
            u_k = tau[k]

            # Predicted next state
            x_pred = Ad @ x_k + Bd.flatten() * u_k

            # Actual next state
            x_actual = np.array([angle[k+1], dangle[k+1]])

            # Prediction errors
            angle_pred_errors.append(x_actual[0] - x_pred[0])
            dangle_pred_errors.append(x_actual[1] - x_pred[1])

        angle_pred_errors = np.array(angle_pred_errors)
        dangle_pred_errors = np.array(dangle_pred_errors)

        # Process noise covariance estimates
        Q_angle = np.var(angle_pred_errors)
        Q_dangle = np.var(dangle_pred_errors)

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

    # Estimate for both axes using plant dynamics
    pitch_est = estimate_axis_noise_plant(pitch, dpitch, tau_pitch, dt_array, d_array, 'pitch')
    yaw_est = estimate_axis_noise_plant(yaw, dyaw, tau_yaw, dt_array, d_array, 'yaw')

    # Combine estimates: use max for robustness (conservative)
    R_combined = max(pitch_est['R'], yaw_est['R'])
    Q_angle_combined = max(pitch_est['Q_angle'], yaw_est['Q_angle'])
    Q_dangle_combined = max(pitch_est['Q_dangle'], yaw_est['Q_dangle'])

    # Also compute average for reference
    R_avg = (pitch_est['R'] + yaw_est['R']) / 2
    Q_angle_avg = (pitch_est['Q_angle'] + yaw_est['Q_angle']) / 2
    Q_dangle_avg = (pitch_est['Q_dangle'] + yaw_est['Q_dangle']) / 2

    is_stationary_dataset = pitch_est['is_stationary'] and yaw_est['is_stationary']

    # Compute average system parameters for display
    d_mean = np.mean(d_array)
    I_A_mean = inertia_B + sphere_mass * d_mean**2
    b_mean = linear_drag * d_mean * d_mean

    # === Create analysis plot ===
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    # Row 1: Pitch and Yaw errors with stationary periods
    for col, (angle, est, name, color) in enumerate([
        (pitch, pitch_est, 'Pitch', 'blue'),
        (yaw, yaw_est, 'Yaw', 'red')
    ]):
        axes[0, col].plot(timestamps_sec, np.degrees(angle), f'{color[0]}-', alpha=0.7, label=f'{name} Error')
        stationary_times = timestamps_sec[est['stationary_mask']]
        stationary_angle = angle[est['stationary_mask']]
        axes[0, col].scatter(stationary_times, np.degrees(stationary_angle),
                             c='green', s=10, alpha=0.5, label='Stationary')
        axes[0, col].set_ylabel(f'{name} Error (deg)')
        axes[0, col].set_title(f'{name}: R = {est["R"]:.2e} rad² ({np.degrees(np.sqrt(est["R"])):.3f}° std)')
        axes[0, col].legend(loc='upper right')
        axes[0, col].grid(True, alpha=0.3)

    # Row 2: Position prediction errors (using plant dynamics)
    for col, (est, name, color) in enumerate([
        (pitch_est, 'Pitch', 'blue'),
        (yaw_est, 'Yaw', 'red')
    ]):
        axes[1, col].plot(timestamps_sec[1:], np.degrees(est['prediction_errors']), f'{color[0]}-', alpha=0.7)
        axes[1, col].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        std_deg = np.degrees(np.sqrt(est['Q_angle']))
        axes[1, col].axhline(y=std_deg, color='g', linestyle='--', alpha=0.5)
        axes[1, col].axhline(y=-std_deg, color='g', linestyle='--', alpha=0.5)
        axes[1, col].set_ylabel('Prediction Error (deg)')
        axes[1, col].set_title(f'{name}: Q_angle = {est["Q_angle"]:.2e} rad² ({std_deg:.3f}° std)')
        axes[1, col].grid(True, alpha=0.3)

    # Row 3: Derivative prediction errors (using plant dynamics)
    for col, (est, name, color) in enumerate([
        (pitch_est, 'Pitch', 'blue'),
        (yaw_est, 'Yaw', 'red')
    ]):
        axes[2, col].plot(timestamps_sec[1:], np.degrees(est['dangle_prediction_errors']), color, alpha=0.7)
        axes[2, col].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        std_deg = np.degrees(np.sqrt(est['Q_dangle']))
        axes[2, col].set_ylabel('d(angle)/dt Pred Error (deg/s)')
        axes[2, col].set_title(f'{name}: Q_dangle = {est["Q_dangle"]:.2e} ({std_deg:.3f}°/s std)')
        axes[2, col].grid(True, alpha=0.3)

    # Row 4: Combined summary
    axes[3, 0].text(0.5, 0.5,
        f"PLANT DYNAMICS MODEL ESTIMATES\n"
        f"(max of pitch/yaw)\n\n"
        f"Measurement Noise:\n"
        f"  R = {R_combined:.6e} rad²\n"
        f"    ({np.degrees(np.sqrt(R_combined)):.4f}° std)\n\n"
        f"Process Noise:\n"
        f"  Q_angle  = {Q_angle_combined:.6e} rad²\n"
        f"  Q_dangle = {Q_dangle_combined:.6e} (rad/s)²\n\n"
        f"For Kalman state [θ, θ̇]:\n"
        f"  Q = diag([{Q_angle_combined:.2e},\n"
        f"            {Q_dangle_combined:.2e}])",
        transform=axes[3, 0].transAxes, fontsize=10, verticalalignment='center',
        horizontalalignment='center', fontfamily='monospace',
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
        f"  b = {b_mean:.6f} N·m·s/rad\n"
        f"  linear_drag = {linear_drag:.6f} N·s/m\n\n"
        f"Individual R estimates:\n"
        f"  Pitch: {pitch_est['R']:.2e}\n"
        f"  Yaw: {yaw_est['R']:.2e}",
        transform=axes[3, 1].transAxes, fontsize=10, verticalalignment='center',
        horizontalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[3, 1].axis('off')
    axes[3, 1].set_title('System Parameters')

    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')

    plt.tight_layout()

    plot_filename = csv_path.parent / f'{csv_path.stem}_kalman_noise.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()

    # Write results to text file
    txt_filename = csv_path.parent / f'{csv_path.stem}_kalman_noise.txt'
    with open(txt_filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("KALMAN FILTER NOISE ESTIMATES (Plant Dynamics Model)\n")
        f.write("="*70 + "\n\n")

        f.write("Plant Model:\n")
        f.write("  θ̈ = -(b/I_A)·θ̇ + (1/I_A)·τ\n")
        f.write("  x_{k+1} = Ad·x_k + Bd·u_k + w_k\n")
        f.write("  z_k = H·x_k + v_k,  H = [1, 0]\n\n")

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

        if is_stationary_dataset:
            f.write("Dataset type: STATIONARY (controller off)\n")
            f.write("  -> R estimate is VALID (ideal conditions)\n")
            f.write("  -> Q estimates may be less accurate\n\n")
        else:
            f.write("Dataset type: MOTION (controller on)\n")
            f.write("  -> Both R and Q estimates are valid\n\n")

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
        f.write(f"  Q_angle = {pitch_est['Q_angle']:.6e} rad²\n")
        f.write(f"  Q_dangle = {pitch_est['Q_dangle']:.6e} (rad/s)²\n")
        f.write(f"Yaw:\n")
        f.write(f"  R = {yaw_est['R']:.6e} rad²\n")
        f.write(f"  Q_angle = {yaw_est['Q_angle']:.6e} rad²\n")
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


def visualize_pointclouds_over_time(point_clouds, normals, centroids, torque_vectors, timestamps, output_path):
    """
    Create an Open3D visualization with all point clouds colored by time gradient.
    Also shows normals and torque vectors at each centroid.

    Args:
        point_clouds: List of numpy arrays (N, 3) for each timestep
        normals: List of numpy arrays (3,) surface normals
        centroids: List of numpy arrays (3,) surface centroids
        torque_vectors: List of numpy arrays (3,) torque commands [pitch, yaw, roll]
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
    torque_lines = []  # Separate list for torque vectors

    # Process each point cloud
    for points, normal, centroid, torque_vec, t in zip(point_clouds, normals, centroids, torque_vectors, timestamps):
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

        # Create normal arrow at centroid (flip normal sign) - colored by time
        if centroid is not None and normal is not None:
            flipped_normal = -normal  # Flip the normal vector
            arrow_length = 0.05  # 5cm arrow
            line_points = [centroid, centroid + flipped_normal * arrow_length]
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(line_points)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([color])
            geometries.append(line)

        # Create torque vector arrow at centroid - magenta color
        if centroid is not None and torque_vec is not None:
            torque_norm = np.linalg.norm(torque_vec)
            if torque_norm > 1e-6:
                # Normalize and scale for visualization
                torque_dir = torque_vec / torque_norm
                arrow_length = 0.03  # 3cm arrow for torque direction
                line_points = [centroid, centroid + torque_dir * arrow_length]
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector(line_points)
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0]])  # Magenta
                torque_lines.append(line)

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

    # Add torque vector lines to geometries
    geometries.extend(torque_lines)

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


def debag(bag_file,
          sphere_mass: float = 2.5,
          sphere_radius: float = 0.65,
          fluid_viscosity: float = 1.0):
    """
    Process a ROS2 bag file containing OrientationControlData messages.

    Args:
        bag_file: Path to the bag file
        sphere_mass: Mass of the virtual sphere for admittance control (kg)
        sphere_radius: Radius of the virtual sphere for admittance control (m)
        fluid_viscosity: Viscosity coefficient for damping (Pa·s)
    """
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
    all_torque_vectors = []  # [pitch_torque, yaw_torque, roll_torque]
    all_timestamps = []

    # Initialize CSV writer
    csv_filename = output_base.with_suffix('.csv')
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # CSV Header matching new message structure with separate pitch/yaw
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
        # Pitch control (X-axis rotation) - filtered
        'pitch_error',
        'dpitch_error',
        'ipitch_error',
        # Pitch control - raw (unfiltered)
        'pitch_error_raw',
        'dpitch_error_raw',
        'pitch_torque_command',
        # Yaw control (Y-axis rotation) - filtered
        'yaw_error',
        'dyaw_error',
        'iyaw_error',
        # Yaw control - raw (unfiltered)
        'yaw_error_raw',
        'dyaw_error_raw',
        'yaw_torque_command',
        # Roll control (Z-axis rotation)
        'roll_error',
        'droll_error',
        'iroll_error',
        'roll_torque_command',
        # PID gains
        'k_p',
        'k_d',
        'k_i',
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

                # Store torque vector [pitch, yaw, roll] and transform to object frame
                torque_cam = np.array([
                    msg.pitch_torque_command,
                    msg.yaw_torque_command,
                    msg.roll_torque_command
                ])
                torque_obj = transform_vector(torque_cam, msg.camera_transform)
                all_torque_vectors.append(torque_obj)

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
            # Pitch control (X-axis rotation) - filtered
            msg.pitch_error,
            msg.dpitch_error,
            msg.ipitch_error,
            # Pitch control - raw (unfiltered)
            msg.pitch_error_raw,
            msg.dpitch_error_raw,
            msg.pitch_torque_command,
            # Yaw control (Y-axis rotation) - filtered
            msg.yaw_error,
            msg.dyaw_error,
            msg.iyaw_error,
            # Yaw control - raw (unfiltered)
            msg.yaw_error_raw,
            msg.dyaw_error_raw,
            msg.yaw_torque_command,
            # Roll control (Z-axis rotation)
            msg.roll_error,
            msg.droll_error,
            msg.iroll_error,
            msg.roll_torque_command,
            # PID gains
            msg.k_p,
            msg.k_d,
            msg.k_i,
        ])

    csv_file.close()
    print(f"CSV saved to {csv_filename} ({msg_count} messages)")

    # Generate the pitch/yaw and roll control plots
    generate_pitch_yaw_control_plots(csv_filename)
    generate_roll_control_plots(csv_filename)

    # Estimate Kalman filter noise parameters (analyzes both pitch and yaw)
    estimate_kalman_noise_parameters(
        csv_filename,
        sphere_mass=sphere_mass,
        sphere_radius=sphere_radius,
        fluid_viscosity=fluid_viscosity
    )

    # Generate Open3D point cloud visualization
    print(f"Creating point cloud visualization from {len(all_point_clouds)} clouds...")
    visualize_pointclouds_over_time(
        all_point_clouds,
        all_normals,
        all_centroids,
        all_torque_vectors,
        all_timestamps,
        output_base
    )

    # Delete the original bag file (directory) to save space
    if bag_file.exists():
        shutil.rmtree(bag_file)
        print(f"Deleted bag file: {bag_file}")
