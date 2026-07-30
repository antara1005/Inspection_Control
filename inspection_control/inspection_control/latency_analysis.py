#!/usr/bin/env python3
# coding: utf-8
"""Offline latency analysis for the admittance control chain.

Consumes a rosbag recorded while ``latency_injection_node`` drives a broadband
excitation, reduces every waypoint in the chain

    wrench in -> twist cmd -> joint vel cmd -> joint_states -> IMU

to a signed scalar time series, and reports:

  * a per-hop latency budget (cross-correlation lag between adjacent stages),
  * the cumulative latency from the admittance output to the IMU,
  * a transfer function (Bode magnitude/phase + group delay) for the physical
    hop (commanded EE angular velocity -> IMU gyro),
  * overlay / cross-correlation / Bode plots.

Why bag *receive* time (default): every message is stamped on the one host
clock as it lands, so cross-correlation lags are directly comparable. The EOAT
IMU reaches ROS via a native, host-timestamped driver, so there is no serial
offset to subtract (leave --serial-latency-ms at 0). You may instead pass
``--time-source header`` since the driver's stamps are trustworthy host stamps.

Usage
-----
    python3 -m inspection_control.latency_analysis BAG_DIR \
        --axis tz --imu-topic /eoat/imu --out /data/latency_report

Requires: numpy, scipy, matplotlib, and a ROS 2 env (rosbag2_py, rclpy,
rosidl_runtime_py) on the path.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    _HAVE_MPL = False

try:
    from scipy import signal as sp_signal
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# Map the excited wrench axis to the natural velocity component to track at the
# task level, and the IMU field/component that should respond.
_AXIS_TO_TWIST = {
    'fx': ('linear', 0), 'fy': ('linear', 1), 'fz': ('linear', 2),
    'tx': ('angular', 0), 'ty': ('angular', 1), 'tz': ('angular', 2),
}
_AXIS_TO_IMU = {
    # (imu field, component). Torque -> gyro (angular_velocity),
    # force -> accelerometer (linear_acceleration).
    'tx': ('angular_velocity', 0), 'ty': ('angular_velocity', 1),
    'tz': ('angular_velocity', 2),
    'fx': ('linear_acceleration', 0), 'fy': ('linear_acceleration', 1),
    'fz': ('linear_acceleration', 2),
}
_WRENCH_COMPONENT = {
    'fx': ('force', 'x'), 'fy': ('force', 'y'), 'fz': ('force', 'z'),
    'tx': ('torque', 'x'), 'ty': ('torque', 'y'), 'tz': ('torque', 'z'),
}


@dataclass
class Stage:
    """A single waypoint reduced to a scalar time series."""
    name: str
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __bool__(self):
        return self.t.size > 1


# --------------------------------------------------------------------------- IO
def read_bag(bag_path, wanted_topics, time_source='receive'):
    """Return {topic: [(t_sec, msg), ...]} for the wanted topics.

    time_source: 'receive' uses the bag receive timestamp (single host clock,
    recommended); 'header' uses msg.header.stamp when present, else falls back
    to receive time for that message.
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message

    reader = SequentialReader()
    opened = False
    for storage_id in ('', 'mcap', 'sqlite3'):
        try:
            reader.open(
                StorageOptions(uri=bag_path, storage_id=storage_id),
                ConverterOptions('', ''))
            opened = True
            break
        except Exception:
            reader = SequentialReader()
    if not opened:
        raise RuntimeError(f'could not open bag at {bag_path}')

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = set(wanted_topics)
    missing = wanted - set(type_map)
    if missing:
        print(f'[warn] topics not in bag: {sorted(missing)}', file=sys.stderr)

    out = {t: [] for t in wanted}
    msg_cls = {}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in wanted:
            continue
        if topic not in msg_cls:
            msg_cls[topic] = get_message(type_map[topic])
        msg = deserialize_message(data, msg_cls[topic])
        t = t_ns * 1e-9
        if time_source == 'header' and hasattr(msg, 'header'):
            st = msg.header.stamp
            hs = st.sec + st.nanosec * 1e-9
            if hs > 0.0:
                t = hs
        out[topic].append((t, msg))
    return out


# --------------------------------------------------------------- stage builders
def _dominant_signed(cols, t):
    """Given a list of columns (each an array over time), pick the one with the
    largest variance and return it signed (used for joint-space vectors whose
    element ordering we don't want to depend on)."""
    variances = [np.var(c) for c in cols]
    idx = int(np.argmax(variances)) if variances else 0
    return np.asarray(cols[idx], dtype=float)


def build_stages(data, cfg):
    """Reduce raw messages into scalar Stages keyed by chain position."""
    stages = []

    # 1. Input wrench (excited component) --------------------------------
    obj, comp = _WRENCH_COMPONENT[cfg.axis]
    rows = data.get(cfg.wrench_topic, [])
    t = np.array([r[0] for r in rows])
    y = np.array([getattr(getattr(r[1].wrench, obj), comp) for r in rows])
    stages.append(Stage('input_wrench', t, y))

    # 2. Admittance twist command (task-level velocity) ------------------
    field_name, k = _AXIS_TO_TWIST[cfg.axis]
    rows = data.get(cfg.twist_topic, [])
    t = np.array([r[0] for r in rows])
    y = np.array([getattr(getattr(r[1].twist, field_name),
                          ('x', 'y', 'z')[k]) for r in rows])
    stages.append(Stage('twist_cmd', t, y))

    # 3. Servo joint velocity command (Float64MultiArray, order unknown) --
    rows = data.get(cfg.joint_cmd_topic, [])
    if rows:
        t = np.array([r[0] for r in rows])
        arr = [list(r[1].data) for r in rows]
        width = max((len(a) for a in arr), default=0)
        cols = [np.array([a[j] if len(a) > j else 0.0 for a in arr])
                for j in range(width)]
        y = _dominant_signed(cols, t) if cols else np.zeros_like(t)
        stages.append(Stage('joint_cmd', t, y))

    # 4. joint_states velocity (measured, dominant joint) -----------------
    rows = data.get(cfg.joint_states_topic, [])
    if rows:
        t = np.array([r[0] for r in rows])
        arr = [list(r[1].velocity) for r in rows]
        width = max((len(a) for a in arr), default=0)
        cols = [np.array([a[j] if len(a) > j else 0.0 for a in arr])
                for j in range(width)]
        y = _dominant_signed(cols, t) if cols else np.zeros_like(t)
        stages.append(Stage('joint_states', t, y))

    # 5. IMU (gyro for torque axis / accel for force axis) ----------------
    imu_field, k = _AXIS_TO_IMU[cfg.axis]
    rows = data.get(cfg.imu_topic, [])
    if rows:
        t = np.array([r[0] for r in rows]) + cfg.serial_latency_s
        y = np.array([getattr(getattr(r[1], imu_field),
                              ('x', 'y', 'z')[k]) for r in rows])
        # Remove DC/gravity bias so xcorr keys on the AC excitation.
        if y.size:
            y = y - np.mean(y)
        stages.append(Stage('imu', t, y))

    return [s for s in stages if s]


# ------------------------------------------------------------------- resampling
def resample_common(stages, cfg):
    """Interpolate every stage onto one uniform grid over the overlapping,
    active window. Returns (t_grid, {name: y}, fs)."""
    t_start = max(s.t[0] for s in stages)
    t_end = min(s.t[-1] for s in stages)
    # Trim the quiet lead-in / tail if the user gave the injection window.
    if cfg.active_start is not None:
        t_start = max(t_start, t_start + cfg.active_start)
    if t_end <= t_start:
        raise RuntimeError('stages do not overlap in time')

    fs = cfg.resample_hz
    n = int((t_end - t_start) * fs)
    t_grid = t_start + np.arange(n) / fs

    band = (cfg.band_lo, cfg.band_hi)
    resampled = {}
    for s in stages:
        y = np.interp(t_grid, s.t, s.y)
        y = _bandpass(y, fs, band) if _HAVE_SCIPY else (y - np.mean(y))
        resampled[s.name] = y
    return t_grid, resampled, fs


def _bandpass(y, fs, band):
    lo, hi = band
    hi = min(hi, 0.49 * fs)
    if lo <= 0.0:
        sos = sp_signal.butter(4, hi, btype='low', fs=fs, output='sos')
    else:
        sos = sp_signal.butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    return sp_signal.sosfiltfilt(sos, y - np.mean(y))


# ------------------------------------------------------------- lag estimation
def xcorr_lag(a, b, fs, max_lag_s=1.0):
    """Lag (seconds) that best aligns b to a: positive => b lags a.

    Sign-insensitive (picks the extreme of the correlation, so a gain inversion
    between differently-scaled stages doesn't matter for timing)."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float('nan'), 0.0
    corr = np.correlate(b, a, mode='full') / (na * nb)
    lags = np.arange(-len(a) + 1, len(b)) / fs
    m = np.abs(lags) <= max_lag_s
    corr_w, lags_w = corr[m], lags[m]
    i = int(np.argmax(np.abs(corr_w)))
    # Parabolic refine around the peak for sub-sample resolution.
    lag = lags_w[i]
    if 0 < i < len(corr_w) - 1:
        y0, y1, y2 = np.abs(corr_w[i - 1:i + 2])
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            lag = lags_w[i] + 0.5 * (y0 - y2) / denom / fs
    return float(lag), float(corr_w[i])


def transfer_function(x, y, fs, band):
    """Welch estimate of Y/X: returns (freqs, mag, phase_rad, group_delay_s)."""
    nper = min(len(x), 1024)
    f, pxx = sp_signal.welch(x, fs=fs, nperseg=nper)
    _, pxy = sp_signal.csd(x, y, fs=fs, nperseg=nper)
    H = pxy / np.where(pxx == 0, np.nan, pxx)
    mag = np.abs(H)
    phase = np.unwrap(np.angle(H))
    m = (f >= band[0]) & (f <= min(band[1], 0.49 * fs))
    gd = float('nan')
    if np.count_nonzero(m) >= 2:
        # group delay = -dphase/domega, fit a line over the excitation band.
        w = 2 * np.pi * f[m]
        slope = np.polyfit(w, phase[m], 1)[0]
        gd = -slope
    return f, mag, phase, gd


# ------------------------------------------------------------------- reporting
def run(cfg):
    if not _HAVE_SCIPY:
        print('[warn] scipy not found: bandpass + transfer function disabled; '
              'cross-correlation still runs on mean-removed signals.',
              file=sys.stderr)

    topics = [cfg.wrench_topic, cfg.twist_topic, cfg.joint_cmd_topic,
              cfg.joint_states_topic, cfg.imu_topic]
    data = read_bag(cfg.bag, topics, cfg.time_source)
    stages = build_stages(data, cfg)
    names = [s.name for s in stages]
    print(f'stages present: {names}')
    if len(stages) < 2:
        print('need at least two stages to measure a hop; check topic names.',
              file=sys.stderr)
        return 1

    t_grid, ys, fs = resample_common(stages, cfg)

    # Adjacent-hop and cumulative latencies (relative to twist_cmd, the
    # admittance output, since that is "the input to the robot side").
    print('\n=== per-hop latency (cross-correlation) ===')
    print(f'{"hop":<28}{"lag [ms]":>12}{"peak corr":>12}')
    order = names
    ref = 'twist_cmd' if 'twist_cmd' in ys else order[0]
    hop_rows = []
    for a, b in zip(order[:-1], order[1:]):
        lag, peak = xcorr_lag(ys[a], ys[b], fs, cfg.max_lag_s)
        hop_rows.append((a, b, lag, peak))
        print(f'{a+" -> "+b:<28}{lag*1e3:>12.2f}{peak:>12.3f}')

    print('\n=== cumulative latency from admittance output (twist_cmd) ===')
    if ref in ys:
        for name in order:
            if name == ref:
                continue
            lag, peak = xcorr_lag(ys[ref], ys[name], fs, cfg.max_lag_s)
            tag = ' (incl. serial offset subtracted)' if name == 'imu' else ''
            print(f'{ref+" -> "+name:<28}{lag*1e3:>12.2f}{peak:>12.3f}{tag}')

    # Physical transfer function: twist_cmd -> imu.
    gd = float('nan')
    tf = None
    if _HAVE_SCIPY and 'twist_cmd' in ys and 'imu' in ys:
        f, mag, phase, gd = transfer_function(
            ys['twist_cmd'], ys['imu'], fs, (cfg.band_lo, cfg.band_hi))
        tf = (f, mag, phase)
        print('\n=== physical hop transfer function (twist_cmd -> imu) ===')
        print(f'group delay over [{cfg.band_lo}, {cfg.band_hi}] Hz: '
              f'{gd*1e3:.2f} ms')
        # -3 dB bandwidth relative to low-freq gain.
        m = (f >= cfg.band_lo) & (f <= cfg.band_hi)
        if np.count_nonzero(m) > 3:
            g0 = np.median(mag[m][:3])
            below = np.where(mag[m] <= g0 / np.sqrt(2))[0]
            if below.size:
                print(f'approx -3 dB bandwidth: {f[m][below[0]]:.2f} Hz')

    if cfg.out and _HAVE_MPL:
        _write_plots(cfg, t_grid, ys, order, tf, gd)
    return 0


def _write_plots(cfg, t_grid, ys, order, tf, gd):
    os.makedirs(os.path.dirname(cfg.out) or '.', exist_ok=True)
    t0 = t_grid[0]

    # Time-series overlay (normalised so shapes line up visually).
    fig, ax = plt.subplots(figsize=(13, 5))
    for name in order:
        y = ys[name]
        s = np.std(y)
        ax.plot(t_grid - t0, y / s if s > 1e-9 else y, label=name, alpha=0.8)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('normalised amplitude (std=1)')
    ax.set_title('control-chain waypoints (band-passed, normalised)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = cfg.out + '_timeseries.png'
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # Bode of the physical hop.
    if tf is not None:
        f, mag, phase = tf
        fig, (axm, axp) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axm.semilogx(f, 20 * np.log10(np.maximum(mag, 1e-12)))
        axm.set_ylabel('|H| [dB]')
        axm.set_title(f'twist_cmd -> IMU  (group delay {gd*1e3:.1f} ms)')
        axm.grid(True, which='both', alpha=0.3)
        axp.semilogx(f, np.degrees(phase))
        axp.set_ylabel('phase [deg]')
        axp.set_xlabel('frequency [Hz]')
        axp.axvspan(cfg.band_lo, cfg.band_hi, color='tab:green', alpha=0.08)
        axp.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        p2 = cfg.out + '_bode.png'
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        print(f'\nwrote {p1} and {p2}')
    else:
        print(f'\nwrote {p1}')


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', help='path to the rosbag directory')
    ap.add_argument('--axis', default='tz', choices=list(_WRENCH_COMPONENT),
                    help='excited wrench axis used during injection')
    ap.add_argument('--wrench-topic', default='/teleop/wrench_cmds')
    ap.add_argument('--twist-topic', default='/servo_node/delta_twist_cmds')
    ap.add_argument('--joint-cmd-topic',
                    default='/ur5e_forward_velocity_controller/commands')
    ap.add_argument('--joint-states-topic', default='/joint_states')
    ap.add_argument('--imu-topic', default='/eoat/imu')
    ap.add_argument('--time-source', default='receive',
                    choices=['receive', 'header'],
                    help='clock for message times (receive = single host clock)')
    ap.add_argument('--serial-latency-ms', type=float, default=0.0,
                    help='constant IMU one-way latency to subtract (0 for a '
                         'native host-timestamped driver)')
    ap.add_argument('--resample-hz', type=float, default=500.0)
    ap.add_argument('--band-lo', type=float, default=0.2,
                    help='analysis band low edge [Hz] (match injection min_freq)')
    ap.add_argument('--band-hi', type=float, default=15.0,
                    help='analysis band high edge [Hz] (match injection max_freq)')
    ap.add_argument('--max-lag-s', type=float, default=0.5,
                    help='max |lag| searched in cross-correlation [s]')
    ap.add_argument('--active-start', type=float, default=None,
                    help='seconds to skip after overlap start (drop lead-in)')
    ap.add_argument('--out', default='/data/latency_report',
                    help='output path prefix for plots ("" to disable)')
    a = ap.parse_args(argv)

    class Cfg:
        pass
    c = Cfg()
    c.bag = a.bag
    c.axis = a.axis
    c.wrench_topic = a.wrench_topic
    c.twist_topic = a.twist_topic
    c.joint_cmd_topic = a.joint_cmd_topic
    c.joint_states_topic = a.joint_states_topic
    c.imu_topic = a.imu_topic
    c.time_source = a.time_source
    c.serial_latency_s = a.serial_latency_ms * 1e-3
    c.resample_hz = a.resample_hz
    c.band_lo = a.band_lo
    c.band_hi = a.band_hi
    c.max_lag_s = a.max_lag_s
    c.active_start = a.active_start
    c.out = a.out or None
    return c


def main(argv=None):
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    return run(cfg)


if __name__ == '__main__':
    raise SystemExit(main())
