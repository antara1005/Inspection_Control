#!/usr/bin/env python3
# coding: utf-8
"""Latency excitation injector.

Publishes a known, broadband excitation wrench onto the admittance controller's
input so the whole control chain

    wrench in -> admittance -> servo -> UR/RTDE -> robot motion -> IMU

can be characterised. Pair this with a rosbag recording and
``latency_analysis.py`` to recover per-hop latency and the physical (IMU)
transfer function.

Design notes
------------
* We drive a SINGLE axis (default torque about Z) so every downstream waypoint
  can be reduced to a scalar EE angular/linear velocity about that axis and the
  hops compared by cross-correlation.
* A linear ``chirp`` (frequency sweep) is the default: it is broadband, so one
  short run excites the system across the whole bandwidth and the cross-corr /
  transfer-function estimates are well conditioned. ``prbs`` is an alternative
  broadband input; ``step`` / ``impulse`` are there for quick eyeball checks.
* The signal is cosine-ramped in and out (``ramp_s``) so we never command a
  jerk into the real robot.
* The publish header stamp is the ROS clock at publish time. When you record a
  bag, the analyzer can key off either the header stamp or the bag receive time
  (both live on the host clock).

SAFETY: this moves the real robot. Start with a small ``amplitude`` and a low
``max_freq``, keep the workspace clear, and keep a hand on the e-stop. Do NOT
run a teleop/space-mouse node onto the same ``wrench_topic`` at the same time.
"""

import json
import math
import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import WrenchStamped


# Which scalar of the wrench each axis name maps to.
_AXIS_INDEX = {
    'fx': 0, 'fy': 1, 'fz': 2,   # force components (N)
    'tx': 3, 'ty': 4, 'tz': 5,   # torque components (N·m)
}


class LatencyInjectionNode(Node):
    def __init__(self):
        super().__init__('latency_injection')

        self.declare_parameters(
            namespace='',
            parameters=[
                # Where admittance listens. Default is the teleop wrench input.
                ('wrench_topic', '/teleop/wrench_cmds'),
                ('frame_id', 'eoat_camera_link'),

                # Excitation shape: 'chirp' | 'prbs' | 'step' | 'impulse'.
                ('waveform', 'chirp'),
                # Which wrench component to drive: fx/fy/fz/tx/ty/tz.
                ('axis', 'tz'),
                # Peak amplitude (N for f*, N·m for t*). Keep small to start.
                ('amplitude', 0.5),

                # Output/sampling rate of the excitation (Hz). Should exceed
                # admittance publish_rate so the input is well sampled.
                ('rate', 500.0),
                # Total active excitation duration (s), excluding start delay.
                ('duration', 20.0),
                # Quiet lead-in before excitation so the bag captures baseline.
                ('start_delay', 2.0),
                # Cosine ramp in/out (s) to avoid commanding a jerk.
                ('ramp_s', 0.75),

                # chirp: sweep from min_freq -> max_freq over `duration` (Hz).
                ('min_freq', 0.2),
                ('max_freq', 15.0),

                # prbs: bit (hold) period in seconds -> ~1/(2*bit) bandwidth.
                ('prbs_bit_s', 0.05),
                ('prbs_seed', 1),

                # impulse: pulse width (s), applied once at start.
                ('impulse_width_s', 0.05),

                # Where to drop the run metadata sidecar for the analyzer.
                ('output_dir', '/data'),
                # Auto-shutdown after the run completes (+ ramp/tail).
                ('shutdown_when_done', True),
            ],
        )

        g = self.get_parameter
        self.wrench_topic = str(g('wrench_topic').value)
        self.frame_id = str(g('frame_id').value)
        self.waveform = str(g('waveform').value).lower()
        self.axis = str(g('axis').value).lower()
        self.amplitude = float(g('amplitude').value)
        self.rate = float(g('rate').value)
        self.duration = float(g('duration').value)
        self.start_delay = float(g('start_delay').value)
        self.ramp_s = float(g('ramp_s').value)
        self.min_freq = float(g('min_freq').value)
        self.max_freq = float(g('max_freq').value)
        self.prbs_bit_s = float(g('prbs_bit_s').value)
        self.prbs_seed = int(g('prbs_seed').value)
        self.impulse_width_s = float(g('impulse_width_s').value)
        self.output_dir = str(g('output_dir').value)
        self.shutdown_when_done = bool(g('shutdown_when_done').value)

        if self.axis not in _AXIS_INDEX:
            raise ValueError(
                f"axis must be one of {list(_AXIS_INDEX)}, got '{self.axis}'")
        self.axis_index = _AXIS_INDEX[self.axis]

        self._rng = np.random.default_rng(self.prbs_seed)
        self._prbs_state = None  # (level, next_toggle_t)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.pub = self.create_publisher(WrenchStamped, self.wrench_topic, qos)

        self._msg = WrenchStamped()
        self._msg.header.frame_id = self.frame_id

        self._t0 = None            # ROS time (s) of first tick
        self._done_logged = False
        # Total wall time we keep publishing (zeros during delay/tail).
        self._tail_s = self.ramp_s + 0.5
        self._end_t = self.start_delay + self.duration + self._tail_s

        self.timer = self.create_timer(1.0 / self.rate, self._tick)

        self._write_metadata()
        self.get_logger().info(
            f"latency_injection: {self.waveform} on {self.axis} "
            f"(amp={self.amplitude}), topic={self.wrench_topic}, "
            f"start_delay={self.start_delay}s duration={self.duration}s")

    # ------------------------------------------------------------------ util
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ramp(self, t_active: float) -> float:
        """Cosine ramp: 0->1 over ramp_s at the start, 1->0 at the end."""
        if self.ramp_s <= 0.0:
            return 1.0
        rise = 1.0
        if t_active < self.ramp_s:
            rise = 0.5 * (1.0 - math.cos(math.pi * t_active / self.ramp_s))
        fall = 1.0
        t_from_end = self.duration - t_active
        if t_from_end < self.ramp_s:
            fall = 0.5 * (1.0 - math.cos(math.pi * max(t_from_end, 0.0) / self.ramp_s))
        return max(0.0, min(rise, fall))

    def _signal(self, t_active: float) -> float:
        """Return the (un-ramped) excitation value at active time t_active."""
        if self.waveform == 'chirp':
            # Linear chirp; instantaneous phase = 2*pi*(f0*t + k/2*t^2).
            k = (self.max_freq - self.min_freq) / max(self.duration, 1e-9)
            phase = 2.0 * math.pi * (self.min_freq * t_active
                                     + 0.5 * k * t_active * t_active)
            return self.amplitude * math.sin(phase)

        if self.waveform == 'prbs':
            if self._prbs_state is None:
                self._prbs_state = (
                    1.0 if self._rng.random() < 0.5 else -1.0,
                    self.prbs_bit_s,
                )
            level, next_toggle = self._prbs_state
            if t_active >= next_toggle:
                level = -level
                next_toggle = t_active + self.prbs_bit_s
                self._prbs_state = (level, next_toggle)
            return self.amplitude * level

        if self.waveform == 'step':
            return self.amplitude

        if self.waveform == 'impulse':
            return self.amplitude if t_active < self.impulse_width_s else 0.0

        return 0.0

    # ------------------------------------------------------------------ loop
    def _tick(self):
        now = self._now()
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0

        value = 0.0
        if self.start_delay <= t < (self.start_delay + self.duration):
            t_active = t - self.start_delay
            value = self._signal(t_active) * self._ramp(t_active)

        m = self._msg
        m.header.stamp = self.get_clock().now().to_msg()
        m.wrench.force.x = m.wrench.force.y = m.wrench.force.z = 0.0
        m.wrench.torque.x = m.wrench.torque.y = m.wrench.torque.z = 0.0
        comp = ('force.x', 'force.y', 'force.z',
                'torque.x', 'torque.y', 'torque.z')[self.axis_index]
        obj, attr = comp.split('.')
        setattr(getattr(m.wrench, obj), attr, float(value))
        self.pub.publish(m)

        if t >= self._end_t and not self._done_logged:
            self._done_logged = True
            self.get_logger().info(
                'latency_injection: excitation complete.')
            if self.shutdown_when_done:
                # One clean zero was just published; stop the timer and exit.
                self.timer.cancel()
                rclpy.shutdown()

    # -------------------------------------------------------------- metadata
    def _write_metadata(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError:
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        meta = {
            'saved_at': ts,
            'start_epoch_sec': self._now(),
            'wrench_topic': self.wrench_topic,
            'frame_id': self.frame_id,
            'waveform': self.waveform,
            'axis': self.axis,
            'amplitude': self.amplitude,
            'rate': self.rate,
            'duration': self.duration,
            'start_delay': self.start_delay,
            'ramp_s': self.ramp_s,
            'min_freq': self.min_freq,
            'max_freq': self.max_freq,
            'prbs_bit_s': self.prbs_bit_s,
        }
        path = os.path.join(self.output_dir, f'latency_injection_{ts}.json')
        try:
            with open(path, 'w') as f:
                json.dump(meta, f, indent=2)
            self.get_logger().info(f'wrote run metadata: {path}')
        except OSError as e:
            self.get_logger().warn(f'could not write metadata: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LatencyInjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # rclpy may already be shut down by the auto-stop path.
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
