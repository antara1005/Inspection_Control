#!/usr/bin/env python3
"""focus_volume_node: the feature front end of the hybrid autofocus, live.

Runs inspection_control.focus_volume (AKAZE-acquire, KLT-sustain, FB-gate-retire,
grid spawn manager, local focus per feature) on the macro camera, and publishes:

* ``~/annotations`` (foxglove_msgs/ImageAnnotations): the features over the camera
  image, coloured by focus relative to each feature's own best, with motion
  tails, the spawn grid, per-feature focus readouts and a summary. In Foxglove:
  an Image panel on the camera topic, with this topic ticked under annotations.
  Pixel coordinates are the full-resolution camera image's, which the H.264
  remote-viewing copy (/pi_camera/video/foxglove) shares.
* ``~/features`` (inspection_control_interfaces/FocusFeatures): every feature
  and the summary, for the autofocus controller. Published only once the base
  layer has been rebuilt with that message; the annotations don't need it.

Pairs each frame with the focal distance from ``distance_topic`` while it is
fresh, so each feature records the distance at which it was sharpest.
"""

import math
import os

import cv2
from foxglove_msgs.msg import (
    CircleAnnotation, Color, ImageAnnotations, Point2, PointsAnnotation, TextAnnotation)
from inspection_control.focus_volume import (
    ACTIVE, COASTING, FocusVolumeConfig, FocusVolumeTracker, TENTATIVE)
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64

try:
    from inspection_control_interfaces.msg import FocusFeature, FocusFeatures
except ImportError:     # base layer built before these messages existed
    FocusFeature = FocusFeatures = None

# Parameters that rebuild the tracker when changed (FocusVolumeConfig fields).
_CONFIG_FIELDS = tuple(FocusVolumeConfig.__dataclass_fields__)


def _parse_cpuset(text):
    """'8-23' / '0,2,4-6' -> set of ints."""
    cpus = set()
    for part in filter(None, (p.strip() for p in text.split(','))):
        lo, _, hi = part.partition('-')
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _rgba(r, g, b, a=1.0):
    return Color(r=float(r), g=float(g), b=float(b), a=float(a))


def _focus_color(norm, alpha=1.0):
    """Red (0) -> yellow (0.5) -> green (1); grey when unknown."""
    if not math.isfinite(norm):
        return _rgba(0.6, 0.6, 0.6, alpha)
    x = min(1.0, max(0.0, norm))
    if x < 0.5:
        return _rgba(1.0, 2.0 * x, 0.0, alpha)
    return _rgba(2.0 * (1.0 - x), 1.0, 0.0, alpha)


class FocusVolumeNode(Node):
    def __init__(self):
        super().__init__('focus_volume')
        defaults = FocusVolumeConfig()
        self.declare_parameters('', [
            ('image_topic', '/pi_camera/image_raw/compressed'),
            ('distance_topic', '/tsdf_pose/focal_distance_m'),
            ('distance_timeout_s', 0.5),
            # Keep this node off the real-time cores: a cpuset such as '8-23'
            # ('' = leave as is), and a cap on OpenCV's worker threads.
            ('cpu_affinity', ''),
            ('opencv_threads', 2),
            # Annotations
            ('publish_annotations', True),
            ('annotation_rate_hz', 15.0),
            ('annotation_scale', 1.0),      # multiplies every size below
            ('show_grid', True),
            ('show_trails', True),
            ('show_summary', True),
            # Focus readouts at up to this many features, one per grid cell
            # (its longest-tracked active feature). 0 = none.
            ('label_features', 36),
            ('label_cells', False),         # median focus per cell, at its corner
            # |d ln(focus)/dt| above this shows an arrow in the readout (1/s).
            # A still feature's smoothed rate wanders by about 0.2/s.
            ('trend_threshold', 1.0),
        ] + [(name, getattr(defaults, name) if name != 'detect_region'
              else list(defaults.detect_region)) for name in _CONFIG_FIELDS])

        affinity = self.get_parameter('cpu_affinity').value
        if affinity:
            self._set_affinity(affinity)
        cv2.setNumThreads(int(self.get_parameter('opencv_threads').value))

        self.tracker = FocusVolumeTracker(self._config())
        self._rebuild = False
        self._latest_distance = None
        self._latest_distance_t = None
        self._last_annot_t = -math.inf
        self._last_frame_t = None
        self._fps = 0.0
        self._got_first_image = False
        self._dims = None

        image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        image_topic = self.get_parameter('image_topic').value
        self.create_subscription(CompressedImage, image_topic, self._on_image, image_qos)
        distance_topic = self.get_parameter('distance_topic').value
        if distance_topic:
            self.create_subscription(Float64, distance_topic, self._on_distance, 10)

        self.annot_pub = self.create_publisher(ImageAnnotations, '~/annotations', 1)
        self.features_pub = None
        if FocusFeatures is not None:
            self.features_pub = self.create_publisher(FocusFeatures, '~/features', 5)
        else:
            self.get_logger().warn(
                'inspection_control_interfaces has no FocusFeatures yet (rebuild the base '
                'layer): publishing annotations only')
        self.add_on_set_parameters_callback(self._on_params)
        self.get_logger().info(
            f'focus volume up: {image_topic}, distance {distance_topic or "none"}, '
            f'operator {self.tracker.cfg.focus_operator}, track scale '
            f'{self.tracker.cfg.track_scale}, grid {self.tracker.cfg.grid_cols}x'
            f'{self.tracker.cfg.grid_rows} x {self.tracker.cfg.per_cell}')

    # ----------------------------------------------------------- setup
    def _config(self):
        values = {}
        for name in _CONFIG_FIELDS:
            v = self.get_parameter(name).value
            values[name] = tuple(v) if name == 'detect_region' else v
        return FocusVolumeConfig(**values)

    def _set_affinity(self, text):
        try:
            want = _parse_cpuset(text) & os.sched_getaffinity(0)
            if not want:
                self.get_logger().warn(f'cpu_affinity {text}: none of those CPUs is available')
                return
            for tid in os.listdir('/proc/self/task'):     # every thread, DDS's included
                os.sched_setaffinity(int(tid), want)
            self.get_logger().info(f'pinned to CPUs {text}')
        except (OSError, ValueError) as e:
            self.get_logger().warn(f'cpu_affinity {text}: {e}')

    def _on_params(self, params):
        # The new values are only set once this returns: rebuild on the next frame.
        if any(p.name in _CONFIG_FIELDS for p in params):
            self._rebuild = True
        return SetParametersResult(successful=True)

    def _rebuild_tracker(self):
        self._rebuild = False
        try:
            self.tracker = FocusVolumeTracker(self._config())
            self.get_logger().info('tracker settings changed: tracks reset')
        except (TypeError, ValueError) as e:
            self.get_logger().error(f'tracker settings rejected, keeping the old ones: {e}')

    # ----------------------------------------------------------- inputs
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_distance(self, msg):
        self._latest_distance = float(msg.data)
        self._latest_distance_t = self._now()

    def _distance(self):
        if self._latest_distance_t is None:
            return math.nan
        if self._now() - self._latest_distance_t > self.get_parameter('distance_timeout_s').value:
            return math.nan
        return self._latest_distance

    def _on_image(self, msg):
        try:
            if self._rebuild:
                self._rebuild_tracker()
            gray = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                self.get_logger().warn(f"couldn't decode a '{msg.format}' frame",
                                       throttle_duration_sec=5.0)
                return
            if not self._got_first_image:
                self._got_first_image = True
                self.get_logger().info(
                    f'first image: {gray.shape[1]}x{gray.shape[0]} ({msg.format})')
            stamp = msg.header.stamp
            t = stamp.sec + stamp.nanosec * 1e-9
            res = self.tracker.process(gray, t, self._distance())

            now = self._now()
            if self._last_frame_t is not None:
                dt = now - self._last_frame_t
                if dt > 0:
                    self._fps += 0.1 * (1.0 / dt - self._fps)
            self._last_frame_t = now

            if self.features_pub is not None:
                self.features_pub.publish(self._features_msg(res, msg.header))
            rate = self.get_parameter('annotation_rate_hz').value
            if (self.get_parameter('publish_annotations').value
                    and (rate <= 0 or now - self._last_annot_t >= 1.0 / rate)):
                self._last_annot_t = now
                self.annot_pub.publish(self._annotations(res, stamp))
            self.get_logger().info(
                f'{res.n_active} active, {res.n_tentative} tentative, {res.n_coasting} '
                f'coasting, coverage {res.coverage:.0%}, {res.process_ms:.0f} ms/frame, '
                f'{self._fps:.1f} fps', throttle_duration_sec=30.0)
        except Exception as e:      # one bad frame must not stop the node
            self.get_logger().error(f'frame failed: {e}', throttle_duration_sec=5.0)

    # ---------------------------------------------------------- outputs
    @staticmethod
    def _features_msg(res, header):
        out = FocusFeatures()
        out.header = header
        out.image_width, out.image_height = res.width, res.height
        out.distance = res.z
        for f in res.features:
            m = FocusFeature()
            m.id, m.state = f.track_id, f.state
            m.u, m.v = f.u, f.v
            m.focus_raw, m.focus = f.focus_raw, f.focus
            m.focus_max, m.focus_norm = f.focus_max, f.focus_norm
            m.focus_rate, m.z_at_max, m.passed_peak = f.focus_rate, f.z_at_max, f.passed_peak
            m.age, m.fb_error = f.age, f.fb_error
            out.features.append(m)
        out.n_tentative, out.n_active, out.n_coasting = (
            res.n_tentative, res.n_active, res.n_coasting)
        out.n_spawned, out.n_retired = res.n_spawned, res.n_retired
        out.coverage = res.coverage
        out.median_focus_norm = res.median_focus_norm
        out.median_flow_px = res.median_flow_px
        out.cells_detected = len(res.cells_detected)
        out.process_ms = res.process_ms
        return out

    def _annotations(self, res, stamp):
        gp = self.get_parameter
        k = gp('annotation_scale').value * res.height / 1000.0   # 1.0 at 1000 px tall
        ann = ImageAnnotations()

        def points(kind, pts, color, thickness, colors=None, fill=None):
            a = PointsAnnotation()
            a.timestamp = stamp
            a.type = kind
            a.points = [Point2(x=float(x), y=float(y)) for x, y in pts]
            a.outline_color = color
            if colors:
                a.outline_colors = colors
            if fill is not None:
                a.fill_color = fill
            a.thickness = float(thickness)
            ann.points.append(a)

        def text(x, y, s, color, size, bg=0.55):
            a = TextAnnotation()
            a.timestamp = stamp
            a.position = Point2(x=float(x), y=float(y))
            a.text = s
            a.font_size = float(size)
            a.text_color = color
            a.background_color = _rgba(0.0, 0.0, 0.0, bg)
            ann.texts.append(a)

        def rect(r):
            x0, y0, x1, y1 = r
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        # Spawn grid: cells without an active feature shaded, cells the
        # detector ran in this frame outlined.
        if gp('show_grid').value and res.cells:
            lines = []
            for x0, y0, x1, y1 in res.cells:
                lines += [(x0, y0), (x1, y0), (x0, y0), (x0, y1)]
            last = res.cells[-1]
            first = res.cells[0]
            lines += [(first[0], last[3]), (last[2], last[3]),
                      (last[2], first[1]), (last[2], last[3])]
            points(PointsAnnotation.LINE_LIST, lines, _rgba(1, 1, 1, 0.25), 1.0 * k)
            for c, r in enumerate(res.cells):
                if res.cell_active[c] == 0:
                    points(PointsAnnotation.LINE_LOOP, rect(r), _rgba(1, 0, 0, 0.0), 0.0,
                           fill=_rgba(1.0, 0.0, 0.0, 0.10))
            for c in res.cells_detected:
                points(PointsAnnotation.LINE_LOOP, rect(res.cells[c]),
                       _rgba(0.3, 0.8, 1.0, 0.7), 2.0 * k)
            if gp('label_cells').value:
                for c, r in enumerate(res.cells):
                    n = res.cell_focus_norm[c]
                    if math.isfinite(n):
                        text(r[0] + 4 * k, r[1] + 18 * k, f'{100 * n:.0f}%',
                             _focus_color(n), 14 * k)

        active = [f for f in res.features if f.state == ACTIVE]
        # Motion tails: where each active feature was a few frames ago.
        if gp('show_trails').value and active:
            segs = []
            for f in active:
                segs += [(f.trail_u, f.trail_v), (f.u, f.v)]
            points(PointsAnnotation.LINE_LIST, segs, _rgba(0.3, 0.9, 1.0, 0.6), 1.5 * k)

        tentative = [(f.u, f.v) for f in res.features if f.state == TENTATIVE]
        if tentative:
            points(PointsAnnotation.POINTS, tentative, _rgba(0.85, 0.85, 0.85, 0.7), 5 * k)
        coasting = [(f.u, f.v) for f in res.features if f.state == COASTING]
        if coasting:
            points(PointsAnnotation.POINTS, coasting, _rgba(1.0, 0.2, 1.0, 0.9), 6 * k)

        # Active features: ring coloured by focus relative to the feature's own
        # best; filled once the feature has passed its peak.
        for f in active:
            c = CircleAnnotation()
            c.timestamp = stamp
            c.position = Point2(x=float(f.u), y=float(f.v))
            c.diameter = float(14 * k)
            c.thickness = float(2 * k)
            c.outline_color = _focus_color(f.focus_norm)
            c.fill_color = (_focus_color(f.focus_norm, 0.45) if f.passed_peak
                            else _rgba(0, 0, 0, 0))
            ann.circles.append(c)

        # Focus readouts at the features: one per cell, its longest-tracked.
        n_labels = int(gp('label_features').value)
        if n_labels > 0 and active:
            best = {}
            for f in active:
                if not math.isfinite(f.focus_norm):
                    continue
                if f.cell not in best or f.age > best[f.cell].age:
                    best[f.cell] = f
            thr = gp('trend_threshold').value
            for f in sorted(best.values(), key=lambda f: -f.age)[:n_labels]:
                arrow = '↑' if f.focus_rate > thr else '↓' if f.focus_rate < -thr else ''
                text(f.u + 10 * k, f.v - 10 * k, f'{100 * f.focus_norm:.0f}%{arrow}',
                     _focus_color(f.focus_norm), 16 * k)

        if gp('show_summary').value:
            cfg = self.tracker.cfg
            z = (f'{1000 * res.z:.1f} mm' if math.isfinite(res.z) else 'no distance')
            med = res.median_focus_norm
            lines = [
                (f'focus volume  AKAZE + KLT  {cfg.focus_operator}', _rgba(1, 1, 1)),
                (f'active {res.n_active}  tentative {res.n_tentative}  coasting '
                 f'{res.n_coasting}  coverage {100 * res.coverage:.0f}%', _rgba(1, 1, 1)),
                (f'median focus {100 * med:.0f}%' if math.isfinite(med)
                 else 'median focus -', _focus_color(med)),
                (f'flow {res.median_flow_px:.1f} px/frame   focal distance {z}',
                 _rgba(0.3, 0.9, 1.0)),
                (f'{res.process_ms:.0f} ms/frame  {self._fps:.1f} fps  detector: '
                 f'{"whole region" if res.full_detect else f"{len(res.cells_detected)} cells"}',
                 _rgba(0.8, 0.8, 0.8)),
                ("ring: focus / feature's best (red 0, green 100%)  filled: past its peak  "
                 '↑↓ sharpening / blurring', _rgba(0.8, 0.8, 0.8)),
            ]
            y = 30 * k
            for s, color in lines:
                text(12 * k, y, s, color, 20 * k, bg=0.6)
                y += 26 * k
        return ann


def main(args=None):
    rclpy.init(args=args)
    node = FocusVolumeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
