#!/usr/bin/env python3
"""Hybrid autofocus.

Fuses compressed macro-camera frames with focal-distance measurements from
``/orientation_controller/focal_distance_m`` (std_msgs/Float64, headerless) and
computes a frame-based focus value on each image using the selected metric.

Two stages:

* **Sensing/fusion**: computes the focus value per frame, publishes it, and the
  ROI box + scalar readouts as a foxglove_msgs/ImageAnnotations overlay
  (composite on the raw feed in a Foxglove Image panel). The same overlay is
  also available as a drawn debug image (``visualize``) for rqt_image_view /
  RViz. Time-series of focus value and distance are viewed as Foxglove plots,
  so the overlay only shows the ROI + current scalars.

* **Active control**: a peak-hold autofocus. While *recording* (joy button) the
  node buffers the focus-vs-distance sweep; on record-stop it parabola-fits the
  samples near the peak and uses the vertex as the refined target (falling back to
  the raw arg-max if the sweep is one-sided/flat). On the *drive* trigger it runs a
  pole-placement PD
  controller that drives the focal distance to that recorded peak by publishing a
  WrenchStamped force along the camera optical (z) axis into the admittance node.
  The plant is the admittance node's virtual point mass with Stokes drag
  (``m d'' + c d' = F``, ``c = 6 pi mu r``), mirroring how the orientation
  controller derives its gains from the system dynamics — but translational
  (no pivot / parallel-axis term).
"""

import math
from collections import deque

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from geometry_msgs.msg import WrenchStamped
from foxglove_msgs.msg import (
    ImageAnnotations, PointsAnnotation, TextAnnotation, Point2, Color)

import inspection_control.focus_metrics as focus_metrics


class AutofocusNode(Node):
    def __init__(self):
        super().__init__('autofocus_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                # Sensing
                ('focus_metric', 'sobel'),
                ('image_topic', '/camera/image_raw/compressed'),
                ('distance_topic', '/orientation_controller/focal_distance_m'),
                # Centered ROI: (roi_x, roi_y) is the fractional centre,
                # roi_width/height are pixel sizes around it.
                ('roi_width', 100),
                ('roi_height', 100),
                ('roi_x', 0.5),
                ('roi_y', 0.5),
                # Debug image
                ('visualize', False),       # publish annotated debug image
                # foxglove_msgs/ImageAnnotations overlay of the ROI box + text.
                # Cheap (vector, not raster): overlay it on the raw camera feed
                # in a Foxglove Image panel instead of the drawn debug image.
                ('publish_annotations', True),
                ('publish_focus', True),

                # --- Active control ---
                ('frame_id', 'eoat_camera_link'),
                ('wrench_topic', '/autofocus/wrench_cmds'),
                # Recording/driving are toggled by teleop_node over the Trigger
                # services below (~/toggle_recording, ~/toggle_driving,
                # ~/disable_autofocus), rather than this node parsing raw Joy.
                # Keeps all button-index/array-bounds handling in teleop_node.
                ('control_rate', 50.0),     # Hz, PD control loop
                # Plant dynamics — must match the admittance node so the PD gains
                # reflect the real virtual plant (m d'' + c d' = F).
                ('sphere_mass', 2.5),
                ('sphere_radius', 0.65),
                ('fluid_viscosity', 1.0),
                # PD pole-placement tuning
                ('zeta', 1.0),              # closed-loop damping ratio (>=1)
                ('v_max', 0.05),            # m/s, focal-axis speed budget
                ('max_distance_error', 0.1),  # m, error scale for omega_n cap
                # +1: +z camera force decreases focal distance. Flip to -1 if the
                # robot drives AWAY from focus.
                ('focus_axis_sign', 1.0),
                # Stop driving once |error| is below this (m).
                ('focus_tolerance_m', 0.002),
                ('deriv_tau', 0.05),        # s, EMA time const for d/dt(distance)
                # Fine refinement: parabola-fit the recording sweep for the target.
                ('fit_enabled', True),
                ('fit_peak_fraction', 0.5),  # window: focus >= frac * max_focus
                ('fit_min_points', 5),       # min window samples to fit
            ]
        )

        gp = self.get_parameter
        self.focus_metric = gp('focus_metric').get_parameter_value().string_value
        self.image_topic = gp('image_topic').get_parameter_value().string_value
        self.distance_topic = gp('distance_topic').get_parameter_value().string_value
        self.roi_width = int(gp('roi_width').value)
        self.roi_height = int(gp('roi_height').value)
        self.roi_x = float(gp('roi_x').value)
        self.roi_y = float(gp('roi_y').value)
        self.visualize = bool(gp('visualize').value)
        self.publish_annotations = bool(gp('publish_annotations').value)
        self.publish_focus = bool(gp('publish_focus').value)

        # Active-control params
        self.frame_id = gp('frame_id').get_parameter_value().string_value
        self.wrench_topic = gp('wrench_topic').get_parameter_value().string_value
        self.control_rate = float(gp('control_rate').value)
        self.sphere_mass = float(gp('sphere_mass').value)
        self.sphere_radius = float(gp('sphere_radius').value)
        self.fluid_viscosity = float(gp('fluid_viscosity').value)
        self.zeta = float(gp('zeta').value)
        self.v_max = float(gp('v_max').value)
        self.max_distance_error = float(gp('max_distance_error').value)
        self.focus_axis_sign = float(gp('focus_axis_sign').value)
        self.focus_tolerance_m = float(gp('focus_tolerance_m').value)
        self.deriv_tau = float(gp('deriv_tau').value)
        self.fit_enabled = bool(gp('fit_enabled').value)
        self.fit_peak_fraction = float(gp('fit_peak_fraction').value)
        self.fit_min_points = int(gp('fit_min_points').value)
        # Plant: virtual point mass with Stokes drag (same as admittance node).
        self.linear_drag = 6.0 * math.pi * self.fluid_viscosity * self.sphere_radius

        self.bridge = CvBridge()

        # Observability flags (first-message logs).
        self._got_first_image = False
        self._got_first_distance = False

        # Latest-value hold for the headerless distance. The default executor is
        # single-threaded, so image and distance callbacks never run
        # concurrently — no locks needed.
        self._latest_distance = None        # metres
        self._latest_distance_t = None      # node-clock seconds
        self._dist_rate = 0.0               # d/dt(distance), EMA-smoothed (m/s)
        self._latest_focus = float('nan')   # most recent focus value (for overlay)

        # --- Peak-hold / control state ------------------------------------
        # mode: 'idle' (no command), 'recording' (track peak, no command),
        #       'driving' (PD force to recorded peak).
        self.mode = 'idle'
        self.max_focus = float('-inf')
        self.target_distance = None         # focal distance at max focus (m)
        self._within_tol = False            # inside the hold deadband
        # (focal_distance, focus_value) samples gathered during 'recording', used
        # to parabola-fit a refined target on record-stop.
        self._sweep = deque(maxlen=5000)

        # --- I/O ----------------------------------------------------------
        # Best-effort sensor QoS: compatible with both BEST_EFFORT and RELIABLE
        # publishers (a RELIABLE subscriber would receive NOTHING from a
        # best-effort camera), and the right policy for a lossy image stream.
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            CompressedImage, self.image_topic, self._on_image, image_qos)
        self.create_subscription(
            Float64, self.distance_topic, self._on_distance, 10)

        # Button-driven actions are triggered by teleop_node over these Trigger
        # services (mirrors orientation_controller's ~/toggle_* interface), so
        # this node no longer subscribes to raw Joy.
        self.create_service(Trigger, '~/toggle_recording', self._on_toggle_recording)
        self.create_service(Trigger, '~/toggle_driving', self._on_toggle_driving)
        self.create_service(Trigger, '~/disable_autofocus', self._on_disable_autofocus)

        self.focus_pub = None
        if self.publish_focus:
            self.focus_pub = self.create_publisher(
                Float64, f'{self.get_name()}/focus_value', 10)

        # Annotated debug image (CompressedImage). View with:
        #   ros2 run rqt_image_view rqt_image_view  -> /autofocus/debug_image
        self.debug_pub = self.create_publisher(
            CompressedImage, f'{self.get_name()}/debug_image/compressed', 1)

        # foxglove_msgs/ImageAnnotations mirroring the debug-image overlay.
        # In Foxglove: Image panel, base = image_topic, add this as an
        # annotation topic (coords are pixels of that image).
        self.annot_pub = self.create_publisher(
            ImageAnnotations, f'{self.get_name()}/annotations', 1)

        # Force command into the admittance node (summed with teleop/orientation).
        self.wrench_pub = self.create_publisher(WrenchStamped, self.wrench_topic, 10)

        # PD control loop.
        self.create_timer(1.0 / self.control_rate, self._control_tick)

        self.add_on_set_parameters_callback(self._on_params)

        self.get_logger().info(
            f"autofocus up — metric={self.focus_metric}, "
            f"image_topic={self.image_topic}, distance_topic={self.distance_topic}, "
            f"visualize={self.visualize}; control: wrench_topic={self.wrench_topic}, "
            f"m={self.sphere_mass}, c={self.linear_drag:.4f}")

    # ----------------------------------------------------------------------
    # Subscriptions
    # ----------------------------------------------------------------------
    def _on_distance(self, msg: Float64):
        if not self._got_first_distance:
            self._got_first_distance = True
            self.get_logger().info(
                f"first focal distance received: {msg.data:.4f} m")
        d = float(msg.data)
        t = self._now()
        # EMA-smoothed derivative d/dt(distance) for the PD damping term.
        if self._latest_distance_t is not None:
            dt = t - self._latest_distance_t
            if dt > 1e-6:
                raw_rate = (d - self._latest_distance) / dt
                alpha = 1.0 - math.exp(-dt / max(1e-6, self.deriv_tau))
                self._dist_rate += alpha * (raw_rate - self._dist_rate)
        self._latest_distance = d
        self._latest_distance_t = t

    def _on_image(self, msg: CompressedImage):
        # Catch-all so a single bad frame can't take down the executor.
        try:
            if not self._got_first_image:
                self._got_first_image = True
                self.get_logger().info(
                    f"first image received on {self.image_topic} "
                    f"(format='{msg.format}')")

            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='bgr8')

            roi, roi_box = self._crop_roi(frame)
            if roi.size == 0:
                self.get_logger().warn("ROI is empty; check roi_* params",
                                       throttle_duration_sec=2.0)
                return

            focus_value = float(focus_metrics.compute(self.focus_metric, roi)[0])
            distance = self._latest_distance  # may be None until first msg
            self._latest_focus = focus_value

            # Peak-hold: while recording, remember the focal distance at the
            # highest focus value seen, and buffer the sweep for parabola fitting.
            if self.mode == 'recording' and distance is not None:
                self._sweep.append((distance, focus_value))
                if focus_value > self.max_focus:
                    self.max_focus = focus_value
                    self.target_distance = distance

            if self.focus_pub is not None:
                self.focus_pub.publish(Float64(data=focus_value))

            if self.publish_annotations:
                self.annot_pub.publish(
                    self._build_annotations(roi_box, msg.header.stamp))

            if self.visualize:
                canvas = self._render(frame, roi_box)
                debug_msg = self.bridge.cv2_to_compressed_imgmsg(canvas)
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(
                f"image callback failed: {e}", throttle_duration_sec=2.0)

    # ----------------------------------------------------------------------
    # Active control
    # ----------------------------------------------------------------------
    def _on_toggle_recording(self, request, response):
        try:
            self._toggle_recording()
            response.success = True
            response.message = f'autofocus mode -> {self.mode}'
        except Exception as e:
            self.get_logger().error(f'toggle_recording failed: {e}')
            response.success = False
            response.message = str(e)
        return response

    def _on_toggle_driving(self, request, response):
        try:
            self._toggle_driving()
            response.success = True
            response.message = f'autofocus mode -> {self.mode}'
        except Exception as e:
            self.get_logger().error(f'toggle_driving failed: {e}')
            response.success = False
            response.message = str(e)
        return response

    def _on_disable_autofocus(self, request, response):
        # Unconditional stop (idempotent): leave any recording/driving mode and
        # zero the force so a single "all stop" button can force the controller
        # off regardless of its current state.
        try:
            self._set_mode('idle')
            response.success = True
            response.message = 'autofocus disabled (mode -> idle)'
        except Exception as e:
            self.get_logger().error(f'disable_autofocus failed: {e}')
            response.success = False
            response.message = str(e)
        return response

    def _toggle_recording(self):
        if self.mode == 'recording':
            self._set_mode('idle')
            argmax = self.target_distance
            if self.fit_enabled:
                vertex = self._fit_peak_parabola()
                if vertex is not None:
                    self.target_distance = vertex
                    self.get_logger().info(
                        f"peak refined: argmax={self._fmt(argmax)} -> "
                        f"fit={self._fmt(vertex)} (N={len(self._sweep)})")
                else:
                    self.get_logger().warn(
                        "parabola fit rejected (one-sided/flat sweep); "
                        f"using argmax {self._fmt(argmax)}")
            self.get_logger().info(
                f"recording stopped; target {self._fmt(self.target_distance)} "
                f"(max focus {self.max_focus:.1f})")
        else:
            self.max_focus = float('-inf')
            self.target_distance = None
            self._sweep.clear()
            self._set_mode('recording')
            self.get_logger().info("recording peak focus distance — sweep the robot")

    def _fit_peak_parabola(self):
        """Least-squares parabola fit to the recording sweep near the peak.

        Returns the refined focal distance (vertex of a concave parabola), or
        None if the sweep is too sparse, one-sided, or non-concave.
        """
        if len(self._sweep) < self.fit_min_points:
            return None
        d = np.array([s[0] for s in self._sweep], dtype=np.float64)
        f = np.array([s[1] for s in self._sweep], dtype=np.float64)

        # Window to the peak region so the local curve is ~parabolic.
        fmax = f.max()
        if not np.isfinite(fmax):
            return None
        keep = f >= self.fit_peak_fraction * fmax
        dw, fw = d[keep], f[keep]
        if dw.size < self.fit_min_points:
            return None

        # Need samples bracketing the arg-max (peak between observed extremes),
        # otherwise the vertex is an extrapolation to the sweep edge.
        d_argmax = dw[int(np.argmax(fw))]
        if not (dw.min() < d_argmax < dw.max()):
            return None

        # Center distances for conditioning, fit f = a*x^2 + b*x + c.
        d0 = dw.mean()
        a, b, _ = np.polyfit(dw - d0, fw, 2)
        if a >= 0:               # not concave -> no interior maximum
            return None
        vertex = d0 - b / (2.0 * a)
        return float(np.clip(vertex, dw.min(), dw.max()))

    def _toggle_driving(self):
        if self.mode == 'driving':
            self._set_mode('idle')
            self.get_logger().info("drive stopped")
        elif self.target_distance is None:
            self.get_logger().warn("no recorded peak yet; record before driving")
        else:
            self._within_tol = False
            self._set_mode('driving')
            self.get_logger().info(
                f"driving to focal distance {self._fmt(self.target_distance)}")

    def _set_mode(self, mode):
        self.mode = mode
        if mode != 'driving':
            self._publish_force(0.0)  # stop the admittance from holding old force

    def _control_tick(self):
        if self.mode != 'driving':
            return
        if self.target_distance is None or self._latest_distance is None:
            return

        # error in focal distance; ė = -ḋ but the PD is expressed directly on the
        # measured distance: F = Kp*(d - d_target) + Kd*ḋ (see module docstring).
        err = self._latest_distance - self.target_distance

        # Within the deadband: zero the force but stay in 'driving' so the loop
        # keeps holding and re-engages if the distance drifts. Only the drive
        # button leaves 'driving'.
        if abs(err) <= self.focus_tolerance_m:
            if not self._within_tol:
                self._within_tol = True
                self.get_logger().info(
                    f"reached target ({self._fmt(self.target_distance)}); holding")
            self._publish_force(0.0)
            return
        self._within_tol = False

        kp, kd = self._pd_gains()
        u = kp * err + kd * self._dist_rate
        f_max = self.linear_drag * self.v_max
        force = self.focus_axis_sign * u
        force = max(-f_max, min(f_max, force))  # saturate to speed budget
        self._publish_force(force)

    def _pd_gains(self):
        """Pole-placement PD gains for the plant  m d'' + c d' = F.

        Kp = m * p1*p2,  Kd = -m*(p1+p2) - c, with poles from (zeta, omega_n);
        omega_n is capped by the force/speed budget like the orientation node.
        """
        m = max(self.sphere_mass, 1e-9)
        c = self.linear_drag
        f_max = c * self.v_max
        omega_n = math.sqrt(max(f_max, 1e-12)
                            / (m * max(self.max_distance_error, 1e-6)))
        zeta = max(self.zeta, 1.0)  # keep poles real / non-oscillatory
        root = math.sqrt(zeta ** 2 - 1.0)
        p1 = -zeta * omega_n + omega_n * root
        p2 = -zeta * omega_n - omega_n * root
        kp = m * (p1 * p2)
        kd = -m * (p1 + p2) - c
        return kp, kd

    def _publish_force(self, fz: float):
        w = WrenchStamped()
        w.header.stamp = self.get_clock().now().to_msg()
        w.header.frame_id = self.frame_id
        w.wrench.force.z = float(fz)
        self.wrench_pub.publish(w)

    @staticmethod
    def _fmt(d):
        return "n/a" if d is None else f"{d:.4f} m"

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _crop_roi(self, frame):
        """Centered ROI. Returns (roi, (x0, y0, x1, y1)) clamped to the frame."""
        h, w = frame.shape[:2]
        cx = int(self.roi_x * w)
        cy = int(self.roi_y * h)
        x0 = max(0, cx - self.roi_width // 2)
        y0 = max(0, cy - self.roi_height // 2)
        x1 = min(w, cx + self.roi_width // 2)
        y1 = min(h, cy + self.roi_height // 2)
        return frame[y0:y1, x0:x1], (x0, y0, x1, y1)

    # ----------------------------------------------------------------------
    # Debug-image rendering (cv2 drawing only — no HighGUI)
    # ----------------------------------------------------------------------
    def _render(self, frame, roi_box):
        """Return an annotated BGR copy of `frame` (ROI box + readouts).

        Mirrors `_build_annotations`: the ROI box is colored by control mode
        (grey idle / red recording / green driving) and the metric/focus/
        distance readouts are stacked just below it. Time-series live in
        Foxglove plots, so only the current scalars are shown.
        """
        canvas = frame.copy()

        x0, y0, x1, y1 = roi_box
        cv2.rectangle(canvas, (x0, y0), (x1, y1), self._bgr(self._mode_color()), 2)

        # Readouts stacked just below the box, left edge aligned to it.
        y = y1 + 22
        for text, color in self._readouts():
            cv2.putText(canvas, text, (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._bgr(color), 2)
            y += 24

        return canvas

    @staticmethod
    def _rgba(r, g, b, a=1.0):
        """foxglove_msgs/Color from 0..1 RGBA components."""
        return Color(r=float(r), g=float(g), b=float(b), a=float(a))

    @staticmethod
    def _bgr(color):
        """foxglove_msgs/Color -> (B, G, R) 0..255 ints for cv2 drawing."""
        return (int(color.b * 255), int(color.g * 255), int(color.r * 255))

    def _mode_color(self):
        """ROI-box color by control mode: grey idle, red recording, green driving."""
        return {
            'recording': self._rgba(1.0, 0.0, 0.0),
            'driving':   self._rgba(0.0, 1.0, 0.0),
        }.get(self.mode, self._rgba(0.7, 0.7, 0.7))

    def _readouts(self):
        """(text, foxglove_msgs/Color) readouts shared by _render + annotations."""
        return [
            (f"metric: {self.focus_metric}",     self._rgba(1.0, 1.0, 1.0)),
            (f"focus: {self._latest_focus:.1f}", self._rgba(1.0, 0.843, 0.0)),
            (f"distance: {self._fmt(self._latest_distance)}", self._rgba(0.0, 0.784, 1.0)),
        ]

    def _build_annotations(self, roi_box, stamp):
        """foxglove_msgs/ImageAnnotations mirroring `_render`'s ROI box + text.

        Vector overlay to composite on the raw camera feed in a Foxglove Image
        panel (no re-encoded debug image needed). Pixel coordinates use the
        top-left image origin; text `position` is the bottom-left of the glyphs
        (like cv2.putText). The box color encodes the control mode; the
        readouts sit just below the box.
        """
        ann = ImageAnnotations()
        ann.timestamp = stamp

        x0, y0, x1, y1 = roi_box
        box = PointsAnnotation()
        box.timestamp = stamp
        box.type = PointsAnnotation.LINE_LOOP
        box.points = [
            Point2(x=float(x0), y=float(y0)),
            Point2(x=float(x1), y=float(y0)),
            Point2(x=float(x1), y=float(y1)),
            Point2(x=float(x0), y=float(y1)),
        ]
        box.outline_color = self._mode_color()  # grey/red/green by mode
        box.thickness = 2.0
        ann.points.append(box)

        # Readouts stacked just below the box, left edge aligned to it.
        # First baseline = box bottom + margin + font size.
        y = float(y1) + 6.0 + 16.0
        for text, color in self._readouts():
            t = TextAnnotation()
            t.timestamp = stamp
            t.position = Point2(x=float(x0), y=y)
            t.text = text
            t.font_size = 16.0
            t.text_color = color
            # Semi-transparent backing so text stays legible over the live feed
            # (cv2 relied on a thick stroke instead); set a=0.0 to disable.
            t.background_color = self._rgba(0.0, 0.0, 0.0, 0.5)
            ann.texts.append(t)
            y += 20.0

        return ann

    # ----------------------------------------------------------------------
    # Parameter updates
    # ----------------------------------------------------------------------
    def _on_params(self, params):
        for p in params:
            if p.name == 'focus_metric':
                self.focus_metric = p.value
            elif p.name == 'roi_width':
                self.roi_width = int(p.value)
            elif p.name == 'roi_height':
                self.roi_height = int(p.value)
            elif p.name == 'roi_x':
                self.roi_x = float(p.value)
            elif p.name == 'roi_y':
                self.roi_y = float(p.value)
            elif p.name == 'visualize':
                self.visualize = bool(p.value)
            elif p.name == 'publish_annotations':
                self.publish_annotations = bool(p.value)
            elif p.name == 'publish_focus':
                self.publish_focus = bool(p.value)
            # Control tunables (live)
            elif p.name == 'zeta':
                self.zeta = float(p.value)
            elif p.name == 'v_max':
                self.v_max = float(p.value)
            elif p.name == 'max_distance_error':
                self.max_distance_error = float(p.value)
            elif p.name == 'focus_axis_sign':
                self.focus_axis_sign = float(p.value)
            elif p.name == 'focus_tolerance_m':
                self.focus_tolerance_m = float(p.value)
            elif p.name == 'deriv_tau':
                self.deriv_tau = float(p.value)
            elif p.name == 'fit_enabled':
                self.fit_enabled = bool(p.value)
            elif p.name == 'fit_peak_fraction':
                self.fit_peak_fraction = float(p.value)
            elif p.name == 'fit_min_points':
                self.fit_min_points = int(p.value)
            elif p.name in ('sphere_mass', 'sphere_radius', 'fluid_viscosity'):
                setattr(self, p.name, float(p.value))
                self.linear_drag = (6.0 * math.pi * self.fluid_viscosity
                                    * self.sphere_radius)
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = AutofocusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
