"""Ported control laws: orientation alignment and known-peak autofocus.

Both mirror the real ROS nodes and output **generalized efforts** ``(Q_a, Q_d, Q_phi)``
on the coupled :class:`~sim2d.dynamics.PendulumPlant` coordinates (tangential slide,
standoff, swing) — orientation owns the swing torque, autofocus the standoff force.

* :class:`OrientationController` — applies the theta-error torque about the surface
  point (parallel-axis pivot plant) and converts it to an equivalent camera wrench via
  Newton-Euler (``orientation_control_node.py``).

* :class:`AutofocusController` — drives the focal distance to a known true peak with a
  pole-placement PD force along the optical axis (``autofocus_node.py``).
"""

from __future__ import annotations

import math

import numpy as np


def pole_placement_poles(m: float, c: float, effort_max: float, max_err: float,
                         zeta: float):
    """Return the two real closed-loop poles ``(p1, p2)`` and ``omega_n``.

    ``omega_n`` is sized by the effort/speed budget (``effort_max = c·v_max``) and the
    error scale ``max_err`` — identical to ``autofocus_node._pd_gains`` and the
    orientation node. Poles at ``-ζωₙ ± ωₙ√(ζ²−1)`` (real / non-oscillatory for
    ``ζ ≥ 1``).
    """
    m = max(m, 1e-9)
    omega_n = math.sqrt(max(effort_max, 1e-12) / (m * max(max_err, 1e-6)))
    zeta = max(zeta, 1.0)
    root = math.sqrt(zeta ** 2 - 1.0)
    p1 = -zeta * omega_n + omega_n * root
    p2 = -zeta * omega_n - omega_n * root
    return p1, p2, omega_n


def pole_placement_pd(m: float, c: float, effort_max: float, max_err: float,
                      zeta: float):
    """PD gains for ``m·q̈ + c·q̇ = u`` (``u = Kp·err + Kd·err_rate``).

    ``Kp = m·p1·p2``, ``Kd = −m(p1+p2) − c``. ``Kd`` may be negative (it cancels
    excess natural drag), but the *total* closed-loop damping ``c + Kd = 2ζωₙ·m`` is
    always positive — matching ``autofocus_node`` / ``orientation_control_node`` with
    no clamping (the decoupled plant makes this stable; see docs §5).
    """
    p1, p2, _ = pole_placement_poles(m, c, effort_max, max_err, zeta)
    return m * (p1 * p2), -m * (p1 + p2) - c


def signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle (rad) to rotate unit vector ``a`` onto unit vector ``b``."""
    dot = float(np.dot(a, b))
    cross = float(a[0] * b[1] - a[1] * b[0])
    return math.atan2(cross, dot)


class _RateEstimator:
    """EMA-smoothed finite-difference derivative (mirrors the nodes' ``deriv_tau``)."""

    def __init__(self, tau: float = 0.05):
        self.tau = tau
        self._last = None
        self.rate = 0.0

    def update(self, value: float, dt: float) -> float:
        if self._last is not None and dt > 1e-6:
            raw = (value - self._last) / dt
            alpha = 1.0 - math.exp(-dt / max(1e-6, self.tau))
            self.rate += alpha * (raw - self.rate)
        self._last = value
        return self.rate

    def reset(self):
        self._last = None
        self.rate = 0.0


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
class OrientationController:
    """Swing the optical axis onto the surface normal with a **pure torque on the swing
    coordinate** of the coupled pendulum plant — a genuine torque about the surface
    pivot, no Newton-Euler feedforward.

    Gains follow ``orientation_control_node.py`` exactly: pole-placement on the pivot
    plant ``I_A·φ̈ + b·φ̇ = τ`` (``I_A = I_B + m·d²``, ``b = c·d² + c_ang``), with poles
    ``p1,p2 = −ζωₙ ± ωₙ√(ζ²−1)`` and ``ωₙ`` from the torque budget
    ``τ_max = c·d·v_max`` and the error scale ``theta_max``. ``controller_type``:

    * **PD**  — ``Kp = I_A·p1·p2``, ``Kd = −I_A(p1+p2) − b``, ``Ki = 0``.
    * **PID** — adds a slow integral pole ``p3 = integral_alpha·p2``:
      ``Kp = I_A(p1p2+p1p3+p2p3)``, ``Ki = −I_A·p1p2p3``, ``Kd = −I_A(p1+p2+p3) − b``,
      with conditional anti-windup integration and an ``ie_clamp`` integral limit.

    Tracks ``φ_ref = angle(−n) + delta`` (``delta`` = teleop reference offset to pivot
    the view about the target). ``compute`` returns ``(Q_a, Q_d, Q_phi) = (0, 0, τ)``.
    """

    def __init__(self, zeta: float = 1.0, v_max: float = 0.5,
                 theta_max: float = math.radians(30.0),
                 controller_type: str = "PD", integral_alpha: float = 5.0,
                 ie_clamp: float = math.radians(8.0), anti_windup: bool = True):
        self.enabled = False
        self.zeta = zeta
        self.v_max = v_max                # speed budget (m/s) -> torque budget c·d·v_max
        self.theta_max = theta_max        # angular error scale (rad)
        self.controller_type = controller_type   # 'PD' or 'PID'
        self.integral_alpha = integral_alpha      # integral pole = alpha·p2
        self.ie_clamp = ie_clamp          # integral-of-error clamp (rad·s)
        self.anti_windup = anti_windup
        self.delta = 0.0                  # reference swing offset about the target (rad)
        self.int_e = 0.0                  # integral state
        self.angle_error = 0.0            # telemetry for the HUD

    def reset_integral(self):
        self.int_e = 0.0

    def compute(self, camera, plant, ray, dt):
        """Return generalized efforts ``(Q_a, Q_d, Q_phi)``."""
        if not self.enabled or not ray.hit:
            self.angle_error = 0.0
            self.int_e = 0.0
            return 0.0, 0.0, 0.0

        # Error to the (offset) reference axis: angle(−n) + delta − phi, wrapped.
        e = signed_angle(camera.optical_axis, -ray.normal) + self.delta
        e = math.atan2(math.sin(e), math.cos(e))
        self.angle_error = e

        d = float(ray.distance)
        I_A = plant.inertia + plant.mass * d * d
        b = plant.linear_drag * d * d + plant.angular_drag   # pivot swing drag
        tau_max = plant.linear_drag * d * self.v_max
        p1, p2, _ = pole_placement_poles(I_A, b, tau_max, self.theta_max, self.zeta)

        if self.controller_type.upper() == "PID":
            p3 = self.integral_alpha * p2
            kp = I_A * (p1 * p2 + p1 * p3 + p2 * p3)
            ki = -I_A * (p1 * p2 * p3)
            kd = -I_A * (p1 + p2 + p3) - b
        else:
            kp = I_A * (p1 * p2)
            ki = 0.0
            kd = -I_A * (p1 + p2) - b

        # err_rate = d/dt(theta_ref − theta) ≈ −ω (uses the clean plant rate, not a
        # differentiated noisy measurement).
        tau_raw = kp * e + ki * self.int_e + kd * (-plant.ang_vel)
        tau = float(np.clip(tau_raw, -tau_max, tau_max))

        # Integrate with conditional anti-windup + clamp (mirrors the node).
        if ki > 1e-12 and dt > 0.0:
            saturated = abs(tau_raw) > tau_max
            if (not self.anti_windup) or (not saturated) or \
                    (tau_raw > 0 and e < 0) or (tau_raw < 0 and e > 0):
                self.int_e += e * dt
            self.int_e = float(np.clip(self.int_e, -self.ie_clamp, self.ie_clamp))

        return 0.0, 0.0, tau


# --------------------------------------------------------------------------- #
# Autofocus
# --------------------------------------------------------------------------- #
class AutofocusController:
    """Drives the focal distance to a *known* true peak ``d_focus`` with a
    pole-placement PD force along the optical axis. No sweep/record/fit — the
    controller is simply told where the focus peak is."""

    def __init__(self, d_focus: float = 1.2, sigma: float = 0.35,
                 zeta: float = 1.0, v_max: float = 0.05,
                 max_distance_error: float = 0.1, tolerance: float = 0.002):
        self.enabled = False                  # 'f' toggles the drive
        self.d_focus = d_focus                # known true focus distance (target)
        self.sigma = sigma                    # focus curve width (HUD readout only)
        self.zeta = zeta
        self.v_max = v_max
        self.max_distance_error = max_distance_error
        self.tolerance = tolerance
        self._dist_rate = _RateEstimator(tau=0.05)
        self.focus_value = float("nan")

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if not self.enabled:
            self._dist_rate.reset()
        return self.enabled

    # -- synthetic camera focus metric (display only) ---------------------- #
    def focus_at(self, d: float) -> float:
        """Sharpness vs focal distance — a Gaussian peak at the true focus."""
        return math.exp(-((d - self.d_focus) / self.sigma) ** 2)

    # -- control tick ------------------------------------------------------ #
    def compute(self, camera, plant, ray, dt):
        """Return generalized efforts ``(Q_a, Q_d, Q_phi)`` — a standoff force on ``d``."""
        if not ray.hit:
            self.focus_value = float("nan")
            return 0.0, 0.0, 0.0

        d = ray.distance
        self.focus_value = self.focus_at(d)

        if not self.enabled:
            self._dist_rate.reset()
            return 0.0, 0.0, 0.0

        err = d - self.d_focus
        if abs(err) <= self.tolerance:
            return 0.0, 0.0, 0.0

        d_rate = self._dist_rate.update(d, dt)
        f_max = plant.linear_drag * self.v_max
        kp, kd = pole_placement_pd(plant.mass, plant.linear_drag,
                                   f_max, self.max_distance_error, self.zeta)
        u = float(np.clip(kp * err + kd * d_rate, -f_max, f_max))
        # Q_d is conjugate to d; +Q_d increases d, so drive d→d_focus with −u.
        return 0.0, -u, 0.0
