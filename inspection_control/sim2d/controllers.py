"""Ported control laws: orientation alignment and peak-hold autofocus.

Both mirror the real ROS nodes and output forces/torques that are summed into the
:class:`~sim2d.dynamics.AdmittancePlant` — exactly as the nodes publish ``WrenchStamped``
into the admittance node.

* :class:`OrientationController` — applies the theta-error torque about the surface
  point (parallel-axis pivot plant) and converts it to an equivalent camera wrench via
  Newton-Euler (``orientation_control_node.py``).

* :class:`AutofocusController` — drives the focal distance to a known true peak with a
  pole-placement PD force along the optical axis (``autofocus_node.py``).
"""

from __future__ import annotations

import math

import numpy as np


def pole_placement_pd(m: float, c: float, effort_max: float, max_err: float,
                      zeta: float):
    """Real-pole PD gains for the 1-DOF plant  ``m·q̈ + c·q̇ = u``.

    ``omega_n`` is sized by the effort/speed budget (``effort_max = c·v_max``) and the
    error scale ``max_err`` — identical to ``autofocus_node._pd_gains`` and the
    orientation node. Returns ``(Kp, Kd)`` for ``u = Kp·err + Kd·err_rate`` with poles
    placed at ``-ζωₙ ± ωₙ√(ζ²−1)`` (non-oscillatory for ``ζ ≥ 1``).
    """
    m = max(m, 1e-9)
    omega_n = math.sqrt(max(effort_max, 1e-12) / (m * max(max_err, 1e-6)))
    zeta = max(zeta, 1.0)
    root = math.sqrt(zeta ** 2 - 1.0)
    p1 = -zeta * omega_n + omega_n * root
    p2 = -zeta * omega_n - omega_n * root
    kp = m * (p1 * p2)
    kd = -m * (p1 + p2) - c
    return kp, kd


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
    """Aligns the optical axis to the surface normal by applying the theta-error
    torque **about the surface point** (a pivot at distance ``d``), then converting
    it — via Newton-Euler — into the equivalent force + torque at the camera that is
    fed to the admittance plant.

    The pivot plant is the parallel-axis reduction used by
    ``orientation_control_node.py``::

        I_A = I_B + m·d²              (inertia about the surface point)
        I_A·θ̈ + (c·d²)·θ̇ = τ_θ      (c·d² = camera linear drag as a pivot moment)

    A pole-placement PD sets ``τ_θ``; Newton-Euler then back-solves the camera wrench
    so that, once the admittance node re-adds drag, the camera+plant swing rigidly
    about the contact point.
    """

    def __init__(self, zeta: float = 1.0, v_max: float = 0.1,
                 theta_max: float = math.radians(30.0)):
        self.enabled = False
        self.zeta = zeta
        self.v_max = v_max            # linear speed budget (m/s) at the camera
        self.theta_max = theta_max    # angular error scale (rad)
        self.angle_error = 0.0        # telemetry for the HUD

    def compute(self, camera, plant, ray, dt):
        """Return the equivalent ``(force_xy, torque)`` at the camera."""
        if not self.enabled or not ray.hit:
            self.angle_error = 0.0
            return np.zeros(2), 0.0

        e = signed_angle(camera.optical_axis, -ray.normal)  # theta_des - theta
        self.angle_error = e

        # Lever arm from the surface contact (pivot A) to the camera (centroid B).
        r = camera.pos - ray.point
        d = float(np.hypot(r[0], r[1]))

        # Plant about the pivot: parallel-axis inertia; the camera's linear drag
        # appears as a moment about A (= -c·d²·ω for pure pivoting).
        inertia_A = plant.inertia + plant.mass * d * d
        force_drag_B = -plant.linear_drag * plant.lin_vel
        moment_drag_A = r[0] * force_drag_B[1] - r[1] * force_drag_B[0]

        # Pole-placement PD torque about the pivot (gains referenced to I_A, c·d²).
        tau_max = plant.linear_drag * d * self.v_max
        kp, kd = pole_placement_pd(inertia_A, plant.linear_drag * d * d,
                                   tau_max, self.theta_max, self.zeta)
        tau_theta = float(np.clip(kp * e + kd * (-plant.ang_vel),
                                  -tau_max, tau_max))

        # Newton-Euler: angular accel about the pivot, then the equivalent camera
        # wrench. force_B/tau_B pre-subtract drag so the admittance plant, which
        # re-adds it, reproduces the pivot motion (a_B = α × r).
        alpha = (moment_drag_A + tau_theta) / inertia_A
        a_B = alpha * np.array([-r[1], r[0]])          # (α ẑ) × r
        force_B = plant.mass * a_B - force_drag_B
        tau_B = plant.inertia * alpha + plant.angular_drag * plant.ang_vel
        return force_B, tau_B


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
        if not ray.hit:
            self.focus_value = float("nan")
            return np.zeros(2), 0.0

        d = ray.distance
        self.focus_value = self.focus_at(d)

        if not self.enabled:
            self._dist_rate.reset()
            return np.zeros(2), 0.0

        err = d - self.d_focus
        if abs(err) <= self.tolerance:
            return np.zeros(2), 0.0

        d_rate = self._dist_rate.update(d, dt)
        f_max = plant.linear_drag * self.v_max
        kp, kd = pole_placement_pd(plant.mass, plant.linear_drag,
                                   f_max, self.max_distance_error, self.zeta)
        u = float(np.clip(kp * err + kd * d_rate, -f_max, f_max))
        return u * camera.optical_axis, 0.0
