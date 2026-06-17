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


def pole_placement_pd(m: float, c: float, effort_max: float, max_err: float,
                      zeta: float, clamp_kd: bool = True):
    """Real-pole PD gains for the 1-DOF plant  ``m·q̈ + c·q̇ = u``.

    ``omega_n`` is sized by the effort/speed budget (``effort_max = c·v_max``) and the
    error scale ``max_err`` — identical to ``autofocus_node._pd_gains`` and the
    orientation node. Returns ``(Kp, Kd)`` for ``u = Kp·err + Kd·err_rate`` with poles
    placed at ``-ζωₙ ± ωₙ√(ζ²−1)`` (non-oscillatory for ``ζ ≥ 1``).

    ``Kd = −m(p1+p2) − c`` goes **negative** when the plant's natural damping ``c``
    already exceeds the target ``2ζωₙ·m`` (a heavily over-damped, drag-dominated
    regime — true here because ``c·d²`` is large). A negative ``Kd`` is anti-damping:
    it cancels real drag and is fragile under the slide/swing coupling, so by default
    we **clamp it to ≥ 0** — the controller then leans on the natural damping instead
    of fighting it (the closed loop stays over-damped and stable). Set
    ``clamp_kd=False`` for the exact (unclamped) placement.
    """
    m = max(m, 1e-9)
    omega_n = math.sqrt(max(effort_max, 1e-12) / (m * max(max_err, 1e-6)))
    zeta = max(zeta, 1.0)
    root = math.sqrt(zeta ** 2 - 1.0)
    p1 = -zeta * omega_n + omega_n * root
    p2 = -zeta * omega_n - omega_n * root
    kp = m * (p1 * p2)
    kd = -m * (p1 + p2) - c
    if clamp_kd:
        kd = max(kd, 0.0)
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
    """Swing the optical axis onto the surface normal by applying a **pure torque on
    the swing coordinate** of the coupled pendulum plant — a genuine torque about the
    surface pivot, no Newton-Euler feedforward.

    The pivot plant (``PendulumPlant``) gives the parallel-axis dynamics
    ``I_A·φ̈ + (c·d² + c_ang)·φ̇ = τ_θ`` with ``I_A = I_B + m·d²``; a pole-placement PD
    sets ``τ_θ`` to track ``φ_ref = angle(−n) + delta``. ``delta`` is a reference swing
    offset (teleop rotation) that lets you pivot the camera about the target.

    ``compute`` returns generalized efforts ``(Q_a, Q_d, Q_phi) = (0, 0, τ_θ)``.
    """

    def __init__(self, zeta: float = 1.0, v_max: float = 0.1,
                 theta_max: float = math.radians(30.0)):
        self.enabled = False
        self.zeta = zeta
        self.v_max = v_max            # linear speed budget (m/s) at the camera
        self.theta_max = theta_max    # angular error scale (rad)
        self.delta = 0.0              # reference swing offset about the target (rad)
        self.angle_error = 0.0        # telemetry for the HUD

    def compute(self, camera, plant, ray, dt):
        """Return generalized efforts ``(Q_a, Q_d, Q_phi)``."""
        if not self.enabled or not ray.hit:
            self.angle_error = 0.0
            return 0.0, 0.0, 0.0

        # Error to the (offset) reference axis: angle(−n) + delta − phi.
        e = signed_angle(camera.optical_axis, -ray.normal) + self.delta
        e = math.atan2(math.sin(e), math.cos(e))          # wrap to (−π, π]
        self.angle_error = e

        d = float(ray.distance)
        inertia_A = plant.inertia + plant.mass * d * d
        # Swing drag about the pivot is c·d² (relative-velocity) plus the camera's own
        # angular drag; place the closed-loop poles against that.
        tau_max = plant.linear_drag * d * self.v_max
        kp, kd = pole_placement_pd(inertia_A,
                                   plant.linear_drag * d * d + plant.angular_drag,
                                   tau_max, self.theta_max, self.zeta)
        tau_theta = float(np.clip(kp * e + kd * (-plant.ang_vel), -tau_max, tau_max))
        return 0.0, 0.0, tau_theta


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
