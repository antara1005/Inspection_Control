"""Ported control laws: orientation alignment and known-peak autofocus.

Both mirror the real ROS nodes and output **generalized efforts** ``(Q_a, Q_d, Q_phi)``
on the coupled :class:`~sim2d.dynamics.PendulumPlant` coordinates (tangential slide,
standoff, swing) — orientation owns the swing torque, autofocus the standoff force.

* :class:`OrientationController` — applies the theta-error torque about the surface
  point (parallel-axis pivot plant) and converts it to an equivalent camera wrench via
  Newton-Euler (``orientation_control_node.py``).

* :class:`AutofocusController` — drives the focal distance to a known true peak with a
  pole-placement PD force along the optical axis (``autofocus_node.py``).

Transient-response improvements over the original PID design
-------------------------------------------------------------
1. **Underdamped poles allowed** (``zeta ≥ 0.5`` instead of ``≥ 1.0``).
   For ``zeta = 0.8`` the closed-loop rise time is ~2× faster than critically
   damped (``zeta = 1``), at the cost of ≈ 5 % overshoot.  ``Kp = I_A·ωₙ²``
   and ``Kd = 2·I_A·ζ·ωₙ − b`` are always real even when the poles are complex
   conjugates, so the gain formulas are unchanged.

2. **Real integral pole for underdamped PID**.  Setting ``p3 = integral_alpha·p2``
   makes ``p3`` complex when ``ζ < 1``.  Instead the integral pole is pinned to
   the real axis at ``p3 = −integral_alpha·ζ·ωₙ``, giving all-real gains
   regardless of ``zeta``.

3. **Back-calculation anti-windup** replaces the binary freeze.  The integrand
   gains a correction ``(τ_clipped − τ_raw) / Tt``.  When the output saturates,
   this term actively reduces the integrator state instead of simply halting; when
   unsaturated the correction is zero and integration is normal.  ``Tt`` defaults
   to ``1 / (ζ·ωₙ)`` (pole-placed tracking constant).

4. **Integral deadzone** (``integral_deadzone``, default ``radians(15)``).
   During large-error transients the controller runs pure PD at maximum authority;
   the integral only activates once ``|e| < integral_deadzone``.  This removes
   integrator lag from the approach phase entirely.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- #
# Gain helpers
# --------------------------------------------------------------------------- #

def pole_placement_poles(m: float, c: float, effort_max: float, max_err: float,
                         zeta: float):
    """Return the two closed-loop poles ``(p1, p2)`` and ``omega_n``.

    ``omega_n`` is sized by the effort/speed budget and the error scale, identical
    to ``autofocus_node._pd_gains`` and the orientation node.

    Unlike the original, ``zeta`` is clamped to ``[0.5, ∞)`` so underdamped
    (faster) responses are allowed.  For ``ζ ≥ 1`` the poles are real; for
    ``ζ < 1`` they are complex conjugates — but the PD gains derived from their
    sum and product are always real.
    """
    m = max(m, 1e-9)
    omega_n = math.sqrt(max(effort_max, 1e-12) / (m * max(max_err, 1e-6)))
    zeta = max(zeta, 0.5)  # allow underdamped; 0.5 ≤ ζ is a reasonable floor
    if zeta >= 1.0:
        root = math.sqrt(zeta ** 2 - 1.0)
        p1 = -zeta * omega_n + omega_n * root
        p2 = -zeta * omega_n - omega_n * root
    else:
        # Complex conjugate pair; callers that need real gains use sum/product.
        imag = omega_n * math.sqrt(1.0 - zeta ** 2)
        p1 = complex(-zeta * omega_n, imag)
        p2 = complex(-zeta * omega_n, -imag)
    return p1, p2, omega_n


def pole_placement_pd(m: float, c: float, effort_max: float, max_err: float,
                      zeta: float):
    """PD gains for ``m·q̈ + c·q̇ = u`` (``u = Kp·err + Kd·err_rate``).

    ``Kp = m·p1·p2 = m·ωₙ²``, ``Kd = −m(p1+p2) − c = 2mζωₙ − c``.
    Both are always real (the imaginary parts of the complex-conjugate poles
    cancel), so no ``.real`` cast produces numerical error.
    """
    p1, p2, omega_n = pole_placement_poles(m, c, effort_max, max_err, zeta)
    # p1*p2 = omega_n^2 and p1+p2 = -2*zeta*omega_n — real for any zeta.
    kp = m * omega_n ** 2
    kd = 2.0 * m * max(zeta, 0.5) * omega_n - c
    return kp, kd


def _pid_gains_3pole(I_A: float, b: float, omega_n: float, zeta: float,
                     integral_alpha: float):
    """PID gains for a 3-pole system with a real integral pole.

    Dominant poles: ``p1,p2 = −ζωₙ ± jωₙ√(1−ζ²)`` (or real for ``ζ≥1``).
    Integral pole:  ``p3 = −integral_alpha · ζ · ωₙ``  (always real).

    Characteristic polynomial ``(s−p1)(s−p2)(s−p3)``:
      s³ + (2ζωₙ + αζωₙ)s² + (ωₙ² + 2αζ²ωₙ²)s + αζωₙ³

    Controller: ``u = Kp·e + Ki·∫e + Kd·ė``
      Kp = I_A·(ωₙ² + 2αζ²ωₙ²) − b·αζωₙ / (... see below)

    The standard matching gives (with ``σ = αζωₙ`` for the integral pole):
      Kp = I_A·(p1p2 + (p1+p2)·p3) = I_A·(ωₙ² − 2ζωₙ·(−σ)) = I_A·ωₙ²(1+2αζ²)
      Ki = −I_A·p1·p2·p3             = I_A·ωₙ²·σ = I_A·αζωₙ³
      Kd = −I_A·(p1+p2+p3) − b       = I_A·ωₙ(2ζ+αζ) − b = I_A·ζωₙ(2+α) − b
    """
    zeta = max(zeta, 0.5)
    sigma = integral_alpha * zeta * omega_n      # magnitude of integral pole
    kp = I_A * omega_n ** 2 * (1.0 + 2.0 * integral_alpha * zeta ** 2)
    ki = I_A * integral_alpha * zeta * omega_n ** 3
    kd = I_A * zeta * omega_n * (2.0 + integral_alpha) - b
    return kp, ki, kd, sigma


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
    coordinate** of the coupled pendulum plant.

    Gains follow ``orientation_control_node.py`` exactly: pole-placement on the pivot
    plant ``I_A·φ̈ + b·φ̇ = τ``.  ``controller_type``:

    * **PD**  — ``Kp = I_A·ωₙ²``, ``Kd = 2I_Aζωₙ − b``, ``Ki = 0``.
    * **PID** — real integral pole at ``p3 = −integral_alpha·ζ·ωₙ``, back-calculation
      anti-windup, integral deadzone.  See module docstring for the redesign rationale.

    Transient-response tuning levers vs the original
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``zeta`` — lower to ``0.7–0.8`` for ~2× faster rise (slight overshoot).
    * ``integral_deadzone`` — pure PD during large transients; integral engages
      only for final regulation within this window (rad).
    * ``Tt`` (``None`` → auto) — back-calculation tracking time constant (s).
      Smaller → faster integrator wind-down during saturation.

    Tracks ``φ_ref = angle(−n) + delta``.  ``compute`` returns ``(Q_a, Q_d, Q_phi)``.
    """

    def __init__(self, zeta: float = 0.8, v_max: float = 0.5,
                 theta_max: float = math.radians(30.0),
                 controller_type: str = "PID",
                 integral_alpha: float = 3.0,
                 ie_clamp: float = math.radians(8.0),
                 integral_deadzone: float = math.radians(15.0),
                 Tt: float | None = None):
        self.enabled = False
        self.zeta = zeta
        self.v_max = v_max
        self.theta_max = theta_max
        self.controller_type = controller_type
        self.integral_alpha = integral_alpha
        self.ie_clamp = ie_clamp
        # Integral only accumulates when |e| is inside this window (rad).
        # Set to 0 to disable deadzone (always integrate).
        self.integral_deadzone = integral_deadzone
        # Back-calculation tracking time constant.  None → auto (1 / (ζ·ωₙ)).
        self.Tt = Tt
        self.delta = 0.0
        self.int_e = 0.0
        self.angle_error = 0.0

    def reset_integral(self):
        self.int_e = 0.0

    def compute(self, camera, plant, ray, dt):
        """Return generalized efforts ``(Q_a, Q_d, Q_phi)``."""
        if not self.enabled or not ray.hit:
            self.angle_error = 0.0
            self.int_e = 0.0
            return 0.0, 0.0, 0.0

        e = signed_angle(camera.optical_axis, -ray.normal) + self.delta
        e = math.atan2(math.sin(e), math.cos(e))
        self.angle_error = e

        d = float(ray.distance)
        I_A = plant.inertia + plant.mass * d * d
        b = plant.linear_drag * d * d + plant.angular_drag
        tau_max = plant.linear_drag * d * self.v_max
        _, _, omega_n = pole_placement_poles(I_A, b, tau_max, self.theta_max,
                                             self.zeta)

        if self.controller_type.upper() == "PID":
            kp, ki, kd, sigma = _pid_gains_3pole(I_A, b, omega_n, self.zeta,
                                                  self.integral_alpha)
            # Derivative uses the clean plant angular rate (no extra smoothing needed).
            tau_raw = kp * e + ki * self.int_e + kd * (-plant.ang_vel)
            tau = float(np.clip(tau_raw, -tau_max, tau_max))

            # Back-calculation anti-windup: when saturated, the tracking error
            # (tau − tau_raw) feeds back into the integrand and actively unwinds
            # the integrator, unlike a binary freeze.
            if ki > 1e-12 and dt > 0.0:
                Tt = self.Tt if self.Tt is not None else 1.0 / max(
                    self.zeta * omega_n, 1e-6)
                tracking_correction = (tau - tau_raw) / max(Tt, 1e-9)

                # Integral deadzone: skip accumulation for large errors so the
                # approach phase is pure PD at full authority.
                if abs(e) < self.integral_deadzone:
                    self.int_e += (e + tracking_correction) * dt
                else:
                    # Still apply tracking correction to prevent windup if the
                    # deadzone is active during saturation.
                    if tracking_correction != 0.0:
                        self.int_e += tracking_correction * dt

                self.int_e = float(np.clip(self.int_e, -self.ie_clamp,
                                           self.ie_clamp))
        else:
            # Pure PD
            kp = I_A * omega_n ** 2
            kd = 2.0 * I_A * max(self.zeta, 0.5) * omega_n - b
            tau_raw = kp * e + kd * (-plant.ang_vel)
            tau = float(np.clip(tau_raw, -tau_max, tau_max))
            self.int_e = 0.0

        return 0.0, 0.0, tau


# --------------------------------------------------------------------------- #
# Autofocus
# --------------------------------------------------------------------------- #
class AutofocusController:
    """Drives the focal distance to a *known* true peak ``d_focus`` with a
    pole-placement PD force along the optical axis. No sweep/record/fit — the
    controller is simply told where the focus peak is."""

    def __init__(self, d_focus: float = 1.2, sigma: float = 0.35,
                 zeta: float = 0.8, v_max: float = 0.05,
                 max_distance_error: float = 0.1, tolerance: float = 0.002):
        self.enabled = False
        self.d_focus = d_focus
        self.sigma = sigma
        self.zeta = zeta
        self.v_max = v_max
        self.max_distance_error = max_distance_error
        self.tolerance = tolerance
        self._dist_rate = _RateEstimator(tau=0.02)   # tighter tau for quicker D response
        self.focus_value = float("nan")

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if not self.enabled:
            self._dist_rate.reset()
        return self.enabled

    def focus_at(self, d: float) -> float:
        """Sharpness vs focal distance — a Gaussian peak at the true focus."""
        return math.exp(-((d - self.d_focus) / self.sigma) ** 2)

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
