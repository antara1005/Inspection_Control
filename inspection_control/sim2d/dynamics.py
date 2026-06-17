"""Coupled pivot-referenced pendulum plant — the dynamics the camera rides on.

This replaces the earlier free **point-mass** admittance plant. The camera (bob, mass
``m``, inertia ``I_B``) hangs at standoff ``d`` from the surface contact point ``A``
(the pivot) along the optical axis. We integrate the *coupled* dynamics in the
generalized coordinates

    q = (a, d, phi)     a   = tangential slide of the pivot along the surface tangent t
                        d   = standoff (camera <-> surface)
                        phi = swing (optical-axis angle)

so the controllers each apply a generalized effort on one coordinate (orientation -> a
torque on ``phi``, autofocus -> a force on ``d``, teleop -> surface-frame translation),
and drag + the centrifugal/Coriolis coupling fall out of the equations of motion
**dissipatively** — no Newton-Euler feedforward, no algebraic loop through drag.

Route (b) "locally-flat + re-project": each ``step`` integrates on the local surface
frame ``{t, n}`` reported by the raycast, and the *next* tick's raycast snaps the pivot
back onto the true (curved, noisy) surface.

**Decoupled** pendulum (one clean second-order system per coordinate), coupled only
through the standoff-dependent swing inertia ``I_A = I_B + m·d²`` and the centrifugal /
Coriolis terms — see ``docs/coupled-pendulum-dynamics.md``::

    a : m·ä   = Q_a − c·ȧ
    d : m·d̈   = Q_d − c·ḋ + m·d·φ̇²                       (centrifugal)
    φ : I_A·φ̈ = Q_φ − (c·d² + c_ang)·φ̇ − 2·m·d·ḋ·φ̇      (Coriolis)

Drag is **relative to the pivot** and diagonal: the swing sees only ``c·d²·φ̇`` (the
reference's ``b = 6πμR·d²``), not the bob's absolute velocity. The inertial slide↔swing
cross-coupling is deliberately omitted so teleop translation never torques the swing —
each coordinate is independently dissipative, so the plant is passive and stable.
"""

from __future__ import annotations

import math

import numpy as np

# Below this |t x û| the optical axis is ~tangent to the surface (grazing); the pivot
# chart is singular, so fall back to a free-body step.
_GRAZE_EPS = 1e-3


class PendulumPlant:
    def __init__(self, mass: float = 2.5, radius: float = 0.65,
                 viscosity: float = 1.0):
        self.set_inertia_and_drag(mass, radius, viscosity)
        self.lin_vel = np.zeros(2, dtype=float)   # world frame, m/s
        self.ang_vel = 0.0                          # rad/s

    def set_inertia_and_drag(self, mass: float, radius: float, viscosity: float):
        self.mass = float(mass)
        self.radius = float(radius)
        self.viscosity = float(viscosity)
        self.inertia = (2.0 / 5.0) * self.mass * self.radius ** 2   # I_B
        self.linear_drag = 6.0 * math.pi * self.viscosity * self.radius        # c
        self.angular_drag = 2.4 * math.pi * self.viscosity * self.radius ** 3  # c_ang

    def reset(self):
        self.lin_vel[:] = 0.0
        self.ang_vel = 0.0

    # ------------------------------------------------------------------ #
    def step(self, camera, ray, q_a: float, q_d: float, q_phi: float, dt: float,
             pivot: bool = True):
        """Integrate one tick from generalized efforts ``(q_a, q_d, q_phi)``.

        With ``pivot`` (orientation engaged): pendulum about the surface contact —
        ``q_a`` slides the pivot along the tangent, ``q_d`` is the standoff force,
        ``q_phi`` the swing torque. Without ``pivot`` (or off-surface / grazing): a
        **free body in camera coordinates** — ``q_a`` is camera-right, ``q_d`` along the
        optical axis, ``q_phi`` a torque about the camera COM.
        """
        u = camera.optical_axis                      # û = (cosφ, sinφ)
        u_perp = np.array([-u[1], u[0]])             # û⊥

        if not pivot or not ray.hit:
            return self._free_step(camera, u, u_perp, q_a, q_d, q_phi, dt)

        t = ray.tangent                              # surface tangent (unit)
        a1 = float(t @ u)
        a2 = float(t @ u_perp)
        d = float(ray.distance)

        # Decompose world velocity (v_c, ω) into generalized velocities (ȧ, ḋ, φ̇):
        #   v_c = ȧ·t − ḋ·û − d·φ̇·û⊥  ⇒  [t, −û]·[ȧ; ḋ] = v_c + d·ω·û⊥
        basis = np.column_stack([t, -u])
        if abs(np.linalg.det(basis)) < _GRAZE_EPS:   # grazing: chart singular
            return self._free_step(camera, u, u_perp, q_a, q_d, q_phi, dt)
        phidot = self.ang_vel
        rhs = self.lin_vel + d * phidot * u_perp
        adot, ddot = np.linalg.solve(basis, rhs)

        m, c, I_B, c_ang = self.mass, self.linear_drag, self.inertia, self.angular_drag
        I_A = I_B + m * d * d                                # parallel-axis inertia

        # Decoupled pendulum dynamics: one clean second-order system per coordinate,
        # coupled only through the standoff-dependent swing inertia I_A(d) and the
        # centrifugal / Coriolis terms. Drag is **relative to the pivot** and diagonal
        # (the swing sees c·d²·φ̇ = the reference's b = 6πμR·d²). This deliberately
        # drops the slide↔swing cross-coupling so teleop translation never torques the
        # swing — each DOF is dissipative, so the plant stays passive and stable.
        #   a : m·ä  = Q_a − c·ȧ
        #   d : m·d̈  = Q_d − c·ḋ + m·d·φ̇²            (centrifugal: spin flings standoff)
        #   φ : I_A·φ̈ = Q_φ − (c·d² + c_ang)·φ̇ − 2·m·d·ḋ·φ̇   (Coriolis)
        addot = (q_a - c * adot) / m
        dddot = (q_d - c * ddot + m * d * phidot * phidot) / m
        phiddot = (q_phi - (c * d * d + c_ang) * phidot
                   - 2.0 * m * d * ddot * phidot) / I_A

        adot += addot * dt
        ddot += dddot * dt
        phidot += phiddot * dt

        # Recompose world velocity and integrate the pose.
        self.lin_vel = adot * t - ddot * u - d * phidot * u_perp
        self.ang_vel = phidot
        camera.pos += self.lin_vel * dt
        camera.theta += self.ang_vel * dt

    # ------------------------------------------------------------------ #
    def _free_step(self, camera, u, u_perp, q_a, q_d, q_phi, dt):
        """Off-surface / grazing fallback: free point mass.

        Map the generalized efforts to world forces with no pivot coupling — tangential
        effort along camera-right, standoff effort along the optical axis — so teleop
        still flies the camera when there is no contact.
        """
        right = np.array([u[1], -u[0]])
        F = q_a * right + q_d * (-u)          # +q_d increases d (moves away along −û)
        m, c, I_B, c_ang = self.mass, self.linear_drag, self.inertia, self.angular_drag
        self.lin_vel = self.lin_vel + (F - c * self.lin_vel) / max(m, 1e-9) * dt
        self.ang_vel = self.ang_vel + (q_phi - c_ang * self.ang_vel) / max(I_B, 1e-9) * dt
        camera.pos += self.lin_vel * dt
        camera.theta += self.ang_vel * dt
