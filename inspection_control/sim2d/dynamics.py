"""Admittance plant — the virtual mass the camera rides on.

This is the 2D reduction of ``admittance_control_node.py``: a sphere of mass ``m`` and
radius ``r`` in a viscous fluid (viscosity ``mu``) with Stokes drag. External wrenches
(teleop, orientation, autofocus) are summed and integrated into velocity, then pose:

    linear:   m·v̇  = F_total − c·v,     c     = 6·π·μ·r
    angular:  I·ω̇  = τ_total − c_ang·ω,  I     = (2/5)·m·r²,  c_ang = 2.4·π·μ·r³

Linear state is 2D ``(x, y)``; angular state is the scalar heading ``θ`` (the single
out-of-plane rotation), mirroring the node's ``update_inertia_and_drag`` and the
force/torque summation in ``publish_twist``.
"""

from __future__ import annotations

import math

import numpy as np


class AdmittancePlant:
    def __init__(self, mass: float = 2.5, radius: float = 0.65,
                 viscosity: float = 1.0):
        self.set_inertia_and_drag(mass, radius, viscosity)
        self.lin_vel = np.zeros(2, dtype=float)   # m/s
        self.ang_vel = 0.0                         # rad/s

    def set_inertia_and_drag(self, mass: float, radius: float, viscosity: float):
        self.mass = float(mass)
        self.radius = float(radius)
        self.viscosity = float(viscosity)
        self.inertia = (2.0 / 5.0) * self.mass * self.radius ** 2
        self.linear_drag = 6.0 * math.pi * self.viscosity * self.radius
        self.angular_drag = 2.4 * math.pi * self.viscosity * self.radius ** 3

    def reset(self):
        self.lin_vel[:] = 0.0
        self.ang_vel = 0.0

    def step(self, camera, force_xy, torque, dt: float):
        """Integrate one tick and write the new pose back into ``camera``.

        ``force_xy`` is the summed external force (2-vector, world frame); ``torque``
        the summed scalar torque (about +z). Drag is added internally.
        """
        f = np.asarray(force_xy, dtype=float)

        lin_acc = (f - self.linear_drag * self.lin_vel) / max(self.mass, 1e-9)
        ang_acc = (float(torque) - self.angular_drag * self.ang_vel) / max(self.inertia, 1e-9)

        self.lin_vel += lin_acc * dt
        self.ang_vel += ang_acc * dt

        camera.pos += self.lin_vel * dt
        camera.theta += self.ang_vel * dt
        return lin_acc, ang_acc
