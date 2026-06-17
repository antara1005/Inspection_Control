"""The simulation world: surface + camera + plant + controllers.

``step(dt)`` performs one control tick exactly like the real stack:

1. Raycast the camera against the surface -> focal distance + normal (the "sensor").
2. Run each enabled controller -> forces/torques (the wrench publishers).
3. Sum them with the manual teleop wrench and integrate the admittance plant.

Keeping this loop tiny and synchronous makes the dynamics easy to reason about; the
single-threaded order mirrors the nodes' executor processing one measurement per tick.
"""

from __future__ import annotations

import math

import numpy as np

from .camera import Camera2D, RayHit
from .controllers import AutofocusController, OrientationController
from .dynamics import AdmittancePlant
from .shapes import make_shape


class World:
    def __init__(self, shape_name: str = "sine", shape_kwargs: dict | None = None):
        self.surface = make_shape(shape_name, **(shape_kwargs or {}))
        self.shape_name = shape_name
        self.camera = Camera2D()
        self.plant = AdmittancePlant()
        self.orientation = OrientationController()
        self.autofocus = AutofocusController()

        # Manual teleop wrench (set by keyboard), summed like /teleop/wrench_cmds.
        self.teleop_force = np.zeros(2, dtype=float)
        self.teleop_torque = 0.0

        # White sensor noise injected into the measurement (controllers + viz see
        # the noisy values). std in metres / radians; 0 disables.
        self.distance_noise_std = 0.0
        self.normal_noise_std = 0.0
        self._rng = np.random.default_rng()

        self.last_hit = None

    def set_shape(self, shape_name: str, **kwargs):
        self.surface = make_shape(shape_name, **kwargs)
        self.shape_name = shape_name

    def reset(self):
        self.camera = Camera2D()
        self.plant.reset()
        self.teleop_force[:] = 0.0
        self.teleop_torque = 0.0
        self.autofocus.enabled = False

    def _apply_noise(self, ray: RayHit) -> RayHit:
        """Return a measurement copy with white noise on distance + normal angle."""
        if not ray.hit or (self.distance_noise_std <= 0.0
                           and self.normal_noise_std <= 0.0):
            return ray

        rng = self._rng
        d = ray.distance
        if self.distance_noise_std > 0.0:
            d = max(0.0, d + rng.normal(0.0, self.distance_noise_std))

        n = ray.normal
        if self.normal_noise_std > 0.0:
            dphi = rng.normal(0.0, self.normal_noise_std)   # rotate the normal
            c, s = math.cos(dphi), math.sin(dphi)
            n = np.array([c * n[0] - s * n[1], s * n[0] + c * n[1]])

        # Recompute the hit point along the optical axis so the drawn ray stays
        # straight while its endpoint reflects the noisy distance.
        point = self.camera.pos + d * self.camera.optical_axis
        return RayHit(hit=True, point=point, distance=d, normal=n,
                      tangent=np.array([-n[1], n[0]]))

    def step(self, dt: float):
        # 1. Sense (raycast, then inject white measurement noise).
        ray = self._apply_noise(self.camera.cast_ray(self.surface))
        self.last_hit = ray

        # 2. Controllers -> wrenches (summed, as in the admittance node).
        f_total = np.array(self.teleop_force, dtype=float)
        tau_total = float(self.teleop_torque)

        f_o, tau_o = self.orientation.compute(self.camera, self.plant, ray, dt)
        f_a, tau_a = self.autofocus.compute(self.camera, self.plant, ray, dt)
        f_total = f_total + f_o + f_a
        tau_total += tau_o + tau_a

        # 3. Integrate the virtual mass and update the pose.
        self.plant.step(self.camera, f_total, tau_total, dt)
        return ray
