"""2D camera pose and raycasting against a :class:`~sim2d.shapes.Surface`.

The camera is a point ``p = (x, y)`` with heading ``theta``; its optical axis (the
viewing direction, analogous to the real camera's +z) is ``zhat = (cos θ, sin θ)``.

``cast_ray`` marches the implicit field ``f`` along the optical axis until it changes
sign, then bisects to the crossing. It returns the hit point, the focal distance
``d`` (what the real depth/orientation stack measures), and the surface normal +
tangent at the hit. This is generic over every registered shape — no per-shape ray
code is ever needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RayHit:
    hit: bool = False
    point: np.ndarray = field(default_factory=lambda: np.zeros(2))
    distance: float = float("inf")
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    tangent: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


class Camera2D:
    """A 2D camera with position and heading and an outgoing ray."""

    def __init__(self, x: float = 0.0, y: float = 3.0, theta: float = -np.pi / 2):
        self.pos = np.array([x, y], dtype=float)
        self.theta = float(theta)

    # -- geometry ----------------------------------------------------------- #
    @property
    def optical_axis(self) -> np.ndarray:
        """Unit viewing direction ``ẑ = (cosθ, sinθ)``."""
        return np.array([np.cos(self.theta), np.sin(self.theta)])

    # -- raycasting --------------------------------------------------------- #
    def cast_ray(self, surface, t_max: float = 20.0, step: float = 0.02,
                 bisect_iters: int = 16) -> RayHit:
        """March ``surface.f`` along the optical axis; bisect the first sign flip.

        All sample points are evaluated in a single vectorised ``surface.f`` call
        (every registered shape accepts numpy arrays), replacing the original Python
        while-loop.  ``bisect_iters=16`` gives sub-micrometre accuracy, so 40 was
        already overkill.  Returns a miss if the surface is not crossed within
        ``t_max``.
        """
        p = self.pos
        d = self.optical_axis

        # Batch all march samples: shape (N, 2) -> f values shape (N,).
        ts = np.arange(0.0, t_max + step, step)   # includes t=0 (camera position)
        qs = p + np.outer(ts, d)
        fs = np.asarray(surface.f(qs[:, 0], qs[:, 1]), dtype=float)

        # Find first index where the sign flips (zero treated as positive so we
        # don't stall when the camera sits exactly on the surface).
        signs = np.sign(fs)
        signs[signs == 0.0] = 1.0
        crossings = np.where(np.diff(signs) != 0.0)[0]
        if crossings.size == 0:
            return RayHit(hit=False, distance=float("inf"))

        i = int(crossings[0])
        lo, hi = float(ts[i]), float(ts[i + 1])
        flo = float(fs[i])
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            fm = float(surface.f(*(p + mid * d)))
            if fm == 0.0:
                lo = hi = mid
                break
            if np.sign(fm) != np.sign(flo):
                hi = mid
            else:
                lo, flo = mid, fm

        t_hit = 0.5 * (lo + hi)
        return self._make_hit(surface, p + t_hit * d, t_hit, d)

    @staticmethod
    def _make_hit(surface, point, t_hit, d) -> RayHit:
        n = surface.normal(point[0], point[1])
        # Orient the normal to face the camera (against the ray direction).
        if np.dot(n, d) > 0.0:
            n = -n
        tangent = np.array([-n[1], n[0]])
        return RayHit(hit=True, point=np.asarray(point, dtype=float),
                      distance=float(t_hit), normal=n, tangent=tangent)
