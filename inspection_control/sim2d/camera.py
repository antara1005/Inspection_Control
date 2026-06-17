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
                 bisect_iters: int = 40) -> RayHit:
        """March ``surface.f`` along the optical axis; bisect the first sign flip.

        ``step`` is the coarse march resolution; ``bisect_iters`` refines the root to
        ~``step / 2**iters``. Returns a miss if the surface is not crossed within
        ``t_max`` (e.g. the camera faces away from the object).
        """
        p = self.pos
        d = self.optical_axis

        f0 = float(surface.f(p[0], p[1]))
        t_prev = 0.0
        f_prev = f0
        t = step
        while t <= t_max:
            q = p + t * d
            ft = float(surface.f(q[0], q[1]))
            if f_prev == 0.0:
                return self._make_hit(surface, p + t_prev * d, t_prev, d)
            if np.sign(ft) != np.sign(f_prev):
                # Bracket [t_prev, t] contains a crossing -> bisect.
                lo, hi = t_prev, t
                flo = f_prev
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
            t_prev, f_prev = t, ft
            t += step

        return RayHit(hit=False, distance=float("inf"))

    @staticmethod
    def _make_hit(surface, point, t_hit, d) -> RayHit:
        n = surface.normal(point[0], point[1])
        # Orient the normal to face the camera (against the ray direction).
        if np.dot(n, d) > 0.0:
            n = -n
        tangent = np.array([-n[1], n[0]])
        return RayHit(hit=True, point=np.asarray(point, dtype=float),
                      distance=float(t_hit), normal=n, tangent=tangent)
