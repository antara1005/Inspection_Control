"""Surface library for the 2D inspection sandbox.

A *surface* is an implicit signed field ``f(x, y)`` with the convention

    f < 0  inside the object
    f = 0  on the surface
    f > 0  in free space (where the camera lives)

The outward normal falls out of the gradient: ``n = ∇f / |∇f|``. The base class
provides a finite-difference gradient and an implicit-field polyline extractor, so a
new shape only needs to define ``f``. Register it with :func:`register_shape` and it is
immediately available to the camera raycaster, the world, and the viz shape selector —
no other file needs to change.

Adding a shape::

    @register_shape("parabola")
    class Parabola(Surface):
        def __init__(self, a=1.0, y0=0.0):
            self.a, self.y0 = a, y0

        def f(self, x, y):
            return y - (self.a * x**2 + self.y0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
SHAPES: dict[str, type["Surface"]] = {}


def register_shape(name: str):
    """Class decorator registering a :class:`Surface` subclass under ``name``."""
    def _decorator(cls):
        SHAPES[name] = cls
        cls.shape_name = name
        return cls
    return _decorator


def make_shape(name: str, **kwargs) -> "Surface":
    """Factory: instantiate a registered shape by name."""
    if name not in SHAPES:
        raise KeyError(f"unknown shape '{name}'; registered: {sorted(SHAPES)}")
    return SHAPES[name](**kwargs)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Surface(ABC):
    """Implicit 2D surface ``f(x, y)`` (negative inside, positive outside)."""

    shape_name: str = "surface"

    @abstractmethod
    def f(self, x, y):
        """Signed implicit field. Must accept scalars or numpy arrays."""
        raise NotImplementedError

    # -- gradient / normal -------------------------------------------------- #
    def grad(self, x, y, eps: float = 1e-5):
        """Gradient ``(∂f/∂x, ∂f/∂y)`` via central differences.

        Override with an analytic gradient for accuracy/speed; the FD default keeps
        new shapes to a single ``f`` method.
        """
        fx = (self.f(x + eps, y) - self.f(x - eps, y)) / (2.0 * eps)
        fy = (self.f(x, y + eps) - self.f(x, y - eps)) / (2.0 * eps)
        return np.array([fx, fy], dtype=float)

    def normal(self, x, y):
        """Unit outward normal at ``(x, y)`` (points toward increasing ``f``)."""
        g = self.grad(x, y)
        n = np.linalg.norm(g)
        if n < 1e-12:
            return np.array([0.0, 1.0])
        return g / n

    # -- plotting ----------------------------------------------------------- #
    def curve_points(self, bounds, resolution: int = 600):
        """Return an (N, 2) polyline of the zero level set within ``bounds``.

        Default: sample the implicit field on a grid and extract the ``f = 0``
        contour with matplotlib. Parametric shapes (circle, line) override this with
        an exact, cheaper parametrization.

        ``bounds`` is ``(xmin, xmax, ymin, ymax)``.
        """
        import matplotlib.pyplot as plt

        xmin, xmax, ymin, ymax = bounds
        xs = np.linspace(xmin, xmax, resolution)
        ys = np.linspace(ymin, ymax, resolution)
        gx, gy = np.meshgrid(xs, ys)
        gz = self.f(gx, gy)
        # Use a throwaway contour to pull out the polyline vertices.
        cs = plt.contour(gx, gy, gz, levels=[0.0])
        segs = []
        for path in cs.get_paths():
            segs.append(path.vertices)
        plt.close()
        if not segs:
            return np.empty((0, 2))
        return np.vstack(segs)


# --------------------------------------------------------------------------- #
# Concrete shapes
# --------------------------------------------------------------------------- #
@register_shape("circle")
class Circle(Surface):
    """Circle of radius ``r`` centered at ``(cx, cy)`` (solid disk, inside < 0)."""

    def __init__(self, cx: float = 0.0, cy: float = 0.0, r: float = 1.0):
        self.cx, self.cy, self.r = cx, cy, r

    def f(self, x, y):
        return np.hypot(x - self.cx, y - self.cy) - self.r

    def grad(self, x, y, eps: float = 1e-5):
        dx, dy = x - self.cx, y - self.cy
        d = np.hypot(dx, dy)
        d = d if d > 1e-12 else 1e-12
        return np.array([dx / d, dy / d], dtype=float)

    def curve_points(self, bounds, resolution: int = 600):
        t = np.linspace(0.0, 2.0 * np.pi, resolution)
        return np.column_stack([self.cx + self.r * np.cos(t),
                                self.cy + self.r * np.sin(t)])


@register_shape("ellipse")
class Ellipse(Surface):
    """Axis-aligned ellipse with semi-axes ``(a, b)`` centered at ``(cx, cy)``."""

    def __init__(self, cx: float = 0.0, cy: float = 0.0, a: float = 1.5, b: float = 1.0):
        self.cx, self.cy, self.a, self.b = cx, cy, a, b

    def f(self, x, y):
        # Scaled implicit so |∇f| ~ 1 near the surface (well-conditioned raymarch).
        return np.hypot((x - self.cx) / self.a, (y - self.cy) / self.b) - 1.0

    def curve_points(self, bounds, resolution: int = 600):
        t = np.linspace(0.0, 2.0 * np.pi, resolution)
        return np.column_stack([self.cx + self.a * np.cos(t),
                                self.cy + self.b * np.sin(t)])


@register_shape("sine")
class SineWall(Surface):
    """Horizontal sinusoidal wall ``y = amp·sin(k·x) + y0`` (object below)."""

    def __init__(self, amp: float = 0.5, k: float = 1.5, y0: float = 0.0):
        self.amp, self.k, self.y0 = amp, k, y0

    def f(self, x, y):
        return y - (self.amp * np.sin(self.k * x) + self.y0)

    def grad(self, x, y, eps: float = 1e-5):
        # ∂f/∂x = -amp·k·cos(kx),  ∂f/∂y = 1
        fx = -self.amp * self.k * np.cos(self.k * x)
        fy = np.ones_like(fx) if np.ndim(fx) else 1.0
        return np.array([fx, fy], dtype=float)

    def curve_points(self, bounds, resolution: int = 600):
        xmin, xmax, *_ = bounds
        xs = np.linspace(xmin, xmax, resolution)
        return np.column_stack([xs, self.amp * np.sin(self.k * xs) + self.y0])


@register_shape("plane")
class Plane(Surface):
    """Flat wall: half-space with outward unit normal ``n`` through ``point``.

    ``f(p) = n · (p - point)`` — exactly a signed distance.
    """

    def __init__(self, px: float = 0.0, py: float = 0.0, nx: float = 0.0, ny: float = 1.0):
        n = np.hypot(nx, ny) or 1.0
        self.px, self.py = px, py
        self.nx, self.ny = nx / n, ny / n

    def f(self, x, y):
        return self.nx * (x - self.px) + self.ny * (y - self.py)

    def grad(self, x, y, eps: float = 1e-5):
        shape = np.shape(x)
        return np.array([np.full(shape, self.nx) if shape else self.nx,
                         np.full(shape, self.ny) if shape else self.ny], dtype=float)

    def curve_points(self, bounds, resolution: int = 600):
        xmin, xmax, ymin, ymax = bounds
        # Draw the wall line across the view: tangent is perpendicular to the normal.
        tx, ty = -self.ny, self.nx
        span = 2.0 * max(xmax - xmin, ymax - ymin)
        s = np.linspace(-span, span, resolution)
        return np.column_stack([self.px + tx * s, self.py + ty * s])
