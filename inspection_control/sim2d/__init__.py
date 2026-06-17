"""2D inspection control sandbox (admittance / orientation / autofocus).

Pure numpy + matplotlib; no ROS. Run with ``python sim2d/run.py``.
"""

from .camera import Camera2D, RayHit
from .controllers import AutofocusController, OrientationController
from .dynamics import AdmittancePlant
from .shapes import SHAPES, Surface, make_shape, register_shape
from .world import World

__all__ = [
    "Camera2D", "RayHit", "AdmittancePlant",
    "OrientationController", "AutofocusController",
    "Surface", "SHAPES", "make_shape", "register_shape", "World",
]
