"""The virtual sphere the cell's assistive controllers share.

admittance_control integrates it: it is the plant the arm follows. The
orientation controller and autofocus place their poles on the same plant, so
all three must use the same values. admittance_control owns them (its
sphere_mass, sphere_radius and fluid_viscosity parameters, the only place they
are set) and publishes them, latched, on ~/virtual_sphere
(inspection_control_interfaces/msg/VirtualSphere) at startup and on every
change. The other nodes follow that topic with SphereFollower and hold their
output at zero until the first message.

The formulas live here only. The dataclass imports nothing from ROS, so offline
tools (debag_orientation_data) use it too.

The message is in the base layer (inspection_control_interfaces). Until base is
rebuilt with it, message_type() is None: admittance_control runs on its own
parameters without publishing, and the followers hold their output at zero,
each logging why, instead of failing to start.
"""

from dataclasses import dataclass
import math

DEFAULT_TOPIC = '/admittance_control/virtual_sphere'
PARAMETERS = ('sphere_mass', 'sphere_radius', 'fluid_viscosity')


@dataclass(frozen=True)
class VirtualSphere:
    """A solid sphere of `mass` and `radius` in a fluid of `viscosity`."""

    mass: float       # kg
    radius: float     # m
    viscosity: float  # Pa·s

    @property
    def inertia(self) -> float:
        """Moment of inertia about the centre, kg·m² (2/5 m r²)."""
        return 0.4 * self.mass * self.radius ** 2

    @property
    def linear_drag(self) -> float:
        """Stokes drag coefficient, N·s/m (6 π μ r)."""
        return 6.0 * math.pi * self.viscosity * self.radius

    @property
    def angular_drag(self) -> float:
        """Rotational drag coefficient, N·m·s/rad (2.4 π μ r³)."""
        return 2.4 * math.pi * self.viscosity * self.radius ** 3

    def swing_inertia(self, d: float) -> float:
        """Inertia swinging about a pivot at distance d, kg·m² (parallel axis)."""
        return self.inertia + self.mass * d * d

    def swing_drag(self, d: float) -> float:
        """Drag of that swing, N·m·s/rad (c d²)."""
        return self.linear_drag * d * d

    def coast_time(self) -> float:
        """Time constant, s, of coasting to rest with no effort (m/c; the same for roll)."""
        return self.mass / self.linear_drag

    def swing_coast_time(self, d: float) -> float:
        """Time constant, s, of a swing about a pivot at d coasting to rest."""
        return self.swing_inertia(d) / self.swing_drag(d)

    def problem(self):
        """Why these values can't be used, or None. All must be finite and > 0."""
        for name, value in zip(PARAMETERS, (self.mass, self.radius, self.viscosity)):
            if not (math.isfinite(value) and value > 0.0):
                return f'{name} must be > 0 (got {value})'
        return None

    def describe(self) -> str:
        """One line for logs."""
        return (f'm={self.mass:g} kg, r={self.radius:g} m, mu={self.viscosity:g} Pa·s '
                f'(I={self.inertia:.4g} kg·m², c={self.linear_drag:.4g} N·s/m)')

    def to_msg(self, stamp=None):
        """As an inspection_control_interfaces/msg/VirtualSphere (needs message_type())."""
        msg = message_type()()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.mass = float(self.mass)
        msg.radius = float(self.radius)
        msg.viscosity = float(self.viscosity)
        msg.inertia = float(self.inertia)
        msg.linear_drag = float(self.linear_drag)
        msg.angular_drag = float(self.angular_drag)
        return msg

    @classmethod
    def from_msg(cls, msg) -> 'VirtualSphere':
        """From a VirtualSphere message: its three inputs (derived values are recomputed)."""
        return cls(float(msg.mass), float(msg.radius), float(msg.viscosity))


MISSING_MESSAGE = (
    'inspection_control_interfaces has no VirtualSphere message: rebuild the base '
    'layer (a full inspection-restart, or inspection-build base with the cell stopped)')


def message_type():
    """Return the VirtualSphere message type, or None if base predates it."""
    try:
        from inspection_control_interfaces.msg import VirtualSphere as VirtualSphereMsg
    except ImportError:
        return None
    return VirtualSphereMsg


def latched_qos():
    """Return the QoS of ~/virtual_sphere: one message, kept for late subscribers."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1)


class SphereFollower:
    """Follow admittance_control's virtual sphere on a node.

    `sphere` is None until the first valid message: a node using it must hold
    its output at zero until then. `on_change(sphere)` is called on every new
    value (not on repeats).
    """

    def __init__(self, node, topic=DEFAULT_TOPIC, on_change=None, callback_group=None):
        """Subscribe on `node` to `topic` (latched)."""
        self._node = node
        self.topic = topic
        self.sphere = None
        self._on_change = on_change
        msg_type = message_type()
        if msg_type is None:
            node.get_logger().error(f'{MISSING_MESSAGE}. No plant: holding zero output.')
            return
        node.create_subscription(
            msg_type, topic, self._on_msg, latched_qos(),
            callback_group=callback_group)

    def _on_msg(self, msg):
        sphere = VirtualSphere.from_msg(msg)
        problem = sphere.problem()
        if problem is not None:
            self._node.get_logger().error(
                f'Ignoring the virtual sphere on {self.topic}: {problem}')
            return
        if sphere == self.sphere:
            return
        self.sphere = sphere
        self._node.get_logger().info(f'Virtual sphere from {self.topic}: {sphere.describe()}')
        if self._on_change is not None:
            self._on_change(sphere)

    def warn_missing(self, what='holding zero output'):
        """Throttled warning for a node that has no sphere yet."""
        self._node.get_logger().warn(
            f'No virtual sphere on {self.topic} yet (is admittance_control running?): {what}',
            throttle_duration_sec=5.0)
