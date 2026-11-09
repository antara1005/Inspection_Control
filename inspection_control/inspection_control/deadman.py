"""Deadman input for the admittance controllers.

teleop publishes std_msgs/Bool on its deadman topic every tick: true only while
the joystick is live and its enable_button is held, and only for the arm the
joystick drives. An admittance controller applies no wrench at all (teleop,
orientation control, autofocus) unless that is true and fresh, so the assistive
controllers only move the arm while an operator holds the deadman. Released,
the plant coasts to rest on its drag.
"""

from std_msgs.msg import Bool


class DeadmanMonitor:
    """Subscribes to teleop's deadman and says whether wrenches may apply."""

    def __init__(self, node, topic: str, timeout: float, qos):
        """
        Subscribe to the deadman topic on node.

        topic: empty disables the deadman (wrenches always apply).
        timeout: a message older than this (s) counts as released; 0 disables
        the check.
        """
        self._node = node
        self.topic = topic
        self.timeout = timeout
        self._held = False
        self._t = None         # receive time (s) of the last message
        self._engaged = False  # last result of engaged(), for logging changes
        self._stale = False
        self._first_call = None  # time (s) of the first engaged() call
        if topic:
            node.create_subscription(Bool, topic, self._on_msg, qos)
            node.get_logger().info(
                f'Deadman: wrenches apply only while {topic} is true'
                + (f' (and newer than {timeout:.2f} s)' if timeout > 0.0 else ''))
        else:
            node.get_logger().warn(
                'deadman_topic is empty: wrenches apply WITHOUT a deadman held')

    def _on_msg(self, msg: Bool):
        self._held = bool(msg.data)
        self._t = self._node.get_clock().now().nanoseconds * 1e-9

    def state(self, now_s: float):
        """(engaged, age of the last message in s or None), without logging."""
        if not self.topic:
            return True, None
        if self._t is None:
            return False, None
        age = now_s - self._t
        stale = self.timeout > 0.0 and age > self.timeout
        return self._held and not stale, age

    def engaged(self, now_s: float) -> bool:
        """Whether wrenches may move the arm now."""
        if not self.topic:
            return True
        log = self._node.get_logger()
        if self._t is None:
            # Quiet for the first seconds: at startup the controllers' wrenches
            # often arrive before teleop's first tick.
            if self._first_call is None:
                self._first_call = now_s
            elif now_s - self._first_call > 2.0:
                log.warn(f'Ignoring all wrenches: no deadman message on {self.topic} yet '
                         '(is teleop running?)', throttle_duration_sec=10.0)
            self._engaged = False
            return False

        age = now_s - self._t
        stale = self.timeout > 0.0 and age > self.timeout
        if stale and not self._stale:
            log.warn(f'No deadman message on {self.topic} for {age:.2f} s: '
                     'ignoring all wrenches until it is back')
        elif self._stale and not stale:
            log.info('Deadman messages are back')
        self._stale = stale

        engaged = self._held and not stale
        if engaged != self._engaged:
            log.debug('Deadman held: applying wrenches' if engaged
                      else 'Deadman released: ignoring all wrenches')
        self._engaged = engaged
        return engaged
