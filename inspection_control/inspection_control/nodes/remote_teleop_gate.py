"""Remote teleop gate: the only path from a remote operator to the arm.

Runs only with remote_teleop:=true. It sits between the joystick and the nodes
that read it (teleop, turntable_joy_node):

    joy_node (local) ---- joy_local ----\
                                         remote_teleop_gate ---- joy ----> teleop,
    Remote Teleop panel -- remote/joy --/                                turntable_joy_node
    (Foxglove, over Tailscale)

Rules:
- Not armed: the local joystick passes straight through; remote input is ignored.
- Arming (remote_teleop/arm, a Trigger) succeeds only while exactly one remote
  client is streaming, its deadman is released, and the local joystick is idle.
  The gate then binds to that client (its id is the messages' header.frame_id)
  and ignores any other.
- Armed: that client's messages pass through while fresh. The deadman itself
  is enforced downstream: teleop and turntable_joy_node only move while their
  enable_button (the same button) is held, and admittance control applies no
  wrench at all (orientation control and autofocus included) without it.
- A gap in the remote stream longer than remote_timeout: the gate sends the
  last message with the deadman released, so the arm stops at once (assistive
  controllers too), and keeps doing so. After link_loss_timeout it disarms and switches orientation
  control and autofocus off.
- Any local joystick activity while armed: local takeover. The gate disarms and
  the local joystick passes through again.
- While armed, the admittance controllers' speed clamps are lowered to the
  remote limits; disarming restores them. Arming fails if they can't be set.
- remote_teleop/status (std_msgs/String, JSON, 5 Hz) reports all of this for
  the panel.

None of this is a safety function: the UR safety configuration, the remote-mode
key and keeping people out of reach are (see the Inspection_Control README).
"""

import json
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger

LIMIT_PARAMS = ('max_linear_speed', 'max_angular_speed')


class RemoteTeleopGate(Node):
    def __init__(self):
        super().__init__('remote_teleop_gate')
        self.declare_parameters('', [
            ('local_joy_topic', 'joy_local'),
            ('remote_joy_topic', 'remote/joy'),
            ('output_joy_topic', 'joy'),
            # The lab controller's layout, from the controller profile
            # (xbox_controller.yaml, 8bitdo_pro2.yaml, ...). The Remote Teleop
            # panel converts the remote gamepad to this layout.
            ('layout', 'xbox'),
            # Must match teleop's and turntable_joy_node's enable_button.
            ('deadman_button', 4),
            # The remote stream must be newer than this to pass (s). The panel
            # sends at 20 Hz.
            ('remote_timeout', 0.3),
            # A remote stream gap this long disarms (s).
            ('link_loss_timeout', 2.0),
            # Arming needs the local joystick to have been idle this long (s).
            ('local_idle_time', 2.0),
            # How far an axis must move between local messages to count as
            # someone using the local joystick.
            ('local_axis_threshold', 0.15),
            ('rate', 50.0),
            # Admittance nodes whose speed clamps are lowered while armed.
            ('admittance_nodes', ['/admittance_control']),
            ('remote_max_linear_speed', 0.05),   # m/s
            ('remote_max_angular_speed', 0.2),   # rad/s
            # Called when the link is lost (Trigger services).
            ('stop_services', ['/orientation_controller/disable_orientation_control',
                               '/autofocus/disable_autofocus']),
            # Remote messages must have at least this many (the lab's Xbox layout).
            ('min_axes', 8),
            ('min_buttons', 11),
        ])
        p = self.get_parameter
        self.layout = str(p('layout').value)
        self.deadman = int(p('deadman_button').value)
        self.remote_timeout = float(p('remote_timeout').value)
        self.link_loss_timeout = float(p('link_loss_timeout').value)
        self.local_idle_time = float(p('local_idle_time').value)
        self.local_axis_threshold = float(p('local_axis_threshold').value)
        self.admittance_nodes = [n for n in p('admittance_nodes').value if n]
        self.remote_limits = (float(p('remote_max_linear_speed').value),
                              float(p('remote_max_angular_speed').value))
        self.min_axes = int(p('min_axes').value)
        self.min_buttons = int(p('min_buttons').value)

        self.cb = ReentrantCallbackGroup()
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE, depth=10)
        self.out_pub = self.create_publisher(Joy, p('output_joy_topic').value, qos)
        self.status_pub = self.create_publisher(
            String, 'remote_teleop/status',
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1))
        self.create_subscription(Joy, p('local_joy_topic').value, self.on_local, qos,
                                 callback_group=self.cb)
        self.create_subscription(Joy, p('remote_joy_topic').value, self.on_remote, qos,
                                 callback_group=self.cb)
        self.create_service(Trigger, 'remote_teleop/arm', self.on_arm, callback_group=self.cb)
        self.create_service(Trigger, 'remote_teleop/disarm', self.on_disarm,
                            callback_group=self.cb)
        self.param_clients = {
            n: (self.create_client(GetParameters, f'{n}/get_parameters', callback_group=self.cb),
                self.create_client(SetParameters, f'{n}/set_parameters', callback_group=self.cb))
            for n in self.admittance_nodes}
        self.stop_clients = [self.create_client(Trigger, s, callback_group=self.cb)
                             for s in p('stop_services').value if s]

        # State
        self.armed = False
        self.client_id = ''
        self.remote_clients = {}          # client id -> last receive time (monotonic s)
        self.last_remote = None           # last message from the bound client
        self.last_remote_t = None
        self.holding_stop = False         # sending deadman-released copies
        self.last_local = None
        self.last_local_activity = None   # monotonic s
        self.saved_limits = {}            # node -> (linear, angular)
        self.last_event = 'not armed'
        # Guards armed/client_id against the callbacks running on other
        # threads: a remote message must never be forwarded after a disarm's
        # release message. Never held while waiting on a service.
        self.lock = threading.RLock()

        self.create_timer(1.0 / float(p('rate').value), self.tick, callback_group=self.cb)
        self.create_timer(0.2, self.publish_status, callback_group=self.cb)
        self.get_logger().info(
            f'Remote teleop gate ready: layout {self.layout}, deadman button {self.deadman}, remote timeout '
            f'{self.remote_timeout:.2f} s, disarm after {self.link_loss_timeout:.1f} s, '
            f'remote limits {self.remote_limits[0]} m/s / {self.remote_limits[1]} rad/s '
            f'on {", ".join(self.admittance_nodes) or "(no admittance nodes)"}')

    # ------------------------------------------------------------------ inputs
    def on_local(self, msg: Joy):
        active = self._local_active(msg)
        self.last_local = msg
        if active:
            self.last_local_activity = time.monotonic()
        with self.lock:
            if self.armed and not active:
                return  # remote has the arm; the local joystick is idle
        if self.armed and active:
            self.disarm('local joystick took over')
        self.out_pub.publish(msg)

    def _local_active(self, msg: Joy) -> bool:
        if any(msg.buttons):
            return True
        prev = self.last_local
        if prev is None or len(prev.axes) != len(msg.axes):
            return False
        return any(abs(a - b) > self.local_axis_threshold for a, b in zip(msg.axes, prev.axes))

    def on_remote(self, msg: Joy):
        now = time.monotonic()
        cid = msg.header.frame_id or 'anonymous'
        self.remote_clients[cid] = now
        if len(msg.axes) < self.min_axes or len(msg.buttons) < self.min_buttons:
            self.last_event = (f'ignored a remote message with {len(msg.axes)} axes / '
                               f'{len(msg.buttons)} buttons (need {self.min_axes} / {self.min_buttons})')
            return
        with self.lock:
            if cid != self.client_id:
                return
            self.last_remote = msg
            self.last_remote_t = now
            if self.armed:
                if self.holding_stop:
                    self.holding_stop = False
                    self.get_logger().info('Remote link back')
                self.out_pub.publish(self._restamped(msg))

    # --------------------------------------------------------------- services
    def on_arm(self, request, response):
        now = time.monotonic()
        if self.armed:
            response.success, response.message = True, f'already armed for {self.client_id}'
            return response
        fresh = [c for c, t in self.remote_clients.items() if now - t <= self.remote_timeout]
        if not fresh:
            return self._refuse(response, 'no remote client is streaming (is the panel connected '
                                          'and a gamepad detected?)')
        if len(fresh) > 1:
            return self._refuse(response, f'{len(fresh)} remote clients are streaming; '
                                          'close the other Remote Teleop panels first')
        if self.last_local_activity is not None and now - self.last_local_activity < self.local_idle_time:
            return self._refuse(response, 'the local joystick is in use')
        cid = fresh[0]
        self.client_id = cid  # start accepting this client's messages
        # The deadman must be released when arming.
        deadline = time.monotonic() + 0.5
        while self.last_remote is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.last_remote is None:
            self.client_id = ''
            return self._refuse(response, 'no valid message from the remote client')
        if self._deadman(self.last_remote):
            self.client_id = ''
            self.last_remote = None
            return self._refuse(response, 'release the deadman (left bumper) before arming')
        ok, why = self._apply_remote_limits()
        if not ok:
            self.client_id = ''
            self.last_remote = None
            return self._refuse(response, why)
        with self.lock:
            self.armed = True
            self.holding_stop = False
            self.last_remote_t = time.monotonic()
            self.last_event = f'armed for {cid}'
        self.get_logger().warn(f'ARMED: remote client {cid} drives the arm '
                               f'(limits {self.remote_limits[0]} m/s, {self.remote_limits[1]} rad/s)')
        response.success, response.message = True, self.last_event
        return response

    def on_disarm(self, request, response):
        was = self.armed
        self.disarm('disarmed by the operator')
        response.success = True
        response.message = 'disarmed' if was else 'was not armed'
        return response

    def _refuse(self, response, why):
        self.last_event = f'arming refused: {why}'
        self.get_logger().info(self.last_event)
        response.success, response.message = False, why
        return response

    # ------------------------------------------------------------------- loop
    def tick(self):
        with self.lock:
            if not self.armed or self.last_remote_t is None:
                return
            age = time.monotonic() - self.last_remote_t
            if age <= self.remote_timeout:
                return
            if age <= self.link_loss_timeout:
                if not self.holding_stop:
                    self.holding_stop = True
                    self.get_logger().warn(f'Remote stream {age:.2f} s old: stopping the arm')
                self._send_release()
                return
        self.disarm(f'remote link lost for {age:.1f} s', stop_controllers=True)

    def disarm(self, reason, stop_controllers=False):
        with self.lock:
            if not self.armed:
                return
            # From here on no remote message is forwarded; the release goes
            # out last.
            self.armed = False
            self._send_release()
            self.client_id = ''
            self.last_remote = None
            self.holding_stop = False
            self.last_event = f'disarmed: {reason}'
        self.get_logger().warn(f'DISARMED: {reason}')
        if stop_controllers:
            for c in self.stop_clients:
                if c.service_is_ready():
                    c.call_async(Trigger.Request())
        self._restore_limits()

    def _send_release(self):
        """The last remote message with the deadman released: teleop and
        turntable_joy_node stop at once."""
        if self.last_remote is None:
            return
        msg = self._restamped(self.last_remote)
        buttons = list(msg.buttons)
        if 0 <= self.deadman < len(buttons):
            buttons[self.deadman] = 0
        msg.buttons = buttons
        self.out_pub.publish(msg)

    def _restamped(self, msg: Joy) -> Joy:
        out = Joy()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'remote'
        out.axes = list(msg.axes)
        out.buttons = list(msg.buttons)
        return out

    def _deadman(self, msg: Joy) -> bool:
        return 0 <= self.deadman < len(msg.buttons) and bool(msg.buttons[self.deadman])

    # ------------------------------------------------------- speed limits
    def _call(self, client, request, timeout=1.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result() if future.done() else None

    def _apply_remote_limits(self):
        if not self.admittance_nodes:
            return False, 'no admittance nodes configured'
        saved = {}
        for node, (get_c, set_c) in self.param_clients.items():
            res = self._call(get_c, GetParameters.Request(names=list(LIMIT_PARAMS)))
            if res is None or len(res.values) != 2:
                self._restore(saved)
                return False, f'{node} is not answering (is admittance control running?)'
            saved[node] = tuple(v.double_value for v in res.values)
            # Lower, never raise: 0 means "no clamp", so it counts as infinite.
            new = tuple(min(old, lim) if old > 0.0 else lim
                        for old, lim in zip(saved[node], self.remote_limits))
            if not self._set(set_c, new):
                self._restore(saved)
                return False, f'could not set the speed limits on {node}'
        self.saved_limits = saved
        return True, ''

    def _restore_limits(self):
        self._restore(self.saved_limits)
        self.saved_limits = {}

    def _restore(self, saved):
        for node, values in saved.items():
            if not self._set(self.param_clients[node][1], values):
                self.get_logger().error(
                    f'could not restore the speed limits on {node} to {values}; set them by hand')

    def _set(self, client, values):
        req = SetParameters.Request(parameters=[
            ParameterMsg(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                                         double_value=float(v)))
            for name, v in zip(LIMIT_PARAMS, values)])
        res = self._call(client, req)
        return res is not None and all(r.successful for r in res.results)

    # ---------------------------------------------------------------- status
    def publish_status(self):
        now = time.monotonic()
        fresh = sorted(c for c, t in self.remote_clients.items() if now - t <= self.remote_timeout)
        status = {
            'armed': self.armed,
            'client': self.client_id,
            'clients_streaming': fresh,
            'remote_age_ms': (round((now - self.last_remote_t) * 1000)
                              if self.armed and self.last_remote_t is not None else None),
            'stopped_by_gate': self.holding_stop,
            'deadman': bool(self.last_remote is not None and self._deadman(self.last_remote)),
            'local_in_use': bool(self.last_local_activity is not None
                                 and now - self.last_local_activity < self.local_idle_time),
            'remote_limits': {'linear': self.remote_limits[0], 'angular': self.remote_limits[1]},
            'layout': self.layout,
            'deadman_button': self.deadman,
            'event': self.last_event,
        }
        self.status_pub.publish(String(data=json.dumps(status)))


def main(args=None):
    rclpy.init(args=args)
    node = RemoteTeleopGate()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.armed:
            node.disarm('gate shutting down')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
