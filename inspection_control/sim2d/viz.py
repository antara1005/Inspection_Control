"""Interactive matplotlib front end for the 2D inspection sandbox.

A ``FuncAnimation`` advances :meth:`World.step` at a fixed rate and redraws the
surface, camera, optical-axis ray, surface normal, and a text HUD. Keyboard nudges the
camera (manual teleop) and toggles controllers; sliders tune gains live.

Controls — teleop meaning depends on whether orientation is engaged.
--------
  Orientation ON (pendulum about the surface pivot):
    W / S    standoff −/+ (toward / away from the surface, changes d)
    A / D    slide the pivot along the surface tangent (−/+ t)
    Q / E    nudge the orientation reference Δ (pivot the view about the target)
  Orientation OFF (free body in camera coordinates):
    W / S    forward / backward along the optical axis
    A / D    left / right (camera frame)
    Q / E    rotate the camera about its COM
  o          toggle orientation control
  f          toggle autofocus drive (to the known true peak distance)
  space      reset camera + plant
  (sliders)  zeta, orient v_max, theta_max, integral_alpha, autofocus v_max,
             mass, viscosity, sensor-noise std
  (radios)   orientation controller (PD / PID), surface shape
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Slider

from .shapes import SHAPES
from .world import World

BOUNDS = (-5.0, 5.0, -4.0, 4.0)   # xmin, xmax, ymin, ymax
DT = 0.02                          # s, 50 Hz tick
TELEOP_FORCE = 8.0                 # generalized force per key press (slide / standoff)
TELEOP_TORQUE = 3.0                # manual swing torque per key press
TELEOP_DELTA = np.radians(3.0)     # reference-offset nudge per key press (rad)
TELEOP_DECAY = 0.55                # per-frame decay so held keys fade on release

# Keys we bind for control. matplotlib's default keymap claims several of these
# (s=save, f=fullscreen, r=home, q=quit, o=zoom, ...); strip them so our handler
# is the sole responder.
CONTROL_KEYS = {"w", "a", "s", "d", "q", "e", "o", "f", " "}


def _release_default_keymaps():
    for name in list(plt.rcParams):
        if name.startswith("keymap."):
            plt.rcParams[name] = [k for k in plt.rcParams[name]
                                  if k not in CONTROL_KEYS]


class SimApp:
    def __init__(self, world: World | None = None):
        self.world = world or World()
        _release_default_keymaps()

        self.fig = plt.figure(figsize=(11, 8))
        self.ax = self.fig.add_axes([0.30, 0.08, 0.66, 0.88])
        self.ax.set_xlim(BOUNDS[0], BOUNDS[1])
        self.ax.set_ylim(BOUNDS[2], BOUNDS[3])
        self.ax.set_aspect("equal")
        self.ax.set_title("2D inspection sandbox")
        self.ax.grid(True, alpha=0.2)

        # -- artists -------------------------------------------------------- #
        (self.surf_line,) = self.ax.plot([], [], color="0.2", lw=2, label="surface")
        (self.cam_dot,) = self.ax.plot([], [], "o", color="tab:blue", ms=9,
                                       label="camera")
        (self.ray_line,) = self.ax.plot([], [], "--", color="tab:orange", lw=1.2,
                                        label="ray")
        (self.hit_dot,) = self.ax.plot([], [], "x", color="tab:red", ms=9, mew=2)
        self.heading_q = self.ax.quiver([0], [0], [0], [0], color="tab:blue",
                                        scale=12, width=0.006, zorder=5)
        self.normal_q = self.ax.quiver([0], [0], [0], [0], color="tab:green",
                                       scale=12, width=0.006, zorder=5)
        self.hud = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                va="top", ha="left", family="monospace", fontsize=9,
                                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        self.ax.legend(loc="lower right", fontsize=8)

        self._draw_surface()
        self._build_widgets()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.anim = FuncAnimation(self.fig, self._update, interval=int(DT * 1000),
                                  blit=False, cache_frame_data=False)

    # -- widgets ------------------------------------------------------------ #
    def _build_widgets(self):
        w = self.world
        o = w.orientation
        # Stack of sliders down the left margin.
        defs = [
            ("s_zeta",   "zeta",          1.0,  4.0,  o.zeta),
            ("s_ovmax",  "orient v_max",  0.05, 2.0,  o.v_max),
            ("s_thetam", "theta_max °",   5.0,  60.0, np.degrees(o.theta_max)),
            ("s_ialpha", "integral_alpha", 1.0, 10.0, o.integral_alpha),
            ("s_vmax",   "af v_max",      0.01, 0.5,  w.autofocus.v_max),
            ("s_mass",   "mass",          0.5,  8.0,  w.plant.mass),
            ("s_visc",   "viscosity",     0.1,  4.0,  w.plant.viscosity),
            ("s_dn",     "dist noise σ",  0.0,  0.3,  w.distance_noise_std),
            ("s_nn",     "norm noise σ°", 0.0,  30.0, np.degrees(w.normal_noise_std)),
        ]
        y = 0.95
        self._sliders = []
        for attr, label, lo, hi, val in defs:
            ax = self.fig.add_axes([0.07, y, 0.16, 0.022])
            s = Slider(ax, label, lo, hi, valinit=val)
            s.label.set_fontsize(8)
            s.on_changed(self._on_sliders)
            setattr(self, attr, s)
            self._sliders.append(s)
            y -= 0.042

        # Controller type (PD / PID), then surface shape.
        ax_ctrl = self.fig.add_axes([0.07, 0.40, 0.16, 0.10])
        ax_ctrl.set_title("orient controller", fontsize=8)
        types = ["PD", "PID"]
        self.radio_ctrl = RadioButtons(ax_ctrl, types,
                                       active=types.index(o.controller_type))
        self.radio_ctrl.on_clicked(self._on_ctrl_type)

        ax_radio = self.fig.add_axes([0.07, 0.12, 0.16, 0.24])
        ax_radio.set_title("shape", fontsize=8)
        names = list(SHAPES.keys())
        self.radio = RadioButtons(ax_radio, names,
                                  active=names.index(w.shape_name))
        self.radio.on_clicked(self._on_shape)

    def _on_sliders(self, _):
        w = self.world
        w.orientation.zeta = self.s_zeta.val
        w.orientation.v_max = self.s_ovmax.val
        w.orientation.theta_max = np.radians(self.s_thetam.val)
        w.orientation.integral_alpha = self.s_ialpha.val
        w.autofocus.zeta = self.s_zeta.val
        w.autofocus.v_max = self.s_vmax.val
        w.plant.set_inertia_and_drag(self.s_mass.val, w.plant.radius,
                                     self.s_visc.val)
        w.distance_noise_std = self.s_dn.val
        w.normal_noise_std = np.radians(self.s_nn.val)

    def _on_ctrl_type(self, label):
        self.world.orientation.controller_type = label
        self.world.orientation.reset_integral()

    def _on_shape(self, name):
        self.world.set_shape(name)
        self._draw_surface()

    def _draw_surface(self):
        pts = self.world.surface.curve_points(BOUNDS)
        if pts.size:
            self.surf_line.set_data(pts[:, 0], pts[:, 1])
        else:
            self.surf_line.set_data([], [])

    # -- keyboard ----------------------------------------------------------- #
    def _on_key(self, event):
        k = (event.key or "").lower()
        w = self.world
        # Generalized efforts. The plant interprets them in the surface-pivot frame
        # when orientation is ON (pendulum) and in the camera frame when OFF (free
        # body): standoff↔optical axis, tangential↔camera-right, swing↔COM rotation.
        if k == "w":                       # toward the surface / forward -> decrease d
            w.teleop_standoff -= TELEOP_FORCE
        elif k == "s":                     # away / backward -> increase d
            w.teleop_standoff += TELEOP_FORCE
        elif k == "d":                     # +tangential (slide) / camera-right
            w.teleop_tangential += TELEOP_FORCE
        elif k == "a":                     # -tangential (slide) / camera-left
            w.teleop_tangential -= TELEOP_FORCE
        elif k in ("q", "e"):              # swing
            sign = 1.0 if k == "q" else -1.0
            if w.orientation.enabled:      # pivot about the target via the reference Δ
                w.orientation.delta += sign * TELEOP_DELTA
            else:                          # manual swing torque about the pivot
                w.teleop_swing += sign * TELEOP_TORQUE
        elif k == "o":
            w.orientation.enabled = not w.orientation.enabled
        elif k == "f":
            w.autofocus.toggle()
        elif k == " ":
            w.reset()

    # -- animation tick ----------------------------------------------------- #
    def _update(self, _frame):
        w = self.world
        ray = w.step(DT)

        # Manual teleop fades after release (matplotlib has no reliable key-up).
        # The orientation reference Δ is persistent, so it is not decayed.
        w.teleop_tangential *= TELEOP_DECAY
        w.teleop_standoff *= TELEOP_DECAY
        w.teleop_swing *= TELEOP_DECAY

        cam = w.camera
        self.cam_dot.set_data([cam.pos[0]], [cam.pos[1]])
        zhat = cam.optical_axis
        self.heading_q.set_offsets([cam.pos])
        self.heading_q.set_UVC(zhat[0], zhat[1])

        if ray.hit:
            self.ray_line.set_data([cam.pos[0], ray.point[0]],
                                   [cam.pos[1], ray.point[1]])
            self.hit_dot.set_data([ray.point[0]], [ray.point[1]])
            self.normal_q.set_offsets([ray.point])
            self.normal_q.set_UVC(ray.normal[0], ray.normal[1])
        else:
            far = cam.pos + 20.0 * zhat
            self.ray_line.set_data([cam.pos[0], far[0]], [cam.pos[1], far[1]])
            self.hit_dot.set_data([], [])
            self.normal_q.set_UVC(0, 0)

        self.hud.set_text(self._hud_text(ray))
        return ()

    def _hud_text(self, ray) -> str:
        w = self.world
        af = w.autofocus
        d = f"{ray.distance:.3f} m" if ray.hit else "  (no hit)"
        ang = np.degrees(w.orientation.angle_error)
        delta = np.degrees(w.orientation.delta)
        return (
            f"distance : {d}\n"
            f"focus    : {af.focus_value:5.3f}  (true peak {af.d_focus:.2f} m)\n"
            f"orient   : {'ON ' if w.orientation.enabled else 'off'}  "
            f"[{w.orientation.controller_type}]  err {ang:+6.1f} deg  Δ {delta:+5.1f} deg\n"
            f"autofocus: {'ON ' if af.enabled else 'off'}  target {af.d_focus:.3f} m\n"
            f"shape    : {w.shape_name}"
        )


def main():
    app = SimApp()
    plt.show()
    return app


if __name__ == "__main__":
    main()
