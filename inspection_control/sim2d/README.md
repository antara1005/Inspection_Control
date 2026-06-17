# 2D Inspection Control Sandbox

A lightweight, interactive 2D playground for the inspection control stack
(**admittance**, **orientation**, **autofocus**). A curve stands in for the object
surface; a 2D "camera" casts a ray to measure focal distance + surface normal, and the
ported control laws close the loop. Pure `numpy` + `matplotlib` — no ROS.

The control math mirrors the real nodes so insights port straight back:

| sandbox file        | mirrors node                  | law |
|---------------------|-------------------------------|-----|
| `dynamics.py`       | `admittance_control_node.py`  | virtual mass + Stokes drag (`m·v̇+c·v=F`, `I·ω̇+c_ang·ω=τ`) |
| `controllers.py`    | `orientation_control_node.py` | theta-error torque about the surface point (parallel-axis pivot), converted to an equivalent camera wrench via Newton-Euler |
| `controllers.py`    | `autofocus_node.py`           | PD drive of the focal distance to a known true peak |

## Run

```bash
python sim2d/run.py                 # default sine-wall surface
python sim2d/run.py --shape circle  # circle / ellipse / sine / plane
```

Requires `numpy` and `matplotlib` (`pip install numpy matplotlib`). Use an interactive
backend (the default TkAgg/Qt on a desktop); a headless `Agg` backend will construct but
not display the window.

## Controls

| key        | action |
|------------|--------|
| `W`/`S`    | teleop force forward/backward (along the optical axis) |
| `A`/`D`    | teleop force left/right (camera frame) |
| `Q`/`E`    | teleop torque (rotate CCW/CW) |
| `o`        | toggle **orientation** alignment torque (rotation only) |
| `f`        | toggle **autofocus** drive (to the known true peak distance) |
| `space`    | reset camera + plant |

Sliders (left): `zeta`, autofocus `v_max`, `mass`, `viscosity`, and white **sensor
noise** std — `dist noise σ` (m) and `norm noise σ°` (degrees, rotates the measured
normal). Noise is injected into the measurement, so the controllers *and* the drawn
ray/normal reflect the noisy values.
Radio buttons (left): surface shape. The HUD (top-left of the plot) shows focal
distance, focus value, orientation error, and autofocus mode/target.

### Orientation = pivot about the surface point
The theta-error torque is applied **about the surface contact point**, not the camera
center. Using the parallel-axis pivot inertia `I_A = I_B + m·d²` and the pivot drag
`b = 6πμR·d²`, a pole-placement PD sets the pivot torque to realize
`I_A·θ̈ + (b+k₂)·θ̇ + k₁·θ = 0`; a Newton-Euler step converts it to the equivalent
camera force + torque fed to the admittance plant. The net camera force is purely
**lateral** (the camera swings about the contact point) — orientation never pushes
along the optical axis, so focal distance stays owned by autofocus and teleop.

Following the reference method, **only orientation is controlled** — there is no
position/standoff term. With a free base this leaves lateral position as an undamped
mode, so engaging orientation from a large offset makes the camera glide/roll along
the surface as it re-aims; hold position with teleop (or keep initial misalignment
small). See <https://macs-lab.github.io/robotic-inspection-orientation-control-2026/>.

### Typical autofocus session
1. Engage `o` so orientation swings the camera onto the surface normal.
2. Press `f` — the PD force drives the focal distance straight to the known true peak
   (`d_focus`). The HUD `focus` value rises toward 1.0 as it converges.

## Adding a shape

Subclass `Surface` and register it — nothing else changes (the raycaster, world, and
shape selector pick it up automatically). Only `f(x, y)` is required; the base class
supplies a finite-difference gradient and an implicit-field contour for plotting.

```python
# in shapes.py
@register_shape("parabola")
class Parabola(Surface):
    def __init__(self, a=0.4, y0=-1.0):
        self.a, self.y0 = a, y0

    def f(self, x, y):                       # < 0 inside, > 0 in free space
        return y - (self.a * x**2 + self.y0)

    # optional: analytic gradient / exact curve for speed & accuracy
    def grad(self, x, y, eps=1e-5):
        import numpy as np
        return np.array([-2*self.a*x, np.ones_like(x) if np.ndim(x) else 1.0])
```

## Architecture

```
run.py → viz.SimApp → world.World.step(dt):
    1. camera.cast_ray(surface)          # sense: distance + normal  (RayHit)
    2. orientation.compute(...)          # → force, torque
       autofocus.compute(...)            # → force, torque
    3. plant.step(camera, ΣF, Στ, dt)    # integrate virtual mass → new pose
```

`World.step` sums the controller wrenches with the manual teleop wrench exactly as the
admittance node sums `/teleop`, `/orientation_controller`, and `/autofocus` topics.
