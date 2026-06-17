# 2D Inspection Control Sandbox

A lightweight, interactive 2D playground for the inspection control stack
(**admittance**, **orientation**, **autofocus**). A curve stands in for the object
surface; a 2D "camera" casts a ray to measure focal distance + surface normal, and the
ported control laws close the loop. Pure `numpy` + `matplotlib` — no ROS.

The control math mirrors the real nodes so insights port straight back:

| sandbox file        | mirrors node                  | law |
|---------------------|-------------------------------|-----|
| `dynamics.py`       | `admittance_control_node.py`  | coupled **pendulum** plant about the surface pivot — generalized coords `(slide a, standoff d, swing φ)`, `I_A=I_B+m·d²` |
| `controllers.py`    | `orientation_control_node.py` | pole-placement **PD/PID torque on the swing** `φ` about the surface point (no Newton-Euler feedforward) |
| `controllers.py`    | `autofocus_node.py`           | PD **force on the standoff** `d` toward a known true peak |

The coupled-dynamics rewrite (why the old point-mass + Newton-Euler feedforward went
unstable, and the new model) is documented in
[`docs/coupled-pendulum-dynamics.md`](docs/coupled-pendulum-dynamics.md).

## Run

```bash
python sim2d/run.py                 # default sine-wall surface
python sim2d/run.py --shape circle  # circle / ellipse / sine / plane
```

Requires `numpy` and `matplotlib` (`pip install numpy matplotlib`). Use an interactive
backend (the default TkAgg/Qt on a desktop); a headless `Agg` backend will construct but
not display the window.

## Controls

Teleop meaning depends on whether orientation is engaged — **surface-pivot frame** when
ON (pendulum), **camera frame** when OFF (free body, rotation about the COM):

| key      | orientation ON (pendulum) | orientation OFF (free body) |
|----------|---------------------------|-----------------------------|
| `W`/`S`  | standoff −/+ (toward / away, changes `d`) | forward / backward along the optical axis |
| `A`/`D`  | slide the pivot along the surface tangent | left / right (camera frame) |
| `Q`/`E`  | nudge the reference `Δ` (pivot the view about the target) | rotate the camera about its COM |
| `o`      | toggle **orientation** control | |
| `f`      | toggle **autofocus** drive (to the known true peak) | |
| `space`  | reset camera + plant | |

Sliders (left): orientation tuning — `zeta`, `orient v_max` (speed: raises the torque
budget `c·d·v_max`), `theta_max`, `integral_alpha` (PID) — plus autofocus `v_max`,
`mass`, `viscosity`, and white **sensor noise** std (`dist noise σ` in m, `norm noise σ°`
in degrees). Noise is injected into the measurement, so the controllers *and* the drawn
ray/normal reflect the noisy values.
Radio buttons (left): **orientation controller (PD / PID)** and surface shape. The HUD
shows focal distance, focus value, orientation controller + swing error + `Δ`, and
autofocus state/target.

> Orientation feels slow? Raise `orient v_max` (≈1.0 settles in ~5 s vs tens of seconds
> at 0.1). PID adds integral action for steady-state/disturbance rejection.

### Coupled pendulum dynamics
The plant models the camera as a **pendulum bob** swinging about the surface contact
point (the pivot) at standoff `d`, in generalized coordinates `(a, d, φ)` = tangential
slide / standoff / swing. Each controller owns one coordinate, so they no longer fight
through a feedforward:

- **orientation** applies a pure pole-placement PD **torque on the swing** `φ` to track
  `φ_ref = angle(−n) + Δ` (`Δ` = `Q/E` reference offset → pivot the view about the target);
- **autofocus** applies a PD **force on the standoff** `d`;
- **teleop** translates in the surface frame and, by construction, does **not** swing the
  camera (the slide↔swing coupling is removed).

The swing inertia is the parallel-axis `I_A = I_B + m·d²` and the swing drag is the
reference's `b = 6πμR·d²`. Following the reference method, **only orientation is
controlled** (no position/standoff term), so with a free base the pivot can slide along
the surface — drive it with teleop. See
<https://macs-lab.github.io/robotic-inspection-orientation-control-2026/> and
[`docs/coupled-pendulum-dynamics.md`](docs/coupled-pendulum-dynamics.md).

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
    1. camera.cast_ray(surface) + noise         # sense: distance + normal  (RayHit)
    2. orientation.compute(...) → (Q_a,Q_d,Q_phi)   # swing torque
       autofocus.compute(...)   → (Q_a,Q_d,Q_phi)   # standoff force
    3. plant.step(camera, ray, ΣQ_a, ΣQ_d, ΣQ_phi, dt)   # integrate pendulum → new pose
```

`World.step` sums the controllers' **generalized efforts** with the manual teleop efforts
(tangential / standoff / swing), then the `PendulumPlant` integrates the coupled dynamics
and re-projects the pivot onto the surface via the next raycast (route b).
