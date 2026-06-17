# Coupled Pendulum Dynamics for the 2D Inspection Sandbox

**Status:** planned · **Scope:** `sim2d/` dynamics + controllers · **Route:** (b) locally-flat + re-project

This document records (1) the diagnosis of why the current orientation control goes
unstable, and (2) the plan to replace the point-mass plant with a coupled
pivot-referenced pendulum so orientation applies a *pure torque about the surface target
point* and the dynamics handle the coupling/drag correctly.

---

## 1. Diagnosis

### 1.1 Current model

`dynamics.py` integrates the camera as a **free point mass** (the admittance plant):

```
m·v̇_c = ΣF − c·v_c          c     = 6πμR          (linear)
I_B·ω̇ = Στ − c_ang·ω        c_ang = 2.4πμR³       (angular)
```

To make a *free* body rotate about a point that is **not** its center of mass (the
surface contact point `A`), the orientation controller supplies, in addition to the
swing torque, the centripetal force via a **Newton-Euler feedforward**
(`orientation_control_node.py`, mirrored in `controllers.py`):

```
r            = camera − A                (lever arm, |r| = d)
I_A          = I_B + m·d²                (parallel-axis inertia about the pivot)
force_drag_B = −c·v_camera
moment_drag_A = r × force_drag_B
α            = (moment_drag_A + τ_θ) / I_A
force_B      = m·(α × r) − force_drag_B
τ_B          = I_B·α − moment_drag_B
```

`force_B`/`τ_B` pre-subtract drag so that the admittance plant — which re-adds it —
reproduces the pivot motion `a_B = α × r`.

### 1.2 Why it is unstable

The feedforward is only valid when the camera's velocity is **exactly** the pivot
rotation `ω × r`. Two independent effects break that assumption:

1. **Drag uses the absolute camera velocity, not the velocity relative to the target.**
   `moment_drag_A = r × (−c·v_camera)` and `v_camera = v_A + ω×r`. When teleop or
   autofocus inject a translation `v_A` that is *not* `ω×r`, that translation leaks into
   `moment_drag_A`, corrupts `α`, and therefore corrupts the very force `force_B` that is
   fed back into the plant. This closes an **algebraic loop through drag** that can go
   unstable. The physically correct rotational drag should see only the relative
   (rotational) part `v_camera − v_A = ω×r`.

2. **The reproduced motion `a_B = α×r` no longer matches the real COM acceleration**
   once teleop/autofocus also push the COM, so the "pivot motion" is wrong.

### 1.3 Symptom even in isolation: target drift

With orientation alone (free base, no position control), a lateral camera velocity
`v_lat` produces a drag moment `≈ −c·d·v_lat` about the pivot, which the PD balances with
a **standing angle error** (`τ_θ = Kp·e`). That equilibrium is a constant lateral
**glide** with a fixed angle bias — the "target point keeps moving" behavior. With
relative-velocity drag this term vanishes and the glide damps out.

### 1.4 Root cause

Faking pivot dynamics with feedforward on a decoupled point-mass plant is fragile. The
fix is to **integrate the true coupled dynamics**: each controller applies a generalized
effort on one coordinate, and drag/coupling fall out of the equations of motion
dissipatively (so they cannot destabilize a passive plant).

The reference method confirms orientation is controlled **exclusively** (pivot torque,
virtual inertia about the pivot with `b = 6πμR·d² = c·d²`, Newton-Euler conversion),
with no position/standoff term:
<https://macs-lab.github.io/robotic-inspection-orientation-control-2026/>.

---

## 2. New model — coupled pivot-referenced pendulum

### 2.1 Geometry (2D)

Local frame at the contact from the raycast: tangent **t**, outward normal **n**.
Pivot/target **A** on the surface; optical axis **û(φ) = (cos φ, sin φ)**; camera (bob)
`B = A − d·û`. Generalized coordinates:

```
q = (a, d, φ)     a = tangential slide of the pivot along t
                  d = standoff (camera ↔ surface)
                  φ = swing (optical-axis angle)
```

Bob velocity: `v_B = ȧ·t − ḋ·û − d·φ̇·û⊥`.

### 2.2 Equations of motion — decoupled pendulum

The full Lagrangian (`T = ½m|v_B|² + ½I_B φ̇²`, with `α1 = t·û`, `α2 = t·û⊥`) yields a
configuration-dependent mass matrix whose **off-diagonals couple the tangential slide to
the swing** — dragging/accelerating the bob sideways torques the pivot. That cross
coupling is exactly what we do *not* want (teleop should translate, not swing the camera)
and it drives an instability (§5). So we use the **decoupled pendulum**: one clean
second-order system per coordinate, coupled only through the standoff-dependent swing
inertia `I_A = I_B + m·d²` and the centrifugal/Coriolis terms:

```
a : m·ä   = Q_a − c·ȧ
d : m·d̈   = Q_d − c·ḋ + m·d·φ̇²                     (centrifugal: spin flings standoff)
φ : I_A·φ̈ = Q_φ − (c·d² + c_ang)·φ̇ − 2·m·d·ḋ·φ̇    (Coriolis)
```

This keeps the pendulum essence (parallel-axis swing inertia, centrifugal & Coriolis
coupling between `d` and `φ`) while dropping the slide↔swing cross terms.

### 2.3 Generalized forces  `Q = Q_drag + Q_control`

**Drag** — **relative to the pivot**, diagonal. The swing sees only `−c·d²·φ̇` (the
reference's `b = 6πμR·d²`), never the bob's absolute velocity — this is the
relative-velocity fix:

```
Q_drag,a = −c·ȧ
Q_drag,d = −c·ḋ
Q_drag,φ = −(c·d² + c_ang)·φ̇
```

Diagonal and positive ⇒ each coordinate is independently **dissipative/passive** ⇒
teleop translation cannot pump energy into the swing.

**Control** — one coordinate each:

| Controller   | Effort | Law |
|--------------|--------|-----|
| Orientation  | `Q_φ += τ_θ` | pole-placement **PD or PID** on `φ − φ_ref`, `φ_ref = atan2(−n) + Δ`, poles from `(I_A, b=c·d²+c_ang, τ_max=c·d·v_max, θ_max, ζ)`; PID adds `p3 = integral_alpha·p2` (per `orientation_control_node.py`) |
| Autofocus    | `Q_d += F`   | PD on `d − d_focus`, `pole_placement_pd(m, c, …)` |
| Teleop       | `Q_a, Q_d`   | surface-frame translation (slide along **t**, standoff along **n**) |
| Teleop (rot) | `Δ`          | sets the reference swing offset → pivot the view about the target |

### 2.4 Per-tick algorithm (route b)

1. `ray = cast_ray(surface)`, inject noise (existing `World._apply_noise`) → `A, n, t, d, φ`.
2. Decompose persistent world velocity `(v_c, ω)` into `(ȧ, ḋ, φ̇)` in this tick's frame
   (2×2 solve on basis `{t, −û}` for `v_c + d·ω·û⊥`; `φ̇ = ω`).
3. Evaluate the decoupled accelerations `(ä, d̈, φ̈)` from §2.2 (no matrix inverse).
4. Integrate `(ȧ,ḋ,φ̇) += q̈·dt`; recompose `v_c = ȧ·t − ḋ·û − d·φ̇·û⊥`, `ω = φ̇`.
5. `camera.pos += v_c·dt; camera.theta += ω·dt`. **Next tick's raycast re-projects** the
   pivot onto the true (curved, noisy) surface — the route-(b) corrector.

Edge cases: no-hit → free point-mass fallback so teleop still works off-surface;
near-grazing (`t ∥ û`) → clamp/skip the pivot coupling.

---

## 3. Files to change

- **`sim2d/dynamics.py`** — replace point-mass `AdmittancePlant` with `PendulumPlant`:
  keep `set_inertia_and_drag`; persist world `(lin_vel, ang_vel)`; new
  `step(camera, ray, Q_a, Q_d, Q_phi, dt)` doing decompose → solve → recompose; frame
  helper (`t, û, α1, α2`); no-hit and grazing handling.
- **`sim2d/controllers.py`** — controllers return **generalized efforts**:
  - `OrientationController.compute(...) -> (0, 0, τ_θ)`; target `φ_ref = atan2(−n) + delta`;
    gains referenced to `I_A`, `c·d² + c_ang`; drop the Newton-Euler block. Add `self.delta`.
  - `AutofocusController.compute(...) -> (0, F_d, 0)` (PD unchanged).
  - Keep `pole_placement_pd`, `signed_angle`, `_RateEstimator`.
- **`sim2d/world.py`** — aggregate generalized efforts (replace force/torque summation);
  teleop fields become `teleop_tangential`, `teleop_standoff`, `teleop_delta_rate`; call
  `plant.step(camera, ray, Q_a, Q_d, Q_phi, dt)`.
- **`sim2d/viz.py`** — remap keys: `W/S` → standoff `Q_d`, `A/D` → tangential `Q_a`,
  `Q/E` → nudge `orientation.delta` when orientation ON else manual `Q_phi`. HUD shows
  `d`, swing error, `Δ`, focus.
- **`sim2d/README.md`** — update mirror table, controls, and dynamics section.

---

## 4. Verification

- **Stability (the bug):** orientation ON **with** teleop sliding, and ON **with**
  autofocus driving — both converge and stay bounded; zero-effort motion decays (passive).
- **No glide:** orientation from a modest offset → ~0° error, pivot tangential velocity → 0.
- **Pivot on surface:** per-tick re-projection error small for circle/sine/ellipse.
- **Decoupling:** teleop slide moves the pivot without inducing swing; `Q/E` pivots about a
  roughly-fixed target via Δ; autofocus still drives `d → d_focus`.
- **Coupling sanity:** spin increases `d` (centrifugal); extending `d` while spinning shows
  the Coriolis reaction.
- **Regression:** all shapes raycast/render; noise sliders still perturb the measured
  `d`/normal feeding the plant; `SimApp` builds and callbacks run.

---

## 5. Implementation findings (stability)

Two refinements were needed beyond the initial design, both stemming from the sandbox's
**high-drag regime** (`mass≈2.5, radius≈0.65, viscosity≈1.0` ⇒ `c·d² ≈ 50`, far more
damped than the near-inertial real system at `viscosity≈0.001`):

1. **Decouple the drag and inertia (§2.2–2.3) — the real fix.** The exact Lagrangian's
   slide↔swing cross terms let a fast tangential slide torque the swing; the camera
   tilted, the measured `d` grew, and that fed back (tilt → larger `d` → larger
   disturbance) into a runaway. A diagonal mass + pivot-relative diagonal drag removes
   the cross-coupling: teleop slide now produces **exactly zero** swing (the intended
   "teleop translates, doesn't pivot" behavior) and each coordinate is passive.

2. **No `Kd` clamp — match the nodes.** With the decoupled swing dynamics, the
   pole-placement gains (`Kd = −I_A(p1+p2) − b`, possibly negative) give a closed loop
   whose *total* damping `b + Kd = 2ζωₙ·I_A > 0` is always positive, so it is stable
   **without** clamping. An earlier stopgap clamped `Kd ≥ 0`, but that made the
   controller heavily over-damped/slow; it was removed so PD/PID reproduce
   `orientation_control_node.py` exactly. **Speed is tuned via `v_max`** (raises the
   torque budget `τ_max = c·d·v_max` → higher `ωₙ`): settling drops from tens of
   seconds at `v_max=0.1` to ~5 s at `v_max=1.0`.

**PD / PID** (`controller_type`) follow the node: PID adds a slow integral pole
`p3 = integral_alpha·p2` with conditional anti-windup and an `ie_clamp` limit. Both,
plus `zeta`, `v_max`, `theta_max`, `integral_alpha`, are exposed as GUI sliders/radio.

Residual (acceptable): sliding *aggressively across a curved surface* makes the swing
**lag** the fast-changing normal (bounded error, stays on the surface) — a bandwidth
limit, not an instability; raise `v_max` to tighten it.
