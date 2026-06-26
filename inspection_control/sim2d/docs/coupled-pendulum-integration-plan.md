# Integrating the Coupled Pendulum into the ROS Orientation Control Stack

**Status:** plan (2-D per-axis) · **Scope:** `orientation_control_node.py` + `admittance_control_node.py` · **Route:** (b) locally-flat + re-project, ported to the existing node graph · **Out of scope:** the §6 full-3-D tensor port from `coupled-pendulum-dynamics.md`.

---

## 1. Context — what we're fixing and why

The sim2d branch (`coupled-pendulum-dynamics.md`) proved the camera's instability and "target glide" came from **faking pivot dynamics with a Newton-Euler feedforward on a free point-mass plant**, and fixed it by integrating the **coupled pivot-referenced pendulum** with **pivot-relative diagonal drag**. The real ROS stack is structurally identical to the sim's *pre-pendulum* design, so it has the same bug. This plan ports the **2-D-per-axis** fix into the ROS nodes (pitch/yaw = two independent copies of the 2-D swing). It deliberately does **not** implement the §6 full-3-D tensor equations.

### How the sim maps onto the ROS nodes

| sim2d (2-D reference) | ROS code |
|---|---|
| `PendulumPlant.step()` | `admittance_control_node.py` (point-mass integrator) |
| `OrientationController → (0,0,τ_φ)` | `orientation_control_node.py` pole-placement → `tau_pitch/tau_yaw` |
| `AutofocusController → (0,F_d,0)` | `autofocus_node.py` wrench |
| teleop → `Q_a, Q_d` | `teleop_node.py` wrench |
| generalized efforts summed by the plant | the three wrench topics summed in admittance |

Already pendulum-correct: pole-placement at `orientation_control_node.py:1961-1982` uses `inertia_A = I_B + m·d²` and `b_damping = c·d²`. The controller already thinks in pendulum terms — only the **plant coupling** is wrong.

---

## 2. Diagnosis

### 2.1 Current model

`admittance_control_node.py:135-200` integrates a **free point mass** — sums the teleop+orientation+autofocus wrenches, subtracts **absolute-velocity** drag, integrates:

```
F_cmd = ΣF − c·v        a = F_cmd/m       v += a·dt
T_cmd = Στ − c_ang·ω    α = T_cmd/I       ω += α·dt
```

To rotate this free body about the surface contact `A` (not its COM), the orientation node adds a Newton-Euler feedforward (`orientation_control_node.py:2026-2038`):

```
r              = camera − A                  (|r| = d)
inertia_A      = I_B + m·d²
force_drag_B   = −c·lin_vel_cam              ← ABSOLUTE camera velocity  (the bug)
moment_dragB_A = r × force_drag_B
ang_acc        = (moment_dragB_A + tau_theta_vec + moment_tele_A) / inertia_A
force_B        = m·(ang_acc × r) − force_drag_B − force_teleop
tau_B          = I_B·ang_acc − moment_drag_B − torque_teleop
```

### 2.2 Why it's unstable

`v_camera = v_A + ω×r`. Using the **full** `lin_vel_cam` for the swing drag means a teleop/autofocus translation `v_A` leaks into `moment_dragB_A`, corrupts `ang_acc`, and corrupts the `force_B` fed back into the plant — an **algebraic loop through drag** that can diverge. Correct rotational drag must see only the relative part `v_camera − v_A = ω×r`.

### 2.3 The glide symptom

Orientation alone, with a lateral velocity, produces a drag moment `≈ −c·d·v_lat` about the pivot that the PD balances with a **standing angle error** → constant lateral glide ("the target keeps moving"). Pivot-relative drag makes this term vanish.

---

## 3. Decision — two routes (pick ONE)

The fragile thing is the feedforward block. You can **patch it** or **delete it**. These are alternative end-states, not an additive stack.

| | Feedforward | Relative-velocity drag | Centrifugal/Coriolis | Effort |
|---|---|---|---|---|
| **Route A: Tier 1** | patched (1 line) | ✅ | ❌ | ~1 line |
| **Route A: Tier 1 + Tier 2** | patched more | ✅ | ✅ | small, 1 file |
| **Route B: Tier 3** | **deleted**, plant rewritten | ✅ built-in | ✅ built-in | ~1 day, 2 files + 1 topic |

- **Route A** = Tier 1, optionally + Tier 2. Quick, low-risk, but keeps the feedforward.
- **Route B** = **Tier 3 alone**. It already contains everything Tier 1 + Tier 2 do (the correct drag and the coupling terms live inside the pendulum equations), and it *deletes* the lines Tier 1/2 edit. Do **not** combine B with A.

**Recommendation:** apply **Tier 1** now (immediate relief), validate, and only move to **Tier 3** later if you want the sandbox-exact passive plant. Tier 1 is a stepping stone you remove if you go to Tier 3.

---

## ROUTE A

### Tier 1 — pivot-relative drag (the core fix, required)

`orientation_control_node.py`, in `process_controller` (~L2029).

**Find:**
```python
            force_drag_B = -self.linear_drag * self.lin_vel_cam
```
**Replace:**
```python
            v_rot = np.cross(self.rot_vel_cam, r)   # camera velocity from swinging about pivot A
            force_drag_B = -self.linear_drag * v_rot
```
Everything below (`moment_dragB_A`, `ang_acc`, `force_B`, `tau_B`) is unchanged — it now just gets the correct drag. A pure teleop slide now injects **zero** swing torque; the slide is still damped downstream by the admittance plant.

**Optional consistency tweak** (~L1824): make the controller's damping match the plant's.
```python
            b_damping = self.linear_drag * d * d            # ← was this
            b_damping = self.linear_drag * d * d + self.angular_drag   # ← make it this
```

### Tier 2 — centrifugal / Coriolis coupling (optional refinement)

Adds the `d↔φ` coupling: `standoff: m·d̈ = Q_d − c·ḋ + m·d·φ̇²` and `swing: I_A·φ̈ = … − 2·m·d·ḋ·φ̇`. All edits in `orientation_control_node.py`, after Tier 1. The one new quantity is `ḋ` (standoff rate).

1. **State** (near `self._last_err_t = None`, ~L1033):
   ```python
   self.last_d = None
   ```
2. **Standoff rate** (after `inertia_A`/`b_damping`, ~L1824):
   ```python
   if self.last_d is not None and dt_ctrl > 0.0:
       ddot = (d - self.last_d) / dt_ctrl
   else:
       ddot = 0.0
   self.last_d = d
   phidot_sq = float(self.rot_vel_cam[0] ** 2 + self.rot_vel_cam[1] ** 2)   # pitch/yaw only
   ```
   (Raw `ddot` is noisy — optionally EMA-smooth it.)
3. **Coriolis on swing** (~L2032) — fold `−2·m·d·ḋ·ω` into the `ang_acc` numerator:
   ```python
   omega_swing = np.array([self.rot_vel_cam[0], self.rot_vel_cam[1], 0.0], dtype=np.float32)
   moment_coriolis_A = -2.0 * self.mass_B * d * ddot * omega_swing
   self.ang_acc = (moment_dragB_A + moment_coriolis_A
                   + tau_theta_vec + moment_tele_A) / inertia_A
   ```
4. **Centrifugal on standoff** (~L2036) — add `m·φ̇²·r` to `force_B`:
   ```python
   force_centrifugal = self.mass_B * phidot_sq * r       # m·d·φ̇² along outward optical axis
   self.force_B = (self.mass_B * np.cross(self.ang_acc, r) + force_centrifugal
                   - force_drag_B - self.force_teleop)
   ```

Caveat: Tier 2 layers more feedforward onto the point-mass plant (the fragility §2 warns about) and leans on a noisy `ḋ`; the sim rates it second-order vs. Tier 1. Skippable. The robust home for these terms is Tier 3.

---

## ROUTE B

### Tier 3 — pendulum plant (full faithful port, multi-node refactor)

**Inversion of responsibility:** the admittance node *becomes* the pendulum; the orientation node goes back to emitting only a swing torque. Per-axis 3-D (pitch/yaw = two copies of the 2-D swing, plus roll), dropping the gyroscopic `ω×(I_A·ω)` cross term — the decoupled approximation the reference endorses, **not** the §6 tensor port. Do this *instead of* Tier 1/2.

**Frame at the contact** (admittance integration frame): `r = camera − A`, `d = |r|`, `û = r/d`. Split `ω` into tilt `ω_t = ω − (ω·û)û` and roll `ω_r = (ω·û)û`.

**Step 1 — orientation node publishes the surface frame.** In `__init__`:
```python
self.pub_pivot = self.create_publisher(PointStamped, f'/{self.get_name()}/pivot_r', 10)
```
In `process_controller` (where `r` exists):
```python
ps = PointStamped()
ps.header.stamp = self.get_clock().now().to_msg()
ps.header.frame_id = self.main_camera_frame
ps.point.x, ps.point.y, ps.point.z = float(r[0]), float(r[1]), float(r[2])
self.pub_pivot.publish(ps)
```

**Step 2 — orientation node deletes the feedforward** (`orientation_control_node.py:2026-2038`). Replace the whole block with a pure torque:
```python
self.force_B = np.zeros(3, dtype=np.float32)
self.tau_B = np.array([tau_pitch, tau_yaw, tau_roll], dtype=np.float32)
```
All the error/KF/pole-placement math above is unchanged.

**Step 3 — admittance node subscribes to the pivot frame.** In `__init__`:
```python
self.r_pivot = None
self.create_subscription(PointStamped, '/orientation_controller/pivot_r',
                         self.on_pivot, qos_profile)
```
```python
def on_pivot(self, msg):
    self.r_pivot = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
    self._pivot_t = self.get_clock().now().nanoseconds * 1e-9
```

**Step 4 — admittance node: replace the integrator with the pendulum** (`admittance_control_node.py:166-177`). Also remove the old drag subtraction at `admittance_control_node.py:151-163` — drag now lives inside the EOM. Use the **summed raw** force/torque:
```python
F = teleop_F + orient_F + autofocus_F            # summed raw force (no drag subtracted)
T = teleop_T + orient_T + autofocus_T
v = np.array(self.linear_vel); w = np.array(self.angular_vel)
m, c, I_B, c_ang = self.mass, self.linear_drag, self.inertia, self.angular_drag

stale = (self.r_pivot is None) or \
        (now - getattr(self, '_pivot_t', 0.0) > 0.2)   # ~0.2 s timeout → free-flight
if (not stale) and np.linalg.norm(self.r_pivot) > 1e-4:
    r = self.r_pivot; d = float(np.linalg.norm(r)); u = r / d
    w_roll = (w @ u) * u; w_tilt = w - w_roll
    Q_d = float(F @ u); Q_a = F - Q_d * u
    tau_roll = (T @ u) * u; tau_tilt = T - tau_roll
    ddot = float(v @ u); v_sw = np.cross(w, r); v_a = v - ddot * u - v_sw
    I_A = I_B + m * d * d
    a_a    = (Q_a - c * v_a) / m
    d_ddot = (Q_d - c * ddot + m * d * float(w_tilt @ w_tilt)) / m
    a_tilt = (tau_tilt - (c*d*d + c_ang) * w_tilt - 2.0*m*d*ddot*w_tilt) / I_A
    a_roll = (tau_roll - c_ang * w_roll) / I_B
    v_a += a_a*dt; ddot += d_ddot*dt; w_tilt += a_tilt*dt; w_roll += a_roll*dt
    v = v_a + ddot*u + np.cross(w_tilt, r); w = w_tilt + w_roll
else:
    # Fallback: free point mass (= today's behaviour) so teleop flies off-surface
    v = v + (F - c * v) / max(m, 1e-9) * dt
    w = w + (T - c_ang * w) / max(I_B, 1e-9) * dt

self.linear_vel = list(v); self.angular_vel = list(w)
```
The speed clamps and twist publish below (`admittance_control_node.py:180-200`) stay unchanged.

**Nodes NOT touched:** `teleop_node.py`, `autofocus_node.py` keep publishing wrenches; the plant projects them onto the pivot frame.

---

## 4. Verification

Mirror the sim's §4 checks (rosbag `wrench_cmds` + `current_twist`, watch `yaw_error_raw`, `focal_distance_m`):

- **Stability:** orientation ON **with** teleop sliding, and ON **with** autofocus driving — both bounded/convergent; zero-effort motion decays.
- **No glide:** from a modest angle offset → swing error ≈ 0°, pivot tangential velocity → 0.
- **Decoupling:** pure teleop slide → ≈ 0 change in `tau_pitch/tau_yaw` (Route A) / `w_tilt` (Route B).
- **Centrifugal/Coriolis** (Tier 2/3): fast swing bumps `focal_distance_m` outward then settles; changing `d` mid-rotation shows a bounded swing reaction.
- **Tier 3 extras:** orientation node publishes ≈ 0 force yet camera still pivots (plant, not feedforward, does it); off-surface teleop still flies (fallback); **verify the admittance integration frame == `main_camera_frame`** or transform `r`.
- **Regression:** wrench still publishes, KF still runs, `OrientationControlData` still logs, values finite across `d_min..d_max`.

Tune speed via `v_max` (raises `tau_max = c·d·v_max`), not via extra damping clamps.

---

## 5. Recommendation

1. **Do Tier 1** (one line). Validate the glide/instability is gone.
2. Add **Tier 2** only if spin↔standoff interaction visibly matters.
3. Move to **Tier 3** only if you want the sandbox-exact passive plant — and when you do, **remove the Tier 1/2 patches** (Tier 3 replaces that block). Budget ~1 day + careful frame/timing validation.
