# Inspection Control: Perception-Driven Admittance-Based Orientation Control

## Overview
This repository presents a perception-driven orientation control framework for robotic inspection tasks. The system integrates real-time point cloud perception, orientation error estimation, and an admittance-based control strategy to enable compliant and stable end-effector alignment with target surfaces.

The framework is designed for human-in-the-loop robotic inspection, where the controller assists operators by autonomously regulating orientation while respecting physical constraints such as torque and velocity limits.

## Key Features
- Perception-driven control using surface normals from point clouds
- Admittance-based outer-loop controller for compliant behavior
- PD-based orientation regulation with physically interpretable dynamics
- Compatible with ROS 2 and MoveIt Servo pipelines
- Real-time integration of perception, control, and execution
- Validated on UR5e manipulator hardware

## System Architecture
The control pipeline consists of the following stages:

```text
Depth Image -> Point Cloud -> Surface Normal Estimation
            -> Orientation Error Computation
            -> PD Controller (Torque Output)
            -> Admittance Dynamics (Virtual Sphere Model)
            -> Velocity Commands -> Robot Execution
```

## Packages

| Package | Contents |
|---|---|
| `inspection_control` | The control nodes: admittance and orientation control, autofocus and its feature front end (`focus_volume_node`), teleop, turntable joystick, Servo logging |
| `inspection_control_interfaces` | The messages they publish: `AutofocusData`, `FocusFeature(s)`, `FocusValue`, `OrientationControlData` (previously in ViewpointGeneration's `viewpoint_generation_interfaces`), and `VirtualSphere` (below) |

`tsdf_pose_node` moved to Inspection_Perception's `tsdf_pose` package. Its node
name and topics (`/tsdf_pose/*`) are unchanged, so autofocus and orientation
control read it as before; in Inspection_Workspace the perception subsystem
starts it. The autofocus sweeps still call `MoveToPoseStamped`, viewpoint
traversal's service, from `viewpoint_generation_interfaces`.

In Inspection_Workspace this repository is the `control` subsystem:
`inspection-restart control` rebuilds and restarts it without touching the arms'
connection or the other subsystems.

## The virtual sphere

`admittance_control` integrates a virtual sphere: the plant the arm follows.
The orientation controller and autofocus place their poles on the same plant, so
all three must use the same values, and they do by construction:

- `admittance_control` owns them: `sphere_mass` (kg), `sphere_radius` (m) and
  `fluid_viscosity` (Pa·s), set in `config/admittance_control.yaml` and nowhere
  else. It refuses values that aren't finite and > 0, at startup (the node
  doesn't start) and live (the set fails with the reason, nothing changes).
- It publishes them, latched, on `/admittance_control/virtual_sphere`
  (`inspection_control_interfaces/msg/VirtualSphere`, with the derived inertia
  and drag) at startup and on every change.
- `orientation_controller` and `autofocus` follow that topic
  (`virtual_sphere_topic`) and have no sphere parameters of their own. Until the
  first message they hold a zero wrench or force and log why.

So change the plant in one place, live (or with Inspection_Workspace's
**Admittance Control** panel):

```bash
ros2 param set /admittance_control sphere_mass 3.0
ros2 topic echo /admittance_control/virtual_sphere   # what the controllers now use
```

Released, the plant coasts to rest with time constant τ = m / (6 π μ r) (the
same for roll); a swing about the surface pivot at distance d takes
(I + m d²) / (c d²), which is longer at short standoffs: with the defaults,
0.20 s and 1.07 s at 0.2 m.

The formulas (inertia 2/5 m r², Stokes drag 6 π μ r, rotational drag
2.4 π μ r³, and the swing inertia and drag about a pivot at d) live only in
`inspection_control/virtual_sphere.py`, which `debag_orientation_data` uses too.
The Zivid arm's `/zivid/admittance_control` keeps its own sphere
(`admittance_control_zivid.yaml`); nothing follows it.

`VirtualSphere` is in the base layer. If `inspection_control` is rebuilt
against a base that predates it, `admittance_control` runs on its own
parameters without publishing (teleop is unaffected), and the orientation
controller and autofocus hold zero output, all logging "rebuild the base
layer". A full `inspection-restart` rebuilds base first.

## Admittance control: state and tuning

| Interface | Type | |
|---|---|---|
| `~/status` | `std_msgs/String` (JSON), latched, 5 Hz | whether the deadman lets wrenches through (and its age), the plant mode (coupled pendulum with its pivot distance, free point mass, or idle), the output twist and whether a clamp cut it, each input's force, torque and age (teleop, orientation, autofocus), the clamps and timeouts, and the sphere with its coast time constants |
| `~/set_tuning` | `rcl_interfaces/srv/SetParametersAtomically` | sets only `sphere_mass`, `sphere_radius` and `fluid_viscosity` |

The Foxglove bridge admits `~/get_parameters` and `~/set_tuning` on both arms'
nodes (Inspection_Workspace's **Admittance Control** panel), never
`~/set_parameters`. The speed clamps stay out of reach from there:
`remote_teleop_gate` lowers them while a remote operator drives and restores
them on disarm. So do the deadman, teleop and pivot timeouts.

Every parameter update, from any setter, is checked first: `publish_rate` must
be > 0 (0 divided by zero when the timer was recreated), and the clamps and
timeouts ≥ 0 (0 still turns one off). `pivot_timeout` now applies live.

## Orientation control: state, on/off and tuning

`orientation_controller` places its gains every cycle by pole placement on the
virtual sphere: Kp, Ki and Kd follow from `controller_type` (PD or PID), `zeta`
(the damping ratio, at least 1), `v_max` and `theta_max_deg`, and the pivot
distance. Setting the `Kp`, `Ki` and `Kd` parameters has no effect, nor do
`integral_alpha`, `anti_windup_enabled`, `d_min` and `d_max`.

What it offers besides teleop's buttons (which still use the toggles):

| Interface | Type | |
|---|---|---|
| `~/status` | `std_msgs/String` (JSON), latched, 5 Hz and on every change | on/off, roll control, TF and plant ready, whether it has a surface, the pitch/yaw/roll errors (deg), the pivot distance, the gains it placed, the torque cap and command, whether it is saturated, recording and the last recording's folder |
| `~/enable_orientation_control`, `~/disable_orientation_control` | `std_srvs/Trigger` | idempotent on and off (the arm still moves only while the deadman is held) |
| `~/set_tuning` | `rcl_interfaces/srv/SetParametersAtomically` | sets only its tunable parameters: the controller, measurement and Kalman ones, `roll_control_enabled` and `object`. Not frames, topics, `data_path`, `save_data` or `orientation_control_enabled` |

The Foxglove bridge admits these and `~/get_parameters` and `~/toggle_save_data`
(Inspection_Workspace's **Orientation Control** panel), never `~/set_parameters`.

Every parameter update, from any setter, is checked before anything is applied,
and refused with the reason if the controller couldn't run on it: `zeta` below
1, `v_max` ≤ 0 or `theta_max_deg` outside (0, 180] used to stop the node with a
math error in the pole placement. Ranges must not be inside out
(`crop_z_min` < `crop_z_max`, `dmap_filter_min` < `dmap_filter_max`), choices
must be known, and `object` (it names the bag file) may only use letters,
digits, `.`, `_` and `-`.

Also: `kalman_r` and `kalman_q_ddangle` now reach the KKF filter live (they only
reached PLANT and CA before, so tuning them did nothing on the default model),
and `save_data` set as a parameter opens and closes the bag like
`~/toggle_save_data` (it used to set the flag alone). A recording's CSV and
plots land in `<data_path>/<object>_<time>/` when it stops; the bag itself is
deleted.

## Joystick and controller profiles

Everything controller-specific is in one **controller profile** in `config/`,
with a section per node: the joystick node (`joy`: which device, deadzone),
`teleop` (the mapping), `turntable_joy_node` (jog buttons) and
`remote_teleop_gate` (the layout the Remote Teleop panel converts to). Pick one
with `controller_config_file:=` (Inspection_System: `controller:=`; in
Inspection_Workspace, put it in the machine's `.env`, e.g.
`LAUNCH_ARGS=controller:=8bitdo_pro2.yaml`).

| Profile | Controller |
|---|---|
| `xbox_controller.yaml` (default) | Xbox controller, `xpad` driver |
| `8bitdo_pro2.yaml` | 8BitDo Pro 2 in D-input mode (mode switch on D), over Bluetooth |

The controls are the same on both; face buttons are assigned by position:

| Control | Does | Xbox | 8BitDo Pro 2 |
|---|---|---|---|
| **Left bumper, held** | **Deadman.** The arm and the turntable only move while it is held, under orientation control and autofocus too | LB | L |
| Left stick | Camera x / y | | |
| Right stick | Pitch / yaw; **roll** instead of yaw while the right bumper is held | | |
| Right bumper, held | Roll modifier | RB | R |
| Left / right trigger | Pull / push (along the line of sight) | LT / RT | ZL / ZR |
| D-pad | Fine x / y nudge | | |
| Turntable, with the deadman held | Counter-clockwise / clockwise | View / Start | left / right back paddle |
| Bottom face button | Toggle orientation control | A | B |
| Right face button | All stop: orientation control and autofocus off (works without the deadman) | B | A |
| Left face button | Autofocus: toggle peak recording | X | Y |
| Top face button | Autofocus: toggle drive-to-peak | Y | X |
| Guide / home | Toggle orientation-control data logging | Guide | home |
| Left stick click | Zivid capture | | |
| Right stick click | Switch the joystick between the arms (`dual_arm:=true`) | | |

Until November 2026 there was no deadman, the bumpers rolled the camera, and
the same bumpers also jogged the turntable. If roll goes the wrong way, flip
`invert_roll` in the profile.

The deadman is enforced in admittance control, so it covers every input
there, not just the sticks: `teleop` publishes it on `/teleop/deadman`
(`std_msgs/Bool`, every tick and at once on release), and `admittance_control`
applies no wrench at all (teleop, orientation control, autofocus) unless it is
true and newer than `deadman_timeout` (0.5 s). Released, the plant coasts to
rest on its drag. Orientation control and autofocus keep running meanwhile, so
they act as soon as the deadman is held again. With
`dual_arm:=true`, only the arm the joystick drives gets its deadman
(`/zivid/teleop/deadman` for the Zivid arm). An empty `deadman_topic` on the
admittance node turns the deadman off.

### Adding a controller

1. On the computer it will be used with, connected the way it will be used
   (USB or Bluetooth, and its mode switch), run Inspection_Workspace's
   `tools/joy_probe/joy_probe.sh`. It prompts for each control and prints its
   indices, in the layout the cell uses and two others.
2. Copy the closest profile, set the indices, and set the `joy` section's
   `device_name` to the name the probe's joystick log shows, so the joystick node
   opens this controller even when others are connected.
3. For remote teleop, add the layout to the Remote Teleop panel (`LAYOUTS` in
   `RemoteTeleopPanel.tsx`) under the profile's `remote_teleop_gate.layout` name.

The joystick node runs with `SDL_JOYSTICK_HIDAPI=0` (the launch file), so it
reads the kernel's evdev layout. Otherwise, where SDL can open
`/dev/hidraw*` (as in the containers), it may use its own HIDAPI driver, whose
layout differs and can change with an SDL update. With several controllers
connected, the joystick node without a `device_name` opens whichever comes
first.

### When input stops

Each stage stops the arm on its own when its input goes quiet, so a lost
joystick, a crashed node or a dropped remote link can't leave the arm moving
on the last command:

| Stage | Parameter | Behaviour |
|---|---|---|
| `joy_node` | `autorepeat_rate: 20.0` | Repeats its last message while a stick is held, so silence means the joystick is gone |
| `teleop` | `joy_timeout: 0.5` | No joystick message for 0.5 s: zero wrench |
| `turntable_joy_node` | `joy_timeout: 0.5` | No joystick message for 0.5 s: stop jogging |
| `admittance_control` | `teleop_timeout: 0.5` | No teleop wrench for 0.5 s: ignore teleop |
| `admittance_control` | `deadman_timeout: 0.5` | No deadman message for 0.5 s: ignore every wrench |
| `admittance_control` | `max_linear_speed: 0.25`, `max_angular_speed: 0.8` | Per-axis clamps on the output twist. Full stick settles at about 0.05 m/s and 0.2 rad/s, so these only catch runaways |

None of this is a safety function. The UR safety configuration on each
controller is, together with keeping people out of reach.

## Remote teleop

A remote operator (at home, over Tailscale) can drive admittance control with
a gamepad through the **Remote Teleop** panel of the `inspection-panels`
Foxglove extension. Off by default; start the control subsystem with it:

```bash
inspection-restart control remote_teleop:=true      # in Inspection_Workspace
inspection-restart control                          # back to lab-only
ros2 launch inspection_control admittance_control.launch.py remote_teleop:=true   # by hand
```

Restarting only `control` leaves the arms connected. (The argument comes from
Inspection_System's `inspection_bringup`, in the workspace's base layer: after
updating it, do one full `inspection-restart` so base is rebuilt.)

`remote_teleop_gate` is then the only path from the panel to the arm:

```text
joy_node (lab joystick) --- joy_local ---\
                                          remote_teleop_gate --- joy ---> teleop, turntable_joy_node
Remote Teleop panel ------- remote/joy --/
```

- **Not armed:** the lab joystick passes straight through; remote input is ignored.
- **Arming** (`/remote_teleop/arm`, from the panel) works only while exactly one
  panel is streaming, its deadman (left bumper) is released, and the lab
  joystick has been idle for 2 s. The gate then accepts that panel only.
- **Armed:** the panel's gamepad passes through while its stream is fresh, in
  the lab controller's layout (the profile's `remote_teleop_gate.layout`, which
  the panel converts to). The left bumper is the deadman, as at the lab. The
  turntable turns with the panel's hold buttons (browsers don't report back
  paddles). The admittance speed clamps drop to `remote_max_linear_speed`
  (0.05 m/s) and `remote_max_angular_speed` (0.2 rad/s).
- **Stream older than 0.3 s:** the gate sends the last message with the deadman released,
  which stops the arm at once, orientation control and autofocus included. **After 2 s:** it disarms, and switches
  orientation control and autofocus off.
- **Anyone touches the lab joystick:** the gate disarms and the lab joystick
  has the arm.
- **Disarm** (panel button, Escape, or leaving the panel's window) restores the
  speed clamps.
- `/remote_teleop/status` (JSON, 5 Hz) reports the state for the panel.

The Foxglove bridge whitelists only `/remote/joy` and the arm/disarm services
for this (Inspection_System's `system.launch.py`).

Before driving remotely for the first time, compare the panel's sent values
(its "Controls and sent values" section) with the lab joystick
(`ros2 topic echo /joy_local`) for the same inputs: browsers report gamepads
in a different layout, and the panel converts it to the lab's. (The 8BitDo
conversion is checked against the probe's measurements; the gamepad at home
is what's left to check.)

This is one layer of several. The cell itself must be set up for remote
operation first: UR safety limits and a remote (Reduced) mode, nobody within
reach of the arms, and an overview camera that doesn't depend on ROS.

## Focus volume

`focus_volume_node` is the feature front end of the hybrid autofocus (the
design and test plan are in the robotic-inspection notebook, "Hybrid Autofocus
Design"). Instead of one focus value over a fixed region, it tracks features
across the macro image and measures focus at each one: **AKAZE finds them, KLT
tracks them, a forward-backward check retires them.** Each feature's focus over
time is a column of the sparse focus volume the controller will consume. It
starts with the control subsystem; to leave it out:

```bash
inspection-restart control focus_volume:=false      # in Inspection_Workspace
ros2 launch inspection_control admittance_control.launch.py focus_volume:=false   # by hand
```

**What it does per frame** (`inspection_control/focus_volume.py`, no ROS, so
it also runs offline on recorded frames):

- **Spawn:** the frame is gridded (6 x 6). AKAZE runs only in cells holding
  fewer than half their quota of features, on a crop of the cell, emptiest
  cells first, a few per frame. A cell where it finds nothing is left alone for
  15 frames. Its threshold is low (1e-4), since the point of AKAZE here is
  finding structure in blur.
- **Track:** pyramidal Lucas-Kanade on half-resolution frames, checked by
  tracking back again. New features are *tentative* until they have tracked
  for 3 frames, then *active*. An active feature that fails the check
  *coasts* at the median motion of the others for up to 5 frames, and comes back
  with its history if tracking catches it again.
- **Focus:** at every feature, the Laplacian energy in a 61 px window of the
  full-resolution frame, smoothed over 0.1 s, and its ratio to the feature's
  own running max (`focus_norm`). The ratio is scale-free: 100% means as
  sharp as this feature has been. Smoothing matters: on a still scene the raw
  value varies by about 8%, and the running max of a noisy value keeps
  creeping up. The node pairs each frame with `/tsdf_pose/focal_distance_m`,
  so each feature records the distance at which it was sharpest (`z_at_max`,
  corrected for the smoothing delay).

**In Foxglove:** an Image panel on `/pi_camera/image_raw/compressed` (or the
H.264 `/pi_camera/video/foxglove`), with `/focus_volume/annotations` ticked
under annotations.

- **Rings:** active features. Colour = focus relative to the feature's best,
  red (0) to green (100%). Filled once the feature has dropped below 70% of its
  best, i.e. passed the peak of its focus curve.
- **Readouts:** the focus percentage at one feature per cell (its
  longest-tracked), with an arrow while it sharpens or blurs.
- Grey dots tentative, magenta dots coasting, cyan tails the motion over the
  last 8 frames.
- **Grid:** cells with no active feature shaded red, cells searched this frame
  outlined in blue.
- **Summary** at the top left: counts, coverage, median focus, image motion,
  focal distance, time per frame.

**For the controller:** `/focus_volume/features`
(`inspection_control_interfaces/FocusFeatures`), every frame: each feature's
id, state, position, raw and smoothed focus, running max, ratio, trend,
`z_at_max`, and a summary. Published once the base layer has been rebuilt with
those messages; until then the node logs that and publishes the annotations
only.

**Cost:** about 25 ms per 2028 x 1520 frame, so it keeps up with the camera's
30 fps, on two OpenCV threads. It pins itself to `cpu_affinity` (8-23, the lab
PC's E-cores), since the control container isn't pinned and would otherwise
share the cell's real-time cores.

**Not done yet** (test plan S2-S6): per-feature depth from
`/tsdf_pose/depth/macro_synthetic`, promotion on a coherent focus response
rather than survival alone, motion-predicted spawning, and the estimator that
turns the features into a focus error.
