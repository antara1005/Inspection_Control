# Surface-Source Interface for Orientation Control — Design & Implementation Plan

## Context & goal

`orientation_control_node` currently estimates the surface normal from a **single live
compressed-depth frame** (deproject → crop a cylinder around the optical axis → PCA/RANSAC
plane fit). This is noisy.

We want orientation control to instead consume a **temporally stable, fused surface**.
`tsdf_pose_node` already produces one. Crucially, **a Zivid scanner will eventually replace
TSDF** as the high-precision source. So this is fundamentally an **interface design problem**:
define a source-agnostic surface contract that orientation control consumes, with TSDF as the
first provider and Zivid as a drop-in second provider.

### What each side does today

**`tsdf_pose_node` produces:**
- `tsdf_pose/fused_cloud` — `PointCloud2`, **XYZ only**, in `object_frame`. Normals *are*
  computed in `_extract_fused_cloud()` but **dropped** before publishing. Published only on the
  **registration cadence** (`register_every_n_frames`), so slow/bursty.
- `tsdf_pose/full_cloud` — same, larger crop.
- `tsdf_pose/depth/synthetic` (+compressedDepth) — organized depth in the **camera frame**,
  raycast from the **CAD mesh** at the registered pose, every frame.
- The TSDF/VBG itself lives in `object_frame`.
- **Fusion is currently CAD-gated:** `_depth_cb` early-returns on `not self.model_ready`, and
  `model_ready` only flips true after `_load_models` loads **both** mesh and reference cloud.

**`orientation_control_node` needs (and only needs):** a stable **centroid + surface normal
along the optical axis, expressed in `main_camera_frame` (`eoat_camera_link`)**. Everything
downstream of the centroid/normal (EMA → Kalman → PD, goal pose, error publishing) is
frame-agnostic and source-agnostic.

### Load-bearing assumption: a stale cloud is fine

The surface is static in its `frame_id`; the camera moves relative to it. The 10 Hz control loop
re-transforms the **latest** cloud with **current** TF, so the camera-frame normal stays current
even when the cloud itself is seconds old. Only genuinely newly-revealed geometry lags —
irrelevant for aligning to a known patch. This holds identically for TSDF and Zivid (both are
low-rate sources).

---

## Part 1 — The interface contract

### Topic & type
- **Type:** `sensor_msgs/PointCloud2` (standard; RViz/PCL-friendly; Open3D ↔ Zivid both map cleanly).
- **Canonical topic:** neutral, source-agnostic, set by parameter on both ends — default
  `/perception/surface_cloud`. TSDF publishes there (or publishes `tsdf_pose/fused_cloud` and is
  remapped in launch); Zivid later remaps onto the same name. **The controller never names a
  specific sensor.**

### Field layout
Little-endian, `is_dense=True`, `height=1` (treated as **unorganized** — do not depend on
organized structure, since that is the one thing that differs between sources):

| field | datatype | offset |
|---|---|---|
| `x`, `y`, `z` | FLOAT32 | 0, 4, 8 |
| `normal_x`, `normal_y`, `normal_z` | FLOAT32 | 12, 16, 20 |
| `confidence` *(optional)* | FLOAT32 | 24 |

`point_step` = 24 (no confidence) or 28 (with). **Confidence is optional and detected by field
presence** — TSDF v1 omits it (publishes 6 floats); the consumer treats absence as "all points
valid." Zivid adds it later from SNR. (If strict PCL `PointNormal` interop is wanted instead,
rename `confidence` → `curvature`; `confidence` chosen here for semantic clarity.)

### Frame convention
- The **source sets `frame_id`** to whatever frame its points are expressed in (TSDF →
  `object_frame`; Zivid → its calibrated optical frame).
- The **consumer always transforms into `eoat_camera_link`** via TF at lookup `Time(0)` (latest
  available). Points get `R·p + t`; **normals get `R·n` only** (rotation, no translation).

### Normal orientation semantics
- The **source SHOULD** publish consistently oriented (outward-facing) normals.
- The **consumer is the authority**: after transforming into the camera frame it **re-orients
  each used normal toward the camera** (enforce `normal_z ≥ 0` in `eoat_camera_link`, matching the
  current `_pca_plane_normal` sign convention). Robust to a source that gets the sign wrong.

### Confidence semantics
- FLOAT32, **higher = better**, nominal range [0, 1]. TSDF → normalized voxel weight (deferred to
  v2; Open3D `extract_point_cloud` does not hand back per-point weight trivially). Zivid →
  normalized SNR. Consumer may threshold/weight by it; absence ⇒ uniform weight.

### QoS & cadence
- **Low-rate, latest-wins:** `RELIABLE`, `KEEP_LAST` depth 1, **`TRANSIENT_LOCAL`** (latched), so a
  late-joining controller immediately gets the last surface. Both ends use the same profile.
- Publish cadence is **decoupled from the registration cadence** (new param), so the surface
  republishes faster than ~1 Hz.

---

## Part 2 — Detailed implementation plan

### Phase 0 — Transition strategy (no behavior change yet)
Add a `surface_source` parameter to `orientation_control_node`: `'depth'` (current, default) or
`'cloud'` (new). Both paths converge on a single `_emit_measurement(...)`. Lets you A/B the two
live and roll back instantly.

### Phase 1 — Publisher: `tsdf_pose_node`

**1a. Emit normals + decoupled cadence + latched QoS**
1. **Serialize normals.** Generalize `_publish_cloud` to write a structured array with
   `x, y, z, normal_x, normal_y, normal_z`. The normals already exist
   (`_extract_fused_cloud()` calls `estimate_normals`) — they are just dropped today. Orient them
   outward before publishing.
2. **Decouple cadence.** Add `surface_publish_every_n_frames` (or a dedicated timer). Extract +
   downsample + estimate-normals + publish independent of `_run_registration`, guarded by
   `integ_lock`.
3. **QoS:** publish the surface topic `RELIABLE / KEEP_LAST 1 / TRANSIENT_LOCAL`.
4. **Topic param:** `surface_cloud_topic`, default `/perception/surface_cloud`.
5. *(Defer)* confidence field.

**1b. Decouple TSDF fusion from CAD availability**

Split the single `model_ready` flag into independent capabilities:

| flag | precondition | gates |
|---|---|---|
| `tsdf_ready` | VBG created (no CAD) | **integration** + surface-cloud publishing |
| `reference_ready` | `ref_cloud` + `ref_fpfh` loaded (CAD point cloud) | **registration** (FPFH/RANSAC + ICP) |
| `mesh_ready` | `mesh` + `ray_scene` loaded (CAD mesh) | **CAD synthetic-depth render** |

Changes:
1. **Create the VBG eagerly.** Call `_reset_tsdf()` in `__init__` (needs only voxel params +
   device, no CAD); set `tsdf_ready=True`. VBG no longer waits on `_on_model_params`.
2. **Re-gate `_depth_cb`:**
   - Integrate when `self.intrinsics is not None and self.vbg is not None` — **drop `model_ready`
     from this guard.**
   - Enter `_run_registration` only if `reference_ready` (plus the existing
     `min_int_before_reg` / `register_every` cadence).
   - Call `_render_synthetic_depth` only if `mesh_ready and T_model_in_fusion is not None`.
3. **Split `_load_models` → `_load_reference()` and `_load_mesh()`**, each setting its own flag.
   `_on_model_params` loads **whichever files are present** and no longer early-returns when one is
   missing — a missing mesh disables only synthetic depth; a missing point cloud disables only
   registration; **neither blocks fusion.**
4. **Reset policy nuance.** Today any model-param change calls `_reset_tsdf()`, which would wipe
   accumulated fusion the moment CAD arrives. New rule: **reset the VBG only when an already-loaded
   non-empty path changes to a *different* non-empty path** (object identity change). The
   empty→set transition (fusing CAD-free, then CAD provided for the same object) must **keep** the
   fused data and merely flip `reference_ready` / `mesh_ready` on — so ICP starts refining against
   the surface already built.
5. **Move `full_cloud` (and the new `surface_cloud`) publishing into the decoupled, timer-driven
   path**, so both flow without CAD. Only the registration-pose and CAD-depth outputs stay dark
   until a reference/mesh appears.

**Net behavior:** on startup with no viewpoint-generation CAD, the node integrates depth, builds
the TSDF, and publishes `surface_cloud` (with normals) the moment geometry accumulates. When CAD
later becomes available, registration/ICP and the pose/TF outputs light up **without clearing** the
fused surface. Matches the eventual Zivid story — geometry capture is never CAD-gated; only
model-relative pose estimation is.

### Phase 2 — Shared utility in `inspection_control`
New module `inspection_control/surface_patch.py` (pure functions, unit-testable, reusable by any
future consumer):
- `transform_cloud(points, normals, R, t)` → camera-frame points/normals.
- `select_optical_axis_patch(points, normals, crop_radius, z_min, z_max)` → boolean mask (cylinder
  around camera Z), reusing the controller's existing crop semantics.
- `patch_centroid_normal(points, normals, confidence=None, method)` → `(centroid, normal)`:
  - **provided-normals mode:** centroid = near-axis point (min `r_xy`, matching current behavior);
    normal = (confidence-weighted) mean of patch normals, normalized, **re-oriented toward camera**;
    optional angular-outlier rejection.
  - **fallback mode:** call existing `_pca_plane_normal` / `_ransac_plane_normal` when no normals
    are present.

### Phase 3 — Consumer: `orientation_control_node`
1. **Extract `_emit_measurement(cen_s, nrm_s, stamp, surface_points)`** from the current
   `process_depth` block (≈ lines 1610–1707): the `r=-cen_s` → surface-normal pose/TF → standoff
   `d` → goal pose → pitch/yaw/roll errors → `_latest_measurement` fill → viz publishes. Both
   `cen_s` / `nrm_s` are in `eoat_camera_link`, so this block is source-independent.
2. **Legacy depth path:** keep `process_depth`, but its tail now just calls `_emit_measurement(...)`.
   Zero behavior change when `surface_source='depth'`.
3. **New cloud path:**
   - Subscribe to `surface_cloud_topic` (matching latched QoS); store latest
     `(points, normals, frame_id, stamp)` under a lock.
   - In the existing 10 Hz `process_controller` tick (or a thin callback), when
     `surface_source='cloud'`: TF `eoat_camera_link ← cloud.frame_id` → `surface_patch` utility →
     `_emit_measurement(...)`.
   - **Prefer provided normals; fall back to fitting.** EMA/Kalman/PD untouched.
4. **Guards:** empty patch ⇒ invalid measurement ⇒ existing `no_target_timeout_s` logic fires.
   Optional `max_cloud_age_s` (default 0 = ignore, since geometry is frame-anchored) for
   diagnostics.
5. **Shared params:** `crop_radius`, `crop_z_min/max` now drive both paths.

### Phase 4 — Wiring
- Launch/params (`admittance_control.launch.py` + yaml): `surface_cloud_topic` on both nodes,
  `surface_source`, `surface_publish_every_n_frames`, matched QoS.
- Keep the legacy `depth_topic` plumbing intact for rollback.

### Phase 5 — Validation
1. RViz: confirm `surface_cloud` normals look stable/outward.
2. Bag legacy vs. cloud: compare normal/pitch/yaw jitter (the stability win).
3. Run closed loop with `surface_source='cloud'`; verify `_emit_measurement` outputs (goal pose,
   errors, `surface_normal` TF) match the legacy path geometrically.
4. TF-currency test: hold the cloud fixed, move the camera, confirm the camera-frame normal tracks.
5. Failure modes: drop the publisher → `no_target_timeout` engages; subscribe late → latched cloud
   arrives.
6. **CAD-free test:** start with viewpoint-generation CAD unset → confirm fusion + `surface_cloud`
   flow; then set CAD → confirm registration/ICP start **without** clearing fused data.

### Phase 6 — Zivid readiness (documentation, no code)
A Zivid node satisfies the contract by publishing `/perception/surface_cloud` (PointCloud2 +
`normal_*`, `frame_id` = its optical frame, latched, `confidence` from SNR). **No controller
changes** — flip the publisher, keep `surface_source='cloud'`.

---

## Risks / edge cases
- **Normal transform & sign:** rotation-only transform + camera-ward re-orientation is mandatory;
  the most likely bug source.
- **Centroid definition:** preserving the "near-axis point" centroid keeps standoff/goal-pose
  semantics identical to today.
- **Threading:** lock the latest-cloud handoff between subscriber and the control timer; copy out
  under the lock.
- **Cost:** the cropped, ~3 mm-downsampled object-bbox cloud is small; full transform per 10 Hz
  tick in numpy is negligible (no KD-tree needed).
- **EMA role shrinks:** keep a light EMA on the transformed normal for motion smoothing, not noise.

## Reversible decisions made
- `confidence` vs PCL `curvature` naming.
- Deferring TSDF confidence to v2.
- Canonical topic name `/perception/surface_cloud`.
- Keeping the depth path alive behind `surface_source` rather than deleting it.
