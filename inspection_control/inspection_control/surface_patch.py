#!/usr/bin/env python3
"""
Source-agnostic surface-patch utilities for orientation control.

These pure-numpy helpers turn a denoised surface cloud (TSDF fused cloud today,
Zivid later) into the single thing the orientation controller needs: a stable
**centroid + surface normal along the optical axis, expressed in the camera
frame**. They make no ROS or Open3D calls so they are trivially unit-testable and
reusable by any future consumer.

Conventions (matching the legacy `_pca_plane_normal` path in
`orientation_control_node`):
  * Points are processed in the camera frame (`eoat_camera_link`), camera looking
    along +Z, so observed surface points have z > 0.
  * The reported normal is re-oriented into the +Z hemisphere (`normal_z >= 0`),
    i.e. pointing along the camera view direction; the controller drives pitch/yaw
    to align the camera Z axis with it.
  * The centroid is the patch point closest to the optical axis (min radial
    distance), preserving standoff/goal-pose semantics.
"""

import numpy as np
import numpy.linalg as LA


def transform_points_normals(points, normals, R, t):
    """Transform points (and optional normals) by a rigid transform.

    Args:
        points:  (N, 3) array in the source frame.
        normals: (N, 3) array or None.
        R:       (3, 3) rotation, t: (3,) translation mapping source -> target.

    Returns:
        (points_out, normals_out). `normals_out` is None if `normals` is None or
        length-mismatched. Normals get the rotation only (no translation).
    """
    points = np.asarray(points, dtype=np.float64)
    pts_out = (R @ points.T).T + np.asarray(t, dtype=np.float64)
    nrm_out = None
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape == points.shape:
            nrm_out = (R @ normals.T).T
    return pts_out, nrm_out


def optical_axis_patch_mask(points, crop_radius, z_min, z_max):
    """Boolean mask selecting a cylinder around the optical axis (camera +Z).

    Args:
        points: (N, 3) in camera frame.
    Returns:
        (N,) bool mask: z in [z_min, z_max] and radial distance <= crop_radius.
    """
    points = np.asarray(points, dtype=np.float64)
    z = points[:, 2]
    r2 = points[:, 0] ** 2 + points[:, 1] ** 2
    return (z >= z_min) & (z <= z_max) & (r2 <= crop_radius * crop_radius)


def near_axis_centroid(points):
    """Patch point closest to the optical axis (min radial distance).

    Matches the legacy `_pca_plane_normal` centroid convention so standoff and
    goal-pose math are unchanged.
    """
    points = np.asarray(points, dtype=np.float64)
    r_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    return points[int(np.argmin(r_xy))]


def nearest_axis_point(points, z_min, z_max):
    """Return the point closest to the optical axis (min x^2 + y^2) within the
    depth window [z_min, z_max], or None if no point qualifies.

    Independent of any radial crop — used for a robust focal-distance estimate
    (distance to the surface point on the camera's optical axis).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return None
    z = pts[:, 2]
    mask = (z >= z_min) & (z <= z_max)
    if not np.any(mask):
        return None
    sel = pts[mask]
    r2 = sel[:, 0] ** 2 + sel[:, 1] ** 2
    return sel[int(np.argmin(r2))]


def aggregate_normal(normals, confidence=None, max_angle_deg=None):
    """Combine per-point patch normals into one stable unit normal.

    Signs are first made consistent (flipped into the +Z hemisphere) so opposing
    normals do not cancel, then a (confidence-weighted) mean is taken. Optionally
    rejects outliers whose angle from the first-pass mean exceeds `max_angle_deg`
    and recomputes.

    Args:
        normals:       (M, 3) patch normals in camera frame.
        confidence:    (M,) non-negative weights, or None for uniform.
        max_angle_deg: optional angular outlier threshold (degrees).

    Returns:
        (3,) unit normal with normal_z >= 0, or None if it cannot be resolved.
    """
    n = np.asarray(normals, dtype=np.float64)
    if n.ndim != 2 or n.shape[0] == 0:
        return None
    n = n.copy()

    # Consistent hemisphere (camera view direction) before averaging.
    flip = n[:, 2] < 0.0
    n[flip] = -n[flip]

    if confidence is not None and len(confidence) == len(n):
        w = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, None)
    else:
        w = np.ones(len(n), dtype=np.float64)

    def _weighted_unit(vecs, weights):
        m = (vecs * weights[:, None]).sum(axis=0)
        norm = LA.norm(m)
        if norm < 1e-9:
            return None
        return m / norm

    mean = _weighted_unit(n, w)
    if mean is None:
        return None

    if max_angle_deg is not None and len(n) >= 4:
        cos_thresh = np.cos(np.radians(max_angle_deg))
        keep = (n @ mean) >= cos_thresh
        if keep.sum() >= 3:
            refined = _weighted_unit(n[keep], w[keep])
            if refined is not None:
                mean = refined

    if mean[2] < 0.0:
        mean = -mean
    return mean


def estimate_patch_normal(points_cam, normals_cam, crop_radius, z_min, z_max,
                          confidence=None, min_points=10, max_angle_deg=30.0):
    """High-level helper: select the optical-axis patch and report centroid+normal.

    Args:
        points_cam:  (N, 3) cloud already in the camera frame.
        normals_cam: (N, 3) normals in the camera frame, or None.
        confidence:  (N,) per-point confidence, or None.

    Returns:
        dict with keys {'centroid', 'normal', 'patch_points', 'has_normal'}, or
        None if the patch has fewer than `min_points`. When `normals_cam` is None,
        'normal' is None and 'has_normal' is False so the caller can fall back to
        plane fitting on 'patch_points'.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.shape[0] == 0:
        return None

    mask = optical_axis_patch_mask(points_cam, crop_radius, z_min, z_max)
    patch = points_cam[mask]
    if patch.shape[0] < min_points:
        return None

    centroid = near_axis_centroid(patch)

    normal = None
    if normals_cam is not None:
        nrm = np.asarray(normals_cam, dtype=np.float64)[mask]
        conf = None
        if confidence is not None:
            conf = np.asarray(confidence, dtype=np.float64)[mask]
        normal = aggregate_normal(nrm, conf, max_angle_deg=max_angle_deg)

    return {
        'centroid': centroid.astype(np.float32),
        'normal': None if normal is None else normal.astype(np.float32),
        'patch_points': patch.astype(np.float32),
        'has_normal': normal is not None,
    }
