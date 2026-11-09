"""Tests of the focus volume front end on synthetic frames (no ROS)."""

import math

import cv2
from inspection_control.focus_volume import (
    ACTIVE, COASTING, FocusVolumeConfig, FocusVolumeTracker)
import numpy as np
import pytest


def texture(h=480, w=640, seed=0):
    """Blob texture that AKAZE and KLT can both hold on to."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.0)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def shifted(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def blurred(img, sigma):
    return img if sigma <= 0 else cv2.GaussianBlur(img, (0, 0), sigma)


def by_id(result):
    return {f.track_id: f for f in result.features}


def test_rejects_unknown_operator():
    with pytest.raises(ValueError):
        FocusVolumeTracker(FocusVolumeConfig(focus_operator='nope'))


def test_first_frame_fills_the_grid_within_limits():
    cfg = FocusVolumeConfig(per_cell=6)
    res = FocusVolumeTracker(cfg).process(texture(), 0.0)
    assert res.full_detect
    assert res.n_spawned > 0.8 * cfg.per_cell * cfg.grid_cols * cfg.grid_rows
    per_cell = np.bincount([f.cell for f in res.features], minlength=len(res.cells))
    assert per_cell.max() <= cfg.per_cell
    assert res.n_active == 0                    # every new feature starts tentative


@pytest.mark.parametrize('scale', [1.0, 0.5])
def test_tracks_follow_a_translation(scale):
    cfg = FocusVolumeConfig(track_scale=scale)
    tracker = FocusVolumeTracker(cfg)
    base = texture()
    step = np.array([1.5, -1.0])
    results = [tracker.process(shifted(base, *(k * step)), k / 30.0) for k in range(8)]

    assert results[-1].n_active > 100
    assert results[-1].coverage > 0.9
    assert results[-1].median_flow_px == pytest.approx(np.hypot(*step), abs=0.2)
    early, late = by_id(results[3]), by_id(results[7])
    common = [i for i in early if i in late and late[i].state == ACTIVE]
    assert len(common) > 100
    moved = np.array([[late[i].u - early[i].u, late[i].v - early[i].v] for i in common])
    assert np.median(np.abs(moved - 4 * step), axis=0).max() < 0.3


@pytest.mark.parametrize('operator', ['laplacian', 'tenengrad'])
def test_focus_curve_through_a_blur_sweep(operator):
    """Blur, sharp, blur again: each feature peaks at the sharp frame's distance."""
    # Next to no smoothing: this checks the bookkeeping, frame by frame.
    tracker = FocusVolumeTracker(FocusVolumeConfig(
        focus_operator=operator, focus_tau=1e-3, focus_rate_tau=0.05))
    base = texture()
    sigmas = [2.0, 1.5, 1.0, 0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    zs = np.linspace(0.29, 0.31, len(sigmas))       # sharpest at z = 0.300 m
    results = [tracker.process(blurred(base, s), k / 30.0, z)
               for k, (s, z) in enumerate(zip(sigmas, zs))]

    sharpening, blurring, last = results[3], results[7], results[-1]
    active = [f for f in last.features if f.state == ACTIVE]
    assert len(active) > 100
    # Sharpest frame -> every feature's best focus, at that frame's distance.
    assert np.median([f.z_at_max for f in active]) == pytest.approx(0.300, abs=1e-6)
    assert np.mean([f.passed_peak for f in active]) > 0.9
    assert last.median_focus_norm < 0.5
    rate = [f.focus_rate for f in sharpening.features if f.state == ACTIVE]
    assert np.median(rate) > 0
    rate = [f.focus_rate for f in blurring.features if f.state == ACTIVE]
    assert np.median(rate) < 0


def test_smoothed_peak_is_found_at_the_right_distance():
    """A slow sweep at 30 fps: the smoothing delay must not shift z_at_max."""
    tracker = FocusVolumeTracker()              # default smoothing
    sharp = texture().astype(np.float32)
    soft = cv2.GaussianBlur(sharp, (0, 0), 3.0)
    n = 61
    zs = np.linspace(0.29, 0.31, n)             # sharpest at frame 30, z = 0.300 m
    for k in range(n):
        # Cross-fade: a smooth focus curve with one well-defined peak (small
        # Gaussian sigmas would leave a flat top: OpenCV's kernel degenerates).
        w = math.exp(-((k - 30) / 10.0) ** 2)
        res = tracker.process((w * sharp + (1 - w) * soft).astype(np.uint8), k / 30.0, zs[k])
    peaks = [f.z_at_max for f in res.features if f.state == ACTIVE and f.passed_peak]
    assert len(peaks) > 100
    assert np.median(peaks) == pytest.approx(0.300, abs=(zs[1] - zs[0]))


def test_still_noisy_features_stay_near_their_best():
    """Sensor noise alone must not read as defocus (the running max creeping up)."""
    tracker = FocusVolumeTracker()
    base = texture().astype(np.float32)
    rng = np.random.default_rng(1)
    for k in range(90):
        noisy = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
        res = tracker.process(noisy, k / 30.0)
    active = [f for f in res.features if f.state == ACTIVE]
    assert len(active) > 100
    assert res.median_focus_norm > 0.85
    assert np.mean([f.passed_peak for f in active]) < 0.05


def test_occluded_features_coast_then_retire():
    cfg = FocusVolumeConfig(coast_frames=3)
    tracker = FocusVolumeTracker(cfg)
    base = texture()
    for k in range(4):
        tracker.process(base, k / 30.0)
    occluded = base.copy()
    occluded[:, 320:] = 128                     # right half goes flat
    retired = {}
    states = []
    for k in range(4, 4 + cfg.coast_frames + 3):
        res = tracker.process(occluded, k / 30.0)
        states.append({f.state for f in res.features if f.u > 360})
        for reason, n in res.retired_by.items():
            retired[reason] = retired.get(reason, 0) + n
    assert COASTING in states[0]                # lost, but kept for a while
    assert not any(f.u > 360 for f in res.features)
    assert retired.get('coast_timeout', 0) > 20
    assert sum(1 for f in res.features if f.u < 300 and f.state == ACTIVE) > 50


def test_textureless_cells_back_off():
    cfg = FocusVolumeConfig(empty_cell_backoff=5)
    tracker = FocusVolumeTracker(cfg)
    img = texture()
    img[:, 320:] = 128                          # right half: nothing to find
    right = {c for c in range(cfg.grid_rows * cfg.grid_cols)
             if c % cfg.grid_cols >= cfg.grid_cols // 2 + 1}
    searched = [set(tracker.process(img, k / 30.0).cells_detected) for k in range(8)]
    assert right <= searched[0]                 # first frame: every cell
    for s in searched[1:5]:                     # then left alone while backing off
        assert not (right & s)
    assert right & searched[5]                  # and retried after the back-off


def test_no_distance_leaves_z_at_max_unknown():
    tracker = FocusVolumeTracker()
    base = texture()
    for k in range(4):
        res = tracker.process(base, k / 30.0)
    assert all(math.isnan(f.z_at_max) for f in res.features)
    assert math.isnan(res.z)
