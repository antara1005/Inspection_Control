"""Focus-metric functions for passive (frame-based) autofocus.

Every metric takes a BGR image (numpy array) and returns
``(focus_value: float, image_out: np.ndarray)`` where ``image_out`` is an
RGB visualization of the metric response. Select a metric by name via
``compute(name, image)`` or the ``METRICS`` registry.
"""

import cv2
import numpy as np


def sobel(image_in):
    gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    sobel_image = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=1, dy=1, ksize=3)
    focus_value = sobel_image.var()
    image_out = cv2.convertScaleAbs(sobel_image)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)
    return focus_value, image_out


def squared_gradient(image_in):
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Finite differences, cropped to matching dimensions.
    gradient_x = np.diff(gray_image, axis=1)[:-1, :]   # (H-1, W-1)
    gradient_y = np.diff(gray_image, axis=0)[:, :-1]   # (H-1, W-1)

    squared = gradient_x ** 2 + gradient_y ** 2
    focus_value = np.var(squared)

    gradient_magnitude = np.sqrt(squared).astype(np.float32)
    normalized = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return focus_value, image_out


def fswm(image_in):
    """Frequency-Selective Weighted Median: median high-pass with center weighting."""
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY).astype(np.float64)

    ksize = 5  # median kernel (odd) — acts as a low-pass filter
    median_filtered = cv2.medianBlur(
        gray_image.astype(np.uint8), ksize).astype(np.float64)

    # High-pass: original minus low-pass.
    high_freq = np.abs(gray_image - median_filtered)

    # Gaussian center weighting (emphasizes image center).
    rows, cols = high_freq.shape
    center_y, center_x = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    sigma_weight = max_distance / 3
    weights = np.exp(-(distance ** 2) / (2 * sigma_weight ** 2))

    focus_value = np.sum(high_freq * weights)

    normalized = cv2.normalize(
        high_freq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return focus_value, image_out


def sobel_laplacian(image_in):
    gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    combined = sobel_magnitude + np.abs(laplacian)
    focus_value = np.var(combined)

    normalized = cv2.normalize(
        combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return focus_value, image_out


# Registry: parameter name -> metric function.
METRICS = {
    'sobel': sobel,
    'squared_gradient': squared_gradient,
    'fswm': fswm,
    'sobel_laplacian': sobel_laplacian,
}


def compute(name, image_in):
    """Run the named metric. Raises KeyError if the name is unknown."""
    try:
        metric_fn = METRICS[name]
    except KeyError:
        raise KeyError(
            f"unknown focus metric '{name}'; valid: {sorted(METRICS)}")
    return metric_fn(image_in)
