"""
Soil Health Card image preprocessing: deskew + adaptive threshold.

Same backend-abstraction rule as the gatekeeper: OpenCV implementation when
available, a pure-NumPy fallback otherwise (so the guided-photo path never
silently dies in the offline runtime - though full OCR itself stays online-only
and offline degrades to manual entry with a clear notice).
"""
from __future__ import annotations

import numpy as np

from ..gatekeeper.backends import get_backend


def deskew_and_prepare(rgb: np.ndarray, target_min: int = 1600) -> np.ndarray:
    """
    Returns a preprocessed grayscale image ready for OCR (deskewed, contrast
    enhanced, thresholded).  Pure-NumPy fallback keeps the pipeline alive
    offline; it performs edge-preserving contrast enhancement instead of a true
    projective deskew, which is documented in the fallback path.
    """
    backend = get_backend()
    if backend.has_cv2:
        return _deskew_cv2(rgb, target_min)
    return _prepare_numpy(rgb)


def _deskew_cv2(rgb: np.ndarray, target_min: int) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = max(1.0, target_min / max(h, w))
    if scale > 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # Local contrast boost - helps washed-out phone photos of glossy cards.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Estimate skew angle from the longest text line via minAreaRect on contours.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    angle = 0.0
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = [c for c in contours if cv2.boundingRect(c)[2] > 0.3 * gray.shape[1]]
    if lines:
        # Rotate back if the card is tilted by more than ~0.5 deg.
        rect = cv2.minAreaRect(max(lines, key=cv2.contourArea))
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)
    return gray


def _prepare_numpy(rgb: np.ndarray) -> np.ndarray:
    """Fallback: grayscale + local contrast normalization (no cv2)."""
    gray = get_backend().to_gray(rgb).astype(np.float64)
    # Local mean/std contrast stretch via a coarse sliding-window approximation.
    k = 31
    mean = _box_filter(gray, k)
    sq = _box_filter(gray * gray, k)
    std = np.sqrt(np.maximum(sq - mean * mean, 1e-6))
    norm = (gray - mean) / (std + 1.0)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    return (norm * 255).astype(np.uint8)


def _box_filter(img: np.ndarray, k: int) -> np.ndarray:
    """Integral-image box filter in pure NumPy (O(1) per pixel)."""
    h, w = img.shape
    cum = np.cumsum(np.cumsum(img, axis=0), axis=1)
    # Pad cumulative sums for clean window math.
    cum_p = np.zeros((h + 1, w + 1))
    cum_p[1:, 1:] = cum
    r = k // 2
    # Window sums at (i,j) == sum over rows i-r..i+r, cols j-r..j+r.
    hh, ww = h, w
    xs = np.arange(ww)
    ys = np.arange(hh)
    x0 = np.clip(xs - r, 0, w)
    x1 = np.clip(xs + r + 1, 0, w + 1)
    y0 = np.clip(ys - r, 0, h)
    y1 = np.clip(ys + r + 1, 0, h + 1)
    area = (x1 - x0)[None, :] * (y1 - y0)[:, None]
    sums = (cum_p[y1[:, None], x1[None, :]]
            - cum_p[y0[:, None], x1[None, :]]
            - cum_p[y1[:, None], x0[None, :]]
            + cum_p[y0[:, None], x0[None, :]])
    return sums / np.maximum(area, 1)