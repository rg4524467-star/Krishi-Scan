"""
Image loading helpers that work identically online and offline.

All gatekeeper/inference code consumes a numpy uint8 RGB array in HxWx3.
This module hides the "where did the bytes come from" detail (Streamlit
upload, camera capture, filesystem path) and normalizes to that contract.
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np

from .runtime import is_pyodide

try:  # PIL is available in both runtimes (pyodide ships Pillow).
    from PIL import Image
except Exception:  # pragma: no cover - only reached if Pillow is missing
    Image = None


def to_np_rgb(image) -> Optional[np.ndarray]:
    """Convert a PIL.Image or numpy array (any common channel order) to uint8 RGB."""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:          # RGBA -> RGB
            arr = arr[..., :3]
        elif arr.shape[2] == 1:        # gray
            arr = np.repeat(arr, 3, axis=-1)
    return arr


def load_image_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes (upload / camera) to a uint8 RGB numpy array."""
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return to_np_rgb(img)
    except Exception:
        # Never let a corrupt upload take down the whole pipeline; the
        # gatekeeper will surface an explicit "unreadable image" check.
        return None


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Downscale preserving aspect ratio using PIL (fast enough in both runtimes)."""
    h, w = image.shape[:2]
    if w <= width:
        return image
    new_h = max(1, int(round(h * width / w)))
    if Image is not None:
        return np.asarray(Image.fromarray(image).resize((width, new_h)), dtype=np.uint8)
    # pure-numpy nearest fallback (used if Pillow is absent in some pyodide build)
    rows = np.linspace(0, h - 1, new_h).astype(int)
    cols = np.linspace(0, w - 1, width).astype(int)
    return image[rows][:, cols]


# Indic-capable font stack.  Arial/Helvetica lack Devanagari + Tamil glyphs, so
# the PDF/share card would render empty boxes (tofu) for hi/ta.  Nirmala UI
# (Windows) and Noto (Linux) cover Latin + Devanagari + Tamil in one family.
_FONT_PATHS = [
    ("C:/Windows/Fonts/Nirmala.ttc", [0, 2, 1]),      # Nirmala UI (Regular/Bold)
    ("C:/Windows/Fonts/arialbd.ttf", []),
    ("C:/Windows/Fonts/arial.ttf", []),
    ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", []),
    ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", []),
]

_FONT_INDEX = {}


def load_font(size: int, bold: bool = False):
    """
    Best-effort font load covering Latin + Devanagari + Tamil.

    Returns a PIL ImageFont.  Falls back to arial, then to the default bitmap
    font so drawing code never crashes on odd machines.
    """
    from PIL import ImageFont

    key = (size, bold)
    if key in _FONT_INDEX:
        return _FONT_INDEX[key]

    # Prefer a font that actually has Devanagari glyphs.
    for path, idxs in _FONT_PATHS:
        try:
            if "nirmala" in path.lower():
                idx = 1 if bold else 0          # Nirmala.ttc: 0=Regular, 1=Bold
                f = ImageFont.truetype(path, size, index=idx)
                _FONT_INDEX[key] = f
                return f
            f = ImageFont.truetype(path, size)
            _FONT_INDEX[key] = f
            return f
        except Exception:
            continue

    _FONT_INDEX[key] = ImageFont.load_default()
    return _FONT_INDEX[key]
