"""Utility helpers: runtime detection, image IO, small misc helpers."""
from .runtime import is_browser, is_pyodide, runtime_name
from .image import load_image_bytes, to_np_rgb, resize_to_width
