"""
Runtime detection.

AgroVision must run in three places with different capabilities:
  1. Native desktop/server (pip, full opencv/tensorflow).
  2. stlite/Pyodide inside the browser (reduced package set, no cv2/tf).
  3. Tests (native but deterministic).

Detecting the runtime at import time lets every module pick its backend
explicitly instead of guessing from ImportError at odd moments.
"""
from __future__ import annotations

import os
import sys


def is_pyodide() -> bool:
    return sys.platform == "emscripten" or "pyodide" in sys.modules


def is_browser() -> bool:
    return is_pyodide()


def is_native() -> bool:
    return not is_pyodide()


def runtime_name() -> str:
    if is_pyodide():
        return "pyodide"
    return "native"


def test_mode() -> bool:
    return os.environ.get("AGRO_TEST", "0") == "1"
