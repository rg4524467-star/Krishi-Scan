"""
Backend factory - the single place that decides WHERE inference runs.

Selection order:
  1. Pyodide/browser            -> ONNX backend (asset must be present)
  2. Native + fine-tuned ckpt   -> TensorFlow backend
  3. Anything else              -> DemoBackend (explicitly labeled in UI)

All three share the ModelBackend interface so the rest of the app is
backend-agnostic.  Loading is cached: model weights are large and should load
once per session (a crucial detail for the weak-connection PWA target).
"""
from __future__ import annotations

import threading
from typing import Optional

from ..config import INF
from .base import ModelBackend
from .classes import load_classes

_lock = threading.Lock()
_model: Optional[ModelBackend] = None


def inference_kind() -> str:
    """Human-readable label of the active engine (used by UI/README)."""
    if INF.onnx_model.exists():
        return "onnx"
    if INF.demo_mode or not INF.checkpoint_path:
        return "demo"
    return "tensorflow"


def get_model(force_reload: bool = False) -> ModelBackend:
    global _model
    with _lock:
        if _model is not None and not force_reload:
            return _model
        classes = _safe_classes()
        if INF.onnx_model.exists():
            from .onnx_backend import ONNXBackend
            _model = ONNXBackend(classes=classes)
        elif INF.demo_mode or not INF.checkpoint_path:
            from .demo_backend import DemoBackend
            _model = DemoBackend(classes=classes)
        else:
            from .tf_backend import TFBackend
            _model = TFBackend(classes=classes)
        return _model


def _safe_classes():
    try:
        return load_classes()
    except Exception:
        from .classes import Classes
        return Classes(order=[], info={})


def reset_model() -> None:
    global _model
    with _lock:
        _model = None