"""
ONNX inference backend (offline/Pyodide path).

Two executors, same result:

* `onnxruntime` - the fast C++ engine used on native desktop.  It has NO
  Pyodide/WebAssembly wheel, so it must never be imported in the browser.
* onnxruntime-web via the JS bridge (`agrovision.offline.bridge`) - the WASM
  engine that runs in Pyodide/stlite.  It is driven from Python through the
  pyodide `js` proxy and gives the browser real CNN inference plus a
  weights-based CAM explanation (see gradcam.ONNXCAM).

If the ONNX file is missing at import we raise a clear error; model.py only
builds this backend when the asset exists (demo backend otherwise) - never a
silent degrade.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import INF
from ..util.runtime import is_pyodide
from .base import ModelBackend, Prediction
from .classes import Classes
from .gradcam import ONNXCAM

OOD_CONFIDENCE_TOP1 = 0.35


def _open_session(model_path: str, external_data=None):
    """Open an ONNX session with the fastest executor available.

    Prefers onnxruntime (native); in Pyodide/stlite it uses the
    onnxruntime-web WASM engine via the JS bridge (onnx/onnxruntime have no
    installable Pyodide wheels, and the pure-python ReferenceEvaluator is too
    slow for real photos anyway).
    """
    if is_pyodide():  # pragma: no cover - exercised in browser/Pyodide
        from ..offline import bridge
        return bridge.create_session_from_file(model_path, external_data=external_data)
    import onnxruntime as ort
    try:
        return ort.InferenceSession(model_path, providers=None)
    except Exception:
        return ort.InferenceSession(model_path)


def _run(session, output_names, feeds) -> List[np.ndarray]:
    """session.run(...) that works for both onnxruntime and ReferenceEvaluator."""
    return session.run(output_names, feeds)


def _input_names(session) -> List[str]:
    get_inputs = getattr(session, "get_inputs", None)
    if get_inputs is not None:  # onnxruntime
        return [i.name for i in get_inputs()]
    return list(session.input_names)  # ReferenceEvaluator


class ONNXBackend(ModelBackend):
    name = "onnx"

    def __init__(self, model_path=None, cam_model_path=None, classes: Optional[Classes] = None,
                 ood_threshold: float = OOD_CONFIDENCE_TOP1):
        self.classes = classes or Classes(order=[], info={})
        self.ood_threshold = ood_threshold
        model_path = Path(model_path or INF.onnx_model)
        cam_model_path = Path(cam_model_path or INF.onnx_cam_model)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session = _open_session(str(model_path))
        self.input_name = _input_names(self.session)[0]
        self.cam = None
        if cam_model_path.exists():
            try:
                external_data = None
                if is_pyodide():
                    data_path = Path(str(cam_model_path) + ".data")
                    if data_path.exists():
                        external_data = [
                            {"path": data_path.name, "data": data_path.read_bytes()}
                        ]
                cam_session = _open_session(str(cam_model_path), external_data=external_data)
                self.cam = ONNXCAM(cam_session,
                                   input_name=_input_names(cam_session)[0],
                                   input_size=INF.input_size)
            except Exception:
                self.cam = None

    def predict_topk(self, rgb: np.ndarray) -> List[Prediction]:
        from .gradcam import _preprocess_input
        x = _preprocess_input(rgb, INF.input_size)
        input_name = self.input_name

        # Test-time augmentation: average the softmax over the image and its
        # flips so a single unusual framing of the leaf is less decisive.
        # Skipped in WASM/Pyodide where 4x inference would be too slow.
        variants = [x]
        if not is_pyodide():
            variants += [
                np.flip(x, axis=1),        # horizontal flip
                np.flip(x, axis=2),        # vertical flip
                np.flip(x, axis=(1, 2)),   # 180 deg
            ]

        acc = None
        for v in variants:
            logits = _run(self.session, None, {input_name: v})[0]
            p = _softmax(np.asarray(logits[0], dtype=np.float64))
            acc = p if acc is None else acc + p
        probs = acc / len(variants)

        idx = np.argsort(probs)[::-1][: INF.top_k]
        return [Prediction(code=self.classes.code_at(i) or f"class_{i}",
                           probability=float(probs[i]), index=int(i)) for i in idx]

    def explain(self, rgb: np.ndarray, code: str) -> np.ndarray:
        if self.cam is None:
            raise RuntimeError("CAM model not available; explanation disabled.")
        index = self.classes.order.index(code) if code in self.classes.order else 0
        return self.cam.heatmap(rgb, index)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()