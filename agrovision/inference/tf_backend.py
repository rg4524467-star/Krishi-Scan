"""
TensorFlow inference backend (online/cloud path).

Loads a fine-tuned MobileNetV2 checkpoint (h5/keras SavedModel) if present.
When no checkpoint is configured this backend is not used - model.py falls back
to the deterministic demo backend so the app still runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import INF
from .base import ModelBackend, Prediction
from .classes import Classes
from .gradcam import TFGradCAM

# Out-of-distribution gate: below this top-1 score we do not force a label.
OOD_CONFIDENCE_TOP1 = 0.35


class TFBackend(ModelBackend):
    name = "tensorflow"

    def __init__(self, checkpoint_path: Optional[str] = None, classes: Optional[Classes] = None,
                 ood_threshold: float = OOD_CONFIDENCE_TOP1, kind: str = "keras"):
        self.classes = classes or Classes(order=[], info={})
        self.ood_threshold = ood_threshold
        self.kind = kind
        self.model, self.gradcam = self._build(checkpoint_path)

    def _build(self, checkpoint_path: Optional[str]):
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        path = checkpoint_path or INF.checkpoint_path
        if not path:
            raise FileNotFoundError("TF backend requires a checkpoint path.")
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = tf.keras.models.load_model(path, compile=False)
        return model, TFGradCAM(model, input_size=INF.input_size)

    def predict_topk(self, rgb: np.ndarray) -> List[Prediction]:
        x = np.asarray(rgb, dtype=np.float32) / 255.0
        if x.shape[:2] != (INF.input_size, INF.input_size):
            from PIL import Image
            x = np.asarray(Image.fromarray(rgb).resize(
                (INF.input_size, INF.input_size)), dtype=np.float32) / 255.0
        x = x[None, ...]
        probs = self.model.predict(x, verbose=0)[0]
        idx = np.argsort(probs)[::-1][: INF.top_k]
        return [Prediction(code=self.classes.code_at(i) or f"class_{i}",
                           probability=float(probs[i]), index=int(i)) for i in idx]

    def explain(self, rgb: np.ndarray, code: str) -> np.ndarray:
        index = self._index_of(code)
        return self.gradcam.heatmap(rgb, index)

    def _index_of(self, code: str) -> int:
        if code in self.classes.order:
            return self.classes.order.index(code)
        return 0