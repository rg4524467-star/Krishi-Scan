"""
Deterministic demo backend.

Used ONLY when a real checkpoint/ONNX export is not present, so the entire app
flow (gatekeeper -> classify -> heatmap -> recommendation) can be exercised on
a phone/PWA before model weights are downloaded.  It is explicitly NOT a real
classifier: results are derived deterministically from the image content hash
and are clearly flagged as demo output in the UI (classifier.model_warning).

The heatmap is a true edge-density map (Sobel magnitude) - it reflects where
structure exists in the leaf but is not a model attribution.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..config import INF
from .base import ModelBackend, Prediction

_DEMO_POOL = ("rice_leaf_blast", "tomato_late_blight", "wheat_leaf_rust",
              "cotton_leaf_curl", "sugarcane_red_rot", "maize_common_rust")


class DemoBackend(ModelBackend):
    name = "demo"

    def __init__(self, classes, seed_offset: int = 0):
        from .classes import Classes
        self.classes: Classes = classes or Classes(order=[], info={})
        self.seed_offset = seed_offset

    def predict_topk(self, rgb: np.ndarray) -> List[Prediction]:
        # Deterministic hash of a tiny downsampled image.
        small = rgb[:: max(1, rgb.shape[0] // 8), :: max(1, rgb.shape[1] // 8)]
        h = hash(small.tobytes()) & 0xFFFFFFFF
        rng = np.random.RandomState((h + self.seed_offset) & 0xFFFFFFFF)
        pool = [c for c in _DEMO_POOL if c in self.classes.order]
        if not pool:
            pool = self.classes.order[: min(4, self.classes.size)]
        picks = rng.choice(pool, size=min(3, len(pool)), replace=False)
        raw = rng.dirichlet(np.ones(len(picks))) * 0.82
        raw = np.sort(raw)[::-1]
        return [Prediction(code=str(c), probability=float(p), index=self.classes.order.index(c))
                for c, p in zip(picks, raw)]

    def explain(self, rgb: np.ndarray, code: str) -> np.ndarray:
        g = np.asarray(rgb, dtype=np.float64).mean(axis=2)
        gx = np.abs(np.diff(g, axis=1))
        gy = np.abs(np.diff(g, axis=0))
        mag = np.abs(gx[:-1, :]) + np.abs(gy[:, :-1])
        mag = np.pad(mag, ((0, 1), (0, 1)))
        mn, mx = float(mag.min()), float(mag.max())
        return (mag - mn) / (mx - mn + 1e-8)