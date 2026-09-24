"""
Model backend contract shared by every inference engine.

Every backend must:
  * predict_topk(rgb)  -> list[Prediction] sorted by probability descending
  * explain(rgb, code) -> (H, W) float32 heatmap normalized to [0,1]

The UI only talks to this interface, which is what lets the same Streamlit
pages drive TensorFlow (online) and ONNX (offline/Pyodide) transparently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Prediction:
    code: str
    probability: float
    index: int


class NotConfidentError(RuntimeError):
    """Raised when the top-1 confidence is below the out-of-distribution gate."""


class ModelBackend(ABC):
    name: str = "base"

    @abstractmethod
    def predict_topk(self, rgb: np.ndarray) -> List[Prediction]:
        ...

    @abstractmethod
    def explain(self, rgb: np.ndarray, code: str) -> np.ndarray:
        ...

    def health(self) -> str:
        return "ok"
