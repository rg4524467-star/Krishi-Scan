"""
Reference-table loader + closeness features for the fusion layer.

conditions.json maps disease code -> {dimension: [lo, hi]} favorites.
`closeness(obs, range)` turns an observation into a 0..1 feature:
  1.0 when inside the favorable range, gaussian tail outside.

No paired multimodal training set is assumed (the project explicitly forbids
it); this table is the guardrail against fooling the classifier with
environment lookup or vice versa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import RECO_DIR

DEFAULT_TABLE_PATH = RECO_DIR / "conditions.json"

# Canonical dimension names understood by the fusion scorer.
KNOWN_DIMS = ("temp_c", "humidity_pct", "ph")


def load_reference(path=None) -> Dict[str, Dict[str, List[float]]]:
    path = path or DEFAULT_TABLE_PATH
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: Dict[str, Dict[str, List[float]]] = {}
    for code, dims in raw.items():
        if code == "note":
            continue
        out[code] = {k: [float(v[0]), float(v[1])] for k, v in dims.items() if k in KNOWN_DIMS}
    return out


def closeness(obs: float, lo: float, hi: float, soften: float = 0.35) -> float:
    """1.0 inside [lo, hi]; gaussian falloff outside scaled to the range width."""
    if lo <= obs <= hi:
        return 1.0
    width = max(1.0, hi - lo)
    dist = min(abs(obs - lo), abs(obs - hi))
    sigma = max(0.5, soften * width)
    return float(np.exp(-0.5 * (dist / sigma) ** 2))


def feature_vector(code: str, conditions: Dict[str, float], table: Dict) -> np.ndarray:
    """Vector of closeness features for one disease given observed conditions."""
    range_d = table.get(code, {})
    vec = []
    for dim in KNOWN_DIMS:
        if dim not in conditions or dim not in range_d:
            vec.append(0.5)          # neutral when dimension not observed
        else:
            lo, hi = range_d[dim]
            vec.append(closeness(conditions[dim], lo, hi))
    return np.asarray(vec, dtype=np.float64)


def present_dims(conditions: Dict[str, float]) -> List[str]:
    return [d for d in KNOWN_DIMS if d in conditions]


def favorability(code: str, conditions: Dict[str, float],
                 table: Optional[Dict] = None) -> float:
    """
    Mean closeness over the dimensions we can actually observe (temperature,
    humidity, pH).  1.0 = today's conditions sit right in the disease's
    favorable band; lower = less favorable.  Unknown dims are ignored, so a
    cold/dry day still gets a fair score from what we do know.
    """
    table = table if table is not None else load_reference()
    ranges = table.get(code, {})
    dims = [d for d in KNOWN_DIMS if d in conditions and d in ranges]
    if not dims:
        return 0.5                            # neutral when nothing observed
    vals = [closeness(conditions[d], *ranges[d]) for d in dims]
    return float(np.mean(vals))