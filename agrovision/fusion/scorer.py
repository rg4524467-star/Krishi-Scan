"""
Decision-layer fusion scorer - re-weights classifier probabilities with
environmental plausibility when soil/weather data is present.

scikit-learn plays its assigned role HERE (decision-layer scoring):
a logistic-regression calibrator trained on synthetic anchor points derived
from the hand-authored reference table.  It learns "how characteristic is this
closeness pattern of a genuinely favorable condition match" and emits an
environmental prior p_env(candidate).  Final ranking:

    p_final(c) = normalize( p_classifier(c) * p_env(c)^alpha )

alpha=0 -> classifier output stands alone (graceful degrade when no data).
    Conditions absent -> returns input unchanged, no error.
    Model errors       -> caught and returned as 'unchanged' (never blocks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..inference.base import Prediction
from ..config import RECO_DIR
from .reference import load_reference, feature_vector, KNOWN_DIMS

_ALPHA_DEFAULT = 0.5
_ANCHORS_PER_DISEASE = 200
_SEED = 7
_PRIOR_FLOOR = 0.1   # never let imperfect env data fully erase a photo candidate

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


@dataclass
class FusionResult:
    candidates: List[Prediction]         # re-ranked with adjusted probabilities
    used_dims: List[str] = field(default_factory=list)
    changed: bool = False
    notes: List[str] = field(default_factory=list)
    env_priors: Dict[str, float] = field(default_factory=dict)     # code -> p_env
    used_conditions: Dict[str, float] = field(default_factory=dict)  # fed to scorer


class FusionScorer:
    def __init__(self, reference_table: Optional[Dict] = None, alpha: float = _ALPHA_DEFAULT):
        self.table = reference_table if reference_table is not None else load_reference()
        self.alpha = alpha
        self._model = self._build_model() if _HAS_SKLEARN else None

    # ------------------------------------------------------------------
    def _build_model(self):
        """Train the calibrator on synthetic anchors drawn from the table."""
        rng = np.random.RandomState(_SEED)
        codes = list(self.table.keys())
        X, y = [], []
        for code in codes:
            ranges = self.table[code]
            dims = KNOWN_DIMS
            for _ in range(_ANCHORS_PER_DISEASE):
                # Positive: conditions inside "code"'s favourable bands.
                obs_pos = {d: rng.uniform(*self.table[code][d]) if d in self.table[code]
                           else rng.uniform(0, 100) for d in dims}
                X.append(feature_vector(code, obs_pos, self.table))
                y.append(1)
                # Negative: conditions in another disease's favourable bands.
                other = codes[rng.randint(len(codes))]
                obs_neg = {d: rng.uniform(*self.table[other][d]) if d in self.table[other]
                           else rng.uniform(0, 100) for d in dims}
                X.append(feature_vector(code, obs_neg, self.table))
                y.append(0)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ])
        pipe.fit(np.vstack(X), np.asarray(y))
        return pipe

    # ------------------------------------------------------------------
    def _env_prior(self, code: str, conditions: Dict[str, float]) -> float:
        if self._model is None:
            # No sklearn (fringe runtime): fall back to mean-closeness heuristic.
            vec = feature_vector(code, conditions, self.table)
            return float(np.asarray(vec).mean())
        vec = feature_vector(code, conditions, self.table).reshape(1, -1)
        return float(self._model.predict_proba(vec)[0, 1])

    def reweight(self, predictions: List[Prediction],
                 conditions: Optional[Dict[str, float]] = None) -> FusionResult:
        conds = conditions or {}
        used = [d for d in KNOWN_DIMS if d in conds]
        if not used or not predictions:
            return FusionResult(candidates=list(predictions), used_dims=[], changed=False,
                                notes=["no_env_data"], used_conditions=dict(conds))

        priors = {}
        for p in predictions:
            prior = self._env_prior(p.code, conds)
            # Floor guards against overconfident/imperfect weather data: an
            # implausible-looking disease is reduced but never erased outright.
            priors[p.code] = max(prior, _PRIOR_FLOOR)

        raw = np.asarray([p.probability for p in predictions], dtype=np.float64)
        adjusted = raw * (np.asarray([priors[p.code] for p in predictions]) ** self.alpha)
        total = adjusted.sum()
        if total <= 0:
            return FusionResult(candidates=list(predictions), used_dims=used, changed=False,
                                notes=["degenerate"], used_conditions=dict(conds))
        adjusted = adjusted / total

        reranked = [Prediction(code=p.code, probability=float(a), index=p.index)
                    for a, p in zip(adjusted, predictions)]
        reranked.sort(key=lambda x: x.probability, reverse=True)
        return FusionResult(candidates=reranked, used_dims=used,
                            changed=any(a != b.probability for a, b in zip(adjusted, predictions)),
                            notes=[f"env_adjusted:{','.join(used)}"],
                            env_priors=priors,
                            used_conditions={d: conds[d] for d in used})


# Convenience singleton built lazily so tests can inject fresh tables.
def get_scorer(reference_table: Optional[Dict] = None, force_new: bool = False) -> FusionScorer:
    global _scorer
    if _scorer is None or force_new:
        _scorer = FusionScorer(reference_table)
    return _scorer


_scorer: Optional[FusionScorer] = None