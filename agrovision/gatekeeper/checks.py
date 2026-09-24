"""
The four gatekeeper checks plus the "unreadable image" gate.

Each check returns a CheckResult carrying:
  * check_id          stable machine key (persisted with the diagnosis)
  * passed            bool
  * value             measured quantity (for debugging / tests)
  * feedback_key      translation key for the FAILURE message
  * hint_key          translation key for the corrective hint

Feedback strings are keys, never hardcoded text, so the gatekeeper module
stays language-agnostic (UI translates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import GK
from .backends import get_backend

LOWER_GREEN_HSV = (35, 40, 40)
UPPER_GREEN_HSV = (95, 255, 255)


@dataclass
class CheckResult:
    check_id: str
    passed: bool
    value: float
    feedback_key: Optional[str] = None
    hint_key: Optional[str] = None
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.passed


def check_unreadable(image) -> CheckResult:
    ok = image is not None and image.size > 0 and image.ndim >= 2
    return CheckResult(
        check_id="unreadable",
        passed=ok,
        value=0.0 if not ok else 1.0,
        feedback_key="gatekeeper.unreadable",
    )


def check_blur(gray: np.ndarray, threshold: Optional[float] = None) -> CheckResult:
    threshold = GK.blur_min_variance if threshold is None else threshold
    value = get_backend().laplacian_variance(gray)
    return CheckResult(
        check_id="blur",
        passed=value >= threshold,
        value=value,
        feedback_key="gatekeeper.blur",
        hint_key="gatekeeper.blur_hint",
        details={"threshold": threshold},
    )


def check_brightness(gray: np.ndarray) -> CheckResult:
    value = float(gray.mean())
    passed = GK.brightness_min <= value <= GK.brightness_max
    feedback_key = "gatekeeper.bright" if value > GK.brightness_max else "gatekeeper.dark"
    hint_key = "gatekeeper.bright_hint" if value > GK.brightness_max else "gatekeeper.dark_hint"
    return CheckResult(
        check_id="brightness",
        passed=passed,
        value=value,
        feedback_key=feedback_key,
        hint_key=hint_key,
        details={"low": GK.brightness_min, "high": GK.brightness_max},
    )


def check_leaf_coverage(rgb: np.ndarray, min_ratio: Optional[float] = None) -> CheckResult:
    min_ratio = GK.leaf_min_coverage if min_ratio is None else min_ratio
    value = get_backend().green_ratio(rgb)
    return CheckResult(
        check_id="leaf_coverage",
        passed=value >= min_ratio,
        value=value,
        feedback_key="gatekeeper.leaf",
        hint_key="gatekeeper.leaf_hint",
        details={"min_ratio": min_ratio},
    )


def check_glare(gray: np.ndarray, max_ratio: Optional[float] = None) -> CheckResult:
    max_ratio = GK.glare_max_ratio if max_ratio is None else max_ratio
    value = get_backend().near_white_ratio(gray, GK.glare_brightness)
    return CheckResult(
        check_id="glare",
        passed=value <= max_ratio,
        value=value,
        feedback_key="gatekeeper.glare",
        hint_key="gatekeeper.glare_hint",
        details={"max_ratio": max_ratio, "brightness": GK.glare_brightness},
    )


ALL_CHECKS = (check_blur, check_brightness, check_leaf_coverage, check_glare)
