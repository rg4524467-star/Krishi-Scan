"""
Gatekeeper orchestrator.

Runs: unreadable gate -> resize -> [blur, brightness, leaf, glare] checks.
Returns a verdict plus a language-agnostic list of failed CheckResults so the
UI can render per-check feedback and drive the capture->check->feedback->retake
loop.  Thresholds live in config (GK).

Deliberately does NOT do live-video processing (Streamlit's execution model
doesn't support continuous loops; we operate on a captured still).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import GK
from ..util.image import resize_to_width
from . import checks
from .checks import CheckResult
from .backends import get_backend


@dataclass
class GatekeeperVerdict:
    passed: bool
    results: List[CheckResult]
    unreadable: bool = False

    @property
    def failed(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed]


class Gatekeeper:
    def __init__(self, max_retakes: Optional[int] = None, config=None):
        self.max_retakes = max_retakes if max_retakes is not None else GK.max_retakes
        self.config = config

    # -- main entry -------------------------------------------------------
    def check(self, image: Optional[np.ndarray]) -> GatekeeperVerdict:
        unreadable = checks.check_unreadable(image)
        if not unreadable.passed:
            return GatekeeperVerdict(passed=False, results=[unreadable], unreadable=True)

        rgb = resize_to_width(image, GK.preview_width)
        backend = get_backend()
        gray = backend.to_gray(rgb)
        results = [
            checks.check_blur(gray),
            checks.check_brightness(gray),
            checks.check_leaf_coverage(rgb),
            checks.check_glare(gray),
        ]
        return GatekeeperVerdict(passed=all(r.passed for r in results), results=results)

    # -- retake policy ------------------------------------------------------
    def can_retry(self, retries_used: int) -> bool:
        """True if the farmer can still retake within the configured limit."""
        return retries_used < self.max_retakes

    def policy_after_fail(self, retries_used: int) -> str:
        """Apply the 3+ retry edge case: offer to proceed anyway."""
        if self.can_retry(retries_used):
            return "retake"
        return "offer_proceed"  # never trap the farmer in an infinite loop


__all__ = ["Gatekeeper", "GatekeeperVerdict", "CheckResult"]