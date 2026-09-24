"""
Soil module orchestrator - the OPTIONAL input with the mandatory confirm step.

Flow: manual entry OR guided SHC photo -> OCR -> parse -> SHOW BACK the values
for one-tap confirm/edit.  Nothing is used downstream until `confirmed=True`.

The module is fully skippable: `run()` with no input returns None and the
pipeline proceeds on image-only diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..util.runtime import is_pyodide
from .ocr import ocr_text
from .parser import parse_shc_text
from .preprocess import deskew_and_prepare


class SoilSource:
    MANUAL = "manual"
    OCR = "ocr"


@dataclass
class SoilValues:
    n: Optional[float] = None
    p: Optional[float] = None
    k: Optional[float] = None
    ph: Optional[float] = None
    source: str = SoilSource.MANUAL
    ocr_raw_lines: List[str] = None  # captured but unconfirmed OCR lines

    def __post_init__(self):
        if self.ocr_raw_lines is None:
            self.ocr_raw_lines = []

    @property
    def present(self) -> bool:
        return any(v is not None for v in (self.n, self.p, self.k, self.ph))

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {"n": self.n, "p": self.p, "k": self.k, "ph": self.ph}

    def as_json(self) -> Optional[str]:
        import json
        if not self.present:
            return None
        return json.dumps({"n": self.n, "p": self.p, "k": self.k, "ph": self.ph,
                           "source": self.source})


# Agronomic reference bands (kg/ha, Indian SHC convention) used ONLY for the
# nutrient advisory note - they do NOT re-rank the disease.
_N_BANDS = (120.0, 280.0)
_P_BANDS = (22.0, 56.0)
_K_BANDS = (110.0, 280.0)


def nutrient_advisory(values: "SoilValues") -> List[Dict]:
    """
    Return advisory notes for low/high N/P/K.  Each item is {key, value} usable
    with the translator (`soil.low_n`, ...).  Empty when nothing entered.
    Deliberately advisory-only: disease ranking uses temperature, humidity and
    pH; N/P/K guide the agronomic recommendation instead.
    """
    if values is None:
        return []
    notes: List[Dict] = []
    for dim, lo, hi, key in (("n", *_N_BANDS, "n"), ("p", *_P_BANDS, "p"), ("k", *_K_BANDS, "k")):
        v = getattr(values, dim, None)
        if v is None:
            continue
        if v < lo:
            notes.append({"key": f"soil.low_{key}", "value": v})
        elif v > hi:
            notes.append({"key": f"soil.high_{key}", "value": v})
    return notes


class SoilModule:
    def manual(self, n=None, p=None, k=None, ph=None) -> SoilValues:
        return SoilValues(n=n, p=p, k=k, ph=ph, source=SoilSource.MANUAL)
    def run_ocr(self, image: np.ndarray) -> Optional[SoilValues]:
        """
        Preprocess -> OCR -> parse.  Returns SoilValues with source='ocr' when
        at least one value was extracted, else None.  The UI must then run the
        confirm/edit step before using the values.  Unconfirmed OCR results are
        never passed downstream.
        """
        if is_pyodide():
            # No OCR engine in the browser build - degrade explicitly.
            return None
        if image is None:
            return None
        prepared = deskew_and_prepare(image)
        lines = ocr_text(prepared)
        values, matched = parse_shc_text(lines)
        if not values:
            return None
        return SoilValues(
            n=values.get("n"), p=values.get("p"), k=values.get("k"),
            ph=values.get("ph"), source=SoilSource.OCR,
            ocr_raw_lines=matched,
        )

    def apply_edits(self, values: SoilValues, n=None, p=None, k=None, ph=None) -> SoilValues:
        """Overwrite any field the farmer corrected; keeps source unchanged."""
        return SoilValues(
            n=n if n is not None else values.n,
            p=p if p is not None else values.p,
            k=k if k is not None else values.k,
            ph=ph if ph is not None else values.ph,
            source=values.source,
            ocr_raw_lines=values.ocr_raw_lines,
        )