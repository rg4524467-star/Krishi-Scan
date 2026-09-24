"""
OCR text extraction for Soil Health Cards - ONLINE ONLY.

Tesseract (pytesseract) is the default; EasyOCR is opt-in via USE_EASYOCR=1.
Neither engine exists inside Pyodide, so the offline/PWA path never attempts
this and the SoilModule cleanly degrades to guided manual entry (documented
in the UI).  All failures return [] rather than raising, letting the caller
offer the "try again or enter manually" fallback.

Tesseract is configured against the app-bundled tessdata dir
(agrovision/assets/tessdata) when present, which ships both `eng` and `hin`
so bilingual Soil Health Cards (English labels + Devanagari) are recognised.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# App-bundled language data (eng + hin).  Used instead of the system tessdata
# so the language set is deterministic on every machine with the Tesseract
# binary installed.  Falls back to the engine default if the dir is absent.
_TESSDATA_DIR = Path(__file__).resolve().parents[1] / "assets" / "tessdata"
_TESSDATA_DIR_STR = str(_TESSDATA_DIR)
_HAS_HINDI = _TESSDATA_DIR.joinpath("hin.traineddata").exists()

# Tesseract is a system binary (not a pip package) and may sit off PATH when
# freshly installed or run from a shell that predates the install.  Auto-detect
# the common locations so pytesseract never silently fails to find it.
_TESSERACT_CANDIDATES = (
    "tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
)


def _tesseract_cmd() -> str:
    for candidate in _TESSERACT_CANDIDATES:
        if os.sep in candidate:
            found = Path(candidate).exists()
        else:
            found = shutil.which(candidate) is not None
        if found:
            return candidate
    return "tesseract"


def ocr_text(image: np.ndarray) -> List[str]:
    """Return OCR'd text lines from a preprocessed grayscale image."""
    engine = os.environ.get("AGRO_OCR", "tesseract")
    if engine.lower() == "easyocr":
        return _ocr_easyocr(image)
    return _ocr_tesseract(image)


def _tess_lang() -> str:
    """Prefer English+Hindi when the bundled hin data exists, else English."""
    return "eng+hin" if _HAS_HINDI else "eng"


def _tess_config() -> str:
    """Tesseract CLI config; pins tessdata to the app bundle when available."""
    if _TESSDATA_DIR.exists():
        return f"--tessdata-dir {_TESSDATA_DIR_STR}"
    return ""


# Tesseract layout modes to try and merge.  Different SHC layouts (single
# column, table grid, sparse cells) read best under different modes, so a
# merged union of several modes catches more text than any one mode alone.
_PSM_MODES = (3, 6, 11)


def _ocr_tesseract(image: np.ndarray) -> List[str]:
    try:
        from PIL import Image as PILImage
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd()
        img = _upscale(PILImage.fromarray(image))
        config = _tess_config()
        seen: set = set()
        lines: List[str] = []
        for psm in _PSM_MODES:
            text = pytesseract.image_to_string(
                img, lang=_tess_lang(), config=f"{config} --psm {psm}"
            )
            for raw in text.splitlines():
                ln = raw.strip()
                key = _dedupe_key(ln)
                if ln and key not in seen:
                    seen.add(key)
                    lines.append(ln)
        return lines
    except Exception as exc:
        logger.warning("Tesseract OCR failed (%s): %s", type(exc).__name__, exc)
        return []


def _upscale(img) -> "PIL.Image":
    """Resize small crops up before OCR - Tesseract reads ~300dpi text best."""
    from PIL import Image as PILImage
    if img.width < 1600 or img.height < 1600:
        scale = max(1600 / img.width, 1600 / img.height)
        if scale > 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)),
                             PILImage.LANCZOS)
    return img


def _dedupe_key(line: str) -> str:
    """Order-preserving dedupe key so merged PSM passes don't repeat lines."""
    return " ".join(line.lower().split())


def _ocr_easyocr(image: np.ndarray) -> List[str]:
    try:
        import easyocr
        # Cache the reader so successive calls don't re-initialize the model.
        reader = getattr(_ocr_easyocr, "reader", None)
        if reader is None:
            reader = easyocr.Reader(["en"], gpu=False)
            _ocr_easyocr.reader = reader
        results = reader.readtext(image, detail=0)
        return [str(r) for r in results]
    except Exception:
        return []