"""
SHC label -> value parser.

Not free-form table parsing: we match OCR text against known label strings
(English + Hindi, as printed on standard Soil Health Cards) and pull the number
that follows the matched label.  Unknown/ambiguous cells are simply ignored
instead of hallucinating values - the "never auto-apply unverified input" rule
starts right here.

Returns parsed values + the raw matched lines so the UI can show the farmer
exactly what was read and where it came from.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Label synonyms per field (lowercased, spaces normalized).
FIELD_LABELS: Dict[str, List[str]] = {
    "n": ["nitrogen", "nitrogen (kg/ha)", "nitrogen kg/ha", "नाइट्रोजन", "n (kg/ha)", "nitrogen(n)"],
    "p": ["phosphorus", "phosphorus (kg/ha)", "phosphorus kg/ha", "phosphorous", "फॉस्फोरस", "p (kg/ha)", "phosphorus(p)"],
    "k": ["potassium", "potassium (kg/ha)", "potassium kg/ha", "पोटाश", "potash", "k (kg/ha)", "potassium(k)"],
    "ph": ["ph", "soil ph", "मिट्टी का ph", "ph (1:2.5)", "ph(1:2.5)"],
}

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
# Ratio substrings like "1:2.5" (standard pH notation) carry no field value.
# The colon form is consumed whole (incl. any trailing ".5" of the ratio) so a
# leftover digit is never mistaken for the value; "1.2.5" (OCR turned the colon
# into a dot) is covered too.  Plain decimals like "6.8" (single separator)
# are NOT ratios and are left intact.
_RATIO = re.compile(r"\d+\s*:\s*\d+(?:\.\d+)?|\d+(?:[.:]\d+){2,}")
# Indian-style thousands separators: "1,425.50" -> "1425.50".
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
# How far past a label line to keep looking for its value when the card's
# table splits the label and the number into separate cells/lines.
_LOOKAHEAD = 2


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).replace("_", " ")


def match_field(text: str) -> Optional[str]:
    """Return the field key whose label appears in `text`, else None."""
    t = _normalize(text)
    for field, labels in FIELD_LABELS.items():
        for label in labels:
            if label in t:
                # 'phosphorus' must not be satisfied by just 'phosphorus(p)' etc - fine.
                return field
    return None


def extract_number(text: str) -> Optional[float]:
    """
    Pull the VALUE out of an SHC line.  Value sits at the end of the row
    ("Nitrogen (kg/ha) 142.50"); ratio fragments are stripped first so the
    standard "pH (1:2.5) 6.8" notation yields 6.8, not 1.
    """
    cleaned = _THOUSANDS.sub("", _RATIO.sub("", text))
    nums = _NUMBER.findall(cleaned)
    if not nums:
        return None
    return float(nums[-1])


def _plausible(field: str, value: float) -> bool:
    """Range sanity check so a misread digit is never silently accepted."""
    if field == "ph":
        return 0.0 <= value <= 14.0
    return 0.0 <= value <= 10000.0


def parse_shc_text(lines) -> Tuple[Dict[str, float], List[str]]:
    """
    Given an iterable of OCR text lines, return:
      (values, matched_lines)
    where values maps 'n'/'p'/'k'/'ph' -> confirmed number and matched_lines
    are the original lines that produced each field (for the confirm/edit UI).

    A label and its value may land on the same line ("Nitrogen (kg/ha) 142.50")
    or on separate lines ("Nitrogen (kg/ha)" / "142.50") when the card's table
    is grid-lined; both layouts are supported.
    """
    values: Dict[str, float] = {}
    matched: List[str] = []
    items = [str(r).strip() for r in lines if str(r).strip()]
    for i, text in enumerate(items):
        field = match_field(text)
        if not field:
            continue
        num = extract_number(text)
        if num is None or not _plausible(field, num):
            # Label seen but no credible value on this line - look ahead a few
            # lines (stopping at the next label) for the value cell.
            for j in range(i + 1, min(i + 1 + _LOOKAHEAD, len(items))):
                if match_field(items[j]) is not None:
                    break
                candidate = extract_number(items[j])
                if candidate is not None and _plausible(field, candidate):
                    num, text = candidate, text + " " + items[j]
                    break
        if num is None or not _plausible(field, num):
            continue
        # Prefer the first credible value for each field (cards list each once).
        if field not in values:
            values[field] = num
            matched.append(text)
    return values, matched