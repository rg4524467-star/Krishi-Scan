"""
Class-list (index -> disease code) management.

classes.json is the single contract between the training/export scripts and the
runtime.  It carries the canonical ordered list plus every localized label and
recommendation so language strings stay externalized (never hardcoded) while
traveling together with the model asset (offline cache friendly).

Shape:
{
  "order": ["rice_leaf_blast", "wheat_leaf_rust", ...],
  "diseases": { "<code>": { "en": {...}, "hi": {...}, "ta": {...} } }
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import INF


@dataclass
class Classes:
    order: List[str]
    info: Dict[str, Dict]          # code -> {lang: {name, advice, next}}

    @property
    def size(self) -> int:
        return len(self.order)

    def code_at(self, index: int) -> Optional[str]:
        return self.order[index] if 0 <= index < len(self.order) else None

    def label(self, code: str, lang: str) -> str:
        entry = self.info.get(code, {})
        if lang in entry and entry[lang].get("name"):
            return entry[lang]["name"]
        for fallback in ("en", "hi"):
            if fallback in entry and entry[fallback].get("name"):
                return entry[fallback]["name"]
        return code

    def recommendation(self, code: str, lang: str) -> Dict[str, str]:
        entry = self.info.get(code, {})
        out = {}
        if lang in entry:
            out.update(entry[lang])
        if not out:
            out.update(entry.get("en", {}))
        return out

    def diseases(self) -> List[Dict]:
        return [self.info[c] for c in self.order]


def load_classes(path=None) -> Classes:
    path = path or INF.classes_file
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return Classes(order=list(raw.get("order", [])), info=raw.get("diseases", {}))


def write_classes(classes: Classes, path=None) -> None:
    path = path or INF.classes_file
    payload = {"order": classes.order, "diseases": classes.info}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
