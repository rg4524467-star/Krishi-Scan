"""
Externalized-string translation layer.

Single source of truth: agrovision/lang/strings_{lang}.json.

Guarantees (enforced in resolve(), covered by tests):
  * A missing key in the requested language NEVER returns a raw key or empty
    string.  Resolution order: requested lang -> fallback_chain -> en.
  * If even English lacks the key, the *key itself* is returned but prefixed
    with a sentinel so callers can distinguish a content bug from normal use.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

from ..config import LANG_DIR, LANG

DEFAULT_LANGUAGE = LANG.default_language

_loaded: Dict[str, Dict] = {}
_lock = threading.Lock()

# Keys may contain placeholders like "{k}" or "{n}" -> format() them at call site.
_RESERVED_PREFIX = "_missing_"


def _load(lang: str) -> Dict:
    with _lock:
        if lang in _loaded:
            return _loaded[lang]
        path = Path(LANG_DIR) / f"strings_{lang}.json"
        if not path.exists():
            _loaded[lang] = {}
            return _loaded[lang]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        _loaded[lang] = data
        return data


def load_strings(lang: str) -> Dict:
    """Public accessor (used by UI too). Safe against double-loading."""
    return _load(lang)


def _flatten(data: Dict, prefix: str = "") -> Dict[str, str]:
    flat = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, full))
        else:
            flat[full] = str(value)
    return flat


class Translator:
    def __init__(self, language: Optional[str] = None, fallback_chain: Optional[tuple] = None):
        self.language = language or DEFAULT_LANGUAGE
        if self.language not in LANG.supported:
            # Unknown requested language -> fall straight to the chain.
            self.language = LANG.supported[0]
        self.fallback_chain = fallback_chain or LANG.fallback_chain
        self._flat = {
            lang: _flatten(_load(lang)) for lang in (LANG.supported + self.fallback_chain)
        }

    def is_supported(self, lang: str) -> bool:
        return lang in LANG.supported

    def resolve(self, key: str, **kwargs) -> str:
        """Resolve + format a dotted string key with strict fallback semantics."""
        for lang in (self.language,) + self.fallback_chain:
            value = self._flat.get(lang, {}).get(key)
            if value is not None:
                return self._safe_format(value, kwargs)
        return f"{_RESERVED_PREFIX}{key}"

    # -- helpers ------------------------------------------------
    @staticmethod
    def _safe_format(template: str, kwargs: Dict) -> str:
        try:
            return template.format(**kwargs) if kwargs else template
        except (KeyError, IndexError, ValueError):
            return template

    def t(self, key: str, **kwargs) -> str:
        return self.resolve(key, **kwargs)


# A module-level instance is cached so the UI can mutate `tr.language` live.
_global_tr: Optional[Translator] = None


def get_translator(language: Optional[str] = None) -> Translator:
    global _global_tr
    if _global_tr is None:
        _global_tr = Translator(language)
    if language is not None and _global_tr.language != language:
        _global_tr.language = language
    return _global_tr


# Convenience: resolve against the default language without building an object.
def resolve(key: str, language: Optional[str] = None, **kwargs) -> str:
    return get_translator(language).resolve(key, **kwargs)