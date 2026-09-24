"""
TTS (text-to-speech) for diagnosis readout.

Tiered strategy so the feature works everywhere without hard failures:
  1. Browser (Pyodide/stlite): use the Web Speech API (speechSynthesis) with a
     BCP-47 language code.  Zero install, works offline for installed voices.
  2. Native desktop: pyttsx3 if installed and an engine is available.
  3. Neither: return False quietly - the UI simply shows text.

This is OUTPUT ONLY.  No speech recognition anywhere (per project scope).
"""
from __future__ import annotations

from typing import Optional

from ..config import LANG
from .translator import resolve as _resolve

# BCP-47 language codes used by both the browser TTS and pyttsx3.
LANGUAGE_CODES = {"hi": "hi-IN", "en": "en-IN", "ta": "ta-IN"}


def tts_available() -> bool:
    if not LANG.tts_enabled:
        return False
    if _in_browser():
        return True
    try:
        import pyttsx3  # noqa: F401
        return True
    except Exception:
        return False


def _in_browser() -> bool:
    try:
        import sys
        return sys.platform == "emscripten"
    except Exception:
        return False


def speak(text: str, lang: Optional[str] = None) -> bool:
    """Speak `text` for `lang`.  Returns True if speech was requested."""
    if not LANG.tts_enabled or not text:
        return False
    code = LANGUAGE_CODES.get(lang or LANG.default_language, "hi-IN")
    if _in_browser():
        return _speak_browser(text, code)
    return _speak_native(text, code)


def _speak_browser(text: str, code: str) -> bool:
    try:
        import js  # Pyodide global bridging to the DOM
        opts = {"rate": 0.95, "lang": code}
        # Cancel any in-flight utterance to avoid overlapping readout.
        js.window.speechSynthesis.cancel()
        js.window.speechSynthesis.speak(js.Object.new(js.window.SpeechSynthesisUtterance, text))
        return True
    except Exception:
        return False


def _speak_native(text: str, code: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        try:
            for v in engine.getProperty("voices") or []:
                if code.split("-")[0].lower() in str(getattr(v, "languages", "")).lower():
                    engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False


def speak_key(key: str, lang: Optional[str] = None, **kwargs) -> bool:
    """Convenience: speak the translated value of a string key."""
    return speak(_resolve(key, lang, **kwargs), lang)