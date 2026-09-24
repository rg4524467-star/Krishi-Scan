"""
Shared UI helpers: translator glue, TTS readout button, heatmap overlay.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from ..lang import get_translator, speak, tts_available


def tr(key: str, **kwargs) -> str:
    """Translate using the current session language."""
    lang = st.session_state.get("lang", "hi")
    return get_translator(lang).t(key, **kwargs)


def lang() -> str:
    return st.session_state.get("lang", "hi")


def speak_button(text: Optional[str] = None, key: Optional[str] = None, lang_code: Optional[str] = None) -> None:
    """Small TTS readout button.  Text-only fallback when TTS unavailable."""
    if not st.session_state.get("tts", True) or not text:
        return
    label = "🔊" if tts_available() else "🔇 (voice unavailable)"
    if st.button(label, key=key, use_container_width=False):
        if tts_available():
            speak(text, lang_code or lang())
    st.caption("")


def tts_toggle() -> bool:
    st.session_state["tts"] = st.toggle(
        tr("lang.tts"), value=st.session_state.get("tts", True),
        help=tr("lang.tts_off"))
    return st.session_state["tts"]


def small_hint(text: str) -> None:
    """Muted one-line hint (forwarded to the design-system renderer)."""
    from .styles import small_hint as _render
    _render(text)


def empty_state(icon: str, text: str) -> None:
    """Centred illustration + caption for blank screens."""
    from .styles import empty_state as _render
    _render(icon, text)


def cam_guide(text: str) -> None:
    """Leaf-positioning hint card shown under the capture widget."""
    from .styles import cam_guide as _render
    _render(text)


# Auto weather strip state: show the box once per session and cache the last
# successful fetch so page nav does not re-hit the network on every rerun.
_WEATHER_CACHE_KEY = "wx_auto_cache"


@st.cache_data(ttl=1800, show_spinner=False)
def _auto_weather_cached(placeish: str):
    """Best-effort current condition snapshot (dict) for the auto strip."""
    from ..weather import WeatherClient, locate_by_ip
    loc = locate_by_ip()
    if not loc:
        return None
    lat, lon, label = loc
    data = WeatherClient().fetch(lat, lon)
    return {
        "location": label or placeish,
        "temp_c": data.temp_c,
        "humidity_pct": data.humidity_pct,
        "rain_mm": data.rain_mm,
        "condition": data.condition,
    }


def weather_strip() -> None:
    """Small always-visible weather box under the header.

    Online -> auto-locate by IP and fetch live conditions (Open-Meteo, no key).
    Offline -> show the last cached snapshot with an 'offline' tag, if any.
    Never blocks the page: failures quietly render the neutral offline note.
    """
    from .styles import weather_bar as _render

    online = st.session_state.get("__online", True)
    cached = st.session_state.get(_WEATHER_CACHE_KEY)

    if not online:
        if cached:
            _render(cached.get("location", ""), cached, offline=True, stale=True,
                    tag=tr("weather.cached_offline"))
        else:
            _render("", {}, offline=True, tag="")
        return

    if cached is not None:
        _render(cached.get("location", ""), cached, offline=False)
        return

    try:
        with st.spinner(tr("weather.loading_short")):
            data = _auto_weather_cached("")
    except Exception:
        data = None
    if data:
        st.session_state[_WEATHER_CACHE_KEY] = data
        _render(data.get("location", ""), data, offline=False)
    else:
        _render("", {}, offline=True, tag="")


def weather_alerts() -> list:
    """Diseases whose favourable env-ranges match today's live weather.

    Reads the same cached snapshot the weather strip uses, so this never
    triggers its own network call.  Returns up to 3 (code, favorability) pairs
    above the alert threshold, or [].
    """
    cached = st.session_state.get(_WEATHER_CACHE_KEY)
    if not cached or cached.get("temp_c") is None:
        return []
    from ..fusion.reference import load_reference, favorability
    from ..inference.classes import load_classes
    classes = load_classes()
    table = load_reference()
    conditions = {
        "temp_c": cached["temp_c"],
        "humidity_pct": cached["humidity_pct"],
    }
    known = set(classes.order)
    scored = []
    for code, ranges in table.items():
        if code not in known or "temp_c" not in ranges or "healthy" in code:
            continue
        fav = favorability(code, conditions, table)
        if fav >= 0.80:
            scored.append((code, fav))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:3]


def weather_alert_strip() -> None:
    """Small 'high risk today' badge under the weather strip on the home page."""
    from ..inference.classes import load_classes

    alerts = weather_alerts()
    if not alerts:
        return
    st.markdown(f"<b>{tr('wx.risk_title')}</b>", unsafe_allow_html=True)
    for code, _fav in alerts:
        label = load_classes().label(code, lang())
        st.markdown(
            f"<div class='result-card'><b>⚠️ {tr('wx.risk_high', disease=label)}</b></div>",
            unsafe_allow_html=True)


def heatmap_overlay(leaf: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Blend the leaf image with a jet-colored Grad-CAM heatmap.
    Pure NumPy colormap so it works in Pyodide without matplotlib.
    """
    h = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    if h.shape[:2] != leaf.shape[:2]:
        from PIL import Image
        h = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize(
            (leaf.shape[1], leaf.shape[0])), dtype=np.float32) / 255.0
    cmap = _jet_colormap()
    idx = np.clip((h * 255).astype(int), 0, 255)
    colored = cmap[idx]                       # (H,W,3)
    blended = (0.55 * leaf.astype(np.float32) + 0.45 * colored * 255.0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _jet_colormap() -> np.ndarray:
    # Standard 256-entry jet lookup table (build once).
    out = np.zeros((256, 3))
    n = 256
    out[:, 0] = np.interp(np.arange(n),
                          [0, 85, 170, 255], [0, 0, 255, 128])
    out[:, 1] = np.interp(np.arange(n),
                          [0, 85, 170, 255], [0, 255, 255, 0])
    out[:, 2] = np.interp(np.arange(n),
                          [0, 85, 170, 255], [128, 255, 0, 0])
    return out / 255.0