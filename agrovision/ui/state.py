"""
Session-state keys and helpers.

Everything the wizard stores lives here with defaults, so pages can be rebuilt
in any order without KeyErrors (Streamlit reruns the whole script each click).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


def init() -> None:
    defaults = {
        # language + tts toggle (persist across reruns)
        "lang": "hi",
        "tts": True,
        # farmer profile
        "farmer": {"name": "", "phone": "", "location": ""},
        # navigation
        "page": "home",
        "nav_tabs": "home",
        # diagnose wizard state
        "dg_crop": None,
        "dg_slot": 0,              # bumped each reset -> fresh capture widget keys
        "dg_photo": None,          # np.ndarray RGB
        "dg_retakes": 0,
        "dg_verdict": None,        # GatekeeperVerdict
        "dg_proceed_anyway": False,
        "dg_predictions": None,    # list[Prediction] (post-fusion, final)
        "dg_raw_predictions": None,
        "dg_heatmap": None,        # np.ndarray (H,W) 0..1
        "dg_soil": None,           # SoilValues
        "dg_soil_loaded_saved": False,   # used saved-profile soil this test
        "dg_weather": None,        # WeatherData
        "dg_fusion": None,         # FusionResult
        "dg_done": None,           # saved-record dict
        # soil OCR confirm state
        "ocr_values": None,        # SoilValues pending confirmation
        # misc transient
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_wizard() -> None:
    keys = ["dg_crop", "dg_photo", "dg_retakes", "dg_verdict", "dg_proceed_anyway",
            "dg_predictions", "dg_raw_predictions", "dg_heatmap", "dg_soil",
            "dg_soil_loaded_saved", "dg_weather", "dg_fusion", "dg_done", "ocr_values"]
    for k in keys:
        st.session_state[k] = None
    st.session_state["dg_retakes"] = 0
    # Fresh widget identity so a previously-uploaded file can't silently
    # re-trigger an old photo on the next capture screen.
    st.session_state["dg_slot"] = st.session_state.get("dg_slot", 0) + 1


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:
    st.session_state[key] = value