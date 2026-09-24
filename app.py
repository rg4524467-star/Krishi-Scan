"""
Krishi Scan - Streamlit entry point.

Run online:    streamlit run app.py
Run offline:   build the stlite PWA shell (see README) which serves this same
               package through Pyodide.

The app uses internal routing (not Streamlit's pages/ dir) so the exact same
codebase runs in the browser where multi-file page discovery differs.

Chrome policy: Streamlit's default header/deploy menu/toolbar are hidden via
CSS (ui/styles.py) and replaced by a branded header + tab-bar nav, so the page
reads as a native mobile app instead of a bare Streamlit page.
"""
from __future__ import annotations

import streamlit as st

from agrovision.data import init_db
from agrovision.ui import about, diagnose, history, home, soil_page, styles
from agrovision.ui import components as c   
from agrovision.ui.state import init as init_state, set
from agrovision.lang import get_translator

# Brand + page config.  Chrome is hidden in CSS, so no sidebar needed.
st.set_page_config(
    page_title="Krishi Scan",
    page_icon="🌾",
    initial_sidebar_state="collapsed",
    layout="centered",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

init_state()
init_db()  # no-op when DB unreachable; records go to the offline journal
get_translator(st.session_state.get("lang", "hi"))

PAGES = {
    "home": ("home", home),
    "diagnose": ("diagnose", diagnose),
    "soil": ("soil", soil_page),
    "history": ("history", history),
    "about": ("about", about),
}
ICONS = {"home": "🏠", "diagnose": "📷", "soil": "🧪", "history": "📋", "about": "ℹ️"}


def _nav() -> None:
    tr = get_translator(st.session_state.get("lang", "hi"))
    current = st.session_state.get("page", "home")
    items = [
        {"key": key, "label": tr.t(f"nav.short_{key}"), "icon": ICONS[key],
         "active": key == current}
        for key in PAGES
    ]
    styles.nav_bar(items)


class _Banner:
    """Cached connectivity snapshot to avoid a network probe on every rerun."""
    _state: tuple = ()


def _connectivity() -> tuple:
    from agrovision.data import is_online
    if not _Banner._state:
        _Banner._state = (is_online(),)
    return _Banner._state


def main() -> None:
    styles.inject()
    tr = get_translator(st.session_state.get("lang", "hi"))
    online = _connectivity()[0]
    st.session_state["__online"] = online
    chip_label = tr.t("offline.online") if online else tr.t("offline.offline")
    farmer = st.session_state.get("farmer", {})
    styles.brand_header(tr.t("app.subtitle"), online=online, chip_label=chip_label,
                        farmer_loc=farmer.get("location", ""))
    c.weather_strip()
    with st.container(border=False):
        lcol, rcol = st.columns([3, 1], vertical_alignment="center")
        with lcol:
            styles.language_switcher({"hi": "हिन्दी", "en": "English", "ta": "தமிழ்"})
        with rcol:
            c.tts_toggle()
    _nav()

    current = st.session_state.get("page", "home")
    PAGES.get(current, home)[1].render()


if __name__ == "__main__":
    main()