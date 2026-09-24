"""Home page: farmer profile, offline/online status banner."""
from __future__ import annotations

import streamlit as st

from ..ui import components as c
from ..ui.state import get, set

LANG_NAMES = {"hi": "हिन्दी", "en": "English", "ta": "தமிழ்"}


def render() -> None:
    # Brand header already shows the title; give a warm welcome card instead.
    st.markdown(
        f"<div class='result-card'><b>{c.tr('home.welcome')}</b>"
        f"<p>{c.tr('home.howto')}</p></div>",
        unsafe_allow_html=True)

    # -- offline banner ----------------------------------------------------
    _render_connectivity_banner()

    # -- weather-based disease risk alerts ---------------------------------
    c.weather_alert_strip()

    # -- farmer profile -----------------------------------------------------
    _render_profile()

    if st.button(c.tr("home.cta"), key="go_diagnose", type="primary"):
        set("page", "diagnose")
        set("nav_tabs", "diagnose")
        st.rerun()


def _render_profile() -> None:
    """Farmer details as a card with avatar + icon fields + save button."""
    farmer = get("farmer", {})
    saved = bool(farmer.get("name"))

    st.markdown(
        f"""
        <div class='av-profile-head'>
          <div class='av-avatar'>🧑‍🌾</div>
          <div>
            <div class='av-profile-title'>{c.tr('farmer.title')}</div>
            <div class='av-profile-sub'>
              {'✓ ' + farmer.get('name', '') if saved else c.tr('farmer.not_saved')}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True)

    with st.container(border=True):
        name = st.text_input(c.tr("farmer.name"), value=farmer.get("name", ""),
                             placeholder=c.tr("farmer.name_ph"), key="f_name")
        phone = st.text_input(c.tr("farmer.phone"), value=farmer.get("phone", ""),
                              placeholder=c.tr("farmer.phone_ph"), key="f_phone")
        loc = st.text_input(c.tr("farmer.location"), value=farmer.get("location", ""),
                            placeholder=c.tr("farmer.location_ph"), key="f_loc")
        if st.button(c.tr("farmer.save"), key="farmer_save", type="primary"):
            set("farmer", {"name": name, "phone": phone, "location": loc})
            st.success(c.tr("farmer.saved"))
            st.rerun()


def _render_connectivity_banner() -> None:
    from ..data import SyncEngine, is_online
    online = is_online()
    if online:
        st.markdown(f"<span class='av-chip online'>● {c.tr('offline.online')}</span>",
                    unsafe_allow_html=True)
        # Auto-sync without a manual button (requirement), showing status.
        engine = SyncEngine()
        status = engine.sync_pending(force=True)
        if status.pending == 0:
            st.caption(c.tr("offline.synced"))
        else:
            st.caption(c.tr("offline.pending", n=status.pending))
    else:
        st.markdown(f"<span class='av-chip offline'>● {c.tr('offline.offline')}</span>",
                    unsafe_allow_html=True)
        # Edge case: very first open with no internet -> clear guidance.
        if not _has_local_assets():
            st.warning(c.tr("offline.needs_first_setup"))
            st.info(c.tr("offline.first_run_online_required"))


def _has_local_assets() -> bool:
    from ..config import INF
    return INF.classes_file.exists() or INF.onnx_model.exists()