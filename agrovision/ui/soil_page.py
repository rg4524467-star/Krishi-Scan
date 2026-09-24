"""
Soil details page - save N/P/K/pH once on the farmer's profile and reuse it in
every diagnosis.  Saving here writes to the `soil_records` table (keyed by the
farmer's phone).  The diagnosis wizard then offers "use my saved soil values".

Update path: the inputs pre-fill with the last saved values, so saving again
simply overwrites - the latest record always wins.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from ..data import load_latest_soil, save_soil
from ..soil import SoilModule
from ..ui import components as c
from ..ui.state import set as sset
from ..util.image import load_image_bytes


def _phone() -> str:
    return st.session_state.get("farmer", {}).get("phone", "")


def _prefill(saved: dict) -> dict:
    """Current input values: last saved values (or empty) for update."""
    return {
        "n": saved.get("n") if saved else 0.0,
        "p": saved.get("p") if saved else 0.0,
        "k": saved.get("k") if saved else 0.0,
        "ph": saved.get("ph") if saved else 6.5,
    }


def render() -> None:
    st.title(c.tr("soil.title"))
    st.caption(c.tr("soil.units_note"))

    phone = _phone()
    saved = load_latest_soil(phone)

    if saved:
        st.info(c.tr("soil.saved_used"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N", saved["n"]); c2.metric("P", saved["p"])
        c3.metric("K", saved["k"]); c4.metric("pH", saved["ph"])
        st.caption(c.tr("soil.saved_from",
                        time=datetime.now().strftime("%Y-%m-%d %H:%M"),
                        source=c.tr("soil.source_manual") if saved.get("source") != "ocr"
                        else c.tr("soil.source_ocr")))
    else:
        st.caption(c.tr("soil.saved_none"))

    mode = st.radio(c.tr("soil.title"), ("manual", "ocr"),
                    format_func=lambda x: c.tr("soil.manual") if x == "manual" else c.tr("soil.ocr"),
                    horizontal=True, key="soil_page_mode")
    if mode == "manual":
        _render_manual(prefill=_prefill(saved), phone=phone)
    else:
        _render_ocr(phone=phone)


def _render_manual(prefill: dict, phone: str) -> None:
    n = st.number_input(c.tr("soil.n"), 0.0, 10000.0, step=1.0, value=float(prefill["n"]), key="sp_n")
    p = st.number_input(c.tr("soil.p"), 0.0, 10000.0, step=1.0, value=float(prefill["p"]), key="sp_p")
    k = st.number_input(c.tr("soil.k"), 0.0, 10000.0, step=1.0, value=float(prefill["k"]), key="sp_k")
    ph = st.number_input(c.tr("soil.ph"), 0.0, 14.0, step=0.1, value=float(prefill["ph"]), key="sp_ph")
    if st.button(c.tr("common.save"), key="sp_save", type="primary"):
        _persist(phone, n=n, p=p, k=k, ph=ph, source="manual")


def _persist(phone: str, *, n, p, k, ph, source: str) -> None:
    if not phone:
        st.warning(c.tr("soil.save_no_phone"))
        return
    ok = save_soil(phone, n=n, p=p, k=k, ph=ph, source=source)
    if ok:
        st.success(c.tr("soil.saved_updated"))
    else:
        st.warning(c.tr("soil.saved_failed"))
    st.rerun()


def _render_ocr(phone: str) -> None:
    from ..util.runtime import is_pyodide
    if is_pyodide():
        st.info(c.tr("soil.invalid"))
        st.info(c.tr("soil.manual"))
        return
    uploaded = st.file_uploader(c.tr("soil.upload_shc"), type=["jpg", "jpeg", "png"],
                                key="sp_shc")
    if uploaded is not None:
        raw = load_image_bytes(uploaded.getvalue())
        if raw is None:
            st.error(c.tr("gatekeeper.unreadable"))
            return
        with st.spinner(c.tr("soil.ocr_running")):
            values = SoilModule().run_ocr(raw)
        if values is None or not values.present:
            st.warning(c.tr("soil.invalid"))
            return
        # Mandatory confirm step (same rule as the wizard).
        st.markdown(f"<div class='gk-pass'>{c.tr('soil.ocr_result')}</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N", values.n); c2.metric("P", values.p)
        c3.metric("K", values.k); c4.metric("pH", values.ph)
        if st.button(c.tr("soil.confirm"), key="sp_confirm", type="primary"):
            _persist(phone, n=values.n, p=values.p, k=values.k, ph=values.ph,
                     source="ocr")