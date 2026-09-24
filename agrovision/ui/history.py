"""History page: past diagnoses + sync status (auto and manual triggers)."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from ..data import OfflineStore, SyncEngine
from ..ui import components as c


def render() -> None:
    st.title(c.tr("history.title"))

    store = OfflineStore()
    engine = SyncEngine(store=store)

    # -- sync status / manual trigger ---------------------------------------
    pending = store.pending_count
    col1, col2 = st.columns([2, 1])
    with col1:
        if pending:
            st.warning(c.tr("offline.pending", n=pending))
            st.caption(c.tr("history.sync_hint", n=pending))
        else:
            st.success(c.tr("offline.synced"))
    with col2:
        if st.button(c.tr("history.sync_now"), key="sync_now"):
            with st.spinner(c.tr("history.sync_progress")):
                status = engine.sync_pending(force=True)
            if status.pending:
                st.warning(c.tr("offline.pending", n=status.pending))
            else:
                st.success(c.tr("offline.synced"))

    # -- locally journaled (offline) entries --------------------------------
    entries = store.pending()
    if entries:
        st.subheader(c.tr("offline.offline"))
        for e in entries:
            data = e.get("data", {})
            when = datetime.fromtimestamp(e.get("created_offline_at", 0)).strftime("%Y-%m-%d %H:%M")
            st.markdown(
                f"<div class='result-card'><b>{data.get('predicted_disease', '?')}</b>"
                f"<p>{c.tr('history.date')}: {when} &middot; "
                f"{c.tr('history.pending')}</p></div>", unsafe_allow_html=True)

    # -- server-side history --------------------------------------------------
    try:
        from ..data import Diagnosis, get_session
        with get_session() as session:
            rows = session.query(Diagnosis).order_by(Diagnosis.timestamp.desc()).limit(50).all()
        if not rows and not entries:
            st.info(c.tr("history.empty"))
            return
        st.subheader(c.tr("history.disease"))
        for r in rows:
            when = r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else "?"
            tag = c.tr("history.synced") if r.synced else c.tr("history.pending")
            st.markdown(
                f"<div class='result-card'><b>{r.predicted_disease}</b> "
                f"({r.confidence:.0%})<p>{when} &middot; {tag}</p></div>",
                unsafe_allow_html=True)
    except Exception:
        # DB unreachable (offline) - journal view above already covered it.
        if not entries:
            st.info(c.tr("history.empty"))