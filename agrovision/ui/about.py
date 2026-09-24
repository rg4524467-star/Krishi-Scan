"""About/help page - offline explanation and disclaimer."""
from __future__ import annotations

import streamlit as st

from ..ui import components as c


def render() -> None:
    st.title(c.tr("about.title"))

    st.subheader(c.tr("about.offline_info"))
    st.info(c.tr("about.offline_text"))

    st.subheader(c.tr("about.disclaimer"))
    st.warning(c.tr("about.disclaimer_text"))

    st.caption(f"Krishi Scan · MIT-style scaffold · v0.1.0")