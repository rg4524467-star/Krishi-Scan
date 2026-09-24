"""
Diagnose wizard.

The heart of the app.  Implements the capture -> gatekeeper -> feedback ->
retake -> classify -> (optional soil/weather) -> fusion -> recommendation flow
as a session-state state machine (Streamlit reruns the page per interaction,
so we branch on what's already stored).

Edge cases handled here (see README):
  * 3+ gatekeeper retakes -> "proceed anyway" + visible low-confidence warning.
  * unreadable image      -> explicit retake prompt, no silent failure.
  * weather/soil offline  -> diagnosis continues on photo alone.
  * OOD (low confidence)  -> "possibly outside known conditions", no forced top-1.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO as _BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

from ..config import ASSETS_DIR, INF, WX
from ..data import SyncEngine, load_latest_soil
from ..fusion import get_scorer
from ..gatekeeper import Gatekeeper
from ..inference import get_model
from ..reco import get_advice
from ..soil import SoilModule, SoilSource, SoilValues, nutrient_advisory
from ..ui import components as c
from ..ui import styles
from ..ui.state import reset_wizard, set
from ..util.image import load_image_bytes, to_np_rgb
from ..weather import WeatherClient, geocode_place, locate_by_ip

OOD_THRESHOLD = 0.35

# Ordered list shown in the crop selector.  Crops with a None family are NOT
# covered by the model - the UI says so explicitly instead of guessing.
CROPS = ("apple", "blueberry", "cherry", "grape", "maize",
         "peach", "pepper", "potato", "raspberry", "rice", "strawberry",
         "tomato", "wheat")

_CROP_FAMILY = {
    "apple": "apple", "blueberry": "blueberry", "cherry": "cherry",
    "grape": "grape", "maize": "corn", "peach": "peach",
    "pepper": "pepper", "potato": "potato", "raspberry": "raspberry",
    "strawberry": "strawberry", "tomato": "tomato",
    "rice": None, "wheat": None,
}


def _class_family(code: str) -> str:
    """Crop family of a model class code, e.g. 'corn_common_rust' -> 'corn'."""
    return code.split("_")[0]


# Reverse lookup for display labels (family -> selectable crop code).
_FAMILY_TO_CROP = {"corn": "maize", "apple": "apple", "blueberry": "blueberry",
                   "cherry": "cherry", "grape": "grape", "orange": "orange",
                   "peach": "peach", "pepper": "pepper", "potato": "potato",
                   "raspberry": "raspberry", "soybean": "soybean",
                   "squash": "squash", "strawberry": "strawberry",
                   "tomato": "tomato"}

_MODEL = None


@st.cache_data(ttl=900, show_spinner=False)
def _geocode_cached(place: str):
    """Place name -> (lat, lon); cached 15 min so repeat lookups are instant."""
    return geocode_place(place)


@st.cache_data(ttl=900, show_spinner=False)
def _weather_cached(lat: float, lon: float):
    """Current + history weather for coordinates; cached 15 min per location."""
    return WeatherClient().fetch(lat, lon)


@st.cache_data(ttl=1800, show_spinner=False)
def _locate_by_ip_cached():
    """(lat, lon, label) from the client's public IP; cached 30 min."""
    return locate_by_ip()


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = get_model()
    return _MODEL


_DIM_LABELS = {"temp_c": "weather.temp", "humidity_pct": "weather.humidity", "ph": "soil.ph"}


def fusion_dim_label(dim: str, value: Optional[float]) -> str:
    """Human label + value for one fused condition dim, e.g. 'Temperature 24.0°C'."""
    key = _DIM_LABELS.get(dim)
    label = c.tr(key) if key else dim
    if value is None:
        return label
    if dim == "temp_c":
        return f"{label} {value:.1f}°C"
    if dim == "humidity_pct":
        return f"{label} {value:.0f}%"
    return f"{label} {value:.1f}"


# ---------------------------------------------------------------------------
def render() -> None:
    st.title(c.tr("diagnose.title"))

    if st.session_state.get("dg_done"):
        _render_saved()
        return

    # -- progress stepper ---------------------------------------------------
    steps = [
        ("photo", c.tr("diagnose.step_photo")),
        ("quality", c.tr("diagnose.step_quality")),
        ("result", c.tr("diagnose.step_result")),
        ("advice", c.tr("diagnose.step_advice")),
    ]
    if st.session_state.get("dg_photo") is None:
        step_idx = 0
    elif st.session_state.get("dg_verdict") is None:
        step_idx = 1
    elif st.session_state.get("dg_raw_predictions") is None:
        step_idx = 2
    else:
        step_idx = 3
    styles.stepper(steps, step_idx)

    crop = st.selectbox(
        c.tr("diagnose.crop"),
        options=CROPS,
        format_func=lambda x: c.tr(f"diagnose.crop_{x}"),
        key="dg_crop_select",
    )
    if st.session_state.get("dg_crop") != crop:
        reset_wizard()
        set("dg_crop", crop)
    st.session_state["dg_crop"] = crop

    # -- capture stage ----------------------------------------------------
    if st.session_state.get("dg_photo") is None:
        c.empty_state("🍃", c.tr("home.illustration"))
        _render_capture(crop)
        return

    st.image(st.session_state["dg_photo"], width=320, caption=c.tr("diagnose.capture_hint"))

    # Always-available escape hatch: start a fresh test on the same crop.
    if st.button(c.tr("diagnose.new_test"), key="dg_new_test", type="secondary"):
        reset_wizard()
        st.rerun()

    # -- gatekeeper stage --------------------------------------------------
    verdict = st.session_state.get("dg_verdict")
    if verdict is None:
        _run_gatekeeper()
        st.rerun()

    _render_gatekeeper(verdict)

    # Only proceed to classification when the photo passed or farmer insisted.
    if not (verdict.passed or st.session_state.get("dg_proceed_anyway")):
        return

    # -- classification ------------------------------------------------------
    if st.session_state.get("dg_raw_predictions") is None:
        _run_classification()
        st.rerun()

    _render_classification()

    # -- optional soil + weather ---------------------------------------------
    _render_soil_section()
    _render_weather_section()

    # -- fusion + recommendation + save ---------------------------------------
    # Always re-fuse: soil/weather edits on later reruns must change the result.
    _run_fusion()
    _render_recommendation()
    _render_save_button()


# ---------------------------------------------------------------------------
def _render_capture(crop: str) -> None:
    st.info(c.tr("diagnose.capture_hint"))
    method = st.radio(c.tr("diagnose.photo_method"), ("camera", "upload"),
                      horizontal=True, key="dg_method")
    # Fresh keys per test so an old upload/capture never silently re-fires.
    slot = st.session_state.get("dg_slot", 0)

    uploaded = None
    if method == "camera":
        c.cam_guide(c.tr("diagnose.cam_guide"))
        uploaded = st.camera_input(c.tr("diagnose.capture"), key=f"dg_cam_{slot}")
    else:
        c.cam_guide(c.tr("diagnose.cam_guide"))
        uploaded = st.file_uploader(c.tr("diagnose.upload_hint"), type=["jpg", "jpeg", "png"],
                                    key=f"dg_upload_{slot}")

    if uploaded is not None:
        raw = load_image_bytes(uploaded.getvalue())
        if raw is None:
            st.error(c.tr("gatekeeper.unreadable"))
        else:
            set("dg_photo", raw)
            # The photo was stored during a rerun that already passed the
            # capture gate, so advance immediately - otherwise the user would
            # have to touch another widget (e.g. switch to camera) to see the
            # image and the diagnosis.
            st.rerun()


def _run_gatekeeper() -> None:
    photo = st.session_state.get("dg_photo")
    gk = Gatekeeper()
    verdict = gk.check(to_np_rgb(photo))
    set("dg_verdict", verdict)


def _render_gatekeeper(verdict) -> None:
    st.subheader(c.tr("gatekeeper.title"))

    if verdict.unreadable:
        st.markdown(f"<div class='gk-fail'>{c.tr('gatekeeper.unreadable')}</div>",
                    unsafe_allow_html=True)
        _retake_buttons(verdict, retries_used=st.session_state.get("dg_retakes", 0))
        return

    if verdict.passed:
        st.markdown(f"<div class='gk-pass'>{c.tr('gatekeeper.pass')}</div>", unsafe_allow_html=True)
        return

    st.markdown(f"<div class='gk-fail'>{c.tr('gatekeeper.fail')}</div>", unsafe_allow_html=True)
    for res in verdict.failed:
        st.markdown(
            f"<div class='gk-item'><b>{c.tr(res.feedback_key)}</b>"
            f"<p>{c.tr(res.hint_key)}</p></div>",
            unsafe_allow_html=True)

    _retake_buttons(verdict, retries_used=st.session_state.get("dg_retakes", 0))


def _retake_buttons(verdict, retries_used: int) -> None:
    gk = Gatekeeper()
    can_retry = gk.can_retry(retries_used) and not verdict.unreadable

    if can_retry:
        st.caption(c.tr("gatekeeper.retakes_left", n=gk.max_retakes - retries_used))
    else:
        # 3+ failures: never trap the farmer - offer proceed-anyway.
        st.markdown(f"<div class='gk-warn'>{c.tr('gatekeeper.low_confidence_warning')}</div>",
                    unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(c.tr("gatekeeper.retake"), key="btn_retake", type="primary"):
            set("dg_retakes", retries_used + 1)
            for k in ("dg_photo", "dg_verdict"):
                st.session_state[k] = None
            st.rerun()
    with col2:
        # Always available so the farmer is never trapped; warning shown above.
        if st.button(c.tr("gatekeeper.proceed_anyway"), key="btn_proceed"):
            set("dg_proceed_anyway", True)
            st.rerun()


# ---------------------------------------------------------------------------
def _run_classification() -> None:
    try:
        photo = st.session_state.get("dg_photo")
        rgb = to_np_rgb(photo)
        model = _model()
        preds = model.predict_topk(rgb)
        top_code = preds[0].code
        heatmap = model.explain(rgb, top_code)
        set("dg_raw_predictions", preds)
        set("dg_heatmap", heatmap)
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        set("dg_raw_predictions", [])


def _severity(code: str, probability: float) -> str:
    """Urgency level for a prediction: 'high' | 'medium' | 'low'."""
    if "healthy" in code:
        return "low"
    urgent = ("late_blight", "huanglongbing", "mosaic", "yellow_leaf_curl",
              "bacterial_spot", "spider_mites", "black_rot", "esca", "target_spot")
    if any(u in code for u in urgent) or probability < 0.5:
        return "high"
    return "medium"


def _render_classification() -> None:
    preds = st.session_state.get("dg_raw_predictions")
    if not preds:
        st.error(c.tr("common.error"))
        if st.session_state.get("last_error"):
            st.code(str(st.session_state["last_error"]))
        return

    st.subheader(c.tr("classifier.title"))

    if INF.demo_mode:
        st.warning(c.tr("classifier.model_warning"))

    top = preds[0]
    low_confidence = top.probability < OOD_THRESHOLD
    if low_confidence:
        st.warning(c.tr("classifier.not_confident"))
        st.warning(c.tr("classifier.outside_training"))

    _render_crop_consistency(preds)

    # -- submitted photo next to the result -------------------------------
    photo = st.session_state.get("dg_photo")
    if photo is not None:
        with st.container(border=True):
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(photo, width=180, caption=c.tr("diagnose.step_photo"))
            with cols[1]:
                label = _model().classes.label(top.code, c.lang())
                st.markdown(f"<div class='result-card'><b>{label}</b>"
                            f"<div class='pct'>{top.probability * 100:.1f}%</div></div>",
                            unsafe_allow_html=True)
                styles.severity_badge(_severity(top.code, top.probability),
                                      _severity_label(top.code, top.probability))
                st.progress(min(1.0, top.probability))

    if low_confidence:
        # Extra honesty for the floor: what the model actually saw.
        with st.expander(c.tr("common.why_toggle")):
            st.caption(c.tr("classifier.why_hint"))
            for p in preds:
                label = _model().classes.label(p.code, c.lang())
                st.markdown(f"<b>{label}</b> — {p.probability * 100:.1f}%", unsafe_allow_html=True)
                st.progress(min(1.0, p.probability))
    else:
        for i, p in enumerate(preds):
            label = _model().classes.label(p.code, c.lang())
            sev = _severity(p.code, p.probability)
            sev_label = _severity_label(p.code, p.probability)
            mark = "★ " if i == 0 else ""
            st.markdown(f"<div class='result-card'><b>{mark}{label}</b>"
                        f"<div class='pct'>{p.probability * 100:.1f}%</div>"
                        f"<div>{sev_label}</div></div>", unsafe_allow_html=True)
            st.progress(min(1.0, p.probability))

    _render_class_explanation(preds)

    # -- before/after heatmap slider ---------------------------------------
    heatmap = st.session_state.get("dg_heatmap")
    photo = st.session_state.get("dg_photo")
    if heatmap is not None and photo is not None:
        st.subheader(c.tr("classifier.heatmap"))
        blend = st.slider(c.tr("classifier.heatmap_hint"), 0, 100, 60,
                          key="hm_blend")
        overlay = c.heatmap_overlay(photo, heatmap)
        if blend >= 100:
            st.image(overlay, width=320, caption=c.tr("classifier.slider_grad"))
        elif blend <= 0:
            st.image(photo, width=320, caption=c.tr("classifier.slider_orig"))
        else:
            st.image(_blend_images(photo, overlay, blend / 100.0),
                     width=320, caption=f"{c.tr('classifier.slider_orig')} ⇄ {c.tr('classifier.slider_grad')}")
        c.speak_button(c.tr("classifier.heatmap_hint"))

    # -- one-tap retake (same crop, new photo) ------------------------------
    if st.button(c.tr("classifier.retake_photo"), key="btn_retake_same_crop"):
        for k in ("dg_photo", "dg_verdict", "dg_raw_predictions", "dg_heatmap",
                  "dg_soil", "dg_weather", "dg_fusion", "dg_predictions"):
            st.session_state[k] = None
        st.rerun()


def _blend_images(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Cross-fade two same-shape RGB arrays (t in 0..1)."""
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    return np.clip((1 - t) * a32 + t * b32, 0, 255).astype(np.uint8)


def _severity_label(code: str, probability: float) -> str:
    level = _severity(code, probability)
    key = {"high": "classifier.urgency_high",
           "medium": "classifier.urgency_medium",
           "low": "classifier.urgency_low"}[level]
    return c.tr(key)


def _render_crop_consistency(preds) -> None:
    """Honest guard: selected crop vs what the model actually recognizes."""
    crop = st.session_state.get("dg_crop")
    crop_label = c.tr(f"diagnose.crop_{crop}") if crop else ""
    fam = _CROP_FAMILY.get(crop)
    if fam is None:
        st.warning(c.tr("classifier.crop_uncovered", crop=crop_label,
                        covered=c.tr("classifier.covered_crops")))
        return
    top_fam = _class_family(preds[0].code)
    if top_fam != fam:
        st.info(c.tr("classifier.crop_mismatch", crop=crop_label,
                     matched=c.tr(f"diagnose.crop_{_FAMILY_TO_CROP.get(top_fam, top_fam)}")))


def _render_class_explanation(preds) -> None:
    """Plain-language explanation for confusing or healthy top results."""
    top, second = preds[0], preds[1] if len(preds) > 1 else None
    top_label = _model().classes.label(top.code, c.lang())
    if "healthy" in top.code:
        st.info(c.tr("classifier.healthy_explanation", label=top_label))
    elif second is not None and (top.probability - second.probability) < 0.08:
        second_label = _model().classes.label(second.code, c.lang())
        # Clarity: the second-best pattern is one tap away from the top.
        with st.expander(c.tr("common.why_toggle")):
            st.markdown(f"<div class='gk-warn'><b>{c.tr('classifier.why_title')}</b>"
                        f"<p>{c.tr('classifier.why_hint')}</p></div>",
                        unsafe_allow_html=True)
            for p in preds[:3]:
                label = _model().classes.label(p.code, c.lang())
                sev = _severity_label(p.code, p.probability)
                st.markdown(f"<b>{label}</b> — {p.probability * 100:.1f}% "
                            f"<span class='av-sev {_severity(p.code, p.probability)}'>{sev}</span>",
                            unsafe_allow_html=True)
                st.progress(min(1.0, p.probability))
        st.info(c.tr("classifier.tie_explanation",
                     a=top_label, b=second_label,
                     pa=round(top.probability * 100), pb=round(second.probability * 100)))


# ---------------------------------------------------------------------------
def _render_soil_section() -> None:
    with st.expander(c.tr("soil.collapse"), expanded=False):
        _render_soil_inner()


def _render_soil_inner() -> None:
    soil = st.session_state.get("dg_soil")
    if soil is not None and soil.present:
        st.success(c.tr("soil.saved"))
        _show_soil_values(soil)
        _render_nutrient_advisory(soil)
        return

    pending = st.session_state.get("ocr_values")
    if pending is not None and pending.present:
        _render_ocr_confirm(pending)
        return

    # Saved soil values (Soil page) -> one-tap reuse for this diagnosis.
    phone = st.session_state.get("farmer", {}).get("phone", "")
    saved = load_latest_soil(phone) if phone else None
    if saved and not st.session_state.get("dg_soil_loaded_saved"):
        st.info(c.tr("soil.saved_used"))
        if st.button(c.tr("soil.use_saved"), key="soil_use_saved", type="primary"):
            module = SoilModule()
            set("dg_soil", module.manual(n=saved.get("n"), p=saved.get("p"),
                                         k=saved.get("k"), ph=saved.get("ph")))
            st.session_state["dg_soil_loaded_saved"] = True
            st.rerun()
        st.markdown("")  # spacing before the alternative entry options

    mode = st.radio(c.tr("soil.title"), ("skip", "manual", "ocr"),
                    format_func=lambda x: {
                        "skip": c.tr("soil.skip"),
                        "manual": c.tr("soil.manual"),
                        "ocr": c.tr("soil.ocr"),
                    }[x],
                    horizontal=True, key="soil_mode")

    if mode == "skip":
        return
    if mode == "manual":
        _render_manual_entry()
    else:
        _render_ocr_capture()


def _show_soil_values(soil: SoilValues) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", soil.n)
    c2.metric("P", soil.p)
    c3.metric("K", soil.k)
    c4.metric("pH", soil.ph)
    st.caption(f"{c.tr('soil.source_manual') if soil.source == SoilSource.MANUAL else c.tr('soil.source_ocr')}")


def _render_nutrient_advisory(soil: SoilValues) -> None:
    notes = nutrient_advisory(soil)
    if not notes:
        return
    lines = "".join(f"<p>• {c.tr(n['key'], value=n['value'])}</p>" for n in notes)
    st.markdown(f"<div class='gk-warn'><b>{c.tr('soil.npk_title')}</b>{lines}</div>",
                unsafe_allow_html=True)
    st.caption(c.tr("soil.npk_note"))


def _render_manual_entry() -> None:
    n = st.number_input(c.tr("soil.n"), 0.0, 10000.0, step=1.0, key="soil_n")
    p = st.number_input(c.tr("soil.p"), 0.0, 10000.0, step=1.0, key="soil_p")
    k = st.number_input(c.tr("soil.k"), 0.0, 10000.0, step=1.0, key="soil_k")
    ph = st.number_input(c.tr("soil.ph"), 0.0, 14.0, step=0.1, key="soil_ph")
    c.small_hint(c.tr("soil.units_note"))
    if st.button(c.tr("common.save"), key="soil_save", type="primary"):
        module = SoilModule()
        set("dg_soil", module.manual(n=n, p=p, k=k, ph=ph))
        st.rerun()


def _render_ocr_capture() -> None:
    from ..util.runtime import is_pyodide
    if is_pyodide():
        # No OCR engine in the browser build - degrade explicitly, never silently.
        st.info(c.tr("soil.invalid"))
        st.info(c.tr("soil.manual"))
        return
    uploaded = st.file_uploader(c.tr("soil.upload_shc"), type=["jpg", "jpeg", "png"],
                                key="shc_upload")
    if uploaded is not None:
        raw = load_image_bytes(uploaded.getvalue())
        if raw is None:
            st.error(c.tr("gatekeeper.unreadable"))
            return
        with st.spinner(c.tr("soil.ocr_running")):
            module = SoilModule()
            values = module.run_ocr(raw)
        if values is None or not values.present:
            st.warning(c.tr("soil.invalid"))
        else:
            set("ocr_values", values)
            st.rerun()


def _render_ocr_confirm(pending: SoilValues) -> None:
    """THE mandatory confirm/edit step - OCR output is never auto-applied."""
    st.markdown(f"<div class='gk-pass'>{c.tr('soil.ocr_result')}</div>", unsafe_allow_html=True)
    st.caption(c.tr("soil.ocr_confirm_prompt"))
    _show_soil_values(pending)

    if st.button(c.tr("soil.confirm"), key="ocr_confirm", type="primary"):
        set("dg_soil", pending)
        st.session_state["ocr_values"] = None
        st.rerun()
    if st.button(c.tr("soil.edit"), key="ocr_edit"):
        n = st.number_input(c.tr("soil.n"), 0.0, 10000.0, value=pending.n or 0.0,
                            key="edit_n")
        p = st.number_input(c.tr("soil.p"), 0.0, 10000.0, value=pending.p or 0.0,
                            key="edit_p")
        k = st.number_input(c.tr("soil.k"), 0.0, 10000.0, value=pending.k or 0.0,
                            key="edit_k")
        ph = st.number_input(c.tr("soil.ph"), 0.0, 14.0, value=pending.ph or 6.5,
                             key="edit_ph")
        if st.button(c.tr("common.save"), key="ocr_edit_save", type="primary"):
            module = SoilModule()
            set("dg_soil", module.apply_edits(pending, n=n, p=p, k=k, ph=ph))
            st.session_state["ocr_values"] = None
            st.rerun()


# ---------------------------------------------------------------------------
def _render_weather_section() -> None:
    with st.expander(c.tr("weather.collapse"), expanded=False):
        _render_weather_inner()


def _render_weather_inner() -> None:
    weather = st.session_state.get("dg_weather")
    if weather is False:
        # Explicitly skipped or failed - weather is not used (soil/photo still are).
        st.caption(c.tr("weather.not_used"))
        return
    if isinstance(weather, object) and hasattr(weather, "temp_c"):
        st.success(c.tr("weather.current"))
        c1, c2, c3 = st.columns(3)
        c1.metric(c.tr("weather.temp"), f"{weather.temp_c:.1f}°C")
        c2.metric(c.tr("weather.humidity"), f"{weather.humidity_pct:.0f}%")
        c3.metric(c.tr("weather.rain"), f"{weather.rain_mm:.1f}mm")
        cond = getattr(weather, "condition", "") or ""
        extra = c.tr(f"weather.{cond}") if cond.startswith("condition_") else cond
        src = st.session_state.get("dg_weather_source", "")
        if src == "gps":
            label = st.session_state.get("dg_gps_label", "") or ""
            note = f"{c.tr('weather.gps_using')}: {label}" if label else c.tr("weather.gps_using")
            st.caption(f"{extra} · {note}")
        else:
            st.caption(extra)
        st.caption(c.tr("weather.history", n=WX.history_days) + f" — "
                   f"{weather.history_avg_temp_c:.1f}°C / {weather.history_avg_humidity_pct:.0f}%")
        return

    if st.button(c.tr("weather.skip"), key="wx_skip"):
        set("dg_weather", False)   # explicit skip marker
        st.rerun()

    # -- Auto location (IP-based, no permission popup: reliable in Streamlit) --
    if st.button(c.tr("weather.gps"), key="wx_gps_ask", type="secondary"):
        with st.spinner(c.tr("weather.locating")):
            loc = _locate_by_ip_cached()
        if loc is None:
            st.warning(c.tr("weather.gps_fail"))
            set("dg_weather", False)
        else:
            lat, lon, label = loc
            with st.spinner(c.tr("weather.loading")):
                try:
                    data = _weather_cached(lat, lon)
                    set("dg_weather", data)
                    st.session_state["dg_weather_source"] = "gps"
                    st.session_state["dg_gps_label"] = label
                except Exception:
                    st.warning(c.tr("weather.failed"))
                    set("dg_weather", False)      # degrade gracefully
        st.rerun()

    location = st.text_input(c.tr("weather.location"), c.tr("weather.location_ph"),
                             key="wx_loc")
    farmer_loc = st.session_state.get("farmer", {}).get("location", "")
    use_farmer = st.checkbox(c.tr("weather.use_farmer_location"), value=bool(farmer_loc),
                             key="wx_use_farmer")
    if st.button(c.tr("weather.load"), key="wx_load", type="primary"):
        place = farmer_loc if use_farmer and farmer_loc else location
        if not place:
            st.info(c.tr("weather.location_ph"))
            return
        with st.spinner(c.tr("weather.loading")):
            coords = _geocode_cached(place)
            if coords is None:
                st.warning(c.tr("weather.failed"))      # degrade gracefully
                set("dg_weather", False)
                st.rerun()
            try:
                data = _weather_cached(*coords)
                set("dg_weather", data)
            except Exception:
                st.warning(c.tr("weather.failed"))      # network failure: photo-only
                set("dg_weather", False)
        st.rerun()


# ---------------------------------------------------------------------------
def _run_fusion() -> None:
    raw = st.session_state.get("dg_raw_predictions") or []
    soil = st.session_state.get("dg_soil")
    weather = st.session_state.get("dg_weather")

    conditions = {}
    if isinstance(weather, object) and hasattr(weather, "fusion_conditions") and weather:
        conditions.update(weather.fusion_conditions())
    if soil is not None and soil.present:
        for k, v in soil.as_dict().items():
            if v is not None and k == "ph":
                conditions["ph"] = v

    result = get_scorer().reweight(raw, conditions)
    set("dg_fusion", result)
    set("dg_predictions", result.candidates)


def _render_treatments(treatments) -> None:
    """Structured treatment table: chemicals and farm practices."""
    st.subheader(c.tr("reco.treat_title"))
    st.caption(c.tr("reco.treat_hint"))
    chemical_badge = c.tr("reco.treat_chemical")
    cultural_badge = c.tr("reco.treat_cultural")

    for i, treat in enumerate(treatments):
        ttype = treat.get("type", "cultural")
        is_chemical = ttype == "chemical"
        badge = chemical_badge if is_chemical else cultural_badge
        tag_cls = "av-tag av-tag-chem" if is_chemical else "av-tag av-tag-cult"
        icon = "🧴" if is_chemical else "🌾"
        name = treat.get("name", "")
        dose = treat.get("dose", "")
        when = treat.get("when", "")
        phi = treat.get("phi", "")
        note = treat.get("note", "")

        rows = []
        if dose:
            rows.append(f"<tr><td class='av-k'>{_esc(c.tr('reco.treat_dose'))}</td>"
                        f"<td><b class='av-dose'>{_esc(dose)}</b></td></tr>")
        if when:
            rows.append(f"<tr><td class='av-k'>{_esc(c.tr('reco.treat_when'))}</td>"
                        f"<td>{_esc(when)}</td></tr>")
        if phi:
            rows.append(f"<tr><td class='av-k'>{_esc(c.tr('reco.treat_phi'))}</td>"
                        f"<td><span class='av-phi'>{_esc(phi)}</span></td></tr>")
        if note:
            rows.append(f"<tr><td class='av-k'>{_esc(c.tr('reco.treat_note'))}</td>"
                        f"<td class='av-note'>{_esc(note)}</td></tr>")
        source = treat.get("source", "")
        source_url = treat.get("source_url", "")
        if source:
            src_txt = _esc(source)
            if source_url:
                src_txt = f"<a href='{_esc(source_url)}' target='_blank'>{src_txt} ↗</a>"
            rows.append(f"<tr><td class='av-k'>{_esc(c.tr('reco.treat_source'))}</td>"
                        f"<td class='av-src'>{src_txt}</td></tr>")

        verify = treat.get("verify")
        verify_pill = ""
        if verify:
            verify_pill = (f" <span class='av-tag av-tag-verify' title='{_esc(c.tr('reco.treat_verify'))}'>"
                           f"⚠ {_esc(c.tr('reco.treat_verify'))}</span>")
        head = (f"<span class='av-tag-ico'>{icon}</span>"
                f"<span class='{tag_cls}'>{_esc(badge)}</span> "
                f"<b class='av-treat-name'>{_esc(name)}</b>{verify_pill}")
        html = (f"<div class='result-card av-treat{'-chem' if is_chemical else '-cult'}'>"
                f"<div class='av-treat-head'>{head}</div>"
                f"<table class='av-treat-table'>{''.join(rows)}</table></div>")
        st.markdown(html, unsafe_allow_html=True)

    st.caption(c.tr("reco.treat_disclaimer"))


def _esc(text: str) -> str:
    """HTML-escape a string for embedding in markdown."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _render_recommendation() -> None:
    final = st.session_state.get("dg_predictions") or st.session_state.get("dg_raw_predictions")
    if not final:
        return
    fusion = st.session_state.get("dg_fusion")
    if fusion is not None and fusion.changed:
        st.subheader(c.tr("fusion.title"))
        used = [fusion_dim_label(d, fusion.used_conditions.get(d)) for d in fusion.used_dims]
        st.caption(f"{c.tr('fusion.adjusted')} " + (" · ".join(used) if used else ""))
        for p in fusion.candidates:
            label = _model().classes.label(p.code, c.lang())
            prior = fusion.env_priors.get(p.code)
            fit_html = ""
            if prior is not None:
                fit_html = f" <span class='av-hint'>({c.tr('fusion.env_fit', pct=round(prior * 100))})</span>"
            st.markdown(f"<div class='result-card'><b>{label}</b>"
                        f"<div class='pct'>{p.probability * 100:.1f}%</div>{fit_html}</div>",
                        unsafe_allow_html=True)
            st.progress(min(1.0, p.probability))
    elif fusion is not None:
        st.caption(c.tr("fusion.no_data"))

    # ---- action layer: NEVER end on a bare label -------------------------
    top_code = final[0].code
    advice = get_advice(top_code, c.lang())
    st.subheader(c.tr("reco.title"))
    st.markdown(f"<div class='result-card'><b>{advice.disease_label}</b>"
                f"<p>{advice.advice}</p></div>", unsafe_allow_html=True)

    # ---- structured treatment (what to use, dose, when) -----------------
    if advice.treatments:
        _render_treatments(advice.treatments)

    # ---- plain-language explanation (farmers understand the result) -------
    top, second = final[0], final[1] if len(final) > 1 else None
    top_label = _model().classes.label(top.code, c.lang())
    with st.expander(c.tr("wx.explain_what"), expanded=False):
        st.markdown(c.tr("wx.explain_body", disease=top_label))
        if second is not None and "healthy" not in top.code:
            sec_label = _model().classes.label(second.code, c.lang())
            if abs(top.probability - second.probability) < 0.08:
                st.warning(c.tr("wx.close_call", a=top_label, pa=f"{top.probability * 100:.0f}",
                                b=sec_label, pb=f"{second.probability * 100:.0f}"))

    # -- voice: read the FULL result (disease + advice + next steps) -------
    speech_extra = ""
    if advice.treatments:
        names = [t.get("name", "") for t in advice.treatments if t.get("name")]
        speech_extra = ". " + " ".join(names)
    full_speech = (f"{advice.disease_label}. {advice.advice}{speech_extra}. "
                   + " ".join(advice.next_steps))
    c.speak_button(full_speech, key="speak_reco_full")

    st.markdown(f"<div class='result-card'><b>{c.tr('reco.next')}</b>"
                f"{''.join(f'<p>• {s}</p>' for s in advice.next_steps)}</div>",
                unsafe_allow_html=True)
    st.caption(c.tr("reco.not_sure"))

    _render_share_card(final, advice, top_code)


# ---------------------------------------------------------------------------
def _render_share_card(final, advice, top_code: str) -> None:
    """Officer report download (PDF) for an agri-officer handoff."""
    photo = st.session_state.get("dg_photo")
    if photo is None:
        return

    with st.expander(c.tr("share.title")):
        try:
            _render_officer_report_pdf(top_code, advice)
        except Exception as exc:
            # Never block the diagnosis on a share-card rendering problem.
            import traceback as _tb
            st.session_state["last_error"] = (
                f"{exc}\n{_tb.format_exc(limit=3)}")
            st.caption(c.tr("common.error"))


# ---------------------------------------------------------------------------
def _wrap_text(text: str, font, width: int) -> list:
    """Greedy word-wrap for reportlab-free PIL text."""
    lines, current = [], ""
    for word in text.split():
        trial = (current + " " + word).strip()
        try:
            w = font.getlength(trial)
        except Exception:
            w = len(trial) * 8
        if w <= width and current:
            current = trial
        elif not current:
            current = word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
def _render_officer_report_pdf(top_code: str, advice) -> None:
    """
    Full A4-format PDF report for the agriculture officer, with the leaf photo,
    diagnosis + confidence, recorded weather/soil, treatment guidance and the
    next-step reference.  Built with PIL at 3x scale so printed/zoomed text
    stays crisp (PIL-PDF embeds a raster; at 72dpi it looks blurry).
    """
    SC = 3                                  # supersampling factor -> ~216 dpi
    from PIL import Image, ImageDraw
    from datetime import datetime

    photo = st.session_state.get("dg_photo")
    if photo is None:
        return

    try:
        from ..util.image import load_font
        font_brand = load_font(26 * SC, bold=True)
        font_h1 = load_font(15 * SC, bold=True)
        font_body = load_font(11 * SC)
        font_small = load_font(9 * SC)

        final = (st.session_state.get("dg_predictions")
                 or st.session_state.get("dg_raw_predictions") or [])
        conf_pct = final[0].probability * 100 if final else 0.0
        crop = st.session_state.get("dg_crop")
        crop_label = c.tr(f"diagnose.crop_{crop}") if crop else "-"

        W, H = 595 * SC, 842 * SC           # A4 portrait
        card = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(card)
        pad = int(36 * SC)
        lh = int(20 * SC)                    # row height

        # -- header band ---------------------------------------------------
        d.rectangle([0, 0, W, int(80 * SC)], fill=(11, 94, 59))
        d.rectangle([0, int(80 * SC), W, int(86 * SC)], fill=(240, 196, 60))
        tx = pad
        try:
            from ..util.brand import make_logo, BRAND_NAME
            logo = make_logo(int(48 * SC))
            # Flatten alpha onto the card (no mask-paste: works identically on
            # desktop Pillow and the Pillow build shipped inside Pyodide).
            tile = Image.new("RGB", logo.size, (255, 255, 255))
            tile.paste(logo, (0, 0))
            card.paste(tile, (pad, int(11 * SC)))
            tx = pad + tile.width + int(20 * SC)
            d.text((tx, int(24 * SC)), BRAND_NAME, fill=(255, 255, 255),
                   font=font_brand)
            d.text((tx, int(52 * SC)), c.tr("wx.report_title"),
                   fill=(220, 240, 230), font=font_small)
        except Exception:
            # Branding is cosmetic - never let it break the share-card PDF.
            d.text((tx, int(20 * SC)), c.tr("wx.report_title"), fill=(220, 240, 230),
                   font=font_small)

        y = int(110 * SC)
        for label, value in (
            (c.tr("wx.report_date"), datetime.now().strftime("%d %b %Y, %H:%M")),
            (c.tr("wx.report_crop"), crop_label),
            (c.tr("wx.report_disease"), advice.disease_label),
            (c.tr("wx.report_conf"), f"{conf_pct:.1f}%"),
        ):
            d.text((pad, y), f"{label}:", fill=(90, 100, 94), font=font_small)
            d.text((pad + int(130 * SC), y), value, fill=(20, 30, 24), font=font_body)
            y += lh
        y += int(8 * SC)

        # -- weather/soil text (left) + leaf photo (right) ------------------
        env_lines = []
        weather = st.session_state.get("dg_weather")
        if isinstance(weather, object) and hasattr(weather, "temp_c") and weather:
            env_lines.append(f"{c.tr('weather.temp')}: {weather.temp_c:.1f}°C   "
                             f"{c.tr('weather.humidity')}: {weather.humidity_pct:.0f}%   "
                             f"{c.tr('weather.rain')}: {weather.rain_mm:.1f}mm")
        soil = st.session_state.get("dg_soil")
        if soil is not None and soil.present:
            env_lines.append(", ".join(f"{k}={v}" for k, v in soil.as_dict().items()
                                       if v is not None))
        if not env_lines:
            env_lines = [c.tr("wx.report_no_env")]

        leaf = Image.fromarray(to_np_rgb(photo)).convert("RGB")
        leaf.thumbnail((int(240 * SC), int(240 * SC)))

        d.text((pad, y), c.tr("wx.report_conditions"), fill=(17, 82, 52),
               font=font_h1)
        # environment text stays left of the photo's column
        env_w = W - pad * 2 - leaf.width - int(20 * SC)
        yy = y + int(26 * SC)
        for line in env_lines:
            for sub in _wrap_text(line, font_body, max(int(120 * SC), env_w)):
                d.text((pad, yy), sub, fill=(40, 50, 44), font=font_body)
                yy += int(18 * SC)
        # photo anchored to the top of the conditions block
        card.paste(leaf, (W - pad - leaf.width, y))

        # -- advice + treatment box (full width below the photo/conditions) --
        y = max(yy, y + leaf.height) + int(18 * SC)
        content_w = W - 2 * pad - int(16 * SC)

        advice_lines = _wrap_text(advice.advice, font_body, content_w)
        treat_lines = []
        if advice.treatments:
            for t in advice.treatments:
                line = t.get("name") or ""
                bits = []
                if t.get("dose"):
                    bits.append(t["dose"])
                if t.get("when"):
                    bits.append(t["when"])
                if bits:
                    line = f"{line} — {', '.join(bits)}" if line else ", ".join(bits)
                for sub in _wrap_text(line, font_body, content_w - int(14 * SC)):
                    treat_lines.append(sub)
                if t.get("note"):
                    for sub in _wrap_text(t["note"], font_small, content_w - int(14 * SC)):
                        treat_lines.append(f"({sub})")

        next_lines = []
        for step in advice.next_steps:
            for i, sub in enumerate(_wrap_text(step, font_body, content_w - int(14 * SC))):
                next_lines.append(("• " if i == 0 else "   ") + sub)

        # compute box height to fit advice + treatments + next steps
        box_h = (int(34 * SC)
                 + len(advice_lines) * int(20 * SC)
                 + (int(30 * SC) + len(treat_lines) * int(20 * SC) if treat_lines else 0)
                 + (int(30 * SC) + len(next_lines) * int(20 * SC) if next_lines else 0)
                 + int(24 * SC))
        # never overflow past the footer
        max_box_h = H - int(58 * SC) - int(12 * SC) - y
        if box_h > max_box_h:
            box_h = max(int(60 * SC), max_box_h)
        d.rectangle([pad - int(8 * SC), y - int(6 * SC),
                     W - pad + int(8 * SC), y + box_h - int(2 * SC)],
                    fill=(238, 247, 242), outline=(200, 220, 208))
        d.text((pad, y), c.tr("wx.report_advice"), fill=(17, 82, 52), font=font_h1)
        yy = y + int(30 * SC)
        for line in advice_lines:
            d.text((pad, yy), line, fill=(40, 50, 44), font=font_body)
            yy += int(20 * SC)

        if treat_lines:
            yy += int(6 * SC)
            d.text((pad, yy), c.tr("reco.treat_title"), fill=(17, 82, 52), font=font_h1)
            yy += int(26 * SC)
            for line in treat_lines:
                d.text((pad + int(8 * SC), yy), line, fill=(40, 50, 44), font=font_body)
                yy += int(20 * SC)

        if next_lines:
            yy += int(6 * SC)
            d.text((pad, yy), c.tr("wx.report_next"), fill=(17, 82, 52), font=font_h1)
            yy += int(26 * SC)
            for line in next_lines:
                d.text((pad, yy), line, fill=(40, 50, 44), font=font_body)
                yy += int(20 * SC)

        # -- footer -----------------------------------------------------------
        footer_y = H - int(58 * SC)
        d.rectangle([0, footer_y, W, H], fill=(245, 247, 246))
        d.text((pad, footer_y + int(14 * SC)), c.tr("wx.report_footer"),
               fill=(120, 130, 122), font=font_small)

        buf = _BytesIO()
        card.save(buf, format="PDF", resolution=72.0 * SC)
        buf.seek(0)
        st.download_button(c.tr("wx.share_pdf"), data=buf.getvalue(),
                           file_name=f"krishi_scan_report_{top_code}.pdf",
                           mime="application/pdf", key="share_pdf_dl",
                           type="secondary", use_container_width=True)
    except Exception:
        st.caption(c.tr("common.error"))


# ---------------------------------------------------------------------------
def _render_save_button() -> None:
    final = st.session_state.get("dg_predictions") or st.session_state.get("dg_raw_predictions")
    if not final:
        return
    if st.button(c.tr("common.save"), key="save_diag", type="primary"):
        try:
            _persist()
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            st.error(c.tr("common.error"))


def _persist() -> None:
    photo = st.session_state.get("dg_photo")
    image_ref = _save_image(photo)
    top = (st.session_state.get("dg_predictions") or st.session_state.get("dg_raw_predictions"))[0]
    soil = st.session_state.get("dg_soil")
    weather = st.session_state.get("dg_weather")

    soil_json = soil.as_json() if soil is not None else None
    weather_json = weather.as_json() if isinstance(weather, object) and hasattr(weather, "as_json") and weather else None

    verdict = st.session_state.get("dg_verdict")
    gatekeeper_failed = verdict is not None and not verdict.passed

    engine = SyncEngine()
    result = engine.record_diagnosis(
        image_ref=image_ref,
        crop=st.session_state.get("dg_crop"),
        predicted_disease=top.code,
        confidence=top.probability,
        soil_data_json=soil_json,
        weather_data_json=weather_json,
        gatekeeper_failed=gatekeeper_failed,
    )
    set("dg_done", {"record": result, "disease": top.code,
                    "synced": result.get("synced")})


def _save_image(photo: Optional[np.ndarray]) -> str:
    uploads = ASSETS_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    name = f"diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(photo) % 1000:03d}.png"
    from PIL import Image
    Image.fromarray(to_np_rgb(photo)).save(uploads / name)
    return str(uploads / name)


def _render_saved() -> None:
    done = st.session_state.get("dg_done", {})
    if done.get("synced"):
        st.success(c.tr("offline.online"))
    else:
        st.warning(c.tr("offline.offline"))
    if st.button(c.tr("common.continue"), key="new_diag", type="primary"):
        reset_wizard()
        st.rerun()
    st.caption(c.tr("diagnose.start_another"))