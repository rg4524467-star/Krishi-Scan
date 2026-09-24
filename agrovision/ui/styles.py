"""
AgroVision design system.

PALETTE POLICY: this app is deliberately LIGHT-THEME ONLY.
  * `.streamlit/config.toml` pins Streamlit's theme.base to "light", and the
    settings toolbar is hidden (no user toggle), so the framework itself always
    renders light.
  * The custom CSS below therefore uses a single light palette and sets
    `color-scheme: light` so native inputs (date pickers, scrollbars) stay
    light too.  No dark-mode media queries exist here, so a dark OS can never
    produce a broken half-dark UI.

Goals:
  1. Hide Streamlit's default chrome (header, deploy menu, toolbar, footer,
     sidebar) so the page reads as a native full-bleed phone app.
  2. Indian design language: saffron/green tricolor accent, deep-leaf green
     primary, warm neutral background, rounded touch-friendly cards, subtle
     textile-like dot motif in the header.
  3. No fixed widths anywhere (Hindi/Tamil text runs longer than English).
"""

_HEADER_THEME_CSS = """
<style>
:root {
  --av-bg: #f5f6f0;
  --av-card: #ffffff;
  --av-text: #17291f;
  --av-muted: #5b6b60;
  --av-border: #dde7dd;
  --av-primary: #0b5e3b;
  --av-primary-soft: #e3f1e6;
  --av-saffron: #e8871e;
  --av-saffron-soft: #fdf1e0;
  --av-red: #a6352a;
  --av-shadow: 0 4px 14px rgba(20, 60, 40, 0.10);
  --av-header-text: #123623;
  color-scheme: light;
}

/* ------------------------------------------------------------------ */
/* Kill Streamlit chrome                                               */
/* ------------------------------------------------------------------ */
#MainMenu, footer, [data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"] { display: none !important; }

.stApp { background: var(--av-bg); }
html, body, .stApp { min-height: 100%; }
.block-container { padding-top: 0.5rem; padding-bottom: 4rem; max-width: 760px; }

/* ------------------------------------------------------------------ */
/* Brand header                                                        */
/* ------------------------------------------------------------------ */
.av-tricolor {
  height: 5px; border-radius: 0 0 6px 6px;
  background: linear-gradient(90deg, #FF9933 0%, #FF9933 33.3%,
              #FFFFFF 33.3%, #FFFFFF 66.6%, #138808 66.6%, #138808 100%);
}
.av-header {
  margin: 0.6rem 0 0.8rem;
  background:
    radial-gradient(circle at 92% 20%, var(--av-primary-soft) 0 34px, transparent 36px),
    radial-gradient(circle at 6% 90%, var(--av-saffron-soft) 0 26px, transparent 28px),
    var(--av-card);
  border: 1px solid var(--av-border);
  border-radius: 16px;
  box-shadow: var(--av-shadow);
  padding: 0.9rem 1rem;
  display: flex; align-items: center; gap: 0.75rem;
}
.av-logo {
  width: 52px; height: 52px; flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: inset 0 -3px 0 rgba(0,0,0,.18);
}
.av-logo img { width: 100%; height: 100%; display: block; border-radius: 14px; }
.av-brand-name { font-size: 1.35rem; font-weight: 800; color: var(--av-header-text); letter-spacing: .2px; }
.av-brand-name span { color: var(--av-saffron); }
.av-tagline { font-size: .85rem; color: var(--av-muted); margin-top: 2px; }
.av-status { margin-left: auto; text-align: right; }

/* Offline/online chips */
.av-chip {
  display: inline-block; font-size: .72rem; font-weight: 700;
  padding: 3px 10px; border-radius: 20px; letter-spacing: .3px;
}
.av-chip.online { background: var(--av-primary-soft); color: var(--av-primary); }
.av-chip.offline { background: var(--av-saffron-soft); color: var(--av-saffron); }
.av-chip.warn { background: var(--av-red); color: #fff; }

/* ------------------------------------------------------------------ */
/* Weather strip (small minimal box under the header)                  */
/* ------------------------------------------------------------------ */
.av-weather {
  display: flex; align-items: center; flex-wrap: wrap; gap: 4px 12px;
  font-size: .83rem; color: var(--av-muted);
  background: var(--av-card); border: 1px solid var(--av-border);
  border-radius: 12px; padding: 7px 14px; margin: 0 0 .7rem;
}
.av-weather .av-wx-ic { font-size: 1.05rem; }
.av-weather .av-wx-loc { font-weight: 700; color: var(--av-text); }
.av-weather .av-wx-val { font-weight: 600; color: var(--av-text); }
.av-weather .av-wx-tag {
  font-size: .68rem; font-weight: 700; letter-spacing: .3px;
  background: var(--av-saffron-soft); color: var(--av-saffron);
  border-radius: 20px; padding: 1px 8px; margin-left: auto;
}

/* ------------------------------------------------------------------ */
/* Navigation tab bar (native segmented control)                       */
/* ------------------------------------------------------------------ */
div[data-testid="stSegmentedControl"] {
  background: var(--av-card);
  border: 1px solid var(--av-border);
  border-radius: 14px;
  box-shadow: var(--av-shadow);
  padding: 4px;
  margin-bottom: 1.0rem;
}

/* ------------------------------------------------------------------ */
/* Buttons + inputs                                                    */
/* ------------------------------------------------------------------ */
.stButton > button, .stDownloadButton > button {
  min-height: 54px; border-radius: 14px; font-size: 1.02rem; font-weight: 600;
  width: 100%; border: 1px solid var(--av-border); background: var(--av-card);
  color: var(--av-text);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(160deg, var(--av-primary), #0d7a4a);
  color: #fff; border-color: transparent; box-shadow: var(--av-shadow);
}
.stButton > button:hover { border-color: var(--av-primary); }
.stButton > button[kind="primary"]:hover { filter: brightness(1.05); }
input, textarea, select, .stTextInput input, .stNumberInput input {
  min-height: 52px !important; font-size: 1.05rem !important;
}
/* Compact language dropdown + TTS toggle row */
div[data-testid="stSelectbox"] > div {
  min-height: 44px; border-radius: 12px;
}
div[data-testid="stSelectbox"] > div > div { min-height: 42px; }
div[data-testid="stBaseWidgetToggle"] { min-height: 44px; }
[data-testid="stToggle"] { display: flex; align-items: center; }
[data-testid="stToggle"] p { font-size: .85rem; font-weight: 600; color: var(--av-muted); }
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div {
  background: var(--av-card) !important; color: var(--av-text) !important;
  border-color: var(--av-border) !important; border-radius: 12px !important;
}

/* ------------------------------------------------------------------ */
/* Info cards (gatekeeper, results, advice)                            */
/* ------------------------------------------------------------------ */
.gk-pass { background: var(--av-primary-soft); color: var(--av-primary); padding: 12px; border-radius: 12px; font-weight: 700; }
.gk-fail { background: var(--av-red); color: #fff; padding: 12px; border-radius: 12px; font-weight: 700; }
.gk-warn { background: var(--av-saffron-soft); color: var(--av-saffron); padding: 12px; border-radius: 12px; }
.gk-item { padding: 10px 0; border-bottom: 1px solid var(--av-border); }
.gk-item b { font-size: 1.05rem; color: var(--av-text); }
.gk-item p { color: var(--av-muted); margin: 4px 0 0; }

.result-card {
  background: var(--av-card); border: 1px solid var(--av-border);
  border-radius: 14px; padding: 14px; margin: 10px 0; box-shadow: var(--av-shadow);
  color: var(--av-text);
}
.result-card .pct { font-size: 1.3rem; font-weight: 800; color: var(--av-primary); }

/* ------------------------------------------------------------------ */
/* Farmer profile card (home)                                          */
/* ------------------------------------------------------------------ */
.av-profile-head {
  display: flex; align-items: center; gap: .7rem;
  margin: .2rem 0 .9rem;
}
.av-avatar {
  width: 46px; height: 46px; flex: 0 0 auto; border-radius: 50%;
  background: linear-gradient(145deg, var(--av-primary), #0d7a4a);
  color: #fff; display: flex; align-items: center; justify-content: center;
  font-size: 1.35rem; box-shadow: var(--av-shadow);
}
.av-profile-title { font-size: 1.02rem; font-weight: 800; color: var(--av-text); }
.av-profile-sub { font-size: .78rem; color: var(--av-muted); }
/* icon prefix inside text inputs */
.av-field-icon {
  position: absolute; left: 12px; top: 50%; transform: translateY(-50%);
  font-size: 1.1rem; pointer-events: none; z-index: 1;
}
div[data-testid="stTextInput"] > div { position: relative; }

/* Metrics render as cards */
div[data-testid="stMetric"] {
  background: var(--av-card); border: 1px solid var(--av-border);
  border-radius: 12px; padding: 12px; color: var(--av-text);
}

/* taller progress bars, easier to read */
div[data-testid="stProgress"] > div > div > div { height: 22px; border-radius: 12px; }

/* streamlit info/warning/error boxes follow our palette */
.stAlert { border-radius: 12px !important; }

/* captions + hints */
.av-hint { color: var(--av-muted); font-size: .82rem; }

/* ------------------------------------------------------------------ */
/* Treatment table (recommendations)                                   */
/* ------------------------------------------------------------------ */
.av-treat-head { margin-bottom: 8px; display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.av-tag {
  display: inline-block; background: var(--av-primary); color: #fff;
  border-radius: 999px; padding: 2px 10px; font-size: .72rem; font-weight: 700;
  letter-spacing: .2px; white-space: nowrap;
}
.av-tag-chem { background: #b45309; }   /* amber: pesticide */
.av-tag-cult { background: #15803d; }   /* green: farm practice */
.av-tag-verify { background: #8a5a00; } /* gold: confirm locally */
.av-tag-ico { font-size: 1.1rem; }
.av-treat-name { font-size: 1rem; }
.av-treat { border-left: 5px solid var(--av-border); }
.av-treat-chem { border-left-color: #b45309; }
.av-treat-cult { border-left-color: #15803d; }
.av-treat-table { width: 100%; border-collapse: collapse; }
.av-treat-table td { padding: 6px 0; vertical-align: top; font-size: .9rem; }
.av-treat-table td.av-k {
  width: 34%; color: var(--av-muted); font-size: .8rem; font-weight: 600;
  padding-right: 10px;
}
.av-treat-table tr + tr { border-top: 1px dashed var(--av-border); }
.av-dose { color: var(--av-primary); font-size: .95rem; }
.av-phi {
  display: inline-block; color: #b45309; border: 1px solid #f3d9a6;
  background: #fdf3e0; border-radius: 999px; padding: 1px 10px; font-size: .78rem;
  font-weight: 600;
}
.av-note { color: var(--av-muted); font-size: .82rem; font-style: italic; }
.av-src { color: var(--av-muted); font-size: .8rem; }
.av-src a { color: var(--av-primary); text-decoration: none; }
.av-src a:hover { text-decoration: underline; }

/* ------------------------------------------------------------------ */
/* Progress stepper                                                    */
/* ------------------------------------------------------------------ */
.av-stepper { display: flex; gap: 4px; margin: 0.5rem 0 1rem; }
.av-step {
  flex: 1; text-align: center; font-size: .78rem; font-weight: 700;
  color: var(--av-muted); background: var(--av-card);
  border: 1px solid var(--av-border); border-radius: 12px;
  padding: 8px 4px;
}
.av-step .av-dot {
  display: block; width: 22px; height: 22px; line-height: 22px;
  margin: 0 auto 4px; border-radius: 50%; font-size: .72rem;
  background: var(--av-border); color: var(--av-text);
}
.av-step.done { color: var(--av-primary); }
.av-step.done .av-dot { background: var(--av-primary-soft); }
.av-step.active {
  background: var(--av-primary); color: #fff; border-color: transparent;
  box-shadow: var(--av-shadow);
}
.av-step.active .av-dot { background: rgba(255,255,255,.25); color: #fff; }

/* ------------------------------------------------------------------ */
/* Severity badges                                                     */
/* ------------------------------------------------------------------ */
.av-sev {
  display: inline-block; font-size: .72rem; font-weight: 700;
  padding: 3px 10px; border-radius: 20px; letter-spacing: .3px;
}
.av-sev.high { background: var(--av-red); color: #fff; }
.av-sev.medium { background: var(--av-saffron-soft); color: var(--av-saffron); }
.av-sev.low { background: var(--av-primary-soft); color: var(--av-primary); }

/* ------------------------------------------------------------------ */
/* Result card entrance animation                                      */
/* ------------------------------------------------------------------ */
@keyframes avFadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.result-card { animation: avFadeUp .35s ease both; }

/* ------------------------------------------------------------------ */
/* Camera / capture guide card                                         */
/* ------------------------------------------------------------------ */
.av-cam-guide {
  display: flex; align-items: center; gap: .6rem;
  background: var(--av-primary-soft); color: var(--av-primary);
  border: 1px dashed var(--av-primary); border-radius: 14px;
  padding: .65rem .8rem; font-size: .9rem; margin: .4rem 0 .9rem;
}
.av-cam-guide .av-leaf-icon {
  flex: 0 0 auto; width: 40px; height: 40px; border-radius: 50%;
  background: var(--av-card); display: flex; align-items: center;
  justify-content: center; font-size: 1.3rem;
  border: 2px solid var(--av-primary);
}

/* ------------------------------------------------------------------ */
/* Empty state / illustration                                          */
/* ------------------------------------------------------------------ */
.av-empty {
  text-align: center; padding: 1.6rem 1rem; color: var(--av-muted);
  background: var(--av-card); border: 1px solid var(--av-border);
  border-radius: 16px; margin: 1rem 0;
}
.av-empty .av-empty-icon { font-size: 3rem; display: block; margin-bottom: .5rem; animation: avFloat 3s ease-in-out infinite; }
@keyframes avFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }

/* ------------------------------------------------------------------ */
/* Share card preview                                                  */
/* ------------------------------------------------------------------ */
.av-share-preview {
  background: var(--av-card); border: 1px solid var(--av-border);
  border-radius: 14px; padding: 12px; box-shadow: var(--av-shadow);
}
.av-share-preview img { border-radius: 10px; }
</style>
"""


def inject() -> None:
    import streamlit as st
    st.markdown(_HEADER_THEME_CSS, unsafe_allow_html=True)


def small_hint(text: str) -> None:
    import streamlit as st
    st.markdown(f"<small class='av-hint'>{text}</small>", unsafe_allow_html=True)


def stepper(steps: list, current: int) -> None:
    """
    Horizontal progress indicator.
    `steps`: list of (key, label) in order; `current`: index of the active step.
    Steps before `current` render as done; the current one is highlighted.
    """
    import streamlit as st
    html = []
    for i, (key, label) in enumerate(steps):
        cls = "av-step done" if i < current else ("av-step active" if i == current else "av-step")
        icon = "✓" if i < current else str(i + 1)
        html.append(f"<div class='{cls}'><span class='av-dot'>{icon}</span>{label}</div>")
    st.markdown(f"<div class='av-stepper'>{''.join(html)}</div>", unsafe_allow_html=True)


def severity_badge(level: str, label: str) -> None:
    """Coloured urgency pill: level in {'high','medium','low'}."""
    import streamlit as st
    st.markdown(f"<span class='av-sev {level}'>{label}</span>", unsafe_allow_html=True)


def cam_guide(text: str) -> None:
    """Leaf-positioning hint card shown under the capture widget."""
    import streamlit as st
    st.markdown(
        f"<div class='av-cam-guide'><span class='av-leaf-icon'>🍃</span>"
        f"<span>{text}</span></div>",
        unsafe_allow_html=True)


def empty_state(icon: str, text: str) -> None:
    """Centred illustration + caption for blank screens."""
    import streamlit as st
    st.markdown(
        f"<div class='av-empty'><span class='av-empty-icon'>{icon}</span>"
        f"<div>{text}</div></div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable header / nav renderers (called from app.py)
# ---------------------------------------------------------------------------
def brand_header(tagline: str, online: bool, chip_label: str, farmer_loc: str = "") -> None:
    import streamlit as st
    chip = ("<span class='av-chip online'>● " + chip_label + "</span>") if online else \
           ("<span class='av-chip offline'>● " + chip_label + "</span>")
    farm_html = f"<div class='av-tagline'>{farmer_loc}</div>" if farmer_loc else ""
    logo_html = "<span style='font-size:1.7rem'>🌾</span>"
    try:
        from ..util.brand import logo_data_uri
        logo_html = (f"<img src='{logo_data_uri(192)}' "
                     f"alt='Krishi Scan' style='width:52px;height:52px;"
                     f"border-radius:14px;display:block' />")
    except Exception:
        pass                     # stale offline bundle without util.brand
    st.markdown(
        f"""
        <div class='av-tricolor'></div>
        <div class='av-header'>
          <div class='av-logo'>{logo_html}</div>
          <div>
            <div class='av-brand-name'>Krishi <span>Scan</span></div>
            <div class='av-tagline'>{tagline}</div>
            {farm_html}
          </div>
          <div class='av-status'>{chip}</div>
        </div>
        """,
        unsafe_allow_html=True)


def weather_bar(location: str, conditions: dict, offline: bool = False,
                stale: bool = False, tag: str = "") -> None:
    """
    Small minimal weather strip shown under the header on every page.

    `conditions` keys: temp_c, humidity_pct, rain_mm, condition (a string key
    like 'condition_clear').  `location` is the place label (may be empty).
    `offline` -> no live network; `stale` -> value is a cached snapshot;
    `tag` -> optional right-aligned pill text (e.g. translated 'cached'/'offline').
    """
    import streamlit as st
    from agrovision.lang import get_translator
    tr = get_translator(st.session_state.get("lang", "hi"))

    if offline and not conditions:
        st.markdown(
            f"<div class='av-weather'><span class='av-wx-ic'>🌦️</span>"
            f"<span class='av-wx-val'>{tr.t('weather.offline_note')}</span>"
            f"<span class='av-wx-tag'>{tag or tr.t('offline.offline')}</span></div>",
            unsafe_allow_html=True)
        return

    icon = {"condition_clear": "☀️", "condition_cloudy": "☁️",
            "condition_rainy": "🌧️", "condition_unknown": "🌦️"}.get(
        conditions.get("condition", ""), "🌦️")
    loc_html = f"<span class='av-wx-loc'>📍 {location}</span>" if location else ""
    tag_html = (f"<span class='av-wx-tag'>{tag}</span>" if tag else "")
    st.markdown(
        f"<div class='av-weather'><span class='av-wx-ic'>{icon}</span>"
        f"{loc_html}"
        f"<span class='av-wx-val'>{conditions.get('temp_c', 0):.1f}°C</span>"
        f"<span class='av-wx-val'>💧{conditions.get('humidity_pct', 0):.0f}%</span>"
        f"<span class='av-wx-val'>🌧️{conditions.get('rain_mm', 0):.1f}mm</span>"
        f"{tag_html}</div>",
        unsafe_allow_html=True)


def nav_bar(items: list) -> None:
    """
    Slim single-row tab bar rendered as a native segmented control, so it
    stays one line tall on phones and matches a proper app nav bar.
    `items`: list of dicts {key, label, icon, active}.

    The widget is keyed `nav_tabs`.  On a genuine user tap the on_change callback
    moves `st.session_state.page`; programmatic navigation must set BOTH `page`
    and `nav_tabs` (see home.py) so the two stay in step.
    """
    import streamlit as st
    from agrovision.ui.state import set

    def _on_nav() -> None:
        chosen = st.session_state.get("nav_tabs")
        if chosen and chosen != st.session_state.get("page"):
            set("page", chosen)

    # Never pre-set the widget value here: doing so would discard the frontend's
    # freshly-tapped selection before on_change reads it.  Programmatic page
    # changes are handled by callers keeping page and nav_tabs in sync.
    st.segmented_control(
        "nav",
        [item["key"] for item in items],
        format_func=lambda key: next(
            (f"{i['icon']} {i['label']}" for i in items if i["key"] == key), key),
        label_visibility="collapsed",
        key="nav_tabs",
        on_change=_on_nav,
    )


def language_switcher(languages: dict) -> None:
    """
    Compact language dropdown (one line on phones instead of stacked pills).
    `languages`: {code: display_name} in the order to show.
    """
    import streamlit as st
    from agrovision.ui.state import set

    current = st.session_state.get("lang", "hi")
    choices = list(languages.items())            # [(code, display_name)]
    index = next((i for i, (code, _) in enumerate(choices) if code == current), 0)
    chosen = st.selectbox(
        "Language",
        choices,
        index=index,
        format_func=lambda item: item[1],
        label_visibility="collapsed",
        key="lang_select",
    )
    if chosen and chosen[0] != current:
        set("lang", chosen[0])
        st.rerun()
