"""
Central configuration for AgroVision.

Every tunable knob (gatekeeper thresholds, model paths, language defaults,
weather provider, retry limits) lives here so modules never hardcode values.
Values are overridable via environment variables for deployment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "agrovision" / "assets"
MODEL_DIR = ASSETS_DIR / "model"
LANG_DIR = BASE_DIR / "agrovision" / "lang"
RECO_DIR = ASSETS_DIR / "recommendations"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Language / UI
# ---------------------------------------------------------------------------
@dataclass
class LanguageConfig:
    default_language: str = os.environ.get("AGRO_LANGUAGE", "hi")   # hi | en | ta
    supported: tuple = ("hi", "en", "ta")
    # Missing key fallback chain: requested -> fallbacks -> en.  Never a raw key.
    fallback_chain: tuple = ("en",)           # after the requested language
    tts_enabled: bool = os.environ.get("AGRO_TTS", "1") != "0"


LANG = LanguageConfig()

# ---------------------------------------------------------------------------
# CV Gatekeeper thresholds (tuning starting points per project spec)
# ---------------------------------------------------------------------------
@dataclass
class GatekeeperConfig:
    blur_min_variance: float = _env_float("AGRO_BLUR_MIN_VAR", 100.0)
    brightness_min: float = _env_float("AGRO_BRIGHT_MIN", 40.0)
    brightness_max: float = _env_float("AGRO_BRIGHT_MAX", 220.0)
    leaf_min_coverage: float = _env_float("AGRO_LEAF_MIN_COVER", 0.15)   # fraction 0..1
    glare_max_ratio: float = _env_float("AGRO_GLARE_MAX_RATIO", 0.05)    # fraction 0..1
    glare_brightness: int = 240                                          # pixels above this = blown out
    max_retakes: int = int(os.environ.get("AGRO_MAX_RETAKES", 3))
    preview_width: int = 480             # resize before checks: cheaper + consistent


GK = GatekeeperConfig()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    # Path to a fine-tuned checkpoint.  Leave unset to run in demo/baseline mode.
    checkpoint_path: str = os.environ.get("AGRO_CHECKPOINT", "")
    # Pre-trained baseline fallback (MobileNetV2 ImageNet) used only in demo mode.
    baseline_weights: str = os.environ.get("AGRO_BASELINE", "imagenet")
    classes_file: Path = MODEL_DIR / "classes.json"
    onnx_model: Path = MODEL_DIR / "mobilenetv2_quant.onnx"
    onnx_cam_model: Path = MODEL_DIR / "mobilenetv2_cam.onnx"
    input_size: int = 224
    top_k: int = 3
    demo_mode: bool = _env_bool("AGRO_DEMO", False)
    # Offline (Pyodide) inference engine: "onnx".  Online engine: "tensorflow".
    offline_engine: str = "onnx"


INF = InferenceConfig()

# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
@dataclass
class WeatherConfig:
    provider: str = os.environ.get("AGRO_WEATHER_PROVIDER", "open-meteo")
    base_url: str = os.environ.get("AGRO_WEATHER_BASE",
                                   "https://api.open-meteo.com/v1/forecast")
    geocode_url: str = os.environ.get("AGRO_WEATHER_GEOCODE",
                                      "https://geocoding-api.open-meteo.com/v1/search")
    timeout_seconds: int = int(os.environ.get("AGRO_WEATHER_TIMEOUT", 8))
    history_days: int = int(os.environ.get("AGRO_WEATHER_HISTORY", 3))
    api_key: str = os.environ.get("AGRO_WEATHER_API_KEY", "")   # required for some providers


WX = WeatherConfig()

# ---------------------------------------------------------------------------
# Database / sync
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    # Postgres on the hosted backend; SQLite file for local dev / tests.
    database_url: str = os.environ.get(
        "AGRO_DB_URL", f"sqlite:///{BASE_DIR / 'agrovision.db'}")
    offline_journal: Path = BASE_DIR / "agrovision" / "data" / "offline_journal.json"
    auto_sync: bool = True


DATA = DataConfig()
