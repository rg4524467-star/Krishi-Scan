"""
Weather module - OPTIONAL input feeding the decision-layer fusion.

Fetches current conditions + the last few days of history via an external API
(Open-Meteo default: free, no key; provider/URL configurable).  Fully skippable:
`fetch()` returning None is a first-class result, and every network failure
raises WeatherError which the UI catches as "continue on photo only" - the
diagnosis pipeline never blocks on weather.

The history window matters for context and the audit log: we average daily
max-temp and mean humidity across the window so the report shows what the
recent days were like, while the fusion layer uses the CURRENT live readings
(the most accurate, directly verifiable numbers).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests

from ..config import WX


class WeatherError(RuntimeError):
    pass


@dataclass
class WeatherData:
    temp_c: float
    humidity_pct: float
    condition: str                 # weather-code -> human key ('condition_clear' etc.)
    rain_mm: float
    history_avg_temp_c: float
    history_avg_humidity_pct: float

    def as_json(self) -> str:
        import json
        return json.dumps({
            "temp_c": self.temp_c, "humidity_pct": self.humidity_pct,
            "condition": self.condition, "rain_mm": self.rain_mm,
            "history_avg_temp_c": self.history_avg_temp_c,
            "history_avg_humidity_pct": self.history_avg_humidity_pct,
        })

    def fusion_conditions(self) -> Dict[str, float]:
        """Mapping into the fusion scorer's accepted dimension names.

        Uses the CURRENT live readings (most accurate, directly checkable by
        the farmer) rather than the rolling-window averages.  The history
        averages are still shown for context in the UI and audit log.
        """
        return {"temp_c": self.temp_c, "humidity_pct": self.humidity_pct}


_WMO = {
    0: "condition_clear", 1: "condition_clear", 2: "condition_cloudy", 3: "condition_cloudy",
    45: "condition_cloudy", 48: "condition_cloudy",
    51: "condition_rainy", 53: "condition_rainy", 55: "condition_rainy",
    61: "condition_rainy", 63: "condition_rainy", 65: "condition_rainy",
    80: "condition_rainy", 81: "condition_rainy", 82: "condition_rainy",
    95: "condition_rainy",
}


def _condition_key(code: Optional[int]) -> str:
    return _WMO.get(code, "condition_unknown")


class WeatherClient:
    def __init__(self, base_url: str = None, geocode_url: str = None, timeout: int = None,
                 history_days: int = None, provider: str = None):
        self.base_url = base_url or WX.base_url
        self.geocode_url = geocode_url or WX.geocode_url
        self.timeout = timeout or WX.timeout_seconds
        self.history_days = history_days or WX.history_days
        self.provider = provider or WX.provider

    def fetch(self, lat: float, lon: float) -> WeatherData:
        if self.provider == "open-meteo":
            return self._fetch_openmeteo(lat, lon)
        return self._fetch_openmeteo(lat, lon)   # default provider

    def _fetch_openmeteo(self, lat: float, lon: float) -> WeatherData:
        now = datetime.now()
        start = (now - timedelta(days=self.history_days)).strftime("%Y-%m-%d")
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,rain,weather_code",
            "daily": "temperature_2m_max,relative_humidity_2m_mean",
            "timezone": "auto",
            "start_date": start, "end_date": now.strftime("%Y-%m-%d"),
        }
        try:
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise WeatherError(f"weather fetch failed: {exc}") from exc

        cur = data.get("current", {})
        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_max") or []
        hums = daily.get("relative_humidity_2m_mean") or []

        return WeatherData(
            temp_c=float(cur.get("temperature_2m", 25.0)),
            humidity_pct=float(cur.get("relative_humidity_2m", 60.0)),
            condition=_condition_key(cur.get("weather_code")),
            rain_mm=float(cur.get("rain", 0.0)),
            history_avg_temp_c=_avg(temps, 25.0),
            history_avg_humidity_pct=_avg(hums, 60.0),
        )


def _avg(values: list, default: float) -> float:
    if not values:
        return default
    return sum(float(v) for v in values) / len(values)


def geocode_place(place: str) -> Optional[tuple]:
    """Forward geocode a place name to (lat, lon).  None when unresolvable."""
    try:
        resp = requests.get(WX.geocode_url, params={"name": place, "count": 1},
                            timeout=WX.timeout_seconds)
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        r = results[0]
        return float(r["latitude"]), float(r["longitude"])
    except (requests.RequestException, KeyError, ValueError):
        return None


_IP_GEO_PROVIDERS = (
    ("ipwho", "https://ipwho.is/"),
    ("ip-api", "http://ip-api.com/json/?fields=status,lat,lon,city,regionName&lang=en"),
)


def locate_by_ip() -> Optional[tuple]:
    """
    Best-effort server-side location from the client's public IP.
    Returns (lat, lon, label) or None.  No browser permission needed, so it
    works even where iframe-based geolocation is blocked.  Tries providers in
    order; every failure degrades to None (caller falls back to manual entry).
    """
    for provider, url in _IP_GEO_PROVIDERS:
        try:
            resp = requests.get(url, timeout=WX.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            if provider == "ipwho":
                lat, lon = data.get("latitude"), data.get("longitude")
                if lat is None or lon is None:
                    continue
                parts = [p for p in (data.get("city"), data.get("region")) if p]
                label = ", ".join(parts) if parts else f"{lat:.2f}, {lon:.2f}"
                return float(lat), float(lon), label
            if provider == "ip-api" and data.get("status") == "success":
                parts = [p for p in (data.get("city"), data.get("regionName")) if p]
                label = ", ".join(parts) if parts else ""
                return float(data["lat"]), float(data["lon"]), label
        except (requests.RequestException, KeyError, ValueError, TypeError):
            continue
    return None