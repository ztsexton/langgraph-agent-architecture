"""Weather tool using Open-Meteo.

No API key required.
- Geocoding: https://geocoding-api.open-meteo.com/
- Forecast:   https://api.open-meteo.com/

Returns structured data suitable for rendering in the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import re


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str


def _parse_iso(dt: str) -> datetime:
    # Open-Meteo returns ISO like 2026-01-18T13:00
    return datetime.fromisoformat(dt)


def geocode_location(query: str) -> Optional[Location]:
    raw = (query or "").strip()
    if not raw:
        return None

    # Open-Meteo geocoding is surprisingly picky about qualifiers like
    # "Seattle, WA". Try a small set of progressively simpler variants.
    candidates: list[str] = [raw]

    # Prefer the substring before the first comma.
    if "," in raw:
        before_comma = raw.split(",", 1)[0].strip()
        if before_comma and before_comma not in candidates:
            candidates.append(before_comma)

    # If the query ends with a 2-letter region code (e.g. "Seattle WA"),
    # try removing it.
    m = re.search(r"\s+([A-Za-z]{2})\s*$", raw)
    if m:
        shortened = raw[: m.start(1)].strip()
        # Avoid chopping multi-word cities down to one token.
        if shortened and shortened not in candidates:
            candidates.append(shortened)
        if "," in shortened:
            before_comma = shortened.split(",", 1)[0].strip()
            if before_comma and before_comma not in candidates:
                candidates.append(before_comma)

    # Very common suffixes.
    for suffix in (" usa", " us"):
        if raw.lower().endswith(suffix):
            shortened = raw[: -len(suffix)].strip()
            if shortened and shortened not in candidates:
                candidates.append(shortened)

    # If the user gave a US state abbreviation, bias results toward the US.
    # (Open-Meteo supports country_code filtering.)
    country_code: Optional[str] = None
    if re.search(r"(,\s*[A-Za-z]{2}\s*$)|(\s+[A-Za-z]{2}\s*$)", raw):
        country_code = "US"

    base_params = {
        "count": 1,
        "language": "en",
        "format": "json",
    }
    if country_code:
        base_params["country_code"] = country_code

    for name in candidates:
        params = {**base_params, "name": name}
        r = httpx.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=10.0)
        r.raise_for_status()
        data = r.json() or {}
        results = data.get("results") or []
        if not results:
            continue
        first = results[0]
        return Location(
            name=str(first.get("name") or name),
            latitude=float(first["latitude"]),
            longitude=float(first["longitude"]),
            timezone=str(first.get("timezone") or "auto"),
        )

    return None


def fetch_forecast(location: Location) -> dict[str, Any]:
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current_weather": "true",
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation_probability",
                "precipitation",
                "weathercode",
                "windspeed_10m",
            ]
        ),
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
                "weathercode",
            ]
        ),
        "timezone": "auto",
    }
    r = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10.0)
    r.raise_for_status()
    data = r.json() or {}
    return {
        "location": {
            "name": location.name,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": data.get("timezone") or location.timezone,
        },
        "current": data.get("current_weather") or {},
        "hourly_units": data.get("hourly_units") or {},
        "hourly": data.get("hourly") or {},
        "daily_units": data.get("daily_units") or {},
        "daily": data.get("daily") or {},
    }


def build_today_summary(forecast: dict[str, Any]) -> dict[str, Any]:
    loc = forecast.get("location") or {}
    current = forecast.get("current") or {}

    daily = forecast.get("daily") or {}
    dates = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    pmax = daily.get("precipitation_probability_max") or []

    today = {
        "location": loc,
        "as_of": current.get("time"),
        "temp_c": current.get("temperature"),
        "wind_kph": current.get("windspeed"),
        "today_high_c": tmax[0] if len(tmax) > 0 else None,
        "today_low_c": tmin[0] if len(tmin) > 0 else None,
        "precip_chance_pct": pmax[0] if len(pmax) > 0 else None,
        "date": dates[0] if len(dates) > 0 else None,
    }
    return today


def build_hourly_rows(forecast: dict[str, Any], *, hours: int = 12) -> list[dict[str, Any]]:
    hourly = forecast.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    pop = hourly.get("precipitation_probability") or []
    wind = hourly.get("windspeed_10m") or []
    precip = hourly.get("precipitation") or []

    # Try to start around 'now' when possible
    current_time = (forecast.get("current") or {}).get("time")
    start_idx = 0
    if isinstance(current_time, str):
        try:
            start_idx = times.index(current_time)
        except Exception:
            start_idx = 0

    rows: list[dict[str, Any]] = []
    for i in range(start_idx, min(start_idx + hours, len(times))):
        rows.append(
            {
                "time": times[i],
                "temp_c": temps[i] if i < len(temps) else None,
                "precip_chance_pct": pop[i] if i < len(pop) else None,
                "precip_mm": precip[i] if i < len(precip) else None,
                "wind_kph": wind[i] if i < len(wind) else None,
            }
        )
    return rows


def build_daily_rows(forecast: dict[str, Any], *, days: int = 7) -> list[dict[str, Any]]:
    daily = forecast.get("daily") or {}
    dates = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    pmax = daily.get("precipitation_probability_max") or []

    out: list[dict[str, Any]] = []
    for i in range(0, min(days, len(dates))):
        out.append(
            {
                "date": dates[i],
                "high_c": tmax[i] if i < len(tmax) else None,
                "low_c": tmin[i] if i < len(tmin) else None,
                "precip_chance_pct": pmax[i] if i < len(pmax) else None,
            }
        )
    return out
