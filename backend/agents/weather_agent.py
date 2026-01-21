from __future__ import annotations

import re
from typing import Any, Optional

from ..tools.agent_config import get_agent_settings
from ..tools.langfuse_tracing import end_span, start_span
from ..tools.llm import ask_llm, get_llm
from ..tools.weather import (
    build_daily_rows,
    build_hourly_rows,
    build_today_summary,
    fetch_forecast,
    geocode_location,
)

from .types import AgentState
from .ui import (
    a2ui_weather_card,
    extract_json_object,
    fmt_date_day,
    fmt_num,
    fmt_temp_c_f,
)


def _choose_weather_view(user_text: str) -> tuple[str, int]:
    t = (user_text or "").lower()

    m = re.search(r"\b(\d{1,2})\s*-?\s*day\b", t)
    if m:
        try:
            n = int(m.group(1))
            if 3 <= n <= 10:
                return "daily", n
        except Exception:
            pass

    if "hour" in t or "hourly" in t or "next 12" in t or "next 24" in t:
        return "hourly", 0
    if "10 day" in t or "10-day" in t or "ten day" in t:
        return "daily", 10
    if "7 day" in t or "7-day" in t or "week" in t or "weekly" in t:
        return "daily", 7
    if "3 day" in t or "3-day" in t or "three day" in t:
        return "daily", 3
    if "tomorrow" in t or "next few days" in t or "forecast" in t:
        return "daily", 7
    return "today", 0


def _extract_location_guess(user_text: str) -> Optional[str]:
    if not user_text:
        return None
    m = re.search(r"\b(?:in|for)\s+([^\n\r\?\,]+)", user_text, flags=re.IGNORECASE)
    if not m:
        return None
    loc = m.group(1).strip()
    loc = re.sub(
        r"\b(today|tomorrow|this week|next week|hourly|forecast|next \d+ hours?)\b.*$",
        "",
        loc,
        flags=re.IGNORECASE,
    ).strip()
    return loc or None


def weather_agent(state: AgentState) -> AgentState:
    """Answer weather questions with structured weather cards (A2UI)."""
    _span = start_span(name="agent:weather_agent", input={"state": state}, metadata={"kind": "agent"})
    query = state.get("input", "")
    weather_settings = get_agent_settings("weather_agent")

    view, days = _choose_weather_view(query)
    location_text = _extract_location_guess(query)

    llm = get_llm(weather_settings.model_name)
    if llm is not None:
        try:
            intent_prompt = (
                "Extract weather intent from the user message. Return ONLY valid JSON with keys: "
                "location (string or null), view (one of today|hourly|daily), days (integer 3-10 or null).\n\n"
                f"User message: {query}"
            )
            intent_raw = ask_llm(
                intent_prompt,
                model_name=weather_settings.model_name,
                system_prompt=weather_settings.system_prompt,
            )
            intent = extract_json_object(intent_raw) or {}
            if isinstance(intent.get("location"), str) and intent["location"].strip():
                location_text = intent["location"].strip()
            if intent.get("view") in ("today", "hourly", "daily"):
                view = intent["view"]
            if isinstance(intent.get("days"), int):
                days = max(3, min(10, int(intent["days"])))
        except Exception:
            pass

    if not location_text:
        text = "Which location should I use (e.g., 'Seattle, WA')? You can also say 'hourly' or '7-day'."
        out = {
            "output": text,
            "a2ui": a2ui_weather_card(title="Weather", subtitle="Location needed", kv={"Next": text}),
        }
        end_span(_span, output=out)
        return out

    try:
        loc = geocode_location(location_text)
    except Exception:
        loc = None

    if loc is None:
        text = (
            f"I couldn't find a location matching '{location_text}'. "
            "Try a more specific place name (e.g., 'Austin, TX')."
        )
        out = {
            "output": text,
            "a2ui": a2ui_weather_card(title="Weather", subtitle="Location not found", kv={"Error": text}),
        }
        end_span(_span, output=out)
        return out

    try:
        forecast = fetch_forecast(loc)
    except Exception:
        text = "Weather service is temporarily unavailable. Please try again."
        out = {
            "output": text,
            "a2ui": a2ui_weather_card(title="Weather", subtitle=loc.name, kv={"Error": text}),
        }
        end_span(_span, output=out)
        return out

    if view == "hourly":
        hourly = build_hourly_rows(forecast, hours=12)
        today = build_today_summary(forecast)
        subtitle = f"{forecast['location']['name']} • Next 12 hours"
        narrative = (
            f"Next 12 hours in {forecast['location']['name']}\n"
            f"Now: {fmt_temp_c_f(today.get('temp_c'))} • "
            f"Precip: {fmt_num(today.get('precip_chance_pct'), '%')} • "
            f"Wind: {fmt_num(today.get('wind_kph'), ' km/h')}"
        )

        if llm is not None:
            try:
                llm_text = ask_llm(
                    "Write 1-2 short sentences summarizing this hourly weather forecast for a user. "
                    "Do not add facts not present.\n\n"
                    f"Location: {forecast['location']['name']}\n"
                    f"Hourly rows: {hourly}",
                    model_name=weather_settings.model_name,
                    system_prompt=weather_settings.system_prompt,
                )
                if isinstance(llm_text, str) and llm_text.strip():
                    narrative = llm_text.strip()
            except Exception:
                pass

        kv = {
            "As of": today.get("as_of") or "",
            "Now": fmt_temp_c_f(today.get("temp_c")),
            "High / Low": f"{fmt_temp_c_f(today.get('today_high_c'))} / {fmt_temp_c_f(today.get('today_low_c'))}",
            "Precip chance": fmt_num(today.get("precip_chance_pct"), "%"),
            "Wind": fmt_num(today.get("wind_kph"), " km/h"),
        }
        table = {
            "type": "table",
            "title": "Hourly",
            "columns": ["Time", "Temp", "Precip", "Precip (mm)", "Wind (km/h)"],
            "rows": [
                [
                    r.get("time"),
                    fmt_temp_c_f(r.get("temp_c")),
                    fmt_num(r.get("precip_chance_pct"), "%"),
                    fmt_num(r.get("precip_mm"), ""),
                    fmt_num(r.get("wind_kph"), ""),
                ]
                for r in hourly
            ],
        }
        out = {
            "output": narrative,
            "a2ui": a2ui_weather_card(title="Weather", subtitle=subtitle, kv=kv, table=table),
        }
        end_span(_span, output=out)
        return out

    if view == "daily":
        if days <= 0:
            days = 7
        daily = build_daily_rows(forecast, days=days)
        subtitle = f"{forecast['location']['name']} • {days}-day forecast"

        preview_lines = []
        for r in daily[:3]:
            preview_lines.append(
                f"- {fmt_date_day(r.get('date'))}: {fmt_temp_c_f(r.get('high_c'))} / {fmt_temp_c_f(r.get('low_c'))} • precip {fmt_num(r.get('precip_chance_pct'), '%')}"
            )
        narrative = (
            f"{days}-day forecast for {forecast['location']['name']}\n"
            + ("Next 3 days:\n" + "\n".join(preview_lines) if preview_lines else "")
        )

        if llm is not None:
            try:
                daily_for_llm = [
                    {
                        "date": r.get("date"),
                        "high": fmt_temp_c_f(r.get("high_c")),
                        "low": fmt_temp_c_f(r.get("low_c")),
                        "precip_chance_pct": r.get("precip_chance_pct"),
                    }
                    for r in daily
                ]
                llm_text = ask_llm(
                    "Write a short, human-friendly weather summary based ONLY on the data provided. "
                    "2-4 sentences max. Use plain language.\n\n"
                    "Must include:\n"
                    "- overall precip/rain risk (based on precip %)\n"
                    "- warming/cooling trend if obvious\n"
                    "- warmest day high and coldest night low (with the day)\n\n"
                    "Do NOT invent conditions (no clouds/sun, no storms, no snow) unless implied by the data (it isn't). "
                    "Temperatures are already formatted as strings (F with C in parentheses).\n\n"
                    f"Location: {forecast['location']['name']}\n"
                    f"Days: {daily_for_llm}",
                    model_name=weather_settings.model_name,
                    system_prompt=weather_settings.system_prompt,
                )
                if isinstance(llm_text, str) and llm_text.strip():
                    narrative = llm_text.strip()
            except Exception:
                pass

        kv = {
            "Timezone": forecast["location"].get("timezone"),
        }
        table = {
            "type": "table",
            "title": "Daily",
            "columns": ["Date", "High", "Low", "Precip"],
            "rows": [
                [
                    r.get("date"),
                    fmt_temp_c_f(r.get("high_c")),
                    fmt_temp_c_f(r.get("low_c")),
                    fmt_num(r.get("precip_chance_pct"), "%"),
                ]
                for r in daily
            ],
        }
        out = {
            "output": narrative,
            "a2ui": a2ui_weather_card(title="Weather", subtitle=subtitle, kv=kv, table=table),
        }
        end_span(_span, output=out)
        return out

    today = build_today_summary(forecast)
    subtitle = f"{forecast['location']['name']} • Today"
    narrative = (
        f"Today in {forecast['location']['name']}\n"
        f"Now: {fmt_temp_c_f(today.get('temp_c'))}\n"
        f"High / Low: {fmt_temp_c_f(today.get('today_high_c'))} / {fmt_temp_c_f(today.get('today_low_c'))}\n"
        f"Precip: {fmt_num(today.get('precip_chance_pct'), '%')} • Wind: {fmt_num(today.get('wind_kph'), ' km/h')}"
    )

    if llm is not None:
        try:
            llm_text = ask_llm(
                "Write 1-2 short sentences summarizing today's weather from these values. "
                "Do not add facts not present.\n\n"
                f"Location: {forecast['location']['name']}\n"
                f"Values: {today}",
                model_name=weather_settings.model_name,
                system_prompt=weather_settings.system_prompt,
            )
            if isinstance(llm_text, str) and llm_text.strip():
                narrative = llm_text.strip()
        except Exception:
            pass

    kv = {
        "As of": today.get("as_of") or "",
        "Now": fmt_temp_c_f(today.get("temp_c")),
        "High": fmt_temp_c_f(today.get("today_high_c")),
        "Low": fmt_temp_c_f(today.get("today_low_c")),
        "Precip chance": fmt_num(today.get("precip_chance_pct"), "%"),
        "Wind": fmt_num(today.get("wind_kph"), " km/h"),
    }
    out = {
        "output": narrative,
        "a2ui": a2ui_weather_card(title="Weather", subtitle=subtitle, kv=kv),
    }
    end_span(_span, output=out)
    return out
