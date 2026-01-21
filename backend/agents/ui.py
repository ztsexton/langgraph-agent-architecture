from __future__ import annotations

import re
from typing import Any, Dict, Optional


def a2ui_text(title: str, text: str, *, sources: Optional[list[str]] = None) -> Dict[str, Any]:
    children: list[Dict[str, Any]] = [
        {"type": "heading", "level": 2, "text": title},
        {"type": "text", "text": text},
    ]
    if sources:
        children.append(
            {
                "type": "links",
                "items": [{"text": href, "href": href} for href in sources if href],
            }
        )
    return {
        "schema": "a2ui",
        "version": "0.1",
        "render": {"type": "container", "children": children},
    }


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not text:
        return None
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            import json

            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            import json

            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def a2ui_weather_card(
    *,
    title: str,
    subtitle: str,
    kv: dict[str, Any],
    intro_text: Optional[str] = None,
    table: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    card_children: list[Dict[str, Any]] = []
    if intro_text:
        card_children.append({"type": "text", "text": intro_text})
    card_children.append({"type": "kv", "items": [{"label": k, "value": v} for k, v in kv.items()]})
    if table:
        card_children.append(table)

    return {
        "schema": "a2ui",
        "version": "0.1",
        "render": {
            "type": "container",
            "children": [
                {"type": "card", "title": title, "subtitle": subtitle, "children": card_children}
            ],
        },
    }


def c_to_f(c: Any) -> Optional[float]:
    try:
        if c is None:
            return None
        return round((float(c) * 9.0 / 5.0) + 32.0, 1)
    except Exception:
        return None


def fmt_temp_c_f(c: Any) -> str:
    if c is None:
        return ""
    f = c_to_f(c)
    try:
        c_num = round(float(c), 1)
        if f is None:
            return f"{c_num}°C"
        return f"{f}°F ({c_num}°C)"
    except Exception:
        return str(c)


def fmt_date_day(date_str: Any) -> str:
    if not isinstance(date_str, str) or not date_str:
        return ""
    try:
        dt = __import__("datetime").datetime.fromisoformat(date_str)
        return dt.strftime("%a %b %d").replace(" 0", " ")
    except Exception:
        return date_str


def fmt_num(x: Any, suffix: str = "") -> str:
    if x is None:
        return ""
    try:
        v = float(x)
        if v.is_integer():
            return f"{int(v)}{suffix}"
        return f"{round(v, 1)}{suffix}"
    except Exception:
        return f"{x}{suffix}"
