"""Build and expose a LangGraph‑based multi‑agent architecture.

This module constructs an agent graph consisting of a supervisor node and
three specialised worker nodes: ``web_agent`` for performing web searches,
``meetings_agent`` for interacting with an in‑memory calendar, and
``rag_agent`` for answering questions from a small document corpus.

When a language model is configured (e.g. Azure OpenAI), the supervisor will
prefer an LLM-based router to choose the appropriate worker. If no model is
configured, routing falls back to simple keyword matching.

To compile the graph call :func:`get_agent_graph()` which returns a compiled
``StateGraph`` ready for invocation or streaming.
"""

from __future__ import annotations

import json
import re
from typing import TypedDict, Dict, Literal, Optional, Any

from langgraph.graph import StateGraph, START, END

# Import tool functions from the dedicated tools package.  These provide
# clean interfaces for each domain and encapsulate shared state.  Note that
# the meetings manager and rag search are initialised within their modules.
from .tools.web import search_web
from .tools.meetings import (
    list_meetings,
    create_meeting,
    edit_meeting_agenda,
    edit_meeting_notes,
)
from .tools.rag import answer_question
from .tools.llm import ask_llm, get_llm
from .tools.agent_config import get_agent_settings
from .tools.weather import (
    geocode_location,
    fetch_forecast,
    build_today_summary,
    build_hourly_rows,
    build_daily_rows,
)

from .tools.langfuse_tracing import start_span, end_span


class AgentState(TypedDict, total=False):
    """Schema for the graph’s state.

    The ``input`` key contains the raw user message.  The ``output`` key
    holds the response returned by whichever worker agent processed the
    request.  Additional keys could be added here (e.g. conversation
    history) if you extend the system to support multi‑turn interactions.
    """

    input: str
    output: str
    # Optional structured UI payload ("a2ui"-style schema)
    a2ui: Dict[str, Any]


def _a2ui_text(title: str, text: str, *, sources: Optional[list[str]] = None) -> Dict[str, Any]:
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


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not text:
        return None
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # Fallback: take first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _choose_weather_view(user_text: str) -> tuple[str, int]:
    """Return (view, days) where view in {today, hourly, daily} and days in [3,10]."""
    t = (user_text or "").lower()

    # Explicit N-day intent
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


def _a2ui_weather_card(
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


def _c_to_f(c: Any) -> Optional[float]:
    try:
        if c is None:
            return None
        return round((float(c) * 9.0 / 5.0) + 32.0, 1)
    except Exception:
        return None


def _fmt_temp_c_f(c: Any) -> str:
    if c is None:
        return ""
    f = _c_to_f(c)
    try:
        c_num = round(float(c), 1)
        if f is None:
            return f"{c_num}°C"
        # Prefer Fahrenheit first for readability, keep Celsius in parentheses.
        return f"{f}°F ({c_num}°C)"
    except Exception:
        return str(c)


def _fmt_date_day(date_str: Any) -> str:
    if not isinstance(date_str, str) or not date_str:
        return ""
    try:
        dt = __import__("datetime").datetime.fromisoformat(date_str)
        # Portable day formatting (no %-d on Windows).
        return dt.strftime("%a %b %d").replace(" 0", " ")
    except Exception:
        return date_str


def _fmt_num(x: Any, suffix: str = "") -> str:
    if x is None:
        return ""
    try:
        v = float(x)
        if v.is_integer():
            return f"{int(v)}{suffix}"
        return f"{round(v, 1)}{suffix}"
    except Exception:
        return f"{x}{suffix}"


# Shared resources are managed by the tools modules.  There is no need to
# instantiate any state here.


def supervisor(state: AgentState) -> AgentState:
    """Pass through the state unmodified.

    The supervisor node itself does not update state; it only decides which
    worker node should run next via the routing function defined below.
    """
    return {}


def route(state: AgentState) -> Literal["web_agent", "meetings_agent", "rag_agent", "weather_agent"]:
    """Determine which worker agent should handle the current request.

    This simple router uses keyword matching on the lowercase input to
    choose an agent.  A more sophisticated implementation might use an
    LLM or structured schema to decide which domain to route to.
    """
    user_text = state.get("input", "")

    route_span = start_span(
        name="agent:route",
        input={"input": user_text},
        metadata={"kind": "routing"},
    )

    # Prefer LLM routing when available.
    supervisor_settings = get_agent_settings("supervisor")
    llm = get_llm(supervisor_settings.model_name)
    if llm is not None:
        router_prompt = (
            "Choose the single best specialist agent for the user's request. "
            "Return ONLY one of these exact tokens: web_agent, meetings_agent, rag_agent, weather_agent.\n\n"
            "Routing guidance:\n"
            "- meetings_agent: scheduling, meetings, agendas, notes, calendars\n"
            "- rag_agent: questions that explicitly mention documents, citations, or internal references\n"
            "- weather_agent: weather, forecast, temperature, rain, snow, wind, air quality\n"
            "- web_agent: anything that needs web search/current info or general Q&A\n\n"
            f"User message: {user_text}"
        )
        try:
            decision = ask_llm(
                router_prompt,
                model_name=supervisor_settings.model_name,
                system_prompt=supervisor_settings.system_prompt,
            )
            token = (decision or "").strip().split()[0].strip().lower()
            if token in ("web_agent", "meetings_agent", "rag_agent", "weather_agent"):
                end_span(route_span, output={"decision": token, "mode": "llm"})
                return token  # type: ignore[return-value]
        except Exception:
            # Fall back to keyword matching.
            pass

    # Fallback keyword router (non-LLM).
    text = user_text.lower()
    meeting_keywords = ["meeting", "schedule", "agenda", "notes", "calendar"]
    rag_keywords = ["document", "doc", "citation", "reference"]
    weather_keywords = ["weather", "forecast", "temperature", "rain", "snow", "wind", "humidity", "aqi", "air quality"]
    search_keywords = ["search", "web", "internet", "look up"]
    if any(k in text for k in weather_keywords):
        end_span(route_span, output={"decision": "weather_agent", "mode": "keyword"})
        return "weather_agent"
    if any(k in text for k in meeting_keywords):
        end_span(route_span, output={"decision": "meetings_agent", "mode": "keyword"})
        return "meetings_agent"
    if any(k in text for k in rag_keywords):
        end_span(route_span, output={"decision": "rag_agent", "mode": "keyword"})
        return "rag_agent"
    if any(k in text for k in search_keywords):
        end_span(route_span, output={"decision": "web_agent", "mode": "keyword"})
        return "web_agent"
    end_span(route_span, output={"decision": "web_agent", "mode": "default"})
    return "web_agent"


def weather_agent(state: AgentState) -> AgentState:
    """Answer weather questions with structured weather cards (A2UI)."""
    _span = start_span(name="agent:weather_agent", input={"state": state}, metadata={"kind": "agent"})
    query = state.get("input", "")
    weather_settings = get_agent_settings("weather_agent")

    view, days = _choose_weather_view(query)
    location_text = _extract_location_guess(query)

    # If we have an LLM configured, let it refine intent/location.
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
            intent = _extract_json_object(intent_raw) or {}
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
        return {
            "output": text,
            "a2ui": _a2ui_weather_card(title="Weather", subtitle="Location needed", kv={"Next": text}),
        }

    try:
        loc = geocode_location(location_text)
    except Exception:
        loc = None

    if loc is None:
        text = f"I couldn't find a location matching '{location_text}'. Try a more specific place name (e.g., 'Austin, TX')."
        return {
            "output": text,
            "a2ui": _a2ui_weather_card(title="Weather", subtitle="Location not found", kv={"Error": text}),
        }

    try:
        forecast = fetch_forecast(loc)
    except Exception:
        text = "Weather service is temporarily unavailable. Please try again."
        return {
            "output": text,
            "a2ui": _a2ui_weather_card(title="Weather", subtitle=loc.name, kv={"Error": text}),
        }

    if view == "hourly":
        hourly = build_hourly_rows(forecast, hours=12)
        today = build_today_summary(forecast)
        subtitle = f"{forecast['location']['name']} • Next 12 hours"
        narrative = (
            f"Next 12 hours in {forecast['location']['name']}\n"
            f"Now: {_fmt_temp_c_f(today.get('temp_c'))} • "
            f"Precip: {_fmt_num(today.get('precip_chance_pct'), '%')} • "
            f"Wind: {_fmt_num(today.get('wind_kph'), ' km/h')}"
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
            "Now": _fmt_temp_c_f(today.get("temp_c")),
            "High / Low": f"{_fmt_temp_c_f(today.get('today_high_c'))} / {_fmt_temp_c_f(today.get('today_low_c'))}",
            "Precip chance": _fmt_num(today.get("precip_chance_pct"), "%"),
            "Wind": _fmt_num(today.get("wind_kph"), " km/h"),
        }
        table = {
            "type": "table",
            "title": "Hourly",
            "columns": ["Time", "Temp", "Precip", "Precip (mm)", "Wind (km/h)"],
            "rows": [
                [
                    r.get("time"),
                    _fmt_temp_c_f(r.get("temp_c")),
                    _fmt_num(r.get("precip_chance_pct"), "%"),
                    _fmt_num(r.get("precip_mm"), ""),
                    _fmt_num(r.get("wind_kph"), ""),
                ]
                for r in hourly
            ],
        }
        return {
            "output": narrative,
            "a2ui": _a2ui_weather_card(title="Weather", subtitle=subtitle, kv=kv, table=table),
        }

    if view == "daily":
        if days <= 0:
            days = 7
        daily = build_daily_rows(forecast, days=days)
        subtitle = f"{forecast['location']['name']} • {days}-day forecast"
        # Simple non-LLM narrative: mention next 3 days.
        preview_lines = []
        for r in daily[:3]:
            preview_lines.append(
                f"- {_fmt_date_day(r.get('date'))}: { _fmt_temp_c_f(r.get('high_c')) } / { _fmt_temp_c_f(r.get('low_c')) } • precip { _fmt_num(r.get('precip_chance_pct'), '%') }"
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
                        "high": _fmt_temp_c_f(r.get("high_c")),
                        "low": _fmt_temp_c_f(r.get("low_c")),
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
                    _fmt_temp_c_f(r.get("high_c")),
                    _fmt_temp_c_f(r.get("low_c")),
                    _fmt_num(r.get("precip_chance_pct"), "%"),
                ]
                for r in daily
            ],
        }
        return {
            "output": narrative,
            "a2ui": _a2ui_weather_card(title="Weather", subtitle=subtitle, kv=kv, table=table),
        }

    today = build_today_summary(forecast)
    subtitle = f"{forecast['location']['name']} • Today"
    narrative = (
        f"Today in {forecast['location']['name']}\n"
        f"Now: {_fmt_temp_c_f(today.get('temp_c'))}\n"
        f"High / Low: {_fmt_temp_c_f(today.get('today_high_c'))} / {_fmt_temp_c_f(today.get('today_low_c'))}\n"
        f"Precip: {_fmt_num(today.get('precip_chance_pct'), '%')} • Wind: {_fmt_num(today.get('wind_kph'), ' km/h')}"
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
        "Now": _fmt_temp_c_f(today.get("temp_c")),
        "High": _fmt_temp_c_f(today.get("today_high_c")),
        "Low": _fmt_temp_c_f(today.get("today_low_c")),
        "Precip chance": _fmt_num(today.get("precip_chance_pct"), "%"),
        "Wind": _fmt_num(today.get("wind_kph"), " km/h"),
    }
    out = {
        "output": narrative,
        "a2ui": _a2ui_weather_card(title="Weather", subtitle=subtitle, kv=kv),
    }
    end_span(_span, output=out)
    return out


def web_agent(state: AgentState) -> AgentState:
    """Perform a web search and populate the output key with formatted results.
    """
    """Perform a web search and summarise using an LLM when configured."""
    _span = start_span(name="agent:web_agent", input={"state": state}, metadata={"kind": "agent"})
    query = state.get("input", "")
    results = search_web(query, max_results=3)
    web_settings = get_agent_settings("web_agent")
    # Format the results for display or LLM summarisation
    lines = []
    for idx, res in enumerate(results, start=1):
        title = res.get("title") or "Untitled"
        body = res.get("body") or ""
        href = res.get("href") or ""
        line = f"{idx}. {title} – {body[:120]}"
        if href:
            line += f" ({href})"
        lines.append(line)
    summary = "\n".join(lines)
    # Attempt to use LLM to answer. If search yielded nothing, the LLM can still
    # respond (and should be explicit about uncertainty).
    llm = get_llm(web_settings.model_name)
    if llm:
        if results:
            prompt = (
                "A user asked the following question:\n\n"
                f"{query}\n\n"
                "Here are the top web search results:\n\n"
                f"{summary}\n\n"
                "Answer the user concisely based only on these results. If the results are insufficient, say so."
            )
        else:
            prompt = (
                "A user asked the following question, but web search returned no results. "
                "Answer as best you can from general knowledge, and clearly say that live search returned no results.\n\n"
                f"User question: {query}"
            )
        try:
            answer = ask_llm(
                prompt,
                model_name=web_settings.model_name,
                system_prompt=web_settings.system_prompt,
            )
            out = {"output": answer}
            end_span(_span, output=out)
            return out
        except Exception:
            pass
    # Non-LLM fallback: return raw results or a friendly message.
    if not results:
        out = {"output": "No search results found."}
        end_span(_span, output=out)
        return out
    out = {"output": summary}
    end_span(_span, output=out)
    return out


def meetings_agent(state: AgentState) -> AgentState:
    """Handle simple meeting operations based on the user input.

    The logic here is deliberately naive; it uses keyword patterns to decide
    which operation to perform.  In a real system you would likely parse
    structured commands or rely on an LLM to extract parameters.
    """
    _span = start_span(name="agent:meetings_agent", input={"state": state}, metadata={"kind": "agent"})
    meetings_settings = get_agent_settings("meetings_agent")
    text = state.get("input", "").lower()
    # List meetings
    if "list" in text and "meeting" in text:
        meetings = list_meetings()
        if not meetings:
            text = "There are no meetings scheduled."
            out = {"output": text, "a2ui": _a2ui_text("Meetings", text)}
            end_span(_span, output=out)
            return out
        lines = [
            f"{m.id}. {m.title} on {m.date} – agenda: {m.agenda}; notes: {m.notes or 'None'}"
            for m in meetings
        ]
        raw = "\n".join(lines)
        llm = get_llm(meetings_settings.model_name)
        if llm:
            try:
                return {
                    "output": ask_llm(
                        f"Format these meeting entries for a user:\n\n{raw}",
                        model_name=meetings_settings.model_name,
                        system_prompt=meetings_settings.system_prompt,
                    ),
                    "a2ui": _a2ui_text("Meetings", raw),
                }
            except Exception:
                pass
        out = {"output": raw, "a2ui": _a2ui_text("Meetings", raw)}
        end_span(_span, output=out)
        return out
    # Create a meeting – expect 'create meeting TITLE on DATE agenda AGENDA'
    if "create" in text and "meeting" in text:
        title = "Untitled meeting"
        date = "2026-01-17"
        agenda = ""
        if "agenda" in text:
            parts = text.split("agenda", 1)
            agenda = parts[1].strip()
            text_before_agenda = parts[0]
        else:
            text_before_agenda = text
        tokens = text_before_agenda.split()
        if "on" in tokens:
            idx = tokens.index("on")
            if idx + 1 < len(tokens):
                date = tokens[idx + 1]
        try:
            meeting_idx = tokens.index("meeting")
            if "on" in tokens:
                on_idx = tokens.index("on")
                title_tokens = tokens[meeting_idx + 1 : on_idx]
            else:
                title_tokens = tokens[meeting_idx + 1 :]
            if title_tokens:
                title = " ".join(title_tokens).title()
        except ValueError:
            pass
        meeting = create_meeting(title=title, date=date, agenda=agenda)
        raw = f"Created meeting {meeting.id}: {meeting.title} on {meeting.date}."
        llm = get_llm(meetings_settings.model_name)
        if llm:
            try:
                out = {
                    "output": ask_llm(
                        f"User requested to create a meeting. Result: {raw}",
                        model_name=meetings_settings.model_name,
                        system_prompt=meetings_settings.system_prompt,
                    )
                }
                end_span(_span, output=out)
                return out
            except Exception:
                pass
        out = {"output": raw, "a2ui": _a2ui_text("Meetings", raw)}
        end_span(_span, output=out)
        return out
    # Edit agenda
    if "edit" in text and "agenda" in text:
        tokens = text.split()
        meeting_id = None
        for token in tokens:
            if token.isdigit():
                meeting_id = int(token)
                break
        new_agenda = text.split("agenda", 1)[1].strip()
        if meeting_id is None:
            out = {"output": "Please specify the meeting ID to edit."}
            end_span(_span, output=out)
            return out
        meeting = edit_meeting_agenda(meeting_id, new_agenda)
        if meeting is None:
            out = {"output": f"Meeting {meeting_id} not found."}
            end_span(_span, output=out)
            return out
        raw = f"Updated agenda for meeting {meeting.id}."
        llm = get_llm(meetings_settings.model_name)
        if llm:
            try:
                out = {
                    "output": ask_llm(
                        f"User requested to update a meeting agenda. Result: {raw}",
                        model_name=meetings_settings.model_name,
                        system_prompt=meetings_settings.system_prompt,
                    )
                }
                end_span(_span, output=out)
                return out
            except Exception:
                pass
        out = {"output": raw, "a2ui": _a2ui_text("Meetings", raw)}
        end_span(_span, output=out)
        return out
    # Edit notes
    if "edit" in text and "notes" in text:
        tokens = text.split()
        meeting_id = None
        for token in tokens:
            if token.isdigit():
                meeting_id = int(token)
                break
        new_notes = text.split("notes", 1)[1].strip()
        if meeting_id is None:
            out = {"output": "Please specify the meeting ID to edit."}
            end_span(_span, output=out)
            return out
        meeting = edit_meeting_notes(meeting_id, new_notes)
        if meeting is None:
            out = {"output": f"Meeting {meeting_id} not found."}
            end_span(_span, output=out)
            return out
        raw = f"Updated notes for meeting {meeting.id}."
        llm = get_llm(meetings_settings.model_name)
        if llm:
            try:
                out = {
                    "output": ask_llm(
                        f"User requested to update meeting notes. Result: {raw}",
                        model_name=meetings_settings.model_name,
                        system_prompt=meetings_settings.system_prompt,
                    )
                }
                end_span(_span, output=out)
                return out
            except Exception:
                pass
        out = {"output": raw, "a2ui": _a2ui_text("Meetings", raw)}
        end_span(_span, output=out)
        return out
    # Default response
    out = {
        "output": (
            "I can manage meetings. Try commands like 'list meetings', 'create meeting "
            "Team Sync on 2026-02-20 agenda Discuss progress', 'edit meeting 1 agenda New agenda' or "
            "'edit meeting 1 notes New notes'."
        ),
        "a2ui": _a2ui_text(
            "Meetings",
            "I can manage meetings. Try: list meetings; create meeting Team Sync on 2026-02-20 agenda ...; edit meeting 1 agenda ...; edit meeting 1 notes ...",
        ),
    }
    end_span(_span, output=out)
    return out


def rag_agent(state: AgentState) -> AgentState:
    """Answer a query using retrieval augmented search."""
    _span = start_span(name="agent:rag_agent", input={"state": state}, metadata={"kind": "agent"})
    query = state.get("input", "")
    result = answer_question(query)
    content = result["content"]
    citation = result["citation"]
    text = f"{content} (Citation: {citation})"
    out = {"output": text, "a2ui": _a2ui_text("RAG", text)}
    end_span(_span, output=out)
    return out


def get_agent_graph() -> "StateGraph[AgentState]":
    """Construct and return a compiled StateGraph for the multi‑agent system."""
    graph_builder: StateGraph[AgentState] = StateGraph(AgentState)
    # Register nodes
    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("web_agent", web_agent)
    graph_builder.add_node("meetings_agent", meetings_agent)
    graph_builder.add_node("rag_agent", rag_agent)
    graph_builder.add_node("weather_agent", weather_agent)
    # Define entry point
    graph_builder.add_edge(START, "supervisor")
    # Route to workers based on supervisor decision
    graph_builder.add_conditional_edges(
        "supervisor",
        route,
        {
            "web_agent": "web_agent",
            "meetings_agent": "meetings_agent",
            "rag_agent": "rag_agent",
            "weather_agent": "weather_agent",
        },
    )
    # Terminate after worker finishes by sending to END
    graph_builder.add_edge("web_agent", END)
    graph_builder.add_edge("meetings_agent", END)
    graph_builder.add_edge("rag_agent", END)
    graph_builder.add_edge("weather_agent", END)
    # Compile and return
    return graph_builder.compile()


# Instantiate a compiled graph at import time so that repeated calls reuse the same
# compiled object.  This avoids the overhead of recompiling on every request.
_compiled_agent_graph = get_agent_graph()

__all__ = [
    "AgentState",
    "get_agent_graph",
    "_compiled_agent_graph",
]
