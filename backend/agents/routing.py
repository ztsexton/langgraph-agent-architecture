from __future__ import annotations

from typing import Literal

from ..tools.agent_config import get_agent_settings
from ..tools.langfuse_tracing import end_span, start_span
from ..tools.llm import ask_llm, get_llm

from .types import AgentState


def supervisor(state: AgentState) -> AgentState:
    """Supervisor node: pass-through state."""
    return {}


def route(state: AgentState) -> Literal["web_agent", "meetings_agent", "rag_agent", "weather_agent"]:
    """Determine which worker agent should handle the current request."""
    user_text = state.get("input", "")

    route_span = start_span(
        name="agent:route",
        input={"input": user_text},
        metadata={"kind": "routing"},
    )

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
            pass

    text = user_text.lower()
    meeting_keywords = ["meeting", "schedule", "agenda", "notes", "calendar"]
    rag_keywords = ["document", "doc", "citation", "reference"]
    weather_keywords = [
        "weather",
        "forecast",
        "temperature",
        "rain",
        "snow",
        "wind",
        "humidity",
        "aqi",
        "air quality",
    ]
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
