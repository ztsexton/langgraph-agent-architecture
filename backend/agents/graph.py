from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .meetings_agent import meetings_agent
from .rag_agent import rag_agent
from .routing import route, supervisor
from .types import AgentState
from .weather_agent import weather_agent
from .web_agent import web_agent


def get_agent_graph() -> "StateGraph[AgentState]":
    """Construct and return a compiled StateGraph for the multi-agent system."""
    graph_builder: StateGraph[AgentState] = StateGraph(AgentState)

    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("web_agent", web_agent)
    graph_builder.add_node("meetings_agent", meetings_agent)
    graph_builder.add_node("rag_agent", rag_agent)
    graph_builder.add_node("weather_agent", weather_agent)

    graph_builder.add_edge(START, "supervisor")
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

    graph_builder.add_edge("web_agent", END)
    graph_builder.add_edge("meetings_agent", END)
    graph_builder.add_edge("rag_agent", END)
    graph_builder.add_edge("weather_agent", END)

    return graph_builder.compile()


_compiled_agent_graph = get_agent_graph()
