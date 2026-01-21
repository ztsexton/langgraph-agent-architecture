from __future__ import annotations

from ..tools.langfuse_tracing import end_span, start_span
from ..tools.rag import answer_question

from .types import AgentState
from .ui import a2ui_text


def rag_agent(state: AgentState) -> AgentState:
    """Answer a query using retrieval augmented search."""
    _span = start_span(name="agent:rag_agent", input={"state": state}, metadata={"kind": "agent"})
    query = state.get("input", "")
    result = answer_question(query)
    content = result["content"]
    citation = result["citation"]
    text = f"{content} (Citation: {citation})"
    out = {"output": text, "a2ui": a2ui_text("RAG", text)}
    end_span(_span, output=out)
    return out
