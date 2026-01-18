"""
Build and expose a LangGraph‑based multi‑agent architecture.

This module constructs a simple agent graph consisting of a supervisor node
and three specialised worker nodes: ``web_agent`` for performing web searches,
``meetings_agent`` for interacting with an in‑memory calendar, and
``rag_agent`` for answering questions from a small document corpus.  A
supervisor examines the user’s input and chooses which worker to invoke.  The
graph exposes streaming updates when run through the LangGraph runtime.

The implementation here is intentionally basic.  It does not invoke any large
language models; instead, the routing logic uses keyword matching and the
worker nodes return heuristic or stubbed responses.  This keeps the example
lightweight while demonstrating the core concepts of multi‑agent delegation
within LangGraph.

To compile the graph call :func:`get_agent_graph()` which returns a compiled
``StateGraph`` ready for invocation or streaming.
"""

from __future__ import annotations

from typing import TypedDict, Dict, Literal, Optional

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


class AgentState(TypedDict):
    """Schema for the graph’s state.

    The ``input`` key contains the raw user message.  The ``output`` key
    holds the response returned by whichever worker agent processed the
    request.  Additional keys could be added here (e.g. conversation
    history) if you extend the system to support multi‑turn interactions.
    """

    input: str
    output: str


# Shared resources are managed by the tools modules.  There is no need to
# instantiate any state here.


def supervisor(state: AgentState) -> AgentState:
    """Pass through the state unmodified.

    The supervisor node itself does not update state; it only decides which
    worker node should run next via the routing function defined below.
    """
    return {}


def route(state: AgentState) -> Literal["web_agent", "meetings_agent", "rag_agent"]:
    """Determine which worker agent should handle the current request.

    This simple router uses keyword matching on the lowercase input to
    choose an agent.  A more sophisticated implementation might use an
    LLM or structured schema to decide which domain to route to.
    """
    text = state.get("input", "").lower()
    # Prioritise meeting‑related queries first
    meeting_keywords = ["meeting", "schedule", "agenda", "notes"]
    rag_keywords = ["document", "doc", "citation", "reference"]
    search_keywords = ["search", "web", "internet", "look up"]
    if any(k in text for k in meeting_keywords):
        return "meetings_agent"
    if any(k in text for k in rag_keywords):
        return "rag_agent"
    if any(k in text for k in search_keywords):
        return "web_agent"
    # Default route – treat as a general search
    return "web_agent"


def web_agent(state: AgentState) -> AgentState:
    """Perform a web search and populate the output key with formatted results.
    """
    """Perform a web search and optionally summarise using an LLM."""
    query = state.get("input", "")
    results = search_web(query, max_results=3)
    # If no results found return immediately
    if not results:
        return {"output": "No search results found."}
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
    # Attempt to use LLM to summarise the search results into a single answer
    llm = get_llm()
    if llm:
        prompt = (
            "You are a helpful assistant.  A user searched for the following query:\n\n"
            f"{query}\n\n"
            "Here are the top search results:\n\n"
            f"{summary}\n\n"
            "Provide a concise summary or answer to the user's query based on these results."
        )
        try:
            answer = ask_llm(prompt)
            return {"output": answer}
        except Exception:
            # Fall back to raw summary if LLM fails
            pass
    return {"output": summary}


def meetings_agent(state: AgentState) -> AgentState:
    """Handle simple meeting operations based on the user input.

    The logic here is deliberately naive; it uses keyword patterns to decide
    which operation to perform.  In a real system you would likely parse
    structured commands or rely on an LLM to extract parameters.
    """
    text = state.get("input", "").lower()
    # List meetings
    if "list" in text and "meeting" in text:
        meetings = list_meetings()
        if not meetings:
            return {"output": "There are no meetings scheduled."}
        lines = [
            f"{m.id}. {m.title} on {m.date} – agenda: {m.agenda}; notes: {m.notes or 'None'}"
            for m in meetings
        ]
        return {"output": "\n".join(lines)}
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
        return {"output": f"Created meeting {meeting.id}: {meeting.title} on {meeting.date}."}
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
            return {"output": "Please specify the meeting ID to edit."}
        meeting = edit_meeting_agenda(meeting_id, new_agenda)
        if meeting is None:
            return {"output": f"Meeting {meeting_id} not found."}
        return {"output": f"Updated agenda for meeting {meeting.id}."}
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
            return {"output": "Please specify the meeting ID to edit."}
        meeting = edit_meeting_notes(meeting_id, new_notes)
        if meeting is None:
            return {"output": f"Meeting {meeting_id} not found."}
        return {"output": f"Updated notes for meeting {meeting.id}."}
    # Default response
    return {
        "output": (
            "I can manage meetings. Try commands like 'list meetings', 'create meeting "
            "Team Sync on 2026-02-20 agenda Discuss progress', 'edit meeting 1 agenda New agenda' or "
            "'edit meeting 1 notes New notes'."
        )
    }


def rag_agent(state: AgentState) -> AgentState:
    """Answer a query using retrieval augmented search."""
    query = state.get("input", "")
    result = answer_question(query)
    content = result["content"]
    citation = result["citation"]
    return {"output": f"{content} (Citation: {citation})"}


def get_agent_graph() -> "StateGraph[AgentState]":
    """Construct and return a compiled StateGraph for the multi‑agent system."""
    graph_builder: StateGraph[AgentState] = StateGraph(AgentState)
    # Register nodes
    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("web_agent", web_agent)
    graph_builder.add_node("meetings_agent", meetings_agent)
    graph_builder.add_node("rag_agent", rag_agent)
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
        },
    )
    # Terminate after worker finishes by sending to END
    graph_builder.add_edge("web_agent", END)
    graph_builder.add_edge("meetings_agent", END)
    graph_builder.add_edge("rag_agent", END)
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
