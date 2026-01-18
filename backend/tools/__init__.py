# This package aggregates tool functions used by the multi‑agent system.
#
# The modules in this directory expose standalone functions that can be passed
# to LangGraph or other orchestration frameworks as "tools".  They
# encapsulate distinct behaviours such as web search, meeting management and
# retrieval‑augmented generation (RAG) with optional LLM integration.  Keeping
# them in a dedicated package makes it easy to reason about the available
# operations and encourages a clean separation of concerns.

from .web import search_web
from .meetings import (
    list_meetings,
    create_meeting,
    edit_meeting_agenda,
    edit_meeting_notes,
)
from .rag import answer_question

__all__ = [
    "search_web",
    "list_meetings",
    "create_meeting",
    "edit_meeting_agenda",
    "edit_meeting_notes",
    "answer_question",
]
