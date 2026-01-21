"""Build and expose a LangGraph-based multi-agent architecture.

This package constructs an agent graph consisting of a supervisor node and
specialised worker nodes:
- ``web_agent`` for web search / general Q&A
- ``meetings_agent`` for calendar interactions
- ``rag_agent`` for retrieval-augmented answers
- ``weather_agent`` for structured weather responses

Public API is kept compatible with the prior single-file module:
- ``AgentState``
- ``get_agent_graph``
- ``_compiled_agent_graph``
"""

from .types import AgentState
from .graph import get_agent_graph, _compiled_agent_graph

__all__ = [
    "AgentState",
    "get_agent_graph",
    "_compiled_agent_graph",
]
