from __future__ import annotations

from typing import Any, Dict, TypedDict


class AgentState(TypedDict, total=False):
    """Schema for the graph’s state."""

    input: str
    output: str
    # Optional structured UI payload ("a2ui"-style schema)
    a2ui: Dict[str, Any]
