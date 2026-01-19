"""
Web search tool.

This module exposes a function that wraps the existing ``web_search`` helper
into a stable tool API.  Tools in this directory are simple callables that can
be passed directly to LangGraph or other orchestration frameworks.  They
accept plain arguments and return results without referencing any global state.
"""

from typing import List, Dict

from ..web_search import search as _search

from .langfuse_tracing import traced_tool


@traced_tool("web.search")
def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search the web for a query.

    Args:
        query: The search phrase.
        max_results: Maximum number of results to return.

    Returns:
        A list of dictionaries with keys ``title``, ``body`` and ``href``.  If
        the underlying search fails or returns no results, an empty list is
        returned.
    """
    return _search(query, max_results=max_results)
