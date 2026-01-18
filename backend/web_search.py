"""
Web search utilities for the multi‑agent system.

This module attempts to perform a real web search using the ``duckduckgo_search``
package.  If the dependency is not installed or network access is not allowed,
it gracefully falls back to returning a static response.  The search results
returned are a list of dictionaries containing a title, a brief snippet and a
URL (href) when available.
"""

from __future__ import annotations

import logging
from typing import List, Dict


logger = logging.getLogger("agent_backend.web_search")


def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using DuckDuckGo.

    Args:
        query: Query string to search for.
        max_results: Maximum number of results to return.

    Returns:
        List of dictionaries with keys ``title``, ``body`` and ``href``.
    """
    try:
        # `duckduckgo_search` has been renamed to `ddgs`.
        # Prefer `ddgs` when available, but keep compatibility.
        try:
            from ddgs import DDGS  # type: ignore
        except Exception:  # pragma: no cover
            from duckduckgo_search import DDGS  # type: ignore

        results: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, safesearch="moderate", max_results=max_results):
                results.append(
                    {
                        "title": result.get("title", ""),
                        "body": result.get("body", ""),
                        "href": result.get("href", ""),
                    }
                )
        return results
    except Exception as e:
        # No mock/stub content: callers should decide how to handle an empty result set.
        logger.warning("Web search failed; returning no results", exc_info=e)
        return []


__all__ = ["search"]
