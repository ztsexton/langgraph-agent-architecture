"""
Web search utilities for the multi‑agent system.

This module attempts to perform a real web search using the ``duckduckgo_search``
package.  If the dependency is not installed or network access is not allowed,
it gracefully falls back to returning a static response.  The search results
returned are a list of dictionaries containing a title, a brief snippet and a
URL (href) when available.
"""

from __future__ import annotations

from typing import List, Dict


def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using DuckDuckGo or return a mock response.

    Args:
        query: Query string to search for.
        max_results: Maximum number of results to return.

    Returns:
        List of dictionaries with keys ``title``, ``body`` and ``href``.
    """
    try:
        from duckduckgo_search import DDGS  # type: ignore

        # Use the new async API if available; otherwise fall back to simple
        # synchronous usage.  We wrap in a context manager to ensure clean
        # shutdown of underlying HTTP session.
        results: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, safesearch="moderate", max_results=max_results):
                # Each ``result`` dict has keys 'title', 'body', 'href'
                results.append(
                    {
                        "title": result.get("title", ""),
                        "body": result.get("body", ""),
                        "href": result.get("href", ""),
                    }
                )
        return results
    except Exception:
        # If we cannot import the package or the search fails (e.g. no internet
        # connectivity) return a mock result.  This fallback ensures the
        # web agent remains functional in constrained environments.
        return [
            {
                "title": "Web search unavailable",
                "body": (
                    "The DuckDuckGo search API is not installed or network access is disabled. "
                    "This is a mock response."
                ),
                "href": "",
            }
        ]


__all__ = ["search"]
