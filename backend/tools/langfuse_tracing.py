"""Langfuse tracing helpers.

Goal
- Make LLM generations, tool calls, and agent routing observable in Langfuse.
- Keep instrumentation optional (no-op unless LANGFUSE_* env vars are set).

This repo currently uses a simple LangGraph StateGraph with plain Python
functions for nodes/tools. We instrument at three levels:
1) A per-request top-level trace (created from `backend.main`)
2) LLM calls in `backend.tools.llm.ask_llm`
3) Tool calls via a tiny decorator usable by any tool callable

We store the current trace/span in context variables so nested calls (agent -> tool
-> http) can automatically attach to the active trace.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any, Callable, Optional, TypeVar, cast

try:
    from langfuse import Langfuse  # type: ignore
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore


_T = TypeVar("_T")

_langfuse_client: Optional[Any] = None

_current_trace: ContextVar[Optional[Any]] = ContextVar("langfuse_current_trace", default=None)
_current_span: ContextVar[Optional[Any]] = ContextVar("langfuse_current_span", default=None)


def _enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_HOST"))


def get_langfuse() -> Optional[Any]:
    """Return a singleton Langfuse client if configured, else None."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if not _enabled() or Langfuse is None:
        _langfuse_client = None
        return None

    _langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )
    return _langfuse_client


def start_trace(
    *,
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    input: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[Any]:
    """Start a Langfuse trace and set it as current."""
    client = get_langfuse()
    if client is None:
        return None

    trace = client.trace(
        name=name,
        user_id=user_id,
        session_id=session_id,
        input=input,
        metadata=metadata,
    )
    _current_trace.set(trace)
    _current_span.set(None)
    return trace


def get_current_trace() -> Optional[Any]:
    return _current_trace.get()


def start_span(
    *,
    name: str,
    input: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[Any]:
    """Start a span under the current trace/span and set as current."""
    trace = get_current_trace()
    if trace is None:
        return None

    parent = _current_span.get() or trace
    span = parent.span(
        name=name,
        input=input,
        metadata=metadata,
    )
    _current_span.set(span)
    return span


def end_span(span: Optional[Any], *, output: Optional[Any] = None, error: Optional[str] = None) -> None:
    if span is None:
        return
    if error:
        span.update(level="ERROR", status_message=error)
    if output is not None:
        span.update(output=output)
    span.end()
    # restore parent (best-effort)
    _current_span.set(None)


def end_trace(trace: Optional[Any], *, output: Optional[Any] = None, error: Optional[str] = None) -> None:
    if trace is None:
        return
    if error:
        trace.update(level="ERROR", status_message=error)
    if output is not None:
        trace.update(output=output)
    trace.end()
    _current_trace.set(None)
    _current_span.set(None)


def traced_tool(
    name: Optional[str] = None,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator to record tool calls as spans.

    Usage:
        @traced_tool("web.search")
        def search_web(...):
            ...
    """

    def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
        tool_name = name or fn.__name__

        def wrapped(*args: Any, **kwargs: Any) -> _T:
            span = start_span(
                name=f"tool:{tool_name}",
                input={"args": args, "kwargs": kwargs} if capture_input else None,
                metadata={"kind": "tool", "tool_name": tool_name},
            )
            try:
                out = fn(*args, **kwargs)
                if capture_output:
                    end_span(span, output=out)
                else:
                    end_span(span)
                return out
            except Exception as e:
                end_span(span, error=str(e))
                raise

        return cast(Callable[..., _T], wrapped)

    return deco
