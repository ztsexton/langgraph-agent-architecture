"""
FastAPI server providing a streaming API for the LangGraph multi‑agent system.

This application exposes a single endpoint, ``/stream``, that accepts a query
parameter ``message``. It compiles the agent graph on startup and uses
LangGraph’s ``stream`` method to emit incremental state updates as the graph
executes. The responses are transmitted as Server‑Sent Events (SSE), making
them compatible with modern event‑streaming clients and the AG‑UI protocol.

It also serves a simple static frontend under the ``/ui`` prefix, allowing you
to interact with the agents from a browser without any build tooling. To start
the server run ``uvicorn agent_project.backend.main:app --reload`` from the
project root after installing dependencies.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agents import _compiled_agent_graph as agent_graph
from .agents import AgentState

from .tools.langfuse_tracing import start_trace, end_trace

import logging


app = FastAPI(title="LangGraph Multi‑Agent Backend")

# Configure a simple application-wide logger.  The log level can be set via
# the LOG_LEVEL environment variable (default: INFO).  Logs are emitted to
# standard output, which can be captured by the hosting environment.
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("agent_backend.main")


@app.middleware("http")
async def no_cache_ui_assets(request: Request, call_next):
    response = await call_next(request)
    # Avoid stale frontend assets during development.
    if request.url.path == "/ui" or request.url.path.startswith("/ui/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/stream")
async def stream(message: str) -> StreamingResponse:
    """Stream graph updates as Server‑Sent Events for a given user message.

    Clients should open an EventSource on this endpoint and supply a
    ``message`` query parameter.  The agent graph will run once and
    yield a series of JSON objects describing state updates.  Each event is
    prepended with ``data:`` as required by the SSE specification.

    Args:
        message: The user’s request to send to the supervisor.

    Returns:
        A streaming HTTP response with ``text/event-stream`` media type.
    """

    # Start a Langfuse trace (optional) so all LLM/tool spans are linked.
    trace = start_trace(
        name="/stream",
        input={"message": message},
        metadata={"endpoint": "/stream"},
    )

    # Prepare the initial state; leave output blank – the worker will populate it
    initial_state: AgentState = {"input": message, "output": ""}

    # Log the incoming message
    logger.info(f"Received stream request: {message}")

    def generate_events() -> Iterator[str]:
        try:
            # Run the graph synchronously.  LangGraph will emit incremental
            # updates to the state on each step.  We choose the "updates" stream
            # mode so that only the changes are sent rather than the entire state.
            last_chunk = None
            for chunk in agent_graph.stream(initial_state, stream_mode="updates"):
                last_chunk = chunk
                # Log each update emitted by the graph
                logger.info(f"Graph update: {chunk}")
                # Each ``chunk`` is a dict keyed by node name with updated values.
                yield f"data: {json.dumps(chunk)}\n\n"
            end_trace(trace, output={"last_chunk": last_chunk})
        except Exception as e:
            end_trace(trace, error=str(e))
            raise

    return StreamingResponse(generate_events(), media_type="text/event-stream")


@app.get("/")
async def root() -> JSONResponse:
    """Return a brief description of the API."""
    return JSONResponse(
        {
            "message": "LangGraph multi‑agent backend is running. Access the frontend under /ui and use /stream?message=... for streaming.",
        }
    )


# Mount static files for the simple frontend.  We resolve the path relative to
# this file to locate the sibling 'frontend' directory in the project root.
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")
