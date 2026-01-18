"""
FastAPI server providing a streaming API for the LangGraph multi‑agent system.

This application exposes a single endpoint, ``/stream``, that accepts a query
parameter ``message``.  It compiles the agent graph on startup and uses
LangGraph’s ``stream`` method to emit incremental state updates as the graph
executes.  The responses are transmitted as Server‑Sent Events (SSE), making
them compatible with modern event‑streaming clients and the AG‑UI protocol.

It also serves a simple static frontend under the ``/ui`` prefix, allowing you
to interact with the agents from a browser without any build tooling.  To start
the server run ``uvicorn agent_project.backend.main:app --reload`` from the
project root after installing dependencies.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agents import _compiled_agent_graph as agent_graph
from .agents import AgentState


app = FastAPI(title="LangGraph Multi‑Agent Backend")


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

    # Prepare the initial state; leave output blank – the worker will populate it
    initial_state: AgentState = {"input": message, "output": ""}

    def generate_events() -> AsyncIterator[str]:
        # Run the graph synchronously.  LangGraph will emit incremental
        # updates to the state on each step.  We choose the "updates" stream
        # mode so that only the changes are sent rather than the entire state.
        for chunk in agent_graph.stream(initial_state, stream_mode="updates"):
            # Each ``chunk`` is a dict keyed by node name with updated values.
            yield f"data: {json.dumps(chunk)}\n\n"

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
