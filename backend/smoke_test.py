"""Tiny local smoke test for the FastAPI app.

Runs without starting Uvicorn: it imports the app and calls endpoints via
FastAPI's TestClient.

Usage:
  /path/to/.venv/bin/python backend/smoke_test.py
"""

import os
import sys

from fastapi.testclient import TestClient


# Allow running as `python backend/smoke_test.py` from the repo root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app  # noqa: E402


def main() -> None:
    client = TestClient(app)

    r = client.get("/")
    assert r.status_code == 200, r.text

    r = client.get("/ui/")
    assert r.status_code in (200, 307), r.text

    # SSE endpoint: we just ensure we at least get a response and some bytes.
    r = client.get("/stream", params={"message": "hello"})
    assert r.status_code == 200, r.text
    assert "data:" in r.text or r.text.strip() != "", "Expected SSE response body"

    # If Langfuse is configured, the request should still succeed and emit a trace.
    # (We don't assert trace existence here since it depends on external services.)
    if os.getenv("LANGFUSE_HOST") and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        r = client.get("/stream", params={"message": "search langfuse observability"})
        assert r.status_code == 200, r.text

    print("smoke_test.py: PASS")


if __name__ == "__main__":
    main()
