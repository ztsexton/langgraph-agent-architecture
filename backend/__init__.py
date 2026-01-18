"""Package exposing the backend modules of the multi-agent demo."""

from .main import app  # Re-export FastAPI application for uvicorn

__all__ = ["app"]
