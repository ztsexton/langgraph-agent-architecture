#!/bin/bash
# Activate the Python virtual environment and run the FastAPI server.

set -e

# Activate virtual environment
. .venv/bin/activate

# Run the server
uvicorn agent_project.backend.main:app --host 0.0.0.0 --port 8000 --reload
