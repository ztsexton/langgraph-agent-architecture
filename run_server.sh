#!/bin/bash
# Activate the Python virtual environment and run the FastAPI server.

set -e

# Determine the project root directory (the location of this script)
ROOT_DIR="$(cd "$(dirname "$0")"; pwd)"

# Change to project root to ensure relative paths are correct
cd "$ROOT_DIR"

# Activate virtual environment
. .venv/bin/activate

# Add project root to PYTHONPATH so Python can import agent_project
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Run the server with reload for development
uvicorn agent_project.backend.main:app --host 0.0.0.0 --port 8000 --reload
