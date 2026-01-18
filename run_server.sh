#!/usr/bin/env bash

# Launch the LangGraph multi‑agent backend server using uvicorn.  This script
# activates the Python virtual environment, loads environment variables from
# a .env file if present, adds the project root to the Python path so that
# the `agent_project` package can be imported, and then runs Uvicorn.

set -euo pipefail

# Determine the absolute path to the directory containing this script.  Using
# $(cd ... && pwd) resolves symlinks and ensures PROJECT_ROOT is an absolute
# path.  This directory is assumed to be the root of the repository.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Ensure a Python virtual environment exists.  We expect the setup script
# (setup_env.sh) to have created the .venv directory.
if [ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  echo "Virtual environment not found. Run ./setup_env.sh first." >&2
  exit 1
fi

# Activate the virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Load environment variables from .env if present.  This allows developers
# to specify API keys and configuration in a single file without exporting
# them manually.  If .env does not exist, this step does nothing.
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +o allexport
fi

# Add the project root to PYTHONPATH so that local packages can be imported
# when invoking uvicorn. If PYTHONPATH isn't set, default it to empty (this
# script runs with `set -u`, so referencing an unset var would error).
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Change to project root to ensure relative paths (e.g. agent_project) are
# resolved correctly by uvicorn and Python.
cd "$PROJECT_ROOT"

# Determine the Uvicorn app path.
# This repository layout is `backend/` at the project root.
# Some downstream forks may wrap it under an `agent_project/` top-level package.
# Prefer the local `backend.main:app` unless that doesn't exist.
APP_MODULE="backend.main:app"
if [ -f "$PROJECT_ROOT/agent_project/backend/main.py" ]; then
  APP_MODULE="agent_project.backend.main:app"
fi

# Use Uvicorn to serve the FastAPI app.  ``--host`` and ``--port`` are
# explicitly specified to listen on all interfaces; ``--reload`` enables
# auto‑reload for development convenience.  In production you may remove
# the ``--reload`` flag.
exec uvicorn "$APP_MODULE" --host 0.0.0.0 --port 8000 --reload
