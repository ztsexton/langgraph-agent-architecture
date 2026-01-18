#!/usr/bin/env bash

# Simple bootstrap script to create a Python virtual environment, install
# dependencies and provide instructions for running the multi‑agent project.

set -euo pipefail

PROJECT_ROOT="$(dirname "$0")"

echo "Creating virtual environment in $PROJECT_ROOT/.venv..."
python3 -m venv "$PROJECT_ROOT/.venv"

echo "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

echo "Upgrading pip..."
pip3 install --upgrade pip

echo "Installing dependencies..."
# Determine the correct requirements file.  If ``backend/requirements.txt`` exists
# alongside this script use it; otherwise fall back to
# ``agent_project/backend/requirements.txt``.  This supports both directory
# layouts when pulling the project.
REQ_FILE="$PROJECT_ROOT/agent_project/backend/requirements.txt"
if [ -f "$PROJECT_ROOT/backend/requirements.txt" ]; then
  REQ_FILE="$PROJECT_ROOT/backend/requirements.txt"
fi
pip3 install -r "$REQ_FILE"

# Create a .env file with dummy values if it does not exist.  This makes it
# easier to run the application locally without having to supply environment
# variables manually.  Real deployments should replace these dummy values with
# valid keys.
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Creating .env file with dummy environment variables..."
  cat > "$ENV_FILE" <<'EOF'
# Dummy environment variables for local development.
# Replace these with real values for production.
OPENAI_API_KEY=changeme
AZURE_OPENAI_API_KEY=changeme
AZURE_OPENAI_API_BASE=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo

# Alternative auth method: Azure Entra ID (AAD) Client Credentials
# If these are set, the backend will acquire a bearer token and use that
# instead of AZURE_OPENAI_API_KEY.
AZURE_TENANT_ID=changeme
AZURE_CLIENT_ID=changeme
AZURE_CLIENT_SECRET=changeme
LOG_LEVEL=INFO
EOF
  echo ".env file created with dummy values at $ENV_FILE"
else
  echo ".env file already exists at $ENV_FILE; leaving it unchanged."
fi

echo "Setup complete."
echo "To activate the environment run:"
echo "source $PROJECT_ROOT/.venv/bin/activate"
echo "Then start the server with:"
echo "bash run_server.sh"
