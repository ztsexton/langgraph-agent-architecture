#!/bin/bash
# Setup a Python virtual environment and install dependencies. Also writes a .env file with dummy values if missing.

set -e

# Determine project root directory (location of this script)
ROOT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "$ROOT_DIR"

echo "Creating virtual environment in $ROOT_DIR/.venv..."
python -m venv "$ROOT_DIR/.venv"

echo "Activating virtual environment..."
. "$ROOT_DIR/.venv/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from agent_project/backend/requirements.txt..."
pip install -r "$ROOT_DIR/agent_project/backend/requirements.txt"

# Create .env file with dummy environment variables if it doesn't exist
ENV_FILE="$ROOT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
cat > "$ENV_FILE" <<'EOF'
# Dummy environment variables for development. Replace with real keys.
OPENAI_API_KEY=dummy-openai-key
AZURE_OPENAI_API_KEY=dummy-azure-key
AZURE_OPENAI_API_BASE=https://dummy-api.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=dummy-deployment
LOG_LEVEL=INFO
EOF
echo "Created .env file with dummy values."
fi

echo "Setup complete."
echo "Activate the virtual environment with 'source $ROOT_DIR/.venv/bin/activate' and start the server using ./run_server.sh"
