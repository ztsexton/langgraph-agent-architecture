#!/bin/bash
# Setup a Python virtual environment and install dependencies.

set -e

python -m venv .venv
# Activate virtual environment
. .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install backend requirements
pip install -r agent_project/backend/requirements.txt

echo "Setup complete."
echo "Activate the virtual environment with 'source .venv/bin/activate' and start the server using ./run_server.sh"
