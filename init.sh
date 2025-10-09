#!/bin/bash
# ==============================================
# Install dependencies for ALTA custom node
# ==============================================

# Find script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_ROOT="$SCRIPT_DIR/../.."  # adjust if your structure differs
VENV_PYTHON="$COMFYUI_ROOT/.venv/bin/python"

# Check if venv python exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Could not find ComfyUI venv at: $VENV_PYTHON"
    echo "Please run ComfyUI at least once to create the virtual environment."
    exit 1
fi

# Install dependencies
echo "🚀 Installing ALTA node dependencies..."
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

if [ $? -eq 0 ]; then
    echo "✅ ALTA NODES LOAD SUCCESS"
else
    echo "❌ Installation failed"
    exit 1
fi