#!/bin/bash
# Always install into the ComfyUI venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_ROOT="$SCRIPT_DIR/../.."  # adjust as needed

"$COMFYUI_ROOT/.venv/Scripts/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
echo "✅ ALTA NODES LOAD SUCCESS"