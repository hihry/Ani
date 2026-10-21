#!/usr/bin/env bash
set -euo pipefail

export COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
TIER="${M2A_TIER:-t2}"

if [ ! -d "${COMFY_DIR}" ]; then
  /workspace/scripts/comfyui/setup_local.sh
fi

python3 /workspace/scripts/comfyui/download_models.py --comfy-dir "${COMFY_DIR}" --tier "${TIER}" --dry-run

cd "${COMFY_DIR}"
# shellcheck disable=SC1091
source "${COMFY_DIR}/.venv/bin/activate"
exec python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch
