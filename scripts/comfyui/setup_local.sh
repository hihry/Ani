#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
MANIFEST_PATH="${REPO_ROOT}/workflows/comfyui/manifest.yaml"
COMFY_COMMIT="7a28332047b6af638dc6c7f72ef3481a718769fd"

# Prevent uv from timing out or saturating bandwidth on large CUDA wheels
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"

# Default model storage on the high-capacity partition when /run/media/hihry/OS is present
if [ -z "${M2A_MODELS_DIR:-}" ]; then
  if [ -d "/run/media/hihry/OS" ] && [ -w "/run/media/hihry/OS" ]; then
    M2A_MODELS_DIR="/run/media/hihry/OS/m2a_models"
  else
    M2A_MODELS_DIR="${COMFY_DIR}/models"
  fi
fi

# Keep wheel caches off /home (which has limited btrfs space) when /run/media/hihry/OS is writable
if [ -d "/run/media/hihry/OS" ] && [ -w "/run/media/hihry/OS" ]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/run/media/hihry/OS/.cache/uv}"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/run/media/hihry/OS/.cache/pip}"
  mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}"
fi

echo "[m2a-comfyui] Setting up ComfyUI in ${COMFY_DIR} (manifest: ${MANIFEST_PATH})"
echo "[m2a-comfyui] Model storage directory: ${M2A_MODELS_DIR}"

if [ ! -d "${COMFY_DIR}/.git" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFY_DIR}"
fi

cd "${COMFY_DIR}"
git fetch --all --quiet || true
git checkout "${COMFY_COMMIT}" || echo "[m2a-comfyui] Warning: commit ${COMFY_COMMIT} not found locally, staying on HEAD"

# Ensure the ComfyUI virtualenv uses Python 3.12 (PyTorch wheels do not exist for Python 3.15)
if [ -x "${COMFY_DIR}/.venv/bin/python" ]; then
  VENV_PY_VER="$("${COMFY_DIR}/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "${VENV_PY_VER}" != "3.12" ]; then
    echo "[m2a-comfyui] Existing .venv uses Python ${VENV_PY_VER} (no PyTorch wheels); recreating with Python 3.12..."
    rm -rf "${COMFY_DIR}/.venv"
  fi
fi

if [ ! -x "${COMFY_DIR}/.venv/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "[m2a-comfyui] Installing/locating Python 3.12 via uv and creating ${COMFY_DIR}/.venv..."
    uv python install 3.12
    uv venv "${COMFY_DIR}/.venv" --python 3.12 --seed
  elif command -v python3.12 >/dev/null 2>&1; then
    echo "[m2a-comfyui] Creating ${COMFY_DIR}/.venv with system python3.12..."
    python3.12 -m venv "${COMFY_DIR}/.venv"
  elif command -v mise >/dev/null 2>&1; then
    echo "[m2a-comfyui] Installing Python 3.12 via mise and creating ${COMFY_DIR}/.venv..."
    mise install python@3.12
    mise exec python@3.12 -- python -m venv "${COMFY_DIR}/.venv"
  else
    echo "[m2a-comfyui] ERROR: Python 3.12 is required for PyTorch/ComfyUI (found $(python3 --version)). Install uv or python3.12." >&2
    exit 1
  fi
fi

# shellcheck disable=SC1091
source "${COMFY_DIR}/.venv/bin/activate"
echo "[m2a-comfyui] Active venv Python: $(python --version) ($(which python))"

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${COMFY_DIR}/.venv/bin/python" "$@"
  else
    "${COMFY_DIR}/.venv/bin/pip" install "$@"
  fi
}

echo "[m2a-comfyui] Installing PyTorch (CUDA 12.4) from PyPI CDN..."
pip_install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

# Ensure torch._library.infer_schema supports Python 3.9+ built-in generic list[T] used by comfy_kitchen
"${COMFY_DIR}/.venv/bin/python" - << 'PYEOF'
import pathlib, torch
p = pathlib.Path(torch.__file__).parent / "_library" / "infer_schema.py"
if p.is_file():
    txt = p.read_text(encoding="utf-8")
    if "list[typ]" not in txt:
        txt = txt.replace(
            "typing.List[typ],  # type: ignore[valid-type]\n        ]",
            "typing.List[typ],  # type: ignore[valid-type]\n            list[typ],  # type: ignore[valid-type]\n        ]",
        )
        txt = txt.replace(
            'typing.List[Tensor]: "Tensor[]",',
            'typing.List[Tensor]: "Tensor[]",\n    list[Tensor]: "Tensor[]",',
        )
        p.write_text(txt, encoding="utf-8")
PYEOF

echo "[m2a-comfyui] Installing ComfyUI requirements..."
pip_install -r "${COMFY_DIR}/requirements.txt"
pip_install huggingface_hub pyyaml

mkdir -p "${COMFY_DIR}/custom_nodes"
cd "${COMFY_DIR}/custom_nodes"

clone_or_update_node() {
  local name="$1"
  local repo="$2"
  if [ ! -d "${name}/.git" ]; then
    git clone "${repo}" "${name}"
  fi
  if [ -f "${name}/requirements.txt" ]; then
    echo "[m2a-comfyui] Installing requirements for custom node ${name}..."
    pip_install -r "${name}/requirements.txt" || echo "[m2a-comfyui] Warning: optional requirements for ${name} had warnings"
  fi
}

clone_or_update_node "ComfyUI-GGUF" "https://github.com/city96/ComfyUI-GGUF.git"
clone_or_update_node "ComfyUI-LTXVideo" "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
clone_or_update_node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git"

# Configure extra_model_paths.yaml so ComfyUI loads models from the large partition
mkdir -p "${M2A_MODELS_DIR}"/{checkpoints,clip,diffusion_models,loras,text_encoders,unet,vae}
cat > "${COMFY_DIR}/extra_model_paths.yaml" <<EOF
m2a:
  base_path: ${M2A_MODELS_DIR}
  checkpoints: checkpoints
  clip: text_encoders
  text_encoders: text_encoders
  diffusion_models: |
    diffusion_models
    unet
  unet: |
    unet
    diffusion_models
  loras: loras
  vae: vae
EOF

echo "[m2a-comfyui] Wrote ${COMFY_DIR}/extra_model_paths.yaml -> ${M2A_MODELS_DIR}"
echo "[m2a-comfyui] Local setup complete."
