FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY_DIR=/opt/ComfyUI

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-venv python3-pip ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY workflows/comfyui/manifest.yaml /workspace/workflows/comfyui/manifest.yaml
COPY scripts/comfyui/setup_local.sh /workspace/scripts/comfyui/setup_local.sh
COPY scripts/comfyui/download_models.py /workspace/scripts/comfyui/download_models.py
COPY scripts/comfyui/cloud/bootstrap.sh /workspace/scripts/comfyui/cloud/bootstrap.sh

RUN chmod +x /workspace/scripts/comfyui/setup_local.sh /workspace/scripts/comfyui/cloud/bootstrap.sh

EXPOSE 8188
ENTRYPOINT ["/workspace/scripts/comfyui/cloud/bootstrap.sh"]
