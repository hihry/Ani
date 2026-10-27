#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== M2A Machine Invariant & Toolchain Guards ==="

# 1. Toolchain Preflight (Go >= 1.26, python3, ffmpeg)
if ! command -v go >/dev/null 2>&1; then
  echo "FAIL: 'go' is not installed or not in PATH (requires Go >= 1.26)." >&2
  exit 1
fi
GO_VER="$(go version | awk '{print $3}' | sed 's/^go//')"
GO_MAJOR="$(echo "$GO_VER" | cut -d. -f1)"
GO_MINOR="$(echo "$GO_VER" | cut -d. -f2)"
if [[ "$GO_MAJOR" -lt 1 ]] || { [[ "$GO_MAJOR" -eq 1 ]] && [[ "$GO_MINOR" -lt 26 ]]; }; then
  echo "FAIL: Go >= 1.26 required, found go$GO_VER" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "FAIL: 'python3' is not installed or not in PATH." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FAIL: 'ffmpeg' is not installed or not in PATH." >&2
  exit 1
fi
echo "  [OK] Toolchain preflight (go$GO_VER, $(python3 --version 2>&1), ffmpeg present)"

# 2. Zero Ground-Truth Leakage in Production Code
LEAK_FILES="$(grep -rnE "ground_truth\.json|_load_eval_ground_truth" \
  workers/ src/m2a/nodes/ src/m2a/providers/ internal/ cmd/ api/ ui/ \
  --exclude="eval_*.py" \
  --exclude-dir="__pycache__" 2>/dev/null || true)"

if [[ -n "$LEAK_FILES" ]]; then
  echo "FAIL: Ground-truth leakage detected in production code:" >&2
  echo "$LEAK_FILES" >&2
  exit 1
fi
echo "  [OK] Zero ground-truth leakage across workers/ (excl. eval_*), src/m2a/nodes/, internal/, cmd/, api/, ui/"

# 3. Zero Tracked Model Weights or Run Artifacts in Git
TRACKED_FORBIDDEN="$(git ls-files | grep -E '\.(safetensors|gguf|ckpt|pth|onnx|pt|bin)$|^runs/' || true)"
if [[ -n "$TRACKED_FORBIDDEN" ]]; then
  echo "FAIL: Forbidden model weights or runs/ artifacts are tracked in git:" >&2
  echo "$TRACKED_FORBIDDEN" >&2
  exit 1
fi
echo "  [OK] Zero tracked model weights or runs/ artifacts in git index"

# 4. License Guard: hunyuan15.enabled must be false in configs/models.yaml
python3 -c '
import sys, yaml
from pathlib import Path
data = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
if data.get("hunyuan15", {}).get("enabled") is not False:
    print("FAIL: hunyuan15.enabled must remain False in configs/models.yaml (EU/UK/KR license exclusion)", file=sys.stderr)
    sys.exit(1)
'
echo "  [OK] License guard (hunyuan15.enabled == false)"
