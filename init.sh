#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="full"
if [[ "${1:-}" == "--quick" ]]; then
  MODE="quick"
fi

echo "=== M2A Master Harness Initialization ($MODE mode) ==="

echo "=== 0. Machine-Enforced Invariant & Toolchain Guards (./harnesses/guards.sh) ==="
./harnesses/guards.sh

if [[ ! -f data/eval/sample_page_1.png || ! -f data/eval/sample_page_2.png || ! -f data/eval/sample_page_3.png ]]; then
  echo "=== [Bootstrap] Generating missing eval sample pages (data/eval/sample_page_*.png) ==="
  python3 data/eval/generate_sample_pages.py
fi

echo "=== 1. Go Static Analysis (go vet ./...) ==="
go vet ./...

echo "=== 2. Go Build Check (go build -o m2a ./cmd/m2a) ==="
go build -o m2a ./cmd/m2a

echo "=== 3. Go Unit & Integration Tests (go test ./...) ==="
go test ./...

echo "=== 4. Python Syntax & Bytecode Check (compileall) ==="
python3 -m compileall -q -x '(^|/)(\.?venv|env|node_modules|build|dist|__pycache__|runs)(/|$)' .

if [[ "$MODE" == "quick" ]]; then
  echo "=== 5. Python Fast Contract Smoke Tests (pytest -m contract -q) ==="
  python3 -m pytest -m "contract" -q
  echo ""
  echo "[NOTE] '--quick' is a fast iteration smoke check only."
  echo "       You MUST run './init.sh' (full mode, 45 pytest tests) before claiming any feature 'done'."
else
  echo "=== 5. Python Full End-to-End & Phase 3 Test Suite (pytest tests/ -q) ==="
  python3 -m pytest tests/ -q
fi

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Read feature_list.json to see current feature state"
echo "2. Pick ONE unfinished feature to work on (and check its domain harness in harnesses/)"
echo "3. Implement only that feature following harnesses/playbooks/new-feature-blueprint.md"
echo "4. Re-run ./init.sh (full mode) before claiming done"
echo "5. For live GPU / ComfyUI model readiness & anti-slideshow audit, run: ./harnesses/eval-bakeoff/probe-models.sh"

