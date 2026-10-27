#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== API & Review UI Harness Initialization ==="

if [[ ! -f data/eval/sample_page_1.png || ! -f data/eval/sample_page_2.png || ! -f data/eval/sample_page_3.png ]]; then
  echo "=== [Bootstrap] Generating missing eval sample pages ==="
  python3 data/eval/generate_sample_pages.py
fi

echo "=== 1. Python Syntax & Bytecode Check (api, ui, src/m2a) ==="
python3 -m compileall -q api ui src/m2a

echo "=== 2. API Job Queue, CP1/CP2 Checkpoint & RunState Integration Tests (pytest -m api_ui -q) ==="
python3 -m pytest -m "api_ui" -q

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Read root feature_list.json to see current feature state"
echo "2. Pick ONE unfinished API/UI feature to work on"
echo "3. Implement only that feature"
echo "4. Re-run ./harnesses/api-ui/init.sh and ./init.sh before claiming done"

