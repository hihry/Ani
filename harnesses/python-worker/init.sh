#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="full"
if [[ "${1:-}" == "--quick" ]]; then
  MODE="quick"
fi

echo "=== Python AI Worker Harness Initialization ($MODE mode) ==="

echo "=== 0. Machine-Enforced Invariant & Toolchain Guards (./harnesses/guards.sh) ==="
./harnesses/guards.sh

if [[ ! -f data/eval/sample_page_1.png || ! -f data/eval/sample_page_2.png || ! -f data/eval/sample_page_3.png ]]; then
  echo "=== [Bootstrap] Generating missing eval sample pages ==="
  python3 data/eval/generate_sample_pages.py
fi

echo "=== 1. Python Syntax & Bytecode Check (workers, src/m2a, tests) ==="
python3 -m compileall -q workers src/m2a tests

if [[ "$MODE" == "quick" ]]; then
  echo "=== 2. Fast Worker, FastAPI Endpoints & Video Provider Smoke Tests (-m worker) ==="
  python3 -m pytest tests/test_phase3.py tests/test_pipeline.py::test_worker_fastapi_endpoints tests/test_pipeline.py::test_zero_ground_truth_leakage_in_production_code -m "worker" -q
  echo ""
  echo "[NOTE] '--quick' is a smoke check only; run './harnesses/python-worker/init.sh' (full) and './init.sh' before claiming done."
else
  echo "=== 2. Full Worker & Phase 3 Pytest Suite (pytest -m worker -q) ==="
  python3 -m pytest -m "worker" -q
fi

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Read root feature_list.json to see current feature state"
echo "2. Pick ONE unfinished Python worker feature to work on"
echo "3. Implement only that feature"
echo "4. Re-run ./harnesses/python-worker/init.sh and ./init.sh before claiming done"

