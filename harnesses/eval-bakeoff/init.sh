#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Evaluation, Bake-off & QC Calibration Harness Initialization ==="

if [[ ! -f data/eval/sample_page_1.png || ! -f data/eval/sample_page_2.png || ! -f data/eval/sample_page_3.png ]]; then
  echo "=== [Bootstrap] Generating missing eval sample pages ==="
  python3 data/eval/generate_sample_pages.py
fi

echo "=== 1. Python Syntax & Bytecode Check (src/m2a/eval, workers/eval_*.py, workers/qc_metrics.py) ==="
python3 -m compileall -q src/m2a/eval workers/eval_segmentation.py workers/eval_understanding.py workers/qc_metrics.py

echo "=== 2. Phase 1 Segmentation F1 & IoU Benchmark ==="
python3 workers/eval_segmentation.py

echo "=== 3. Phase 2 & 3 Eval, QC Metrics v2, Adversarial Fixtures, Bake-off, Rubric & Probe Tests (pytest -m eval_qc -q) ==="
python3 -m pytest -m "eval_qc" -q

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Read root feature_list.json to see current feature state"
echo "2. Pick ONE unfinished eval/bake-off/QC feature to work on"
echo "3. For live GPU model verification & anti-slideshow audit, run: ./harnesses/eval-bakeoff/probe-models.sh"
echo "4. Re-run ./harnesses/eval-bakeoff/init.sh and ./init.sh before claiming done"

