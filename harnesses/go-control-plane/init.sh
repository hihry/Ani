#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Go Control Plane Harness Initialization ==="

if [[ ! -f data/eval/sample_page_1.png || ! -f data/eval/sample_page_2.png || ! -f data/eval/sample_page_3.png ]]; then
  echo "=== [Bootstrap] Generating missing eval sample pages ==="
  python3 data/eval/generate_sample_pages.py
fi

echo "=== 1. Go Vet (go vet ./...) ==="
go vet ./...

echo "=== 2. Go Unit & Integration Tests (go test ./...) ==="
go test ./...

echo "=== 3. Go CLI Build Check (go build -o m2a ./cmd/m2a) ==="
go build -o m2a ./cmd/m2a

echo "=== 4. Go <-> Python RunState Contract & Seed Parity (pytest -m contract -q) ==="
python3 -m pytest -m "contract" -q

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Read root feature_list.json to see current feature state"
echo "2. Pick ONE unfinished Go control-plane feature to work on"
echo "3. Implement only that feature"
echo "4. Re-run ./harnesses/go-control-plane/init.sh and ./init.sh before claiming done"

