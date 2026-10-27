#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VALIDATOR="${HARNESS_VALIDATOR:-/home/hihry/.gemini/config/skills/harness-creator/scripts/validate-harness.mjs}"

DOMAIN_HARNESSES=(
  "harnesses/go-control-plane"
  "harnesses/python-worker"
  "harnesses/eval-bakeoff"
  "harnesses/api-ui"
)

echo "========================================================"
echo " Validating M2A Root Master Harness + 4 Domain Harnesses"
echo "========================================================"

echo ""
echo ">>> 1. Checking Root Master Harness ($REPO_ROOT)"
for req in AGENTS.md CLAUDE.md GEMINI.md feature_list.json progress.md session-handoff.md init.sh; do
  if [[ ! -f "$REPO_ROOT/$req" ]]; then
    echo "FAIL: Missing root artifact: $req"
    exit 1
  fi
done
if [[ ! -x "$REPO_ROOT/init.sh" ]]; then
  echo "FAIL: Root init.sh is not executable"
  exit 1
fi
python3 -c '
import json, pathlib, sys
data = json.loads(pathlib.Path("feature_list.json").read_text(encoding="utf-8"))
features = data.get("features", [])
assert isinstance(features, list) and len(features) > 0, "feature_list.json must contain non-empty features list"
for f in features:
    for k in ("id", "name", "description", "status", "harness"):
        assert k in f, f"Feature {f} missing required key {k}"
    if f["status"] == "done":
        assert f.get("evidence"), f"Feature {f['id']} marked done without verification evidence"
print(f"PASS: root feature_list.json valid ({len(features)} features tracked)")
'

if [[ -f "$VALIDATOR" ]] && command -v node >/dev/null 2>&1; then
  node "$VALIDATOR" --target "$REPO_ROOT" --min-score 100
fi

echo ""
echo ">>> 2. Checking 4 Domain Harnesses (AGENTS.md + init.sh, zero duplicate state files)"
for dom in "${DOMAIN_HARNESSES[@]}"; do
  echo "  - $dom"
  if [[ ! -f "$REPO_ROOT/$dom/AGENTS.md" ]]; then
    echo "FAIL: Missing $dom/AGENTS.md"
    exit 1
  fi
  if [[ ! -x "$REPO_ROOT/$dom/init.sh" ]]; then
    echo "FAIL: Missing or non-executable $dom/init.sh"
    exit 1
  fi
  for section in "## Startup Workflow" "## Scope & Ownership Boundary" "## Definition of Done" "## Verification Commands"; do
    if ! grep -Fq "$section" "$REPO_ROOT/$dom/AGENTS.md"; then
      echo "FAIL: $dom/AGENTS.md missing required section: $section"
      exit 1
    fi
  done
  for dup in feature_list.json progress.md session-handoff.md; do
    if [[ -e "$REPO_ROOT/$dom/$dup" ]]; then
      echo "FAIL: Duplicate state file '$dom/$dup' found! State files must live only at repo root."
      exit 1
    fi
  done
done
echo "PASS: All 4 domain harnesses verified with zero duplicate state files."

echo ""
echo ">>> 3. Checking Real-Model GPU Probe & Machine Guards"
if [[ ! -x "$REPO_ROOT/harnesses/eval-bakeoff/probe-models.sh" ]]; then
  echo "FAIL: harnesses/eval-bakeoff/probe-models.sh is missing or not executable"
  exit 1
fi
if [[ ! -x "$REPO_ROOT/harnesses/guards.sh" ]]; then
  echo "FAIL: harnesses/guards.sh is missing or not executable"
  exit 1
fi
"$REPO_ROOT/harnesses/guards.sh"

echo ""
echo "========================================================"
echo " ALL HARNESSES & MACHINE GUARDS PASSED VALIDATION"
echo "========================================================"

