# Multi-Agent Coordination Playbook — Manhwa → Anime (M2A)

Use this playbook when decomposing a large M2A feature across multiple subagents so workers never duplicate research, overwrite each other's files, or break the `Go ↔ Python` contract.

---

## 1. Ownership Partitioning Across Planes

When spawning parallel implementers, assign disjoint directory ownership aligned with the 4 domain harnesses:

| Agent Role | Domain Harness | Write-Allowed Paths | Verification Command |
|---|---|---|---|
| **Go Control Plane Implementer** | `harnesses/go-control-plane/` | `cmd/m2a/`, `internal/` | `./harnesses/go-control-plane/init.sh` |
| **Python Worker Implementer** | `harnesses/python-worker/` | `workers/`, `src/m2a/providers/`, `workflows/comfyui/` | `./harnesses/python-worker/init.sh` |
| **Eval & Calibration Implementer** | `harnesses/eval-bakeoff/` | `src/m2a/eval/`, `workers/eval_*.py`, `configs/{bakeoff,qc}/`, `tests/fixtures/` | `./harnesses/eval-bakeoff/init.sh` |
| **API & Review UI Implementer** | `harnesses/api-ui/` | `api/`, `ui/` | `./harnesses/api-ui/init.sh` |
| **Shared Contract Lock (`Coordinator Only`)** | Root `./AGENTS.md` | `internal/model/state.go`, `src/m2a/state.py`, `configs/default.yaml` | `./init.sh --quick` |

> **Golden Invariant**: If a feature changes `RunState` (`internal/model/state.go` and `src/m2a/state.py`), the **Coordinator** updates and verifies the shared contract **first** before dispatching parallel Go and Python worker subagents.

---

## 2. Three Delegation Patterns

1. **Coordinator Pattern (Recommended for Cross-Plane Features)**:
   - Phase 1: Spawn read-only `research` subagent to inspect affected Go and Python files.
   - Phase 2: Coordinator synthesizes findings and updates `internal/model/state.go` + `src/m2a/state.py` contract if needed.
   - Phase 3: Spawn scoped implementer subagent(s) with self-contained prompts and disjoint write paths.
   - Phase 4: Coordinator runs `./init.sh` and updates `feature_list.json` + `progress.md`.
2. **Single-Level Fork Pattern**:
   - Use only when splitting independent tests or read-only audits that share loaded context. Children must never fork recursively.
3. **Flat Swarm Pattern**:
   - Multiple agents pick disjoint features from `feature_list.json` where `"harness"` fields point to different domain harnesses.

---

## 3. Self-Contained Subagent Prompt Template

Always synthesize findings before delegating — never write *"Based on your findings, implement X"*.

```markdown
# Self-Contained M2A Worker Prompt

## Context (Synthesized by Coordinator)
- **Feature ID**: feat-XXX (<Name>)
- **Domain Harness**: `harnesses/<domain>/AGENTS.md`
- **Contract State**: `RunState` schema in `internal/model/state.go` and `src/m2a/state.py` is already locked at commit `<sha>`.

## Your Role
You are the **<Go Control Plane | Python Worker | Eval | API-UI> Implementer**.

## Exact Specification
1. Modify `<exact file paths and functions>`.
2. Enforce M2A fail-closed invariants (no fabricated QC scores, no dummy panels, no ground-truth leakage).
3. Add unit/integration tests in `<test file>`.

## Scope Constraints
- **Write-Allowed**: `<owned paths from table above>`
- **Do NOT Modify**: Unrelated planes or root harness state files (`feature_list.json`, `progress.md` — Coordinator updates these after verification).

## Verification Gate
Run `./harnesses/<domain>/init.sh` and confirm exit code 0 before returning.

## Deliverable
Return:
1. Summary of files changed
2. Exact output of `./harnesses/<domain>/init.sh`
3. Any blockers or edge cases discovered
```
