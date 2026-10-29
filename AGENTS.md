# AGENTS.md — Manhwa → Anime (M2A) Master Harness

Foolproof multi-harness system for reliable agent-assisted development in the **Manhwa → Anime (M2A)** hybrid Go + Python repository.

## Startup Workflow

Before writing code:

1. **Confirm working directory** with `pwd` (`/run/media/hihry/OS/hihry/Ani`)
2. **Read this file** completely and select the matching sub-harness if working in a single plane
3. **Read project docs if present** ([README.md](README.md), [ARCHITECTURE_EXCALIDRAW_MERMAID.md](ARCHITECTURE_EXCALIDRAW_MERMAID.md), [Phase3_implemn.md](Phase3_implemn.md), [optim.md](optim.md))
4. **Run `./init.sh`** (or `./init.sh --quick` as a fast smoke check during iteration; never sufficient for claiming `done`) to verify the Go + Python environment is healthy
5. **Read `feature_list.json` and `progress.md`** (single root source of truth across all planes) to see current feature state and active objective
6. **Review recent commits** with `git log --oneline -5`

If baseline verification is failing, repair that first before adding new scope.

## Multi-Harness Routing Map

This repository uses a **Root Master Harness** plus **4 Specialized Domain Harnesses** and **Playbooks** so any future feature can be implemented in strict isolation or cross-plane coordination. State files (`feature_list.json`, `progress.md`, `session-handoff.md`) live **only at the repository root**; domain directories contain only `AGENTS.md` and `init.sh` (plus `probe-models.sh` in `eval-bakeoff`):

| Harness / Playbook | Directory / Path | Target Scope | Verification Entrypoint |
|---|---|---|---|
| **Root Master Harness** | `./` (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`) | Cross-plane features, full 12-stage pipeline, contract changes | `./init.sh` |
| **Go Control Plane Harness** | `harnesses/go-control-plane/` | `cmd/m2a/`, `internal/{config,model,client,qc,pipeline}/` | `./harnesses/go-control-plane/init.sh` (`-m contract`) |
| **Python AI Worker Harness** | `harnesses/python-worker/` | `workers/`, `src/m2a/`, `workflows/comfyui/` | `./harnesses/python-worker/init.sh` (`-m worker`) |
| **Eval, Bake-off & QC Harness** | `harnesses/eval-bakeoff/` | `src/m2a/eval/`, `workers/eval_*.py`, `configs/{bakeoff,qc}/`, `data/eval/` | `./harnesses/eval-bakeoff/init.sh` (`-m eval_qc`) |
| **Live GPU & Slideshow Probe** | `harnesses/eval-bakeoff/probe-models.sh` | Model SHA-256 on `/run/media/hihry/OS/m2a_models/`, ComfyUI `:8188`, 1-panel I2V render, slideshow audit | `./harnesses/eval-bakeoff/probe-models.sh` |
| **API & Review UI Harness** | `harnesses/api-ui/` | `api/`, `ui/` (FastAPI Job Queue & CP1/CP2 Streamlit Editor) | `./harnesses/api-ui/init.sh` (`-m api_ui`) |
| **Machine Invariant Guards** | `harnesses/guards.sh` | Toolchain preflight, zero ground-truth leakage, zero tracked weights, `hunyuan15.enabled: false` | `./harnesses/guards.sh` |
| **New Feature Blueprint** | `harnesses/playbooks/new-feature-blueprint.md` | Step-by-step recipe for adding any future stage, provider, tier, or metric | `./harnesses/validate-all.sh` |
| **Multi-Agent Coordination** | `harnesses/playbooks/multi-agent-coordination.md` | Coordinator / Fork / Swarm prompt templates & ownership boundaries | `./harnesses/validate-all.sh` |
| **Memory & Context Budget** | `harnesses/playbooks/memory-and-context-budget.md` | Context engineering, tool safety, and M2A failure-mode guardrails | `./harnesses/validate-all.sh` |

## Working Rules & M2A Invariants

- **One feature at a time**: Pick exactly one unfinished feature from root `feature_list.json` (unless dispatching bounded parallel workers via `harnesses/playbooks/multi-agent-coordination.md`).
- **Stay in scope**: Do not modify files unrelated to the active feature or violate plane boundaries.
- **Verification required**: Never claim done without running full `./init.sh` (and the relevant domain harness `init.sh`, plus `./harnesses/eval-bakeoff/probe-models.sh` when claiming live GPU I2V verification) and recording exact command output.
- **Update artifacts**: Before ending a session, update root `progress.md`, `feature_list.json`, and `session-handoff.md`.
- **Leave clean state**: Next session must be able to run `./init.sh` immediately from a clean, restartable checkout.
- **Zero silent stubs (Fail-Closed)**: Never fabricate QC scores, dummy panels, or fake output paths when a provider or stage fails. Record honest `route_used` degradations (`i2v -> parallax -> kenburns`) or return explicit errors.
- **Zero ground-truth leakage**: Production code (`workers/` excluding `eval_*.py`, `src/m2a/nodes/`, `src/m2a/providers/`, `internal/`, `cmd/`, `api/`, `ui/`) must never load `data/eval/ground_truth.json` (enforced by `./harnesses/guards.sh` and `test_zero_ground_truth_leakage_in_production_code`).
- **Go ↔ Python `RunState` Lockstep**: Any schema addition or change to `RunState`, `PanelJSON`, `CharacterBible`, `ClipCandidate`, or `QCScores` must be updated simultaneously in `internal/model/state.go` and `src/m2a/state.py` with matching JSON tags and golden tests.
- **Go Engine is Canonical**: `internal/pipeline/` is the single source of truth for DAG orchestration, budget routing, and QC fallback ladders. `api/` and `ui/` must delegate to `m2a` or read `runs/<id>/state.json` rather than duplicating orchestration logic (`feat-005` / `optim.md` A1 & P0-07).
- **Disk & Environment Safety**: Never commit model weights (`.safetensors`, `.gguf`, `.pth`, `.onnx`) or `runs/` artifacts. Heavy weights belong on `/run/media/hihry/OS/m2a_models/`, not `/home` (19 GB cap). ComfyUI runs in its own Python 3.12 venv over HTTP `:8188`.

## Required Artifacts

- `feature_list.json` — Single root feature state tracker and dependency graph (source of truth)
- `progress.md` — Single root session continuity log and verification evidence record
- `init.sh` — Standard fail-fast startup and verification gate
- `session-handoff.md` — Single root cross-session handoff state
- `harnesses/*/AGENTS.md` & `harnesses/*/init.sh` — Domain-specific instructions and marker-scoped verification gates

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] Target behavior is implemented with zero placeholder stubs or fabricated metrics
- [ ] Go ↔ Python `RunState` and config schemas remain 100% compatible (if touched)
- [ ] Required verification actually ran (`./init.sh` passes `./harnesses/guards.sh`, Go vet, Go build, Go tests, Python compile, and all 47 pytest tests; plus `./harnesses/eval-bakeoff/probe-models.sh` for live GPU features)
- [ ] Verification Evidence (exact command and output summary) is recorded in `feature_list.json` and `progress.md`
- [ ] Repository remains clean and restartable from the standard startup path

## End of Session

Before ending a session:

1. Run `./init.sh` and `./harnesses/validate-all.sh` to confirm code and harness integrity
2. Update `progress.md` with current state, modified files, and verification evidence
3. Update `feature_list.json` with new feature `status` (`not-started` | `in-progress` | `done` | `blocked`) and `evidence`
4. Update `session-handoff.md` with blockers, decisions, and recommended next step
5. Leave the repository clean enough for the next session to run `./init.sh` immediately

## Verification Commands

```bash
# Full master verification (REQUIRED before claiming done)
./init.sh

# Fast iteration smoke check ONLY (runs 9 contract tests; NOT sufficient for claiming done)
./init.sh --quick

# Validate all 5 project harnesses + machine guards
./harnesses/validate-all.sh

# Live GPU model SHA-256 + ComfyUI :8188 + 1-panel real I2V render + anti-slideshow detector
./harnesses/eval-bakeoff/probe-models.sh --tier t1s
./harnesses/eval-bakeoff/probe-models.sh --check-run runs/<run_id>
```

Required checks executed by `./init.sh`:
- `./harnesses/guards.sh`
- `go vet ./...`
- `go build -o m2a ./cmd/m2a`
- `go test ./...`
- `python3 -m compileall -q -x '(^|/)(\.?venv|env|node_modules|build|dist|__pycache__|runs)(/|$)' .`
- `python3 -m pytest tests/ -q`

## Escalation

If you encounter:
- **Architecture or cross-plane contract changes**: Consult [ARCHITECTURE_EXCALIDRAW_MERMAID.md](ARCHITECTURE_EXCALIDRAW_MERMAID.md) and `harnesses/playbooks/new-feature-blueprint.md`, otherwise ask user
- **Model licensing or cloud GPU budget questions**: Check `configs/models.yaml` and [Phase3_implemn.md](Phase3_implemn.md) §1–§2 (`hunyuan15` must stay `enabled: false` by default), otherwise ask user
- **Repeated test failures**: Update `progress.md` and `session-handoff.md`, do not weaken tests or thresholds, and flag for human review
- **Scope ambiguity**: Re-read `feature_list.json` for dependencies and definition of done

