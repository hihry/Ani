# AGENTS.md — FastAPI Job Queue & Streamlit Review UI Harness (`api/` & `ui/`)

Specialized harness for the **API & Human Checkpoint Review Plane** of Manhwa → Anime (M2A): FastAPI job submission and checkpoint override API (`api/app.py`), background job poller (`api/worker.py`), and the Streamlit checkpoint editor for `CHECKPOINT_1` (narrative/dialogue/bbox edits) and `CHECKPOINT_2` (route & motion overrides) (`ui/streamlit_app.py`, `ui/pages/`).

## Startup Workflow

Before writing code:

1. **Confirm working directory** with `pwd` (repository root `/run/media/hihry/OS/hihry/Ani`)
2. **Read this file** and root [AGENTS.md](../../AGENTS.md) completely
3. **Read project docs if present** ([README.md](../../README.md), [optim.md](../../optim.md) A1/P0-07)
4. **Run `./harnesses/api-ui/init.sh`** to verify API and UI syntax and state/checkpoint tests pass
5. **Read root `feature_list.json` and `progress.md`** to see current feature state
6. **Review recent commits** with `git log --oneline -5`

If baseline verification is failing, repair that first before adding new scope.

## Scope & Ownership Boundary

- **Owned Paths**: `api/app.py`, `api/worker.py`, `ui/streamlit_app.py`, `ui/pages/`
- **Cross-Plane Contract Files**: Reads and writes `runs/<run_id>/state.json` using the canonical `RunState` schema (`src/m2a/state.py` ↔ `internal/model/state.go`)
- **Out of Scope**: Core CV/AI worker implementations in `workers/`

## Working Rules & API/UI Invariants

- **One feature at a time**: Pick exactly one unfinished feature from root `feature_list.json`
- **Stay in scope**: Only modify `api/` and `ui/` files and their tests
- **Verification required**: Run `./harnesses/api-ui/init.sh` and `./init.sh` before claiming done
- **Update artifacts**: Before ending session, update root `progress.md`, `feature_list.json`, and `session-handoff.md`
- **Leave clean state**: Next session must be able to run `./harnesses/api-ui/init.sh` immediately
- **Delegate Orchestration to Go Engine**: Do not add duplicate routing, QC, or I2V orchestration logic in `api/` or `ui/`; delegate to the Go `m2a` binary (`run` / `resume`) and read/update `runs/<run_id>/state.json` atomically (`feat-005` / `optim.md` A1 & P0-07)
- **Preserve Unknown/Future Fields on Checkpoint Save**: When `PUT /jobs/{id}/checkpoint` or `ui/streamlit_app.py` updates `panels` or `routes` in `state.json`, preserve `stage_history`, `cost_ledger`, `clip_candidates`, and normalized forward-slash paths

## Required Artifacts

- `../../feature_list.json` — Root feature tracker (single source of truth across all planes)
- `../../progress.md` — Root session continuity log
- `init.sh` — API/UI verification script (`compileall` + `pytest -m api_ui`)
- `../../session-handoff.md` — Root handoff notes for the next session

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] Target API or UI behavior is implemented cleanly without duplicating Go engine logic
- [ ] Required verification actually ran (`./harnesses/api-ui/init.sh` and `./init.sh`)
- [ ] Evidence recorded in root `feature_list.json` and `progress.md`
- [ ] Repository remains restartable from standard startup path

## End of Session

Before ending a session:

1. Update root `progress.md` with current state and verification command output
2. Update root `feature_list.json` with new feature status and evidence
3. Record any unresolved risks or blockers in root `session-handoff.md`
4. Commit with descriptive message once work is in safe state
5. Leave repo clean enough for next session to run `./harnesses/api-ui/init.sh` immediately

## Verification Commands

```bash
# Full API & UI verification (recommended)
./harnesses/api-ui/init.sh
```

Required checks:
- `python3 -m compileall -q api ui src/m2a`
- `python3 -m pytest -m "api_ui" -q`

## Escalation

If you encounter:
- **Architecture decisions**: Consult [ARCHITECTURE_EXCALIDRAW_MERMAID.md](../../ARCHITECTURE_EXCALIDRAW_MERMAID.md), otherwise ask user
- **Unclear requirements**: Check [Phase3_implemn.md](../../Phase3_implemn.md) CQ9, otherwise ask user
- **Repeated test failures**: Update progress, flag for human review
- **Scope ambiguity**: Re-read root `feature_list.json` for definition of done

