# AGENTS.md — Go Control Plane Harness (`cmd/m2a` & `internal/`)

Specialized harness for the **Go Control Plane** of Manhwa → Anime (M2A): CLI entrypoint (`cmd/m2a`), YAML config & tier loader (`internal/config`), canonical state models (`internal/model`), dual-mode HTTP/CLI worker client (`internal/client`), QC threshold evaluator & fallback router (`internal/qc`), and the 12-stage deterministic DAG engine (`internal/pipeline`).

## Startup Workflow

Before writing code:

1. **Confirm working directory** with `pwd` (repository root `/run/media/hihry/OS/hihry/Ani`)
2. **Read this file** and root [AGENTS.md](../../AGENTS.md) completely
3. **Read project docs if present** ([README.md](../../README.md), [ARCHITECTURE_EXCALIDRAW_MERMAID.md](../../ARCHITECTURE_EXCALIDRAW_MERMAID.md), [optim.md](../../optim.md))
4. **Run `./harnesses/go-control-plane/init.sh`** to verify Go packages build, vet, and pass contract tests
5. **Read root `feature_list.json` and `progress.md`** to see current feature state
6. **Review recent commits** with `git log --oneline -5`

If baseline verification is failing, repair that first before adding new scope.

## Scope & Ownership Boundary

- **Owned Paths**: `cmd/m2a/`, `internal/config/`, `internal/model/`, `internal/client/`, `internal/qc/`, `internal/pipeline/`, `go.mod`, `go.sum`
- **Cross-Plane Contract Files**: Whenever `internal/model/state.go` or `internal/config/config.go` changes, you MUST verify parity with `src/m2a/state.py` and `configs/*.yaml`
- **Out of Scope**: Python AI inference logic in `workers/` (unless updating CLI/HTTP transport payloads in lockstep)

## Working Rules & Go Plane Invariants

- **One feature at a time**: Pick exactly one unfinished feature from root `feature_list.json`
- **Stay in scope**: Only modify Go control-plane packages and their paired contract tests
- **Verification required**: Run `./harnesses/go-control-plane/init.sh` and `./init.sh` before claiming done
- **Update artifacts**: Before ending session, update root `progress.md`, `feature_list.json`, and `session-handoff.md`
- **Leave clean state**: Next session must be able to run `./harnesses/go-control-plane/init.sh` immediately
- **Atomic Checkpointing**: `RunState.Save()` must always write via temp file + atomic rename and call `NormalizePaths()` (`/` separators)
- **Fail-Closed Nodes (`internal/pipeline/nodes.go`)**: Never invent fake QC scores (`0.92/0.08/0.85`) or silent dummy panels when `WorkerClient` returns an error
- **Per-Operation Timeouts (`internal/client/worker_client.go`)**: Use stage-appropriate timeouts (async job polling `/render/jobs` for GPU I2V; bounded synchronous timeouts for CPU lanes) and never double-submit non-idempotent GPU jobs on timeout

## Required Artifacts

- `../../feature_list.json` — Root feature tracker (single source of truth across all planes)
- `../../progress.md` — Root session continuity log
- `init.sh` — Go plane verification script (`go vet`, `go test`, `go build`, `pytest -m contract`)
- `../../session-handoff.md` — Root handoff notes for the next session

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] Target behavior is implemented in `cmd/m2a` or `internal/` with unit test coverage (`*_test.go`)
- [ ] Required verification actually ran (`./harnesses/go-control-plane/init.sh` and `./init.sh`)
- [ ] Evidence recorded in root `feature_list.json` and `progress.md`
- [ ] Repository remains restartable from standard startup path

## End of Session

Before ending a session:

1. Update root `progress.md` with current state and verification command output
2. Update root `feature_list.json` with new feature status and evidence
3. Record any unresolved risks or blockers in root `session-handoff.md`
4. Commit with descriptive message once work is in safe state
5. Leave repo clean enough for next session to run `./harnesses/go-control-plane/init.sh` immediately

## Verification Commands

```bash
# Go control-plane verification
./harnesses/go-control-plane/init.sh

# Full master verification before claiming done
./init.sh
```

Required checks:
- `go vet ./...`
- `go test ./...`
- `go build -o m2a ./cmd/m2a`
- `python3 -m pytest -m "contract" -q`

## Escalation

If you encounter:
- **Architecture decisions**: Consult [ARCHITECTURE_EXCALIDRAW_MERMAID.md](../../ARCHITECTURE_EXCALIDRAW_MERMAID.md), otherwise ask user
- **Unclear requirements**: Check [Phase3_implemn.md](../../Phase3_implemn.md) and [optim.md](../../optim.md), otherwise ask user
- **Repeated test failures**: Update progress, flag for human review
- **Scope ambiguity**: Re-read root `feature_list.json` for definition of done

