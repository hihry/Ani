# AGENTS.md — Python AI Worker & Video Plane Harness (`workers/`, `src/m2a/`, `workflows/comfyui/`)

Specialized harness for the **Python AI Worker Plane** of Manhwa → Anime (M2A): FastAPI worker service (`workers/server.py` `:8100`), CLI worker bridge (`workers/pipeline_cli.py`), strip ingestion (`workers/ingest.py`), 2D panel slicer (`workers/slicer.py`), VLM/CV understanding (`workers/understanding.py`), Ken Burns & Parallax renderers (`workers/render_*.py`), ComfyUI I2V provider (`workers/video/`, `workflows/comfyui/`), QC metrics v2 (`workers/qc_metrics.py`), and FFmpeg timeline/audio assembler (`workers/assembler.py`).

## Startup Workflow

Before writing code:

1. **Confirm working directory** with `pwd` (repository root `/run/media/hihry/OS/hihry/Ani`)
2. **Read this file** and root [AGENTS.md](../../AGENTS.md) completely
3. **Read project docs if present** ([README.md](../../README.md), [Phase3_implemn.md](../../Phase3_implemn.md), [optim.md](../../optim.md))
4. **Run `./harnesses/python-worker/init.sh`** to verify guards, Python syntax, and worker tests pass
5. **Read root `feature_list.json` and `progress.md`** to see current feature state
6. **Review recent commits** with `git log --oneline -5`

If baseline verification is failing, repair that first before adding new scope.

## Scope & Ownership Boundary

- **Owned Paths**: `workers/`, `src/m2a/state.py`, `src/m2a/config.py`, `src/m2a/providers/`, `workflows/comfyui/`, `configs/models.yaml`, `configs/video_models.yaml`
- **Cross-Plane Contract Files**: Any change to `src/m2a/state.py` or HTTP endpoint payloads in `workers/server.py` must be synchronized with `internal/model/state.go` and `internal/client/worker_client.go`
- **Out of Scope**: Go DAG orchestration logic in `internal/pipeline/` or Streamlit UI layout in `ui/`

## Working Rules & Python Worker Invariants

- **One feature at a time**: Pick exactly one unfinished feature from root `feature_list.json`
- **Stay in scope**: Only modify Python worker modules, ComfyUI workflow JSONs, and their tests
- **Verification required**: Run `./harnesses/python-worker/init.sh` and `./init.sh` before claiming done
- **Update artifacts**: Before ending session, update root `progress.md`, `feature_list.json`, and `session-handoff.md`
- **Leave clean state**: Next session must be able to run `./harnesses/python-worker/init.sh` immediately
- **Zero Ground-Truth Leakage**: `workers/` (excluding `eval_*.py`), `src/m2a/nodes/`, and `src/m2a/providers/` must NEVER read `data/eval/ground_truth.json` (enforced by `./harnesses/guards.sh` and `test_zero_ground_truth_leakage_in_production_code`)
- **Honest `route_used` & Hard-Fail on Frame Mismatch**: `assemble_from_run_state` must reuse accepted `i2v` clips when frame count matches (`abs(frames - expected) <= 2`) and raise a `ValueError` rather than silently re-rendering parallax if an accepted I2V clip is missing or mismatched
- **ComfyUI Workflow Tags**: Every workflow JSON in `workflows/comfyui/*_api.json` must preserve all `REQUIRED_TAGS` (`M2A_IMAGE`, `M2A_POS`, `M2A_NEG`, `M2A_SEED`, `M2A_LATENT`, `M2A_SAMPLER`, `M2A_OUTPUT`)
- **Torch-Free Core Worker**: Keep `workers/requirements.txt` free of heavy `torch` dependencies so the worker runs cleanly on Python 3.15 while talking to ComfyUI (Python 3.12 venv) over HTTP `:8188`

## Required Artifacts

- `../../feature_list.json` — Root feature tracker (single source of truth across all planes)
- `../../progress.md` — Root session continuity log
- `init.sh` — Python worker verification script (`guards.sh` + `compileall` + `pytest -m worker`)
- `../../session-handoff.md` — Root handoff notes for the next session

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] Target behavior is implemented in `workers/` or `src/m2a/` with zero silent stubs
- [ ] Required verification actually ran (`./harnesses/python-worker/init.sh` and `./init.sh`)
- [ ] Evidence recorded in root `feature_list.json` and `progress.md`
- [ ] Repository remains restartable from standard startup path

## End of Session

Before ending a session:

1. Update root `progress.md` with current state and verification command output
2. Update root `feature_list.json` with new feature status and evidence
3. Record any unresolved risks or blockers in root `session-handoff.md`
4. Commit with descriptive message once work is in safe state
5. Leave repo clean enough for next session to run `./harnesses/python-worker/init.sh` immediately

## Verification Commands

```bash
# Full Python worker verification (recommended)
./harnesses/python-worker/init.sh

# Fast syntax + Phase 3 worker smoke verification (iteration only; not sufficient for done)
./harnesses/python-worker/init.sh --quick
```

Required checks:
- `./harnesses/guards.sh`
- `python3 -m compileall -q workers src/m2a tests`
- `python3 -m pytest -m "worker" -q`

## Escalation

If you encounter:
- **Architecture decisions**: Consult [Phase3_implemn.md](../../Phase3_implemn.md) and [optim.md](../../optim.md), otherwise ask user
- **Unclear requirements**: Check `configs/models.yaml` and `configs/video_models.yaml`, otherwise ask user
- **Repeated test failures**: Update progress, flag for human review
- **Scope ambiguity**: Re-read root `feature_list.json` for definition of done

