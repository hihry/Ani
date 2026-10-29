# Session Progress Log — Manhwa → Anime (M2A)

## Current State

**Last Updated:** 2026-10-28 23:12
**Session ID:** refactor-arch-debt-005-006-010
**Active Feature:** feat-004 - Phase 3 (M6-M7): Cloud GPU Video Bake-off Execution & Empirical QC Calibration

## Status

### What's Done

- [x] **feat-001**: Phase 0 & 1 Core Foundations, 2D Slicer (`F1=1.0000`, `Mean IoU=0.9968`), and CPU Motion Comic Baseline (`go test ./...` 5 packages with tests / 29 Go tests PASS; `pytest tests/test_pipeline.py` 35 passed).
- [x] **feat-002**: Phase 2 Narrative Scene Understanding, Character Bible & LaMa/Telea Clean Plates. Removed `_load_eval_ground_truth` and `M2A_EVAL_ORACLE` ground-truth loading from `workers/understanding.py` (and unused import in `workers/pipeline_cli.py`); enforced zero ground-truth leakage via `./harnesses/guards.sh` and `test_zero_ground_truth_leakage_in_production_code`.
- [x] **feat-003**: Phase 3 (M0–M5) ComfyUI I2V workflows (all 6 API templates), `ComfyClient` + `ClipCache`, aspect-safe `conform` (with sub-pixel continuous `Image.Transform.EXTENT` tail drift), optical-flow/LPIPS/OCR `qc_metrics.py`, bake-off + ROC/Youden calibration infrastructure, and real-model GPU probe + anti-slideshow detector (`src/m2a/eval/probe_models.py` & `harnesses/eval-bakeoff/probe-models.sh`; `tests/test_phase3.py` 12/12 passing).
- [x] **feat-005**: Unify Job Queue API & Streamlit UI onto Canonical Go DAG Engine (`optim.md` A1 & P0-07). Added `--run-dir` and `config.snapshot.yaml` loading to `cmd/m2a/main.go`, replaced the duplicate Python `PipelineGraph` stage loop in `src/m2a/graph.py` with canonical Go `m2a` CLI delegation (`run_go_engine`, `resume_go_engine`, `inspect_go_engine`, `save_checkpoint_edits`), removed the silent `i2v -> parallax` fallback in `src/m2a/nodes/generate.py`, and updated `api/app.py`, `api/worker.py`, `ui/streamlit_app.py`, and `src/m2a/cli.py`.
- [x] **feat-006**: Streaming FFmpeg Assembly & Bounded Peak RSS Memory (`optim.md` P1-01 & P1-02). Added `StreamingMjpegAviWriter`, `count_video_frames`, `iter_video_frames`, `create_subtitle_overlay`, `_encode_mp4_via_ffmpeg_filtergraph`, and `assemble_clips_streaming` in `workers/assembler.py`, and refactored `assemble_from_run_state` in `workers/pipeline_cli.py` to validate clip frame counts in O(1) memory and stream crossfade assembly without decoding all panel clips into `clips: List[List[Image.Image]]`.
- [x] **feat-010**: QC Terminal Fallback `needs_review` Flag & Real Local Model Residency (`optim.md` P0-02 & §3). Added `NeedsReview` (`needs_review`) to `PanelJSON` in `internal/model/panel.go` and `src/m2a/state.py`, updated `QCNode` in `internal/pipeline/nodes.go` to mark `p.NeedsReview = true` and record a `state.Errors` warning when terminal `kenburns` fails QC thresholds, and wired real loader/unloader lifecycle management into `ModelResidencyManager` in `workers/server.py`.
- [x] **Harness Audit Remediation (C1–C4, H1–H6, M1–M4)**:
  - **C1 (`probe-models.sh` & Slideshow Detector)**: Added `src/m2a/eval/probe_models.py` and `harnesses/eval-bakeoff/probe-models.sh` outside CPU `init.sh`. Verified `./harnesses/eval-bakeoff/probe-models.sh --tier t1s --files-only` (`exit 0`, verifying all 3 `wan22-5b-gguf-q4` files + SHA-256 on `/run/media/hihry/OS/m2a_models`), `./harnesses/eval-bakeoff/probe-models.sh --tier t1s` (`exit 1` fail-closed when ComfyUI `:8188` is offline), and `./harnesses/eval-bakeoff/probe-models.sh --check-run runs/solo_gpu_01` (`exit 1` catching `2/2` `i2v` panels degraded to `kenburns`/`parallax`).
  - **C2 & M1/M2 (Zero Ground-Truth Leakage & `harnesses/guards.sh`)**: Deleted `_load_eval_ground_truth` from `workers/understanding.py`, added `harnesses/guards.sh` (checking `go >= 1.26`, `python3`, `ffmpeg`, zero `ground_truth.json` references in production code, zero tracked model weights, and `hunyuan15.enabled == false`), and wired it into `init.sh` and `validate-all.sh`.
  - **C3 (Honest Evidence & `optim.md` Tracking)**: Fixed test counts (`45 passed` across `test_pipeline.py` [33] and `test_phase3.py` [12]; 5 Go packages / 29 Go tests), fixed `test_m3_m7_contract_and_tier_configs` / `test_go_state_json_interoperability` references in `harnesses/playbooks/new-feature-blueprint.md`, and tracked open `optim.md` items in `feat-005`, `feat-006`, and `feat-010`.
  - **C4 (Pytest Markers)**: Registered `--strict-markers` (`contract`, `worker`, `eval_qc`, `eval`, `api_ui`, `api`, `gpu`) in `pyproject.toml`, tagged all 45 tests, expanded `test_job_api_lifecycle`, and updated all `init.sh` scripts to use `-m <marker>`.
  - **H1–H6 & M3–M4**: Removed 12 duplicate state files from `harnesses/*/` so state lives only at root; turned `CLAUDE.md` and `GEMINI.md` into short pointers to `AGENTS.md`; added auto-generation of `data/eval/sample_page_*.png` to `init.sh`; unified `Makefile` (`BIN ?= m2a`) and `init.sh` (`go build -o m2a ./cmd/m2a`); and made `harnesses/validate-all.sh` portable.

### What's In Progress

- [ ] **feat-004**: Provision local Python 3.12 ComfyUI venv, copy `t1s` weights off `rclone` FUSE mount, and execute GPU bake-off (`M6`–`M7`)
  - Blockers: Local ComfyUI Python 3.12 venv needs `./scripts/comfyui/setup_local.sh` execution with network access (`:8188` is currently offline); `t1`/`t2` heavy models require a 24–48 GB cloud GPU.

### What's Next

1. Copy the `11.6 GB` `t1s` weights (`Wan2.2-TI2V-5B-Q4_K_M.gguf`, `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `wan2.2_vae.safetensors`) from `/home/hihry/google_ai_cloud/m2a_models` onto local `/run/media/hihry/OS/m2a_models_local/` (`146 GB` free) to eliminate `rclone` FUSE stalls (`RC6`), run `./scripts/comfyui/setup_local.sh`, and verify live I2V rendering with `./harnesses/eval-bakeoff/probe-models.sh --tier t1s`.
2. Or proceed with Phase 4 / Phase 6 features: `feat-007` (Multi-Speaker Multilingual TTS & Character Voice Locking, `harnesses/python-worker`) or `feat-009` (Multi-Chapter Batch Processing & Episode Stitching CLI, `harnesses/go-control-plane` — now unblocked since `feat-005` and `feat-006` are done).

## Blockers / Risks

- [ ] **ComfyUI Venv Missing (`:8188` Offline)**: `/run/media/hihry/OS/venvs/` is empty (`/run/media/hihry/OS/venvs/comfyui` does not exist) and host Python 3.15 has no PyTorch wheels.
- [ ] **`rclone` Google Drive FUSE Mount Stalls (`RC6`)**: `/run/media/hihry/OS/m2a_models` and `/home/hihry/ComfyUI` symlink to `/home/hihry/google_ai_cloud/`, causing `32–50s` OpenCV/FFmpeg stream timeouts and `900s` ComfyUI render timeouts when read directly over FUSE.

## Decisions Made

- **Canonical Go Engine Bridge (`feat-005`)**: `src/m2a/graph.py` now delegates `run`, `resume`, and `inspect` to the compiled Go `m2a` binary (`--run-dir` + `config.snapshot.yaml`) and syncs CP1 (`understanding/<panel_id>.json`) and CP2 (`routes/routing_plan.json`) edits on disk so Go's `Engine.Resume` picks them up identically across CLI, FastAPI (`api/`), and Streamlit (`ui/`).
- **Bounded-Memory Streaming Assembly (`feat-006`)**: `assemble_from_run_state` validates clip frame counts via RIFF header scanning (`count_video_frames`) and streams crossfade transitions (`StreamingMjpegAviWriter` + FFmpeg `-filter_complex` `overlay`/`xfade`), holding at most `2 * transition_frames` full-resolution frames in RAM.
- **Terminal QC Review Flag & Model Residency Lifecycle (`feat-010`)**: `QCNode` sets `PanelJSON.NeedsReview = true` when terminal `kenburns` fallback still fails QC thresholds, and `ModelResidencyManager` manages registered loader/unloader hooks, weight path resolution, garbage collection, and CUDA/ComfyUI cache eviction.

## Files Modified This Session

- `cmd/m2a/main.go`, `internal/model/panel.go`, `internal/pipeline/nodes.go`, `internal/pipeline/engine_test.go` - Added `--run-dir` / `config.snapshot.yaml` support to `m2a run` and `NeedsReview` terminal QC fallback flagging in `QCNode`
- `src/m2a/graph.py`, `src/m2a/nodes/generate.py`, `src/m2a/config.py`, `src/m2a/state.py`, `src/m2a/cli.py`, `api/app.py`, `api/worker.py`, `ui/streamlit_app.py` - Unified API, UI, and Python CLI onto canonical Go `m2a` engine and removed silent `i2v -> parallax` fallback
- `workers/assembler.py`, `workers/pipeline_cli.py`, `workers/server.py` - Implemented streaming MJPEG AVI writer, O(1) frame counting, FFmpeg `filter_complex` crossfade/overlay assembly, and `ModelResidencyManager` weight lifecycle management
- `tests/test_pipeline.py`, `AGENTS.md`, `feature_list.json`, `progress.md`, `session-handoff.md` - Added regression tests for `feat-005`, `feat-006`, and `feat-010` and updated verification evidence

## Evidence of Completion

- [x] Machine guards pass: `./harnesses/guards.sh` -> `ALL 4 MACHINE INVARIANT GUARDS PASSED` (`exit 0`)
- [x] Full master suite passes: `./init.sh` -> `go vet ./...` (`exit 0`), `go build -o m2a ./cmd/m2a` (`exit 0`), `go test ./...` (5 packages with tests / 29 Go tests PASS), `compileall` (`exit 0`), `python3 -m pytest tests/ -q` (`47 passed in 46.35s`)
- [x] Domain harnesses pass with strict markers:
  - `./harnesses/go-control-plane/init.sh` -> `10 passed, 37 deselected` (`-m contract`)
  - `./harnesses/python-worker/init.sh` -> `42 passed, 5 deselected` (`-m worker`)
  - `./harnesses/eval-bakeoff/init.sh` -> `F1=1.0000, Mean IoU=0.9968` + `16 passed, 31 deselected` (`-m eval_qc`)
  - `./harnesses/api-ui/init.sh` -> `8 passed, 39 deselected` (`-m api_ui`)
- [x] Real-model probe & slideshow detector verified:
  - `./harnesses/eval-bakeoff/probe-models.sh --tier t1s --files-only` -> `exit 0` (3 `wan22-5b-gguf-q4` files + SHA-256 verified)
  - `./harnesses/eval-bakeoff/probe-models.sh --tier t1s` -> `exit 1` (fails closed because ComfyUI `:8188` is offline)
  - `./harnesses/eval-bakeoff/probe-models.sh --check-run runs/solo_dramatic_01` -> `exit 0` (passes with `3/3` valid I2V clips)
  - `./harnesses/eval-bakeoff/probe-models.sh --check-run runs/solo_gpu_01` -> `exit 1` (detects `2/2` `i2v` panels degraded to `kenburns`/`parallax`)
- [x] Harness validation passes: `./harnesses/validate-all.sh` -> `exit 0` (`100/100` root score + 4 domain harnesses + zero duplicate state files + machine guards)

## Notes for Next Session

Start by running `./init.sh --quick` (`~4s` smoke check; always run full `./init.sh` before claiming any feature `done`). Use `./harnesses/eval-bakeoff/probe-models.sh` whenever verifying live GPU I2V readiness or auditing `runs/<id>` for slideshow fallback.
