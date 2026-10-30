# Session Handoff — Manhwa → Anime (M2A)

## Current Objective

- Goal: Complete architectural refactoring across `feat-005` (unify API/UI/Python CLI onto canonical Go `m2a` DAG engine and remove silent `i2v -> parallax` fallback), `feat-006` (streaming FFmpeg/MJPEG assembly with bounded peak RSS memory), and `feat-010` (QC terminal fallback `needs_review` flag + real `ModelResidencyManager` lifecycle).
- Current status: `feat-005`, `feat-006`, and `feat-010` are all `done` and verified (`47 passed` in pytest, 5 Go test packages / 29 Go tests passing, `./harnesses/guards.sh` passing, `./harnesses/validate-all.sh` passing). `feat-004` remains `in-progress` pending local ComfyUI venv setup or cloud GPU execution; `feat-007` and `feat-009` are now unblocked on CPU.
- Branch: `Harness`

## Completed This Session

- [x] **`feat-005` — Unify Job Queue API & Streamlit UI onto Canonical Go DAG Engine (`optim.md` A1 & P0-07)**: Added `--run-dir` and `config.snapshot.yaml` support to `cmd/m2a/main.go`, replaced the duplicate Python stage machine in `src/m2a/graph.py` with `run_go_engine`/`resume_go_engine`/`inspect_go_engine`/`save_checkpoint_edits`, removed silent `i2v -> parallax` fallback in `src/m2a/nodes/generate.py`, and updated `api/app.py`, `api/worker.py`, `ui/streamlit_app.py`, and `src/m2a/cli.py`.
- [x] **`feat-006` — Streaming FFmpeg Assembly & Bounded Peak RSS Memory (`optim.md` P1-01 & P1-02)**: Added `StreamingMjpegAviWriter`, `count_video_frames`, `iter_video_frames`, `create_subtitle_overlay`, `_encode_mp4_via_ffmpeg_filtergraph`, and `assemble_clips_streaming` in `workers/assembler.py`, and refactored `assemble_from_run_state` in `workers/pipeline_cli.py` to validate clip frame counts in O(1) memory and stream crossfade assembly without decoding all panel clips into `clips: List[List[Image.Image]]`.
- [x] **`feat-010` — QC Terminal Fallback `needs_review` Flag & Real Local Model Residency (`optim.md` P0-02 & §3)**: Added `NeedsReview` (`needs_review`) to `PanelJSON` in `internal/model/panel.go` and `src/m2a/state.py`, updated `QCNode` in `internal/pipeline/nodes.go` to set `p.NeedsReview = true` and append a warning to `state.Errors` when terminal `kenburns` fails QC thresholds, and wired loader/unloader callbacks, weight path resolution, and GC/CUDA/ComfyUI eviction into `ModelResidencyManager` (`workers/server.py`).

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Machine Invariant Guards | `./harnesses/guards.sh` | PASS (exit 0) | `go1.27.3`, `Python 3.15.1`, `ffmpeg`, zero GT leakage, zero tracked weights, `hunyuan15.enabled == false` |
| Master Harness (Full) | `./init.sh` | PASS (exit 0) | `guards.sh` + `go vet` + `go build -o m2a` + `go test ./...` (5 pkgs / 29 tests) + `compileall` + `pytest tests/ -q` (`47 passed in 46.35s`) |
| Go Control Plane Harness | `./harnesses/go-control-plane/init.sh` | PASS (exit 0) | `go vet` + `go test` + `go build` + `pytest -m contract` (`10 passed, 37 deselected`) |
| Python Worker Harness | `./harnesses/python-worker/init.sh` | PASS (exit 0) | `guards.sh` + `compileall` + `pytest -m worker` (`42 passed, 5 deselected`) |
| API & Review UI Harness | `./harnesses/api-ui/init.sh` | PASS (exit 0) | `compileall` + `pytest -m api_ui` (`8 passed, 39 deselected`) |
| All-Harness Structural Validator | `./harnesses/validate-all.sh --quick` | PASS (exit 0) | Root `100/100` + 4 domain harnesses + zero duplicate state files + `guards.sh` |

## Files Changed

- `cmd/m2a/main.go`, `internal/model/panel.go`, `internal/pipeline/nodes.go`, `internal/pipeline/engine_test.go`
- `src/m2a/graph.py`, `src/m2a/nodes/generate.py`, `src/m2a/config.py`, `src/m2a/state.py`, `src/m2a/cli.py`
- `api/app.py`, `api/worker.py`, `ui/streamlit_app.py`
- `workers/assembler.py`, `workers/pipeline_cli.py`, `workers/server.py`
- `tests/test_pipeline.py`, `AGENTS.md`, `feature_list.json`, `progress.md`, `session-handoff.md`

## Decisions Made

- Unified all Python orchestration callers (`api/`, `ui/`, `src/m2a/cli.py`, `src/m2a/graph.py`) onto the compiled Go `m2a` binary (`run`/`resume`/`inspect`) and `runs/<id>/state.json`, persisting CP1 edits to `understanding/<panel_id>.json` and CP2 edits to `routes/routing_plan.json`.
- Replaced in-memory `clips: List[List[Image.Image]]` assembly with header-level RIFF `00dc` frame counting (`count_video_frames`), streaming crossfade MJPEG AVI encoding (`StreamingMjpegAviWriter`), and FFmpeg `-filter_complex` (`overlay` + `xfade`) MP4 encoding.
- Added lockstep `NeedsReview` / `needs_review` to Go & Python `PanelJSON` and wired real model loader/unloader lifecycle hooks into `ModelResidencyManager`.

## Blockers / Risks

- **ComfyUI `:8188` Offline (`RC2`)**: `/run/media/hihry/OS/venvs/` is empty; `./scripts/comfyui/setup_local.sh` must be run to provision the Python 3.12 + PyTorch 2.6.0 ComfyUI environment.
- **`rclone` FUSE Mount Stalls (`RC6`)**: `/run/media/hihry/OS/m2a_models` points to `/home/hihry/google_ai_cloud/m2a_models`; copy the `11.6 GB` `t1s` weights to local disk before running live ComfyUI inference.

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json`, `progress.md`, and `session-handoff.md`.
3. Run `./init.sh --quick` for a fast smoke check (and full `./init.sh` before claiming any feature `done`).

## Recommended Next Step

- **For GPU bake-off (`feat-004`)**: Copy `t1s` weights off `/home/hihry/google_ai_cloud/m2a_models` to local `/run/media/hihry/OS/m2a_models_local/`, run `./scripts/comfyui/setup_local.sh`, start ComfyUI on `:8188`, and run `./harnesses/eval-bakeoff/probe-models.sh --tier t1s`.
- **For CPU features (`feat-007` or `feat-009`)**: Start with `feat-007` (`./harnesses/python-worker/init.sh`) for multi-speaker multilingual TTS & SFX bed mixing, or `feat-009` (`./harnesses/go-control-plane/init.sh`) for multi-chapter batch processing & episode stitching CLI.
