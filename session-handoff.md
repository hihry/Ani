# Session Handoff — Manhwa → Anime (M2A)

## Current Objective

- Goal: Remediate all critical, high, and medium issues from the M2A harness audit (`C1`–`C4`, `H1`–`H6`, `M1`–`M4`): add a real-model GPU probe & anti-slideshow detector, remove `ground_truth.json` leakage from production worker code, enforce machine guards in `init.sh`, register strict pytest domain markers across all 45 tests, consolidate duplicate state files to the repository root, and record accurate verification evidence.
- Current status: Root Master Harness + 4 Domain Sub-Harnesses + Machine Guards + Real-Model Probe active and verified (`45 passed` in pytest, 5 Go test packages / 29 Go tests passing, `./harnesses/guards.sh` passing, `./harnesses/validate-all.sh` passing). `feat-004` remains `in-progress` pending local ComfyUI venv setup or cloud GPU execution; open `optim.md` architectural items are explicitly tracked in `feat-005`, `feat-006`, and `feat-010`.
- Branch: `Harness`

## Completed This Session

- [x] **C1 — Real-Model GPU Probe & Anti-Slideshow Detector**: Created `src/m2a/eval/probe_models.py` and `harnesses/eval-bakeoff/probe-models.sh` outside CPU `init.sh`, and synced video model SHA-256 hashes in `configs/models.yaml` with `workflows/comfyui/manifest.yaml`.
- [x] **C2 & M1/M2 — Zero Ground-Truth Leakage & Machine Guards**: Removed `_load_eval_ground_truth`, `matches_gt_panel_crop`, and `M2A_EVAL_ORACLE` ground-truth loading from `workers/understanding.py` (and unused import in `workers/pipeline_cli.py`). Added `harnesses/guards.sh` (toolchain preflight for `go >= 1.26`, `python3`, `ffmpeg`; zero `ground_truth.json` in production code; zero tracked model weights; `hunyuan15.enabled == false`) and `test_zero_ground_truth_leakage_in_production_code`.
- [x] **C3 — Accurate Evidence & Honest `optim.md` Statuses**: Updated `feature_list.json`, `progress.md`, `session-handoff.md`, and `harnesses/playbooks/new-feature-blueprint.md` with exact test names (`test_m3_m7_contract_and_tier_configs`, `test_go_state_json_interoperability`), exact test counts (`45` pytest tests, `29` Go tests across `5` packages), and explicit tracking of open `optim.md` issues (`feat-005`, `feat-006`, `feat-010`).
- [x] **C4 — Pytest Domain Markers**: Configured `--strict-markers` in `pyproject.toml`, tagged all 45 tests in `tests/test_pipeline.py` (33) and `tests/test_phase3.py` (12), expanded `test_job_api_lifecycle`, and updated all `init.sh` scripts to filter via `-m <marker>`.
- [x] **H1–H6 & M3–M4 — State Consolidation, Prompt Deduplication & Clean-Checkout Bootstrap**: Deleted the 12 duplicate `feature_list.json`, `progress.md`, and `session-handoff.md` files from `harnesses/*/`, converted `CLAUDE.md` and `GEMINI.md` into short pointers to `AGENTS.md`, added auto-generation of `data/eval/sample_page_*.png` to `init.sh`, unified `go build -o m2a ./cmd/m2a` across `init.sh` and `Makefile`, and made `harnesses/validate-all.sh` portable.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Machine Invariant Guards | `./harnesses/guards.sh` | PASS (exit 0) | `go1.27.3`, `Python 3.15.1`, `ffmpeg`, zero GT leakage, zero tracked weights, `hunyuan15.enabled == false` |
| Master Harness (Full) | `./init.sh` | PASS (exit 0) | `guards.sh` + `go vet` + `go build -o m2a` + `go test ./...` (5 pkgs / 29 tests) + `compileall` + `pytest tests/ -q` (`45 passed`) |
| Master Harness (Quick Smoke) | `rm -f data/eval/sample_page_*.png && ./init.sh --quick` | PASS (exit 0) | Auto-generated missing `sample_page_1..3.png` + ran `-m contract` (`9 passed, 36 deselected`) |
| Go Control Plane Harness | `./harnesses/go-control-plane/init.sh` | PASS (exit 0) | `go vet` + `go test` + `go build` + `pytest -m contract` (`9 passed`) |
| Python Worker Harness | `./harnesses/python-worker/init.sh --quick` | PASS (exit 0) | `guards.sh` + `compileall` + `pytest tests/test_phase3.py -m worker` (`11 passed`) |
| Eval & Bake-off Harness | `./harnesses/eval-bakeoff/init.sh` | PASS (exit 0) | `eval_segmentation.py` (`F1=1.0000`, `IoU=0.9968`) + `pytest -m eval_qc` (`16 passed`) |
| API & Review UI Harness | `./harnesses/api-ui/init.sh` | PASS (exit 0) | `compileall` + `pytest -m api_ui` (`8 passed`) |
| Real-Model Files & SHA-256 Probe | `./harnesses/eval-bakeoff/probe-models.sh --tier t1s --files-only` | PASS (exit 0) | Verified all 3 `wan22-5b-gguf-q4` files + SHA-256 on `/run/media/hihry/OS/m2a_models` (with `rclone` mount warning) |
| Live ComfyUI `:8188` Probe | `./harnesses/eval-bakeoff/probe-models.sh --tier t1s` | FAIL-CLOSED (exit 1) | Expected: ComfyUI `:8188` is offline (`Connection refused`) |
| Anti-Slideshow Run Audit | `./harnesses/eval-bakeoff/probe-models.sh --check-run runs/solo_gpu_01` | FAIL-CLOSED (exit 1) | Expected: detected `2/2 (100.0%)` `i2v` panels (`panel_001`, `panel_018`) degraded to `kenburns`/`parallax` |
| All-Harness Structural Validator | `./harnesses/validate-all.sh` | PASS (exit 0) | Root `100/100` + 4 domain harnesses + zero duplicate state files + `guards.sh` |

## Files Changed

- `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `init.sh`, `Makefile`, `pyproject.toml`
- `feature_list.json`, `progress.md`, `session-handoff.md`
- `configs/models.yaml`
- `workers/understanding.py`, `workers/pipeline_cli.py`
- `src/m2a/eval/probe_models.py`, `harnesses/eval-bakeoff/probe-models.sh`
- `harnesses/guards.sh`, `harnesses/validate-all.sh`
- `harnesses/{go-control-plane,python-worker,eval-bakeoff,api-ui}/{AGENTS.md,init.sh}` (and removed duplicate `{feature_list.json,progress.md,session-handoff.md}`)
- `harnesses/playbooks/{new-feature-blueprint.md,memory-and-context-budget.md}`
- `tests/test_pipeline.py`, `tests/test_phase3.py`

## Decisions Made

- Consolidated `feature_list.json`, `progress.md`, and `session-handoff.md` to the repository root and enforced absence of domain duplicates in `harnesses/validate-all.sh`.
- Replaced `CLAUDE.md` and `GEMINI.md` with 3-line pointers to `AGENTS.md` to prevent 3x prompt token duplication.
- Built `harnesses/eval-bakeoff/probe-models.sh` (`src/m2a/eval/probe_models.py`) to gate live GPU claims on SHA-256 weight verification, `:8188` custom node readiness, a real 1-panel I2V render (`route_used == "i2v"`, `motion_score >= 0.02`, `frozen_ratio <= 5%`), and run-directory slideshow auditing (`--check-run`).

## Blockers / Risks

- **ComfyUI `:8188` Offline (`RC2`)**: `/run/media/hihry/OS/venvs/` is empty; `./scripts/comfyui/setup_local.sh` must be run to provision the Python 3.12 + PyTorch 2.6.0 ComfyUI environment.
- **`rclone` FUSE Mount Stalls (`RC6`)**: `/run/media/hihry/OS/m2a_models` points to `/home/hihry/google_ai_cloud/m2a_models`; copy the `11.6 GB` `t1s` weights to local disk before running live ComfyUI inference.
- **Open Architectural Debt (`optim.md`)**: Tracked in `feat-005` (duplicate Python `PipelineGraph`), `feat-006` (in-memory `assemble_from_run_state` frame list), and `feat-010` (terminal `kenburns` QC `needs_review` flag + `ModelResidencyManager` lifecycle).

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json`, `progress.md`, and `session-handoff.md`.
3. Run `./init.sh --quick` for a fast smoke check (and full `./init.sh` before claiming any feature `done`).

## Recommended Next Step

- **For GPU bake-off (`feat-004`)**: Copy `t1s` weights off `/home/hihry/google_ai_cloud/m2a_models` to local `/run/media/hihry/OS/m2a_models_local/`, run `./scripts/comfyui/setup_local.sh`, start ComfyUI on `:8188`, and run `./harnesses/eval-bakeoff/probe-models.sh --tier t1s`.
- **For CPU/architecture debt (`feat-005`, `feat-006`, `feat-010`)**: Start with `feat-005` (`./harnesses/api-ui/init.sh`) to unify `api/` and `ui/` onto the canonical Go `m2a` binary.
