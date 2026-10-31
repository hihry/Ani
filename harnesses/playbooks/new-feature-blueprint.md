# New Feature Blueprint — Manhwa → Anime (M2A)

Use this step-by-step blueprint whenever adding **any new feature** to M2A so that cross-plane contracts (`Go ↔ Python`), quality gates, and tests never drift.

---

## Step 0: Register the Feature Before Coding

1. Add a single entry to the root `feature_list.json` (single source of truth across all planes):
   ```json
   {
     "id": "feat-010",
     "name": "<Short Concrete Feature Title>",
     "description": "<Exact observable behavior, inputs, outputs, and files touched>",
     "dependencies": ["feat-003"],
     "harness": "harnesses/<go-control-plane|python-worker|eval-bakeoff|api-ui>",
     "status": "in-progress",
     "evidence": ""
   }
   ```
2. Set `Active Feature` in `progress.md` and run `./init.sh --quick` to confirm a clean starting point.

---

## Recipe A: Adding or Modifying a `RunState` / `PanelJSON` Schema Field

Because Go (`cmd/m2a`, `internal/`) and Python (`workers/`, `src/m2a/`, `api/`, `ui/`) read and write the same `runs/<run_id>/state.json`, schema changes MUST happen in lockstep:

- [ ] **1. Go Struct (`internal/model/state.go`)**: Add the field with explicit `json:"snake_case_name"` tag (use `omitempty` only if optional across legacy runs).
- [ ] **2. Python Pydantic Model (`src/m2a/state.py`)**: Add the matching field with identical type and default value so both Go-written and Python-written `state.json` files validate.
- [ ] **3. Path Normalization**: If the field stores a file path, add it to `RunState.NormalizePaths()` in `internal/model/state.go`.
- [ ] **4. Golden Contract Test**: Add/update the round-trip assertion in `tests/test_phase3.py` (`test_m3_m7_contract_and_tier_configs`), `tests/test_pipeline.py` (`test_go_state_json_interoperability`), and `internal/model/` tests.
- [ ] **5. Verify**: Run `./harnesses/go-control-plane/init.sh` (which runs `pytest -m contract`).

---

## Recipe B: Adding a New Pipeline Stage or Modifying the 12-Stage DAG

- [ ] **1. Stage Constant (`internal/model/state.go` & `src/m2a/state.py`)**: Register the stage enum/constant in both planes.
- [ ] **2. Go DAG Wiring (`internal/pipeline/engine.go` & `nodes.go`)**:
  - Register the stage transition in `Engine.Run()`.
  - Implement the handler in `internal/pipeline/nodes.go` with atomic `state.Save()` checkpointing and `StageHistory` recording.
  - **Fail-Closed Rule**: Never return `nil` with fake outputs when `WorkerClient` returns an error.
- [ ] **3. Worker Client (`internal/client/worker_client.go`)**:
  - Add the typed request/response struct and HTTP endpoint call (`POST /<stage>`) with stage-specific timeout.
  - Add the CLI fallback subcommand in `workers/pipeline_cli.py` (only for idempotent CPU operations; GPU jobs must use async `/render/jobs`).
- [ ] **4. Worker Endpoint (`workers/server.py`)**: Register the FastAPI route and coordinate with `ModelResidencyManager` if GPU VRAM is used.
- [ ] **5. Verify**: Run `./harnesses/go-control-plane/init.sh` and `./harnesses/python-worker/init.sh`.

---

## Recipe C: Adding a New AI Model Provider (Video / VLM / Inpaint / TTS)

- [ ] **1. Model Manifest (`configs/models.yaml` & `configs/video_models.yaml`)**:
  - Record exact HuggingFace `repo`, `revision` SHA, file `sha256`, `vram_gb`, and verified `license` (check regional exclusions like `excluded_regions: [EU, UK, KR]`).
- [ ] **2. ComfyUI Workflow Template (`workflows/comfyui/<model>_api.json`)** (for Video/I2V):
  - Export in ComfyUI **API format** (not UI graph format).
  - Include all `REQUIRED_TAGS` validated by `workers/video/workflows.py` (`M2A_IMAGE`, `M2A_POS`, `M2A_NEG`, `M2A_SEED`, `M2A_LATENT`, `M2A_SAMPLER`, `M2A_OUTPUT`).
- [ ] **3. Provider Registration (`workers/video/provider.py` or `src/m2a/providers/`)**:
  - Implement real execution + conform to target FPS/aspect ratio (`workers/video/conform.py`).
  - Record GPU seconds in `CostLedger` and cache key in `ClipCache`.
- [ ] **4. Tier Configuration (`configs/tiers/{t0,t1s,t1,t2}.yaml`)**:
  - Wire the model into the appropriate VRAM tier and fallback ladder.
- [ ] **5. Verify**: Add workflow injection + `FakeComfyUI` tests in `tests/test_phase3.py` and run `./harnesses/python-worker/init.sh` (plus `./harnesses/eval-bakeoff/probe-models.sh --tier <tier>` on live GPU).

---

## Recipe D: Adding a New QC Metric or Calibration Rule

- [ ] **1. Metric Computation (`workers/qc_metrics.py`)**:
  - Implement the metric on decoded clip frames. Return `-1.0` (not measured) if optional heavy weights are absent — never return a fake passing constant.
- [ ] **2. Schema & Evaluator (`internal/model/state.go`, `src/m2a/state.py`, `internal/qc/evaluator.go`)**:
  - Add the score field to `QCScores` and threshold check to `EvaluateClip()` per route (`i2v`, `parallax`, `kenburns`).
- [ ] **3. Adversarial Fixture (`tests/fixtures/qc_adversarial.py`)**:
  - Create a deterministic synthetic failure clip that triggers the new metric and assert in `tests/test_phase3.py` that `EvaluateClip` rejects it.
- [ ] **4. Calibrator (`src/m2a/eval/calibrate_qc.py`)**:
  - Include the metric in ROC/AUC + Youden's J threshold fitting.
- [ ] **5. Verify**: Run `./harnesses/eval-bakeoff/init.sh`.
