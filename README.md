# Manhwa → Anime (M2A)

Manhwa → Anime (M2A) is an end-to-end multimodal pipeline that transforms static vertical-scroll manhwa webtoon strips into animated 9:16 portrait anime video sequences with synthesized voice acting, burned anime subtitles, and dynamic camera motion. Built with a **Hybrid Go + Python Backend**, M2A pairs a deterministic compiled **Go Control Plane** (`cmd/m2a`, `internal/`) with a modular **Python AI Worker Plane** (`workers/`, `src/m2a/`) to automate vertical strip stitching, 2D panel segmentation, narrative scene understanding, speech bubble inpainting, shot routing, video generation, automatic quality control (QC), TTS audio synthesis, and FFmpeg timeline composition.

## Architecture (v0.2 Hybrid Go + Python)

The system separates orchestration from AI/CV execution across two planes that read and write a single unified `RunState` contract (`runs/<run_id>/state.json`):

```text
+-----------------------------------------------------------------------------------+
|                                 Input Manhwa Strip                                |
|                        (2-3 vertical webtoon pages: .png/.jpg)                    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Go Control Plane (cmd/m2a & internal/)                     |
|  - CLI (run, resume, inspect) + FastAPI Job Queue API (api/app.py) + Streamlit UI |
|  - Config Loader (configs/default.yaml + tiers/t0.yaml, t1.yaml, t2.yaml)         |
|  - Deterministic 12-Stage DAG Engine (internal/pipeline/engine.go + nodes.go)     |
|  - Dual-Mode WorkerClient (internal/client: HTTP :8100 first, CLI fallback)       |
|  - Auto-QC Gate & Route Degrader (internal/qc: i2v -> parallax -> kenburns)       |
+-----------------------------------------------------------------------------------+
            |                                                   |
            | HTTP (:8100) / CLI Fallback                       | Atomic Checkpoints
            v                                                   v
+-----------------------------------------+     +-----------------------------------+
|     Python AI Worker Plane (:8100)      |     |    Artifact & State Store         |
|   (workers/server.py + pipeline_cli.py) |     |      (runs/<run_id>/)             |
+-----------------------------------------+     +-----------------------------------+
| POST /ingest      Strip stitch (800px)  |     | state.json       Unified RunState |
| POST /segment     2D Gutter + Magiv3    |     | stitched.png     Normalized strip |
| POST /understand  Qwen3-VL + Char Bible |     | panels/          Panel crops      |
| POST /inpaint     LaMa / Telea Clean    |     | understanding/   PanelJSON + Bible|
| POST /render      I2V / Parallax / KB   |     | clean_plates/    Bubble-free art  |
| POST /tts         MeloTTS / WAV Synth   |     | clips/           Rendered clips   |
| POST /assemble    FFmpeg MP4/AVI/WebP   |     | audio/           Dialogue WAVs    |
+-----------------------------------------+     | output/          final_anime.mp4  |
                                                +-----------------------------------+
```

### 12-Stage End-to-End Pipeline Flow
1. **`INGEST`**: Validates source images, copies to `pages/`, normalizes width to `800px`, and stitches `stitched.png` (`workers/ingest.py`).
2. **`SEGMENT`**: Runs 2D horizontal + vertical gutter slicing (`workers/slicer.py`, `F1 = 1.0000`, `Mean IoU = 0.9968`) and saves `panels/panel_00X.png`.
3. **`UNDERSTAND`**: Extracts shot framing, action summary, dialogue lines, SFX, character bible (`understanding/character_bible.json`), and motion prompts (`PanelJSON`).
4. **`CHECKPOINT_1`**: Human review checkpoint for `PanelJSON` text, speakers, and bounding boxes (`PUT /jobs/{id}/checkpoint` or `--auto-approve`).
5. **`CLEAN_PLATES`**: Creates speech bubble masks and inpaints text regions via LaMa / OpenCV Telea into `clean_plates/panel_00X_clean.png`.
6. **`ROUTE`**: Assigns each panel to `i2v`, `parallax`, or `kenburns` using shot framing, visual motion intensity, camera motion taxonomy, and GPU budget (`routes/routing_plan.json`).
7. **`CHECKPOINT_2`**: Human review checkpoint for animation route overrides before GPU rendering.
8. **`GENERATE`**: Renders per-panel video clips (`clips/panel_00X_clip_01.avi`) from clean plates using P10 aspect-safe Ken Burns (blurred-fill letterbox for wide panels), occlusion-filled 2.5D Parallax, or ComfyUI I2V (`Wan 2.2`, `HunyuanVideo 1.5`, `LTX-2.3`).
9. **`QC`**: Evaluates every clip (`internal/qc/evaluator.go`) on Identity (`>= 0.85`), Flicker (`<= 0.15`), Text Leakage (`0`), and Motion Sanity; automatically degrades failed clips (`i2v -> parallax -> kenburns`) and re-renders.
10. **`TTS`**: Synthesizes playable PCM WAV speech tracks (`audio/*.wav`) per dialogue line and expands `panel.DurationS = max(duration_s, speech_dur + 0.3s)`.
11. **`ASSEMBLE`**: Chains accepted clips with linear alpha crossfades, burns anime subtitles, writes `output/subtitles.srt` and `output/timeline.json`, mixes WAV audio, and encodes `output/final_anime.mp4` (plus `.avi` and `.webp`).
12. **`DONE`**: Closes `StageHistory` timestamps, records `CostLedger`, and saves final `state.json`.

## Quick Start

### Prerequisites
- [Go 1.26+](https://go.dev/)
- [Python 3.10+](https://www.python.org/)
- [FFmpeg](https://ffmpeg.org/) (installed and available in `PATH`)

### 1. Install Worker Dependencies
```bash
make worker-install
```

### 2. Optional: Start the Persistent Python AI Worker Server
```bash
make worker
```
The FastAPI worker server listens at `http://127.0.0.1:8100` with `ModelResidencyManager` keeping at most one heavy GPU model resident in VRAM. If `:8100` is not running, the Go `WorkerClient` automatically falls back to invoking `python workers/pipeline_cli.py`.

### 3. Build & Run the Go Orchestrator
```bash
# Build Go binary
go build -o m2a.exe ./cmd/m2a

# Run full pipeline (auto-approving checkpoints CP1 & CP2)
./m2a.exe run --pages data/eval/sample_page_1.png,data/eval/sample_page_2.png,data/eval/sample_page_3.png --tier t0 --lang en --auto-approve

# Or pause at CP1/CP2 and resume after editing state.json
./m2a.exe resume --run-id <run_id>
./m2a.exe inspect --run-id <run_id>
```

### 4. Run Tests & Evaluation Benchmarks
```bash
# Go unit & integration tests
go test ./... -v

# Python unit & end-to-end tests
python -m pytest tests/test_pipeline.py -v

# Phase 1 Segmentation F1 & IoU benchmark
python workers/eval_segmentation.py

# Phase 2 Understanding & OCR benchmark
python workers/eval_understanding.py

# Phase 3 ComfyUI setup, Video Bake-off & QC Calibration
./scripts/comfyui/setup_local.sh
python scripts/comfyui/download_models.py --tier t1 --dry-run
python -m m2a.eval.bakeoff --config configs/bakeoff/phase3.yaml --out runs/bakeoff_phase3
python -m m2a.eval.calibrate_qc --bakeoff runs/bakeoff_phase3/records.jsonl --out configs/qc/calibrated_t1.yaml
```

## Project Structure

```text
Ani/
├── cmd/m2a/main.go              # Go CLI entrypoint (run, resume, inspect, version)
├── internal/                    # Go Control Plane packages
│   ├── config/                  # YAML configuration loader & tier overlay validator
│   ├── model/                   # Canonical structs: PanelJSON, CharacterBible, RunState
│   ├── client/                  # Dual-mode HTTP (:8100) + CLI fallback WorkerClient
│   ├── qc/                      # Quality control threshold evaluator & fallback router
│   └── pipeline/                # Go DAG Engine (engine.go) & 9 stage handlers (nodes.go)
├── workers/                     # Python AI Worker Plane
│   ├── server.py                # FastAPI worker service (:8100) + ModelResidencyManager
│   ├── pipeline_cli.py          # CLI bridge for all 7 worker commands + state assembly
│   ├── ingest.py                # Vertical strip normalization & stitching with page offsets
│   ├── slicer.py                # 2D horizontal + vertical gutter panel detector (F1=1.00)
│   ├── render_kenburns.py       # P10 blurred-fill letterbox + cubic-eased Ken Burns
│   ├── render_parallax.py       # Occlusion-filled 2.5D depth parallax renderer
│   ├── assembler.py             # Bubble inpainter, WAV synth, QC metrics, MP4/AVI/WebP
│   └── eval_segmentation.py     # Segmentation IoU / Precision / Recall / F1 evaluator
├── src/m2a/                     # Interoperable Python package (state, config, graph, nodes)
├── api/                         # FastAPI Job Queue API (api/app.py) with CP1/CP2 hooks
├── ui/                          # Streamlit Checkpoint Review Editor (ui/streamlit_app.py)
├── configs/                     # default.yaml, models.yaml, tiers/{t0,t1,t2}.yaml
├── data/eval/                   # Synthetic manhwa pages + ground_truth.json + schema
├── tests/test_pipeline.py       # Python end-to-end pytest suite
└── Manhwa_to_Anime_Prototype_Proposal.pdf # v0.2 Architecture & Phase 0/1 Flaw Audit PDF
```

## Compute Tiers

| Tier | Hardware Target | Video Generation Model | Vision-Language Model (VLM) | TTS Audio Engine |
| :--- | :--- | :--- | :--- | :--- |
| **T0** | CPU Only (No GPU) | Aspect-Safe Ken Burns & 2.5D Parallax | Qwen3-VL-2B / CV Analyzer | Kokoro / Harmonic WAV Synth |
| **T1S** | 8 GB VRAM (RTX 3070/4060) | Wan2.2-Ti2V-5B GGUF Q4_K_M (`wan22-5b-gguf-q4`) | Qwen3-VL-8B (4-bit quantized) | MeloTTS / Kokoro |
| **T1** | 12–16 GB VRAM (RTX 3080/4070) | Wan2.2-Ti2V-5B FP8 (`wan22-ti2v-5b`, fallback: `ltx23-fp8`) | Qwen3-VL-8B (4-bit quantized) | MeloTTS (fallback: Kokoro) |
| **T2** | 24–48 GB VRAM (RTX 4090/A5000/Cloud) | Wan2.2-I2V-A14B FP8 / LightX2V 4-Step LoRA | Qwen3-VL-8B (FP16) | MeloTTS / Chatterbox |

## License

This project is licensed under the Apache 2.0 License.
