# Manhwa → Anime (M2A)

Manhwa → Anime (M2A) is an end-to-end multimodal pipeline that transforms static manhwa webtoon strips into fully animated anime video sequences with synthesized voice acting, background music, and dynamic motion. Built with a high-performance Go orchestrator and a modular Python AI worker backend, M2A automates panel segmentation, visual narrative understanding via Vision-Language Models (VLMs), speech bubble inpainting, AI video generation, text-to-speech audio synthesis, and final video composition.

## Architecture

The system uses a hybrid architecture: a compiled Go orchestrator manages the pipeline lifecycle, state machine, file artifacts, and quality checks, delegating heavyweight AI inference tasks over HTTP to a lightweight FastAPI worker server.

```text
+-----------------------------------------------------------------------------+
|                               Input Manhwa Strip                            |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                            Go Backend Orchestrator                          |
|  - CLI Interface & Config Loader (default.yaml + compute tiers)             |
|  - DAG Stage Execution & Artifact Pipeline Management                       |
|  - Quality Control (QC) Gates & Validation Checks                           |
+-----------------------------------------------------------------------------+
          |                      ^                      |
          | HTTP Request         | JSON Response        | Composition
          v                      |                      v
+----------------------------------+          +-------------------------------+
|     Python AI Worker Server      |          |       FFmpeg Video Engine     |
|         (FastAPI :8100)          |          |  - Video stitching & crossfade|
+----------------------------------+          |  - Dialogue audio mixing      |
| /segment     OpenCV / MagiV3     |          |  - Subtitles & visual effects |
| /understand  Qwen3-VL VLM        |          +-------------------------------+
| /inpaint     LaMa inpainting     |                          |
| /tts         MeloTTS / Kokoro    |                          v
+----------------------------------+          +-------------------------------+
                                              |       Final Anime Video       |
                                              |        (.mp4, 9:16)           |
                                              +-------------------------------+
```

### Pipeline Stages
1. **Perception**: Detects panel bounding boxes and gutters using OpenCV slicing or MagiV3.
2. **Understanding**: Uses Vision-Language Models (e.g. Qwen3-VL) to extract scene descriptions, shot framing, camera motion prompts, and dialogue lines.
3. **Cleanup**: Identifies speech bubbles and inpaints background artwork using LaMa.
4. **Generation**: Synthesizes video clips via Image-to-Video models (Wan2.2, HunyuanVideo, Ken Burns) and generates dialogue audio with TTS (MeloTTS, Kokoro).
5. **Composition**: Combines animated video clips, audio tracks, and transitions into a coherent vertical anime video.
6. **Quality Control (QC)**: Validates character consistency, flicker thresholds, and text leakage before delivery.

## Quick Start

### Prerequisites
- [Go 1.26+](https://go.dev/)
- [Python 3.10+](https://www.python.org/)
- [FFmpeg](https://ffmpeg.org/) (installed and available in `PATH`)

### 1. Install Worker Dependencies
```bash
make worker-install
```
Or manually:
```bash
cd workers
pip install -r requirements.txt
cd ..
```

### 2. Start the Python AI Worker Server
```bash
make worker
```
The FastAPI worker server will start listening at `http://127.0.0.1:8100`.

### 3. Build & Run the Go Orchestrator
In a separate terminal:
```bash
# Build binary
make build

# Run pipeline on a sample manhwa image with Tier 1 profile
./m2a.exe run --pages sample.png --tier t1
```

## Project Structure

```text
c:\hihry\Ani\
├── cmd\
│   └── m2a\                 # Main Go CLI entrypoint
├── internal\                # Go backend internal packages
│   ├── config\              # Configuration parsing and tier overlays
│   ├── pipeline\            # Execution engine and DAG orchestration
│   ├── client\              # HTTP client for Python AI worker API
│   ├── models\              # Shared Go data structures
│   └── qc\                  # Quality control evaluators
├── workers\                 # Python AI worker microservice
│   ├── __init__.py
│   ├── server.py            # FastAPI service exposing AI inference endpoints
│   └── requirements.txt     # Python ML and API dependencies
├── configs\                 # YAML configuration profiles
│   ├── default.yaml         # Base pipeline configuration
│   └── tiers\               # Hardware tier overrides (T0, T1, T2)
│       ├── t0.yaml
│       ├── t1.yaml
│       └── t2.yaml
├── data\                    # Dataset and evaluation assets
│   └── eval\                # Ground truth schemas and benchmark annotations
│       └── ground_truth_schema.json
├── Makefile                 # Automation targets for build, test, and worker
├── .gitignore               # Ignored build artifacts, models, and virtual environments
└── README.md                # Project documentation
```

## Compute Tiers

M2A provides configurable tiers to match local or cloud GPU compute:

| Tier | Hardware Target | Video Generation Model | Vision-Language Model (VLM) | TTS Audio Engine |
| :--- | :--- | :--- | :--- | :--- |
| **T0** | CPU Only (No GPU) | Ken Burns Pan/Zoom (rule-based) | Qwen3-VL-2B | Kokoro / Edge-TTS |
| **T1** | 12–16 GB VRAM (RTX 3080/4070) | Wan2.2-Ti2V-5B (fallback: Ken Burns) | Qwen3-VL-8B (4-bit quantized) | MeloTTS (fallback: Kokoro) |
| **T2** | 24–48 GB VRAM (RTX 4090/A5000/Cloud) | Wan2.2-I2V-A14B (fallback: Wan2.2-Ti2V-5B, Hunyuan1.5) | Qwen3-VL-8B | MeloTTS (fallback: Kokoro) |

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
