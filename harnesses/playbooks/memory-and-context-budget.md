# Memory, Context Budget & Safety Guardrails — Manhwa → Anime (M2A)

This playbook codifies context engineering, persistent memory layers, tool safety boundaries, and the non-obvious M2A failure modes discovered during Phases 0–3.

---

## 1. Progressive Disclosure Context Budget

Keep startup token cost low by loading files in three strict tiers:

| Tier | When to Load | Files | Target Budget |
|---|---|---|---|
| **Tier 1: Always-On Metadata** | Every session start | Root `AGENTS.md`, `feature_list.json`, `progress.md` | <= 3,000 tokens |
| **Tier 2: Domain Instructions** | When a feature is selected | `harnesses/<domain>/AGENTS.md`, `session-handoff.md` | <= 2,000 tokens |
| **Tier 3: Deep Reference Docs** | On-demand only when relevant | `Phase3_implemn.md`, `optim.md`, `ARCHITECTURE_EXCALIDRAW_MERMAID.md`, `harnesses/playbooks/*.md` | Loaded selectively by section |

** Never load** `codebase.txt` (181 KB dump) or `runs/*/state.json` artifacts into context unless debugging a specific run ID.

---

## 2. Layered State & Memory Rules

1. **Human-Curated Instruction Layer (Version-Controlled)**:
   - `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `harnesses/*/AGENTS.md`, `harnesses/playbooks/*.md`.
2. **Structured Feature & Session Continuity Layer**:
   - `feature_list.json` (status + verification evidence per feature).
   - `progress.md` (what's done, in-progress, blockers, decisions, verification outputs).
   - `session-handoff.md` (clean restart checklist for the next session).
3. **What Never Belongs in Memory**:
   - Derivable code summaries, full file dumps, or transient `runs/` UUIDs.

---

## 3. Tool Permission & Environment Safety Boundaries

### Bypass-Immune Protected Paths & Commands
- **Never commit or stage**:
  - Model weights (`*.safetensors`, `*.gguf`, `*.ckpt`, `*.pth`, `*.onnx`, `*.bin`)
  - Generated run directories (`runs/*`) or root `.mp4`/`.avi`/`.webp` outputs
  - Private copyrighted evaluation panels (`data/bakeoff/private/*`)
- **Partition & Storage Rule**:
  - `/home` (btrfs) has only **~19 GB free**: only place the Python 3.12 ComfyUI virtualenv (`~/ComfyUI/.venv`) there.
  - `/run/media/hihry/OS/` (NTFS) has **~146 GB free**: store all model weights under `/run/media/hihry/OS/m2a_models/` and mount via ComfyUI `extra_model_paths.yaml`.
  - Because NTFS (`ntfs3`) does not support POSIX chmod/symlinks reliably, do not rely on symlinked model weight trees inside the repo.

### Concurrency Classification for Tool Calls
- **Safe to run in parallel (Read-Only)**: `read_file`, `grep`, `go vet ./...`, inspecting `configs/*.yaml`.
- **Must run serially (State-Mutating / Resource-Contending)**:
  - Editing `internal/model/state.go` and `src/m2a/state.py`
  - Running `pytest` or `go test ./...` that write to shared temporary caches
  - Any GPU/ComfyUI job on `:8188` or `:8100` (single-VRAM residency constraint on 6 GB local GPU)

---

## 4. M2A Non-Obvious Gotchas Checklist

1. **Python 3.15 vs Python 3.12 Split**:
   - Host Python is **3.15.1** (where `pytest` and the FastAPI M2A worker run). PyTorch/ComfyUI wheels require **Python 3.12** in `~/ComfyUI/.venv`. Keep `workers/requirements.txt` torch-free so the M2A worker talks to ComfyUI over HTTP (`:8188`).
2. **Wan 2.2 `4k + 1` and LTX `8k + 1` Frame Quantization**:
   - Diffusion video models only accept specific frame counts (`4k + 1` for Wan 2.2, `8k + 1` for LTX-2.3). Always use `workers/video/conform.py` (`frames_for()` + `conform()`) so the output clip matches `round(duration_s * fps)` before ASSEMBLE.
3. **1:4 Extreme Vertical Manhwa Strips**:
   - Raw manhwa panels can have 1:4 aspect ratios. Never center-crop or anisotropic-stretch them into 9:16; use `prepare_input()` (blurred-fill letterbox canvas) so characters and speech context are never clipped.
4. **HunyuanVideo 1.5 Regional License Exclusion**:
   - Tencent Hunyuan Community License excludes the **EU, UK, and South Korea**. Keep `hunyuan15.enabled: false` in `configs/models.yaml` unless explicitly overridden by the user.
5. **Deterministic Seeds Across Go & Python**:
   - Both Go (`internal/model/state.go`) and Python (`src/m2a/state.py`) compute panel seeds via 64-bit FNV-1a (`compute_deterministic_seed`). Never replace this with Python's salted `hash()`.
