# AGENTS.md — Evaluation, Bake-off & QC Calibration Harness (`src/m2a/eval/` & `workers/eval_*.py`)

Specialized harness for the **Evaluation, Video Model Bake-off, Blind Rubric, QC Calibration & Real-Model GPU Probe Plane** of Manhwa → Anime (M2A): Phase 1 segmentation evaluator (`workers/eval_segmentation.py`), Phase 2 understanding evaluator (`workers/eval_understanding.py`), Phase 3 video bake-off engine (`src/m2a/eval/bakeoff.py`), 3-reviewer blind rubric & Krippendorff's alpha (`src/m2a/eval/rubric.py`), ROC/AUC & Youden's J threshold calibrator (`src/m2a/eval/calibrate_qc.py`), real-model GPU probe & anti-slideshow detector (`src/m2a/eval/probe_models.py`, `harnesses/eval-bakeoff/probe-models.sh`), and adversarial QC test fixtures (`tests/fixtures/qc_adversarial.py`).

## Startup Workflow

Before writing code:

1. **Confirm working directory** with `pwd` (repository root `/run/media/hihry/OS/hihry/Ani`)
2. **Read this file** and root [AGENTS.md](../../AGENTS.md) completely
3. **Read project docs if present** ([Phase3_implemn.md](../../Phase3_implemn.md) §6–§11, [README.md](../../README.md))
4. **Run `./harnesses/eval-bakeoff/init.sh`** to verify segmentation F1/IoU, QC adversarial fixtures, bake-off, calibration, and probe tests pass
5. **Read root `feature_list.json` and `progress.md`** to see current feature state
6. **Review recent commits** with `git log --oneline -5`

If baseline verification is failing, repair that first before adding new scope.

## Scope & Ownership Boundary

- **Owned Paths**: `src/m2a/eval/`, `workers/eval_segmentation.py`, `workers/eval_understanding.py`, `workers/qc_metrics.py`, `tests/fixtures/qc_adversarial.py`, `configs/bakeoff/`, `configs/qc/`, `configs/calibration/`, `data/eval/`, `scripts/comfyui/`, `harnesses/eval-bakeoff/probe-models.sh`
- **Out of Scope**: Core Go CLI/DAG orchestration (`cmd/m2a/`, `internal/pipeline/`) and FastAPI/Streamlit UI (`api/`, `ui/`)

## Working Rules & Evaluation Invariants

- **One feature at a time**: Pick exactly one unfinished feature from root `feature_list.json`
- **Stay in scope**: Only modify evaluation, bake-off, calibration, probe, and QC metric files
- **Verification required**: Run `./harnesses/eval-bakeoff/init.sh` and `./init.sh` before claiming done
- **Live GPU / Real-Model Verification (`probe-models.sh`)**: Before claiming any real-GPU I2V run or bake-off is verified, run `./harnesses/eval-bakeoff/probe-models.sh --tier <tier>` (verifies model SHA-256 on `/run/media/hihry/OS/m2a_models/`, live ComfyUI `:8188` custom nodes, and 1 short real I2V render with `route_used == "i2v"` and `motion_score >= 0.02`) and `./harnesses/eval-bakeoff/probe-models.sh --check-run runs/<id>` (fails if `i2v` panels degraded to `parallax`/`kenburns` or produced frozen frames)
- **Update artifacts**: Before ending session, update root `progress.md`, `feature_list.json`, and `session-handoff.md`
- **Leave clean state**: Next session must be able to run `./harnesses/eval-bakeoff/init.sh` immediately
- **Zero Hard-Coded Benchmark Numbers**: `bakeoff.py`, `calibrate_qc.py`, and `eval_*.py` must always compute metrics from real decoded frames and labels — never emit static `0.90 / 0.80` placeholders
- **100% Adversarial QC Rejection Gate (E6)**: Any change to `workers/qc_metrics.py` or `configs/qc/*.yaml` must maintain 100% rejection across all 6 synthetic adversarial failure modes (`frozen`, `shuffled`, `flicker`, `text_overlay`, `hue_shift`, `face_swap`)
- **Segmentation Regression Floor**: `workers/eval_segmentation.py` must maintain `F1 >= 0.98` (`1.0000` on synthetic baseline) and `Mean IoU >= 0.98`
- **Data Privacy & Copyright**: Keep copyrighted real manhwa bake-off panels inside gitignored `data/bakeoff/private/`; only commit synthetic or CC-BY/licensed panels to `data/eval/`

## Required Artifacts

- `../../feature_list.json` — Root feature tracker (single source of truth across all planes)
- `../../progress.md` — Root session continuity log
- `init.sh` — CPU evaluation verification script (segmentation eval + `pytest -m eval_qc`)
- `probe-models.sh` — Live GPU model SHA-256 + ComfyUI `:8188` + short real render + anti-slideshow detector
- `../../session-handoff.md` — Root handoff notes for the next session

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] Target evaluation, metric, or calibration behavior is implemented with reproducible JSONL/YAML outputs
- [ ] Required verification actually ran (`./harnesses/eval-bakeoff/init.sh` and `./init.sh`, plus `./harnesses/eval-bakeoff/probe-models.sh` when claiming live GPU verification)
- [ ] Evidence recorded in root `feature_list.json` and `progress.md`
- [ ] Repository remains restartable from standard startup path

## End of Session

Before ending a session:

1. Update root `progress.md` with current state and benchmark numbers
2. Update root `feature_list.json` with new feature status and evidence
3. Record any unresolved risks or blockers in root `session-handoff.md`
4. Commit with descriptive message once work is in safe state
5. Leave repo clean enough for next session to run `./harnesses/eval-bakeoff/init.sh` immediately

## Verification Commands

```bash
# CPU evaluation & calibration verification
./harnesses/eval-bakeoff/init.sh

# Live GPU model SHA-256 + ComfyUI :8188 + 1-panel real I2V render probe
./harnesses/eval-bakeoff/probe-models.sh --tier t1s

# Offline model file + SHA-256 check only (no ComfyUI required)
./harnesses/eval-bakeoff/probe-models.sh --tier t1s --files-only

# Audit a completed run directory for silent I2V -> pan/zoom slideshow fallback
./harnesses/eval-bakeoff/probe-models.sh --check-run runs/<run_id>
```

Required CPU checks:
- `python3 -m compileall -q src/m2a/eval workers/eval_segmentation.py workers/eval_understanding.py workers/qc_metrics.py`
- `python3 workers/eval_segmentation.py`
- `python3 -m pytest -m "eval_qc" -q`

## Escalation

If you encounter:
- **Architecture decisions**: Consult [Phase3_implemn.md](../../Phase3_implemn.md) §7–§10, otherwise ask user
- **Unclear requirements**: Check `configs/bakeoff/phase3.yaml` and `configs/qc/`, otherwise ask user
- **Repeated test failures**: Update progress, flag for human review
- **Scope ambiguity**: Re-read root `feature_list.json` for definition of done

