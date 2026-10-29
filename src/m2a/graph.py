"""Canonical Go Engine bridge for the Manhwa-to-Anime (M2A) pipeline (`optim.md` A1 & P0-07).

Delegates `run`, `resume`, and `inspect` orchestration to the compiled Go `m2a` binary
(`cmd/m2a`) and reads/writes `runs/<id>/state.json` directly, eliminating the duplicate
Python stage-machine loop.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from m2a.config import PipelineConfig, load_config
from m2a.state import PipelineStage, RunState

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_m2a_binary() -> Path:
    """Resolves the compiled Go `m2a` binary path, building it if absent or stale."""
    env_bin = os.environ.get("M2A_BIN", "").strip()
    if env_bin:
        p = Path(env_bin)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.is_file():
            return p

    bin_name = "m2a.exe" if os.name == "nt" else "m2a"
    bin_path = REPO_ROOT / bin_name
    main_go = REPO_ROOT / "cmd" / "m2a" / "main.go"

    needs_build = not bin_path.is_file()
    if not needs_build and main_go.is_file():
        try:
            if main_go.stat().st_mtime > bin_path.stat().st_mtime:
                needs_build = True
        except OSError:
            pass

    if needs_build:
        subprocess.run(
            ["go", "build", "-o", str(bin_path), "./cmd/m2a"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    return bin_path


def write_config_snapshot(config: PipelineConfig, run_dir: Path) -> Path:
    """Persists the effective PipelineConfig to `run_dir/config.snapshot.yaml` for the Go engine."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_dict: dict[str, Any] = config.model_dump(exclude={"tier"}, exclude_none=True)
    snap_dict.setdefault("general", {})["run_dir"] = str(run_dir.as_posix())
    snap_path = run_dir / "config.snapshot.yaml"
    snap_path.write_text(yaml.safe_dump(snap_dict, sort_keys=False), encoding="utf-8")
    return snap_path


def save_checkpoint_edits(state: RunState, run_dir: Path) -> None:
    """Saves human edits made via API/UI at CHECKPOINT_1 or CHECKPOINT_2 to disk and state.json."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if state.current_stage == PipelineStage.CHECKPOINT_1:
        vlm_dir = run_dir / "understanding"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        for p in state.panels:
            (vlm_dir / f"{p.panel_id}.json").write_text(
                p.model_dump_json(indent=2),
                encoding="utf-8",
            )
        if state.character_bible is not None:
            (vlm_dir / "character_bible.json").write_text(
                state.character_bible.model_dump_json(indent=2),
                encoding="utf-8",
            )
    elif state.current_stage == PipelineStage.CHECKPOINT_2:
        routes_dir = run_dir / "routes"
        routes_dir.mkdir(parents=True, exist_ok=True)
        plan_file = routes_dir / "routing_plan.json"
        wrote_structured = False
        if plan_file.is_file():
            try:
                raw_plan = json.loads(plan_file.read_text(encoding="utf-8"))
                if isinstance(raw_plan, dict) and isinstance(raw_plan.get("panels"), list):
                    by_id = {p.panel_id: p for p in state.panels}
                    for entry in raw_plan["panels"]:
                        if isinstance(entry, dict) and entry.get("panel_id") in by_id:
                            pan = by_id[entry["panel_id"]]
                            entry["route"] = pan.route
                            entry["duration_s"] = pan.duration_s
                            entry["motion_prompt"] = pan.motion_prompt
                    plan_file.write_text(json.dumps(raw_plan, indent=2), encoding="utf-8")
                    wrote_structured = True
            except Exception:
                pass
        if not wrote_structured:
            plan_file.write_text(
                json.dumps([p.model_dump() for p in state.panels], indent=2),
                encoding="utf-8",
            )

    state.save(run_dir)


def run_m2a_cli(args: list[str], run_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Executes the canonical Go `m2a` CLI binary with the given arguments."""
    m2a_bin = resolve_m2a_binary()
    proc = subprocess.run(
        [str(m2a_bin), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if run_dir is not None:
        try:
            run_dir_path = Path(run_dir)
            run_dir_path.mkdir(parents=True, exist_ok=True)
            with open(run_dir_path / "engine.log", "a", encoding="utf-8") as log_fh:
                if proc.stdout:
                    log_fh.write(proc.stdout)
                if proc.stderr:
                    log_fh.write(proc.stderr)
        except OSError:
            pass
    return proc


def run_go_engine(
    state: RunState,
    run_dir: Path,
    config: PipelineConfig | None = None,
    auto_approve: bool = False,
) -> RunState:
    """Runs the pipeline from INGEST via the canonical Go `m2a run` command and returns updated RunState."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    eff_cfg = config or load_config(tier=state.config.compute_tier, config_dir=REPO_ROOT / "configs")
    write_config_snapshot(eff_cfg, run_dir)
    state.save(run_dir)

    if not state.config.source_pages:
        state.record_error("ingest: no source pages specified")
        state.advance_to(PipelineStage.FAILED)
        state.save(run_dir)
        return state

    cmd_args = [
        "run",
        "--pages",
        ",".join(state.config.source_pages),
        "--tier",
        state.config.compute_tier or "t1",
        "--lang",
        state.config.source_language or "ko",
        "--config",
        str((REPO_ROOT / "configs").as_posix()),
        "--run-id",
        state.config.run_id,
        "--run-dir",
        str(run_dir),
    ]
    if auto_approve:
        cmd_args.append("--auto-approve")

    proc = run_m2a_cli(cmd_args, run_dir=run_dir)
    loaded = RunState.load(run_dir)
    if proc.returncode != 0 and loaded.current_stage != PipelineStage.FAILED:
        err_msg = (proc.stderr or proc.stdout or f"m2a run exited with code {proc.returncode}").strip()
        loaded.record_error(err_msg)
        loaded.advance_to(PipelineStage.FAILED)
        loaded.save(run_dir)
    return loaded


def resume_go_engine(
    run_dir: Path,
    state: RunState | None = None,
    config: PipelineConfig | None = None,
    auto_approve: bool = False,
) -> RunState:
    """Resumes a paused or failed run via the canonical Go `m2a resume` command and returns updated RunState."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if state is not None and not (run_dir / "state.json").is_file():
        state.save(run_dir)

    if config is not None:
        write_config_snapshot(config, run_dir)

    cmd_args = [
        "resume",
        "--run-dir",
        str(run_dir),
        "--config",
        str((REPO_ROOT / "configs").as_posix()),
    ]
    if auto_approve:
        cmd_args.append("--auto-approve")

    proc = run_m2a_cli(cmd_args, run_dir=run_dir)
    loaded = RunState.load(run_dir)
    if proc.returncode != 0 and loaded.current_stage != PipelineStage.FAILED:
        err_msg = (proc.stderr or proc.stdout or f"m2a resume exited with code {proc.returncode}").strip()
        loaded.record_error(err_msg)
        loaded.advance_to(PipelineStage.FAILED)
        loaded.save(run_dir)
    return loaded


def inspect_go_engine(run_dir: Path) -> str:
    """Inspects a run directory via the canonical Go `m2a inspect` command and returns formatted output."""
    run_dir = Path(run_dir).resolve()
    proc = run_m2a_cli(["inspect", "--run-dir", str(run_dir)])
    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or f"m2a inspect failed ({proc.returncode})").strip()
        raise RuntimeError(err_msg)
    return proc.stdout


class PipelineGraph:
    """Thin adapter delegating pipeline execution to the canonical Go `m2a` engine."""

    def __init__(
        self,
        config: PipelineConfig,
        run_dir: Path,
        auto_approve_checkpoints: bool = False,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.config.general.run_dir = str(self.run_dir.as_posix())
        self.auto_approve_checkpoints = auto_approve_checkpoints

    def run(self, state: RunState) -> RunState:
        return run_go_engine(
            state=state,
            run_dir=self.run_dir,
            config=self.config,
            auto_approve=self.auto_approve_checkpoints,
        )

    def resume(self, state: RunState | None = None) -> RunState:
        return resume_go_engine(
            run_dir=self.run_dir,
            state=state,
            config=self.config,
            auto_approve=self.auto_approve_checkpoints,
        )

    def inspect(self) -> str:
        return inspect_go_engine(self.run_dir)
