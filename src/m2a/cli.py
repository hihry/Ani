"""Python CLI for Manhwa-to-Anime (M2A) Pipeline."""
from __future__ import annotations

import hashlib
from pathlib import Path
import typer
from rich.console import Console

from m2a.config import load_config
from m2a.eval.bakeoff import BakeoffConfig, run_bakeoff
from m2a.graph import PipelineGraph
from m2a.state import PipelineStage, RunConfig, RunState

app = typer.Typer(help="Manhwa to Anime (M2A) Pipeline CLI")
console = Console()


@app.command()
def run(
    pages: list[Path] = typer.Argument(..., help="Source page images to process"),
    tier: str = typer.Option("t1", "--tier", help="Compute tier configuration (t0, t1, t2)"),
    lang: str = typer.Option("ko", "--lang", help="Source language (ko, ja, en)"),
    auto_approve: bool = typer.Option(True, "--auto-approve/--pause-checkpoints", help="Auto-approve CP1 and CP2"),
    output_dir: Path = typer.Option(Path("./runs"), "--output-dir", help="Directory to save the run"),
) -> None:
    """Run the full pipeline on a set of source pages."""
    console.print("[bold green]Starting M2A Pipeline run...[/bold green]")

    config = load_config(tier=tier)
    hasher = hashlib.sha256()
    for p in pages:
        hasher.update(str(p).encode("utf-8"))
    hasher.update(lang.encode("utf-8"))
    hasher.update(tier.encode("utf-8"))

    run_config = RunConfig(
        config_hash=hasher.hexdigest(),
        source_pages=[str(p.as_posix()) for p in pages],
        source_language=lang,
        output_resolution=config.resolution,
        compute_tier=tier,
    )

    run_folder = output_dir / run_config.run_id
    state = RunState(config=run_config, current_stage=PipelineStage.INGEST)

    graph = PipelineGraph(config=config, run_dir=run_folder, auto_approve_checkpoints=auto_approve)
    final_state = graph.run(state)

    if final_state.current_stage == PipelineStage.DONE:
        console.print(f"[bold green]Pipeline completed successfully in {run_folder}![/bold green]")
    elif final_state.current_stage in (PipelineStage.CHECKPOINT_1, PipelineStage.CHECKPOINT_2):
        console.print(f"[bold yellow]Pipeline paused at {final_state.current_stage.value} in {run_folder}[/bold yellow]")
    else:
        console.print(f"[bold red]Pipeline failed at stage: {final_state.current_stage}[/bold red]")


@app.command()
def resume(
    run_dir: Path = typer.Argument(..., help="Run directory containing state.json to resume"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Auto-approve remaining checkpoints"),
) -> None:
    """Resume a failed or paused pipeline run."""
    console.print(f"Resuming run from {run_dir}...")
    try:
        state = RunState.load(run_dir)
        config = load_config(tier=state.config.compute_tier)
        graph = PipelineGraph(config=config, run_dir=run_dir, auto_approve_checkpoints=auto_approve)

        final_state = graph.resume(state)
        if final_state.current_stage == PipelineStage.DONE:
            console.print("[bold green]Pipeline resumed and completed successfully![/bold green]")
        else:
            console.print(f"[bold yellow]Pipeline stopped at stage: {final_state.current_stage.value}[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Failed to resume run: {e}[/bold red]")


@app.command()
def inspect(
    run_dir: Path = typer.Argument(..., help="Run directory to inspect"),
) -> None:
    """Show the current state of a pipeline run."""
    try:
        state = RunState.load(run_dir)
        console.print(f"Run ID: [cyan]{state.config.run_id}[/cyan]")
        console.print(f"Current Stage: [magenta]{state.current_stage.value}[/magenta]")
        console.print(f"Compute Tier: [green]{state.config.compute_tier}[/green]")
        console.print(f"Panels Processed: {len(state.panels)}")
        console.print(f"Clean Plates: {len(state.clean_plate_paths)}")
        console.print(f"Audio Tracks: {len(state.audio_tracks)}")
        if state.errors:
            console.print("[red]Errors:[/red]")
            for err in state.errors:
                console.print(f" - {err}")
    except Exception as e:
        console.print(f"[bold red]Failed to inspect run: {e}[/bold red]")


@app.command()
def bakeoff(
    panels_dir: Path = typer.Argument(..., help="Directory containing panels for bakeoff"),
    models: list[str] = typer.Option(["wan22-ti2v-5b", "hunyuan15", "ltx23"], "--models", help="Models to compare"),
    output_dir: Path = typer.Option(Path("./runs/bakeoff"), "--output-dir", help="Bakeoff output directory"),
) -> None:
    """Run a video model comparison bakeoff."""
    panel_files = sorted(panels_dir.glob("*.png"))
    cfg = BakeoffConfig(models=models, panel_paths=panel_files, output_dir=output_dir)
    run_bakeoff(cfg)
    console.print(f"[bold green]Bakeoff complete! Results saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
