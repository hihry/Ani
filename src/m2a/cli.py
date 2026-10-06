import typer
from pathlib import Path
from rich.console import Console

from m2a.state import RunState, RunConfig, PipelineStage, QCScores
from m2a.config import load_config
from m2a.graph import PipelineGraph

app = typer.Typer(help="Manhwa to Anime (M2A) Pipeline CLI")
console = Console()

@app.command()
def run(
    pages: list[Path] = typer.Argument(..., help="Source page images to process"),
    tier: str = typer.Option("default", "--tier", help="Compute tier configuration"),
    config_file: Path | None = typer.Option(None, "--config", help="Optional override config file"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to save the run")
):
    """Run the full pipeline on a set of source pages."""
    console.print(f"[bold green]Starting M2A Pipeline run...[/bold green]")
    
    config = load_config(tier=tier)
    
    run_config = RunConfig(
        config_hash="dummy_hash",  # TODO: Compute actual hash
        source_pages=pages,
        source_language="ko",
        output_resolution=config.resolution,
        compute_tier=tier
    )
    
    state = RunState(
        config=run_config,
        current_stage=PipelineStage.INGEST
    )
    
    graph = PipelineGraph(config=config, run_dir=output_dir)
    
    # TODO: Register actual node handlers
    # graph.register_node(PipelineStage.INGEST, ingest_handler)
    
    final_state = graph.run(state)
    
    if final_state.current_stage == PipelineStage.DONE:
        console.print("[bold green]Pipeline completed successfully![/bold green]")
    else:
        console.print(f"[bold red]Pipeline failed at stage: {final_state.current_stage}[/bold red]")

@app.command()
def resume(
    run_dir: Path = typer.Argument(..., help="Run directory containing state.json to resume")
):
    """Resume a failed or paused pipeline run."""
    console.print(f"Resuming run from {run_dir}...")
    try:
        state = RunState.load(run_dir)
        config = load_config(tier=state.config.compute_tier)
        graph = PipelineGraph(config=config, run_dir=run_dir)
        
        final_state = graph.resume(state)
        if final_state.current_stage == PipelineStage.DONE:
            console.print("[bold green]Pipeline resumed and completed successfully![/bold green]")
        else:
            console.print(f"[bold red]Pipeline failed at stage: {final_state.current_stage}[/bold red]")
            
    except Exception as e:
        console.print(f"[bold red]Failed to resume run: {e}[/bold red]")

@app.command()
def inspect(
    run_dir: Path = typer.Argument(..., help="Run directory to inspect")
):
    """Show the current state of a pipeline run."""
    try:
        state = RunState.load(run_dir)
        console.print(f"Run ID: [cyan]{state.config.run_id}[/cyan]")
        console.print(f"Current Stage: [magenta]{state.current_stage.value}[/magenta]")
        console.print(f"Panels Processed: {len(state.panels)}")
        if state.errors:
            console.print("[red]Errors:[/red]")
            for err in state.errors:
                console.print(f" - {err}")
    except Exception as e:
        console.print(f"[bold red]Failed to inspect run: {e}[/bold red]")

@app.command()
def bakeoff(
    panels_dir: Path = typer.Argument(..., help="Directory containing panels for bakeoff"),
    models: list[str] = typer.Option(["svd", "i2v"], "--models", help="Models to compare")
):
    """Run a video model comparison bakeoff."""
    console.print(f"Running model bakeoff on {panels_dir} using models: {', '.join(models)}")
    # TODO: Implement bakeoff logic
    console.print("[yellow]Bakeoff implementation pending...[/yellow]")

if __name__ == "__main__":
    app()
