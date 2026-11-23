"""Trackflow command-line interface."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Annotated

import onnxruntime as ort
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from trackflow import __version__
from trackflow.benchmark import (
    LEGACY_2017_BASELINE,
    generate_synthetic_crowd_video,
    run_benchmark_suite,
)
from trackflow.inspector import serve_inspector
from trackflow.models import MODEL_REGISTRY, REID_REGISTRY
from trackflow.pipeline import TrackflowPipeline
from trackflow.reid import search_track_cards

app = typer.Typer(
    name="trackflow",
    help="Modern real-time people & object tracking — queryable, debuggable, privacy-first.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def run(
    source: Annotated[
        str, typer.Argument(help="Video path, image directory, or 'camera' / webcam index.")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Detector preset, HF repo, or local .onnx path.")
    ] = "rtdetr-v2-r18",
    precision: Annotated[
        str, typer.Option("--precision", help="Model precision: fp32, fp16, or int8.")
    ] = "fp32",
    tracker: Annotated[
        str, typer.Option("--tracker", "-t", help="Tracker: bytetrack or botsort.")
    ] = "bytetrack",
    device: Annotated[
        str, typer.Option("--device", "-d", help="Execution device: auto, coreml, cuda, cpu.")
    ] = "auto",
    threshold: Annotated[
        float, typer.Option("--threshold", help="Detection confidence threshold.")
    ] = 0.45,
    classes: Annotated[
        str, typer.Option("--classes", "-c", help="Comma-separated class names, or 'all'.")
    ] = "person",
    reid: Annotated[
        str, typer.Option("--reid", help="Appearance/VLM backend: none, color, clip, siglip.")
    ] = "none",
    query: Annotated[
        str | None,
        typer.Option(
            "--query", "-q", help='Natural-language prompt-to-track, e.g. "person in red jacket".'
        ),
    ] = None,
    query_threshold: Annotated[
        float,
        typer.Option("--query-threshold", help="Cosine similarity threshold for --query matches."),
    ] = 0.22,
    anonymize: Annotated[
        str,
        typer.Option(
            "--anonymize", help="Privacy mode: none, blur, pixelate, face-blur, skeleton."
        ),
    ] = "none",
    homography: Annotated[
        Path | None, typer.Option("--homography", help="4-point floorplan calibration JSON file.")
    ] = None,
    skip: Annotated[int, typer.Option("--skip", help="Skip N frames between detections.")] = 0,
    max_frames: Annotated[
        int | None, typer.Option("--max-frames", help="Stop after N frames.")
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for run artifacts.")
    ] = Path("runs/latest"),
    save_video: Annotated[
        bool, typer.Option("--save-video/--no-video", help="Save annotated MP4 video.")
    ] = True,
    export: Annotated[
        str, typer.Option("--export", help="Comma-separated exports: parquet,jsonl,csv,html.")
    ] = "parquet,jsonl,html",
    display: Annotated[
        bool, typer.Option("--display", help="Show live preview window if OpenCV GUI is available.")
    ] = False,
) -> None:
    """Run real-time detection, tracking, optional NL querying, privacy anonymization, and exports."""
    class_list = [c.strip() for c in classes.split(",") if c.strip()]
    export_list = [e.strip() for e in export.split(",") if e.strip()]

    console.print(
        f"[bold cyan]Trackflow v{__version__}[/] · model=[bold]{model}[/] ({precision}) · "
        f"tracker=[bold]{tracker}[/] · privacy=[bold]{anonymize}[/]"
    )

    pipeline = TrackflowPipeline(
        model=model,
        precision=precision,  # type: ignore[arg-type]
        tracker=tracker,
        device=device,
        threshold=threshold,
        classes=class_list,
        reid=reid,  # type: ignore[arg-type]
        query=query,
        query_threshold=query_threshold,
        anonymize=anonymize,  # type: ignore[arg-type]
        homography=homography,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Tracking…", total=None)

        def _on_progress(done: int, total: int, fps: float) -> None:
            if total > 0:
                progress.update(task_id, total=total)
            progress.update(task_id, completed=done, description=f"Tracking ({fps:.1f} FPS)")

        summary = pipeline.run(
            source,
            output_dir=output,
            skip=skip,
            max_frames=max_frames,
            save_video=save_video,
            exports=export_list,
            display=display,
            progress_callback=_on_progress,
        )

    table = Table(title="Run Summary", show_header=False, box=None)
    table.add_row("Provider", f"[cyan]{summary.device}[/]")
    table.add_row(
        "Speed",
        f"[bold green]{summary.avg_fps:.1f} FPS[/] (inf {summary.avg_inference_ms:.1f} ms · trk {summary.avg_tracking_ms:.2f} ms)",
    )
    table.add_row(
        "Unique Tracks",
        f"[bold]{summary.unique_tracks}[/] (recovered: {summary.recovered_tracks}, Re-ID merges: {summary.reid_merges})",
    )
    table.add_row("Frames", f"{summary.processed_frames} processed / {summary.total_frames} read")
    for k, v in summary.output_files.items():
        table.add_row(f"Output ({k})", f"[dim]{v}[/]")
    console.print(table)


@app.command()
def search(
    run_dir: Annotated[
        Path, typer.Argument(help="Completed Trackflow run directory (or track_cards.json).")
    ],
    query: Annotated[
        str,
        typer.Argument(help='Natural-language search query, e.g. "person wearing blue backpack".'),
    ],
    top_k: Annotated[
        int, typer.Option("--top-k", "-k", help="Maximum Track Cards to return.")
    ] = 10,
    min_score: Annotated[
        float, typer.Option("--min-score", help="Minimum cosine similarity score.")
    ] = -1.0,
) -> None:
    """Search saved Track Cards in a completed run using natural language."""
    hits = search_track_cards(run_dir, query, top_k=top_k, min_score=min_score)
    table = Table(title=f'Track Card Search: "{query}"')
    table.add_column("Track ID", style="bold cyan")
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("Class")
    table.add_column("Frames")
    table.add_column("Duration", justify="right")
    table.add_column("Thumbnail", style="dim")

    for h in hits:
        table.add_row(
            f"#{h['track_id']}",
            f"{h['search_score']:.4f}",
            str(h["class_name"]),
            f"{h['first_frame']}–{h['last_frame']}",
            f"{h['duration_s']:.1f}s",
            str(h.get("thumbnail") or "-"),
        )
    console.print(table)


@app.command()
def inspect(
    run_dir: Annotated[Path, typer.Argument(help="Completed Trackflow run directory to inspect.")],
    compare: Annotated[
        Path | None,
        typer.Option(
            "--compare", help="Optional second run directory for side-by-side tracker diff."
        ),
    ] = None,
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Local HTTP port.")] = 8765,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not automatically open the browser.")
    ] = False,
) -> None:
    """Launch the interactive Visual Track Debugger & 4-Point Floorplan Studio."""
    console.print(
        f"[bold green]Trackflow Inspector[/] running at [link=http://{host}:{port}]http://{host}:{port}[/]"
    )
    if compare is not None:
        console.print(f"Comparing [cyan]{run_dir}[/] vs [magenta]{compare}[/]")
    console.print("Press Ctrl+C to stop.")
    serve_inspector(
        run_dir,
        compare_dir=compare,
        host=host,
        port=port,
        open_browser=not no_browser,
    )


@app.command()
def compare(
    source: Annotated[str, typer.Argument(help="Video path to compare trackers on.")],
    trackers: Annotated[
        str,
        typer.Option("--trackers", help="Two comma-separated trackers, e.g. bytetrack,botsort."),
    ] = "bytetrack,botsort",
    model: Annotated[str, typer.Option("--model", "-m", help="Detector preset.")] = "rtdetr-v2-r18",
    max_frames: Annotated[
        int | None, typer.Option("--max-frames", help="Limit frames for quick comparison.")
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Parent directory for both runs.")
    ] = Path("runs/compare"),
    open_inspector: Annotated[
        bool, typer.Option("--inspect", help="Open Visual Track Debugger after comparing.")
    ] = False,
) -> None:
    """Run two trackers on the same video and compare FPS, track counts, and ID switches side-by-side."""
    names = [t.strip() for t in trackers.split(",") if t.strip()]
    if len(names) < 2:
        raise typer.BadParameter(
            "Provide at least two comma-separated trackers (e.g. bytetrack,botsort)."
        )

    runs: list[tuple[str, Path, object]] = []
    for tname in names[:2]:
        sub = output / tname
        pipe = TrackflowPipeline(model=model, tracker=tname)
        summary = pipe.run(source, output_dir=sub, max_frames=max_frames, save_video=False)
        runs.append((tname, sub, summary))

    table = Table(title=f"Tracker Comparison ({model})")
    table.add_column("Metric", style="bold")
    for tname, _, _ in runs:
        table.add_column(tname, justify="right")

    s0, s1 = runs[0][2], runs[1][2]
    table.add_row("Avg FPS", f"{s0.avg_fps:.1f}", f"{s1.avg_fps:.1f}")  # type: ignore[attr-defined]
    table.add_row("Unique Tracks", str(s0.unique_tracks), str(s1.unique_tracks))  # type: ignore[attr-defined]
    table.add_row("Track Recoveries", str(s0.recovered_tracks), str(s1.recovered_tracks))  # type: ignore[attr-defined]
    table.add_row(
        "Observations", str(s0.total_track_observations), str(s1.total_track_observations)
    )  # type: ignore[attr-defined]
    console.print(table)

    if open_inspector:
        serve_inspector(runs[0][1], compare_dir=runs[1][1])


@app.command()
def benchmark(
    video: Annotated[
        Path | None,
        typer.Argument(help="Optional video file (generates synthetic crowd video if omitted)."),
    ] = None,
    models: Annotated[
        str, typer.Option("--models", help="Comma-separated models to benchmark.")
    ] = "rtdetr-v2-r18",
    trackers: Annotated[
        str, typer.Option("--trackers", help="Comma-separated trackers to benchmark.")
    ] = "bytetrack,botsort",
    device: Annotated[
        str, typer.Option("--device", help="Device: auto, coreml, cuda, cpu.")
    ] = "auto",
    max_frames: Annotated[int, typer.Option("--max-frames", help="Max frames per run.")] = 90,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Benchmark output directory.")
    ] = Path("runs/benchmark"),
    split_screen: Annotated[
        bool, typer.Option("--split-screen", help="Generate a 2017 vs. 2026 split-screen MP4.")
    ] = False,
) -> None:
    """Run the reproducible 2017 vs. 2026 benchmark suite."""
    output.mkdir(parents=True, exist_ok=True)
    target_video = (
        video
        if video is not None
        else generate_synthetic_crowd_video(output / "synthetic_crowd.mp4")
    )

    res = run_benchmark_suite(
        target_video,
        output_dir=output,
        models=[m.strip() for m in models.split(",") if m.strip()],
        trackers=[t.strip() for t in trackers.split(",") if t.strip()],
        device=device,
        max_frames=max_frames,
        split_screen=split_screen,
    )

    table = Table(title="2017 vs. 2026 Benchmark Results")
    table.add_column("Config")
    table.add_column("Provider")
    table.add_column("Avg FPS", justify="right", style="bold green")
    table.add_column("Inf (ms)", justify="right")
    table.add_column("Tracks", justify="right")

    table.add_row(
        f"[dim]2017 {LEGACY_2017_BASELINE['detector']} + DeepSORT[/]",
        f"[dim]{LEGACY_2017_BASELINE['hardware']}[/]",
        f"[dim]{LEGACY_2017_BASELINE['avg_fps']:.1f}[/]",
        "[dim]~71.4[/]",
        "[dim]—[/]",
    )
    for row in res["results"]:
        table.add_row(
            f"{row['model']} + {row['tracker']}",
            row["device"],
            f"{row['avg_fps']:.1f}",
            f"{row['avg_inference_ms']:.1f}",
            str(row["unique_tracks"]),
        )
    console.print(table)
    console.print(f"Markdown report written to [cyan]{res['report_md']}[/]")


@app.command()
def models() -> None:
    """List built-in Apache-2.0 detectors and vision-language Re-ID models."""
    det_table = Table(title="Built-in ONNX Object Detectors (Auto-Downloaded from Hugging Face)")
    det_table.add_column("Preset", style="bold cyan")
    det_table.add_column("Architecture")
    det_table.add_column("Size", justify="right")
    det_table.add_column("License", style="green")
    det_table.add_column("Hugging Face Repo", style="dim")

    for spec in MODEL_REGISTRY.values():
        det_table.add_row(
            spec.name,
            spec.architecture,
            f"{spec.approx_size_mb:.1f} MB",
            spec.license,
            spec.repo_id,
        )
    console.print(det_table)

    reid_table = Table(title="Built-in Vision-Language Re-ID & Query Models (--reid / --query)")
    reid_table.add_column("Backend", style="bold magenta")
    reid_table.add_column("INT8 Size", justify="right")
    reid_table.add_column("License", style="green")
    reid_table.add_column("Hugging Face Repo", style="dim")
    for rspec in REID_REGISTRY.values():
        reid_table.add_row(
            rspec.name, f"{rspec.approx_size_mb:.0f} MB", rspec.license, rspec.repo_id
        )
    console.print(reid_table)


@app.command()
def info() -> None:
    """Display system environment, Python version, and ONNX Runtime acceleration providers."""
    table = Table(title="Trackflow Environment Info", show_header=False)
    table.add_row("Trackflow Version", f"[bold cyan]{__version__}[/]")
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", f"{platform.system()} {platform.machine()}")
    table.add_row("ONNX Runtime", ort.__version__)
    table.add_row("Available EPs", ", ".join(ort.get_available_providers()))
    console.print(table)


@app.command()
def version() -> None:
    """Print the Trackflow version."""
    console.print(__version__)
