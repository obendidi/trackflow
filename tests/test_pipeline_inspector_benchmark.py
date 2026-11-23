"""End-to-end tests for TrackflowPipeline, Inspector API, Benchmark suite, and CLI."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from typer.testing import CliRunner

from trackflow.benchmark import generate_synthetic_crowd_video, run_benchmark_suite
from trackflow.cli import app
from trackflow.homography import DwellZone, FloorplanHomography
from trackflow.inspector import make_inspector_handler
from trackflow.pipeline import TrackflowPipeline
from trackflow.types import Detection


class _SyntheticMotionDetector:
    """Fast deterministic detector returning two crossing people per frame for CI/unit tests."""

    active_provider = "MockCPU"

    def __init__(self) -> None:
        self.step = 0

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        _ = frame_bgr
        s = self.step
        self.step += 1
        x1 = 40.0 + s * 6.0
        x2 = 520.0 - s * 5.0
        return [
            Detection(
                bbox=(x1, 80.0, x1 + 45.0, 220.0), confidence=0.92, class_id=0, class_name="person"
            ),
            Detection(
                bbox=(x2, 100.0, x2 + 45.0, 240.0), confidence=0.88, class_id=0, class_name="person"
            ),
        ]


def test_end_to_end_pipeline_inspector_and_benchmark(tmp_path: Path) -> None:
    vid_path = generate_synthetic_crowd_video(tmp_path / "crowd.mp4", num_frames=20)

    hom = FloorplanHomography(
        image_points=[(20, 80), (620, 80), (620, 340), (20, 340)],
        world_points=[(0, 0), (10, 0), (10, 6), (0, 6)],
        width_m=10.0,
        height_m=6.0,
        zones=[DwellZone(name="lobby", polygon=[(0, 0), (5, 0), (5, 6), (0, 6)])],
    )

    run_dir = tmp_path / "run_a"
    pipe = TrackflowPipeline(
        model="synthetic",
        tracker="bytetrack",
        reid="color",
        anonymize="face-blur",
        homography=hom,
        detector=_SyntheticMotionDetector(),
    )
    summary = pipe.run(
        str(vid_path),
        output_dir=run_dir,
        save_video=True,
        exports=("parquet", "jsonl", "csv", "html"),
    )

    assert summary.processed_frames == 20
    assert summary.unique_tracks >= 2
    assert (run_dir / "tracks.parquet").is_file()
    assert (run_dir / "tracks.jsonl").is_file()
    assert (run_dir / "tracks.csv").is_file()
    assert (run_dir / "report.html").is_file()
    assert (run_dir / "annotated.mp4").is_file()
    assert (run_dir / "floorplan.png").is_file()
    assert (run_dir / "track_cards.json").is_file()

    # Verify Parquet schema and row count
    table = pq.read_table(str(run_dir / "tracks.parquet"))
    assert table.num_rows == summary.total_track_observations
    assert "world_x" in table.column_names and "zone" in table.column_names

    # Verify Inspector HTTP server API (/ , /api/run, POST /api/homography)
    handler = make_inspector_handler(run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            assert b"Trackflow Inspector" in resp.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["summary"]["processed_frames"] == 20
            assert len(payload["cards"]) >= 2

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/homography",
            data=json.dumps(
                {
                    "image_points": [[20, 80], [620, 80], [620, 340], [20, 340]],
                    "width_m": 12.0,
                    "height_m": 8.0,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            h_res = json.loads(resp.read().decode("utf-8"))
            assert len(h_res["trajectories"]) >= 2
    finally:
        server.shutdown()
        server.server_close()

    # Verify Benchmark suite + 2017 vs. 2026 split-screen video generation
    bench = run_benchmark_suite(
        vid_path,
        output_dir=tmp_path / "bench",
        models=("synthetic",),
        trackers=("bytetrack", "botsort"),
        max_frames=15,
        split_screen=True,
        detector_factory=lambda _m: _SyntheticMotionDetector(),
    )
    assert len(bench["results"]) == 2
    assert Path(bench["report_md"]).is_file()
    assert bench["split_screen_video"] is not None
    assert Path(bench["split_screen_video"]).is_file()


def test_cli_smoke_commands(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    for args in (["--help"], ["models"], ["info"], ["version"]):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, f"CLI failed for {args}: {res.output}"

    # Verify `trackflow run` CLI progress bar and summary rendering end-to-end
    vid = generate_synthetic_crowd_video(tmp_path / "cli_smoke.mp4", num_frames=6)
    monkeypatch.setattr(
        "trackflow.pipeline.OnnxDetrDetector",
        lambda **_kwargs: _SyntheticMotionDetector(),
    )
    res = runner.invoke(
        app,
        ["run", str(vid), "--no-video", "--export", "jsonl", "--output", str(tmp_path / "out")],
    )
    assert res.exit_code == 0, f"CLI run failed: {res.output}"
    assert (tmp_path / "out" / "tracks.jsonl").is_file()
