"""End-to-end tests for TrackflowPipeline, Inspector API, Benchmark suite, and CLI."""

from __future__ import annotations

import json
import signal
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
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


class _InterruptingDetector(_SyntheticMotionDetector):
    """Sends Ctrl-C to its own process, ``presses`` times, while detecting frame ``at_frame``."""

    def __init__(self, at_frame: int, presses: int = 1) -> None:
        super().__init__()
        self.at_frame, self.presses = at_frame, presses

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self.step == self.at_frame:
            for _ in range(self.presses):
                signal.raise_signal(signal.SIGINT)
        return super().detect(frame_bgr)


def test_ctrl_c_stops_the_run_and_writes_every_output(tmp_path: Path) -> None:
    vid_path = generate_synthetic_crowd_video(tmp_path / "crowd.mp4", num_frames=20)
    handler = signal.getsignal(signal.SIGINT)
    pipe = TrackflowPipeline(model="synthetic", detector=_InterruptingDetector(at_frame=5))

    summary = pipe.run(
        str(vid_path), output_dir=tmp_path / "run", save_video=False, exports=("jsonl",)
    )

    assert summary.processed_frames == 6
    for name in ("tracks.jsonl", "track_cards.json", "events.json", "summary.json"):
        assert (tmp_path / "run" / name).is_file()
    assert signal.getsignal(signal.SIGINT) is handler


def test_second_ctrl_c_aborts_the_run(tmp_path: Path) -> None:
    vid_path = generate_synthetic_crowd_video(tmp_path / "crowd.mp4", num_frames=20)
    pipe = TrackflowPipeline(model="synthetic", detector=_InterruptingDetector(5, presses=2))

    with pytest.raises(KeyboardInterrupt):
        pipe.run(str(vid_path), output_dir=tmp_path / "run", save_video=False, exports=("jsonl",))

    assert not (tmp_path / "run" / "summary.json").exists()


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

    # Anonymized runs must not write raw person crops anywhere (files or report thumbnails).
    assert not (run_dir / "crops").exists()
    cards = json.loads((run_dir / "track_cards.json").read_text(encoding="utf-8"))["cards"]
    assert cards and all(c["thumbnail"] is None for c in cards)
    assert "crops/" not in (run_dir / "report.html").read_text(encoding="utf-8")

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
        "trackflow.pipeline.create_detector",
        lambda **_kwargs: _SyntheticMotionDetector(),
    )
    res = runner.invoke(
        app,
        ["run", str(vid), "--no-video", "--export", "jsonl", "--output", str(tmp_path / "out")],
    )
    assert res.exit_code == 0, f"CLI run failed: {res.output}"
    assert (tmp_path / "out" / "tracks.jsonl").is_file()

    # Repeating a command with its default output must not collide with the previous run.
    monkeypatch.chdir(tmp_path)
    for args in (
        ["run", str(vid), "--no-video", "--export", "jsonl"],
        ["compare", str(vid), "--max-frames", "4"],
        ["benchmark", str(vid), "--trackers", "bytetrack", "--max-frames", "4"],
    ):
        for _ in range(2):
            res = runner.invoke(app, args)
            assert res.exit_code == 0, f"CLI {args[0]} failed on repeat: {res.output}"
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [
        "benchmark-1",
        "benchmark-2",
        "compare-1",
        "compare-2",
        "run-1",
        "run-2",
    ]

    # Reusing an explicit non-empty output is a clear usage error, not a traceback.
    res = runner.invoke(app, ["compare", str(vid), "-o", "runs/compare-1"])
    assert res.exit_code == 2 and "not empty" in res.output

    # A run that fails on its input leaves no empty run folder behind.
    res = runner.invoke(app, ["run", str(tmp_path / "missing.mp4")])
    assert res.exit_code == 2
    assert not (tmp_path / "runs" / "run-3").exists()


def test_inspector_serves_source_frames_and_respects_anonymization(tmp_path: Path) -> None:
    vid_path = generate_synthetic_crowd_video(tmp_path / "crowd.mp4", num_frames=6)

    def _serve(run_dir: Path):
        handler = make_inspector_handler(run_dir)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, server.server_address[1], handler

    # 1. Normal --no-video run serves JPEG frames directly from the local source video.
    run_plain = tmp_path / "plain"
    TrackflowPipeline(model="synthetic", detector=_SyntheticMotionDetector()).run(
        str(vid_path), output_dir=run_plain, save_video=False, exports=("jsonl",)
    )
    server, port, handler = _serve(run_plain)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run") as resp:
            assert json.loads(resp.read().decode("utf-8"))["has_frames"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/frame/0") as resp:
            assert resp.status == 200
            assert resp.headers.get_content_type() == "image/jpeg"
            assert resp.read()[:2] == b"\xff\xd8"
        for bad in ("999", "not-an-int"):
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/frame/{bad}")
            assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        handler.frame_store.close()

    # 2. Anonymized --no-video run must never leak raw source frames.
    run_anon_novid = tmp_path / "anon_novid"
    TrackflowPipeline(
        model="synthetic", anonymize="face-blur", detector=_SyntheticMotionDetector()
    ).run(str(vid_path), output_dir=run_anon_novid, save_video=False, exports=("jsonl",))
    server, port, handler = _serve(run_anon_novid)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run") as resp:
            assert json.loads(resp.read().decode("utf-8"))["has_frames"] is False
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/frame/0")
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        handler.frame_store.close()

    # 3. Anonymized run with annotated.mp4 serves frames from annotated.mp4 (not raw source).
    run_anon_vid = tmp_path / "anon_vid"
    TrackflowPipeline(
        model="synthetic", anonymize="blur", detector=_SyntheticMotionDetector()
    ).run(str(vid_path), output_dir=run_anon_vid, save_video=True, exports=("jsonl",))
    vid_path.unlink()  # proves frames come from annotated.mp4, not the raw source
    server, port, handler = _serve(run_anon_vid)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/run") as resp:
            assert json.loads(resp.read().decode("utf-8"))["has_frames"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/frame/0") as resp:
            assert resp.status == 200 and resp.read()[:2] == b"\xff\xd8"
    finally:
        server.shutdown()
        server.server_close()
        handler.frame_store.close()

    # 4. Image-folder source serves frames in natural order.
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    for i in (1, 2, 10):
        frame = np.full((120, 160, 3), i * 20, dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"frame_{i}.png"), frame)
    run_dir_src = tmp_path / "dir_src"
    TrackflowPipeline(model="synthetic", detector=_SyntheticMotionDetector()).run(
        str(img_dir), output_dir=run_dir_src, save_video=False, exports=("jsonl",), fps=10.0
    )
    server, port, handler = _serve(run_dir_src)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/frame/2") as resp:
            assert resp.status == 200 and resp.read()[:2] == b"\xff\xd8"
    finally:
        server.shutdown()
        server.server_close()
        handler.frame_store.close()


def test_agent_cli_json_and_llms_txt(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    info_res = runner.invoke(app, ["info", "--json"])
    assert info_res.exit_code == 0, info_res.output
    info_data = json.loads(info_res.output)
    assert {
        "version",
        "python",
        "platform",
        "onnxruntime",
        "available_providers",
        "nvidia_gpu",
        "apple_p_cores",
        "tensorrt_libs",
        "default_device",
        "recommended_command",
    } <= set(info_data)
    assert isinstance(info_data["available_providers"], list)

    # When an NVIDIA GPU is present on a CPU-only ORT install, `info` prints the 1-line GPU command.
    monkeypatch.setattr("trackflow.cli.detect_nvidia_gpu", lambda: "Tesla T4, 580.82")
    monkeypatch.setattr("trackflow.cli.ort.get_available_providers", lambda: ["CPUExecutionProvider"])
    gpu_info = json.loads(runner.invoke(app, ["info", "--json"]).output)
    assert gpu_info["nvidia_gpu"] == "Tesla T4, 580.82"
    assert "onnxruntime-gpu" in gpu_info["recommended_command"]

    # TRACKFLOW_DEFAULT_DEVICE flows into `trackflow run` when --device is omitted.
    seen_device: list[str] = []

    def fake_create_det(**kwargs):
        seen_device.append(kwargs["device"])
        return _SyntheticMotionDetector()

    vid = generate_synthetic_crowd_video(tmp_path / "env_dev.mp4", num_frames=4)
    monkeypatch.setenv("TRACKFLOW_DEFAULT_DEVICE", "tensorrt")
    monkeypatch.setattr("trackflow.pipeline.create_detector", fake_create_det)
    run_res = runner.invoke(
        app,
        ["run", str(vid), "--no-video", "--export", "jsonl", "-o", str(tmp_path / "env_out")],
    )
    assert run_res.exit_code == 0, run_res.output
    assert seen_device == ["tensorrt"]

    fake_hits = [
        {
            "track_id": 1,
            "class_name": "person",
            "first_frame": 0,
            "last_frame": 20,
            "duration_s": 0.8,
            "search_score": 0.312,
            "thumbnail": "crops/track_1.jpg",
        }
    ]
    monkeypatch.setattr("trackflow.cli.search_track_cards", lambda *_a, **_kw: fake_hits)
    search_res = runner.invoke(app, ["search", str(tmp_path), "person in white shirt", "--json"])
    assert search_res.exit_code == 0, search_res.output
    search_data = json.loads(search_res.output)
    assert search_data == {"query": "person in white shirt", "results": fake_hits}

    repo_root = Path(__file__).resolve().parent.parent
    llms_txt = repo_root / "web" / "llms.txt"
    assert llms_txt.is_file() and "trackflow-python" in llms_txt.read_text(encoding="utf-8")
    assert 'href="llms.txt"' in (repo_root / "web" / "index.html").read_text(encoding="utf-8")


def test_async_pipeline_flushes_video_on_ctrl_c_and_propagates_worker_errors(
    tmp_path: Path, monkeypatch
) -> None:
    vid_path = generate_synthetic_crowd_video(tmp_path / "crowd.mp4", num_frames=16)

    # 1. Single Ctrl-C with save_video=True flushes every processed frame into a valid MP4.
    run_ctrlc = tmp_path / "run_ctrlc"
    pipe_ctrlc = TrackflowPipeline(model="synthetic", detector=_InterruptingDetector(at_frame=5))
    summary = pipe_ctrlc.run(
        str(vid_path), output_dir=run_ctrlc, save_video=True, exports=("jsonl",)
    )
    assert summary.processed_frames == 6
    cap = cv2.VideoCapture(str(run_ctrlc / "annotated.mp4"))
    written = 0
    while cap.read()[0]:
        written += 1
    cap.release()
    assert written == 6

    # 2. Exception raised inside the background writer thread propagates to the caller.
    pipe_werr = TrackflowPipeline(model="synthetic", detector=_SyntheticMotionDetector())
    seen_calls = 0

    def failing_apply(frame, tracks, *, detections, coasting):
        nonlocal seen_calls
        seen_calls += 1
        if seen_calls == 3:
            raise RuntimeError("writer-boom")
        return frame

    monkeypatch.setattr(pipe_werr.anonymizer, "apply", failing_apply)
    with pytest.raises(RuntimeError, match="writer-boom"):
        pipe_werr.run(str(vid_path), output_dir=tmp_path / "run_werr", save_video=True, exports=())

    # 3. Exception raised inside the background frame reader thread propagates cleanly.
    img_dir = tmp_path / "bad_frames"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "0.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(img_dir / "1.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(img_dir / "2.png"), np.zeros((80, 64, 3), dtype=np.uint8))  # wrong height
    pipe_rerr = TrackflowPipeline(model="synthetic", detector=_SyntheticMotionDetector())
    with pytest.raises(ValueError, match="Expected a readable 64x64 image"):
        pipe_rerr.run(str(img_dir), output_dir=tmp_path / "run_rerr", save_video=False, exports=())
