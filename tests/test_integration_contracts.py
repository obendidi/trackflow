"""Contracts for embedding Trackflow, exporting observations, and reusing pipelines."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from trackflow import Detection, DwellZone, FloorplanHomography, Track, TrackflowPipeline
from trackflow.cli import app
from trackflow.inspector import compute_homography_projection, load_run_payload
from trackflow.pipeline import _source_label
from trackflow.sinks import ObservationWriter


class NestedObjects:
    def detect(self, frame_bgr):
        return [
            Detection((4, 4, 56, 60), 0.95, 0, "person"),
            Detection((12, 16, 32, 40), 0.91, 24, "backpack"),
        ]


class FrameCounter:
    """The two protocols need only detect(frame) and update(detections, frame_bgr)."""

    def __init__(self):
        self.count = 0

    def update(self, detections, frame_bgr=None):
        self.count += 1
        return [
            Track(self.count, d.bbox, d.confidence, d.class_id, d.class_name) for d in detections
        ]


def image_sequence(path, count):
    path.mkdir()
    for i in range(count):
        cv2.imwrite(str(path / f"{i}.png"), np.full((64, 64, 3), i, np.uint8))
    return path


def test_stream_matches_file_run_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipe = TrackflowPipeline(detector=NestedObjects(), classes="all")

    def frames():
        for _ in range(4):
            yield np.zeros((64, 64, 3), np.uint8)

    stream = list(pipe.stream(frames(), fps=20))
    assert not list(tmp_path.iterdir())
    assert [[t.track_id for t in f.tracks] for f in stream] == [[1, 2]] * 4
    assert stream[-1].timestamp_s == pytest.approx(0.15)
    source = image_sequence(tmp_path / "images", 4)
    run = pipe.run(
        source, fps=20, output_dir=tmp_path / "data", save_video=False, exports="jsonl,mot"
    )
    rows = [json.loads(line) for line in Path(run.output_files["jsonl"]).read_text().splitlines()]
    assert rows == [t.to_record(f.frame_idx, f.timestamp_s) for f in stream for t in f.tracks]
    mot = list(csv.reader(Path(run.output_files["mot"]).read_text().splitlines()))
    assert list(map(float, mot[0])) == [1, 1, 5, 5, 52, 56, 0.95, -1, -1, -1]
    assert mot[-1][0] == "4"
    assert json.loads(Path(run.output_files["summary"]).read_text()) == run.to_dict()


def test_custom_tracker_factory_owns_one_state_per_stream():
    pipe = TrackflowPipeline(
        detector=NestedObjects(), tracker_factory=FrameCounter, classes=["backpack"]
    )
    frames = [np.zeros((64, 64, 3), np.uint8)] * 3
    first, second = pipe.stream(frames), pipe.stream(frames)
    for stream, expected in [(first, 1), (first, 2), (second, 1), (first, 3)]:
        tracks = next(stream).tracks
        assert [(t.track_id, t.class_name) for t in tracks] == [(expected, "backpack")]


def test_output_isolation_and_concurrent_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = image_sequence(tmp_path / "images", 3)
    pipe = TrackflowPipeline(detector=NestedObjects())
    previous = pipe.run(source, output_dir="old", exports=["jsonl"])
    assert list(Path("old/crops").glob("*.jpg"))
    before = Path(previous.output_files["summary"]).read_bytes()
    private = TrackflowPipeline(detector=NestedObjects(), anonymize="blur")
    with pytest.raises(FileExistsError, match="fresh"):
        private.run(source, output_dir="old", save_video=False, exports=["csv"])
    assert Path(previous.output_files["summary"]).read_bytes() == before
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(
            pool.map(lambda _: private.run(source, save_video=False, exports=["csv"]), range(2))
        )
    assert len({r.output_files["summary"] for r in runs}) == 2
    for r in runs:
        out = Path(r.output_files["summary"]).parent
        assert not (out / "crops").exists() and not (out / "annotated.mp4").exists()
        assert load_run_payload(out)["diagnostics"]["total_observations"] == 3


def test_privacy_receives_people_even_when_tracking_other_classes(tmp_path, monkeypatch):
    calls = []

    def build(**kwargs):
        assert set(kwargs["classes"]) == {"backpack", "person"}
        return NestedObjects()

    monkeypatch.setattr("trackflow.pipeline.create_detector", build)
    pipe = TrackflowPipeline(classes="backpack", anonymize="face-blur")

    def apply(frame, tracks, *, detections, coasting):
        calls.append(([t.class_name for t in tracks], [d.class_name for d in detections]))
        return frame

    monkeypatch.setattr(pipe.anonymizer, "apply", apply)
    result = pipe.run(image_sequence(tmp_path / "images", 2), output_dir=tmp_path / "run")
    assert result.unique_tracks == 1 and calls == [(["backpack"], ["person", "backpack"])] * 2


def test_floorplan_and_inspector_respect_stride_and_separate_runs(tmp_path):
    pts = [(0, 0), (64, 0), (64, 64), (0, 64)]
    zone = DwellZone("room", pts)
    hom = FloorplanHomography(pts, pts, 64, 64, [zone])
    pipe = TrackflowPipeline(detector=NestedObjects(), homography=hom)
    source = image_sequence(tmp_path / "images", 6)
    for i in range(2):
        out = tmp_path / f"r{i}"
        r = pipe.run(source, output_dir=out, fps=30, skip=1, save_video=False, exports=["csv"])
        assert r.zone_dwell_totals_s["room"] == pytest.approx(0.2)
        data = compute_homography_projection(out, pts, 64, 64, [zone.to_dict()])
        assert data["zone_dwell_totals_s"] == {"room": 0.2}
        assert [f["frame_idx"] for f in load_run_payload(out)["frames"]] == [0, 2, 4]


def test_input_order_and_validation(tmp_path):
    seen = []

    class Empty:
        def detect(self, frame_bgr):
            seen.append(int(frame_bgr[0, 0, 0]))
            return []

    pipe = TrackflowPipeline(detector=Empty())
    src = image_sequence(tmp_path / "images", 12)
    run = pipe.run(src, output_dir=tmp_path / "ok", save_video=False, exports=["parquet"])
    assert seen == list(range(12))
    assert len(load_run_payload(Path(run.output_files["summary"]).parent)["frames"]) == 12
    for kwargs in ({"skip": -1}, {"max_frames": 0}, {"exports": ["typo"]}, {"fps": 0}):
        with pytest.raises(ValueError):
            pipe.run(src, output_dir=tmp_path / "bad", **kwargs)
    with pytest.raises(ValueError, match="uint8"):
        list(pipe.stream([np.zeros((4, 4), np.float32)]))
    assert (
        _source_label("rtsp://user:password@camera:554/lobby?token=secret")
        == "rtsp://camera:554/lobby"
    )


def test_cli_catalog_is_machine_readable():
    r = CliRunner().invoke(app, ["models", "--json"])
    assert r.exit_code == 0
    models = {row["name"]: row for row in json.loads(r.output)["detectors"]}
    assert models["rfdetr-large"]["backend"] == "transformers"
    assert models["deimv2-s"]["precisions"] == ["fp32"]
    assert models["yolo26n"]["license"].startswith("AGPL")
    assert all(len(row["revision"]) == 40 for row in models.values())


def test_batched_exports_keep_order_and_flush_the_last_partial_batch(tmp_path):
    paths = {fmt: tmp_path / fmt for fmt in ("jsonl", "parquet", "mot", "csv")}
    track = Track(1, (1, 2, 4, 8), 0.9, 0, "person")
    with ExitStack() as stack:
        sink = ObservationWriter(paths, stack)
        for idx in range(1027):
            sink.append(track.to_record(idx, idx / 30))
    parquet = pq.ParquetFile(paths["parquet"])
    assert parquet.num_row_groups == 2
    assert parquet.read(columns=["frame_idx"]).column(0).to_pylist() == list(range(1027))
    assert (
        len(paths["jsonl"].read_text().splitlines())
        == len(paths["mot"].read_text().splitlines())
        == 1027
    )
    assert len(paths["csv"].read_text().splitlines()) == 1028


def test_reid_merge_corrects_earlier_rows_in_all_exports(tmp_path):
    class Returning:
        track_buffer = 3

        def update(self, detections, frame_bgr=None):
            idx = int(frame_bgr[0, 0, 0])
            if 4 <= idx < 14:
                return []
            return [Track(10 if idx < 4 else 20, (0, 0, 60, 60), 0.9, 0, "person")]

    source = image_sequence(tmp_path / "images", 24)
    pipeline = TrackflowPipeline(
        detector=NestedObjects(), tracker_factory=Returning, reid="color", reid_merge=True
    )
    run = pipeline.run(
        source, output_dir=tmp_path / "out", save_video=False, exports="parquet,jsonl,csv,mot"
    )
    assert run.reid_merges == 1 and run.unique_tracks == 1
    assert set(pq.read_table(run.output_files["parquet"])["track_id"].to_pylist()) == {10}
    assert {
        json.loads(r)["track_id"] for r in Path(run.output_files["jsonl"]).read_text().splitlines()
    } == {10}
    assert {
        r["track_id"]
        for r in csv.DictReader(Path(run.output_files["csv"]).read_text().splitlines())
    } == {"10"}
    assert {r[1] for r in csv.reader(Path(run.output_files["mot"]).read_text().splitlines())} == {
        "10"
    }
