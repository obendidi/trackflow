"""Unit and end-to-end tests for offline post-processing (--postprocess)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from typer.testing import CliRunner

from trackflow import (
    CountingLine,
    Detection,
    DwellZone,
    FloorplanHomography,
    PostprocessResult,
    Track,
    TrackflowPipeline,
    postprocess_records,
    postprocess_tracks_array,
)
from trackflow.cli import app
from trackflow.inspector import load_run_payload


def _linear_track_rows(
    tid: int,
    frames: list[int],
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    w: float = 40.0,
    h: float = 100.0,
    conf: float = 0.85,
) -> list[list[float]]:
    out = []
    for f in frames:
        cx = x0 + vx * f
        cy = y0 + vy * f
        out.append([float(f), float(tid), cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h, conf])
    return out


def test_postprocess_fills_short_gap_and_respects_frame_stride() -> None:
    # Single person walking smoothly with a missing observation at frame 12 (stride=2 -> frames 0,2,..,20)
    frames = [f for f in range(0, 22, 2) if f not in (10, 12)]
    rows = np.array(_linear_track_rows(1, frames, x0=50.0, y0=150.0, vx=4.0, vy=0.0), dtype=np.float64)

    out = postprocess_tracks_array(
        rows, fps=20.0, frame_width=640, frame_height=480, frame_stride=2
    )
    out_frames = sorted(int(round(f)) for f in out[:, 0])
    assert out_frames == list(range(0, 22, 2))
    assert set(int(round(t)) for t in out[:, 1]) == {1}

    # Check filled midpoint at frame 10 sits near x=90 (cx=50+4*10=90 -> x1=70, x2=110)
    f10 = out[np.isclose(out[:, 0], 10.0)][0]
    cx10 = 0.5 * (f10[2] + f10[4])
    assert abs(cx10 - 90.0) < 2.0


def test_postprocess_stitches_fragmented_tracklet_and_isolates_classes() -> None:
    # Fragment 1 (frames 0..14) and Fragment 2 (frames 25..39) of the same walking person (class 0),
    # plus a different class (class 24) along a similar path that must NEVER merge with class 0.
    r1 = _linear_track_rows(1, list(range(0, 15)), x0=40.0, y0=180.0, vx=5.0, vy=0.0)
    r2 = _linear_track_rows(2, list(range(25, 40)), x0=40.0, y0=180.0, vx=5.0, vy=0.0)
    r_other = _linear_track_rows(3, list(range(25, 40)), x0=42.0, y0=180.0, vx=5.0, vy=0.0)

    rows = np.array(r1 + r2 + r_other, dtype=np.float64)
    class_ids = np.array([0] * len(r1) + [0] * len(r2) + [24] * len(r_other), dtype=np.int64)
    out = postprocess_tracks_array(
        rows, fps=30.0, frame_width=640, frame_height=480, class_ids=class_ids
    )
    # Track 1 and Track 2 merge into 1 track (with gap filled), while class 24 stays separate
    unique_tids = sorted(set(int(round(t)) for t in out[:, 1]))
    assert len(unique_tids) == 2

    # Ensure zero duplicate (frame_idx, track_id) pairs and positive boxes
    pairs = set(zip(np.rint(out[:, 0]).astype(int), np.rint(out[:, 1]).astype(int), strict=True))
    assert len(pairs) == len(out)
    assert np.all(out[:, 4] > out[:, 2]) and np.all(out[:, 5] > out[:, 3])


def test_postprocess_repairs_two_track_crossing_swap() -> None:
    # Person A walks left->right (x: 40 -> 340); Person B walks right->left (x: 340 -> 40).
    # They cross around frame 25, and the raw online tracker swaps IDs 1 <-> 2 at frame 25!
    rows = []
    for f in range(51):
        xa = 40.0 + 6.0 * f
        xb = 340.0 - 6.0 * f
        if f < 25:
            tid_a, tid_b = 1, 2
        else:
            tid_a, tid_b = 2, 1  # online tracker swapped IDs at crossing!
        rows.append([float(f), float(tid_a), xa - 20.0, 100.0, xa + 20.0, 200.0, 0.9])
        rows.append([float(f), float(tid_b), xb - 20.0, 100.0, xb + 20.0, 200.0, 0.9])

    out = postprocess_tracks_array(np.array(rows, dtype=np.float64), fps=30.0, frame_width=640, frame_height=480)
    # After un-crossing, the track starting on the left (x=40 at f=0) should end on the right (x=340 at f=50)
    f0 = out[np.isclose(out[:, 0], 0.0)]
    f50 = out[np.isclose(out[:, 0], 50.0)]
    left_tid = int(round(f0[np.argmin(f0[:, 2]), 1]))
    end_row = f50[np.isclose(f50[:, 1], left_tid)][0]
    end_cx = 0.5 * (end_row[2] + end_row[4])
    assert end_cx > 300.0


def test_postprocess_records_preserves_schema_and_long_reid_merge() -> None:
    # Two distant non-overlapping appearances (gap 60 frames) with distinct strong embeddings
    recs: list[dict] = []
    embs: list[np.ndarray] = []
    e_red = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_blue = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    for f in range(10):
        recs.append(Track(10, (20, 40, 80, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_red)
        recs.append(Track(11, (220, 40, 280, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_blue)
    # Red person returns 60 frames later (far beyond short kinematic window) as raw ID 20
    for f in range(70, 80):
        recs.append(Track(20, (300, 40, 360, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_red)

    res = postprocess_records(
        recs,
        fps=10.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs),
        emb_valid=np.ones(len(embs), dtype=bool),
        allow_long_reid_merge=True,
        min_exit_gap_frames=8,
    )
    assert isinstance(res, PostprocessResult)
    assert res.unique_tracks_before == 3
    assert res.unique_tracks_after == 2
    assert res.merges >= 1
    assert len(res.source_row_indices) == len(res.records)


def _write_image_seq(path: Path, count: int, width: int = 320, height: int = 240) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        cv2.rectangle(frame, (20, 40), (90, 190), (10, 10, 230), -1)
        cv2.rectangle(frame, (200, 40), (260, 190), (10, 10, 230), -1)
        cv2.imwrite(str(path / f"{i:03d}.png"), frame)
    return path


def test_pipeline_postprocess_end_to_end_privacy_lines_and_stream_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    src = _write_image_seq(tmp_path / "seq", 30)

    class _GapDetector:
        active_provider = "MockCPU"

        def __init__(self) -> None:
            self.step = 0

        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            _ = frame_bgr
            f = self.step
            self.step += 1
            # Walk across the counting line x=140 from left to right; drop frames 10..12
            if 10 <= f <= 12:
                return []
            x = 20.0 + f * 6.0
            return [Detection((x, 40.0, x + 50.0, 190.0), 0.92, 0, "person")]

    class _ReturningGapTracker:
        track_buffer = 3
        last_cmc_warp = None

        def __init__(self) -> None:
            self.f = 0

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = frame_bgr
            idx = self.f
            self.f += 1
            if not detections:
                return []
            # Fragment into two IDs before/after the gap so postprocess must stitch & interpolate
            tid = 1 if idx < 10 else 2
            d = detections[0]
            return [Track(tid, d.bbox, d.confidence, d.class_id, d.class_name)]

    pts = [(0, 0), (320, 0), (320, 240), (0, 240)]
    hom = FloorplanHomography(pts, [(0, 0), (10, 0), (10, 8), (0, 8)], 10.0, 8.0, [DwellZone("all", [(0, 0), (10, 0), (10, 8), (0, 8)])])
    line = CountingLine("gate", (140.0, 0.0), (140.0, 240.0))

    # 1. Live stream() remains strictly frame-by-frame online even when postprocess=True
    pipe_stream = TrackflowPipeline(
        model="mock",
        detector=_GapDetector(),
        tracker_factory=_ReturningGapTracker,
        postprocess=True,
    )
    stream_ids = [
        [t.track_id for t in fr.tracks]
        for fr in pipe_stream.stream([np.zeros((240, 320, 3), dtype=np.uint8)] * 20, fps=20.0)
    ]
    assert [1] in stream_ids and [2] in stream_ids and [] in stream_ids

    # 2. Offline run() with postprocess=True + anonymize + save_video + floorplan + line + all exports
    out_dir = tmp_path / "pp_run"
    anon_calls: list[tuple[int, int]] = []
    pipe = TrackflowPipeline(
        model="mock",
        detector=_GapDetector(),
        tracker_factory=_ReturningGapTracker,
        postprocess=True,
        anonymize="face-blur",
        homography=hom,
        lines=[line],
    )
    orig_apply = pipe.anonymizer.apply

    def _spy_apply(frame, tracks, *, detections=(), coasting=()):
        anon_calls.append((len(tracks), len(detections)))
        return orig_apply(frame, tracks, detections=detections, coasting=coasting)

    monkeypatch.setattr(pipe.anonymizer, "apply", _spy_apply)

    summary = pipe.run(
        src,
        output_dir=out_dir,
        fps=20.0,
        save_video=True,
        exports=("parquet", "jsonl", "csv", "mot", "html"),
    )
    assert summary.postprocessed is True
    assert summary.unique_tracks == 1
    assert summary.total_track_observations == 30  # 3 missing frames filled cleanly!
    assert (out_dir / "annotated.mp4").is_file()
    assert (out_dir / "floorplan.png").is_file()
    assert (out_dir / "report.html").is_file()
    # Privacy: zero crops on anonymized run, and Pass 2 anonymizer ran on all 30 frames with raw dets
    assert not (out_dir / "crops").exists()
    assert len(anon_calls) == 30
    assert sum(d for _, d in anon_calls) == 27  # 27 frames had raw detections
    cards = json.loads((out_dir / "track_cards.json").read_text(encoding="utf-8"))["cards"]
    assert len(cards) == 1 and cards[0]["thumbnail"] is None
    assert "crops/" not in (out_dir / "report.html").read_text(encoding="utf-8")

    # All export formats agree on single stitched ID = 1 and 30 observations
    pq_tbl = pq.read_table(out_dir / "tracks.parquet")
    assert pq_tbl.num_rows == 30 and set(pq_tbl["track_id"].to_pylist()) == {1}
    assert len(list(csv.reader((out_dir / "tracks.mot" if False else out_dir / "tracks.txt").read_text().splitlines()))) == 30
    events = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
    pp_ev = [e for e in events if e.get("type") == "offline_postprocess"]
    assert len(pp_ev) == 1 and pp_ev[0]["interpolated_rows"] == 3
    assert summary.line_counts["gate"]["out"] + summary.line_counts["gate"]["in"] == 1
    assert load_run_payload(out_dir)["diagnostics"]["total_observations"] == 30

    # 3. Unification with --reid-merge and no false line crossing across long exit/re-entry
    class _ExitReentryTracker:
        track_buffer = 3

        def __init__(self) -> None:
            self.f = 0

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = detections, frame_bgr
            idx = self.f
            self.f += 1
            if idx < 6:
                return [Track(10, (20, 40, 80, 190), 0.9, 0, "person")]  # left of x=140
            if idx >= 22:
                return [Track(20, (200, 40, 260, 190), 0.9, 0, "person")]  # right of x=140 after exit
            return []

    out_reid = tmp_path / "pp_reid_merge"
    pipe_rm = TrackflowPipeline(
        model="mock",
        detector=_GapDetector(),
        tracker_factory=_ExitReentryTracker,
        reid="color",
        reid_merge=True,
        postprocess=True,
        lines=[line],
    )
    sum_rm = pipe_rm.run(src, output_dir=out_reid, fps=10.0, save_video=False, exports=("jsonl",))
    assert sum_rm.unique_tracks == 1
    assert sum_rm.reid_merges >= 1
    # Must NOT invent a fake line crossing across the 1.6s exit/re-entry gap!
    assert sum_rm.line_counts["gate"] == {"in": 0, "out": 0}


def test_cli_postprocess_flag(tmp_path: Path, monkeypatch) -> None:
    src = _write_image_seq(tmp_path / "cli_seq", 8)

    class _Det:
        active_provider = "MockCPU"

        def detect(self, frame_bgr):
            _ = frame_bgr
            return [Detection((20, 40, 80, 180), 0.9, 0, "person")]

    monkeypatch.setattr("trackflow.pipeline.create_detector", lambda **_kw: _Det())
    runner = CliRunner()
    out_dir = tmp_path / "cli_out"
    res = runner.invoke(
        app,
        ["run", str(src), "--postprocess", "--no-video", "--export", "jsonl", "-o", str(out_dir)],
    )
    assert res.exit_code == 0, res.output
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["postprocessed"] is True
