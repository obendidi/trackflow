"""Unit and end-to-end tests for offline post-processing (--postprocess)."""

from __future__ import annotations

import csv
import json
import signal
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import trackflow.pipeline as pipeline_mod
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


def _write_image_seq(path: Path, count: int, width: int = 320, height: int = 240) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        cv2.rectangle(frame, (20, 40), (90, 190), (10, 10, 230), -1)
        cv2.rectangle(frame, (200, 40), (260, 190), (10, 10, 230), -1)
        cv2.imwrite(str(path / f"{i:03d}.png"), frame)
    return path


def _count_video_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        return n
    finally:
        cap.release()


def test_postprocess_fills_short_gap_and_respects_frame_stride() -> None:
    frames = [f for f in range(0, 22, 2) if f not in (10, 12)]
    rows = np.array(_linear_track_rows(1, frames, x0=50.0, y0=150.0, vx=4.0, vy=0.0), dtype=np.float64)

    out = postprocess_tracks_array(
        rows, fps=20.0, frame_width=640, frame_height=480, frame_stride=2
    )
    assert sorted(int(round(f)) for f in out[:, 0]) == list(range(0, 22, 2))
    assert set(int(round(t)) for t in out[:, 1]) == {1}

    f10 = out[np.isclose(out[:, 0], 10.0)][0]
    assert abs(0.5 * (f10[2] + f10[4]) - 90.0) < 2.0


def test_postprocess_stitches_fragmented_tracklet_and_isolates_classes() -> None:
    r1 = _linear_track_rows(1, list(range(0, 15)), x0=40.0, y0=180.0, vx=5.0, vy=0.0)
    r2 = _linear_track_rows(2, list(range(25, 40)), x0=40.0, y0=180.0, vx=5.0, vy=0.0)
    r_other = _linear_track_rows(3, list(range(25, 40)), x0=42.0, y0=180.0, vx=5.0, vy=0.0)

    rows = np.array(r1 + r2 + r_other, dtype=np.float64)
    class_ids = np.array([0] * len(r1) + [0] * len(r2) + [24] * len(r_other), dtype=np.int64)
    out = postprocess_tracks_array(
        rows, fps=30.0, frame_width=640, frame_height=480, class_ids=class_ids
    )
    assert len(set(int(round(t)) for t in out[:, 1])) == 2
    pairs = set(zip(np.rint(out[:, 0]).astype(int), np.rint(out[:, 1]).astype(int), strict=True))
    assert len(pairs) == len(out)
    assert np.all(out[:, 4] > out[:, 2]) and np.all(out[:, 5] > out[:, 3])


def test_postprocess_repairs_two_track_crossing_swap() -> None:
    rows = []
    for f in range(51):
        xa = 40.0 + 6.0 * f
        xb = 340.0 - 6.0 * f
        tid_a, tid_b = (1, 2) if f < 25 else (2, 1)
        rows.append([float(f), float(tid_a), xa - 20.0, 100.0, xa + 20.0, 200.0, 0.9])
        rows.append([float(f), float(tid_b), xb - 20.0, 100.0, xb + 20.0, 200.0, 0.9])

    out = postprocess_tracks_array(
        np.array(rows, dtype=np.float64), fps=30.0, frame_width=640, frame_height=480
    )
    f0 = out[np.isclose(out[:, 0], 0.0)]
    f50 = out[np.isclose(out[:, 0], 50.0)]
    left_tid = int(round(f0[np.argmin(f0[:, 2]), 1]))
    end_row = f50[np.isclose(f50[:, 1], left_tid)][0]
    assert 0.5 * (end_row[2] + end_row[4]) > 300.0


def test_postprocess_empty_and_input_validation() -> None:
    empty_res = postprocess_records([])
    assert isinstance(empty_res, PostprocessResult)
    assert empty_res.records == []
    assert empty_res.unique_tracks_before == 0
    assert empty_res.unique_tracks_after == 0
    assert empty_res.reid_merges == 0
    assert empty_res.reid_merge_events == []
    assert postprocess_tracks_array(np.zeros((0, 7), dtype=np.float64)).shape == (0, 7)

    neg_rows = np.array([[-1.0, 1.0, 10.0, 20.0, 50.0, 120.0, 0.9]], dtype=np.float64)
    with pytest.raises(ValueError, match="nonnegative"):
        postprocess_tracks_array(neg_rows)

    valid_rows = np.array(_linear_track_rows(1, [0, 1, 2], 50.0, 150.0, 2.0, 0.0), dtype=np.float64)
    with pytest.raises(ValueError, match="embeddings length"):
        postprocess_tracks_array(valid_rows, embeddings=np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="emb_valid length"):
        postprocess_tracks_array(
            valid_rows,
            embeddings=np.ones((3, 4), dtype=np.float32),
            emb_valid=np.ones((2,), dtype=bool),
        )
    with pytest.raises(ValueError, match="class_ids length"):
        postprocess_tracks_array(valid_rows, class_ids=np.array([0, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="cmc_valid length"):
        postprocess_tracks_array(
            valid_rows,
            cmc_warps=np.zeros((3, 2, 3), dtype=np.float32),
            cmc_valid=np.ones((2,), dtype=bool),
        )


def test_postprocess_embeddings_default_emb_valid_and_list_coercion() -> None:
    e_red = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_blue = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    recs: list[dict] = []
    embs: list[np.ndarray] = []
    for f in range(10):
        recs.append(Track(1, (20, 40, 80, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_red)
        recs.append(Track(2, (220, 40, 280, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_blue)
    for f in range(70, 80):
        recs.append(Track(3, (300, 40, 360, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_red)

    res_none = postprocess_records(
        recs,
        fps=10.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs),
        emb_valid=None,
        allow_long_reid_merge=True,
        min_exit_gap_frames=8,
    )
    assert res_none.unique_tracks_after == 2 and res_none.reid_merges == 1

    # Passing a Python list of ints for emb_valid must coerce cleanly to bool
    res_list = postprocess_records(
        recs,
        fps=10.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs),
        emb_valid=[1] * len(embs),  # type: ignore[arg-type]
        allow_long_reid_merge=True,
        min_exit_gap_frames=8,
    )
    assert res_list.unique_tracks_after == 2 and res_list.reid_merges == 1


def test_postprocess_long_horizon_continues_when_medium_pass_empty() -> None:
    e_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    e_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rows: list[list[float]] = []
    embs: list[np.ndarray] = []

    for f in range(200):
        rows.append([float(f), 1.0, 40.0, 60.0, 80.0, 160.0, 0.92])
        embs.append(e_a)
        rows.append([float(f), 2.0, 460.0, 60.0, 500.0, 160.0, 0.92])
        embs.append(e_b)

    # Person C disappears at frame 30 and reappears 4.0s (120 frames) later with no <=2.5s gaps.
    for f in range(0, 31):
        rows.append([float(f), 3.0, 220.0, 200.0, 260.0, 300.0, 0.92])
        embs.append(e_c)
    for f in range(150, 185):
        rows.append([float(f), 4.0, 240.0, 200.0, 280.0, 300.0, 0.92])
        embs.append(e_c)

    out = postprocess_tracks_array(
        np.array(rows, dtype=np.float64),
        fps=30.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs),
    )
    assert len(set(int(round(t)) for t in out[:, 1])) == 3


def test_postprocess_cross_exit_reid_chains_noisy_three_segment_reentry() -> None:
    # Noisy embeddings where sim(A, C) ~= 0.99 > sim(B, C) ~= 0.97 > sim(A, B) ~= 0.92:
    # union-find must still chain A -> B -> C instead of orphaning middle segment B.
    def _unit(v: list[float]) -> np.ndarray:
        a = np.array(v, dtype=np.float32)
        return a / float(np.linalg.norm(a))

    e_a = _unit([1.0, 0.0, 0.0, 0.0])
    e_c = _unit([0.99, 0.141, 0.0, 0.0])
    e_b = _unit([0.92, 0.392, 0.0, 0.0])
    e_other = _unit([0.0, 0.0, 1.0, 0.0])
    assert float(e_a @ e_c) > float(e_b @ e_c) > float(e_a @ e_b) > 0.90

    recs: list[dict] = []
    embs: list[np.ndarray] = []
    for seg_tid, start_f, x0, vec in (
        (10, 0, 20, e_a),
        (20, 45, 220, e_b),
        (30, 90, 420, e_c),
    ):
        for f in range(start_f, start_f + 10):
            recs.append(Track(seg_tid, (x0, 40, x0 + 60, 180), 0.9, 0, "person").to_record(f, f / 10.0))
            embs.append(vec)
    for f in range(0, 10):
        recs.append(Track(11, (520, 40, 580, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_other)

    res = postprocess_records(
        recs,
        fps=10.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs),
        allow_long_reid_merge=True,
        min_exit_gap_frames=8,
    )
    assert res.unique_tracks_before == 4
    assert res.unique_tracks_after == 2
    assert res.reid_merges == 2
    assert len(res.reid_merge_events) == 2


def test_postprocess_cross_exit_reid_rejects_overlapping_lookalikes() -> None:
    e_red = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_la1 = np.array([0.995, 0.09987, 0.0, 0.0], dtype=np.float32)
    e_la1 /= np.linalg.norm(e_la1)
    e_la2 = np.array([0.990, 0.14107, 0.0, 0.0], dtype=np.float32)
    e_la2 /= np.linalg.norm(e_la2)

    recs: list[dict] = []
    embs: list[np.ndarray] = []
    for f in range(10):
        recs.append(Track(1, (20, 40, 80, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_red)
    for f in range(50, 60):
        recs.append(Track(2, (200, 40, 260, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_la1)
        recs.append(Track(3, (400, 40, 460, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_la2)

    res = postprocess_records(
        recs,
        fps=10.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs),
        allow_long_reid_merge=True,
        min_exit_gap_frames=8,
    )
    assert res.unique_tracks_after == 3
    assert res.reid_merges == 0


def test_postprocess_cross_exit_reid_rejects_single_linkage_bridge() -> None:
    # Complete-linkage guard: sim(A, C) ~= 0.86 and sim(B, C) ~= 0.84 both clear 0.80,
    # but sim(A, B) ~= 0.55 < 0.80 so B must NOT be bridged into A+C.
    def _unit(v: list[float]) -> np.ndarray:
        a = np.array(v, dtype=np.float32)
        return a / float(np.linalg.norm(a))

    e_c = _unit([1.0, 0.0, 0.0, 0.0])
    e_a = _unit([0.86, 0.5103, 0.0, 0.0])
    e_b = _unit([0.84, -0.34, 0.423, 0.0])
    assert float(e_a @ e_c) > 0.85 and float(e_b @ e_c) > 0.83 and float(e_a @ e_b) < 0.60

    recs: list[dict] = []
    embs: list[np.ndarray] = []
    for seg_tid, start_f, x0, vec in (
        (1, 0, 20, e_a),
        (2, 45, 220, e_b),
        (3, 90, 420, e_c),
    ):
        for f in range(start_f, start_f + 10):
            recs.append(Track(seg_tid, (x0, 40, x0 + 60, 180), 0.9, 0, "person").to_record(f, f / 10.0))
            embs.append(vec)

    res = postprocess_records(
        recs,
        fps=10.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs),
        allow_long_reid_merge=True,
        min_exit_gap_frames=8,
    )
    assert res.unique_tracks_after == 2
    assert res.reid_merges == 1
    assert all(ev["similarity"] >= 0.80 for ev in res.reid_merge_events)


def test_postprocess_low_fps_horizon_scales_in_seconds() -> None:
    r1 = _linear_track_rows(1, list(range(0, 10)), x0=100.0, y0=150.0, vx=0.0, vy=0.0)
    r_far = _linear_track_rows(2, list(range(70, 80)), x0=102.0, y0=150.0, vx=0.0, vy=0.0)
    out_far = postprocess_tracks_array(
        np.array(r1 + r_far, dtype=np.float64), fps=5.0, frame_width=640, frame_height=480
    )
    assert len(set(int(round(t)) for t in out_far[:, 1])) == 2

    r_near = _linear_track_rows(2, list(range(19, 29)), x0=102.0, y0=150.0, vx=0.0, vy=0.0)
    out_near = postprocess_tracks_array(
        np.array(r1 + r_near, dtype=np.float64), fps=5.0, frame_width=640, frame_height=480
    )
    assert len(set(int(round(t)) for t in out_near[:, 1])) == 1


def test_postprocess_cross_class_overlap_and_stride8_crossing() -> None:
    rows_cc: list[list[float]] = []
    cids_cc: list[int] = []
    for f in range(51):
        xa = 40.0 + 6.0 * f
        xb = 340.0 - 6.0 * f
        rows_cc.append([float(f), 1.0, xa - 20.0, 100.0, xa + 20.0, 200.0, 0.9])
        cids_cc.append(0)
        rows_cc.append([float(f), 2.0, xb - 20.0, 100.0, xb + 20.0, 200.0, 0.9])
        cids_cc.append(2)
    out_cc = postprocess_tracks_array(
        np.array(rows_cc, dtype=np.float64),
        fps=30.0,
        frame_width=640,
        frame_height=480,
        class_ids=np.array(cids_cc, dtype=np.int64),
    )
    assert len(set(int(round(t)) for t in out_cc[:, 1])) == 2

    rows_st: list[list[float]] = []
    for step in range(41):
        f = step * 8
        xa = 40.0 + 7.0 * step
        xb = 320.0 - 7.0 * step
        tid_a, tid_b = (1, 2) if step < 20 else (2, 1)
        rows_st.append([float(f), float(tid_a), xa - 20.0, 100.0, xa + 20.0, 200.0, 0.9])
        rows_st.append([float(f), float(tid_b), xb - 20.0, 100.0, xb + 20.0, 200.0, 0.9])
    out_st = postprocess_tracks_array(
        np.array(rows_st, dtype=np.float64),
        fps=30.0,
        frame_width=640,
        frame_height=480,
        frame_stride=8,
    )
    f0 = out_st[np.isclose(out_st[:, 0], 0.0)]
    f_end = out_st[np.isclose(out_st[:, 0], 40.0 * 8.0)]
    left_tid = int(round(f0[np.argmin(f0[:, 2]), 1]))
    end_row = f_end[np.isclose(f_end[:, 1], left_tid)][0]
    assert 0.5 * (end_row[2] + end_row[4]) > 280.0


def test_pipeline_postprocess_e2e_exports_privacy_events_and_progressive_overlay(
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
            if 10 <= f <= 12:
                return []
            x = 20.0 + f * 6.0
            return [Detection((x, 40.0, x + 50.0, 190.0), 0.92, 0, "person")]

    class _ReturningGapTracker:
        track_buffer = 3
        last_cmc_warp = None
        last_events: list[dict] = []

        def __init__(self) -> None:
            self.f = 0

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = frame_bgr
            idx = self.f
            self.f += 1
            self.last_events = []
            if not detections:
                return []
            tid = 7 if idx < 10 else 9
            if idx == 0:
                self.last_events = [{"type": "track_started", "track_id": 7}]
            elif idx == 13:
                self.last_events = [
                    {"type": "possible_id_switch", "new_track_id": 9, "previous_track_id": 7}
                ]
            d = detections[0]
            return [Track(tid, d.bbox, d.confidence, d.class_id, d.class_name)]

    pts = [(0, 0), (320, 0), (320, 240), (0, 240)]
    hom = FloorplanHomography(
        pts,
        [(0, 0), (10, 0), (10, 8), (0, 8)],
        10.0,
        8.0,
        [DwellZone("all", [(0, 0), (10, 0), (10, 8), (0, 8)])],
    )
    line = CountingLine("gate", (140.0, 0.0), (140.0, 240.0))

    # Live stream() remains strictly online even when postprocess=True
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
    assert [7] in stream_ids and [9] in stream_ids and [] in stream_ids

    out_dir = tmp_path / "pp_run"
    anon_calls: list[tuple[int, int]] = []
    rendered_counts: list[dict[str, dict[str, int]] | None] = []
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
    orig_draw = pipeline_mod.draw_tracks

    def _spy_apply(frame, tracks, *, detections=(), coasting=()):
        anon_calls.append((len(tracks), len(detections)))
        return orig_apply(frame, tracks, detections=detections, coasting=coasting)

    def _spy_draw(*args, **kwargs):
        lc = kwargs.get("line_counts")
        rendered_counts.append({k: dict(v) for k, v in lc.items()} if lc else None)
        return orig_draw(*args, **kwargs)

    monkeypatch.setattr(pipe.anonymizer, "apply", _spy_apply)
    monkeypatch.setattr(pipeline_mod, "draw_tracks", _spy_draw)

    summary = pipe.run(
        src,
        output_dir=out_dir,
        fps=20.0,
        save_video=True,
        exports=("parquet", "jsonl", "csv", "mot", "html"),
    )
    assert summary.postprocessed is True
    assert summary.unique_tracks == 1
    assert summary.reid_merges == 0
    assert summary.total_track_observations == 30
    assert _count_video_frames(out_dir / "annotated.mp4") == 30
    assert (out_dir / "floorplan.png").is_file()
    assert (out_dir / "report.html").is_file()
    assert not (out_dir / "crops").exists()
    assert len(anon_calls) == 30 and sum(d for _, d in anon_calls) == 27
    assert len(rendered_counts) == 30
    assert rendered_counts[0] == {"gate": {"in": 0, "out": 0}}
    assert sum(rendered_counts[-1]["gate"].values()) == 1

    cards = json.loads((out_dir / "track_cards.json").read_text(encoding="utf-8"))["cards"]
    assert len(cards) == 1 and cards[0]["thumbnail"] is None and cards[0]["merged_track_ids"] == []
    assert "crops/" not in (out_dir / "report.html").read_text(encoding="utf-8")

    pq_tbl = pq.read_table(out_dir / "tracks.parquet")
    assert pq_tbl.num_rows == 30 and set(pq_tbl["track_id"].to_pylist()) == {1}
    assert len(list(csv.reader((out_dir / "tracks.txt").read_text().splitlines()))) == 30

    events = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
    frame_seq = [int(e.get("frame_idx", 0)) for e in events]
    assert frame_seq == sorted(frame_seq)
    started = [e for e in events if e.get("type") == "track_started"]
    assert len(started) == 1 and started[0]["track_id"] == 1
    assert not [e for e in events if e.get("type") == "possible_id_switch"]
    pp_ev = [e for e in events if e.get("type") == "offline_postprocess"]
    assert len(pp_ev) == 1 and pp_ev[0]["interpolated_rows"] == 3 and pp_ev[0]["reid_merges"] == 0
    assert summary.line_counts["gate"]["out"] + summary.line_counts["gate"]["in"] == 1
    assert load_run_payload(out_dir)["diagnostics"]["total_observations"] == 30


def test_pipeline_postprocess_reid_merge_unification_suppresses_fake_line_crossing(
    tmp_path: Path,
) -> None:
    src = _write_image_seq(tmp_path / "seq", 30)
    line = CountingLine("gate", (140.0, 0.0), (140.0, 240.0))

    class _Det:
        active_provider = "MockCPU"

        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            _ = frame_bgr
            return []

    class _ExitReentryTracker:
        track_buffer = 3

        def __init__(self) -> None:
            self.f = 0

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = detections, frame_bgr
            idx = self.f
            self.f += 1
            if idx < 9:
                return [Track(10, (20, 40, 80, 190), 0.9, 0, "person")]
            if idx >= 21:
                return [Track(20, (200, 40, 260, 190), 0.9, 0, "person")]
            return []

    out_reid = tmp_path / "pp_reid_merge"
    pipe_rm = TrackflowPipeline(
        model="mock",
        detector=_Det(),
        tracker_factory=_ExitReentryTracker,
        reid="color",
        reid_merge=True,
        postprocess=True,
        lines=[line],
    )
    sum_rm = pipe_rm.run(src, output_dir=out_reid, fps=10.0, save_video=False, exports=("jsonl",))
    assert sum_rm.unique_tracks == 1
    assert sum_rm.reid_merges == 1
    rm_events = json.loads((out_reid / "events.json").read_text(encoding="utf-8"))
    assert len([e for e in rm_events if e.get("type") == "reid_cross_exit_merge"]) == 1
    assert sum_rm.line_counts["gate"] == {"in": 0, "out": 0}


def test_pipeline_postprocess_source_validation(tmp_path: Path) -> None:
    class _Det:
        active_provider = "MockCPU"

        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            _ = frame_bgr
            return []

    pipe = TrackflowPipeline(model="mock", detector=_Det(), postprocess=True)
    with pytest.raises(ValueError, match="recorded video"):
        pipe.run("rtsp://example.com/live", output_dir=tmp_path / "rtsp_out", save_video=False)
    with pytest.raises(ValueError, match="recorded video"):
        pipe.run(0, output_dir=tmp_path / "cam_out", save_video=False)
    with pytest.raises(FileNotFoundError):
        pipe.run(tmp_path / "missing.mp4", output_dir=tmp_path / "miss_out", save_video=False)


def test_pipeline_postprocess_line_keeps_same_raw_id_occlusion_and_interpolated_stitch(
    tmp_path: Path,
) -> None:
    src = _write_image_seq(tmp_path / "occ_seq", 35)
    line = CountingLine("gate", (140.0, 0.0), (140.0, 240.0))

    class _Det:
        active_provider = "MockCPU"

        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            _ = frame_bgr
            return []

    # 1. Same raw ID occluded for 0.7s (14 frames @ 20fps) across line still counts
    class _OccludedSameIdTracker:
        track_buffer = 30

        def __init__(self) -> None:
            self.f = 0

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = detections, frame_bgr
            idx = self.f
            self.f += 1
            if idx < 8:
                return [Track(1, (80.0, 40.0, 120.0, 190.0), 0.9, 0, "person")]
            if idx >= 22:
                return [Track(1, (160.0, 40.0, 200.0, 190.0), 0.9, 0, "person")]
            return []

    pipe1 = TrackflowPipeline(
        model="mock",
        detector=_Det(),
        tracker_factory=_OccludedSameIdTracker,
        postprocess=True,
        lines=[line],
    )
    s1 = pipe1.run(src, output_dir=tmp_path / "same_raw", fps=20.0, save_video=False, exports=("jsonl",))
    assert s1.line_counts["gate"]["in"] + s1.line_counts["gate"]["out"] == 1

    # 2. Different raw IDs bridged by continuous Stage-3 interpolation across >= min_exit_gap
    # (track_buffer=5 -> min_exit_gap=8, 9-frame interpolated crossing) still counts
    class _InterpStitchedCrossTracker:
        track_buffer = 5

        def __init__(self) -> None:
            self.f = 0

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = detections, frame_bgr
            idx = self.f
            self.f += 1
            if 10 <= idx <= 18:
                return []
            x = 70.0 + idx * 5.0
            tid = 1 if idx < 10 else 2
            return [Track(tid, (x, 40.0, x + 40.0, 190.0), 0.92, 0, "person")]

    pipe2 = TrackflowPipeline(
        model="mock",
        detector=_Det(),
        tracker_factory=_InterpStitchedCrossTracker,
        postprocess=True,
        lines=[line],
    )
    s2 = pipe2.run(src, output_dir=tmp_path / "interp_stitch", fps=20.0, save_video=False, exports=("jsonl",))
    assert s2.unique_tracks == 1
    assert s2.total_track_observations == 35
    assert s2.line_counts["gate"]["in"] + s2.line_counts["gate"]["out"] == 1


def test_pipeline_postprocess_ctrl_c_in_pass1_renders_processed_frames(tmp_path: Path) -> None:
    src = _write_image_seq(tmp_path / "p1_ctrlc", 20)

    class _InterruptPass1Detector:
        active_provider = "MockCPU"

        def __init__(self) -> None:
            self.step = 0

        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            _ = frame_bgr
            self.step += 1
            if self.step == 6:
                signal.raise_signal(signal.SIGINT)
            return [Detection((40.0, 40.0, 90.0, 190.0), 0.92, 0, "person")]

    out_dir = tmp_path / "p1_out"
    pipe = TrackflowPipeline(model="mock", detector=_InterruptPass1Detector(), postprocess=True)
    summary = pipe.run(src, output_dir=out_dir, fps=20.0, save_video=True, exports=("jsonl",))
    assert summary.processed_frames == 6
    assert "video" in summary.output_files
    assert _count_video_frames(out_dir / "annotated.mp4") == 6
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "track_cards.json").is_file()
    assert (out_dir / "events.json").is_file()


def test_pipeline_postprocess_ctrl_c_in_pass2_preserves_artifacts_and_unlinks_empty_video(
    tmp_path: Path, monkeypatch
) -> None:
    src = _write_image_seq(tmp_path / "p2_ctrlc", 20)

    class _Det:
        active_provider = "MockCPU"

        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            _ = frame_bgr
            return [Detection((40.0, 40.0, 90.0, 190.0), 0.92, 0, "person")]

    orig_draw = pipeline_mod.draw_tracks

    # 1. SIGINT on Pass-2 frame 5 writes a valid 5-frame MP4 plus all JSON artifacts
    out_mid = tmp_path / "p2_mid"
    replay_frames = 0

    def _interrupt_mid(*args, **kwargs):
        nonlocal replay_frames
        replay_frames += 1
        if replay_frames == 5:
            signal.raise_signal(signal.SIGINT)
        return orig_draw(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "draw_tracks", _interrupt_mid)
    pipe_mid = TrackflowPipeline(model="mock", detector=_Det(), postprocess=True)
    s_mid = pipe_mid.run(src, output_dir=out_mid, fps=20.0, save_video=True, exports=("jsonl",))
    assert replay_frames == 5
    assert _count_video_frames(out_mid / "annotated.mp4") == 5
    assert (out_mid / "summary.json").is_file()
    assert (out_mid / "track_cards.json").is_file()
    assert (out_mid / "events.json").is_file()
    assert s_mid.total_track_observations == 20

    # 2. SIGINT before frame 0 is written unlinks the empty MP4 and omits "video" from summary
    monkeypatch.setattr(pipeline_mod, "draw_tracks", orig_draw)
    out_zero = tmp_path / "p2_zero"
    orig_iter = pipeline_mod._iter_frames
    iter_calls = 0

    def _interrupt_before_first_replay(*args, **kwargs):
        nonlocal iter_calls
        iter_calls += 1
        gen, rate, w, h, total = orig_iter(*args, **kwargs)
        if iter_calls == 2:
            def _wrapped():
                signal.raise_signal(signal.SIGINT)
                yield from gen
            return _wrapped(), rate, w, h, total
        return gen, rate, w, h, total

    monkeypatch.setattr(pipeline_mod, "_iter_frames", _interrupt_before_first_replay)
    pipe_zero = TrackflowPipeline(model="mock", detector=_Det(), postprocess=True)
    s_zero = pipe_zero.run(src, output_dir=out_zero, fps=20.0, save_video=True, exports=("jsonl",))
    assert not (out_zero / "annotated.mp4").exists()
    assert "video" not in s_zero.output_files
    assert (out_zero / "summary.json").is_file()


def test_pipeline_postprocess_skeleton_omits_stale_coasting(tmp_path: Path, monkeypatch) -> None:
    src = _write_image_seq(tmp_path / "skel_seq", 10)

    class _Det:
        active_provider = "MockCPU"

        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            _ = frame_bgr
            return [Detection((40.0, 40.0, 90.0, 190.0), 0.92, 0, "person")]

    class _CoastTracker:
        track_buffer = 5

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = frame_bgr
            d = detections[0]
            return [Track(1, d.bbox, d.confidence, d.class_id, d.class_name)]

        def coasting_tracks(self, max_age: int = 3) -> list[Track]:
            _ = max_age
            return [Track(99, (180.0, 40.0, 220.0, 190.0), 0.5, 0, "person")]

    pipe = TrackflowPipeline(
        model="mock",
        detector=_Det(),
        tracker_factory=_CoastTracker,
        postprocess=True,
        anonymize="skeleton",
    )
    coasting_lens: list[int] = []
    orig_apply = pipe.anonymizer.apply

    def _spy(frame, tracks, *, detections=(), coasting=()):
        coasting_lens.append(len(coasting))
        return orig_apply(frame, tracks, detections=detections, coasting=coasting)

    monkeypatch.setattr(pipe.anonymizer, "apply", _spy)
    pipe.run(src, output_dir=tmp_path / "skel_out", fps=20.0, save_video=True, exports=("jsonl",))
    assert len(coasting_lens) == 10 and all(n == 0 for n in coasting_lens)


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
