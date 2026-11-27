"""Unit and end-to-end tests for offline post-processing (--postprocess)."""

from __future__ import annotations

import csv
import json
import os
import signal
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
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
    # Single person walking smoothly with missing observations at frames 10 and 12 (stride=2)
    frames = [f for f in range(0, 22, 2) if f not in (10, 12)]
    rows = np.array(_linear_track_rows(1, frames, x0=50.0, y0=150.0, vx=4.0, vy=0.0), dtype=np.float64)

    out = postprocess_tracks_array(
        rows, fps=20.0, frame_width=640, frame_height=480, frame_stride=2
    )
    out_frames = sorted(int(round(f)) for f in out[:, 0])
    assert out_frames == list(range(0, 22, 2))
    assert set(int(round(t)) for t in out[:, 1]) == {1}

    # Filled midpoint at frame 10 sits near x=90 (cx=50+4*10=90 -> x1=70, x2=110)
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
    unique_tids = sorted(set(int(round(t)) for t in out[:, 1]))
    assert len(unique_tids) == 2

    pairs = set(zip(np.rint(out[:, 0]).astype(int), np.rint(out[:, 1]).astype(int), strict=True))
    assert len(pairs) == len(out)
    assert np.all(out[:, 4] > out[:, 2]) and np.all(out[:, 5] > out[:, 3])


def test_postprocess_repairs_two_track_crossing_swap() -> None:
    # Person A walks left->right (x: 40 -> 340); Person B walks right->left (x: 340 -> 40).
    # They cross around frame 25, and the raw online tracker swaps IDs 1 <-> 2 at frame 25.
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
    end_cx = 0.5 * (end_row[2] + end_row[4])
    assert end_cx > 300.0


def test_postprocess_edge_cases_and_validation() -> None:
    empty_res = postprocess_records([])
    assert isinstance(empty_res, PostprocessResult)
    assert empty_res.records == []
    assert empty_res.unique_tracks_before == 0
    assert empty_res.unique_tracks_after == 0
    assert empty_res.reid_merges == 0
    assert empty_res.reid_merge_events == []

    empty_arr = postprocess_tracks_array(np.zeros((0, 7), dtype=np.float64))
    assert empty_arr.shape == (0, 7)

    # Negative frame_idx raises ValueError instead of wrapping CMC arrays
    neg_rows = np.array([[-1.0, 1.0, 10.0, 20.0, 50.0, 120.0, 0.9]], dtype=np.float64)
    with pytest.raises(ValueError, match="nonnegative"):
        postprocess_tracks_array(neg_rows)

    # Mismatched embeddings / emb_valid lengths raise ValueError
    valid_rows = np.array(_linear_track_rows(1, [0, 1, 2], 50.0, 150.0, 2.0, 0.0), dtype=np.float64)
    with pytest.raises(ValueError, match="embeddings length"):
        postprocess_tracks_array(valid_rows, embeddings=np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="emb_valid length"):
        postprocess_tracks_array(
            valid_rows,
            embeddings=np.ones((3, 4), dtype=np.float32),
            emb_valid=np.ones((2,), dtype=bool),
        )

    # Passing embeddings with emb_valid=None defaults to all-True (actually uses embeddings)
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
    res_implicit = postprocess_records(
        recs,
        fps=10.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs),
        emb_valid=None,
        allow_long_reid_merge=True,
        min_exit_gap_frames=8,
    )
    assert res_implicit.unique_tracks_after == 2
    assert res_implicit.reid_merges == 1


def test_postprocess_long_horizon_continues_when_medium_pass_empty() -> None:
    # High-trust Re-ID scene with ZERO gaps <= 2.5s and one 4.0s gap (120 frames @ 30fps).
    # Stage 2 must continue from the 2.5s pass to the 6.0s pass instead of breaking early.
    e_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    e_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rows: list[list[float]] = []
    embs: list[np.ndarray] = []

    # Two continuous bystander tracks establish high sequence-level reid_trust (>= 0.50).
    for f in range(200):
        rows.append([float(f), 1.0, 40.0, 60.0, 80.0, 160.0, 0.92])
        embs.append(e_a)
        rows.append([float(f), 2.0, 460.0, 60.0, 500.0, 160.0, 0.92])
        embs.append(e_b)

    # Person C disappears at frame 30 and reappears 4.0s (120 frames) later at frame 150.
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


def test_postprocess_cross_exit_reid_multi_reentry_and_lookalike_guard() -> None:
    e_red = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e_blue = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    # 1. Three non-overlapping appearances A -> B -> C of the same person must chain into 1 track
    recs: list[dict] = []
    embs: list[np.ndarray] = []
    for seg_tid, start_f, x0 in ((10, 0, 20), (20, 45, 220), (30, 90, 420)):
        for f in range(start_f, start_f + 10):
            recs.append(Track(seg_tid, (x0, 40, x0 + 60, 180), 0.9, 0, "person").to_record(f, f / 10.0))
            embs.append(e_red)
    for f in range(0, 10):
        recs.append(Track(11, (520, 40, 580, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs.append(e_blue)

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

    # 2. Two overlapping look-alikes within margin (< 0.05) must NOT merge ambiguously
    e_la1 = np.array([0.995, 0.09987, 0.0, 0.0], dtype=np.float32)
    e_la1 /= np.linalg.norm(e_la1)
    e_la2 = np.array([0.990, 0.14107, 0.0, 0.0], dtype=np.float32)
    e_la2 /= np.linalg.norm(e_la2)
    recs_la: list[dict] = []
    embs_la: list[np.ndarray] = []
    for f in range(10):
        recs_la.append(Track(1, (20, 40, 80, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs_la.append(e_red)
    for f in range(50, 60):
        recs_la.append(Track(2, (200, 40, 260, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs_la.append(e_la1)
        recs_la.append(Track(3, (400, 40, 460, 180), 0.9, 0, "person").to_record(f, f / 10.0))
        embs_la.append(e_la2)
    res_la = postprocess_records(
        recs_la,
        fps=10.0,
        frame_width=640,
        frame_height=480,
        embeddings=np.stack(embs_la),
        allow_long_reid_merge=True,
        min_exit_gap_frames=8,
    )
    assert res_la.unique_tracks_after == 3
    assert res_la.reid_merges == 0


def test_postprocess_low_fps_horizon_stride_and_cross_class_isolation() -> None:
    # 1. At 5 fps without Re-ID, max horizon is 3.2s (~16 frames): a 12s (60-frame) gap must NOT merge.
    r1 = _linear_track_rows(1, list(range(0, 10)), x0=100.0, y0=150.0, vx=0.0, vy=0.0)
    r2 = _linear_track_rows(2, list(range(70, 80)), x0=102.0, y0=150.0, vx=0.0, vy=0.0)
    out_far = postprocess_tracks_array(
        np.array(r1 + r2, dtype=np.float64), fps=5.0, frame_width=640, frame_height=480
    )
    assert len(set(int(round(t)) for t in out_far[:, 1])) == 2

    # 2. Same 5 fps with a 2s (10-frame) gap DOES merge.
    r3 = _linear_track_rows(2, list(range(19, 29)), x0=102.0, y0=150.0, vx=0.0, vy=0.0)
    out_near = postprocess_tracks_array(
        np.array(r1 + r3, dtype=np.float64), fps=5.0, frame_width=640, frame_height=480
    )
    assert len(set(int(round(t)) for t in out_near[:, 1])) == 1

    # 3. Cross-class crossing (person=0 vs car=2): never split or rewire across classes.
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

    # 4. Stride=8 crossing swap: episode grouping and Trigger C scale cleanly with frame_stride.
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
            # Fragment into two raw IDs (7 -> 9) so postprocess must stitch, renumber, and remap events
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
    assert [7] in stream_ids and [9] in stream_ids and [] in stream_ids

    # 2. Offline run() with postprocess=True + anonymize + save_video + floorplan + line + all exports
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

    def _spy_apply(frame, tracks, *, detections=(), coasting=()):
        anon_calls.append((len(tracks), len(detections)))
        return orig_apply(frame, tracks, detections=detections, coasting=coasting)

    import trackflow.pipeline as pipeline_mod

    orig_draw = pipeline_mod.draw_tracks

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
    assert summary.reid_merges == 0  # Kinematic stitches must NOT inflate reid_merges
    assert summary.total_track_observations == 30
    assert (out_dir / "annotated.mp4").is_file()
    assert (out_dir / "floorplan.png").is_file()
    assert (out_dir / "report.html").is_file()
    # Privacy: zero crops on anonymized run, and Pass 2 anonymizer ran on all 30 frames with raw dets
    assert not (out_dir / "crops").exists()
    assert len(anon_calls) == 30
    assert sum(d for _, d in anon_calls) == 27
    # Pass 2 video overlay must show progressive line counts (0 on frame 0, 1 at the end)
    assert len(rendered_counts) == 30
    assert rendered_counts[0] == {"gate": {"in": 0, "out": 0}}
    assert sum(rendered_counts[-1]["gate"].values()) == 1

    cards = json.loads((out_dir / "track_cards.json").read_text(encoding="utf-8"))["cards"]
    assert len(cards) == 1 and cards[0]["thumbnail"] is None
    assert cards[0]["merged_track_ids"] == []
    assert "crops/" not in (out_dir / "report.html").read_text(encoding="utf-8")

    # All export formats agree on single stitched ID = 1 and 30 observations
    pq_tbl = pq.read_table(out_dir / "tracks.parquet")
    assert pq_tbl.num_rows == 30 and set(pq_tbl["track_id"].to_pylist()) == {1}
    assert len(list(csv.reader((out_dir / "tracks.txt").read_text().splitlines()))) == 30

    # Events are chronologically sorted, raw IDs remapped to final ID=1, resolved self-switch dropped
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

    # 3. Unification with --reid-merge and no false line crossing across long exit/re-entry
    class _ExitReentryTracker:
        track_buffer = 3

        def __init__(self) -> None:
            self.f = 0

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = detections, frame_bgr
            idx = self.f
            self.f += 1
            # 9 frames each -> 3 sampled embeddings at reid_sample_every=3, gap = 12 >= 8
            if idx < 9:
                return [Track(10, (20, 40, 80, 190), 0.9, 0, "person")]  # left of x=140
            if idx >= 21:
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
    assert sum_rm.reid_merges == 1
    rm_events = json.loads((out_reid / "events.json").read_text(encoding="utf-8"))
    assert len([e for e in rm_events if e.get("type") == "reid_cross_exit_merge"]) == 1
    assert sum_rm.line_counts["gate"] == {"in": 0, "out": 0}


def test_pipeline_postprocess_same_raw_id_occlusion_line_and_non_replayable_rejected(
    tmp_path: Path,
) -> None:
    src = _write_image_seq(tmp_path / "occ_seq", 30)

    class _Det:
        active_provider = "MockCPU"

        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            _ = frame_bgr
            return []

    # Live stream URL and camera index must fail fast under postprocess=True
    pipe_live = TrackflowPipeline(model="mock", detector=_Det(), postprocess=True)
    with pytest.raises(ValueError, match="recorded video"):
        pipe_live.run("rtsp://example.com/live", output_dir=tmp_path / "rtsp_out", save_video=False)
    with pytest.raises(ValueError, match="recorded video"):
        pipe_live.run(0, output_dir=tmp_path / "cam_out", save_video=False)

    # Same raw ID occluded for 0.7s (14 frames @ 20fps) while stepping across line x=140
    # must NOT reset LineCounter (would undercount occluded crossings).
    class _OccludedSameIdTracker:
        track_buffer = 30

        def __init__(self) -> None:
            self.f = 0

        def update(self, detections: list[Detection], frame_bgr=None) -> list[Track]:
            _ = detections, frame_bgr
            idx = self.f
            self.f += 1
            if idx < 8:
                return [Track(1, (80.0, 40.0, 120.0, 190.0), 0.9, 0, "person")]  # left of x=140
            if idx >= 22:
                return [Track(1, (160.0, 40.0, 200.0, 190.0), 0.9, 0, "person")]  # right of x=140
            return []

    line = CountingLine("gate", (140.0, 0.0), (140.0, 240.0))
    pipe = TrackflowPipeline(
        model="mock",
        detector=_Det(),
        tracker_factory=_OccludedSameIdTracker,
        postprocess=True,
        lines=[line],
    )
    summary = pipe.run(src, output_dir=tmp_path / "occ_out", fps=20.0, save_video=False, exports=("jsonl",))
    assert summary.line_counts["gate"]["in"] + summary.line_counts["gate"]["out"] == 1


def test_pipeline_postprocess_ctrl_c_in_replay_and_skeleton_no_coasting(
    tmp_path: Path, monkeypatch
) -> None:
    src = _write_image_seq(tmp_path / "ctrlc_seq", 20)

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

    # 1. SIGINT during Pass 2 video replay must break cleanly and still write all run artifacts.
    out_ctrlc = tmp_path / "ctrlc_out"
    pipe_cc = TrackflowPipeline(
        model="mock",
        detector=_Det(),
        tracker_factory=_CoastTracker,
        postprocess=True,
    )
    import trackflow.pipeline as pipeline_mod

    orig_draw = pipeline_mod.draw_tracks
    replay_frames = 0

    def _interrupt_mid_replay(*args, **kwargs):
        nonlocal replay_frames
        replay_frames += 1
        if replay_frames == 5:
            os.kill(os.getpid(), signal.SIGINT)
        return orig_draw(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "draw_tracks", _interrupt_mid_replay)
    summary = pipe_cc.run(
        src, output_dir=out_ctrlc, fps=20.0, save_video=True, exports=("jsonl",)
    )
    assert replay_frames < 20
    assert (out_ctrlc / "summary.json").is_file()
    assert (out_ctrlc / "track_cards.json").is_file()
    assert (out_ctrlc / "events.json").is_file()
    assert (out_ctrlc / "tracks.jsonl").is_file()
    assert (out_ctrlc / "annotated.mp4").is_file()
    assert summary.total_track_observations == 20

    # 2. In skeleton mode, Pass 2 passes coasting=() to anonymizer.apply (no stale coasting double-draw).
    monkeypatch.setattr(pipeline_mod, "draw_tracks", orig_draw)
    out_skel = tmp_path / "skel_out"
    pipe_sk = TrackflowPipeline(
        model="mock",
        detector=_Det(),
        tracker_factory=_CoastTracker,
        postprocess=True,
        anonymize="skeleton",
    )
    coasting_lens: list[int] = []
    orig_apply = pipe_sk.anonymizer.apply

    def _spy_skel(frame, tracks, *, detections=(), coasting=()):
        coasting_lens.append(len(coasting))
        return orig_apply(frame, tracks, detections=detections, coasting=coasting)

    monkeypatch.setattr(pipe_sk.anonymizer, "apply", _spy_skel)
    pipe_sk.run(src, output_dir=out_skel, fps=20.0, save_video=True, exports=("jsonl",))
    assert len(coasting_lens) == 20
    assert all(n == 0 for n in coasting_lens)


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
