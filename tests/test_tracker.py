"""Tests for ByteTrack, BoT-SORT, Kalman filtering, and diagnostic events."""

from __future__ import annotations

import numpy as np

from trackflow.tracker import BotSortTracker, ByteTracker, compute_iou_matrix, create_tracker
from trackflow.types import Detection


def _det(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9, emb=None) -> Detection:
    return Detection(
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        class_id=0,
        class_name="person",
        embedding=emb,
    )


def test_iou_matrix_exact_and_disjoint() -> None:
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=np.float32)
    iou = compute_iou_matrix(a, b)
    assert np.isclose(iou[0, 0], 1.0)
    assert np.isclose(iou[0, 1], 50.0 / 150.0, atol=1e-4)
    assert np.isclose(iou[1, 0], 0.0)


def test_bytetrack_persists_id_and_recovers_low_confidence() -> None:
    tracker = ByteTracker(track_high_thresh=0.5, track_low_thresh=0.15, new_track_thresh=0.55)

    # Frame 1: high-confidence person
    t1 = tracker.update([_det(10, 10, 50, 110, 0.9)])
    assert len(t1) == 1
    tid = t1[0].track_id

    # Frame 2: slight motion, still high confidence -> same ID
    t2 = tracker.update([_det(14, 10, 54, 110, 0.85)])
    assert len(t2) == 1 and t2[0].track_id == tid

    # Frame 3: partial occlusion drops confidence to 0.25 -> ByteTrack 2nd stage preserves ID
    t3 = tracker.update([_det(18, 10, 58, 110, 0.25)])
    assert len(t3) == 1 and t3[0].track_id == tid

    # Frame 4: complete occlusion (0 detections), then returns on frame 5 -> recovered flag True
    t4 = tracker.update([])
    assert len(t4) == 0
    t5 = tracker.update([_det(24, 10, 64, 110, 0.88)])
    assert len(t5) == 1 and t5[0].track_id == tid and t5[0].recovered is True


def test_botsort_appearance_fusion_and_cmc() -> None:
    tracker = BotSortTracker(use_cmc=True)
    emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame1[30:200:20, 30:280:20] = 255
    frame2 = np.roll(frame1, shift=2, axis=1)

    out1 = tracker.update(
        [_det(20, 40, 70, 160, 0.9, emb_a), _det(180, 40, 230, 160, 0.9, emb_b)],
        frame_bgr=frame1,
    )
    assert len(out1) == 2
    ids1 = [t.track_id for t in out1]

    out2 = tracker.update(
        [_det(22, 40, 72, 160, 0.9, emb_a), _det(182, 40, 232, 160, 0.9, emb_b)],
        frame_bgr=frame2,
    )
    assert [t.track_id for t in out2] == ids1


def test_create_tracker_factory() -> None:
    assert isinstance(create_tracker("bytetrack"), ByteTracker)
    assert isinstance(create_tracker("bot-sort"), BotSortTracker)


def test_single_frame_false_positive_never_gets_an_id() -> None:
    tracker = ByteTracker(track_high_thresh=0.5, new_track_thresh=0.55)
    tracker.update([_det(10, 10, 50, 110)])  # frame 1: activated immediately (reference behavior)

    # A ghost box appears for one frame only: it must not be output or consume an ID.
    out = tracker.update([_det(12, 10, 52, 110), _det(300, 50, 340, 150)])
    assert [t.track_id for t in out] == [1]
    out = tracker.update([_det(14, 10, 54, 110)])
    assert [t.track_id for t in out] == [1]

    # A real newcomer is confirmed on its second detection and gets the next ID.
    tracker.update([_det(16, 10, 56, 110), _det(200, 40, 240, 140)])
    out = tracker.update([_det(18, 10, 58, 110), _det(203, 40, 243, 140)])
    assert [t.track_id for t in out] == [1, 2]


def test_partial_body_duplicate_does_not_spawn_a_track() -> None:
    tracker = ByteTracker(track_high_thresh=0.5, new_track_thresh=0.55)
    for i in range(3):
        # Full-body box plus a confident upper-body duplicate inside it.
        out = tracker.update([_det(100, 50, 140, 210, 0.9), _det(101 + i, 52, 139, 130, 0.8)])
        assert [t.track_id for t in out] == [1]


def test_coasting_tracks_cover_short_misses_only() -> None:
    tracker = ByteTracker(track_buffer=30)
    tracker.update([_det(10, 10, 50, 110)])
    tracker.update([_det(14, 10, 54, 110)])
    assert tracker.update([]) == []
    coast = tracker.coasting_tracks(max_frames=3)
    assert [t.track_id for t in coast] == [1] and coast[0].coasting is True
    assert coast[0].bbox[0] > 14.0  # predicted forward along its motion
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert tracker.coasting_tracks(max_frames=3) == []


def test_botsort_cmc_reference_frame_never_goes_stale() -> None:
    tracker = BotSortTracker(use_cmc=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[30:200:20, 30:280:20] = 255
    tracker.update([], frame_bgr=frame)  # no tracks yet, CMC must still see this frame
    assert tracker._cmc is not None and tracker._cmc._prev_gray is not None
