"""Tests for ByteTrack, BoT-SORT, Kalman filtering, and diagnostic events."""

from __future__ import annotations

import numpy as np
import pytest

from trackflow.tracker import BotSortTracker, ByteTracker, compute_iou_matrix, create_tracker
from trackflow.types import Detection


@pytest.mark.parametrize("tracker", [ByteTracker, BotSortTracker])
def test_different_classes_can_overlap_without_stealing_ids(tracker):
    mot = tracker()
    person = _det(0, 0, 60, 80)
    bag = Detection((1, 1, 59, 79), 0.92, 24, "backpack")
    ids = {t.class_id: t.track_id for t in mot.update([person, bag])}
    assert ids.keys() == {0, 24}
    # A weak bag cannot match the person. A lost person cannot be deduplicated with the bag.
    bag.confidence = 0.30
    assert [t.track_id for t in mot.update([bag])] == [ids[24]]
    bag.confidence = 0.92
    assert {t.class_id: t.track_id for t in mot.update([bag, person])} == ids


def test_cost_limit_allows_an_unmatched_row_or_column():
    from trackflow.tracker import _linear_assignment

    cost = np.array([[0.20, 0.55, 0.9], [0.55, 1.0, 1.0]])
    assert _linear_assignment(cost, 0.5) == ([(0, 0)], [1], [1, 2])
    assert _linear_assignment(cost.T, 0.5) == ([(0, 0)], [1, 2], [1])
    # Preserve a perfect pair when the alternative is two marginal matches.
    assert _linear_assignment(np.array([[0, 0.44], [0.44, 0.6]]), 0.5)[0] == [(0, 0)]


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


def test_cmc_masks_foreground_motion_and_tracks_camera_pan() -> None:
    from trackflow.tracker import SparseOpticalFlowCMC

    rng = np.random.default_rng(0)
    bg = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    # Static textured background + large bright foreground rectangle moving right by 20px.
    f0 = bg.copy()
    f0[60:200, 80:180] = rng.integers(0, 255, size=(140, 100, 3), dtype=np.uint8)
    f1 = bg.copy()
    f1[60:200, 100:200] = f0[60:200, 80:180]

    cmc = SparseOpticalFlowCMC(downscale=2)
    assert cmc.estimate(f0, exclude_boxes=np.array([[80.0, 60.0, 180.0, 200.0]])) is None
    warp = cmc.estimate(f1, exclude_boxes=np.array([[100.0, 60.0, 200.0, 200.0]]))
    # Background did not move, so masked CMC must report ~zero translation (not +20px foreground drift).
    assert warp is not None
    assert abs(float(warp[0, 2])) < 1.5 and abs(float(warp[1, 2])) < 1.5


def test_hmiou_prefers_matching_height_band() -> None:
    from trackflow.tracker import compute_hmiou_matrix

    a = np.array([[0.0, 100.0, 40.0, 200.0]])
    same_y = np.array([[10.0, 100.0, 50.0, 200.0]])
    shifted_y = np.array([[0.0, 125.0, 40.0, 225.0]])
    # Both have identical 0.60 plain IoU, but shifted_y has lower vertical overlap.
    assert float(compute_hmiou_matrix(a, same_y)[0, 0]) > float(
        compute_hmiou_matrix(a, shifted_y)[0, 0]
    )


def test_cmc_handles_frame_resize_and_full_foreground_occlusion() -> None:
    from trackflow.tracker import SparseOpticalFlowCMC

    rng = np.random.default_rng(1)
    small = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    large = rng.integers(0, 255, size=(192, 192, 3), dtype=np.uint8)
    cmc = SparseOpticalFlowCMC(downscale=2)
    assert cmc.estimate(small) is None
    # Frame size change must reset reference instead of raising an OpenCV assertion.
    assert cmc.estimate(large) is None
    # Full-frame foreground box leaves zero eligible background pixels -> must return None.
    assert cmc.estimate(large, exclude_boxes=np.array([[0.0, 0.0, 192.0, 192.0]])) is None


def test_xywh_kalman_confidence_one_and_botsort_lost_width_velocity() -> None:
    from trackflow.tracker import XywhKalmanFilter

    kf = XywhKalmanFilter((10.0, 20.0, 50.0, 120.0), min_nsa=0.0)
    # Consecutive confidence=1.0 updates must not create a singular innovation matrix.
    for _ in range(3):
        kf.predict()
        kf.update((12.0, 20.0, 52.0, 120.0), confidence=1.0)
    assert np.isfinite(kf.mean).all()

    tr = BotSortTracker(use_cmc=False, track_high_thresh=0.5, new_track_thresh=0.55)
    tr.update([_det(10, 20, 50, 120, 0.9)])
    tr.update([_det(10, 20, 70, 120, 0.9)])  # growing width
    for _ in range(7):  # lost past brief-occlusion window: width velocity must be zeroed
        tr.update([])
    assert tr._tracks[0].kf.mean[6] == 0.0


def test_create_tracker_forwards_new_track_thresh_to_all_trackers() -> None:
    from trackflow.tracker import create_tracker

    for name in ("bytetrack", "botsort", "ocsort", "boosttrack"):
        tr = create_tracker(name, track_high_thresh=0.5, new_track_thresh=0.95)
        assert tr.update([_det(10, 20, 60, 140, 0.60)]) == []


def test_boosttrack_duplicate_removal_preserves_matched_confirmed_track() -> None:
    from trackflow.boosttrack import BoostTracker

    def emb_det(x1: float, emb: list[float]) -> Detection:
        return Detection(
            bbox=(x1, 20.0, x1 + 50.0, 140.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            embedding=np.array(emb, dtype=np.float32),
        )

    tr = BoostTracker(det_thresh=0.5, new_track_thresh=0.55, use_cmc=False, min_hits=1)
    # Track 1 spawns earlier (older age) at x=10; track 2 spawns later at x=80 and confirms.
    assert [t.track_id for t in tr.update([emb_det(10, [1.0, 0.0])])] == [1]
    tr.update([emb_det(10, [1.0, 0.0]), emb_det(80, [0.0, 1.0])])
    out = tr.update([emb_det(10, [1.0, 0.0]), emb_det(70, [0.0, 1.0])])
    assert [t.track_id for t in out] == [1, 2]
    # Track 1 goes lost at x=10 while confirmed track 2 (distinct appearance) moves onto x=10.
    for x in (50, 30, 11):
        out = tr.update([emb_det(x, [0.0, 1.0])])
        assert [t.track_id for t in out] == [2]


def test_xywh_kalman_camera_warp_matches_matrix_form_and_ocsort_direction() -> None:
    from trackflow.ocsort import _direction
    from trackflow.tracker import XywhKalmanFilter

    theta = 0.04
    scale = 1.03
    rot = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    warp = np.hstack([rot, np.array([[12.5], [-4.25]])])

    for height_scaled in (False, True):
        kf = XywhKalmanFilter((100.0, 80.0, 160.0, 220.0), height_scaled=height_scaled)
        kf.predict()
        kf.update((106.0, 83.0, 168.0, 226.0), confidence=0.82)
        kf.predict()

        # Reference explicit 8x8 matrix transform before calling apply_camera_warp.
        mean_before = kf.mean.copy()
        cov_before = kf.cov.copy()
        t = np.zeros((8, 8), dtype=np.float64)
        t[0:2, 0:2] = rot
        t[2, 2] = t[3, 3] = scale
        t[4:6, 4:6] = rot
        t[6, 6] = t[7, 7] = scale
        ref_mean = t @ mean_before
        ref_mean[:2] += warp[:2, 2]
        ref_cov = t @ cov_before @ t.T

        kf.apply_camera_warp(warp)
        assert np.allclose(kf.mean, ref_mean, atol=1e-12)
        assert np.allclose(kf.cov, ref_cov, atol=1e-12)

    assert np.allclose(_direction((0, 0, 10, 10), (3, 4, 13, 14)), [0.8, 0.6], atol=1e-6)
    assert np.allclose(_direction((5, 5, 15, 15), (5, 5, 15, 15)), [0.0, 0.0], atol=1e-6)
