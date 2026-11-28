"""Unit tests for pure Apache-2.0 NonLinearTracker (``nltrack``)."""

from __future__ import annotations

import numpy as np
import pytest

from trackflow import NonLinearTracker, TrackflowPipeline
from trackflow.nltrack import _NlMotion
from trackflow.tracker import TRACKER_NAMES, create_tracker
from trackflow.types import Detection


def _det(
    x: float,
    conf: float = 0.9,
    *,
    y: float = 10.0,
    w: float = 40.0,
    h: float = 100.0,
    cls: int = 0,
    emb: list[float] | np.ndarray | None = None,
) -> Detection:
    name = "person" if cls == 0 else "bus"
    vec = np.asarray(emb, dtype=np.float32) if emb is not None else None
    return Detection((x, y, x + w, y + h), conf, cls, name, embedding=vec)


def _ids(tracks) -> list[int]:
    return [t.track_id for t in tracks]


def test_nltrack_lifecycle_recovery_coasting_and_reset() -> None:
    tracker = NonLinearTracker(use_cmc=False, track_buffer=10)
    assert tracker.track_buffer == 10

    for i in range(4):
        assert _ids(tracker.update([_det(10 + 3 * i)])) == [1]

    # A newcomer after frame 1 needs min_confirm_hits (4) frames before confirmation.
    seen = [_ids(tracker.update([_det(22 + 3 * i), _det(300 + 2 * i)])) for i in range(4)]
    assert seen == [[1], [1], [1], [1, 2]]

    # Miss track 2 for 3 frames -> lost, coasting, then immediate recovery on return.
    for i in range(3):
        out = tracker.update([_det(34 + 3 * i)])
        assert _ids(out) == [1]
    assert _ids(tracker.coasting_tracks(3)) == [2]

    out = tracker.update([_det(43), _det(312)])
    assert _ids(out) == [1, 2]
    t2 = next(t for t in out if t.track_id == 2)
    assert t2.recovered is True
    assert any(e == {"type": "track_recovered", "track_id": 2} for e in tracker.last_events)

    tracker.reset()
    assert _ids(tracker.update([_det(200)])) == [1]


def test_nltrack_factory_and_buffer_floor() -> None:
    assert "nltrack" in TRACKER_NAMES
    tr = create_tracker("nltrack", track_buffer=15)
    assert isinstance(tr, NonLinearTracker)
    assert tr.track_buffer == 60
    assert isinstance(create_tracker("nl-track"), NonLinearTracker)


def test_nltrack_motion_damping_and_oru_interpolation() -> None:
    motion = _NlMotion((10.0, 10.0, 50.0, 110.0), delta_t=3, vel_damp=0.90)
    for age in range(1, 6):
        motion.predict(is_lost=False)
        motion.update((10.0 + 10.0 * age, 10.0, 50.0 + 10.0 * age, 110.0), age, confidence=0.9)
    vx_before = float(motion.kf.mean[4])
    assert vx_before > 5.0

    # Lost frames damp center velocity and zero box-size velocities.
    for _ in range(4):
        motion.miss()
        motion.predict(is_lost=True)
    assert float(motion.kf.mean[4]) < vx_before
    assert float(motion.kf.mean[6]) == 0.0 and float(motion.kf.mean[7]) == 0.0

    # Re-observation triggers ORU linear interpolation from the frozen snapshot.
    motion.update((75.0, 10.0, 115.0, 110.0), age=10, confidence=0.9)
    assert motion._frozen is None
    assert motion._missed == 0
    assert abs(motion.box[0] - 75.0) < 6.0


def test_nltrack_stage2_median_height_is_per_class() -> None:
    tracker = NonLinearTracker(use_cmc=False, min_confirm_hits=1)
    # 3 large buses (h=400) + 1 small person (h=80): person must stay Stage-2 eligible.
    frame1 = [
        _det(10, cls=5, w=120, h=400),
        _det(200, cls=5, w=120, h=400),
        _det(400, cls=5, w=120, h=400),
        _det(600, cls=0, w=30, h=80),
    ]
    out1 = tracker.update(frame1)
    person_id = next(t.track_id for t in out1 if t.class_id == 0)

    # Person drops to low confidence (0.25) -> must still match via Stage 2a.
    frame2 = [
        _det(12, cls=5, w=120, h=400),
        _det(202, cls=5, w=120, h=400),
        _det(402, cls=5, w=120, h=400),
        _det(602, conf=0.25, cls=0, w=30, h=80),
    ]
    out2 = tracker.update(frame2)
    assert any(t.track_id == person_id for t in out2)


def test_nltrack_stage2b_recovers_recently_lost_track_on_low_score_detection() -> None:
    tracker = NonLinearTracker(use_cmc=False)
    for i in range(4):
        assert _ids(tracker.update([_det(50 + 2 * i)])) == [1]

    # Miss for 3 frames, then reappear with a low-confidence detection (0.25) on top of last box.
    for _ in range(3):
        assert tracker.update([]) == []
    out = tracker.update([_det(58, conf=0.25)])
    assert _ids(out) == [1]
    assert out[0].recovered is True


def test_nltrack_clean_bank_appearance_recovers_non_overlapping_reentry() -> None:
    tracker = NonLinearTracker(use_cmc=False)
    emb_a = [1.0, 0.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0, 0.0]
    emb_c = [0.0, 0.0, 1.0, 0.0]

    # Build clean appearance banks and non-matching scene statistics.
    for i in range(4):
        out = tracker.update(
            [
                _det(20 + i, emb=emb_a),
                _det(200 + i, emb=emb_b),
                _det(380 + i, emb=emb_c),
            ]
        )
    assert _ids(out) == [1, 2, 3]

    # Track 1 vanishes for 5 frames and reappears shifted by 90 px (zero IoU) with same emb_a.
    for i in range(4, 9):
        tracker.update([_det(200 + i, emb=emb_b), _det(380 + i, emb=emb_c)])
    out = tracker.update(
        [_det(115, emb=emb_a), _det(209, emb=emb_b), _det(389, emb=emb_c)]
    )
    assert _ids(out) == [1, 2, 3]
    t1 = next(t for t in out if t.track_id == 1)
    assert t1.recovered is True


def test_nltrack_tai_blocks_duplicate_spawn_on_occupied_track() -> None:
    tracker = NonLinearTracker(use_cmc=False)
    for i in range(5):
        out = tracker.update([_det(100 + i, conf=0.9), _det(102 + i, conf=0.85)])
        assert _ids(out) == [1]


def test_pipeline_embeds_high_confidence_detections_for_nltrack() -> None:
    class _StubDet:
        def __init__(self) -> None:
            self.step = 0
            self.seen: list[Detection] = []

        def detect(self, _frame):
            s = self.step
            self.step += 1
            dets = [_det(10 + 3 * s, conf=0.9), _det(200 - 3 * s, conf=0.2)]
            self.seen.extend(dets)
            return dets

    det = _StubDet()
    pipe = TrackflowPipeline(detector=det, tracker="nltrack", reid="color")
    frames = (np.full((240, 320, 3), 80, np.uint8) for _ in range(5))
    results = list(pipe.stream(frames, fps=10))
    assert [_ids(r.tracks) for r in results][-1] == [1]
    assert sum(d.embedding is not None for d in det.seen[-2:]) == 1


@pytest.mark.parametrize("bad", ["nltrack_onnx", "unknown_tracker"])
def test_unknown_tracker_rejected_loudly(bad: str) -> None:
    with pytest.raises(ValueError, match="Unknown tracker"):
        create_tracker(bad)
