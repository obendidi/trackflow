"""Tests for the SORT-style trackers: OC-SORT and BoostTrack."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trackflow import TrackflowPipeline
from trackflow.boosttrack import BoostTracker
from trackflow.ocsort import OcSortTracker
from trackflow.tracker import TRACKER_NAMES, create_tracker
from trackflow.types import Detection

SORT_TRACKERS = [OcSortTracker, lambda: BoostTracker(use_cmc=False)]


def _det(x: float, conf: float = 0.9, *, y: float = 10, cls: int = 0, emb=None) -> Detection:
    name = "person" if cls == 0 else "backpack"
    return Detection((x, y, x + 40, y + 100), conf, cls, name, embedding=emb)


def _ids(tracks) -> list[int]:
    return [t.track_id for t in tracks]


@pytest.mark.parametrize("make", SORT_TRACKERS)
def test_ids_are_confirmed_only_after_a_hit_streak(make) -> None:
    tracker = make()
    for i in range(5):
        assert _ids(tracker.update([_det(10 + 3 * i)])) == [1]
    # A newcomer after the first frames needs three matches after its spawn frame.
    seen = [_ids(tracker.update([_det(25 + 3 * i), _det(300 + 2 * i)])) for i in range(5)]
    assert seen == [[1], [1], [1], [1, 2], [1, 2]]
    # A one-frame ghost never gets an ID, so IDs stay contiguous.
    tracker.update([_det(43), _det(310), _det(150, y=200)])
    for i in range(4):
        assert _ids(tracker.update([_det(46 + 3 * i), _det(312 + 2 * i)])) == [1, 2]
    assert all(e["type"] != "track_started" for e in tracker.last_events)


@pytest.mark.parametrize("make", SORT_TRACKERS)
def test_class_aware_association_keeps_overlapping_objects_apart(make) -> None:
    tracker = make()
    ids = {}
    for i in range(6):
        out = tracker.update([_det(10 + 2 * i), _det(11 + 2 * i, cls=24)])
        ids = {t.class_id: t.track_id for t in out}
    assert ids == {0: 1, 24: 2}
    # With the person missing, the bag keeps its own ID and never takes the person's.
    assert {t.class_id: t.track_id for t in tracker.update([_det(23, cls=24)])} == {24: 2}


@pytest.mark.parametrize("make", SORT_TRACKERS)
def test_recovery_after_a_miss_reports_once_the_streak_is_rebuilt(make) -> None:
    tracker = make()
    for i in range(6):
        tracker.update([_det(10 + 4 * i)])
    assert tracker.update([]) == []
    assert [e["type"] for e in tracker.last_events] == ["track_lost"]
    assert _ids(tracker.coasting_tracks(3)) == [1]
    # Matched again: kept as the same track, shown as coasting until its third new match.
    for i in range(2):
        assert tracker.update([_det(38 + 4 * i)]) == []
        assert _ids(tracker.coasting_tracks(3)) == [1]
    out = tracker.update([_det(46)])
    assert _ids(out) == [1] and out[0].recovered
    assert [e["type"] for e in tracker.last_events] == ["track_recovered"]
    assert tracker.coasting_tracks(3) == []


def test_ocsort_recovers_a_stopped_track_from_its_last_observation() -> None:
    tracker = OcSortTracker()
    for i in range(8):
        tracker.update([_det(10 + 15 * i)])
    for _ in range(6):  # occluded: the filter keeps moving it right, away from where it stopped
        tracker.update([])
    assert tracker.coasting_tracks(30)[0].bbox[0] > 200
    for _ in range(3):
        out = tracker.update([_det(115)])
    assert _ids(out) == [1] and out[0].recovered


def test_ocsort_reupdate_matches_the_reference_filter() -> None:
    from trackflow.ocsort import _OcSortMotion

    motion = _OcSortMotion((10, 10, 50, 110), delta_t=3)
    for age in range(1, 13):
        motion.predict()
        if age < 8:
            motion.update((10 + 10 * age, 10, 50 + 10 * age, 110), age)
        elif age < 12:
            motion.miss()  # hidden while it slows from 10 to 3 px per frame
    motion.update((95, 10, 135, 110), 12)
    # The upstream OC-SORT filter gives cx 116.3 and vx 6.43 on the same input; it applies the
    # final observation a second time, which accounts for the small difference.
    assert motion.x[0] == pytest.approx(116.3, abs=1.0)
    assert motion.x[4] == pytest.approx(6.43, abs=0.15)


def test_boosttrack_keeps_a_low_score_detection_on_a_track() -> None:
    tracker = BoostTracker(use_cmc=False)
    for i in range(8):
        tracker.update([_det(10 + 2 * i)])
    out = tracker.update([_det(26, conf=0.3)])  # below det_thresh, right where the track is
    assert _ids(out) == [1] and out[0].confidence == 0.3


@pytest.mark.parametrize("duo_boost", [True, False])
def test_boosttrack_promotes_an_isolated_low_score_detection(duo_boost) -> None:
    tracker = BoostTracker(use_cmc=False, duo_boost=duo_boost)
    for i in range(5):
        tracker.update([_det(10 + 2 * i)])
    for i in range(4):
        out = tracker.update([_det(20 + 2 * i), _det(300 + 2 * i, conf=0.35)])
    assert _ids(out) == ([1, 2] if duo_boost else [1])


def test_boosttrack_appearance_matches_a_low_overlap_detection() -> None:
    same = np.array([1.0, 0.0, 0.0])
    for emb, expected in ((same, [1]), (None, [])):
        tracker = BoostTracker(use_cmc=False)
        for _ in range(8):
            tracker.update([_det(100, emb=same)])
        # IoU 0.21 is below iou_threshold, but enough together with a matching appearance.
        assert _ids(tracker.update([_det(126, emb=emb)])) == expected


def test_boosttrack_camera_motion_moves_tracks() -> None:
    tracker = BoostTracker(use_cmc=True)
    noise = np.random.default_rng(0).integers(0, 255, (240, 320), dtype=np.uint8)
    frame = cv2.cvtColor(cv2.GaussianBlur(noise, (0, 0), 3), cv2.COLOR_GRAY2BGR)
    for _ in range(4):
        tracker.update([_det(100)], frame_bgr=frame)
    shifted = np.roll(frame, shift=12, axis=1)
    out = tracker.update([_det(112)], frame_bgr=shifted)
    assert _ids(out) == [1]
    assert out[0].bbox[0] == pytest.approx(112, abs=1.0)


def test_reset_restarts_ids() -> None:
    for name in ("ocsort", "boosttrack"):
        tracker = create_tracker(name)
        tracker.update([_det(10)])
        tracker.reset()
        assert _ids(tracker.update([_det(200)])) == [1]


def test_factory_names_and_aliases() -> None:
    assert isinstance(create_tracker("OC-SORT"), OcSortTracker)
    assert isinstance(create_tracker("boosttrack++"), BoostTracker)
    tracker = create_tracker("ocsort", track_high_thresh=0.6, track_buffer=50, iou_buffer=0.2)
    assert (tracker.det_thresh, tracker.track_buffer, tracker.iou_buffer) == (0.6, 50, 0.2)
    with pytest.raises(ValueError, match=", ".join(TRACKER_NAMES)):
        create_tracker("deepsort")


def test_pipeline_rejects_an_unknown_tracker_before_loading_a_model(monkeypatch) -> None:
    def no_download(**_kwargs):
        raise AssertionError("the detector must not load")

    monkeypatch.setattr("trackflow.pipeline.create_detector", no_download)
    with pytest.raises(ValueError, match="Unknown tracker"):
        TrackflowPipeline(tracker="nope")


class _Walkers:
    def __init__(self) -> None:
        self.step = 0
        self.seen: list[Detection] = []

    def detect(self, frame_bgr):
        s = self.step
        self.step += 1
        dets = [_det(10 + 3 * s), _det(200 - 3 * s, conf=0.2)]
        self.seen.extend(dets)
        return dets


@pytest.mark.parametrize(("tracker", "embedded"), [("boosttrack", 2), ("botsort", 1)])
def test_pipeline_embeds_the_detections_each_tracker_uses(tracker, embedded) -> None:
    detector = _Walkers()
    pipe = TrackflowPipeline(detector=detector, tracker=tracker, reid="color")
    frames = (np.full((240, 320, 3), 80, np.uint8) for _ in range(6))
    results = list(pipe.stream(frames, fps=10))
    assert [_ids(r.tracks) for r in results][-1] == [1]
    assert sum(d.embedding is not None for d in detector.seen[-2:]) == embedded
