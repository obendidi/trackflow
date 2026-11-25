"""Tests for Track Cards, opt-in cross-exit Re-ID merges, and natural-language video search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from trackflow import reid
from trackflow.reid import (
    ReidEmbedder,
    TrackCardIndex,
    VlmEmbedder,
    create_vlm_embedder,
    search_track_cards,
)
from trackflow.types import Track

RED = (10, 10, 230)  # BGR
BLUE = (230, 10, 10)
GREEN = (10, 230, 10)


def _mean_color_unit(crop_bgr: np.ndarray) -> np.ndarray:
    b, g, r = crop_bgr.mean(axis=(0, 1))
    vec = np.array([r, g, b], dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-6)


class _MockVlm(VlmEmbedder):
    """Deterministic vision-language embedder mapping red vs blue crops & prompts cleanly."""

    def __init__(self) -> None:
        self.backend = "clip"
        self.precision = "int8"
        self._vision_sess = object()
        self._text_sess = object()
        self._tokenizer = object()  # type: ignore[assignment]
        self._text_cache = {}

    def encode_image(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        return None if crop_bgr is None or crop_bgr.size == 0 else _mean_color_unit(crop_bgr)

    def encode_text(self, query: str) -> np.ndarray:
        q = query.lower()
        if "red" in q:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "blue" in q:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


class _MockReid(ReidEmbedder):
    """Identity embedder: people are distinguished by their (solid) clothing color."""

    def __init__(self) -> None:
        self.backend = "osnet"

    def encode_images(self, crops: list[np.ndarray | None]) -> list[np.ndarray | None]:
        return [None if c is None or c.size == 0 else _mean_color_unit(c) for c in crops]


def _frame(*people: tuple[tuple[int, int, int, int], tuple[int, int, int]]) -> np.ndarray:
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    for (x1, y1, x2, y2), color in people:
        frame[y1:y2, x1:x2] = color
    return frame


def _track(tid: int, box: tuple[int, int, int, int]) -> Track:
    return Track(track_id=tid, bbox=box, confidence=0.9, class_id=0, class_name="person")


LEFT = (20, 20, 80, 140)
RIGHT = (180, 20, 240, 140)
FAR = (300, 20, 360, 140)


def _feed(idx: TrackCardIndex, start: int, n: int, *people) -> list[dict]:
    """Show the same people for n frames; returns all events."""
    events = []
    frame = _frame(*[(box, color) for _, box, color in people])
    for f in range(start, start + n):
        tracks = [_track(tid, box) for tid, box, _ in people]
        events += idx.update(frame_idx=f, timestamp_s=f / 10, frame_bgr=frame, tracks=tracks)
    return events


def test_query_scoring_and_natural_language_search(tmp_path: Path) -> None:
    vlm = _MockVlm()
    idx = TrackCardIndex(vlm=vlm, query="person in red jacket", query_threshold=0.8)
    frame = _frame((LEFT, RED), (RIGHT, BLUE))
    tr_red, tr_blue = _track(1, LEFT), _track(2, RIGHT)
    idx.update(frame_idx=0, timestamp_s=0.0, frame_bgr=frame, tracks=[tr_red, tr_blue])
    assert tr_red.query_match is True
    assert tr_blue.query_match is False

    idx.save_json(tmp_path / "track_cards.json", vlm_backend="clip")
    hits = search_track_cards(tmp_path, "person in blue coat", top_k=2, embedder=vlm)
    assert [h["track_id"] for h in hits] == [2, 1]
    assert hits[0]["search_score"] > hits[1]["search_score"]


def test_search_loads_the_embedder_the_run_was_indexed_with(tmp_path: Path, monkeypatch) -> None:
    vlm = _MockVlm()
    idx = TrackCardIndex(vlm=vlm)
    idx.update(
        frame_idx=0,
        timestamp_s=0.0,
        frame_bgr=_frame((LEFT, RED), (RIGHT, BLUE)),
        tracks=[_track(1, LEFT), _track(2, RIGHT)],
    )
    idx.save_json(tmp_path / "track_cards.json", vlm_backend="embeddinggemma")
    loaded: list[str] = []

    def fake_create(backend: str, **_: object) -> VlmEmbedder:
        loaded.append(backend)
        return vlm

    monkeypatch.setattr(reid, "create_vlm_embedder", fake_create)
    hits = search_track_cards(tmp_path, "person in red", top_k=1)
    assert loaded == ["embeddinggemma"]
    assert hits[0]["track_id"] == 1


def test_query_threshold_defaults_to_the_models_score_scale() -> None:
    from trackflow.pipeline import TrackflowPipeline

    def vlm(backend: str) -> _MockVlm:
        embedder = _MockVlm()
        embedder.backend = backend
        return embedder

    def pipe(backend: str, **kwargs: object) -> float:
        return TrackflowPipeline(
            model="mock",
            detector=object(),
            query="person in red",
            vlm_embedder=vlm(backend),
            **kwargs,
        ).query_threshold

    assert pipe("clip") == 0.27
    assert pipe("siglip") == 0.08
    assert pipe("embeddinggemma") == 0.71
    assert pipe("embeddinggemma", query_threshold=0.5) == 0.5
    assert TrackCardIndex(vlm=vlm("siglip"), query="red").query_threshold == 0.08


def test_embeddinggemma_explains_the_missing_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"trackflow-python\[transformers\]"):
        create_vlm_embedder("embeddinggemma")


def test_track_is_embedded_when_first_seen_then_at_doubling_ages() -> None:
    class _CountingVlm(_MockVlm):
        def __init__(self) -> None:
            super().__init__()
            self.ages: list[float] = []

        def encode_image(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
            self.ages.append(self.now)
            return super().encode_image(crop_bgr)

    vlm = _CountingVlm()
    idx = TrackCardIndex(vlm=vlm)
    frame = _frame((LEFT, RED))
    for f in range(100):  # 10 s at 10 FPS
        vlm.now = f / 10
        idx.update(frame_idx=f, timestamp_s=f / 10, frame_bgr=frame, tracks=[_track(1, LEFT)])

    assert vlm.ages == pytest.approx([0.0, 1.0, 2.0, 4.0, 8.0])
    assert idx.cards[1].embedding == pytest.approx([1.0, 0.0, 0.0], abs=0.05)


def test_query_never_merges_identities() -> None:
    """A --query run (VLM only) must not merge different raw IDs, however similar they look."""
    idx = TrackCardIndex(vlm=_MockVlm(), query="red")
    _feed(idx, 0, 5, (1, LEFT, RED))
    _feed(idx, 30, 5, (7, LEFT, RED))
    assert idx.reid_merges == 0
    assert sorted(idx.cards) == [1, 7]


def test_merge_is_opt_in_and_needs_reid_model() -> None:
    with pytest.raises(ValueError):
        TrackCardIndex(vlm=_MockVlm(), reid_merge=True)

    idx = TrackCardIndex(reid=_MockReid())  # Re-ID model but merging not enabled
    _feed(idx, 0, 5, (1, LEFT, RED))
    _feed(idx, 30, 10, (7, LEFT, RED))
    assert idx.reid_merges == 0


def test_returning_person_gets_earlier_id_after_enough_samples() -> None:
    idx = TrackCardIndex(
        reid=_MockReid(), reid_merge=True, merge_min_samples=3, reid_sample_every=1
    )
    _feed(idx, 0, 10, (1, LEFT, RED), (2, RIGHT, BLUE))

    # Red person leaves, comes back 20 frames later as raw ID 9.
    first = _feed(idx, 30, 2, (9, LEFT, RED))
    assert first == [] and 9 in idx.cards  # not decided yet: provisional ID
    events = _feed(idx, 32, 3, (9, LEFT, RED))
    assert [e["type"] for e in events] == ["reid_cross_exit_merge"]
    assert events[0]["raw_track_id"] == 9 and events[0]["canonical_track_id"] == 1
    assert idx.canonical_id(9) == 1
    assert sorted(idx.cards) == [1, 2]
    assert idx.cards[1].frame_count == 15 and idx.cards[1].merged_track_ids == [9]

    later = _track(9, LEFT)
    idx.update(frame_idx=40, timestamp_s=4.0, frame_bgr=_frame((LEFT, RED)), tracks=[later])
    assert later.track_id == 1 and later.reid_merged_from == 9


def test_no_merge_with_someone_visible_at_the_same_time() -> None:
    idx = TrackCardIndex(reid=_MockReid(), reid_merge=True, reid_sample_every=1)
    # Two people in identical red clothes visible together: the second can't be the first.
    _feed(idx, 0, 10, (1, LEFT, RED), (2, FAR, RED))
    assert idx.reid_merges == 0 and sorted(idx.cards) == [1, 2]


def test_ambiguous_look_alikes_are_not_merged() -> None:
    idx = TrackCardIndex(reid=_MockReid(), reid_merge=True, reid_sample_every=1)
    _feed(idx, 0, 10, (1, LEFT, RED), (2, FAR, RED))  # two people in red, both leave
    _feed(idx, 50, 10, (3, LEFT, RED))  # matches #1 and #2 equally -> leave it alone
    assert idx.reid_merges == 0 and sorted(idx.cards) == [1, 2, 3]

    _feed(idx, 70, 10, (4, LEFT, GREEN))  # nobody similar -> new person
    assert idx.reid_merges == 0


def test_color_reid_embedder_produces_unit_vectors() -> None:
    emb = ReidEmbedder(backend="color")
    crop = np.full((80, 40, 3), (30, 140, 220), dtype=np.uint8)
    vecs = emb.encode_images([crop, None])
    assert vecs[1] is None
    assert vecs[0] is not None and vecs[0].shape == (96,)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-4)
