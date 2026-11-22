"""Tests for Track Cards, cross-exit long-term Re-ID, and natural-language video search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trackflow.reid import AppearanceEmbedder, TrackCardIndex, search_track_cards
from trackflow.types import Track


class _MockVlmEmbedder(AppearanceEmbedder):
    """Deterministic vision-language embedder mapping red vs blue crops & prompts cleanly."""

    def __init__(self) -> None:
        self.backend = "clip"
        self.precision = "int8"
        self._vision_sess = object()
        self._text_sess = object()
        self._tokenizer = object()  # type: ignore[assignment]
        self._text_cache = {}

    @property
    def supports_text(self) -> bool:
        return True

    def encode_image(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        b, g, r = crop_bgr.mean(axis=(0, 1))
        vec = np.array([r, g, b], dtype=np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-6)

    def encode_text(self, query: str) -> np.ndarray:
        q = query.lower()
        if "red" in q:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "blue" in q:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def test_cross_exit_reid_and_natural_language_search(tmp_path: Path) -> None:
    embedder = _MockVlmEmbedder()
    idx = TrackCardIndex(
        embedder=embedder,
        query="person in red jacket",
        query_threshold=0.8,
        reid_merge_threshold=0.95,
        min_exit_gap_frames=5,
        crops_dir=tmp_path / "crops",
    )

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # Red jacket person on left (BGR: 10, 10, 230)
    frame[20:140, 20:80] = (10, 10, 230)
    # Blue coat person on right (BGR: 230, 10, 10)
    frame[20:140, 180:240] = (230, 10, 10)

    tr_red = Track(
        track_id=1, bbox=(20, 20, 80, 140), confidence=0.9, class_id=0, class_name="person"
    )
    tr_blue = Track(
        track_id=2, bbox=(180, 20, 240, 140), confidence=0.88, class_id=0, class_name="person"
    )
    idx.update(frame_idx=0, timestamp_s=0.0, frame_bgr=frame, tracks=[tr_red, tr_blue])

    assert tr_red.query_match is True
    assert tr_blue.query_match is False

    # Red person exits for 10 frames, then re-enters as a fresh raw track_id=9 -> should merge to #1!
    tr_red_reenter = Track(
        track_id=9, bbox=(20, 20, 80, 140), confidence=0.92, class_id=0, class_name="person"
    )
    events = idx.update(frame_idx=12, timestamp_s=0.4, frame_bgr=frame, tracks=[tr_red_reenter])

    assert tr_red_reenter.track_id == 1
    assert tr_red_reenter.reid_merged_from == 9
    assert idx.reid_merges == 1
    assert any(e["type"] == "reid_cross_exit_merge" for e in events)
    assert len(idx.cards) == 2

    # Save Track Cards and run post-run natural-language search
    cards_file = tmp_path / "track_cards.json"
    idx.save_json(cards_file, reid_backend="clip")
    hits = search_track_cards(tmp_path, "person in blue coat", top_k=2, embedder=embedder)
    assert len(hits) == 2
    assert hits[0]["track_id"] == 2
    assert hits[0]["search_score"] > hits[1]["search_score"]


def test_color_embedder_produces_unit_vectors() -> None:
    emb = AppearanceEmbedder(backend="color")
    crop = np.full((80, 40, 3), (30, 140, 220), dtype=np.uint8)
    vec = emb.encode_image(crop)
    assert vec is not None
    assert vec.shape == (96,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)
