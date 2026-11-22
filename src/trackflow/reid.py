"""Vision-language Re-ID, Track Cards, cross-exit identity merging, and natural-language search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from tokenizers import Tokenizer

from trackflow.detector import create_onnx_session
from trackflow.models import Precision, resolve_reid_artifacts
from trackflow.types import Track, TrackCard

ReidBackend = Literal["none", "color", "clip", "siglip"]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _crop_bbox(frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, int(round(bbox[0]))))
    y1 = max(0, min(h - 1, int(round(bbox[1]))))
    x2 = max(x1 + 1, min(w, int(round(bbox[2]))))
    y2 = max(y1 + 1, min(h, int(round(bbox[3]))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame_bgr[y1:y2, x1:x2]


def _color_spatial_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """Fast 96-dim spatial upper/lower-body HSV color descriptor (zero external weights)."""
    resized = cv2.resize(crop_bgr, (48, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    parts = [hsv[:48, :, :], hsv[48:, :, :]]  # upper body + lower body
    feats: list[np.ndarray] = []
    for part in parts:
        h_hist = cv2.calcHist([part], [0], None, [24], [0, 180]).ravel()
        s_hist = cv2.calcHist([part], [1], None, [12], [0, 256]).ravel()
        v_hist = cv2.calcHist([part], [2], None, [12], [0, 256]).ravel()
        feats.extend([h_hist, s_hist, v_hist])
    return _l2_normalize(np.concatenate(feats).astype(np.float32))


class AppearanceEmbedder:
    """Appearance and vision-language embedder supporting CLIP, SigLIP (ONNX), and fast Color."""

    def __init__(
        self,
        backend: ReidBackend = "clip",
        *,
        precision: Precision = "int8",
        device: str = "auto",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.backend: ReidBackend = backend
        self.precision = precision
        self._vision_sess = None
        self._text_sess = None
        self._tokenizer: Tokenizer | None = None
        self._image_size = (224, 224)
        self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._max_length = 77 if backend == "clip" else 64
        self._text_cache: dict[str, np.ndarray] = {}

        if backend in ("clip", "siglip"):
            artifacts = resolve_reid_artifacts(
                backend=backend, precision=precision, cache_dir=cache_dir
            )
            eff_dev = "cpu" if (device == "auto" and precision == "int8") else device
            self._vision_sess, _ = create_onnx_session(artifacts["vision_onnx"], device=eff_dev)
            self._text_sess, _ = create_onnx_session(artifacts["text_onnx"], device=eff_dev)
            self._tokenizer = Tokenizer.from_file(str(artifacts["tokenizer"]))
            self._load_preprocess(artifacts["preprocessor"], artifacts["config"])

    @property
    def supports_text(self) -> bool:
        return self._text_sess is not None and self._tokenizer is not None

    def _load_preprocess(self, prep_path: Path, config_path: Path) -> None:
        if prep_path.is_file():
            cfg = json.loads(prep_path.read_text(encoding="utf-8"))
            size = cfg.get("crop_size") or cfg.get("size") or {}
            if isinstance(size, int):
                self._image_size = (size, size)
            elif isinstance(size, dict):
                h = int(size.get("height", size.get("shortest_edge", 224)))
                w = int(size.get("width", size.get("shortest_edge", 224)))
                self._image_size = (h, w)
            if "image_mean" in cfg:
                self._mean = np.array(cfg["image_mean"], dtype=np.float32).reshape(3, 1, 1)
            if "image_std" in cfg:
                self._std = np.array(cfg["image_std"], dtype=np.float32).reshape(3, 1, 1)

        if config_path.is_file():
            mcfg = json.loads(config_path.read_text(encoding="utf-8"))
            tcfg = mcfg.get("text_config", {})
            if isinstance(tcfg, dict) and "max_position_embeddings" in tcfg:
                self._max_length = int(tcfg["max_position_embeddings"])

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        th, tw = self._image_size
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_CUBIC)
        arr = resized.astype(np.float32) * (1.0 / 255.0)
        chw = np.transpose(arr, (2, 0, 1))
        chw = (chw - self._mean) / self._std
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def encode_image(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        """Encode a BGR person/object crop into an L2-normalized embedding vector."""
        if crop_bgr is None or crop_bgr.size == 0 or self.backend == "none":
            return None
        if self.backend == "color":
            return _color_spatial_embedding(crop_bgr)

        assert self._vision_sess is not None
        inp = self._vision_sess.get_inputs()[0]
        tensor = self._preprocess_crop(crop_bgr)
        if "float16" in inp.type:
            tensor = tensor.astype(np.float16)

        outputs = self._vision_sess.run(None, {inp.name: tensor})
        out_names = [o.name for o in self._vision_sess.get_outputs()]
        by_name = dict(zip(out_names, outputs, strict=False))
        emb = by_name.get("image_embeds")
        if emb is None:
            emb = by_name.get("pooler_output", outputs[0])
        emb_arr = np.asarray(emb, dtype=np.float32)
        if emb_arr.ndim == 3:
            emb_arr = emb_arr[:, 0, :]
        return _l2_normalize(emb_arr[0])

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a natural-language query into the shared vision-language embedding space."""
        cleaned = query.strip()
        if cleaned in self._text_cache:
            return self._text_cache[cleaned]
        if not self.supports_text:
            raise RuntimeError(
                f"Backend '{self.backend}' does not support natural-language text queries. "
                "Use --reid clip or --reid siglip."
            )

        assert self._tokenizer is not None and self._text_sess is not None
        self._tokenizer.enable_truncation(max_length=self._max_length)
        if self.backend == "siglip":
            self._tokenizer.enable_padding(length=self._max_length, pad_id=1)
        else:
            self._tokenizer.no_padding()

        enc = self._tokenizer.encode(cleaned)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)

        feed: dict[str, np.ndarray] = {}
        for inp in self._text_sess.get_inputs():
            if inp.name == "input_ids":
                feed[inp.name] = input_ids
            elif inp.name == "attention_mask":
                feed[inp.name] = attention_mask

        outputs = self._text_sess.run(None, feed)
        out_names = [o.name for o in self._text_sess.get_outputs()]
        by_name = dict(zip(out_names, outputs, strict=False))
        emb = by_name.get("text_embeds")
        if emb is None:
            emb = by_name.get("pooler_output", outputs[0])
        emb_arr = np.asarray(emb, dtype=np.float32)
        if emb_arr.ndim == 3:
            emb_arr = emb_arr[:, 0, :]
        vec = _l2_normalize(emb_arr[0])
        self._text_cache[cleaned] = vec
        return vec

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = _l2_normalize(a)
        nb = _l2_normalize(b)
        if na.shape != nb.shape:
            return 0.0
        return float(np.dot(na, nb))


class TrackCardIndex:
    """Maintains persistent Track Cards, cross-exit long-term Re-ID merges, and NL query scores."""

    def __init__(
        self,
        *,
        embedder: AppearanceEmbedder | None = None,
        query: str | None = None,
        query_threshold: float = 0.22,
        reid_merge_threshold: float = 0.82,
        min_exit_gap_frames: int = 8,
        embed_every_n_frames: int = 5,
        crops_dir: Path | None = None,
    ) -> None:
        self.embedder = embedder
        self.query = query.strip() if query else None
        self.query_threshold = float(query_threshold)
        self.reid_merge_threshold = float(reid_merge_threshold)
        self.min_exit_gap_frames = int(min_exit_gap_frames)
        self.embed_every_n_frames = max(1, int(embed_every_n_frames))
        self.crops_dir = crops_dir

        self._query_vec: np.ndarray | None = None
        if self.query and self.embedder is not None and self.embedder.supports_text:
            self._query_vec = self.embedder.encode_text(self.query)

        self.cards: dict[int, TrackCard] = {}
        self._embeddings: dict[int, np.ndarray] = {}
        self._id_alias: dict[int, int] = {}  # raw_track_id -> canonical_track_id
        self.reid_merges: int = 0

    def canonical_id(self, raw_track_id: int) -> int:
        return self._id_alias.get(raw_track_id, raw_track_id)

    def _try_cross_exit_merge(
        self,
        raw_track_id: int,
        class_name: str,
        frame_idx: int,
        emb: np.ndarray | None,
    ) -> int:
        """Match a newly spawned track against previously exited Track Cards by appearance."""
        if emb is None or raw_track_id in self._id_alias:
            return self.canonical_id(raw_track_id)

        best_cid: int | None = None
        best_sim = self.reid_merge_threshold

        for cid, card in self.cards.items():
            if card.class_name != class_name:
                continue
            if frame_idx - card.last_frame < self.min_exit_gap_frames:
                continue
            other_emb = self._embeddings.get(cid)
            if other_emb is None:
                continue
            sim = AppearanceEmbedder.cosine_similarity(emb, other_emb)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_cid is not None:
            self._id_alias[raw_track_id] = best_cid
            if raw_track_id not in self.cards[best_cid].merged_track_ids:
                self.cards[best_cid].merged_track_ids.append(raw_track_id)
            self.reid_merges += 1
            return best_cid

        self._id_alias[raw_track_id] = raw_track_id
        return raw_track_id

    def update(
        self,
        frame_idx: int,
        timestamp_s: float,
        frame_bgr: np.ndarray,
        tracks: list[Track],
        frame_dt_s: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Update Track Cards, perform cross-exit Re-ID merges, and score NL queries."""
        events: list[dict[str, Any]] = []

        for tr in tracks:
            raw_id = tr.track_id
            is_new_raw = raw_id not in self._id_alias

            # Compute embedding when track is new or on periodic refresh intervals
            emb: np.ndarray | None = None
            crop = _crop_bbox(frame_bgr, tr.bbox)
            should_embed = (
                self.embedder is not None
                and self.embedder.backend != "none"
                and (is_new_raw or (frame_idx % self.embed_every_n_frames == 0))
            )
            if should_embed and crop is not None and self.embedder is not None:
                emb = self.embedder.encode_image(crop)

            cid = self._try_cross_exit_merge(raw_id, tr.class_name, frame_idx, emb)
            if cid != raw_id:
                tr.reid_merged_from = raw_id
                tr.track_id = cid
                if is_new_raw:
                    events.append(
                        {
                            "type": "reid_cross_exit_merge",
                            "raw_track_id": raw_id,
                            "canonical_track_id": cid,
                        }
                    )

            card = self.cards.get(cid)
            if card is None:
                card = TrackCard(
                    track_id=cid,
                    class_name=tr.class_name,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    first_timestamp_s=timestamp_s,
                    last_timestamp_s=timestamp_s,
                    frame_count=1,
                    best_confidence=tr.confidence,
                    best_frame=frame_idx,
                    best_bbox=tr.bbox,
                )
                self.cards[cid] = card
                self._save_crop(card, crop)
            else:
                card.last_frame = frame_idx
                card.last_timestamp_s = timestamp_s
                card.frame_count += 1
                if tr.confidence > card.best_confidence:
                    card.best_confidence = tr.confidence
                    card.best_frame = frame_idx
                    card.best_bbox = tr.bbox
                    self._save_crop(card, crop)

            # Update zone tracking on the card
            if tr.zone is not None:
                if tr.zone not in card.zones_visited:
                    card.zones_visited.append(tr.zone)
                card.dwell_seconds[tr.zone] = card.dwell_seconds.get(tr.zone, 0.0) + frame_dt_s

            # Blend running appearance embedding
            if emb is not None:
                prev = self._embeddings.get(cid)
                if prev is None:
                    self._embeddings[cid] = emb
                else:
                    self._embeddings[cid] = _l2_normalize(0.8 * prev + 0.2 * emb)
                card.embedding = [round(float(x), 5) for x in self._embeddings[cid].tolist()]

            # Score natural-language prompt if active
            if self._query_vec is not None and cid in self._embeddings:
                score = AppearanceEmbedder.cosine_similarity(self._embeddings[cid], self._query_vec)
                card.query_score = score
                card.query_match = score >= self.query_threshold
                tr.query_score = score
                tr.query_match = card.query_match

        return events

    def _save_crop(self, card: TrackCard, crop_bgr: np.ndarray | None) -> None:
        if self.crops_dir is None or crop_bgr is None:
            return
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        filename = f"track_{card.track_id:04d}.jpg"
        out_path = self.crops_dir / filename
        cv2.imwrite(str(out_path), crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        card.thumbnail_relpath = f"crops/{filename}"

    def sorted_cards(self) -> list[TrackCard]:
        return sorted(self.cards.values(), key=lambda c: c.track_id)

    def save_json(self, path: Path, *, reid_backend: str = "none") -> None:
        payload = {
            "reid_backend": reid_backend,
            "query": self.query,
            "query_threshold": self.query_threshold,
            "reid_merges": self.reid_merges,
            "cards": [c.to_dict(include_embedding=True) for c in self.sorted_cards()],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def search_track_cards(
    run_dir: str | Path,
    query: str,
    *,
    top_k: int = 10,
    min_score: float = -1.0,
    embedder: AppearanceEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Search saved Track Cards in a run directory using a natural-language query."""
    run_path = Path(run_dir)
    cards_path = run_path / "track_cards.json" if run_path.is_dir() else run_path
    if not cards_path.is_file():
        raise FileNotFoundError(f"No track_cards.json found at {cards_path}")

    payload = json.loads(cards_path.read_text(encoding="utf-8"))
    backend = payload.get("reid_backend", "clip")
    cards: list[dict[str, Any]] = payload.get("cards", [])

    if embedder is None:
        if backend not in ("clip", "siglip"):
            raise ValueError(
                f"Run was recorded with reid_backend='{backend}', which has no text-aligned "
                "embeddings. Re-run with `--reid clip` or `--reid siglip` to enable NL search."
            )
        embedder = AppearanceEmbedder(backend=backend)

    q_vec = embedder.encode_text(query)
    scored: list[dict[str, Any]] = []
    for card in cards:
        emb = card.get("embedding")
        if not emb:
            continue
        score = AppearanceEmbedder.cosine_similarity(np.asarray(emb, dtype=np.float32), q_vec)
        if score >= min_score:
            item = {k: v for k, v in card.items() if k != "embedding"}
            item["search_score"] = round(float(score), 4)
            scored.append(item)

    scored.sort(key=lambda r: r["search_score"], reverse=True)
    return scored[:top_k]
