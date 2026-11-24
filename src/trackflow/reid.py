"""Person Re-ID, vision-language queries, Track Cards, opt-in cross-exit merges, and NL search.

Two different kinds of embeddings live here and must not be mixed up:

* :class:`ReidEmbedder` (``--reid osnet|color``) produces *identity* embeddings trained to tell
  people apart. Only these are used for BoT-SORT appearance matching and cross-exit merges.
* :class:`VlmEmbedder` (``--vlm clip|siglip``) produces image/text-aligned embeddings. They are
  great for "person in a red jacket" queries and post-run search, but they barely separate
  individuals (on vtest.avi, different people scored 0.83 cosine on average vs 0.86 for the same
  person), so they are never used to decide identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from tokenizers import Tokenizer

from trackflow.detector import create_onnx_session
from trackflow.models import (
    REID_REGISTRY,
    Precision,
    resolve_reid_model,
    resolve_vlm_artifacts,
)
from trackflow.types import Track, TrackCard

ReidBackend = Literal["none", "color", "osnet"]
VlmBackend = Literal["none", "clip", "siglip"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = _l2_normalize(a)
    nb = _l2_normalize(b)
    if na.shape != nb.shape:
        return 0.0
    return float(np.dot(na, nb))


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


class ReidEmbedder:
    """Person identity embeddings: OSNet-AIN (ONNX) or a weight-free HSV color descriptor."""

    def __init__(
        self,
        backend: ReidBackend = "osnet",
        *,
        device: str = "auto",
        cache_dir: str | Path | None = None,
    ) -> None:
        if backend not in ("color", "osnet"):
            raise ValueError(f"Unknown Re-ID backend '{backend}'. Choose 'osnet' or 'color'.")
        self.backend: ReidBackend = backend
        self._sess = None
        self._input_name = ""
        self._input_hw = (256, 128)
        if backend == "osnet":
            spec = REID_REGISTRY["osnet"]
            self._input_hw = spec.input_hw
            self._sess, _ = create_onnx_session(
                resolve_reid_model("osnet", cache_dir=cache_dir), device=device
            )
            self._input_name = self._sess.get_inputs()[0].name

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = self._input_hw
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return ((x - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)

    def encode_images(self, crops: list[np.ndarray | None]) -> list[np.ndarray | None]:
        """Embed a batch of BGR crops; ``None``/empty crops map to ``None``."""
        out: list[np.ndarray | None] = [None] * len(crops)
        valid = [i for i, c in enumerate(crops) if c is not None and c.size > 0]
        if not valid:
            return out
        if self.backend == "color":
            for i in valid:
                out[i] = _color_spatial_embedding(crops[i])  # type: ignore[arg-type]
            return out
        assert self._sess is not None
        batch = np.stack([self._preprocess(crops[i]) for i in valid])  # type: ignore[arg-type]
        embs = self._sess.run(None, {self._input_name: batch})[0]
        for i, e in zip(valid, embs, strict=True):
            out[i] = _l2_normalize(e)
        return out

    def encode_image(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        return self.encode_images([crop_bgr])[0]


class VlmEmbedder:
    """CLIP / SigLIP image+text embeddings for natural-language queries and search."""

    def __init__(
        self,
        backend: VlmBackend = "clip",
        *,
        precision: Precision = "int8",
        device: str = "auto",
        cache_dir: str | Path | None = None,
    ) -> None:
        if backend not in ("clip", "siglip"):
            raise ValueError(f"Unknown VLM backend '{backend}'. Choose 'clip' or 'siglip'.")
        self.backend: VlmBackend = backend
        self.precision = precision
        self._vision_sess = None
        self._text_sess = None
        self._tokenizer: Tokenizer | None = None
        self._image_size = (224, 224)
        self._shortest_edge: int | None = None
        self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self._max_length = 77 if backend == "clip" else 64
        self._text_cache: dict[str, np.ndarray] = {}

        artifacts = resolve_vlm_artifacts(backend=backend, precision=precision, cache_dir=cache_dir)
        eff_dev = "cpu" if (device == "auto" and precision == "int8") else device
        # ONNX Runtime's default CoreML setup: MLProgram can't load these INT8 graphs, and
        # NeuralNetwork limited to CPU and GPU returns NaN for some CLIP crops.
        self._vision_sess, _ = create_onnx_session(
            artifacts["vision_onnx"], device=eff_dev, coreml_format=None
        )
        self._text_sess, _ = create_onnx_session(
            artifacts["text_onnx"], device=eff_dev, coreml_format=None
        )
        self._tokenizer = Tokenizer.from_file(str(artifacts["tokenizer"]))
        self._load_preprocess(artifacts["preprocessor"], artifacts["config"])

    @property
    def supports_text(self) -> bool:
        return self._text_sess is not None and self._tokenizer is not None

    def _load_preprocess(self, prep_path: Path, config_path: Path) -> None:
        if prep_path.is_file():
            cfg = json.loads(prep_path.read_text(encoding="utf-8"))
            if cfg.get("do_center_crop") and isinstance(cfg.get("size"), dict):
                self._shortest_edge = cfg["size"].get("shortest_edge")
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
        if self._shortest_edge:
            h, w = rgb.shape[:2]
            scale = self._shortest_edge / min(h, w)
            rh, rw = max(th, int(h * scale)), max(tw, int(w * scale))
            resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_CUBIC)
            top, left = (rh - th) // 2, (rw - tw) // 2
            resized = resized[top : top + th, left : left + tw]
        else:
            resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_CUBIC)
        arr = resized.astype(np.float32) * (1.0 / 255.0)
        chw = np.transpose(arr, (2, 0, 1))
        chw = (chw - self._mean) / self._std
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def encode_image(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        """Encode a BGR crop into an L2-normalized image embedding (text-aligned)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
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
        return cosine_similarity(a, b)


class TrackCardIndex:
    """Persistent Track Cards, NL query scores, and opt-in cross-exit identity merges.

    Cross-exit merging (``reid_merge=True``, requires a :class:`ReidEmbedder`) gives a returning
    person their earlier ID. A new track is compared only after ``merge_min_samples`` identity
    samples, only against cards that ended at least ``min_exit_gap_frames`` before it started (so
    two people visible at the same time can never share an ID), and it merges only when the best
    match clears ``reid_merge_threshold`` *and* beats the runner-up by ``reid_merge_margin``.
    Look-alikes (e.g. several people in dark jackets) are left unmerged rather than guessed.
    """

    def __init__(
        self,
        *,
        vlm: VlmEmbedder | None = None,
        reid: ReidEmbedder | None = None,
        query: str | None = None,
        query_threshold: float = 0.22,
        reid_merge: bool = False,
        reid_merge_threshold: float = 0.80,
        reid_merge_margin: float = 0.05,
        merge_min_samples: int = 3,
        reid_sample_every: int = 3,
        min_reid_height: float = 48.0,
        min_exit_gap_frames: int = 8,
        embed_every_n_frames: int = 5,
        crops_dir: Path | None = None,
    ) -> None:
        if reid_merge and reid is None:
            raise ValueError("Cross-exit merging needs a person Re-ID embedder (--reid osnet).")
        self.vlm = vlm
        self.reid = reid
        self.query = query.strip() if query else None
        self.query_threshold = float(query_threshold)
        self.reid_merge = bool(reid_merge)
        self.reid_merge_threshold = float(reid_merge_threshold)
        self.reid_merge_margin = float(reid_merge_margin)
        self.merge_min_samples = max(1, int(merge_min_samples))
        self.reid_sample_every = max(1, int(reid_sample_every))
        self.min_reid_height = float(min_reid_height)
        self.min_exit_gap_frames = int(min_exit_gap_frames)
        self.embed_every_n_frames = max(1, int(embed_every_n_frames))
        self.crops_dir = crops_dir

        self._query_vec: np.ndarray | None = None
        if self.query and self.vlm is not None and self.vlm.supports_text:
            self._query_vec = self.vlm.encode_text(self.query)

        self.cards: dict[int, TrackCard] = {}
        self._vlm_emb: dict[int, np.ndarray] = {}
        self._reid_sum: dict[int, np.ndarray] = {}
        self._alias: dict[int, int] = {}  # raw tracker ID -> canonical card ID
        self._raw_first_frame: dict[int, int] = {}
        self._pending: dict[int, list[np.ndarray]] = {}
        self._decided: set[int] = set()
        self.reid_merges: int = 0

    def canonical_id(self, raw_track_id: int) -> int:
        cid = raw_track_id
        while cid in self._alias:
            cid = self._alias[cid]
        return cid

    def _reid_vec(self, cid: int) -> np.ndarray | None:
        s = self._reid_sum.get(cid)
        return None if s is None else _l2_normalize(s)

    def _try_merge(self, raw: int, class_name: str, visible: set[int]) -> tuple[int, float] | None:
        samples = self._pending.pop(raw, [])
        self._decided.add(raw)
        if not samples:
            return None
        new_vec = _l2_normalize(np.mean(samples, axis=0))
        start = self._raw_first_frame[raw]
        sims: list[tuple[float, int]] = []
        for cid, card in self.cards.items():
            if cid in visible or card.class_name != class_name:
                continue
            if start - card.last_frame < self.min_exit_gap_frames:
                continue  # overlaps in time (or too recent) -> must be someone else
            vec = self._reid_vec(cid)
            if vec is not None:
                sims.append((float(np.dot(new_vec, vec)), cid))
        if not sims:
            return None
        sims.sort(reverse=True)
        best, best_cid = sims[0]
        runner_up = sims[1][0] if len(sims) > 1 else -1.0
        if best >= self.reid_merge_threshold and best - runner_up >= self.reid_merge_margin:
            return best_cid, best
        return None

    def _fold_card(self, src_id: int, dst_id: int) -> None:
        src = self.cards.pop(src_id)
        dst = self.cards[dst_id]
        if src.first_frame < dst.first_frame:
            dst.first_frame, dst.first_timestamp_s = src.first_frame, src.first_timestamp_s
        if src.last_frame > dst.last_frame:
            dst.last_frame, dst.last_timestamp_s = src.last_frame, src.last_timestamp_s
        dst.frame_count += src.frame_count
        if src.best_confidence > dst.best_confidence:
            dst.best_confidence, dst.best_frame, dst.best_bbox = (
                src.best_confidence,
                src.best_frame,
                src.best_bbox,
            )
            dst.thumbnail_relpath = src.thumbnail_relpath or dst.thumbnail_relpath
        for z in src.zones_visited:
            if z not in dst.zones_visited:
                dst.zones_visited.append(z)
        for z, s in src.dwell_seconds.items():
            dst.dwell_seconds[z] = dst.dwell_seconds.get(z, 0.0) + s
        dst.merged_track_ids.extend([src_id, *src.merged_track_ids])

        if src_id in self._reid_sum:
            self._reid_sum[dst_id] = self._reid_sum.get(dst_id, 0.0) + self._reid_sum.pop(src_id)
        if src_id in self._vlm_emb:
            if dst_id in self._vlm_emb:
                self._vlm_emb[dst_id] = _l2_normalize(
                    self._vlm_emb[dst_id] + self._vlm_emb.pop(src_id)
                )
            else:
                self._vlm_emb[dst_id] = self._vlm_emb.pop(src_id)
            dst.embedding = [round(float(x), 5) for x in self._vlm_emb[dst_id].tolist()]

    def update(
        self,
        frame_idx: int,
        timestamp_s: float,
        frame_bgr: np.ndarray,
        tracks: list[Track],
        frame_dt_s: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Update Track Cards, perform opt-in cross-exit merges, and score NL queries."""
        events: list[dict[str, Any]] = []

        # Pass 1: crops + batched identity embeddings.
        crops: list[np.ndarray | None] = []
        reid_crops: list[np.ndarray | None] = []
        for tr in tracks:
            raw = tr.track_id
            is_new = raw not in self._raw_first_frame
            if is_new:
                self._raw_first_frame[raw] = frame_idx
            crop = _crop_bbox(frame_bgr, tr.bbox)
            crops.append(crop)
            want = (
                self.reid_merge
                and self.reid is not None
                and tr.class_name.lower() == "person"
                and crop is not None
                and tr.height >= self.min_reid_height
                and (is_new or frame_idx % self.reid_sample_every == 0)
            )
            reid_crops.append(crop if want else None)
        reid_embs = (
            self.reid.encode_images(reid_crops)
            if self.reid is not None and any(c is not None for c in reid_crops)
            else [None] * len(tracks)
        )

        # Pass 2: merges, cards, query scores.
        visible = {self.canonical_id(tr.track_id) for tr in tracks} if self.reid_merge else set()
        for tr, crop, remb in zip(tracks, crops, reid_embs, strict=True):
            raw = tr.track_id
            cid = self.canonical_id(raw)
            is_new_card = cid not in self.cards

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

            if remb is not None:
                self._reid_sum[cid] = self._reid_sum.get(cid, 0.0) + remb

            # Opt-in cross-exit merge, decided once per raw track after enough samples.
            if self.reid_merge and raw not in self._decided and raw not in self._alias:
                if remb is not None:
                    self._pending.setdefault(raw, []).append(remb)
                if len(self._pending.get(raw, [])) >= self.merge_min_samples:
                    hit = self._try_merge(raw, tr.class_name, visible)
                    if hit is not None:
                        dst, sim = hit
                        self._fold_card(raw, dst)
                        self._alias[raw] = dst
                        self.reid_merges += 1
                        events.append(
                            {
                                "type": "reid_cross_exit_merge",
                                "raw_track_id": raw,
                                "canonical_track_id": dst,
                                "similarity": round(sim, 3),
                            }
                        )
                        cid = dst
                        visible.add(dst)
                        card = self.cards[cid]

            if cid != raw:
                tr.reid_merged_from = raw
                tr.track_id = cid

            if tr.zone is not None:
                if tr.zone not in card.zones_visited:
                    card.zones_visited.append(tr.zone)
                card.dwell_seconds[tr.zone] = card.dwell_seconds.get(tr.zone, 0.0) + frame_dt_s

            # Text-aligned embedding for --query scoring and post-run search.
            if (
                self.vlm is not None
                and crop is not None
                and (is_new_card or frame_idx % self.embed_every_n_frames == 0)
            ):
                vemb = self.vlm.encode_image(crop)
                if vemb is not None:
                    prev = self._vlm_emb.get(cid)
                    self._vlm_emb[cid] = (
                        vemb if prev is None else _l2_normalize(0.8 * prev + 0.2 * vemb)
                    )
                    card.embedding = [round(float(x), 5) for x in self._vlm_emb[cid].tolist()]

            if self._query_vec is not None and cid in self._vlm_emb:
                score = cosine_similarity(self._vlm_emb[cid], self._query_vec)
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

    def save_json(
        self, path: Path, *, reid_backend: str = "none", vlm_backend: str = "none"
    ) -> None:
        payload = {
            "reid_backend": reid_backend,
            "vlm_backend": vlm_backend,
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
    embedder: VlmEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Search saved Track Cards in a run directory using a natural-language query."""
    run_path = Path(run_dir)
    cards_path = run_path / "track_cards.json" if run_path.is_dir() else run_path
    if not cards_path.is_file():
        raise FileNotFoundError(f"No track_cards.json found at {cards_path}")

    payload = json.loads(cards_path.read_text(encoding="utf-8"))
    backend = payload.get("vlm_backend")
    if backend in (None, "none") and payload.get("reid_backend") in ("clip", "siglip"):
        backend = payload["reid_backend"]  # runs recorded before --vlm existed
    cards: list[dict[str, Any]] = payload.get("cards", [])

    if embedder is None:
        if backend not in ("clip", "siglip"):
            raise ValueError(
                "This run has no text-aligned embeddings. Re-run with `--vlm clip` (or "
                "`--vlm siglip`, or any `--query`) to enable natural-language search."
            )
        embedder = VlmEmbedder(backend=backend)

    q_vec = embedder.encode_text(query)
    scored: list[dict[str, Any]] = []
    for card in cards:
        emb = card.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(np.asarray(emb, dtype=np.float32), q_vec)
        if score >= min_score:
            item = {k: v for k, v in card.items() if k != "embedding"}
            item["search_score"] = round(float(score), 4)
            scored.append(item)

    scored.sort(key=lambda r: r["search_score"], reverse=True)
    return scored[:top_k]
