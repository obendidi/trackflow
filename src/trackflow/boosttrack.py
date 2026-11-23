"""BoostTrack++ multi-object tracker (Stanojević & Todorović, 2024).

A port of the reference implementation (github.com/vukasin-stanojevic/BoostTrack, MIT license)
with its BoostTrack++ settings, onto Trackflow's track lifecycle. Association is class-aware, and
each assignment only considers pairs that pass the IoU gate. Camera motion is estimated with
sparse optical flow (the reference uses ECC), and shape similarity normalizes the height
difference by height.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trackflow.tracker import (
    SparseOpticalFlowCMC,
    _assign,
    _boxes,
    _expand_boxes,
    _SortStyleTracker,
    _SortTrack,
    compute_iou_matrix,
)
from trackflow.types import Detection, Track

Box = tuple[float, float, float, float]

# 99% quantile of the chi-square distribution with 4 degrees of freedom.
_MAHALANOBIS_LIMIT = 13.2767


def _box_to_z(box: Box) -> np.ndarray:
    """``[cx, cy, h, w/h]``."""
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    return np.array([x1 + 0.5 * w, y1 + 0.5 * h, h, w / h], dtype=np.float64)


def _z_to_box(z: np.ndarray) -> Box:
    h = max(float(z[2]), 1e-3)
    w = max(float(z[3]), 0.0) * h
    cx, cy = float(z[0]), float(z[1])
    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


class _BoostMotion:
    """Constant-velocity Kalman filter on ``[cx, cy, h, w/h]`` with BoostTrack's fixed noise."""

    _F = np.eye(8, dtype=np.float64) + np.eye(8, k=4, dtype=np.float64)
    _H = np.eye(4, 8, dtype=np.float64)
    _Q = np.diag([1.0, 1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2, 1e-2])
    _R = np.diag([1.0, 1.0, 10.0, 1e-2])

    def __init__(self, box: Box) -> None:
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = _box_to_z(box)
        self.P = np.diag([10.0] * 4 + [1e4] * 4)

    @property
    def box(self) -> Box:
        return _z_to_box(self.x)

    def predict(self) -> None:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q

    def update(self, box: Box) -> None:
        s = self._H @ self.P @ self._H.T + self._R
        gain = np.linalg.solve(s, self._H @ self.P).T
        self.x = self.x + gain @ (_box_to_z(box) - self._H @ self.x)
        self.P = self.P - gain @ s @ gain.T

    def warp(self, affine: np.ndarray) -> None:
        """Move the current box by a 2x3 camera-motion transform (velocities are kept)."""
        x1, y1, x2, y2 = self.box
        (ax1, ay1), (ax2, ay2) = (affine[:, :2] @ np.array([[x1, x2], [y1, y2]])).T + affine[:, 2]
        self.x[:4] = _box_to_z((ax1, ay1, ax2, ay2))


def _mahalanobis_similarity(dist: np.ndarray) -> np.ndarray:
    """Softmax over detections of ``limit - distance`` for each track; zero beyond the limit."""
    far = dist > _MAHALANOBIS_LIMIT
    weights = np.exp(_MAHALANOBIS_LIMIT - np.minimum(dist, _MAHALANOBIS_LIMIT))
    sim = weights / weights.sum(axis=0, keepdims=True)
    sim[far] = 0.0
    return sim


def _shape_similarity(dets: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    dw = (dets[:, 2] - dets[:, 0])[:, None]
    dh = (dets[:, 3] - dets[:, 1])[:, None]
    tw = (tracks[:, 2] - tracks[:, 0])[None, :]
    th = (tracks[:, 3] - tracks[:, 1])[None, :]
    return np.exp(
        -(
            np.abs(dw - tw) / np.maximum(np.maximum(dw, tw), 1e-6)
            + np.abs(dh - th) / np.maximum(np.maximum(dh, th), 1e-6)
        )
    )


def _soft_buffered_iou(dets: np.ndarray, tracks: np.ndarray, track_conf: np.ndarray) -> np.ndarray:
    """IoU after growing both boxes in proportion to how uncertain the track is."""
    grow = (1.0 - track_conf)[None, :, None]
    d = dets[:, None, :]
    t = tracks[None, :, :]
    d_size = np.concatenate([d[..., 2:] - d[..., :2]] * 2, axis=-1)
    t_size = np.concatenate([t[..., 2:] - t[..., :2]] * 2, axis=-1)
    sign = np.array([-1.0, -1.0, 1.0, 1.0])
    db = d + sign * d_size * grow * 0.25
    tb = t + sign * t_size * grow * 0.5
    iw = np.maximum(0.0, np.minimum(db[..., 2], tb[..., 2]) - np.maximum(db[..., 0], tb[..., 0]))
    ih = np.maximum(0.0, np.minimum(db[..., 3], tb[..., 3]) - np.maximum(db[..., 1], tb[..., 1]))
    inter = iw * ih
    area_d = (db[..., 2] - db[..., 0]) * (db[..., 3] - db[..., 1])
    area_t = (tb[..., 2] - tb[..., 0]) * (tb[..., 3] - tb[..., 1])
    return inter / np.maximum(area_d + area_t - inter, 1e-6)


def _unit_rows(vectors: list[np.ndarray | None], dim: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(vectors), dim), dtype=np.float64)
    has = np.zeros(len(vectors), dtype=bool)
    for i, v in enumerate(vectors):
        if v is None:
            continue
        norm = float(np.linalg.norm(v))
        if norm > 1e-6 and v.shape[0] == dim:
            out[i] = v / norm
            has[i] = True
    return out, has


class BoostTracker(_SortStyleTracker):
    """BoostTrack++: SORT with a richer similarity and detection-confidence boosting.

    Each frame:

    1. tracks move with camera motion (``use_cmc``), then are predicted;
    2. detection scores are boosted. A detection that closely matches a track (by Mahalanobis
       distance, shape and a soft buffered IoU) is likely a real object (DLO boost). A
       low-score detection far from every track, and stronger than overlapping such detections,
       is likely a new object (DUO boost). Detections still below ``det_thresh`` are dropped;
    3. detections <-> tracks by IoU, plus IoU weighted by detection and track confidence,
       Mahalanobis and shape similarity, and appearance when detections carry Re-ID
       embeddings. A pair needs IoU >= ``iou_threshold``, or half that with appearance
       similarity >= 0.75;
    4. unmatched detections spawn tracks.

    A track's confidence is low while it is young and decays while it goes unmatched.
    Confirmed tracks report the filter's estimate.
    """

    name: str = "boosttrack"

    def __init__(
        self,
        *,
        det_thresh: float = 0.5,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25,
        dlo_boost: bool = True,
        duo_boost: bool = True,
        use_cmc: bool = True,
        iou_buffer: float = 0.0,
    ) -> None:
        self.det_thresh = float(det_thresh)
        self.iou_threshold = float(iou_threshold)
        self.lambda_iou = float(lambda_iou)
        self.lambda_mhd = float(lambda_mhd)
        self.lambda_shape = float(lambda_shape)
        self.lambda_emb = 1.5 * (1.0 + self.lambda_iou + self.lambda_shape + self.lambda_mhd)
        self.dlo_boost = bool(dlo_boost)
        self.duo_boost = bool(duo_boost)
        self.use_cmc = bool(use_cmc)
        # Buffered IoU for association (see iou_buffer_for_interval).
        self.iou_buffer = float(iou_buffer)
        # Boosting can promote any detection, so every detection may need an embedding.
        self.embedding_min_confidence = 0.0
        super().__init__(max_age=max_age, min_hits=min_hits)

    def reset(self) -> None:
        super().reset()
        self._cmc = SparseOpticalFlowCMC() if self.use_cmc else None

    @staticmethod
    def _track_confidence(tr: _SortTrack) -> float:
        if tr.age < 7:
            return 0.9 ** (7 - tr.age)
        return 0.9 ** (tr.time_since_update - 1)

    @staticmethod
    def _mahalanobis(dets: list[Detection], tracks: list[_SortTrack]) -> np.ndarray:
        """Squared distance of each detection to each track, using the state variances."""
        z = np.array([_box_to_z(d.bbox) for d in dets])
        x = np.array([t.motion.x[:4] for t in tracks])
        inv_var = np.array([1.0 / np.diag(t.motion.P)[:4] for t in tracks])
        dist = (((z[:, None, :] - x[None, :, :]) ** 2) * inv_var[None, :, :]).sum(axis=2)
        same_class = np.equal.outer([d.class_id for d in dets], [t.class_id for t in tracks])
        return np.where(same_class, dist, np.inf)

    def _boost(
        self, dets: list[Detection], tracks: list[_SortTrack], track_conf: np.ndarray
    ) -> np.ndarray:
        scores = np.array([d.confidence for d in dets], dtype=np.float64)
        mahalanobis = self._mahalanobis(dets, tracks)
        d_boxes = _boxes(dets)
        t_boxes = np.array([t.motion.box for t in tracks], dtype=np.float64)

        if self.dlo_boost:
            sim = (
                _mahalanobis_similarity(mahalanobis)
                + _shape_similarity(d_boxes, t_boxes)
                + _soft_buffered_iou(d_boxes, t_boxes, track_conf)
            ) / 3.0
            sim[np.isinf(mahalanobis)] = 0.0
            scores = np.maximum(scores, 0.65 * scores + 0.35 * sim.max(axis=1) ** 1.5)
            missed = np.array([t.time_since_update - 1 for t in tracks], dtype=np.float64)
            needed = np.maximum(0.95 - missed * (0.15 / 20), 0.8)
            likely = (sim > needed[None, :]).any(axis=1)
            scores[likely] = np.maximum(scores[likely], self.det_thresh + 1e-5)

        if self.duo_boost and self._frame_id > 1:
            candidates = np.flatnonzero(
                (mahalanobis.min(axis=1) > _MAHALANOBIS_LIMIT) & (scores < self.det_thresh)
            )
            if candidates.size:
                overlap = compute_iou_matrix(d_boxes[candidates], d_boxes[candidates])
                np.fill_diagonal(overlap, 0.0)
                strongest = [
                    i
                    for k, i in enumerate(candidates)
                    if scores[i] >= scores[candidates[overlap[k] > 0.3]].max(initial=-np.inf)
                ]
                scores[strongest] = self.det_thresh + 1e-4
        return scores

    def _associate(
        self,
        tracks: list[_SortTrack],
        dets: list[Detection],
        scores: np.ndarray,
        track_conf: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match detections (rows) to tracks (columns)."""
        if not tracks or not dets:
            return [], list(range(len(dets))), list(range(len(tracks)))
        d_boxes = _boxes(dets)
        t_boxes = np.array([t.motion.box for t in tracks], dtype=np.float64)
        iou = compute_iou_matrix(
            _expand_boxes(d_boxes, self.iou_buffer), _expand_boxes(t_boxes, self.iou_buffer)
        )
        conf = scores[:, None] * track_conf[None, :]
        conf[iou < self.iou_threshold] = 0.0
        mahalanobis = self._mahalanobis(dets, tracks)
        sim = (
            iou
            + self.lambda_iou * conf * iou
            + self.lambda_mhd * _mahalanobis_similarity(mahalanobis)
            + self.lambda_shape * conf * _shape_similarity(d_boxes, t_boxes)
        )
        valid = iou >= self.iou_threshold

        known = [t.embedding for t in tracks if t.embedding is not None]
        if known and any(d.embedding is not None for d in dets):
            dim = known[0].shape[0]
            d_emb, d_has = _unit_rows([d.embedding for d in dets], dim)
            t_emb, t_has = _unit_rows([t.embedding for t in tracks], dim)
            appearance = (d_emb @ t_emb.T) * np.outer(d_has, t_has)
            sim = sim + self.lambda_emb * appearance
            valid |= (iou >= self.iou_threshold / 2) & (appearance >= 0.75)

        return _assign(sim, valid & ~np.isinf(mahalanobis))

    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[Track]:
        """Advance the tracker by one frame and return the tracks reported on it."""
        self._frame_id += 1
        events: list[dict[str, Any]] = []
        if self._cmc is not None and frame_bgr is not None:
            # Estimate on every frame (even with no tracks) so the reference frame never goes stale.
            warp = self._cmc.estimate(frame_bgr)
            if warp is not None:
                for tr in self._tracks:
                    tr.motion.warp(warp)
        self._predict()
        tracks = list(self._tracks)
        track_conf = np.array([self._track_confidence(t) for t in tracks], dtype=np.float64)

        if tracks and detections:
            boosted = self._boost(detections, tracks, track_conf)
        else:
            boosted = np.array([d.confidence for d in detections], dtype=np.float64)
        keep = np.flatnonzero(boosted >= self.det_thresh)
        dets = [detections[i] for i in keep]
        scores = boosted[keep]

        matches, u_det, _ = self._associate(tracks, dets, scores, track_conf)
        for di, ti in matches:
            tr, det = tracks[ti], dets[di]
            tr.motion.update(det.bbox)
            self._observe(tr, det, tr.motion.box)
            trust = (scores[di] - self.det_thresh) / max(1.0 - self.det_thresh, 1e-6)
            tr.blend_embedding(det.embedding, alpha=0.95 + 0.05 * (1.0 - trust))
        for di in u_det:
            tr = self._spawn(dets[di], _BoostMotion(dets[di].bbox))
            tr.blend_embedding(dets[di].embedding, alpha=0.0)
        return self._finish_frame(events)
