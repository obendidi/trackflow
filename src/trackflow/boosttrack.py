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
    XywhKalmanFilter,
    _assign,
    _boxes,
    _expand_boxes,
    _SortStyleTracker,
    _SortTrack,
    _xyxy_to_xywh,
    compute_hmiou_matrix,
    compute_ioa_matrix,
    compute_iou_matrix,
)
from trackflow.types import Detection, Track

Box = tuple[float, float, float, float]

# 99% quantile of the chi-square distribution with 4 degrees of freedom.
_MAHALANOBIS_LIMIT = 13.2767


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
    grow = np.minimum(0.75, 1.0 - track_conf)[None, :, None]
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
        duo_boost: bool = False,
        use_cmc: bool = True,
        iou_buffer: float = 0.0,
        new_track_thresh: float | None = None,
    ) -> None:
        self.det_thresh = float(det_thresh)
        self.new_track_thresh = (
            float(new_track_thresh)
            if new_track_thresh is not None
            else max(self.det_thresh, min(0.95, self.det_thresh + 0.05))
        )
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
        # Keep embeddings available down to low-confidence detections.
        self.embedding_min_confidence = min(0.20, self.det_thresh)
        super().__init__(max_age=max_age, min_hits=min_hits)

    def reset(self) -> None:
        super().reset()
        self._cmc = SparseOpticalFlowCMC() if self.use_cmc else None
        self.last_cmc_warp: np.ndarray | None = None

    def _predict(self) -> None:
        for tr in self._tracks:
            if tr.time_since_update > 0:
                tr.motion.mean[6:] = 0.0  # a missed track keeps moving but stops resizing
        super()._predict()

    @staticmethod
    def _track_confidence(tr: _SortTrack) -> float:
        if tr.age < 7:
            return 0.9 ** (7 - tr.age)
        return 0.9 ** (tr.time_since_update - 1)

    @staticmethod
    def _mahalanobis(dets: list[Detection], tracks: list[_SortTrack]) -> np.ndarray:
        """Squared NSA innovation distance of each detection to each track."""
        z = np.array([_xyxy_to_xywh(d.bbox) for d in dets], dtype=np.float64)
        nsa = np.array(
            [max(0.1, 1.0 - min(float(d.confidence), 0.95)) for d in dets], dtype=np.float64
        )
        x = np.array([t.motion.mean[:4] for t in tracks], dtype=np.float64)
        p_diag = np.array([np.diag(t.motion.cov)[:4] for t in tracks], dtype=np.float64)
        r_base = np.array([t.motion.measurement_var() for t in tracks], dtype=np.float64)
        s_diag = np.maximum(p_diag[None, :, :] + nsa[:, None, None] * r_base[None, :, :], 1e-6)
        dist = (((z[:, None, :] - x[None, :, :]) ** 2) / s_diag).sum(axis=2)
        same_class = np.equal.outer([d.class_id for d in dets], [t.class_id for t in tracks])
        return np.where(same_class, dist, np.inf)

    def _duo_promote(
        self,
        dets: list[Detection],
        tracks: list[_SortTrack],
        scores: np.ndarray,
    ) -> np.ndarray:
        """Promote isolated moderate-confidence detections that are far from existing tracks."""
        promoted = np.zeros(len(dets), dtype=bool)
        if not self.duo_boost or self._frame_id <= 1 or not dets or not tracks:
            return promoted
        duo_floor = max(0.35, self.det_thresh - 0.15)
        mahalanobis = self._mahalanobis(dets, tracks)
        d_boxes = _boxes(dets)
        t_boxes = np.array([t.motion.box for t in tracks], dtype=np.float64)
        max_track_iou = compute_iou_matrix(d_boxes, t_boxes).max(axis=1)
        candidates = np.flatnonzero(
            (mahalanobis.min(axis=1) > _MAHALANOBIS_LIMIT)
            & (max_track_iou < 0.1)
            & (scores >= duo_floor)
            & (scores < self.det_thresh)
        )
        if candidates.size:
            overlap = compute_iou_matrix(d_boxes[candidates], d_boxes[candidates])
            np.fill_diagonal(overlap, 0.0)
            for k, i in enumerate(candidates):
                if scores[i] >= scores[candidates[overlap[k] > 0.3]].max(initial=-np.inf):
                    promoted[i] = True
        return promoted

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
        d_exp = _expand_boxes(d_boxes, self.iou_buffer)
        t_exp = _expand_boxes(t_boxes, self.iou_buffer)
        iou = compute_iou_matrix(d_exp, t_exp)
        sb_iou = _soft_buffered_iou(d_exp, t_exp, track_conf)
        mahalanobis = self._mahalanobis(dets, tracks)
        low_iou_gate = min(self.iou_threshold, 0.20)
        valid = (sb_iou >= self.iou_threshold) | (
            (sb_iou >= low_iou_gate) & (mahalanobis <= _MAHALANOBIS_LIMIT)
        )
        conf = np.where(valid, scores[:, None] * track_conf[None, :], 0.0)
        sim = (
            compute_hmiou_matrix(d_boxes, t_boxes, sb_iou)
            + self.lambda_iou * conf * sb_iou
            + self.lambda_mhd * _mahalanobis_similarity(mahalanobis)
            + self.lambda_shape * conf * _shape_similarity(d_boxes, t_boxes)
        )

        known = [t.embedding for t in tracks if t.embedding is not None]
        if known and any(d.embedding is not None for d in dets):
            dim = known[0].shape[0]
            d_emb, d_has = _unit_rows([d.embedding for d in dets], dim)
            t_emb, t_has = _unit_rows([t.embedding for t in tracks], dim)
            appearance = (d_emb @ t_emb.T) * np.outer(d_has, t_has)
            sim = sim + self.lambda_emb * appearance
            valid |= (iou >= self.iou_threshold / 2) & (appearance >= 0.75)

        return _assign(sim, valid & ~np.isinf(mahalanobis))

    def _associate_low(
        self,
        tracks: list[_SortTrack],
        dets: list[Detection],
        scores: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Second-stage tight-IoU association for active tracks with low-score detections."""
        if not tracks or not dets:
            return []
        d_boxes = _boxes(dets)
        t_boxes = np.array([t.motion.box for t in tracks], dtype=np.float64)
        iou = compute_iou_matrix(d_boxes, t_boxes)
        mahalanobis = self._mahalanobis(dets, tracks)
        mhd_ok = mahalanobis <= _MAHALANOBIS_LIMIT
        valid = ((iou >= 0.45) | ((iou >= 0.35) & mhd_ok & (scores[:, None] >= 0.25))) & ~np.isinf(
            mahalanobis
        )
        sim = (
            iou
            + self.lambda_mhd * _mahalanobis_similarity(mahalanobis)
            + self.lambda_shape * scores[:, None] * _shape_similarity(d_boxes, t_boxes)
        )
        matches, _, _ = _assign(sim, valid)
        return matches

    def _remove_duplicates(self) -> None:
        tracked = [t for t in self._tracks if t.time_since_update == 0]
        lost = [t for t in self._tracks if 0 < t.time_since_update <= self.track_buffer]
        if not tracked or not lost:
            return
        t_boxes = np.array([t.box for t in tracked], dtype=np.float64)
        l_boxes = np.array([t.motion.box for t in lost], dtype=np.float64)
        iou = compute_iou_matrix(t_boxes, l_boxes)
        drop: set[int] = set()
        for p, q in zip(*np.where(iou > 0.85), strict=False):
            a, b = tracked[int(p)], lost[int(q)]
            if a.class_id != b.class_id:
                continue
            # Never delete an already-confirmed active track that matched this frame; only drop
            # the overlapping lost track, or a newly-spawned tentative duplicate under a confirmed one.
            if a.track_id or a.age >= b.age:
                drop.add(b.uid)
            else:
                drop.add(a.uid)
        if drop:
            self._tracks = [t for t in self._tracks if t.uid not in drop]

    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[Track]:
        """Advance the tracker by one frame and return the tracks reported on it."""
        self._frame_id += 1
        events: list[dict[str, Any]] = []
        self.last_cmc_warp = None
        if self._cmc is not None and frame_bgr is not None:
            # Estimate on every frame (even with no tracks) so the reference frame never goes stale.
            warp = self._cmc.estimate(frame_bgr, _boxes(detections))
            self.last_cmc_warp = warp
            if warp is not None:
                for tr in self._tracks:
                    tr.motion.apply_camera_warp(warp)
        self._predict()
        confirmed = [t for t in self._tracks if t.track_id > 0]
        unconfirmed = [t for t in self._tracks if t.track_id == 0]
        all_tracks = confirmed + unconfirmed

        raw_scores = np.array([d.confidence for d in detections], dtype=np.float64)
        duo_promoted = self._duo_promote(detections, all_tracks, raw_scores)
        high_mask = (raw_scores >= self.det_thresh) | duo_promoted
        high_indices = np.flatnonzero(high_mask)
        low_indices = np.flatnonzero(~high_mask)

        high_dets = [detections[i] for i in high_indices]
        high_scores = np.where(
            duo_promoted[high_indices], self.det_thresh + 1e-4, raw_scores[high_indices]
        )
        high_duo = duo_promoted[high_indices]

        updated_boxes: dict[int, list[Box]] = {}
        matched_uids: set[int] = set()

        def _apply_match(tr: _SortTrack, det: Detection, score: float, *, blend_emb: bool) -> None:
            tr.motion.update(det.bbox, confidence=det.confidence)
            self._observe(tr, det, tr.motion.box)
            updated_boxes.setdefault(tr.class_id, []).append(tr.box)
            matched_uids.add(tr.uid)
            if blend_emb:
                trust = (score - self.det_thresh) / max(1.0 - self.det_thresh, 1e-6)
                tr.blend_embedding(det.embedding, alpha=0.95 + 0.05 * (1.0 - trust))

        # Stage 1: match confirmed (active + lost) tracks with high-score detections.
        conf_tconf = np.array([self._track_confidence(t) for t in confirmed], dtype=np.float64)
        matches, u_det, u_conf = self._associate(confirmed, high_dets, high_scores, conf_tconf)
        for di, ti in matches:
            _apply_match(confirmed[ti], high_dets[di], float(high_scores[di]), blend_emb=True)

        # Stage 2: DLO low-score recovery for remaining active and recently missed confirmed tracks.
        used_low: set[int] = set()
        if self.dlo_boost and low_indices.size:
            active_conf = [confirmed[ti] for ti in u_conf if confirmed[ti].time_since_update == 1]
            dlo_floor = min(0.15, self.det_thresh * 0.5)
            low_keep = [i for i in low_indices.tolist() if raw_scores[i] >= dlo_floor]
            if active_conf and low_keep:
                low_dets = [detections[i] for i in low_keep]
                low_scores = raw_scores[low_keep]
                low_matches = self._associate_low(active_conf, low_dets, low_scores)
                for di, ti in low_matches:
                    _apply_match(
                        active_conf[ti], low_dets[di], float(low_scores[di]), blend_emb=False
                    )
                    used_low.add(low_keep[di])
            recent_lost = [
                confirmed[ti]
                for ti in u_conf
                if confirmed[ti].uid not in matched_uids
                and 2 <= confirmed[ti].time_since_update <= 3
            ]
            rec_keep = [
                i for i in low_indices.tolist() if i not in used_low and raw_scores[i] >= 0.30
            ]
            if recent_lost and rec_keep:
                rec_dets = [detections[i] for i in rec_keep]
                rec_scores = raw_scores[rec_keep]
                rec_boxes = _boxes(rec_dets)
                lost_boxes = np.array([t.motion.box for t in recent_lost], dtype=np.float64)
                rec_iou = compute_iou_matrix(rec_boxes, lost_boxes)
                rec_mhd = self._mahalanobis(rec_dets, recent_lost)
                rec_valid = (rec_iou >= 0.50) & (rec_mhd <= _MAHALANOBIS_LIMIT)
                rec_sim = (
                    rec_iou
                    + self.lambda_mhd * _mahalanobis_similarity(rec_mhd)
                    + self.lambda_shape
                    * rec_scores[:, None]
                    * _shape_similarity(rec_boxes, lost_boxes)
                )
                rec_matches, _, _ = _assign(rec_sim, rec_valid)
                for di, ti in rec_matches:
                    _apply_match(
                        recent_lost[ti], rec_dets[di], float(rec_scores[di]), blend_emb=False
                    )
                    used_low.add(rec_keep[di])

        # Stage 3: match unconfirmed tracks with leftover high-score detections.
        rem_high_dets = [high_dets[di] for di in u_det]
        rem_high_scores = high_scores[u_det] if u_det else np.empty(0, dtype=np.float64)
        unconf_tconf = np.array([self._track_confidence(t) for t in unconfirmed], dtype=np.float64)
        un_matches, u_rem_det, u_unconf = self._associate(
            unconfirmed, rem_high_dets, rem_high_scores, unconf_tconf
        )
        for di, ti in un_matches:
            _apply_match(
                unconfirmed[ti], rem_high_dets[di], float(rem_high_scores[di]), blend_emb=True
            )

        # When DUO boost is enabled, let DUO-spawned unconfirmed tracks continue on low-score hits.
        if self.duo_boost and self.dlo_boost and u_unconf and low_indices.size:
            rem_unconf = [
                unconfirmed[ti] for ti in u_unconf if unconfirmed[ti].time_since_update == 1
            ]
            dlo_floor = min(0.15, self.det_thresh * 0.5)
            low_keep = [
                i for i in low_indices.tolist() if i not in used_low and raw_scores[i] >= dlo_floor
            ]
            if rem_unconf and low_keep:
                low_dets = [detections[i] for i in low_keep]
                low_scores = raw_scores[low_keep]
                for di, ti in self._associate_low(rem_unconf, low_dets, low_scores):
                    _apply_match(
                        rem_unconf[ti], low_dets[di], float(low_scores[di]), blend_emb=False
                    )

        # Prune unconfirmed tracks that missed so 1-frame ghosts never linger as zombies.
        self._tracks = [t for t in self._tracks if t.track_id > 0 or t.uid in matched_uids]

        # Spawn new tracks from leftover high-score detections above new_track_thresh (or DUO-promoted).
        leftovers = sorted(
            (u_det[i] for i in u_rem_det), key=lambda di: float(high_scores[di]), reverse=True
        )
        for di in leftovers:
            det = high_dets[di]
            if not high_duo[di] and det.confidence < self.new_track_thresh:
                continue
            same_class = updated_boxes.get(det.class_id)
            if (
                same_class
                and float(compute_ioa_matrix(_boxes([det]), np.array(same_class)).max()) >= 0.8
            ):
                continue
            tr = self._spawn(det, XywhKalmanFilter(det.bbox, height_scaled=True, min_nsa=0.03))
            tr.blend_embedding(det.embedding, alpha=0.0)
            updated_boxes.setdefault(det.class_id, []).append(det.bbox)
        self._remove_duplicates()
        return self._finish_frame(events)
