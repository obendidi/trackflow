"""Non-linear multi-object tracker with dual-anchor motion and adaptive gallery Re-ID.

Clean-room Apache-2.0 tracker combining:

1. **Dual-anchor non-linear motion**: NSA :class:`XywhKalmanFilter` + :class:`SparseOpticalFlowCMC`
   with observation-centric re-update (ORU) and momentum (OCM), exponential velocity damping on
   lost tracks (``vx, vy`` damped per missed frame, ``vw, vh`` zeroed), and dual-anchor spatial
   affinity ``max(HMIoU(damped_pred), HMIoU(last_obs))`` with controlled occlusion-age expansion.
2. **Clean-bank adaptive appearance**: 15-frame ring buffer plus clean EMA per track updated only
   on clean crops (overlap ``<= 0.40``, or when the bank is empty), scored via blended max-bank/EMA
   cosine with per-sequence running non-matching cosine normalization (GHOST-style), virtual-null
   dual-softmax, and row/col margin weighting fused additively with motion.
3. **Confirmed-track immediate recovery + iterative mutual-best association**: iterative mutual-best
   (TPA-style) followed by Hungarian assignment, immediate re-reporting on recovery, extended
   buffer (``>= 60`` frames), guarded low-score recovery for recently lost tracks, and track-aware
   duplicate-spawn blocking (TAI).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trackflow.tracker import (
    SparseOpticalFlowCMC,
    XywhKalmanFilter,
    _assign,
    _boxes,
    _expand_boxes,
    _switch_event,
    compute_hmiou_matrix,
    compute_ioa_matrix,
    compute_iou_matrix,
)
from trackflow.types import Detection, Track

Box = tuple[float, float, float, float]


def _centers(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)


def _direction(a: Box, b: Box) -> np.ndarray:
    """Unit vector ``(dy, dx)`` from center of ``a`` to center of ``b``; zero if nearly static."""
    dx = 0.5 * ((b[0] + b[2]) - (a[0] + a[2]))
    dy = 0.5 * ((b[1] + b[3]) - (a[1] + a[3]))
    norm = float(np.hypot(dy, dx))
    if norm < 1.0:
        return np.zeros(2, dtype=np.float64)
    return np.array([dy / norm, dx / norm], dtype=np.float64)


def _warp_box(box: Box, rot: np.ndarray, trans: np.ndarray, scale: float) -> Box:
    x1, y1, x2, y2 = box
    cx, cy = rot @ np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64) + trans
    hw = max(0.5, (x2 - x1) * scale * 0.5)
    hh = max(0.5, (y2 - y1) * scale * 0.5)
    return (float(cx - hw), float(cy - hh), float(cx + hw), float(cy + hh))


def _expanded_iou_hmiou(
    t_boxes: np.ndarray,
    d_boxes: np.ndarray,
    t_expand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise IoU and HMIoU where each track row ``i`` expands both boxes by ``t_expand[i]``."""
    n, m = len(t_boxes), len(d_boxes)
    if n == 0 or m == 0:
        empty = np.zeros((n, m), dtype=np.float64)
        return empty, empty

    exp = t_expand[:, None]
    tw = (t_boxes[:, 2] - t_boxes[:, 0])[:, None]
    th = (t_boxes[:, 3] - t_boxes[:, 1])[:, None]
    dw = (d_boxes[:, 2] - d_boxes[:, 0])[None, :]
    dh = (d_boxes[:, 3] - d_boxes[:, 1])[None, :]

    tx1 = t_boxes[:, 0:1] - exp * tw
    ty1 = t_boxes[:, 1:2] - exp * th
    tx2 = t_boxes[:, 2:3] + exp * tw
    ty2 = t_boxes[:, 3:4] + exp * th

    dx1 = d_boxes[None, :, 0] - exp * dw
    dy1 = d_boxes[None, :, 1] - exp * dh
    dx2 = d_boxes[None, :, 2] + exp * dw
    dy2 = d_boxes[None, :, 3] + exp * dh

    iw = np.maximum(0.0, np.minimum(tx2, dx2) - np.maximum(tx1, dx1))
    ih = np.maximum(0.0, np.minimum(ty2, dy2) - np.maximum(ty1, dy1))
    inter = iw * ih
    area_t = np.maximum(0.0, tx2 - tx1) * np.maximum(0.0, ty2 - ty1)
    area_d = np.maximum(0.0, dx2 - dx1) * np.maximum(0.0, dy2 - dy1)
    iou = inter / np.maximum(area_t + area_d - inter, 1e-6)

    v_inter = np.minimum(t_boxes[:, None, 3], d_boxes[None, :, 3]) - np.maximum(
        t_boxes[:, None, 1], d_boxes[None, :, 1]
    )
    v_union = np.maximum(t_boxes[:, None, 3], d_boxes[None, :, 3]) - np.minimum(
        t_boxes[:, None, 1], d_boxes[None, :, 1]
    )
    hmiou = iou * (0.75 + 0.25 * np.maximum(v_inter, 0.0) / np.maximum(v_union, 1e-6))
    return iou, hmiou


def _mutual_best_assign(
    similarity: np.ndarray, valid: np.ndarray
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Iterative mutual-best matching over valid pairs, finishing leftovers with Hungarian."""
    n, m = similarity.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    s = np.where(valid, similarity, -np.inf)
    row_alive = np.ones(n, dtype=bool)
    col_alive = np.ones(m, dtype=bool)
    matches: list[tuple[int, int]] = []

    while bool(row_alive.any()) and bool(col_alive.any()):
        sub_r = np.flatnonzero(row_alive)
        sub_c = np.flatnonzero(col_alive)
        sub_s = s[np.ix_(sub_r, sub_c)]
        row_best_c = np.argmax(sub_s, axis=1)
        col_best_r = np.argmax(sub_s, axis=0)
        found = False
        for ri_local, ci_local in enumerate(row_best_c.tolist()):
            if not np.isfinite(float(sub_s[ri_local, ci_local])):
                continue
            if int(col_best_r[ci_local]) == ri_local:
                r_idx = int(sub_r[ri_local])
                c_idx = int(sub_c[ci_local])
                matches.append((r_idx, c_idx))
                row_alive[r_idx] = False
                col_alive[c_idx] = False
                found = True
        if not found:
            break

    rem_r = np.flatnonzero(row_alive).tolist()
    rem_c = np.flatnonzero(col_alive).tolist()
    if rem_r and rem_c:
        sub_m, left_r, left_c = _assign(
            similarity[np.ix_(rem_r, rem_c)], valid[np.ix_(rem_r, rem_c)]
        )
        for ri, ci in sub_m:
            matches.append((rem_r[ri], rem_c[ci]))
        rem_r = [rem_r[i] for i in left_r]
        rem_c = [rem_c[i] for i in left_c]
    return matches, rem_r, rem_c


class _NlMotion:
    """NSA xywh Kalman filter with lost-track velocity damping, OCM direction, and ORU re-update."""

    def __init__(self, box: Box, *, delta_t: int = 3, vel_damp: float = 0.80) -> None:
        self.kf = XywhKalmanFilter(box, height_scaled=True, min_nsa=0.03)
        self.delta_t = int(delta_t)
        self.vel_damp = float(vel_damp)
        self.observations: dict[int, Box] = {0: box}
        self.last_observation: Box = box
        self.direction: np.ndarray = np.zeros(2, dtype=np.float64)
        self._frozen: tuple[np.ndarray, np.ndarray] | None = None
        self._missed: int = 0

    @property
    def box(self) -> Box:
        return self.kf.box

    def predict(self, *, is_lost: bool) -> None:
        if is_lost:
            self.kf.mean[4:6] *= self.vel_damp
            self.kf.mean[6:8] = 0.0
        self.kf.predict()

    def apply_camera_warp(self, warp_2x3: np.ndarray) -> None:
        rot = warp_2x3[:2, :2]
        trans = warp_2x3[:2, 2]
        scale = float(np.sqrt(abs(np.linalg.det(rot))))
        if scale <= 1e-6:
            return
        self.kf.apply_camera_warp(warp_2x3)
        if self._frozen is not None:
            t_mat = np.zeros((8, 8), dtype=np.float64)
            t_mat[0:2, 0:2] = rot
            t_mat[2, 2] = t_mat[3, 3] = scale
            t_mat[4:6, 4:6] = rot
            t_mat[6, 6] = t_mat[7, 7] = scale
            f_mean, f_cov = self._frozen
            f_mean = t_mat @ f_mean
            f_mean[:2] += trans
            self._frozen = (f_mean, t_mat @ f_cov @ t_mat.T)
        if float(np.hypot(self.direction[0], self.direction[1])) > 1e-6:
            v_xy = rot @ np.array([self.direction[1], self.direction[0]], dtype=np.float64)
            v_norm = float(np.hypot(v_xy[0], v_xy[1]))
            if v_norm > 1e-6:
                self.direction = np.array([v_xy[1] / v_norm, v_xy[0] / v_norm], dtype=np.float64)
        self.last_observation = _warp_box(self.last_observation, rot, trans, scale)
        self.observations = {
            age: _warp_box(b, rot, trans, scale) for age, b in self.observations.items()
        }

    def reference_observation(self, age: int) -> Box:
        for dt in range(self.delta_t, 0, -1):
            obs = self.observations.get(age - dt)
            if obs is not None:
                return obs
        return self.last_observation

    def miss(self) -> None:
        if self._frozen is None:
            self._frozen = (self.kf.mean.copy(), self.kf.cov.copy())
        self._missed += 1

    def update(self, box: Box, age: int, confidence: float) -> None:
        self.direction = _direction(self.reference_observation(age), box)
        if self._frozen is None:
            self.kf.update(box, confidence=confidence)
        else:
            f_mean, f_cov = self._frozen
            self.kf.mean = f_mean
            self.kf.cov = f_cov
            gap = self._missed + 1
            self._frozen = None
            self._missed = 0
            x0a, y0a, x0b, y0b = self.last_observation
            x1a, y1a, x1b, y1b = box
            for step in range(1, gap + 1):
                f = step / gap
                interp = (
                    x0a + f * (x1a - x0a),
                    y0a + f * (y1a - y0a),
                    x0b + f * (x1b - x0b),
                    y0b + f * (y1b - y0b),
                )
                step_conf = confidence if step == gap else min(confidence, 0.85)
                self.kf.update(interp, confidence=step_conf)
                if step < gap:
                    self.kf.predict()
        self.last_observation = box
        self.observations[age] = box
        oldest = age + 1 - self.delta_t
        self.observations = {a: b for a, b in self.observations.items() if a >= oldest}


@dataclass(eq=False)
class _NlTrack:
    uid: int
    motion: _NlMotion
    confidence: float
    class_id: int
    class_name: str
    start_frame: int
    peak_height: float
    track_id: int = 0
    is_activated: bool = False
    age: int = 1
    hits: int = 1
    high_hits: int = 1
    time_since_update: int = 0
    state: str = "tracked"  # tracked | lost
    recovered_now: bool = False
    bank_has_clean: bool = False
    emb_bank: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=15), repr=False)
    emb_ema: np.ndarray | None = field(default=None, repr=False)

    @property
    def bbox(self) -> Box:
        return self.motion.box

    def update_embedding(
        self, emb: np.ndarray | None, *, ovl: float, clean_ovl_thresh: float
    ) -> None:
        if emb is None:
            return
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-6:
            return
        unit = np.asarray(emb, dtype=np.float64) / norm
        if ovl <= clean_ovl_thresh:
            if not self.bank_has_clean:
                self.emb_bank.clear()
                self.bank_has_clean = True
                self.emb_ema = unit
            else:
                blended = 0.90 * self.emb_ema + 0.10 * unit if self.emb_ema is not None else unit
                self.emb_ema = blended / max(float(np.linalg.norm(blended)), 1e-6)
            self.emb_bank.append(unit)
        elif not self.emb_bank:
            self.emb_bank.append(unit)
            self.emb_ema = unit

    def to_track(self, *, coasting: bool = False) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
            age=self.age,
            hits=self.hits,
            recovered=self.recovered_now and not coasting,
            coasting=coasting,
        )


class NonLinearTracker:
    """Dual-anchor non-linear tracker with clean-bank adaptive appearance and lost-track recovery."""

    name: str = "nltrack"

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.55,
        track_buffer: int = 60,
        iou_buffer: float = 0.0,
        use_cmc: bool = True,
        vel_damp: float = 0.88,
        delta_t: int = 3,
        inertia: float = 0.15,
        bank_size: int = 15,
        clean_ovl_thresh: float = 0.40,
        emb_weight: float = 3.5,
        softmax_temp: float = 0.13,
        null_std: float = 1.0,
        margin_weight: float = 0.25,
        ema_blend: float = 0.50,
        occ_exp_max: float = 0.20,
        occ_exp_rate: float = 0.03,
        active_iou_gate: float = 0.20,
        lost_iou_gate: float = 0.18,
        reid_active_iou_gate: float = 0.05,
        reid_rescue_z: float = 1.3,
        reid_rescue_margin: float = 0.25,
        reid_rescue_max_disp: float = 8.0,
        active_low_iou: float = 0.50,
        lost_low_max_tsu: int = 5,
        lost_low_iou: float = 0.55,
        lost_low_conf: float = 0.26,
        stage2_min_rel_height: float = 0.52,
        stage2_high_frac: float = 0.92,
        tai_iou_thresh: float = 0.55,
        spawn_overlap_thresh: float = 0.80,
        duplicate_iou: float = 0.85,
        min_confirm_hits: int = 4,
    ) -> None:
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.track_buffer = max(int(track_buffer), 60)
        self.iou_buffer = float(iou_buffer)
        self.use_cmc = bool(use_cmc)
        self.vel_damp = float(vel_damp)
        self.delta_t = int(delta_t)
        self.inertia = float(inertia)
        self.bank_size = int(bank_size)
        self.clean_ovl_thresh = float(clean_ovl_thresh)
        self.emb_weight = float(emb_weight)
        self.softmax_temp = float(softmax_temp)
        self.null_std = float(null_std)
        self.margin_weight = float(margin_weight)
        self.ema_blend = float(ema_blend)
        self.occ_exp_max = float(occ_exp_max)
        self.occ_exp_rate = float(occ_exp_rate)
        self.active_iou_gate = float(active_iou_gate)
        self.lost_iou_gate = float(lost_iou_gate)
        self.reid_active_iou_gate = float(reid_active_iou_gate)
        self.reid_rescue_z = float(reid_rescue_z)
        self.reid_rescue_margin = float(reid_rescue_margin)
        self.reid_rescue_max_disp = float(reid_rescue_max_disp)
        self.active_low_iou = float(active_low_iou)
        self.lost_low_max_tsu = int(lost_low_max_tsu)
        self.lost_low_iou = float(lost_low_iou)
        self.lost_low_conf = float(lost_low_conf)
        self.stage2_min_rel_height = float(stage2_min_rel_height)
        self.stage2_high_frac = float(stage2_high_frac)
        self.tai_iou_thresh = float(tai_iou_thresh)
        self.spawn_overlap_thresh = float(spawn_overlap_thresh)
        self.duplicate_iou = float(duplicate_iou)
        self.min_confirm_hits = int(min_confirm_hits)
        self.embedding_min_confidence: float | None = self.track_high_thresh
        self.reset()

    def reset(self) -> None:
        self._next_id = 1
        self._next_uid = 1
        self._frame_id = 0
        self._tracks: list[_NlTrack] = []
        self.last_events: list[dict[str, Any]] = []
        self._cmc = SparseOpticalFlowCMC(drop_foreground_flow=True) if self.use_cmc else None
        self._bg_count: float = 10.0
        self._bg_mean: float = 0.55
        self._bg_m2: float = 10.0 * (0.12**2)

    def _update_bg_stats(
        self,
        d_emb: np.ndarray,
        d_has: np.ndarray,
        d_cls: np.ndarray,
        det_iou: np.ndarray,
    ) -> None:
        """Update running mean/variance of cosines between distinct detections in this frame."""
        m = len(d_has)
        if m < 2 or int(d_has.sum()) < 2:
            return
        cos_mat = d_emb @ d_emb.T
        tri = np.triu(np.ones((m, m), dtype=bool), k=1)
        valid = tri & np.outer(d_has, d_has) & (d_cls[:, None] == d_cls[None, :]) & (det_iou < 0.30)
        vals = cos_mat[valid]
        if vals.size == 0:
            return
        if self._bg_count > 4000.0:
            scale = 4000.0 / self._bg_count
            self._bg_count *= scale
            self._bg_m2 *= scale
        batch_n = float(vals.size)
        batch_mean = float(vals.mean())
        batch_m2 = float(((vals - batch_mean) ** 2).sum())
        delta = batch_mean - self._bg_mean
        new_count = self._bg_count + batch_n
        self._bg_mean = self._bg_mean + delta * (batch_n / new_count)
        self._bg_m2 = self._bg_m2 + batch_m2 + (delta**2) * (self._bg_count * batch_n / new_count)
        self._bg_count = new_count

    def _bg_mean_std(self) -> tuple[float, float]:
        var = max(self._bg_m2 / max(self._bg_count, 1.0), 0.06**2)
        return self._bg_mean, float(np.sqrt(var))

    @staticmethod
    def _unit_det_embeddings(dets: list[Detection]) -> tuple[np.ndarray, np.ndarray, int]:
        dim = 0
        for d in dets:
            if d.embedding is not None:
                dim = int(d.embedding.shape[0])
                break
        if dim == 0:
            return np.zeros((len(dets), 0), dtype=np.float64), np.zeros(len(dets), dtype=bool), 0
        out = np.zeros((len(dets), dim), dtype=np.float64)
        has = np.zeros(len(dets), dtype=bool)
        for j, d in enumerate(dets):
            if d.embedding is None or d.embedding.shape[0] != dim:
                continue
            n = float(np.linalg.norm(d.embedding))
            if n > 1e-6:
                out[j] = np.asarray(d.embedding, dtype=np.float64) / n
                has[j] = True
        return out, has, dim

    @staticmethod
    def _det_overlaps(dets: list[Detection], d_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = len(dets)
        if m == 0:
            return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
        det_iou = compute_iou_matrix(d_boxes, d_boxes)
        if m == 1:
            return det_iou, np.zeros(1, dtype=np.float64)
        d_cls = np.array([d.class_id for d in dets], dtype=np.int64)
        same_other = (d_cls[:, None] == d_cls[None, :]) & ~np.eye(m, dtype=bool)
        ovl = np.where(same_other, det_iou, 0.0).max(axis=1)
        return det_iou, ovl

    def _appearance_scores(
        self,
        tracks: list[_NlTrack],
        d_emb: np.ndarray,
        d_has: np.ndarray,
        dim: int,
        same_class: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(app_sim, p_dual, z_ghost, margin, has_pair)`` matrices of shape ``[N, M]``."""
        n, m = same_class.shape
        zeros = np.zeros((n, m), dtype=np.float64)
        no_pair = np.zeros((n, m), dtype=bool)
        if dim == 0 or n == 0 or m == 0 or not bool(d_has.any()):
            return zeros, zeros, zeros, zeros, no_pair

        bank_rows: list[np.ndarray] = []
        offsets: list[int] = []
        t_idx_with_emb: list[int] = []
        ema_rows: list[np.ndarray] = []
        t_has = np.zeros(n, dtype=bool)
        cursor = 0
        for i, tr in enumerate(tracks):
            if not tr.emb_bank:
                continue
            vecs = [v for v in tr.emb_bank if v.shape[0] == dim]
            if not vecs:
                continue
            t_has[i] = True
            t_idx_with_emb.append(i)
            offsets.append(cursor)
            bank_rows.extend(vecs)
            ema_rows.append(tr.emb_ema if tr.emb_ema is not None else vecs[-1])
            cursor += len(vecs)

        if not t_idx_with_emb:
            return zeros, zeros, zeros, zeros, no_pair

        all_bank = np.vstack(bank_rows)
        raw_sims = all_bank @ d_emb.T
        max_per_t = np.maximum.reduceat(raw_sims, offsets, axis=0)
        if self.ema_blend > 0.0:
            ema_sims = np.vstack(ema_rows) @ d_emb.T
            max_per_t = (1.0 - self.ema_blend) * max_per_t + self.ema_blend * ema_sims

        max_cos = np.full((n, m), -1.0, dtype=np.float64)
        max_cos[t_idx_with_emb] = max_per_t
        has_pair = t_has[:, None] & d_has[None, :] & same_class
        if not bool(has_pair.any()):
            return zeros, zeros, zeros, zeros, no_pair

        bg_mean, bg_std = self._bg_mean_std()
        c_null = bg_mean + self.null_std * bg_std
        inv_t = 1.0 / max(self.softmax_temp, 1e-3)

        logits = np.where(has_pair, max_cos * inv_t, -1e9)
        null_logit = c_null * inv_t

        row_ext = np.column_stack([logits, np.full((n, 1), null_logit, dtype=np.float64)])
        row_max = row_ext.max(axis=1, keepdims=True)
        row_exp = np.exp(row_ext - row_max)
        p_row = (row_exp[:, :m] / np.maximum(row_exp.sum(axis=1, keepdims=True), 1e-12)) * has_pair

        col_ext = np.vstack([logits, np.full((1, m), null_logit, dtype=np.float64)])
        col_max = col_ext.max(axis=0, keepdims=True)
        col_exp = np.exp(col_ext - col_max)
        p_col = (col_exp[:n, :] / np.maximum(col_exp.sum(axis=0, keepdims=True), 1e-12)) * has_pair

        p_dual = np.sqrt(np.maximum(p_row * p_col, 0.0))

        # Row/col top-1 vs second-best margin (invariant to pool size N).
        c_masked = np.where(has_pair, max_cos, -1e9)
        c_row_ext = np.column_stack([c_masked, np.full((n, 1), c_null, dtype=np.float64)])
        row_part = np.partition(c_row_ext, -2, axis=1)
        row_top1 = row_part[:, -1:]
        row_top2 = row_part[:, -2:-1]
        best_other_row = np.where(c_masked >= row_top1 - 1e-12, row_top2, row_top1)
        m_row = np.maximum(0.0, c_masked - best_other_row)

        c_col_ext = np.vstack([c_masked, np.full((1, m), c_null, dtype=np.float64)])
        col_part = np.partition(c_col_ext, -2, axis=0)
        col_top1 = col_part[-1:, :]
        col_top2 = col_part[-2:-1, :]
        best_other_col = np.where(c_masked >= col_top1 - 1e-12, col_top2, col_top1)
        m_col = np.maximum(0.0, c_masked - best_other_col)

        margin = np.clip(0.5 * (m_row + m_col) / bg_std, 0.0, 1.5)

        z_ghost = np.where(has_pair, (max_cos - bg_mean) / bg_std, 0.0)
        ghost_norm = np.clip((max_cos - bg_mean) / max(1.0 - bg_mean, 0.12), -0.5, 1.0)
        app_sim = np.where(
            has_pair,
            0.55 * p_dual
            + 0.30 * np.maximum(ghost_norm, 0.0)
            + self.margin_weight * margin
            + 0.25 * np.minimum(ghost_norm, 0.0),
            0.0,
        )
        return app_sim, p_dual, z_ghost, margin, has_pair

    def _momentum_bonus(
        self, tracks: list[_NlTrack], d_boxes: np.ndarray, d_scores: np.ndarray
    ) -> np.ndarray:
        n, m = len(tracks), len(d_boxes)
        if n == 0 or m == 0 or self.inertia <= 0.0:
            return np.zeros((n, m), dtype=np.float64)
        refs = np.array([t.motion.reference_observation(t.age) for t in tracks], dtype=np.float64)
        delta = _centers(d_boxes)[None, :, :] - _centers(refs)[:, None, :]
        norm = np.hypot(delta[..., 0], delta[..., 1]) + 1e-6
        dx, dy = delta[..., 0] / norm, delta[..., 1] / norm
        vel = np.array([t.motion.direction for t in tracks], dtype=np.float64)
        has_vel = (np.hypot(vel[:, 0], vel[:, 1]) > 0.5) & (
            np.array([t.time_since_update for t in tracks]) <= 3
        )
        cos = np.clip(vel[:, 0:1] * dy + vel[:, 1:2] * dx, -1.0, 1.0)
        agreement = (np.pi * 0.5 - np.abs(np.arccos(cos))) / np.pi
        bonus = agreement * self.inertia * d_scores[None, :]
        bonus[~has_vel] = 0.0
        return bonus

    def _associate_stage1(
        self,
        pool: list[_NlTrack],
        dets: list[Detection],
        d_boxes: np.ndarray,
        d_emb: np.ndarray,
        d_has: np.ndarray,
        dim: int,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not pool or not dets:
            return [], list(range(len(pool))), list(range(len(dets)))

        n = len(pool)
        same_class = np.equal.outer([t.class_id for t in pool], [d.class_id for d in dets])
        pred_boxes = np.array([t.motion.box for t in pool], dtype=np.float64)
        last_boxes = np.array([t.motion.last_observation for t in pool], dtype=np.float64)
        tsu = np.array([t.time_since_update for t in pool], dtype=np.int64)
        is_lost = tsu >= 2

        base_exp = np.full(n, self.iou_buffer, dtype=np.float64)
        occ_exp = np.where(
            is_lost,
            self.iou_buffer + np.minimum(self.occ_exp_max, self.occ_exp_rate * tsu),
            self.iou_buffer,
        )

        pred_iou, pred_hmiou = _expanded_iou_hmiou(pred_boxes, d_boxes, base_exp)
        spatial_hmiou = pred_hmiou.copy()
        gate_iou = pred_iou.copy()

        if bool(is_lost.any()):
            li = np.flatnonzero(is_lost)
            l_last_raw_iou, l_last_raw_hm = _expanded_iou_hmiou(
                last_boxes[li], d_boxes, base_exp[li]
            )
            raw_hm = np.maximum(pred_hmiou[li], l_last_raw_hm)
            raw_iou = np.maximum(pred_iou[li], l_last_raw_iou)
            if self.occ_exp_max > 0.0:
                l_pred_iou, l_pred_hm = _expanded_iou_hmiou(pred_boxes[li], d_boxes, occ_exp[li])
                l_last_exp_iou, l_last_exp_hm = _expanded_iou_hmiou(
                    last_boxes[li], d_boxes, occ_exp[li]
                )
                exp_hm = np.maximum(l_pred_hm, l_last_exp_hm)
                spatial_hmiou[li] = np.maximum(raw_hm, 0.85 * exp_hm)
                gate_iou[li] = np.maximum.reduce([raw_iou, l_pred_iou, l_last_exp_iou])
            else:
                spatial_hmiou[li] = raw_hm
                gate_iou[li] = raw_iou

        d_scores = np.array([d.confidence for d in dets], dtype=np.float64)
        ocm = self._momentum_bonus(pool, d_boxes, d_scores)
        app_sim, p_dual, z_ghost, margin, has_pair = self._appearance_scores(
            pool, d_emb, d_has, dim, same_class
        )

        iou_thresh = np.where(is_lost, self.lost_iou_gate, self.active_iou_gate)[:, None]
        valid = same_class & (gate_iou >= iou_thresh)

        if bool(has_pair.any()):
            t_h = np.maximum(last_boxes[:, 3] - last_boxes[:, 1], 1.0)[:, None]
            t_w = np.maximum(last_boxes[:, 2] - last_boxes[:, 0], 1.0)[:, None]
            d_h = np.maximum(d_boxes[:, 3] - d_boxes[:, 1], 1.0)[None, :]
            h_ratio = d_h / t_h
            h_ok = (h_ratio >= 0.55) & (h_ratio <= 1.80)

            d_c = _centers(d_boxes)[None, :, :]
            dist_pred = np.hypot(
                d_c[..., 0] - _centers(pred_boxes)[:, None, 0],
                d_c[..., 1] - _centers(pred_boxes)[:, None, 1],
            )
            dist_last = np.hypot(
                d_c[..., 0] - _centers(last_boxes)[:, None, 0],
                d_c[..., 1] - _centers(last_boxes)[:, None, 1],
            )
            min_dist = np.minimum(dist_pred, dist_last)
            t_scale = np.maximum(t_w, t_h)
            max_disp = (
                np.minimum(self.reid_rescue_max_disp, np.maximum(1.5, 0.35 * tsu))[:, None]
                * t_scale
            )
            reid_strong = (z_ghost >= self.reid_rescue_z) & (
                (margin >= self.reid_rescue_margin) | (p_dual >= 0.65)
            )
            reid_lost = is_lost[:, None] & has_pair & h_ok & (min_dist <= max_disp) & reid_strong
            reid_active = (
                (~is_lost[:, None])
                & has_pair
                & (gate_iou >= self.reid_active_iou_gate)
                & h_ok
                & reid_strong
            )
            valid |= reid_lost | reid_active

        sim = spatial_hmiou + ocm + self.emb_weight * app_sim
        sim = np.where(valid, np.maximum(sim, 1e-4), 0.0)
        return _mutual_best_assign(sim, valid)

    def _activate(self, tr: _NlTrack, events: list[dict[str, Any]]) -> None:
        tr.is_activated = True
        tr.track_id = self._next_id
        self._next_id += 1
        events.append({"type": "track_started", "track_id": tr.track_id})
        switch = _switch_event(
            tr.track_id,
            tr.bbox,
            [
                (t.track_id, t.bbox)
                for t in self._tracks
                if t.state == "lost"
                and t.class_id == tr.class_id
                and t.is_activated
                and t.time_since_update <= 5
            ],
        )
        if switch is not None:
            events.append(switch)

    def _apply_detection(
        self,
        tr: _NlTrack,
        det: Detection,
        *,
        update_class: bool,
        ovl: float | None,
    ) -> None:
        tr.motion.update(det.bbox, tr.age, det.confidence)
        tr.confidence = det.confidence
        if update_class:
            tr.class_id = det.class_id
            tr.class_name = det.class_name
            tr.high_hits += 1
            tr.peak_height = max(tr.peak_height, float(det.bbox[3] - det.bbox[1]))
        tr.hits += 1
        tr.time_since_update = 0
        if ovl is not None:
            tr.update_embedding(det.embedding, ovl=ovl, clean_ovl_thresh=self.clean_ovl_thresh)

    def _stage2_eligible(self, tr: _NlTrack, med_h: float) -> bool:
        """Keep persistent low-confidence tiny background tracks out of Stage-2 maintenance."""
        if self.stage2_min_rel_height <= 0.0 or med_h <= 0.0:
            return True
        if tr.peak_height >= self.stage2_min_rel_height * med_h:
            return True
        return (tr.high_hits / max(tr.hits, 1)) >= self.stage2_high_frac

    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[Track]:
        """Advance the tracker by one frame and return confirmed tracks observed this frame."""
        self._frame_id += 1
        events: list[dict[str, Any]] = []

        for tr in self._tracks:
            tr.recovered_now = False
            tr.motion.predict(is_lost=(tr.state != "tracked"))
            tr.age += 1
            tr.time_since_update += 1

        if self._cmc is not None:
            if frame_bgr is None:
                self._cmc = SparseOpticalFlowCMC(drop_foreground_flow=True)
            else:
                warp = self._cmc.estimate(frame_bgr, _boxes(detections))
                if warp is not None:
                    scale = float(np.sqrt(abs(np.linalg.det(warp[:2, :2]))))
                    for tr in self._tracks:
                        tr.motion.apply_camera_warp(warp)
                        if scale > 1e-6:
                            tr.peak_height *= scale

        high_dets = [d for d in detections if d.confidence >= self.track_high_thresh]
        low_dets = [
            d for d in detections if self.track_low_thresh <= d.confidence < self.track_high_thresh
        ]

        high_boxes = _boxes(high_dets)
        det_iou, high_ovl = self._det_overlaps(high_dets, high_boxes)
        d_emb, d_has, dim = self._unit_det_embeddings(high_dets)
        if dim > 0:
            d_cls = np.array([d.class_id for d in high_dets], dtype=np.int64)
            self._update_bg_stats(d_emb, d_has, d_cls, det_iou)

        confirmed = [t for t in self._tracks if t.is_activated and t.state == "tracked"]
        lost = [t for t in self._tracks if t.is_activated and t.state == "lost"]
        unconfirmed = [t for t in self._tracks if not t.is_activated]
        pool = confirmed + lost
        updated: set[int] = set()

        # Stage 1: confirmed + lost tracks <-> high-score detections.
        matches, u_pool, u_high = self._associate_stage1(
            pool, high_dets, high_boxes, d_emb, d_has, dim
        )
        for ti, di in matches:
            tr = pool[ti]
            if tr.state == "lost":
                tr.recovered_now = True
                events.append({"type": "track_recovered", "track_id": tr.track_id})
            tr.state = "tracked"
            self._apply_detection(tr, high_dets[di], update_class=True, ovl=float(high_ovl[di]))
            updated.add(tr.uid)

        med_h = (
            float(np.median([t.bbox[3] - t.bbox[1] for t in confirmed]))
            if len(confirmed) >= 3
            else 0.0
        )

        # Stage 2a: remaining active tracks <-> low-score detections.
        rem_tracked = [
            pool[i]
            for i in u_pool
            if pool[i].state == "tracked" and self._stage2_eligible(pool[i], med_h)
        ]
        used_low: set[int] = set()
        if rem_tracked and low_dets:
            t_boxes = np.array([t.motion.box for t in rem_tracked], dtype=np.float64)
            l_boxes = _boxes(low_dets)
            iou = compute_iou_matrix(t_boxes, l_boxes)
            hmiou = compute_hmiou_matrix(t_boxes, l_boxes, iou)
            same_cls = np.equal.outer(
                [t.class_id for t in rem_tracked], [d.class_id for d in low_dets]
            )
            low_matches, _, _ = _assign(hmiou, same_cls & (iou >= self.active_low_iou))
            for ti, di in low_matches:
                tr = rem_tracked[ti]
                self._apply_detection(tr, low_dets[di], update_class=False, ovl=None)
                updated.add(tr.uid)
                used_low.add(di)

        # Stage 2b: guarded low-score recovery for recently lost tracks.
        if self.lost_low_max_tsu > 0:
            rem_lost_rec = [
                pool[i]
                for i in u_pool
                if pool[i].state == "lost"
                and 2 <= pool[i].time_since_update <= self.lost_low_max_tsu
                and self._stage2_eligible(pool[i], med_h)
            ]
            rem_low_idx = [
                i
                for i in range(len(low_dets))
                if i not in used_low and low_dets[i].confidence >= self.lost_low_conf
            ]
            if rem_lost_rec and rem_low_idx:
                cand_low = [low_dets[i] for i in rem_low_idx]
                l_boxes = _boxes(cand_low)

                upd_by_cls: dict[int, list[Box]] = defaultdict(list)
                for t in self._tracks:
                    if t.uid in updated:
                        upd_by_cls[t.class_id].append(t.bbox)
                low_ok = np.ones(len(cand_low), dtype=bool)
                for j, d in enumerate(cand_low):
                    same_upd = upd_by_cls.get(d.class_id)
                    if same_upd:
                        upd_arr = np.array(same_upd, dtype=np.float64)
                        d_arr = l_boxes[j : j + 1]
                        if (
                            float(compute_ioa_matrix(d_arr, upd_arr).max()) >= 0.70
                            or float(compute_iou_matrix(d_arr, upd_arr).max()) >= 0.55
                        ):
                            low_ok[j] = False

                if bool(low_ok.any()):
                    pred_b = np.array([t.motion.box for t in rem_lost_rec], dtype=np.float64)
                    last_b = np.array(
                        [t.motion.last_observation for t in rem_lost_rec], dtype=np.float64
                    )
                    iou_p = compute_iou_matrix(pred_b, l_boxes)
                    iou_l = compute_iou_matrix(last_b, l_boxes)
                    hm_p = compute_hmiou_matrix(pred_b, l_boxes, iou_p)
                    hm_l = compute_hmiou_matrix(last_b, l_boxes, iou_l)
                    iou_dual = np.maximum(iou_p, iou_l)
                    hm_dual = np.maximum(hm_p, hm_l)

                    t_h = np.maximum(last_b[:, 3] - last_b[:, 1], 1.0)[:, None]
                    d_h = np.maximum(l_boxes[:, 3] - l_boxes[:, 1], 1.0)[None, :]
                    h_ok = (d_h / t_h >= 0.65) & (d_h / t_h <= 1.55)
                    same_cls = np.equal.outer(
                        [t.class_id for t in rem_lost_rec], [d.class_id for d in cand_low]
                    )
                    valid_rec = same_cls & low_ok[None, :] & h_ok & (iou_dual >= self.lost_low_iou)
                    rec_matches, _, _ = _assign(hm_dual, valid_rec)
                    for ti, di in rec_matches:
                        tr = rem_lost_rec[ti]
                        tr.recovered_now = True
                        tr.state = "tracked"
                        events.append({"type": "track_recovered", "track_id": tr.track_id})
                        self._apply_detection(tr, cand_low[di], update_class=False, ovl=None)
                        updated.add(tr.uid)

        # Stage 3: unconfirmed tracks <-> leftover high-score detections.
        rem_high_idx = list(u_high)
        if unconfirmed and rem_high_idx:
            rem_dets = [high_dets[i] for i in rem_high_idx]
            rem_boxes = high_boxes[rem_high_idx]
            u_boxes = np.array([t.motion.box for t in unconfirmed], dtype=np.float64)
            u_iou = compute_iou_matrix(
                _expand_boxes(u_boxes, self.iou_buffer),
                _expand_boxes(rem_boxes, self.iou_buffer),
            )
            u_hmiou = compute_hmiou_matrix(u_boxes, rem_boxes, u_iou)
            same_cls = np.equal.outer(
                [t.class_id for t in unconfirmed], [d.class_id for d in rem_dets]
            )
            un_matches, _, u_rem_local = _assign(u_hmiou, same_cls & (u_iou >= 0.30))
            for ti, di in un_matches:
                tr = unconfirmed[ti]
                orig_idx = rem_high_idx[di]
                self._apply_detection(
                    tr, high_dets[orig_idx], update_class=True, ovl=float(high_ovl[orig_idx])
                )
                updated.add(tr.uid)
                if tr.hits >= self.min_confirm_hits:
                    self._activate(tr, events)
            rem_high_idx = [rem_high_idx[i] for i in u_rem_local]

        # Lost transitions, ORU miss snapshots, expiry, and drop unmatched unconfirmed.
        surviving: list[_NlTrack] = []
        for tr in self._tracks:
            if tr.uid in updated:
                surviving.append(tr)
                continue
            if not tr.is_activated:
                continue
            if tr.state == "tracked":
                tr.state = "lost"
                events.append({"type": "track_lost", "track_id": tr.track_id})
            tr.motion.miss()
            if tr.time_since_update <= self.track_buffer:
                surviving.append(tr)
            else:
                events.append({"type": "track_ended", "track_id": tr.track_id})
        self._tracks = surviving

        # Spawn new tracks with TAI duplicate-spawn blocking.
        occupied: dict[int, list[Box]] = defaultdict(list)
        for t in self._tracks:
            if t.uid in updated:
                occupied[t.class_id].append(t.bbox)
        higher_leftovers: dict[int, list[Box]] = defaultdict(list)

        for idx in sorted(rem_high_idx, key=lambda i: high_dets[i].confidence, reverse=True):
            det = high_dets[idx]
            if det.confidence < self.new_track_thresh:
                higher_leftovers[det.class_id].append(det.bbox)
                continue
            d_arr = high_boxes[idx : idx + 1]
            same_occ = occupied[det.class_id]
            if same_occ:
                occ_arr = np.array(same_occ, dtype=np.float64)
                if (
                    float(compute_iou_matrix(d_arr, occ_arr).max()) >= self.tai_iou_thresh
                    or float(compute_ioa_matrix(d_arr, occ_arr).max()) >= self.spawn_overlap_thresh
                ):
                    higher_leftovers[det.class_id].append(det.bbox)
                    continue
            same_left = higher_leftovers[det.class_id]
            if same_left:
                left_arr = np.array(same_left, dtype=np.float64)
                if float(compute_iou_matrix(d_arr, left_arr).max()) >= self.tai_iou_thresh:
                    higher_leftovers[det.class_id].append(det.bbox)
                    continue

            tr = _NlTrack(
                uid=self._next_uid,
                motion=_NlMotion(det.bbox, delta_t=self.delta_t, vel_damp=self.vel_damp),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                start_frame=self._frame_id,
                peak_height=float(det.bbox[3] - det.bbox[1]),
                emb_bank=deque(maxlen=self.bank_size),
            )
            self._next_uid += 1
            tr.update_embedding(
                det.embedding, ovl=float(high_ovl[idx]), clean_ovl_thresh=self.clean_ovl_thresh
            )
            self._tracks.append(tr)
            updated.add(tr.uid)
            same_occ.append(tr.bbox)
            higher_leftovers[det.class_id].append(det.bbox)
            if self._frame_id == 1:
                self._activate(tr, events)

        self._remove_duplicates(updated)

        self.last_events = events
        outputs = [t.to_track() for t in self._tracks if t.is_activated and t.uid in updated]
        outputs.sort(key=lambda t: t.track_id)
        return outputs

    def _remove_duplicates(self, updated: set[int]) -> None:
        tracked = [t for t in self._tracks if t.state == "tracked" and t.uid in updated]
        lost = [t for t in self._tracks if t.state == "lost"]
        if not tracked or not lost:
            return
        iou = compute_iou_matrix(_boxes(tracked), _boxes(lost))
        drop: set[int] = set()
        for p, q in zip(*np.where(iou > self.duplicate_iou), strict=False):
            a, b = tracked[int(p)], lost[int(q)]
            if a.class_id != b.class_id:
                continue
            if a.is_activated or (self._frame_id - a.start_frame) >= (
                self._frame_id - b.start_frame
            ):
                drop.add(b.uid)
            else:
                drop.add(a.uid)
        if drop:
            self._tracks = [t for t in self._tracks if t.uid not in drop]

    def coasting_tracks(self, max_frames: int = 3) -> list[Track]:
        """Confirmed tracks missed for at most ``max_frames`` frames, at their predicted position."""
        out = [
            t.to_track(coasting=True)
            for t in self._tracks
            if t.is_activated and t.state == "lost" and t.time_since_update <= max_frames
        ]
        out.sort(key=lambda t: t.track_id)
        return out
