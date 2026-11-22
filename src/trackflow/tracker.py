"""Pure NumPy/SciPy ByteTrack and BoT-SORT multi-object trackers with Kalman filtering & CMC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from trackflow.types import Detection, Track


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two [N, 4] and [M, 4] xyxy arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = np.split(boxes_a, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(boxes_b, 4, axis=1)

    inter_x1 = np.maximum(ax1, bx1.T)
    inter_y1 = np.maximum(ay1, by1.T)
    inter_x2 = np.minimum(ax2, bx2.T)
    inter_y2 = np.minimum(ay2, by2.T)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = np.maximum(area_a + area_b.T - inter_area, 1e-6)
    return (inter_area / union).astype(np.float32)


def _xyxy_to_xyah(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    a = w / h
    return np.array([cx, cy, a, h], dtype=np.float64)


def _xyah_to_xyxy(xyah: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, a, h = (
        float(xyah[0]),
        float(xyah[1]),
        max(0.05, float(xyah[2])),
        max(1.0, float(xyah[3])),
    )
    w = a * h
    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


class KalmanBoxFilter:
    """8-state constant-velocity Kalman filter for bounding boxes: [cx, cy, a, h, vx, vy, va, vh]."""

    def __init__(self, initial_bbox: tuple[float, float, float, float]) -> None:
        self._F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        self._H = np.eye(4, 8, dtype=np.float64)

        self._std_pos = 1.0 / 20.0
        self._std_vel = 1.0 / 160.0

        meas = _xyxy_to_xyah(initial_bbox)
        self.mean = np.zeros(8, dtype=np.float64)
        self.mean[:4] = meas

        h = max(10.0, meas[3])
        std = np.array(
            [
                2 * self._std_pos * h,
                2 * self._std_pos * h,
                1e-2,
                2 * self._std_pos * h,
                10 * self._std_vel * h,
                10 * self._std_vel * h,
                1e-5,
                10 * self._std_vel * h,
            ],
            dtype=np.float64,
        )
        self.cov = np.diag(std**2)

    def predict(self) -> tuple[float, float, float, float]:
        h = max(10.0, float(self.mean[3]))
        std = np.array(
            [
                self._std_pos * h,
                self._std_pos * h,
                1e-2,
                self._std_pos * h,
                self._std_vel * h,
                self._std_vel * h,
                1e-5,
                self._std_vel * h,
            ],
            dtype=np.float64,
        )
        q = np.diag(std**2)
        self.mean = self._F @ self.mean
        if self.mean[3] < 1.0:
            self.mean[3] = 1.0
            self.mean[7] = 0.0
        self.cov = self._F @ self.cov @ self._F.T + q
        return _xyah_to_xyxy(self.mean[:4])

    def update(self, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        meas = _xyxy_to_xyah(bbox)
        h = max(10.0, float(self.mean[3]))
        r = np.diag(
            np.array(
                [self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h], dtype=np.float64
            )
            ** 2
        )
        proj_mean = self._H @ self.mean
        proj_cov = self._H @ self.cov @ self._H.T + r
        kalman_gain = np.linalg.solve(proj_cov, self._H @ self.cov).T
        innovation = meas - proj_mean
        self.mean = self.mean + kalman_gain @ innovation
        self.cov = self.cov - kalman_gain @ proj_cov @ kalman_gain.T
        return _xyah_to_xyxy(self.mean[:4])

    def apply_camera_warp(self, warp_2x3: np.ndarray) -> None:
        """Apply a 2x3 affine camera motion transform to the predicted center & velocity."""
        rot_scale = warp_2x3[:2, :2]
        trans = warp_2x3[:2, 2]
        self.mean[:2] = rot_scale @ self.mean[:2] + trans
        self.mean[4:6] = rot_scale @ self.mean[4:6]


@dataclass
class _TrackState:
    track_id: int
    kf: KalmanBoxFilter
    confidence: float
    class_id: int
    class_name: str
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "tracked"  # tracked | lost
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _xyah_to_xyxy(self.kf.mean[:4])

    def update_embedding(self, emb: np.ndarray | None, alpha: float = 0.85) -> None:
        if emb is None:
            return
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-6:
            return
        unit = emb / norm
        if self.embedding is None:
            self.embedding = unit
        else:
            blended = alpha * self.embedding + (1.0 - alpha) * unit
            bnorm = float(np.linalg.norm(blended))
            self.embedding = blended / max(bnorm, 1e-6)


class SparseOpticalFlowCMC:
    """Fast sparse optical-flow Camera Motion Compensation (BoT-SORT style)."""

    def __init__(self, downscale: int = 2) -> None:
        self.downscale = max(1, downscale)
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None

    def estimate(self, frame_bgr: np.ndarray | None) -> np.ndarray | None:
        if frame_bgr is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (max(32, w // self.downscale), max(32, h // self.downscale)))

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=3,
        )

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < 6:
            self._prev_gray = gray
            self._prev_pts = pts
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_pts, None)
        warp: np.ndarray | None = None
        if curr_pts is not None and status is not None:
            valid = status.ravel() == 1
            p0 = self._prev_pts[valid]
            p1 = curr_pts[valid]
            if len(p0) >= 6:
                mat, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
                if mat is not None:
                    mat = mat.astype(np.float64)
                    mat[:2, 2] *= float(self.downscale)
                    warp = mat

        self._prev_gray = gray
        self._prev_pts = pts
        return warp


def _linear_assignment(
    cost_matrix: np.ndarray,
    max_cost: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()

    for r, c in zip(row_ind.tolist(), col_ind.tolist(), strict=False):
        if cost_matrix[r, c] <= max_cost:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class ByteTracker:
    """Two-stage ByteTrack multi-object tracker with diagnostic event reporting."""

    name: str = "bytetrack"

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.55,
        match_thresh: float = 0.8,
        second_match_thresh: float = 0.5,
        track_buffer: int = 30,
    ) -> None:
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_thresh = float(match_thresh)
        self.second_match_thresh = float(second_match_thresh)
        self.track_buffer = int(track_buffer)

        self._next_id = 1
        self._frame_id = 0
        self._tracks: list[_TrackState] = []
        self.last_events: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._next_id = 1
        self._frame_id = 0
        self._tracks.clear()
        self.last_events = []

    def _compute_first_stage_cost(
        self,
        pool: list[_TrackState],
        dets: list[Detection],
    ) -> np.ndarray:
        t_boxes = np.array([t.bbox for t in pool], dtype=np.float32)
        d_boxes = np.array([d.bbox for d in dets], dtype=np.float32)
        iou = compute_iou_matrix(t_boxes, d_boxes)
        return 1.0 - iou

    def _apply_cmc(self, frame_bgr: np.ndarray | None) -> None:
        """No-op in ByteTrack; overridden by BoT-SORT."""
        _ = frame_bgr

    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[Track]:
        """Advance the tracker by one frame and return active Tracks."""
        self._frame_id += 1
        events: list[dict[str, Any]] = []

        # 1. Predict all existing tracked & lost states
        for t in self._tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        # Optional camera motion compensation (active in BoT-SORT)
        self._apply_cmc(frame_bgr)

        # Partition detections into high-confidence and low-confidence tiers
        high_dets = [d for d in detections if d.confidence >= self.track_high_thresh]
        low_dets = [
            d for d in detections if self.track_low_thresh <= d.confidence < self.track_high_thresh
        ]

        pool = list(self._tracks)
        active_outputs: list[Track] = []
        updated_ids: set[int] = set()

        # 2. First association: all tracks (tracked + lost) with high-confidence detections
        if pool and high_dets:
            cost = self._compute_first_stage_cost(pool, high_dets)
            matches, unmatched_t_idx, unmatched_d_idx = _linear_assignment(
                cost, max_cost=self.match_thresh
            )
        else:
            matches = []
            unmatched_t_idx = list(range(len(pool)))
            unmatched_d_idx = list(range(len(high_dets)))

        for ti, di in matches:
            tr = pool[ti]
            det = high_dets[di]
            was_lost = tr.state == "lost"
            tr.kf.update(det.bbox)
            tr.confidence = det.confidence
            tr.class_id = det.class_id
            tr.class_name = det.class_name
            tr.hits += 1
            tr.time_since_update = 0
            tr.state = "tracked"
            tr.update_embedding(det.embedding)
            updated_ids.add(tr.track_id)
            if was_lost:
                events.append({"type": "track_recovered", "track_id": tr.track_id})
            active_outputs.append(
                Track(
                    track_id=tr.track_id,
                    bbox=tr.bbox,
                    confidence=tr.confidence,
                    class_id=tr.class_id,
                    class_name=tr.class_name,
                    age=tr.age,
                    hits=tr.hits,
                    recovered=was_lost,
                )
            )

        # 3. Second association: remaining *tracked* tracks with low-confidence detections (IoU only)
        rem_tracked = [pool[i] for i in unmatched_t_idx if pool[i].state == "tracked"]
        if rem_tracked and low_dets:
            t_boxes = np.array([t.bbox for t in rem_tracked], dtype=np.float32)
            d_boxes = np.array([d.bbox for d in low_dets], dtype=np.float32)
            cost_low = 1.0 - compute_iou_matrix(t_boxes, d_boxes)
            low_matches, _, _ = _linear_assignment(cost_low, max_cost=self.second_match_thresh)
            for ti, di in low_matches:
                tr = rem_tracked[ti]
                det = low_dets[di]
                tr.kf.update(det.bbox)
                tr.confidence = det.confidence
                tr.hits += 1
                tr.time_since_update = 0
                tr.state = "tracked"
                updated_ids.add(tr.track_id)
                active_outputs.append(
                    Track(
                        track_id=tr.track_id,
                        bbox=tr.bbox,
                        confidence=tr.confidence,
                        class_id=tr.class_id,
                        class_name=tr.class_name,
                        age=tr.age,
                        hits=tr.hits,
                        recovered=False,
                    )
                )

        # 4. Mark unmatched tracks as lost and prune expired tracks
        surviving: list[_TrackState] = []
        recently_lost_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        for tr in self._tracks:
            if tr.track_id in updated_ids:
                surviving.append(tr)
            else:
                if tr.state == "tracked":
                    tr.state = "lost"
                    events.append({"type": "track_lost", "track_id": tr.track_id})
                if tr.time_since_update <= self.track_buffer:
                    surviving.append(tr)
                    if tr.time_since_update <= 5:
                        recently_lost_boxes.append((tr.track_id, tr.bbox))
                else:
                    events.append({"type": "track_ended", "track_id": tr.track_id})

        # 5. Spawn new tracks for unmatched high-confidence detections
        for di in unmatched_d_idx:
            det = high_dets[di]
            if det.confidence < self.new_track_thresh:
                continue
            new_id = self._next_id
            self._next_id += 1
            tr = _TrackState(
                track_id=new_id,
                kf=KalmanBoxFilter(det.bbox),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                embedding=None,
            )
            tr.update_embedding(det.embedding)
            surviving.append(tr)
            events.append({"type": "track_started", "track_id": new_id})

            # Flag possible ID switch if a new track spawns near a recently lost track
            if recently_lost_boxes:
                new_box = np.array([det.bbox], dtype=np.float32)
                lost_arr = np.array([b for _, b in recently_lost_boxes], dtype=np.float32)
                ious = compute_iou_matrix(new_box, lost_arr)[0]
                best_j = int(np.argmax(ious))
                if float(ious[best_j]) >= 0.25:
                    events.append(
                        {
                            "type": "possible_id_switch",
                            "new_track_id": new_id,
                            "previous_track_id": recently_lost_boxes[best_j][0],
                            "iou": round(float(ious[best_j]), 3),
                        }
                    )

            active_outputs.append(
                Track(
                    track_id=new_id,
                    bbox=tr.bbox,
                    confidence=tr.confidence,
                    class_id=tr.class_id,
                    class_name=tr.class_name,
                    age=1,
                    hits=1,
                    recovered=False,
                )
            )

        self._tracks = surviving
        self.last_events = events
        active_outputs.sort(key=lambda t: t.track_id)
        return active_outputs


class BotSortTracker(ByteTracker):
    """BoT-SORT tracker: ByteTrack + Sparse Optical Flow CMC + Appearance Re-ID fusion."""

    name: str = "botsort"

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.55,
        match_thresh: float = 0.8,
        second_match_thresh: float = 0.5,
        track_buffer: int = 45,
        use_cmc: bool = True,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.35,
    ) -> None:
        super().__init__(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=match_thresh,
            second_match_thresh=second_match_thresh,
            track_buffer=track_buffer,
        )
        self.use_cmc = use_cmc
        self.proximity_thresh = float(proximity_thresh)
        self.appearance_thresh = float(appearance_thresh)
        self._cmc = SparseOpticalFlowCMC() if use_cmc else None

    def reset(self) -> None:
        super().reset()
        if self.use_cmc:
            self._cmc = SparseOpticalFlowCMC()

    def _apply_cmc(self, frame_bgr: np.ndarray | None) -> None:
        if self._cmc is None or frame_bgr is None or not self._tracks:
            return
        warp = self._cmc.estimate(frame_bgr)
        if warp is None:
            return
        for tr in self._tracks:
            tr.kf.apply_camera_warp(warp)

    def _compute_first_stage_cost(
        self,
        pool: list[_TrackState],
        dets: list[Detection],
    ) -> np.ndarray:
        iou_dist = super()._compute_first_stage_cost(pool, dets)

        # Fuse cosine appearance distance when both track & detection have embeddings
        has_any_emb = any(t.embedding is not None for t in pool) and any(
            d.embedding is not None for d in dets
        )
        if not has_any_emb:
            return iou_dist

        fused = iou_dist.copy()
        for i, tr in enumerate(pool):
            if tr.embedding is None:
                continue
            for j, det in enumerate(dets):
                if det.embedding is None:
                    continue
                if iou_dist[i, j] > self.proximity_thresh:
                    continue
                d_norm = float(np.linalg.norm(det.embedding))
                if d_norm <= 1e-6:
                    continue
                cos_sim = float(np.dot(tr.embedding, det.embedding / d_norm))
                emb_dist = max(0.0, min(1.0, 0.5 * (1.0 - cos_sim)))
                if emb_dist <= self.appearance_thresh:
                    fused[i, j] = min(float(iou_dist[i, j]), emb_dist)
        return fused


def create_tracker(
    tracker_name: str = "bytetrack",
    *,
    track_high_thresh: float = 0.5,
    track_low_thresh: float = 0.15,
    new_track_thresh: float = 0.55,
    track_buffer: int = 30,
) -> ByteTracker:
    """Factory for built-in multi-object trackers."""
    key = tracker_name.lower().strip().replace("-", "_")
    if key in ("bytetrack", "byte"):
        return ByteTracker(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
        )
    if key in ("botsort", "bot_sort"):
        return BotSortTracker(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=max(track_buffer, 45),
        )
    raise ValueError(f"Unknown tracker '{tracker_name}'. Choose 'bytetrack' or 'botsort'.")
