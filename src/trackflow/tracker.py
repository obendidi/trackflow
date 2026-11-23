"""Pure NumPy/SciPy ByteTrack and BoT-SORT multi-object trackers with Kalman filtering & CMC.

``web/tracker.js`` is a direct port of :class:`ByteTracker` for the browser playground. Keep the
two in sync: ``tests/test_web_parity.py`` replays the same recorded detections through both and
requires identical IDs and boxes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from trackflow.types import Detection, Track


class Tracker(Protocol):
    """A stateful per-source tracker. Return the observations for the current frame."""

    def update(
        self, detections: list[Detection], frame_bgr: np.ndarray | None = None
    ) -> list[Track]: ...


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two [N, 4] and [M, 4] xyxy arrays (float64)."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    ax1, ay1, ax2, ay2 = np.split(boxes_a, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(boxes_b, 4, axis=1)

    inter_w = np.maximum(0.0, np.minimum(ax2, bx2.T) - np.maximum(ax1, bx1.T))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2.T) - np.maximum(ay1, by1.T))
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = np.maximum(area_a + area_b.T - inter_area, 1e-6)
    return inter_area / union


def compute_ioa_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Fraction of each box in ``boxes_a`` covered by each box in ``boxes_b`` (intersection / area_a)."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    ax1, ay1, ax2, ay2 = np.split(boxes_a, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(boxes_b, 4, axis=1)
    inter_w = np.maximum(0.0, np.minimum(ax2, bx2.T) - np.maximum(ax1, bx1.T))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2.T) - np.maximum(ay1, by1.T))
    area_a = np.maximum(np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1), 1e-6)
    return (inter_w * inter_h) / area_a


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
    uid: int  # internal handle, assigned at spawn
    kf: KalmanBoxFilter
    confidence: float
    class_id: int
    class_name: str
    start_frame: int
    track_id: int = 0  # public ID, assigned only once the track is confirmed
    is_activated: bool = False
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "tracked"  # tracked | lost
    recovered_now: bool = False
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _xyah_to_xyxy(self.kf.mean[:4])

    def update_embedding(self, emb: np.ndarray | None, alpha: float = 0.9) -> None:
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

    # Cost-limited linear assignment. Each row can take a dummy at the cutoff; a
    # marginal pairing need not displace a strong one just to fill the matrix.
    transpose = n_rows > n_cols
    cost = cost_matrix.T if transpose else cost_matrix
    dummy = np.full((cost.shape[0], cost.shape[0]), np.nextafter(max_cost, np.inf))
    rows, cols = linear_sum_assignment(np.column_stack((cost, dummy)))
    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()

    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if col >= cost.shape[1]:
            continue
        r, c = (col, row) if transpose else (row, col)
        if cost_matrix[r, c] <= max_cost:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


def _boxes(items: list[Any]) -> np.ndarray:
    return np.array([it.bbox for it in items], dtype=np.float64).reshape(-1, 4)


def _expand_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    """Grow each xyxy box by ``scale`` x its width/height on every side (buffered IoU)."""
    if scale <= 0.0:
        return boxes
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack(
        [
            boxes[:, 0] - scale * w,
            boxes[:, 1] - scale * h,
            boxes[:, 2] + scale * w,
            boxes[:, 3] + scale * h,
        ],
        axis=1,
    )


def iou_buffer_for_interval(dt_s: float) -> float:
    """Buffered-IoU scale for a given time between processed frames.

    Pedestrians move a roughly fixed fraction of their own size per second, so when frames are
    processed far apart (``--skip``, or a slow WASM browser at 2-4 FPS) consecutive boxes of the
    same person stop overlapping and plain IoU fragments IDs. Growing the boxes before computing
    IoU (C-BIoU, Yang et al. WACV 2023) restores the overlap. Calibrated on vtest.avi: no buffer at
    <= 0.1 s (it only adds swaps there), 0.3 at 0.3 s, capped at 0.5.
    """
    return float(min(0.5, max(0.0, 1.5 * (dt_s - 0.1))))


class ByteTracker:
    """ByteTrack multi-object tracker (Zhang et al., ECCV 2022) with diagnostic event reporting.

    Follows the reference association order:

    1. confirmed + lost tracks  <->  high-score detections (IoU, optionally buffered)
    2. remaining tracked tracks <->  low-score detections (IoU)
    3. unconfirmed tracks       <->  remaining high-score detections (IoU, optionally buffered);
       unmatched ones are dropped
    4. new *unconfirmed* tracks from leftover high-score detections. A track only gets a public
       ID (and is only output) once a second detection confirms it, so one-frame false positives
       never show up as IDs. Detections mostly covered by a track already updated this frame
       (``spawn_overlap_thresh``, intersection over detection area) don't start new tracks, which
       stops partial-body duplicate boxes from spawning ghost IDs.
    5. tracked/lost tracks overlapping by IoU > ``duplicate_iou`` are de-duplicated (older wins).
    """

    name: str = "bytetrack"

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.55,
        match_thresh: float = 0.8,
        second_match_thresh: float = 0.5,
        unconfirmed_match_thresh: float = 0.7,
        track_buffer: int = 30,
        spawn_overlap_thresh: float = 0.8,
        duplicate_iou: float = 0.85,
        iou_buffer: float = 0.0,
    ) -> None:
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_thresh = float(match_thresh)
        self.second_match_thresh = float(second_match_thresh)
        self.unconfirmed_match_thresh = float(unconfirmed_match_thresh)
        self.track_buffer = int(track_buffer)
        self.spawn_overlap_thresh = float(spawn_overlap_thresh)
        self.duplicate_iou = float(duplicate_iou)
        # Buffered IoU for high-score association (see iou_buffer_for_interval); may be changed
        # between updates when the processing rate changes.
        self.iou_buffer = float(iou_buffer)

        self._next_id = 1
        self._next_uid = 1
        self._frame_id = 0
        self._tracks: list[_TrackState] = []
        self.last_events: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._next_id = 1
        self._next_uid = 1
        self._frame_id = 0
        self._tracks.clear()
        self.last_events = []

    def _compute_first_stage_cost(
        self,
        pool: list[_TrackState],
        dets: list[Detection],
    ) -> np.ndarray:
        return 1.0 - compute_iou_matrix(
            _expand_boxes(_boxes(pool), self.iou_buffer),
            _expand_boxes(_boxes(dets), self.iou_buffer),
        )

    def _apply_cmc(self, frame_bgr: np.ndarray | None) -> None:
        """No-op in ByteTrack; overridden by BoT-SORT."""
        _ = frame_bgr

    def _associate(
        self,
        tracks: list[_TrackState],
        dets: list[Detection],
        max_cost: float,
        *,
        low: bool = False,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        if low:
            cost = 1.0 - compute_iou_matrix(_boxes(tracks), _boxes(dets))
        else:
            cost = self._compute_first_stage_cost(tracks, dets)
        different_class = np.not_equal.outer(
            [t.class_id for t in tracks], [d.class_id for d in dets]
        )
        cost[different_class] = np.inf
        return _linear_assignment(cost, max_cost)

    def _activate(self, tr: _TrackState, events: list[dict[str, Any]]) -> None:
        tr.is_activated = True
        tr.track_id = self._next_id
        self._next_id += 1
        events.append({"type": "track_started", "track_id": tr.track_id})

        # Flag a possible ID switch if the new ID appears where another ID was just lost.
        recently_lost = [
            t
            for t in self._tracks
            if t.state == "lost"
            and t.class_id == tr.class_id
            and t.is_activated
            and t.time_since_update <= 5
        ]
        if recently_lost:
            ious = compute_iou_matrix(_boxes([tr]), _boxes(recently_lost))[0]
            j = int(np.argmax(ious))
            if float(ious[j]) >= 0.25:
                events.append(
                    {
                        "type": "possible_id_switch",
                        "new_track_id": tr.track_id,
                        "previous_track_id": recently_lost[j].track_id,
                        "iou": round(float(ious[j]), 3),
                    }
                )

    @staticmethod
    def _apply_detection(tr: _TrackState, det: Detection, *, update_class: bool) -> None:
        tr.kf.update(det.bbox)
        tr.confidence = det.confidence
        if update_class:
            tr.class_id = det.class_id
            tr.class_name = det.class_name
        tr.hits += 1
        tr.time_since_update = 0
        tr.update_embedding(det.embedding)

    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[Track]:
        """Advance the tracker by one frame and return the confirmed tracks observed this frame."""
        self._frame_id += 1
        events: list[dict[str, Any]] = []

        # 1. Predict every track (lost tracks keep position velocity but stop growing/shrinking).
        for t in self._tracks:
            t.recovered_now = False
            if t.state != "tracked":
                t.kf.mean[7] = 0.0
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        # Optional camera motion compensation (active in BoT-SORT)
        self._apply_cmc(frame_bgr)

        high_dets = [d for d in detections if d.confidence >= self.track_high_thresh]
        low_dets = [
            d for d in detections if self.track_low_thresh <= d.confidence < self.track_high_thresh
        ]

        confirmed = [t for t in self._tracks if t.is_activated and t.state == "tracked"]
        lost = [t for t in self._tracks if t.state == "lost"]
        unconfirmed = [t for t in self._tracks if not t.is_activated]
        pool = confirmed + lost
        updated: set[int] = set()

        # 2. First association: confirmed + lost tracks with high-score detections.
        matches, u_pool, u_high = self._associate(pool, high_dets, self.match_thresh)
        for ti, di in matches:
            tr = pool[ti]
            if tr.state == "lost":
                tr.recovered_now = True
                events.append({"type": "track_recovered", "track_id": tr.track_id})
            tr.state = "tracked"
            self._apply_detection(tr, high_dets[di], update_class=True)
            updated.add(tr.uid)

        # 3. Second association: remaining *tracked* tracks with low-score detections (IoU only).
        rem_tracked = [pool[i] for i in u_pool if pool[i].state == "tracked"]
        if rem_tracked and low_dets:
            low_matches, _, _ = self._associate(
                rem_tracked, low_dets, self.second_match_thresh, low=True
            )
            for ti, di in low_matches:
                self._apply_detection(rem_tracked[ti], low_dets[di], update_class=False)
                updated.add(rem_tracked[ti].uid)

        # 4. Unconfirmed tracks with the high-score detections left over from stage 1.
        rem_high = [high_dets[i] for i in u_high]
        un_matches, _, u_rem = self._associate(unconfirmed, rem_high, self.unconfirmed_match_thresh)
        for ti, di in un_matches:
            tr = unconfirmed[ti]
            self._apply_detection(tr, rem_high[di], update_class=True)
            updated.add(tr.uid)
            self._activate(tr, events)

        # 5. Lost-state transitions, expiry, and removal of unconfirmed tracks that missed.
        surviving: list[_TrackState] = []
        for tr in self._tracks:
            if tr.uid in updated:
                surviving.append(tr)
                continue
            if not tr.is_activated:
                continue  # unconfirmed and unmatched -> removed
            if tr.state == "tracked":
                tr.state = "lost"
                events.append({"type": "track_lost", "track_id": tr.track_id})
            if tr.time_since_update <= self.track_buffer:
                surviving.append(tr)
            else:
                events.append({"type": "track_ended", "track_id": tr.track_id})
        self._tracks = surviving

        # 6. Spawn new (unconfirmed) tracks from leftover high-score detections.
        occupied: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        for t in self._tracks:
            if t.uid in updated:
                occupied[t.class_id].append(t.bbox)
        leftovers = sorted((rem_high[i] for i in u_rem), key=lambda d: d.confidence, reverse=True)
        for det in leftovers:
            if det.confidence < self.new_track_thresh:
                continue
            same_class = occupied[det.class_id]
            if same_class and (
                float(compute_ioa_matrix(_boxes([det]), np.array(same_class)).max())
                >= self.spawn_overlap_thresh
            ):
                continue
            tr = _TrackState(
                uid=self._next_uid,
                kf=KalmanBoxFilter(det.bbox),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                start_frame=self._frame_id,
            )
            self._next_uid += 1
            tr.update_embedding(det.embedding)
            self._tracks.append(tr)
            updated.add(tr.uid)
            same_class.append(tr.bbox)
            if self._frame_id == 1:
                self._activate(tr, events)

        # 7. Remove tracked/lost duplicates (the longer-lived track wins).
        self._remove_duplicates()

        self.last_events = events
        outputs = [t.to_track() for t in self._tracks if t.is_activated and t.uid in updated]
        outputs.sort(key=lambda t: t.track_id)
        return outputs

    def _remove_duplicates(self) -> None:
        tracked = [t for t in self._tracks if t.state == "tracked"]
        lost = [t for t in self._tracks if t.state == "lost"]
        if not tracked or not lost:
            return
        iou = compute_iou_matrix(_boxes(tracked), _boxes(lost))
        drop: set[int] = set()
        for p, q in zip(*np.where(iou > self.duplicate_iou), strict=False):
            a, b = tracked[int(p)], lost[int(q)]
            if a.class_id != b.class_id:
                continue
            if (self._frame_id - a.start_frame) > (self._frame_id - b.start_frame):
                drop.add(b.uid)
            else:
                drop.add(a.uid)
        if drop:
            self._tracks = [t for t in self._tracks if t.uid not in drop]

    def coasting_tracks(self, max_frames: int = 3) -> list[Track]:
        """Confirmed tracks missed for at most ``max_frames`` frames, at their predicted position.

        Useful for display and anonymization so boxes don't blink when the detector misses a
        frame. These are predictions, not observations; they are never written to exports.
        """
        out = [
            t.to_track(coasting=True)
            for t in self._tracks
            if t.is_activated and t.state == "lost" and t.time_since_update <= max_frames
        ]
        out.sort(key=lambda t: t.track_id)
        return out


class BotSortTracker(ByteTracker):
    """BoT-SORT tracker: ByteTrack + Sparse Optical Flow CMC + appearance Re-ID fusion.

    Appearance only helps when embeddings come from a person Re-ID model (``--reid osnet``);
    CLIP/SigLIP image embeddings barely separate different people and must not be used here.
    """

    name: str = "botsort"

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.55,
        match_thresh: float = 0.8,
        second_match_thresh: float = 0.5,
        unconfirmed_match_thresh: float = 0.7,
        track_buffer: int = 45,
        spawn_overlap_thresh: float = 0.8,
        iou_buffer: float = 0.0,
        use_cmc: bool = True,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
    ) -> None:
        super().__init__(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=match_thresh,
            second_match_thresh=second_match_thresh,
            unconfirmed_match_thresh=unconfirmed_match_thresh,
            track_buffer=track_buffer,
            spawn_overlap_thresh=spawn_overlap_thresh,
            iou_buffer=iou_buffer,
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
        if self._cmc is None or frame_bgr is None:
            return
        # Estimate on every frame (even with no tracks) so the reference frame never goes stale.
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
        t_has = [t.embedding is not None for t in pool]
        d_has = [d.embedding is not None for d in dets]
        if not any(t_has) or not any(d_has):
            return iou_dist

        dim = next(t.embedding for t in pool if t.embedding is not None).shape[0]
        t_emb = np.zeros((len(pool), dim), dtype=np.float64)
        d_emb = np.zeros((len(dets), dim), dtype=np.float64)
        for i, t in enumerate(pool):
            if t.embedding is not None:
                t_emb[i] = t.embedding
        for j, d in enumerate(dets):
            if d.embedding is not None:
                n = float(np.linalg.norm(d.embedding))
                if n > 1e-6:
                    d_emb[j] = d.embedding / n

        # Reference BoT-SORT fusion: cosine distance / 2, gated by appearance AND IoU proximity.
        emb_dist = np.clip(0.5 * (1.0 - t_emb @ d_emb.T), 0.0, 1.0)
        emb_dist[~np.array(t_has), :] = 1.0
        emb_dist[:, ~np.array(d_has)] = 1.0
        emb_dist[emb_dist > self.appearance_thresh] = 1.0
        emb_dist[iou_dist > self.proximity_thresh] = 1.0
        return np.minimum(iou_dist, emb_dist)


def create_tracker(
    tracker_name: str = "bytetrack",
    *,
    track_high_thresh: float = 0.5,
    track_low_thresh: float = 0.15,
    new_track_thresh: float = 0.55,
    track_buffer: int = 30,
    iou_buffer: float = 0.0,
) -> ByteTracker:
    """Factory for built-in multi-object trackers."""
    key = tracker_name.lower().strip().replace("-", "_")
    if key in ("bytetrack", "byte"):
        return ByteTracker(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            iou_buffer=iou_buffer,
        )
    if key in ("botsort", "bot_sort"):
        return BotSortTracker(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=max(track_buffer, 45),
            iou_buffer=iou_buffer,
        )
    raise ValueError(f"Unknown tracker '{tracker_name}'. Choose 'bytetrack' or 'botsort'.")
