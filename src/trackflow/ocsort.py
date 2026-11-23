"""OC-SORT multi-object tracker (Cao et al., "Observation-Centric SORT", CVPR 2023).

A port of the reference implementation (github.com/noahcao/OC_SORT, MIT license) onto Trackflow's
track lifecycle. Association is class-aware, and each assignment only considers pairs that pass
the IoU gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trackflow.tracker import (
    _assign,
    _boxes,
    _expand_boxes,
    _SortStyleTracker,
    _SortTrack,
    compute_iou_matrix,
)
from trackflow.types import Detection, Track

Box = tuple[float, float, float, float]


def _box_to_z(box: Box) -> np.ndarray:
    """``[cx, cy, area, w/h]``, SORT's measurement vector."""
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    return np.array([x1 + 0.5 * w, y1 + 0.5 * h, w * h, w / h], dtype=np.float64)


def _z_to_box(z: np.ndarray) -> Box:
    area = max(float(z[2]), 1e-6)
    ratio = max(float(z[3]), 1e-6)
    w = float(np.sqrt(area * ratio))
    h = area / w
    cx, cy = float(z[0]), float(z[1])
    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def _centers(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def _direction(a: Box, b: Box) -> np.ndarray:
    """Unit vector ``(dy, dx)`` from the center of ``a`` to the center of ``b``."""
    (ca, cb) = _centers(np.array([a, b], dtype=np.float64))
    d = np.array([cb[1] - ca[1], cb[0] - ca[0]])
    return d / (float(np.hypot(d[0], d[1])) + 1e-6)


def _transition(n: int) -> np.ndarray:
    f = np.eye(n, dtype=np.float64)
    for i in range(n - 4):
        f[i, i + 4] = 1.0
    return f


class _OcSortMotion:
    """SORT's Kalman filter on ``[cx, cy, area, w/h]`` plus the observations OC-SORT uses.

    While a track goes unmatched the filter coasts on its prediction. When it matches again, the
    state is rolled back to the first missed frame and re-updated along a straight line from the
    last observation to the new one (observation-centric re-update), which undoes the drift
    accumulated while coasting.
    """

    _F = _transition(7)
    _H = np.eye(4, 7, dtype=np.float64)
    _Q = np.diag([1.0, 1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-4])
    _R = np.diag([1.0, 1.0, 10.0, 10.0])

    def __init__(self, box: Box, delta_t: int) -> None:
        self.x = np.zeros(7, dtype=np.float64)
        self.x[:4] = _box_to_z(box)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.delta_t = delta_t
        self.observations: dict[int, Box] = {0: box}  # track age -> observed box
        self.last_observation = box
        self.direction = np.zeros(2)  # unit (dy, dx) of recent motion, zero until known
        self._frozen: tuple[np.ndarray, np.ndarray] | None = None
        self._missed = 0

    @property
    def box(self) -> Box:
        return _z_to_box(self.x)

    def predict(self) -> None:
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self._predict()

    def _predict(self) -> None:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q

    def _correct(self, z: np.ndarray) -> None:
        s = self._H @ self.P @ self._H.T + self._R
        gain = np.linalg.solve(s, self._H @ self.P).T
        self.x = self.x + gain @ (z - self._H @ self.x)
        i_kh = np.eye(7) - gain @ self._H
        self.P = i_kh @ self.P @ i_kh.T + gain @ self._R @ gain.T

    def reference_observation(self, age: int) -> Box:
        """The observation ``delta_t`` frames back, else the closest later one."""
        for dt in range(self.delta_t, 0, -1):
            obs = self.observations.get(age - dt)
            if obs is not None:
                return obs
        return self.last_observation

    def miss(self) -> None:
        if self._frozen is None:
            self._frozen = (self.x.copy(), self.P.copy())
        self._missed += 1

    def update(self, box: Box, age: int) -> None:
        self.direction = _direction(self.reference_observation(age), box)
        if self._frozen is None:
            self._correct(_box_to_z(box))
        else:
            self.x, self.P = self._frozen
            gap = self._missed + 1
            self._frozen, self._missed = None, 0
            start, end = self.last_observation, box
            w0, h0 = start[2] - start[0], start[3] - start[1]
            w1, h1 = end[2] - end[0], end[3] - end[1]
            (c0, c1) = _centers(np.array([start, end], dtype=np.float64))
            for step in range(1, gap + 1):
                f = step / gap
                cx, cy = c0 + f * (c1 - c0)
                w = max(w0 + f * (w1 - w0), 1e-3)
                h = max(h0 + f * (h1 - h0), 1e-3)
                self._correct(np.array([cx, cy, w * h, w / h]))
                if step < gap:
                    self._predict()
        self.last_observation = box
        self.observations[age] = box
        oldest = age + 1 - self.delta_t
        self.observations = {a: b for a, b in self.observations.items() if a >= oldest}


class OcSortTracker(_SortStyleTracker):
    """OC-SORT: SORT made robust to occlusion by trusting observations over predictions.

    Each frame:

    1. high-score detections <-> every track's predicted box, by IoU plus observation-centric
       momentum (OCM): a bonus, weighted by ``inertia`` and the detection score, when the
       direction from the track's observation ``delta_t`` frames back to the detection agrees
       with the track's recent direction of motion;
    2. with ``use_byte``, low-score detections <-> remaining tracks by IoU (ByteTrack's second
       stage);
    3. observation-centric recovery (OCR): remaining high-score detections <-> remaining
       tracks' *last observed* boxes by IoU, which recovers tracks the filter has drifted off;
    4. unmatched high-score detections spawn tracks.

    A track that matches again after missing frames is re-updated along a virtual trajectory
    (ORU). Confirmed tracks report the matched detection's box.
    """

    name: str = "ocsort"

    def __init__(
        self,
        *,
        det_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        delta_t: int = 3,
        inertia: float = 0.2,
        use_byte: bool = False,
        iou_buffer: float = 0.0,
    ) -> None:
        self.det_thresh = float(det_thresh)
        self.low_thresh = float(low_thresh)
        self.iou_threshold = float(iou_threshold)
        self.delta_t = int(delta_t)
        self.inertia = float(inertia)
        self.use_byte = bool(use_byte)
        # Buffered IoU for the first association (see iou_buffer_for_interval).
        self.iou_buffer = float(iou_buffer)
        super().__init__(max_age=max_age, min_hits=min_hits)

    def _gate(self, iou: np.ndarray, tracks: list[_SortTrack], dets: list[Detection]) -> np.ndarray:
        same_class = np.equal.outer([t.class_id for t in tracks], [d.class_id for d in dets])
        return (iou >= self.iou_threshold) & same_class

    def _momentum(self, tracks: list[_SortTrack], dets: list[Detection]) -> np.ndarray:
        """OCM bonus for each (track, detection) pair, in ``[-inertia/2, inertia/2] * score``."""
        refs = np.array([t.motion.reference_observation(t.age) for t in tracks], dtype=np.float64)
        delta = _centers(_boxes(dets))[None, :, :] - _centers(refs)[:, None, :]
        norm = np.hypot(delta[..., 0], delta[..., 1]) + 1e-6
        dx, dy = delta[..., 0] / norm, delta[..., 1] / norm
        velocity = np.array([t.motion.direction for t in tracks])  # (dy, dx) per track
        cos = np.clip(velocity[:, 0:1] * dy + velocity[:, 1:2] * dx, -1.0, 1.0)
        agreement = (np.pi / 2 - np.abs(np.arccos(cos))) / np.pi
        scores = np.array([d.confidence for d in dets])
        return agreement * self.inertia * scores[None, :]

    def _match(self, tr: _SortTrack, det: Detection) -> None:
        tr.motion.update(det.bbox, tr.age)
        self._observe(tr, det, det.bbox)

    def update(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray | None = None,
    ) -> list[Track]:
        """Advance the tracker by one frame and return the tracks reported on it."""
        _ = frame_bgr
        self._frame_id += 1
        events: list[dict[str, Any]] = []
        self._predict()
        tracks = list(self._tracks)

        high = [d for d in detections if d.confidence >= self.det_thresh]
        u_det = list(range(len(high)))
        u_trk = list(range(len(tracks)))

        # 1. High-score detections <-> predicted boxes, IoU plus direction consistency.
        if tracks and high:
            iou = compute_iou_matrix(
                _expand_boxes(np.array([t.motion.box for t in tracks]), self.iou_buffer),
                _expand_boxes(_boxes(high), self.iou_buffer),
            )
            matches, u_trk, u_det = _assign(
                iou + self._momentum(tracks, high), self._gate(iou, tracks, high)
            )
            for ti, di in matches:
                self._match(tracks[ti], high[di])

        # 2. Optional BYTE stage: low-score detections <-> remaining predicted boxes.
        low = (
            [d for d in detections if self.low_thresh <= d.confidence < self.det_thresh]
            if self.use_byte
            else []
        )
        if low and u_trk:
            rest = [tracks[i] for i in u_trk]
            iou = compute_iou_matrix(np.array([t.motion.box for t in rest]), _boxes(low))
            matches, left, _ = _assign(iou, self._gate(iou, rest, low))
            for ti, di in matches:
                self._match(rest[ti], low[di])
            u_trk = [u_trk[i] for i in left]

        # 3. OCR: remaining high-score detections <-> last observed boxes.
        if u_det and u_trk:
            rest = [tracks[i] for i in u_trk]
            dets = [high[i] for i in u_det]
            iou = compute_iou_matrix(
                np.array([t.motion.last_observation for t in rest], dtype=np.float64),
                _boxes(dets),
            )
            matches, left_t, left_d = _assign(iou, self._gate(iou, rest, dets))
            for ti, di in matches:
                self._match(rest[ti], dets[di])
            u_trk = [u_trk[i] for i in left_t]
            u_det = [u_det[i] for i in left_d]

        for i in u_trk:
            tracks[i].motion.miss()
        for i in u_det:
            self._spawn(high[i], _OcSortMotion(high[i].bbox, self.delta_t))
        return self._finish_frame(events)
