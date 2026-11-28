"""OC-SORT multi-object tracker (Cao et al., "Observation-Centric SORT", CVPR 2023).

A port of the reference implementation (github.com/noahcao/OC_SORT, MIT license) onto Trackflow's
track lifecycle. Association is class-aware, and each assignment only considers pairs that pass
the IoU gate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from trackflow.tracker import (
    SparseOpticalFlowCMC,
    _assign,
    _boxes,
    _expand_boxes,
    _SortStyleTracker,
    _SortTrack,
    compute_hmiou_matrix,
    compute_ioa_matrix,
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
    dx = 0.5 * ((b[0] + b[2]) - (a[0] + a[2]))
    dy = 0.5 * ((b[1] + b[3]) - (a[1] + a[3]))
    inv = 1.0 / (math.hypot(dy, dx) + 1e-6)
    return np.array([dy * inv, dx * inv], dtype=np.float64)


def _transition(n: int) -> np.ndarray:
    f = np.eye(n, dtype=np.float64)
    for i in range(n - 4):
        f[i, i + 4] = 1.0
    return f


def _soft_iou(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Blend IoU with expanded IoU to tolerate small prediction errors."""
    raw = compute_iou_matrix(tracks, detections)
    buffered = compute_iou_matrix(_expand_boxes(tracks, 0.15), _expand_boxes(detections, 0.15))
    return (raw + buffered) / 2


def _hm_soft_iou(tracks: np.ndarray, detections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Soft IoU for gating and its height-modulated version (Hybrid-SORT HMIoU) for cost."""
    iou = _soft_iou(tracks, detections)
    return iou, compute_hmiou_matrix(tracks, detections, iou)


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

    def compensate(self, warp: np.ndarray, jacobian: np.ndarray, scale: float) -> None:
        """Map the filter and its stored observations into the current frame."""
        shift = warp[:, 2]
        linear = warp[:, :2]
        self.x = jacobian @ self.x
        self.x[:2] += shift
        self.P = jacobian @ self.P @ jacobian.T
        if self._frozen is not None:
            state, covariance = self._frozen
            state = jacobian @ state
            state[:2] += shift
            self._frozen = (state, jacobian @ covariance @ jacobian.T)
        self.direction = (linear @ self.direction[::-1])[::-1] / scale

        def move(box: Box) -> Box:
            x1, y1, x2, y2 = box
            cx, cy = linear @ ((x1 + x2) / 2, (y1 + y2) / 2) + shift
            hw, hh = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
            return (cx - hw, cy - hh, cx + hw, cy + hh)

        self.last_observation = move(self.last_observation)
        self.observations = {age: move(box) for age, box in self.observations.items()}

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

    1. high-score detections <-> confirmed tracks' predicted boxes, by IoU plus observation-centric
       momentum (OCM): a bonus, weighted by ``inertia`` and the detection score, when the
       direction from the track's observation ``delta_t`` frames back to the detection agrees
       with the track's recent direction of motion;
    2. observation-centric recovery (OCR): remaining high-score detections <-> remaining
       tracks' *last observed* boxes by IoU, which recovers tracks the filter has drifted off;
    3. with ``use_byte``, low-score detections <-> remaining confirmed tracks seen on the
       previous frame, by IoU;
    4. remaining high-score detections <-> tentative (not yet confirmed) tracks, by IoU;
    5. unmatched detections at or above ``new_track_thresh`` spawn tracks, unless mostly
       covered by a box matched this frame. A tentative track that misses a frame is dropped.

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
        use_byte: bool = True,
        iou_buffer: float = 0.0,
        use_cmc: bool = True,
        new_track_thresh: float | None = None,
    ) -> None:
        self.det_thresh = float(det_thresh)
        self.new_track_thresh = (
            float(new_track_thresh)
            if new_track_thresh is not None
            else max(self.det_thresh, min(0.95, self.det_thresh + 0.05))
        )
        self.low_thresh = float(low_thresh)
        self.iou_threshold = float(iou_threshold)
        self.delta_t = int(delta_t)
        self.inertia = float(inertia)
        self.use_byte = bool(use_byte)
        self.use_cmc = bool(use_cmc)
        # Buffered IoU for the first association (see iou_buffer_for_interval).
        self.iou_buffer = float(iou_buffer)
        super().__init__(max_age=max_age, min_hits=min_hits)

    def reset(self) -> None:
        super().reset()
        self._cmc = SparseOpticalFlowCMC() if self.use_cmc else None

    def _compensate(self, frame: np.ndarray, detections: list[Detection]) -> None:
        assert self._cmc is not None
        warp = self._cmc.estimate(frame, _boxes(detections))
        if warp is None:
            return
        linear = warp[:, :2]
        area_scale = float(np.sum(linear[0] ** 2))
        if area_scale <= 1e-6:
            return
        jacobian = np.eye(7)
        jacobian[:2, :2] = jacobian[4:6, 4:6] = linear
        jacobian[2, 2] = jacobian[6, 6] = area_scale
        scale = float(np.sqrt(area_scale))
        for track in self._tracks:
            track.motion.compensate(warp, jacobian, scale)

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
        self._frame_id += 1
        events: list[dict[str, Any]] = []
        if self._cmc is not None:
            if frame_bgr is None:
                self._cmc = SparseOpticalFlowCMC()
            else:
                self._compensate(frame_bgr, detections)
        self._predict()
        # Confirmed tracks associate first; tentative tracks only get the leftover detections.
        tracks = [t for t in self._tracks if t.track_id]
        tentative = [t for t in self._tracks if not t.track_id]

        high = [d for d in detections if d.confidence >= self.det_thresh]
        u_det = list(range(len(high)))
        u_trk = list(range(len(tracks)))

        # 1. High-score detections <-> predicted boxes, IoU plus direction consistency.
        if tracks and high:
            iou, cost = _hm_soft_iou(
                _expand_boxes(np.array([t.motion.box for t in tracks]), self.iou_buffer),
                _expand_boxes(_boxes(high), self.iou_buffer),
            )
            matches, u_trk, u_det = _assign(
                cost + self._momentum(tracks, high), self._gate(iou, tracks, high)
            )
            for ti, di in matches:
                self._match(tracks[ti], high[di])

        # 2. OCR: remaining high-score detections <-> last observed boxes.
        if u_det and u_trk:
            rest = [tracks[i] for i in u_trk]
            dets = [high[i] for i in u_det]
            iou, cost = _hm_soft_iou(
                np.array([t.motion.last_observation for t in rest], dtype=np.float64),
                _boxes(dets),
            )
            matches, left_t, left_d = _assign(cost, self._gate(iou, rest, dets))
            for ti, di in matches:
                self._match(rest[ti], dets[di])
            u_trk = [u_trk[i] for i in left_t]
            u_det = [u_det[i] for i in left_d]

        # 3. Low scores can sustain an active track after both high-score passes.
        if self.use_byte and u_trk:
            active = [i for i in u_trk if tracks[i].track_id and tracks[i].time_since_update == 1]
            low = [d for d in detections if self.low_thresh <= d.confidence < self.det_thresh]
            if active and low:
                rest = [tracks[i] for i in active]
                iou = compute_iou_matrix(np.array([t.motion.box for t in rest]), _boxes(low))
                matches, _, _ = _assign(iou, self._gate(iou, rest, low) & (iou >= 0.5))
                for ti, di in matches:
                    self._match(rest[ti], low[di])

        # 4. Tentative tracks <-> remaining high-score detections.
        if tentative and u_det:
            dets = [high[i] for i in u_det]
            iou, cost = _hm_soft_iou(np.array([t.motion.box for t in tentative]), _boxes(dets))
            matches, _, left_d = _assign(cost, self._gate(iou, tentative, dets))
            for ti, di in matches:
                self._match(tentative[ti], dets[di])
            u_det = [u_det[i] for i in left_d]

        for i in u_trk:
            if tracks[i].time_since_update:
                tracks[i].motion.miss()
        # A tentative track that misses a frame is dropped so it can't take later detections.
        self._tracks = [t for t in self._tracks if t.track_id or not t.time_since_update]
        # Spawn from confident leftovers that aren't mostly covered by a box matched this frame.
        taken: dict[int, list[Box]] = {}
        for t in self._tracks:
            if not t.time_since_update:
                taken.setdefault(t.class_id, []).append(t.box)
        for i in sorted(u_det, key=lambda i: -high[i].confidence):
            det = high[i]
            if det.confidence < self.new_track_thresh:
                continue
            same = taken.get(det.class_id)
            if same and float(compute_ioa_matrix(_boxes([det]), np.array(same)).max()) >= 0.8:
                continue
            self._spawn(det, _OcSortMotion(det.bbox, self.delta_t))
            taken.setdefault(det.class_id, []).append(det.bbox)
        return self._finish_frame(events)
