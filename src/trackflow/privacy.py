"""Privacy anonymization: Gaussian blur, pixelation, head/face blur, and stick-figure-only mode.

Best effort, not a guarantee: every box the detector produced (any score, including people not yet
confirmed as tracks) plus briefly-missed tracks is anonymized, with a margin. A person the detector
never finds in a frame is not. ``face-blur`` assumes the head is in the top 30% of the box.
``skeleton`` draws a fixed stick figure inside each track box (no pose estimation) on a blank
canvas, so no video pixels are written at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np

from trackflow.types import Detection, Track

AnonymizeMode = Literal["none", "blur", "pixelate", "face-blur", "skeleton"]


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic vibrant BGR color per track ID."""
    palette = [
        (92, 214, 148),
        (250, 173, 72),
        (140, 130, 255),
        (240, 200, 70),
        (110, 235, 255),
        (255, 130, 210),
        (135, 255, 190),
        (195, 155, 255),
    ]
    return palette[track_id % len(palette)]


def _fast_gaussian_blur(roi: np.ndarray, kernel_size: int) -> np.ndarray:
    """Gaussian blur with 2x downscale-blur-upsample for large ROIs and wide kernels."""
    rh, rw = roi.shape[:2]
    if kernel_size >= 21 and rh >= 32 and rw >= 32:
        half_k = max(5, (kernel_size // 2) | 1)
        small = cv2.resize(roi, (max(2, rw // 2), max(2, rh // 2)), interpolation=cv2.INTER_AREA)
        blurred_small = cv2.GaussianBlur(small, (half_k, half_k), 0)
        return cv2.resize(blurred_small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)


def _dedup_boxes_preserving_coverage(
    rects: list[tuple[int, int, int, int]],
    *,
    merge_iou: float = 0.85,
) -> list[tuple[int, int, int, int]]:
    """Collapse redundant overlapping rects without ever uncovering any pixel.

    Drops boxes strictly contained inside an already-kept box, and merges near-identical
    overlapping boxes (``IoU >= merge_iou``, e.g. a track box and its raw detection box)
    into their enclosing union rectangle so every original pixel stays covered in one pass.
    """
    if len(rects) <= 1:
        return rects
    # Largest first so strictly contained boxes are eliminated immediately.
    ordered = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in ordered:
        area_a = (x2 - x1) * (y2 - y1)
        merged = False
        for i, (kx1, ky1, kx2, ky2) in enumerate(kept):
            if kx1 <= x1 and ky1 <= y1 and kx2 >= x2 and ky2 >= y2:
                merged = True
                break
            iw = min(x2, kx2) - max(x1, kx1)
            ih = min(y2, ky2) - max(y1, ky1)
            if iw > 0 and ih > 0:
                inter = iw * ih
                area_b = (kx2 - kx1) * (ky2 - ky1)
                if inter / (area_a + area_b - inter) >= merge_iou:
                    kept[i] = (min(x1, kx1), min(y1, ky1), max(x2, kx2), max(y2, ky2))
                    merged = True
                    break
        if not merged:
            kept.append((x1, y1, x2, y2))
    return kept


def _cluster_head_boxes(
    head_rects: list[tuple[int, int, int, int]],
    *,
    cluster_iou: float = 0.50,
) -> list[tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]]]:
    """Group overlapping head rects so face-blur runs one blur per cluster with exact ellipses."""
    if not head_rects:
        return []
    # Deduplicate exact identical head rects first (order-independent for union of ellipses).
    unique_rects = list(dict.fromkeys(head_rects))
    clusters: list[list[tuple[int, int, int, int]]] = []
    unions: list[tuple[int, int, int, int]] = []
    for r in sorted(unique_rects, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        x1, y1, x2, y2 = r
        area_a = (x2 - x1) * (y2 - y1)
        placed = False
        for i, (ux1, uy1, ux2, uy2) in enumerate(unions):
            if ux1 <= x1 and uy1 <= y1 and ux2 >= x2 and uy2 >= y2:
                clusters[i].append(r)
                placed = True
                break
            iw = min(x2, ux2) - max(x1, ux1)
            ih = min(y2, uy2) - max(y1, uy1)
            if iw > 0 and ih > 0:
                inter = iw * ih
                area_b = (ux2 - ux1) * (uy2 - uy1)
                if inter / (area_a + area_b - inter) >= cluster_iou:
                    clusters[i].append(r)
                    unions[i] = (min(x1, ux1), min(y1, uy1), max(x2, ux2), max(y2, uy2))
                    placed = True
                    break
        if not placed:
            clusters.append([r])
            unions.append(r)
    return list(zip(unions, clusters, strict=True))


class PrivacyAnonymizer:
    """Applies privacy-preserving transformations to every detected person box."""

    def __init__(
        self, mode: AnonymizeMode = "none", *, strength: int = 31, margin: float = 0.08
    ) -> None:
        valid = ("none", "blur", "pixelate", "face-blur", "skeleton")
        if mode not in valid:
            raise ValueError(f"Unknown anonymize mode '{mode}'. Choose from: {valid}")
        self.mode: AnonymizeMode = mode
        # Ensure odd kernel size >= 9
        self.kernel_size = max(9, int(strength) | 1)
        self.margin = float(margin)

    def apply(
        self,
        frame_bgr: np.ndarray,
        tracks: Sequence[Track],
        *,
        detections: Sequence[Detection] = (),
        coasting: Sequence[Track] = (),
    ) -> np.ndarray:
        """Return an anonymized copy of ``frame_bgr``.

        Anonymizes the boxes of ``tracks``, of briefly-missed ``coasting`` tracks, and of all raw
        ``detections`` (pass every detection, not only the ones that became tracks).
        """
        if self.mode == "none":
            return frame_bgr

        if self.mode == "skeleton":
            return self._render_skeleton_canvas(frame_bgr, [*tracks, *coasting])

        out = frame_bgr.copy()
        h, w = out.shape[:2]

        rects: list[tuple[int, int, int, int]] = []
        for box in (
            *(t.bbox for t in tracks),
            *(t.bbox for t in coasting),
            *(d.bbox for d in detections),
        ):
            bx1, by1, bx2, by2 = box
            mx, my = (bx2 - bx1) * self.margin, (by2 - by1) * self.margin
            x1 = max(0, min(w - 1, int(round(bx1 - mx))))
            y1 = max(0, min(h - 1, int(round(by1 - my))))
            x2 = max(x1 + 1, min(w, int(round(bx2 + mx))))
            y2 = max(y1 + 1, min(h, int(round(by2 + my))))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            rects.append((x1, y1, x2, y2))

        if not rects:
            return out

        if self.mode == "face-blur":
            head_rects = [
                (x1, y1, x2, min(h, y1 + max(8, int((y2 - y1) * 0.30))))
                for x1, y1, x2, y2 in rects
            ]
            for (ux1, uy1, ux2, uy2), members in _cluster_head_boxes(head_rects):
                roi = out[uy1:uy2, ux1:ux2]
                blurred = _fast_gaussian_blur(roi, self.kernel_size)
                mask = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.uint8)
                for rx1, ry1, rx2, ry2 in members:
                    cv2.ellipse(
                        mask,
                        ((rx1 + rx2) // 2 - ux1, (ry1 + ry2) // 2 - uy1),
                        (max(2, int((rx2 - rx1) * 0.42)), max(2, int((ry2 - ry1) * 0.48))),
                        0,
                        0,
                        360,
                        255,
                        -1,
                    )
                out[uy1:uy2, ux1:ux2] = np.where(mask[:, :, None] > 0, blurred, roi)
            return out

        for x1, y1, x2, y2 in _dedup_boxes_preserving_coverage(rects):
            roi = out[y1:y2, x1:x2]
            if self.mode == "blur":
                out[y1:y2, x1:x2] = _fast_gaussian_blur(roi, self.kernel_size)
            elif self.mode == "pixelate":
                bw, bh = max(4, (x2 - x1) // 10), max(4, (y2 - y1) // 10)
                small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
                out[y1:y2, x1:x2] = cv2.resize(
                    small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
                )

        return out

    @staticmethod
    def _render_skeleton_canvas(frame_bgr: np.ndarray, tracks: Sequence[Track]) -> np.ndarray:
        """Render box-based stick figures (no pose estimation) on a blank grid canvas."""
        h, w = frame_bgr.shape[:2]
        canvas = np.full((h, w, 3), (18, 20, 26), dtype=np.uint8)

        # Subtle analytics grid
        step = max(32, min(w, h) // 12)
        for gx in range(0, w, step):
            cv2.line(canvas, (gx, 0), (gx, h), (28, 32, 42), 1)
        for gy in range(0, h, step):
            cv2.line(canvas, (0, gy), (w, gy), (28, 32, 42), 1)

        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            bw = max(8.0, x2 - x1)
            bh = max(16.0, y2 - y1)
            cx = int(round((x1 + x2) * 0.5))
            color = _track_color(tr.track_id)

            # Anatomical landmarks scaled to bounding box
            head_y = int(round(y1 + bh * 0.12))
            head_r = max(3, int(round(min(bw * 0.18, bh * 0.08))))
            neck = (cx, int(round(y1 + bh * 0.22)))
            hip = (cx, int(round(y1 + bh * 0.56)))
            l_shoulder = (int(round(cx - bw * 0.24)), int(round(y1 + bh * 0.26)))
            r_shoulder = (int(round(cx + bw * 0.24)), int(round(y1 + bh * 0.26)))
            l_hand = (int(round(cx - bw * 0.34)), int(round(y1 + bh * 0.52)))
            r_hand = (int(round(cx + bw * 0.34)), int(round(y1 + bh * 0.52)))
            l_knee = (int(round(cx - bw * 0.15)), int(round(y1 + bh * 0.78)))
            r_knee = (int(round(cx + bw * 0.15)), int(round(y1 + bh * 0.78)))
            l_foot = (int(round(cx - bw * 0.18)), int(round(y2 - 2)))
            r_foot = (int(round(cx + bw * 0.18)), int(round(y2 - 2)))

            # Head & skeleton limbs
            cv2.circle(canvas, (cx, head_y), head_r, color, 2, cv2.LINE_AA)
            bones = [
                (neck, hip),
                (l_shoulder, r_shoulder),
                (l_shoulder, l_hand),
                (r_shoulder, r_hand),
                (hip, l_knee),
                (l_knee, l_foot),
                (hip, r_knee),
                (r_knee, r_foot),
            ]
            for p1, p2 in bones:
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

            for joint in (neck, hip, l_shoulder, r_shoulder, l_knee, r_knee):
                cv2.circle(canvas, joint, 3, (240, 245, 255), -1, cv2.LINE_AA)

        return canvas
