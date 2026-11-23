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

        boxes = [t.bbox for t in tracks] + [t.bbox for t in coasting] + [d.bbox for d in detections]
        for bx1, by1, bx2, by2 in boxes:
            mx, my = (bx2 - bx1) * self.margin, (by2 - by1) * self.margin
            x1 = max(0, min(w - 1, int(round(bx1 - mx))))
            y1 = max(0, min(h - 1, int(round(by1 - my))))
            x2 = max(x1 + 1, min(w, int(round(bx2 + mx))))
            y2 = max(y1 + 1, min(h, int(round(by2 + my))))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            if self.mode == "blur":
                roi = out[y1:y2, x1:x2]
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (self.kernel_size, self.kernel_size), 0)
            elif self.mode == "pixelate":
                roi = out[y1:y2, x1:x2]
                bw, bh = max(4, (x2 - x1) // 10), max(4, (y2 - y1) // 10)
                small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
                out[y1:y2, x1:x2] = cv2.resize(
                    small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
                )
            elif self.mode == "face-blur":
                # Blur upper head/face ellipse (top 30% of person bounding box)
                head_h = max(8, int((y2 - y1) * 0.30))
                fy2 = min(h, y1 + head_h)
                roi = out[y1:fy2, x1:x2]
                blurred = cv2.GaussianBlur(roi, (self.kernel_size, self.kernel_size), 0)
                mask = np.zeros((fy2 - y1, x2 - x1), dtype=np.uint8)
                cv2.ellipse(
                    mask,
                    ((x2 - x1) // 2, (fy2 - y1) // 2),
                    (max(2, int((x2 - x1) * 0.42)), max(2, int((fy2 - y1) * 0.48))),
                    0,
                    0,
                    360,
                    255,
                    -1,
                )
                out[y1:fy2, x1:x2] = np.where(mask[:, :, None] > 0, blurred, roi)

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
