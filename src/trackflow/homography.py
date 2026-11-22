"""4-point floorplan homography calibration, bird's-eye projection, heatmaps, and dwell zones."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class DwellZone:
    """Named polygon in top-down floorplan coordinates."""

    name: str
    polygon: list[tuple[float, float]]
    color: str = "#38bdf8"

    def contains(self, x: float, y: float) -> bool:
        pts = np.array(self.polygon, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in self.polygon],
            "color": self.color,
        }


@dataclass
class FloorplanHomography:
    """Maps 2D camera ground points onto a top-down bird's-eye floorplan via 4-point homography."""

    image_points: list[tuple[float, float]]
    world_points: list[tuple[float, float]]
    width_m: float = 10.0
    height_m: float = 10.0
    zones: list[DwellZone] = field(default_factory=list)
    _matrix: np.ndarray = field(init=False, repr=False)
    _heatmap: np.ndarray = field(init=False, repr=False)
    _trajectories: dict[int, list[tuple[float, float]]] = field(default_factory=dict, repr=False)
    zone_dwell_totals_s: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.image_points) != 4 or len(self.world_points) != 4:
            raise ValueError(
                "FloorplanHomography requires exactly 4 image_points and 4 world_points."
            )
        src = np.array(self.image_points, dtype=np.float32)
        dst = np.array(self.world_points, dtype=np.float32)
        self._matrix = cv2.getPerspectiveTransform(src, dst)
        self._heatmap = np.zeros((100, 100), dtype=np.float32)
        for z in self.zones:
            self.zone_dwell_totals_s.setdefault(z.name, 0.0)

    @classmethod
    def from_json(cls, path: str | Path) -> FloorplanHomography:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        img_pts = [(float(p[0]), float(p[1])) for p in data["image_points"]]
        wld_pts = [(float(p[0]), float(p[1])) for p in data["world_points"]]
        zones = [
            DwellZone(
                name=str(z["name"]),
                polygon=[(float(p[0]), float(p[1])) for p in z["polygon"]],
                color=str(z.get("color", "#38bdf8")),
            )
            for z in data.get("zones", [])
        ]
        width_m = float(data.get("width_m", max(p[0] for p in wld_pts) or 10.0))
        height_m = float(data.get("height_m", max(p[1] for p in wld_pts) or 10.0))
        return cls(
            image_points=img_pts,
            world_points=wld_pts,
            width_m=width_m,
            height_m=height_m,
            zones=zones,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_points": [
                [round(float(x), 2), round(float(y), 2)] for x, y in self.image_points
            ],
            "world_points": [
                [round(float(x), 3), round(float(y), 3)] for x, y in self.world_points
            ],
            "width_m": round(float(self.width_m), 3),
            "height_m": round(float(self.height_m), 3),
            "zones": [z.to_dict() for z in self.zones],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def project(self, image_xy: tuple[float, float]) -> tuple[float, float]:
        """Project a single 2D image pixel point (e.g. foot point) into floorplan (x, y)."""
        pt = np.array([[[float(image_xy[0]), float(image_xy[1])]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._matrix)[0, 0]
        return (float(out[0]), float(out[1]))

    def locate_zone(self, world_xy: tuple[float, float]) -> str | None:
        for zone in self.zones:
            if zone.contains(world_xy[0], world_xy[1]):
                return zone.name
        return None

    def observe(
        self,
        track_id: int,
        foot_xy: tuple[float, float],
        dt_s: float = 0.0,
    ) -> tuple[tuple[float, float], str | None]:
        """Project a track observation, accumulate heatmap & dwell metrics, and return (world_xy, zone)."""
        wx, wy = self.project(foot_xy)
        self._trajectories.setdefault(track_id, []).append((wx, wy))

        gx = int(
            np.clip(
                (wx / max(self.width_m, 1e-3)) * (self._heatmap.shape[1] - 1),
                0,
                self._heatmap.shape[1] - 1,
            )
        )
        gy = int(
            np.clip(
                (wy / max(self.height_m, 1e-3)) * (self._heatmap.shape[0] - 1),
                0,
                self._heatmap.shape[0] - 1,
            )
        )
        self._heatmap[gy, gx] += 1.0

        zone = self.locate_zone((wx, wy))
        if zone is not None:
            self.zone_dwell_totals_s[zone] = self.zone_dwell_totals_s.get(zone, 0.0) + max(
                0.0, dt_s
            )
        return (wx, wy), zone

    def render_floorplan(
        self, out_path: str | Path, canvas_size: tuple[int, int] = (800, 600)
    ) -> Path:
        """Render a high-contrast bird's-eye floorplan PNG with heatmap, dwell zones, and trajectories."""
        cw, ch = canvas_size
        canvas = np.full((ch, cw, 3), (15, 20, 30), dtype=np.uint8)

        # Draw grid lines (1-meter spacing)
        for mx in range(int(self.width_m) + 1):
            px = int((mx / max(self.width_m, 1e-3)) * (cw - 1))
            cv2.line(canvas, (px, 0), (px, ch), (30, 38, 54), 1)
        for my in range(int(self.height_m) + 1):
            py = int((my / max(self.height_m, 1e-3)) * (ch - 1))
            cv2.line(canvas, (0, py), (cw, py), (30, 38, 54), 1)

        # Overlay smoothed density heatmap
        if float(self._heatmap.max()) > 0:
            heat = cv2.GaussianBlur(self._heatmap, (11, 11), 0)
            norm = np.clip((heat / max(float(heat.max()), 1e-6)) * 255.0, 0, 255).astype(np.uint8)
            resized = cv2.resize(norm, (cw, ch), interpolation=cv2.INTER_CUBIC)
            colored = cv2.applyColorMap(resized, cv2.COLORMAP_INFERNO)
            mask = (resized > 15).astype(np.float32)[:, :, None] * 0.55
            canvas = np.clip(canvas * (1.0 - mask) + colored * mask, 0, 255).astype(np.uint8)

        def to_px(wx: float, wy: float) -> tuple[int, int]:
            x = int(np.clip((wx / max(self.width_m, 1e-3)) * (cw - 1), 0, cw - 1))
            y = int(np.clip((wy / max(self.height_m, 1e-3)) * (ch - 1), 0, ch - 1))
            return (x, y)

        # Draw dwell zones
        for zone in self.zones:
            pts = np.array([to_px(x, y) for x, y in zone.polygon], dtype=np.int32)
            if len(pts) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], (60, 120, 180))
                canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
                cv2.polylines(canvas, [pts], True, (110, 210, 255), 2, cv2.LINE_AA)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                dwell = self.zone_dwell_totals_s.get(zone.name, 0.0)
                cv2.putText(
                    canvas,
                    f"{zone.name} ({dwell:.1f}s)",
                    (max(6, cx - 40), max(20, cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (230, 245, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Draw trajectories
        palette = [
            (92, 214, 148),
            (250, 173, 72),
            (140, 130, 255),
            (110, 235, 255),
            (255, 130, 210),
        ]
        for tid, pts in self._trajectories.items():
            if len(pts) < 2:
                continue
            color = palette[tid % len(palette)]
            px_pts = np.array([to_px(x, y) for x, y in pts], dtype=np.int32)
            cv2.polylines(canvas, [px_pts], False, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(px_pts[-1]), 4, (255, 255, 255), -1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"Trackflow Bird's-Eye Floorplan ({self.width_m:.1f}m x {self.height_m:.1f}m)",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 235, 245),
            1,
            cv2.LINE_AA,
        )
        out = Path(out_path)
        cv2.imwrite(str(out), canvas)
        return out
