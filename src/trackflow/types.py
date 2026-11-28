"""Core data models shared across detectors, trackers, Re-ID, spatial analytics, and sinks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Detection:
    """Single object detection in pixel coordinates (x1, y1, x2, y2)."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center ground contact point used for floorplan homography."""
        return ((self.bbox[0] + self.bbox[2]) * 0.5, self.bbox[3])


@dataclass(slots=True)
class Track:
    """Active tracked object at a specific frame."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    age: int = 1
    hits: int = 1
    recovered: bool = False
    coasting: bool = False  # predicted position of a briefly-missed track (display only)
    query_score: float | None = None
    query_match: bool = True
    world_xy: tuple[float, float] | None = None
    zone: str | None = None
    reid_merged_from: int | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def foot_point(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) * 0.5, self.bbox[3])

    def to_record(self, frame_idx: int, timestamp_s: float) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        return {
            "frame_idx": int(frame_idx),
            "timestamp_s": round(float(timestamp_s), 4),
            "track_id": int(self.track_id),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 4),
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
            "width": round(float(x2 - x1), 2),
            "height": round(float(y2 - y1), 2),
            "age": int(self.age),
            "hits": int(self.hits),
            "recovered": bool(self.recovered),
            "query_score": None if self.query_score is None else round(float(self.query_score), 4),
            "query_match": bool(self.query_match),
            "world_x": None if self.world_xy is None else round(float(self.world_xy[0]), 3),
            "world_y": None if self.world_xy is None else round(float(self.world_xy[1]), 3),
            "zone": self.zone,
        }


@dataclass(slots=True)
class FrameResult:
    """Per-frame tracking output and timing telemetry."""

    frame_idx: int
    timestamp_s: float
    tracks: list[Track]
    num_detections: int
    inference_ms: float
    tracking_ms: float
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TrackCard:
    """Persistent entity summary for a tracked person/object across its lifetime."""

    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    first_timestamp_s: float
    last_timestamp_s: float
    frame_count: int
    best_confidence: float
    best_frame: int
    best_bbox: tuple[float, float, float, float]
    thumbnail_relpath: str | None = None
    embedding: list[float] | None = None
    query_score: float | None = None
    query_match: bool = True
    merged_track_ids: list[int] = field(default_factory=list)
    zones_visited: list[str] = field(default_factory=list)
    dwell_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.last_timestamp_s - self.first_timestamp_s)

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_timestamp_s": round(self.first_timestamp_s, 3),
            "last_timestamp_s": round(self.last_timestamp_s, 3),
            "duration_s": round(self.duration_s, 3),
            "frame_count": self.frame_count,
            "best_confidence": round(self.best_confidence, 4),
            "best_frame": self.best_frame,
            "best_bbox": [round(float(v), 2) for v in self.best_bbox],
            "thumbnail": self.thumbnail_relpath,
            "query_score": None if self.query_score is None else round(self.query_score, 4),
            "query_match": self.query_match,
            "merged_track_ids": list(self.merged_track_ids),
            "zones_visited": list(self.zones_visited),
            "dwell_seconds": {k: round(v, 3) for k, v in self.dwell_seconds.items()},
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding
        return data


@dataclass(slots=True)
class HomographyPoint:
    """A 2D image-to-floorplan calibration correspondence."""

    image_xy: tuple[float, float]
    world_xy: tuple[float, float]


@dataclass(slots=True)
class RunSummary:
    """Aggregate summary of a completed Trackflow run."""

    source: str
    model: str
    tracker: str
    device: str
    total_frames: int
    processed_frames: int
    source_fps: float
    avg_fps: float
    avg_inference_ms: float
    avg_tracking_ms: float
    total_detections: int
    total_track_observations: int
    unique_tracks: int
    recovered_tracks: int
    reid_merges: int
    elapsed_s: float
    frame_width: int
    frame_height: int
    classes: list[str]
    query: str | None = None
    anonymize_mode: str = "none"
    output_files: dict[str, str] = field(default_factory=dict)
    zone_dwell_totals_s: dict[str, float] = field(default_factory=dict)
    line_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    frame_stride: int = 1
    model_revision: str | None = None
    precision: str | None = None
    tracking_threshold: float | None = None
    mask_fallback: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "model": self.model,
            "model_revision": self.model_revision,
            "precision": self.precision,
            "tracking_threshold": self.tracking_threshold,
            "tracker": self.tracker,
            "device": self.device,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "source_fps": round(self.source_fps, 2),
            "frame_stride": self.frame_stride,
            "avg_fps": round(self.avg_fps, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "avg_tracking_ms": round(self.avg_tracking_ms, 2),
            "total_detections": self.total_detections,
            "total_track_observations": self.total_track_observations,
            "unique_tracks": self.unique_tracks,
            "recovered_tracks": self.recovered_tracks,
            "reid_merges": self.reid_merges,
            "elapsed_s": round(self.elapsed_s, 3),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "classes": self.classes,
            "query": self.query,
            "anonymize_mode": self.anonymize_mode,
            "output_files": self.output_files,
            "zone_dwell_totals_s": {k: round(v, 2) for k, v in self.zone_dwell_totals_s.items()},
            "line_counts": self.line_counts,
        }
        if self.mask_fallback is not None:
            out["mask_fallback"] = self.mask_fallback
        return out
