"""Trackflow — Modern real-time people & object tracking toolkit."""

from __future__ import annotations

from trackflow.detector import Detector, OnnxDetrDetector, create_detector
from trackflow.homography import DwellZone, FloorplanHomography
from trackflow.models import MODEL_REGISTRY, REID_REGISTRY, VLM_REGISTRY, ModelSpec
from trackflow.pipeline import TrackflowPipeline
from trackflow.privacy import PrivacyAnonymizer
from trackflow.reid import ReidEmbedder, TrackCardIndex, VlmEmbedder
from trackflow.tracker import BotSortTracker, ByteTracker, Tracker, create_tracker
from trackflow.types import (
    Detection,
    FrameResult,
    HomographyPoint,
    RunSummary,
    Track,
    TrackCard,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "BotSortTracker",
    "ByteTracker",
    "Detection",
    "Detector",
    "DwellZone",
    "FloorplanHomography",
    "FrameResult",
    "HomographyPoint",
    "MODEL_REGISTRY",
    "ModelSpec",
    "OnnxDetrDetector",
    "PrivacyAnonymizer",
    "REID_REGISTRY",
    "ReidEmbedder",
    "RunSummary",
    "Track",
    "TrackCard",
    "TrackCardIndex",
    "TrackflowPipeline",
    "Tracker",
    "VLM_REGISTRY",
    "VlmEmbedder",
    "create_detector",
    "create_tracker",
]
