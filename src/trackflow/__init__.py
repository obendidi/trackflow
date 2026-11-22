"""Trackflow — Modern real-time people & object tracking toolkit."""

from __future__ import annotations

from trackflow.detector import OnnxDetrDetector
from trackflow.homography import DwellZone, FloorplanHomography
from trackflow.models import MODEL_REGISTRY, REID_REGISTRY, ModelSpec
from trackflow.pipeline import TrackflowPipeline
from trackflow.privacy import PrivacyAnonymizer
from trackflow.reid import AppearanceEmbedder, TrackCardIndex
from trackflow.tracker import BotSortTracker, ByteTracker, create_tracker
from trackflow.types import (
    Detection,
    FrameResult,
    HomographyPoint,
    RunSummary,
    Track,
    TrackCard,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AppearanceEmbedder",
    "BotSortTracker",
    "ByteTracker",
    "Detection",
    "DwellZone",
    "FloorplanHomography",
    "FrameResult",
    "HomographyPoint",
    "MODEL_REGISTRY",
    "ModelSpec",
    "OnnxDetrDetector",
    "PrivacyAnonymizer",
    "REID_REGISTRY",
    "RunSummary",
    "Track",
    "TrackCard",
    "TrackCardIndex",
    "TrackflowPipeline",
    "create_tracker",
]
