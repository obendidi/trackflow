"""Trackflow — Modern real-time people & object tracking toolkit."""

from __future__ import annotations

from trackflow.boosttrack import BoostTracker
from trackflow.detector import Detector, OnnxDetrDetector, create_detector
from trackflow.homography import DwellZone, FloorplanHomography
from trackflow.lines import CountingLine
from trackflow.models import MODEL_REGISTRY, REID_REGISTRY, VLM_REGISTRY, ModelSpec
from trackflow.ocsort import OcSortTracker
from trackflow.pipeline import TrackflowPipeline
from trackflow.privacy import PrivacyAnonymizer
from trackflow.reid import ReidEmbedder, TrackCardIndex, VlmEmbedder
from trackflow.tracker import (
    TRACKER_NAMES,
    BotSortTracker,
    ByteTracker,
    Tracker,
    create_tracker,
)
from trackflow.types import (
    Detection,
    FrameResult,
    HomographyPoint,
    RunSummary,
    Track,
    TrackCard,
)

__version__ = "0.4.3"

__all__ = [
    "__version__",
    "BoostTracker",
    "BotSortTracker",
    "ByteTracker",
    "CountingLine",
    "Detection",
    "Detector",
    "DwellZone",
    "FloorplanHomography",
    "FrameResult",
    "HomographyPoint",
    "MODEL_REGISTRY",
    "ModelSpec",
    "OcSortTracker",
    "OnnxDetrDetector",
    "PrivacyAnonymizer",
    "REID_REGISTRY",
    "ReidEmbedder",
    "RunSummary",
    "Track",
    "TrackCard",
    "TRACKER_NAMES",
    "TrackCardIndex",
    "TrackflowPipeline",
    "Tracker",
    "VLM_REGISTRY",
    "VlmEmbedder",
    "create_detector",
    "create_tracker",
]
