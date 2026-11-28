"""Trackflow — Modern real-time people & object tracking toolkit."""

from __future__ import annotations

from trackflow.boosttrack import BoostTracker
from trackflow.detector import Detector, OnnxDetrDetector, create_detector
from trackflow.homography import DwellZone, FloorplanHomography
from trackflow.lines import CountingLine
from trackflow.mask_fallback import (
    MaskFallbackConfig,
    MaskFallbackResolver,
    MaskFallbackStats,
    MaskPropagator,
    Sam3MaskPropagator,
)
from trackflow.models import (
    MASK_REGISTRY,
    MODEL_REGISTRY,
    REID_REGISTRY,
    VLM_REGISTRY,
    MaskModelSpec,
    ModelSpec,
)
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

__version__ = "0.6.0"

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
    "MASK_REGISTRY",
    "MODEL_REGISTRY",
    "MaskFallbackConfig",
    "MaskFallbackResolver",
    "MaskFallbackStats",
    "MaskModelSpec",
    "MaskPropagator",
    "ModelSpec",
    "OcSortTracker",
    "OnnxDetrDetector",
    "PrivacyAnonymizer",
    "REID_REGISTRY",
    "ReidEmbedder",
    "RunSummary",
    "Sam3MaskPropagator",
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
