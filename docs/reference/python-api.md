# Python API

Trackflow ships inline type annotations (`py.typed`) and exposes both a high-level batch/streaming pipeline (`TrackflowPipeline`) and every underlying detector, tracker, Re-ID, VLM, and analytics building block.

## Quick examples

### Batch run with exports

```python
from trackflow import CountingLine, TrackflowPipeline

pipeline = TrackflowPipeline(
    model="rtdetr-v2-r18",
    tracker="botsort",
    reid="osnet",
    classes=["person"],
    lines=[CountingLine("door", (640, 300), (640, 720))],
    anonymize="face-blur",
)
summary = pipeline.run("crowd.mp4", exports=("parquet", "jsonl"))
print(f"{summary.unique_tracks} tracks at {summary.avg_fps:.1f} FPS -> {summary.output_files['parquet']}")
```

### Zero-disk in-memory streaming

```python
import cv2
from trackflow import TrackflowPipeline

pipeline = TrackflowPipeline(model="rfdetr-nano", tracker="bytetrack")
cap = cv2.VideoCapture("rtsp://camera.local/stream")

def frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame  # BGR uint8

for result in pipeline.stream(frames(), fps=25.0):
    for track in result.tracks:
        print(result.frame_idx, result.timestamp_s, track.track_id, track.class_name, track.bbox)
```

---

## Pipeline

::: trackflow.TrackflowPipeline

---

## Detectors

::: trackflow.create_detector

::: trackflow.Detector

::: trackflow.OnnxDetrDetector

::: trackflow.ModelSpec

---

## Trackers

::: trackflow.create_tracker

::: trackflow.Tracker

::: trackflow.ByteTracker

::: trackflow.BotSortTracker

::: trackflow.OcSortTracker

::: trackflow.BoostTracker

---

## Re-ID & Text Search

::: trackflow.ReidEmbedder

::: trackflow.VlmEmbedder

::: trackflow.TrackCardIndex

---

## Analytics & Privacy

::: trackflow.CountingLine

::: trackflow.FloorplanHomography

::: trackflow.DwellZone

::: trackflow.PrivacyAnonymizer

---

## Data Types

::: trackflow.Detection

::: trackflow.Track

::: trackflow.FrameResult

::: trackflow.RunSummary

::: trackflow.TrackCard

::: trackflow.HomographyPoint
