# Migrating from Google Cloud Video Intelligence Object Tracking to Trackflow

Google Cloud Video Intelligence was deprecated on **2026-09-14** with full shutdown scheduled for **2027-09-14**. At list price, `OBJECT_TRACKING` costs **$0.15/minute ($9.00/video-hour)** for stored video and **$0.17/minute ($10.20/video-hour)** for streaming (~$7,344/month per 24/7 camera over 30 days).

If your pipeline only needed persistent per-object bounding boxes (`Feature.OBJECT_TRACKING`) written to a table or JSON stream, [Trackflow](index.md) replaces it locally on CPU, CUDA, Apple Silicon CoreML, or TensorRT under an **Apache-2.0** license (both code and default ONNX DETR weights).

## 1. One-command CLI replacement

```bash
uvx trackflow-python run footage.mp4 \
  --tracker botsort --reid osnet \
  --export parquet,jsonl
```

Or install permanently with `pip install trackflow-python` (`uv add trackflow-python`). Weights download once from Hugging Face and stay cached locally; video never leaves your machine.

## 2. Python API: Before and After

**Before (`google-cloud-videointelligence`):**

```python
from google.cloud import videointelligence

client = videointelligence.VideoIntelligenceServiceClient()
op = client.annotate_video(
    request={
        "features": [videointelligence.Feature.OBJECT_TRACKING],
        "input_uri": "gs://my-bucket/footage.mp4",
    }
)
result = op.result(timeout=600)
for ann in result.annotation_results[0].object_annotations:
    for frame in ann.frames:
        box = frame.normalized_bounding_box
        print(ann.track_id, ann.entity.description, frame.time_offset, box)
```

**After (`trackflow-python` — file batch):**

```python
from trackflow import TrackflowPipeline

pipeline = TrackflowPipeline(
    model="rtdetr-v2-r18",   # Apache-2.0 ONNX default (or "rfdetr-medium")
    tracker="botsort",
    reid="osnet",
)
summary = pipeline.run("footage.mp4", exports=("parquet", "jsonl"))
print(summary.output_files["parquet"], summary.unique_tracks)
```

**After (`trackflow-python` — in-memory frame stream, no files written):**

```python
import cv2
from trackflow import TrackflowPipeline

pipeline = TrackflowPipeline(model="rtdetr-v2-r18", tracker="botsort")
cap = cv2.VideoCapture("footage.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

def frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame

for res in pipeline.stream(frames(), fps=fps):
    for t in res.tracks:
        x1, y1, x2, y2 = t.bbox
        norm_box = (x1 / w, y1 / h, x2 / w, y2 / h)
        print(t.track_id, t.class_name, res.timestamp_s, norm_box)
```

## 3. Field Mapping (`OBJECT_TRACKING` → `tracks.parquet` / `tracks.jsonl`)

Trackflow writes one row per tracked observation with the schema defined in `src/trackflow/sinks.py`:

| Google Video Intelligence (`ObjectTrackingAnnotation`) | Trackflow (`tracks.parquet` / `tracks.jsonl`) | Notes |
| :--- | :--- | :--- |
| `object_annotations[].track_id` | `track_id` (`int64`) | Stable integer ID per tracked object |
| `object_annotations[].entity.description` | `class_name` (`string`) | COCO class name (80 classes by default) |
| `object_annotations[].entity.entity_id` | `class_id` (`int64`) | Integer COCO class ID |
| `object_annotations[].confidence` | `confidence` (`float64`) | Per-frame detection score (`0.0–1.0`) |
| `frames[].time_offset` | `timestamp_s` (`float64`), `frame_idx` (`int64`) | Seconds since start + 0-based frame index |
| `frames[].normalized_bounding_box.{left,top,right,bottom}` | `x1`, `y1`, `x2`, `y2`, `width`, `height` (`float64`) | Pixel coordinates; divide by frame width/height for `[0, 1]` |
| `segment.{start_time_offset, end_time_offset}` | Query `MIN(timestamp_s)` / `MAX(timestamp_s)` | Aggregated per `track_id` (see SQL below) |

Reconstruct Video Intelligence-style per-track segments directly with DuckDB:

```sql
SELECT
  track_id,
  ANY_VALUE(class_name) AS entity,
  MIN(timestamp_s) AS start_time_s,
  MAX(timestamp_s) AS end_time_s,
  COUNT(*) AS frames,
  AVG(confidence) AS avg_confidence
FROM 'runs/run-1/tracks.parquet'
GROUP BY track_id
ORDER BY start_time_s;
```

## 4. Cost Comparison

| Workload | Google Video Intelligence (list) | Trackflow (local) |
| :--- | ---: | ---: |
| 100 hours/month stored footage | $900 / month | $0 API fees (local CPU/GPU) |
| 1× 24/7 stream (720 hrs/month) | ~$7,344 / month | $0 API fees (local machine) |

## 5. What Does Not Map 1:1

- **Label taxonomy**: Default ONNX presets detect the 80 COCO classes (`person`, `car`, `truck`, `bicycle`, `backpack`, etc.), not Google's broader Knowledge Graph vocabulary. For domain-specific classes, pass a custom ONNX model (`-m path/to/model.onnx`) or a Hugging Face Transformers detector (`-m hf:owner/repo` with `trackflow-python[transformers]`).
- **Non-tracking Video Intelligence features**: Trackflow does not replace shot change detection, OCR (`TEXT_DETECTION`), explicit content moderation, or speech transcription.
- **GCS ingestion & fleet orchestration**: Trackflow reads local files, image folders, cameras, or RTSP/HTTP URLs one source per process. Download from GCS first (or run workers in a batch job); for 30+ live streams with alerting, look at DeepStream, Frigate, or Roboflow Inference.
