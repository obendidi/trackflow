# Migrating from AWS Rekognition Person Tracking & People Pathing to Trackflow

AWS Rekognition shut down **People Pathing** on **2025-10-31** and moved **Streaming Video Analysis** to maintenance (closed to new accounts on **2026-04-30**). Stored `StartPersonTracking` / `GetPersonTracking` bills at **$0.10/minute ($6.00/video-hour)** with a 1-minute minimum per video.

If you need persistent person IDs, bounding boxes, trajectories, line-crossing counts, or floorplan coordinates without uploading footage to S3, [Trackflow](../README.md) runs locally under **Apache-2.0** (code and default ONNX DETR weights) on CPU, CUDA, Apple Silicon CoreML, or TensorRT.

## 1. One-command CLI replacement

```bash
uvx trackflow-python run footage.mp4 \
  --classes person \
  --tracker botsort --reid osnet \
  --export parquet,jsonl
```

Add `--line "entrance=640,300,640,720"` for directional foot-traffic counting, `--anonymize face-blur` (or `blur` / `pixelate` / `skeleton`) to mask people in exported video, or open `trackflow inspect` to click 4 ground corners and project paths onto a top-down floorplan.

## 2. Python API: Before and After

**Before (`boto3` `StartPersonTracking` + `GetPersonTracking`):**

```python
import boto3

rek = boto3.client("rekognition")
job = rek.start_person_tracking(
    Video={"S3Object": {"Bucket": "my-bucket", "Name": "footage.mp4"}},
    NotificationChannel={"SNSTopicArn": TOPIC_ARN, "RoleArn": ROLE_ARN},
)
# Wait for SNS completion notification, then paginate GetPersonTracking:
resp = rek.get_person_tracking(JobId=job["JobId"])
for item in resp["Persons"]:
    person = item["Person"]
    print(item["Timestamp"], person["Index"], person.get("BoundingBox"))
```

**After (`trackflow-python` — stored video to Parquet/JSONL):**

```python
from trackflow import TrackflowPipeline

pipeline = TrackflowPipeline(
    model="rtdetr-v2-r18",   # Apache-2.0 ONNX default
    classes=["person"],
    tracker="botsort",
    reid="osnet",            # body/clothing Re-ID to cut ID switches
)
summary = pipeline.run("footage.mp4", exports=("parquet", "jsonl"))
print(summary.output_files["parquet"], summary.unique_tracks)
```

**After (`trackflow-python` — RTSP or OpenCV stream, nothing written to disk):**

```python
import cv2
from trackflow import TrackflowPipeline

pipeline = TrackflowPipeline(
    model="rfdetr-nano",
    classes=["person"],
    tracker="botsort",
)
cap = cv2.VideoCapture("rtsp://camera.local/stream")

def frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame

for res in pipeline.stream(frames(), fps=25):
    for t in res.tracks:
        print(int(res.timestamp_s * 1000), t.track_id, t.bbox, t.confidence)
```

## 3. Field Mapping (`GetPersonTracking` → `tracks.parquet` / `tracks.jsonl`)

| AWS Rekognition (`GetPersonTracking`) | Trackflow (`tracks.parquet` / `tracks.jsonl`) | Notes |
| :--- | :--- | :--- |
| `Persons[].Timestamp` (ms) | `timestamp_s` (`float64`, seconds) + `frame_idx` | Multiply `timestamp_s * 1000` for ms |
| `Persons[].Person.Index` | `track_id` (`int64`) | Persistent person track ID |
| `Persons[].Person.BoundingBox.{Left,Top,Width,Height}` | `x1`, `y1`, `x2`, `y2`, `width`, `height` (`float64`) | Pixels; divide by frame width/height for `[0, 1]` |
| `Persons[].Person.BoundingBox` confidence (`0–100`) | `confidence` (`float64`, `0.0–1.0`) | Multiply by `100` if downstream expects percent |
| People Pathing trajectory / zones | `tracks.parquet` + `world_x`, `world_y`, `zone` | Floorplan homography via `trackflow inspect` or `--homography` |
| Entry/exit counts | `summary.json` (`line_counts`) + `events.json` | Via `--line "name=x1,y1,x2,y2"` |

Query per-person dwell time and paths directly with DuckDB:

```sql
SELECT
  track_id,
  MIN(timestamp_s) AS first_seen_s,
  MAX(timestamp_s) AS last_seen_s,
  ROUND(MAX(timestamp_s) - MIN(timestamp_s), 2) AS dwell_s,
  COUNT(*) AS observations
FROM 'runs/run-1/tracks.parquet'
WHERE class_name = 'person'
GROUP BY track_id
ORDER BY dwell_s DESC;
```

## 4. Cost & Privacy Comparison

- **Cost**: 100 hours/month of stored video via `StartPersonTracking` is **$600/month** at list price ($0.10/min). Trackflow runs locally with $0 per-minute API charges.
- **Privacy**: Footage stays on local disks or private infrastructure — no S3 upload required. Optional `--anonymize face-blur|blur|pixelate|skeleton` masks detected people in exported videos and disables saving person crops.

## 5. What Does Not Map 1:1

- **No face recognition or biometric matching**: Rekognition's `Person.Face` attributes, `FaceSearch`, and celebrity matching are intentionally out of scope. `--reid osnet` matches coarse body/clothing appearance only (check MSMT17 weight terms for commercial use; `--tracker botsort` without `--reid` is pure Apache-2.0).
- **No managed S3/SNS or Kinesis Video Streams queue**: Trackflow runs synchronously on local files, image folders, cameras, or RTSP/HTTP URLs. Wrap it in your existing batch runner (e.g., ECS/Batch/Modal/Airflow) for S3 queues.
- **Single source per process**: Not a multi-camera VMS or alerting server. For 30+ concurrent live streams, use DeepStream, Frigate, or Roboflow Inference.
