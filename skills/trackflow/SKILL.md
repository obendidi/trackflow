---
name: trackflow
description: Run local Apache-2.0 people and object tracking on video, image folders, or streams with Trackflow (ONNX detectors + ByteTrack/BoT-SORT/OC-SORT/BoostTrack), export Parquet/JSONL/MOTChallenge, count line crossings, search tracks by text, project onto a floorplan, or anonymize footage. Use when the user asks to track objects/people in video, count crossings, search video tracks, or inspect tracking runs locally.
---

# Trackflow

Local people and object tracking via `trackflow-python` (Apache-2.0 CLI, Python streaming API, and WebGPU browser demo at `https://obendidi.github.io/trackflow/`).

## Quickstart

Run without installing via `uvx`, or install once with `pip`/`uv`:

```bash
uvx trackflow-python run street.mp4
# or
pip install trackflow-python
trackflow run street.mp4
trackflow inspect          # opens local web UI on the latest run
```

- Each run writes to a fresh directory (`runs/run-1`, `runs/run-2`, ...). `-o <dir>` must be empty so runs never mix.
- Pass `--no-video` for batch analytics / exports when you do not need an annotated MP4.
- Sources accepted: video file, image folder (natural sort; pass `--fps`), RTSP/HTTP URL, or camera index (`0`).

## Choosing Detector, Tracker, and Device

Check available presets anytime with `trackflow models --json` and local acceleration with `trackflow info`.

| Need | Flags | Notes |
| :--- | :--- | :--- |
| Best permissive default (crowded scenes) | `-m rtdetr-v2-r18` *(default)* | Apache-2.0, ~81 MB, 45.1 MOT17 val HOTA (ByteTrack) |
| Fast on Apple Silicon / WebGPU | `-m rfdetr-nano` | Apache-2.0, ~108 MB |
| Smallest download | `-m dfine-n` | Apache-2.0, ~15 MB |
| Highest preset accuracy | `-m rfdetr-large` | Apache-2.0; requires `pip install "trackflow-python[transformers]"` |
| Fastest tracker | `-t bytetrack` *(default)* | Best speed; 45.1 HOTA on MOT17 val |
| Fewest ID switches / moving camera | `-t botsort --reid osnet` | Camera motion comp + person Re-ID; 46.1 HOTA, 170 ID switches |
| Heavy occlusions | `-t ocsort` or `-t boosttrack` | Re-matches after occlusions; BoostTrack+OSNet is slower |

- **Weight licenses**: Trackflow and `rtdetr-v2-r18`, `rfdetr-*`, `dfine-n` are **Apache-2.0**. `yolo26*` presets are **AGPL-3.0** and `deimv2-*` are **non-commercial** — warn the user before using them in commercial code.
- **Devices**: `--device auto|cpu|cuda|coreml|tensorrt` (ONNX) or `mps` (`hf:` Transformers). On Apple Silicon, `--device coreml` runs `rtdetr-v2-r18` at ~CPU speed with ~1/3 the CPU core load; `rfdetr-*` is faster on `cpu`.

## Common Workflows

### 1. Export tracks to Parquet and query with DuckDB

```bash
trackflow run crowd.mp4 --tracker botsort --reid osnet \
  --classes person,backpack --export parquet,jsonl,csv,mot,html --no-video
```

```sql
SELECT track_id, COUNT(*) AS frames, AVG(confidence) AS avg_conf
FROM 'runs/run-1/tracks.parquet'
GROUP BY track_id ORDER BY frames DESC;
```

### 2. Count directional line crossings

```bash
trackflow run entrance.mp4 --line "door=640,300,640,720"
```

- Format: `[name=]x1,y1,x2,y2` in image pixels (repeatable).
- Crossing to the right of `(x1,y1) -> (x2,y2)` counts as `in`, opposite as `out`, after a `0.3 s` hold to filter box jitter. Place lines where feet remain visible.
- Totals land in `summary.json` (`line_counts`), HTML report, and `events.json` (`line_crossed`).

### 3. Natural-language track search (and Re-ID caveat)

```bash
# Compute CLIP/SigLIP embeddings during the run (--vlm siglip for finer clothing attributes)
trackflow run mall.mp4 --query "person in red jacket with a backpack" --vlm siglip

# Re-query the saved run later without re-running detection
trackflow search runs/run-1 "person carrying a suitcase"
```

- **Keep search and identity separate**: CLIP/SigLIP match text to track crops and are never used for track IDs. Use `--reid osnet` (and opt-in `--reid-merge` for re-entry linking) for appearance identity.

### 4. Visual debugging, comparison, and floorplan homography

```bash
trackflow compare crowd.mp4 --trackers bytetrack,botsort --inspect
trackflow inspect runs/run-1          # jump to ID switches, search cards, 4-point floorplan
```

### 5. Anonymization

```bash
trackflow run lobby.mp4 --anonymize face-blur   # top 30% of each person box (box heuristic, not a face detector)
trackflow run lobby.mp4 --anonymize blur        # or pixelate / skeleton (no source pixels saved)
```

- Anonymized runs suppress saving person crops/thumbnails. Recommend `blur` or `pixelate` over `face-blur` when subjects bend, sit, or are partially cropped.

### 6. Python streaming API (no disk writes)

```python
import cv2
from trackflow import TrackflowPipeline
from trackflow.lines import CountingLine

pipeline = TrackflowPipeline(
    model="rfdetr-nano",
    tracker="botsort",
    lines=[CountingLine("door", (640, 300), (640, 720))],
)
cap = cv2.VideoCapture("rtsp://camera.local/stream")

def frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame  # BGR uint8

for result in pipeline.stream(frames(), fps=25):
    for track in result.tracks:
        print(result.frame_idx, track.track_id, track.class_name, track.bbox)
```

## Boundaries

- Single video/stream per process — not a multi-camera VMS or alert server.
- FPS and track counts do not prove ID accuracy; for accuracy evaluation, export `--export mot` and score with TrackEval (or run `scripts/eval_mot17.py`).
