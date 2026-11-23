# Trackflow

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/obendidi/trackflow/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://pypi.org/project/trackflow-python/)

Real-time people and object tracking in Python and in the browser. Trackflow runs current detectors (RF-DETR, RT-DETRv2, D-FINE, DEIMv2, YOLO26) with ByteTrack, BoT-SORT, OC-SORT or BoostTrack, streams or exports track data (Parquet, JSONL, CSV, MOTChallenge, HTML), and includes text-based track search, a local run inspector with 4-point floorplan projection, and anonymization.

Try it in the browser without installing anything: [obendidi.github.io/trackflow](https://obendidi.github.io/trackflow/).

## Quickstart

Run directly with `uvx`:

```bash
uvx trackflow-python run street.mp4
```

Or install with `pip` / `uv` (provides the `trackflow` CLI and `import trackflow`):

```bash
pip install trackflow-python
trackflow run street.mp4
trackflow inspect          # opens the most recent run
```

Each run writes to a new folder (`runs/run-1`, `runs/run-2`, …). Pass `-o <dir>` to choose one yourself; it must be empty, so a new run never mixes with files from an old one.

Model weights download from Hugging Face on first run and are cached locally. Every preset is pinned to a specific model revision, so the same command gives the same weights later. ONNX Runtime picks CUDA or CPU automatically; `--device cpu|cuda|coreml|tensorrt` forces one and fails loudly if it isn't available.

## Usage

### Tracking and exports

```bash
trackflow run crowd.mp4 \
  --model rfdetr-medium \
  --tracker botsort --reid osnet \
  --classes person,backpack \
  --export parquet,jsonl,csv,mot,html
```

The source can be a video file, a folder of images (read in natural order, `frame_2` before `frame_10`; set the rate with `--fps`), an RTSP/HTTP stream URL, or a camera index.

The Parquet export can be queried directly with DuckDB:

```sql
SELECT track_id, COUNT(*) AS frames, AVG(confidence) AS avg_conf
FROM 'runs/run-1/tracks.parquet'
GROUP BY track_id ORDER BY frames DESC;
```

`--export mot` writes `tracks.txt` in MOTChallenge format, which [TrackEval](https://github.com/JonathonLuiten/TrackEval) scores directly (HOTA, IDF1, MOTA).

### Trackers

| `--tracker` | Algorithm | How it works |
| :--- | :--- | :--- |
| `bytetrack` *(default)* | [ByteTrack](https://arxiv.org/abs/2110.06864) (ECCV 2022) | Matches boxes by overlap, then uses low-confidence detections to keep tracks through partial occlusion. The fastest option. |
| `botsort` | [BoT-SORT](https://arxiv.org/abs/2206.14651) (2022) | ByteTrack plus camera-motion compensation, and appearance matching with `--reid osnet`. |
| `ocsort` | [OC-SORT](https://arxiv.org/abs/2203.14360) (CVPR 2023) | Trusts detections over motion predictions: after an occlusion it re-matches a track to its last observed box and corrects the motion model along the gap. |
| `boosttrack` | [BoostTrack++](https://arxiv.org/abs/2408.13003) (2024) | Raises the confidence of weak detections that fit a track or look like a new object, and matches with motion, shape and, with `--reid osnet`, appearance. Compensates for camera motion. |

All four keep a person and a bag in the same box as separate tracks, and only give out an ID once a track has been seen on more than one frame. OC-SORT and BoostTrack report a track again after a miss once it has matched three frames in a row; in between, the annotated video shows it as a coasting box. The browser playground uses ByteTrack.

Scores on two labeled MOT17 sequences, with the default detector on CPU:

| Tracker | MOT17-05 HOTA | IDF1 | MOTA | MOT17-09 HOTA | IDF1 | MOTA |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bytetrack` | 42.7 | 57.1 | 40.8 | 47.9 | 57.9 | 51.0 |
| `botsort` | 44.6 | 60.4 | 42.7 | 48.0 | 57.7 | 50.9 |
| `ocsort` | 43.7 | 59.2 | 46.5 | 45.7 | 54.2 | 56.6 |
| `boosttrack` | 43.7 | 59.2 | 43.9 | 45.4 | 52.8 | 53.5 |
| `botsort --reid osnet` | 45.2 | 61.5 | 43.1 | 49.0 | 60.0 | 52.7 |
| `boosttrack --reid osnet` | 45.2 | 61.7 | 43.8 | 47.0 | 55.8 | 53.8 |

No tracker wins everywhere. On MOT17-05, filmed from a moving camera at 14 FPS, every newer tracker beats ByteTrack. On MOT17-09, from a static camera, ByteTrack and BoT-SORT stay ahead on HOTA. OC-SORT has the highest MOTA on both sequences. With OSNet, BoT-SORT and BoostTrack tie on MOT17-05, and BoT-SORT is best on MOT17-09. BoostTrack with `--reid osnet` is slower than BoT-SORT because it embeds every person detection, including weak ones it might boost. Two sequences are not a benchmark, so compare trackers on your own footage with `trackflow compare` or `--export mot`.

### Text search and re-identification

```bash
# Highlight matching people live (uses CLIP by default; --vlm siglip for finer attributes)
trackflow run mall.mp4 --query "person in red jacket with a backpack"

# Search a completed run later without re-running detection
# (requires a run created with --query or --vlm clip|siglip)
trackflow search runs/run-1 "person carrying a suitcase"

# Give people who leave and come back their earlier ID (opt-in, uses person Re-ID)
trackflow run mall.mp4 --reid osnet --reid-merge
```

CLIP and SigLIP match images to *text*; they cannot reliably distinguish two people in similar clothes, so they are never used for identity. `--reid osnet` runs a dedicated person re-identification model. Re-entry merging only considers people who left before the new track appeared and stayed gone longer than the tracker's own recovery window, and merges only when one earlier person is clearly the best match; look-alikes stay separate. It can still make mistakes on low-resolution footage, which is why it is off by default.

### Inspector, tracker comparison, and floorplan projection

```bash
# Compare ByteTrack and BoT-SORT on the same video and open the inspector
trackflow compare crowd.mp4 --trackers bytetrack,botsort --inspect

# Open the inspector on any existing run
trackflow inspect runs/run-1
```

`trackflow inspect` starts a local web UI where you can scrub through frames, jump to flagged ID switches or recoveries, search track cards, compare two runs, and click 4 ground corners to project 2D tracks onto a top-down floorplan.

### Anonymization

```bash
# Blur faces/heads
trackflow run lobby.mp4 --anonymize face-blur

# Draw stick figures on a blank canvas (box-based, no pose model; no video pixels saved)
trackflow run lobby.mp4 --anonymize skeleton
```

Available modes are `face-blur`, `blur`, `pixelate`, and `skeleton`. Anonymization covers every detected person box (including low-confidence detections and people briefly missed by the tracker), even when you track other classes, and anonymized runs do not write person crops or thumbnails. It is best effort: if the detector misses someone in a frame, that person is not blurred in that frame.

### Benchmarking

```bash
trackflow benchmark crowd.mp4 --models rfdetr-nano,rtdetr-v2-r18 --trackers bytetrack,botsort --split-screen
```

Runs the selected detector/tracker pairs on a video (or a generated synthetic clip if no video is given) and writes a Markdown and JSON report. `--split-screen` also renders the first run's annotated video next to a panel with the 2017 version's published speed. The 2017 README only gives that number, not per-frame results to replay.

FPS and track counts don't tell you whether IDs are right. For accuracy, run on a labeled sequence with `--export mot` and score the result with TrackEval.

## Models

Weights download automatically from Hugging Face on first use. `trackflow models` lists every preset (`--json` for scripts).

| Preset | Model | Size (FP32) | License | Notes |
| :--- | :--- | ---: | :--- | :--- |
| `rtdetr-v2-r18` *(default)* | RT-DETRv2 ResNet-18 (640×640) | ~81 MB | Apache-2.0 | Balanced default for crowded scenes |
| `rfdetr-nano` | RF-DETR Nano (384×384) | ~108 MB | Apache-2.0 | Fast on Apple Silicon and WebGPU |
| `rfdetr-small` | RF-DETR Small (512×512) | ~115 MB | Apache-2.0 | |
| `rfdetr-medium` | RF-DETR Medium (576×576) | ~121 MB | Apache-2.0 | |
| `rfdetr-base` | RF-DETR Base (560×560) | ~108 MB | Apache-2.0 | |
| `rfdetr-large` | RF-DETR Large (704×704) | ~136 MB | Apache-2.0 | Most accurate preset; needs the `transformers` extra |
| `dfine-n` | D-FINE Nano (640×640) | ~15 MB | Apache-2.0 | Small download |
| `yolo26n` … `yolo26x` | YOLO26 Nano to XLarge (640×640) | 10–223 MB | **AGPL-3.0** | Commercial use needs an Ultralytics license |
| `deimv2-atto` | DEIMv2 Atto (320×320) | ~2 MB | **Non-commercial** | Tiny edge model, FP32 only |
| `deimv2-s` | DEIMv2 Small, DINOv3 backbone (640×640) | ~41 MB | **Non-commercial** + DINOv3 | FP32 only |

Most presets also have `--precision fp16` and `int8` variants. Trackflow itself is Apache-2.0, but each model's weights carry their own license, so check the table before shipping a product. On two labeled MOT17 sequences, `rfdetr-large` beat the default at about the same CPU speed (HOTA 44.9 vs 42.7 on MOT17-05, 51.2 vs 47.9 on MOT17-09). That's two sequences, so test on footage like yours before you switch.

You aren't limited to presets. `-m` also takes any compatible ONNX repo on Hugging Face (`owner/repo`), a local `.onnx` file, or, with the extra installed, any Transformers object-detection checkpoint as `hf:owner/repo`:

```bash
pip install "trackflow-python[transformers]"
trackflow run crowd.mp4 -m rfdetr-large
trackflow run crowd.mp4 -m hf:PekingU/rtdetr_v2_r50vd --device mps
```

Re-ID and text-search models:

| Flag | Model | Size | License | Used for |
| :--- | :--- | ---: | :--- | :--- |
| `--reid osnet` | OSNet-AIN x1.0 person Re-ID | ~9 MB | MIT code; weights trained on MSMT17 (check dataset terms for commercial use) | BoT-SORT and BoostTrack appearance, and `--reid-merge` |
| `--vlm clip` | CLIP ViT-B/32 (INT8) | ~153 MB | MIT | Default for `--query` and `search` |
| `--vlm siglip` | SigLIP Base (INT8) | ~205 MB | Apache-2.0 | Better on fine clothing/attribute queries |

## Python API

Run a video and write the same files as the CLI:

```python
from trackflow import TrackflowPipeline

pipeline = TrackflowPipeline(
    model="rfdetr-medium",
    tracker="botsort",
    reid="osnet",                         # person Re-ID for BoT-SORT
    query="person wearing a red jacket",  # scored with CLIP (vlm="clip" by default when query is set)
    anonymize="face-blur",
)
summary = pipeline.run("crowd.mp4")       # or output_dir="runs/crowd"
print(f"{summary.unique_tracks} unique tracks at {summary.avg_fps:.1f} FPS")
```

Or feed frames from your own loop and get tracks back, with nothing written to disk:

```python
import cv2
from trackflow import TrackflowPipeline

pipeline = TrackflowPipeline(model="rfdetr-nano")
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

To use your own model, pass any object with a `detect(frame_bgr) -> list[Detection]` method as `detector=`. A custom tracker goes in `tracker_factory=`, which takes any object with `update(detections, frame_bgr=None) -> list[Track]`. The built-in trackers are exported too (`ByteTracker`, `BotSortTracker`, `OcSortTracker`, `BoostTracker`), so you can tune one, for example `tracker_factory=lambda: BoostTracker(use_cmc=False)` for a fixed camera. The package ships type hints (`py.typed`), and `Detector` and `Tracker` are exported as protocols.

## License

Apache-2.0 — see [LICENSE](https://github.com/obendidi/trackflow/blob/main/LICENSE). Model weights are downloaded separately and keep their own licenses (see [Models](#models)).

The original 2017 `Tracking-with-darkflow` codebase (YOLOv2 + DeepSORT + TensorFlow 1.0) is kept on the [`legacy-2017`](https://github.com/obendidi/trackflow/tree/legacy-2017) branch and [`v1.0-legacy-2017`](https://github.com/obendidi/trackflow/releases/tag/v1.0-legacy-2017) tag under GPL-3.0.
