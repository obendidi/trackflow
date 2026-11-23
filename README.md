# Trackflow

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/obendidi/trackflow/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://pypi.org/project/trackflow-python/)

Real-time people and object tracking in Python and in the browser. Runs ONNX detection transformers (RT-DETRv2, RF-DETR, D-FINE) with ByteTrack or BoT-SORT, exports structured track data (Parquet, JSONL, CSV, HTML), and includes text-based track search, a local run inspector with 4-point floorplan projection, and anonymization.

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
```

Model weights download from Hugging Face on first run and are cached locally. ONNX Runtime selects CoreML, CUDA, or CPU automatically (`--device` to override).

## Usage

### Tracking and exports

```bash
trackflow run crowd.mp4 \
  --model dfine-n \
  --tracker botsort --reid osnet \
  --classes person \
  --export parquet,jsonl,csv,html
```

The Parquet export can be queried directly with DuckDB:

```sql
SELECT track_id, COUNT(*) AS frames, AVG(confidence) AS avg_conf
FROM 'runs/latest/tracks.parquet'
GROUP BY track_id ORDER BY frames DESC;
```

### Text search and re-identification

```bash
# Highlight matching people live (uses CLIP by default; --vlm siglip for finer attributes)
trackflow run mall.mp4 --query "person in red jacket with a backpack"

# Search a completed run later without re-running detection
# (requires a run created with --query or --vlm clip|siglip)
trackflow search runs/latest "person carrying a suitcase"

# Give people who leave and come back their earlier ID (opt-in, uses person Re-ID)
trackflow run mall.mp4 --reid osnet --reid-merge
```

CLIP and SigLIP match images to *text*; they cannot reliably distinguish two people in similar clothes, so they are never used for identity. `--reid osnet` runs a dedicated person re-identification model. Re-entry merging compares a new track against people who left *before* it appeared and merges only when one earlier person is clearly the best match; look-alikes stay separate. It can still make mistakes on low-resolution footage, which is why it is off by default.

### Inspector, tracker comparison, and floorplan projection

```bash
# Compare ByteTrack and BoT-SORT on the same video and open the inspector
trackflow compare crowd.mp4 --trackers bytetrack,botsort --inspect

# Open the inspector on any existing run
trackflow inspect runs/latest
```

`trackflow inspect` starts a local web UI where you can scrub through frames, jump to flagged ID switches or recoveries, search track cards, compare two runs, and click 4 ground corners to project 2D tracks onto a top-down floorplan.

### Anonymization

```bash
# Blur faces/heads
trackflow run lobby.mp4 --anonymize face-blur

# Draw stick figures on a blank canvas (box-based, no pose model; no video pixels saved)
trackflow run lobby.mp4 --anonymize skeleton
```

Available modes are `face-blur`, `blur`, `pixelate`, and `skeleton`. Anonymization covers every detected person box (including low-confidence detections and people briefly missed by the tracker), and anonymized runs do not write person crops or thumbnails. It is best effort: if the detector misses someone in a frame, that person is not blurred in that frame.

### Benchmarking

```bash
trackflow benchmark crowd.mp4 --models dfine-n,rtdetr-v2-r18 --trackers bytetrack,botsort --split-screen
```

Runs the selected detector/tracker pairs on a video (or a generated synthetic clip if no video is given), writes a Markdown and JSON report, and with `--split-screen` outputs a side-by-side comparison video against the repo's 2017 baseline.

## Browser demo

The live demo at [obendidi.github.io/trackflow](https://obendidi.github.io/trackflow/) runs locally in WebGPU or WASM — video stays on your machine.

The [`web/`](https://github.com/obendidi/trackflow/tree/main/web) directory runs the same `onnx-community` ONNX models (RF-DETR Nano by default) through `@huggingface/transformers`, using ports of the Python detection post-processing and ByteTrack (`web/detector.js` and `web/tracker.js`, checked against Python by `tests/test_web_parity.py`), along with live anonymization. To run it locally:

```bash
python3 -m http.server 8080 --directory web
```

## Models

Weights download automatically from Hugging Face on first use (`trackflow models` lists all presets and repo IDs):

| Preset / Flag | Model | Size | License | Notes |
| :--- | :--- | ---: | :--- | :--- |
| `rtdetr-v2-r18` *(default)* | RT-DETRv2 ResNet-18 (640×640) | ~81 MB | Apache-2.0 | Good default balance for crowded scenes |
| `rfdetr-nano` | RF-DETR Nano (384×384) | ~108 MB | Apache-2.0 | Fast on Apple Silicon and WebGPU |
| `rfdetr-base` | RF-DETR Base (560×560) | ~108 MB | Apache-2.0 | Higher accuracy, heavier compute |
| `dfine-n` | D-FINE Nano (640×640) | ~15 MB | Apache-2.0 | Smallest download |
| `--reid osnet` | OSNet-AIN x1.0 person Re-ID | ~9 MB | MIT code; weights trained on MSMT17 (check dataset terms for commercial use) | BoT-SORT appearance and `--reid-merge` |
| `--vlm clip` | CLIP ViT-B/32 (INT8) | ~153 MB | MIT | Default for `--query` and `search` |
| `--vlm siglip` | SigLIP Base (INT8) | ~205 MB | Apache-2.0 | Better on fine clothing/attribute queries |

## Python API

```python
from trackflow import FloorplanHomography, TrackflowPipeline

pipeline = TrackflowPipeline(
    model="dfine-n",
    tracker="botsort",
    reid="osnet",                         # person Re-ID for BoT-SORT
    query="person wearing a red jacket",  # scored with CLIP (vlm="clip" by default when query is set)
    anonymize="face-blur",
)
summary = pipeline.run("crowd.mp4", output_dir="runs/crowd")
print(f"{summary.unique_tracks} unique tracks at {summary.avg_fps:.1f} FPS")
```

## License

Apache-2.0 — see [LICENSE](https://github.com/obendidi/trackflow/blob/main/LICENSE).

The original 2017 `Tracking-with-darkflow` codebase (YOLOv2 + DeepSORT + TensorFlow 1.0) is kept on the [`legacy-2017`](https://github.com/obendidi/trackflow/tree/legacy-2017) branch and [`v1.0-legacy-2017`](https://github.com/obendidi/trackflow/releases/tag/v1.0-legacy-2017) tag under GPL-3.0.
