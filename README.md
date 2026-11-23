# Trackflow

**Real-time people & object tracking that goes beyond boxes and IDs.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)

> **Looking for the original 2017 `Tracking-with-darkflow` (YOLOv2 + DeepSORT + TensorFlow 1.0)?**  
> It is preserved untouched on the [`legacy-2017`](https://github.com/obendidi/trackflow/tree/legacy-2017) branch and [`v1.0-legacy-2017`](https://github.com/obendidi/trackflow/releases/tag/v1.0-legacy-2017) tag under GPL-3.0.

---

## Why Trackflow?

In July 2017, this repository wired YOLOv2 (`darkflow`) to DeepSORT using TensorFlow 1.0, OpenCV 3, Cython builds, 3 git submodules, and manual Google Drive weight downloads (~14 FPS on a GTX 1080).

Plain `YOLO + ByteTrack` is now commoditized. **Trackflow v2** is a clean-room **Apache-2.0** toolkit built for everything that happens **above** raw bounding boxes:

1. **Zero-Friction Modern Core:** One command (`uvx trackflow run video.mp4`), Apache-2.0 end-to-end transformers (**D-FINE**, **RT-DETRv2**, **RF-DETR**), **ByteTrack & BoT-SORT**, ONNX Runtime (**Apple Silicon CoreML / NVIDIA CUDA / CPU**), and data-first exports (**Parquet / JSONL / CSV / HTML report**).
2. **Natural-Language Querying & Track Cards:** Prompt-to-track (`--query "person in red jacket with a backpack"` via ONNX CLIP / SigLIP), persistent **Track Cards** with **cross-exit long-term Re-ID**, and post-run `trackflow search`.
3. **Visual Track Debugger & 4-Point Floorplan Homography:** Launch `trackflow inspect` to scrub timelines, flag ID switches, compare trackers side-by-side, and click 4 ground corners to project 2D tracks onto bird's-eye trajectories, heatmaps, and dwell zones.
4. **Privacy-First Mode & Zero-Install WebGPU Playground:** Built-in `--anonymize` (`face-blur`, `blur`, `pixelate`, or zero-PII `skeleton`-only rendering) plus a 100% local in-browser [**WebGPU Playground**](https://obendidi.github.io/trackflow/).
5. **2017 vs. 2026 Benchmark Suite:** Built-in `trackflow benchmark --split-screen` harness celebrating 9 years of real-time tracking progress.

---

## Quickstart

```bash
# Run immediately with uvx (no manual weight downloads or Cython builds)
uvx trackflow-python run street.mp4

# Or install via pip / uv (provides the `trackflow` CLI and `import trackflow`)
pip install trackflow-python
trackflow run street.mp4
```

### 1. Real-Time Tracking + Structured Exports
```bash
trackflow run crowd.mp4 \
  --model dfine-n \
  --tracker botsort \
  --classes person \
  --export parquet,jsonl,csv,html
```
Query the Parquet output directly with DuckDB:
```sql
SELECT track_id, COUNT(*) AS frames, AVG(confidence) AS avg_conf
FROM 'runs/latest/tracks.parquet'
GROUP BY track_id ORDER BY frames DESC;
```

### 2. Natural-Language Prompt-to-Track & Post-Run Search
```bash
# Highlight matching people live + build persistent Track Cards with cross-exit Re-ID
trackflow run mall.mp4 --reid clip --query "person in red jacket with a backpack"

# Search the completed run again anytime without re-running detection
trackflow search runs/latest "person carrying a suitcase"
```

### 3. Visual Track Debugger, Tracker Comparison & 4-Point Floorplan
```bash
# Compare ByteTrack vs BoT-SORT side-by-side on the same video
trackflow compare crowd.mp4 --trackers bytetrack,botsort --inspect

# Or open the Visual Track Debugger & 4-Point Homography Studio on any run
trackflow inspect runs/latest
```

### 4. Privacy-First Anonymization
```bash
# Blur faces/heads only
trackflow run lobby.mp4 --anonymize face-blur

# Zero-PII pose-skeleton-only analytics (no raw video pixels saved)
trackflow run lobby.mp4 --anonymize skeleton
```

### 5. Reproducible 2017 vs. 2026 Benchmark
```bash
trackflow benchmark crowd.mp4 --models dfine-n,rtdetr-v2-r18 --trackers bytetrack,botsort --split-screen
```

---

## Zero-Install WebGPU Browser Playground

Try Trackflow immediately in your browser at **[obendidi.github.io/trackflow](https://obendidi.github.io/trackflow/)** — 100% local WebGPU/WASM execution with zero video upload.

The [`web/`](web/) directory powers the playground via `@huggingface/transformers` using the exact same `onnx-community` ONNX models (`rtdetr_v2_r18vd-ONNX`, `dfine_n_coco-ONNX`) plus in-browser ByteTrack and live privacy anonymization. To run it locally:

```bash
python3 -m http.server 8080 --directory web
```

---

## Built-in Models

All weights auto-download and cache from Hugging Face on first run:

| Preset | Model | Size (FP32) | License | Best For |
| :--- | :--- | ---: | :--- | :--- |
| `rtdetr-v2-r18` *(default)* | RT-DETRv2 ResNet-18 | ~81 MB | Apache-2.0 | Balanced speed & dense crowd accuracy |
| `dfine-n` | D-FINE Nano (COCO) | ~15 MB | Apache-2.0 | Ultra-lightweight edge / CPU fallback |
| `rfdetr-base` | RF-DETR Base (COCO) | ~108 MB | Apache-2.0 | High-accuracy people tracking |
| `--reid clip` | CLIP ViT-B/32 (INT8 ONNX) | ~153 MB | MIT | Fast natural-language querying & Re-ID |
| `--reid siglip` | SigLIP Base (INT8 ONNX) | ~205 MB | Apache-2.0 | Fine-grained clothing/attribute queries |

---

## Python API

```python
from trackflow import FloorplanHomography, TrackflowPipeline

pipeline = TrackflowPipeline(
    model="dfine-n",
    tracker="botsort",
    reid="clip",
    query="person wearing a red jacket",
    anonymize="face-blur",
)
summary = pipeline.run("crowd.mp4", output_dir="runs/crowd")
print(f"{summary.unique_tracks} unique tracks at {summary.avg_fps:.1f} FPS")
```

---

## License

Trackflow v2 (`main`) is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).  
The historical 2017 codebase on [`legacy-2017`](https://github.com/obendidi/trackflow/tree/legacy-2017) remains under GPL-3.0.
