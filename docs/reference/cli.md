# CLI & Output Schema

All three packages (`trackflow-python`, `trackflow-python-cuda`, `trackflow-python-tensorrt`) install the `trackflow` command.

## Commands

### `trackflow run <source>`

Run detection and tracking on a video file, image directory (natural sort: `frame_2` before `frame_10`), RTSP/HTTP URL, or camera index (`0`).

```bash
trackflow run crowd.mp4 \
  --model rtdetr-v2-r18 \
  --tracker botsort \
  --reid osnet \
  --classes person,backpack \
  --line "door=640,300,640,720" \
  --export parquet,jsonl,csv,mot,html
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-m, --model` | `rtdetr-v2-r18` | Built-in preset, Hugging Face ONNX repo (`owner/repo`), local `.onnx`, or `hf:owner/repo` (`[transformers]` extra) |
| `--precision` | `fp32` | ONNX weight precision: `fp32`, `fp16`, or `int8` |
| `-t, --tracker` | `bytetrack` | `bytetrack`, `botsort`, `ocsort`, or `boosttrack` |
| `-d, --device` | `auto` *(or `tensorrt` in tensorrt package)* | ONNX: `auto`, `cpu`, `cuda`, `coreml`, `tensorrt`. Transformers: `auto`, `cpu`, `cuda`, `mps` |
| `--threshold` | `0.45` | High-confidence tracking threshold (`0.0–1.0`) |
| `-c, --classes` | `person` | Comma-separated class names, or `all` |
| `--reid` | `none` | Person Re-ID encoder for BoT-SORT/BoostTrack and `--reid-merge`: `none`, `osnet`, `color` |
| `--reid-merge / --no-reid-merge` | `--no-reid-merge` | Re-link people who leave the frame and return later (requires `--reid osnet`) |
| `-q, --query` | `None` | Natural-language query to highlight matching tracks live |
| `--vlm` | `clip` when `--query` is set, else `none` | Vision-language model: `none`, `clip`, `siglip`, `siglip2`, `embeddinggemma` |
| `--query-threshold` | Per-VLM default | Override cosine similarity cutoff for `--query` matches |
| `--anonymize` | `none` | `none`, `face-blur`, `blur`, `pixelate`, or `skeleton` |
| `--homography` | `None` | Path to 4-point floorplan calibration JSON |
| `--line` | `None` | Directional counting line `[name=]x1,y1,x2,y2` in pixels (repeatable) |
| `--skip` | `0` | Skip `N` frames between detections |
| `--max-frames` | `None` | Stop after `N` source frames |
| `--fps` | Source FPS *(25 for image folders)* | Override source frame rate |
| `-o, --output` | `runs/run-N` | Empty output directory (refuses non-empty directories to prevent mixing runs) |
| `--save-video / --no-video` | `--save-video` | Write `annotated.mp4` |
| `--export` | `parquet,jsonl,html` | Comma-separated formats: `parquet`, `jsonl`, `csv`, `mot`, `html` |
| `--display` | `False` | Open an OpenCV live window (requires `trackflow-python[display]`) |

!!! tip "Graceful Ctrl-C"
    Pressing **Ctrl-C** once during `trackflow run` stops reading frames and still finalizes every export, `summary.json`, and `track_cards.json`. Pressing Ctrl-C a second time aborts immediately without writing outputs.

---

### `trackflow search <run_dir> "<query>"`

Search saved Track Cards in a completed run (requires a run created with `--query` or `--vlm`).

```bash
trackflow search runs/run-1 "person carrying a red backpack" --top-k 5 --pad-s 1.0 --export-prompts prompts.json
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-k, --top-k` | `10` | Maximum matching tracks to return |
| `--min-score` | `-1.0` | Minimum cosine similarity filter |
| `--export-prompts` | `None` | Write top hits as a `trackflow.prompts.v1` JSON manifest (time windows + pixel/normalized boxes for SAM 3 or VLMs) |
| `--pad-s` | `0.0` | Seconds of context added before and after each hit window in `--export-prompts` |
| `--json` | `False` | Machine-readable JSON output |

---

### `trackflow inspect [run_dir]`

Start the local web UI to scrub video frames, jump to recovery events, search Track Cards, calibrate a 4-point floorplan, or compare two runs. Defaults to the newest run under `runs/`.

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--compare` | `None` | Second run directory for side-by-side tracker diff |
| `--host` | `127.0.0.1` | Bind host |
| `-p, --port` | `8765` | Local HTTP port |
| `--no-browser` | `False` | Do not open a browser tab automatically |

---

### `trackflow compare <source>`

Run two trackers on the same clip and print a side-by-side summary table.

```bash
trackflow compare crowd.mp4 --trackers bytetrack,botsort --inspect
```

Flags: `--trackers` (default `bytetrack,botsort`), `-m/--model`, `-d/--device`, `--max-frames`, `-o/--output`, `--inspect`.

---

### `trackflow benchmark [video]`

Measure detector/tracker throughput on a video (or a generated synthetic crowd clip if omitted) and write Markdown + JSON reports.

Flags: `--models`, `--trackers`, `--device`, `--max-frames` (default `90`), `-o/--output`, `--split-screen`.

---

### `trackflow models` · `trackflow info` · `trackflow version`

- `trackflow models [--json]` — Prints all built-in detectors, Re-ID models, and VLMs with sizes, pinned Hugging Face repos, and licenses.
- `trackflow info [--json]` — Prints Python/ORT versions, NVIDIA GPU or Apple Silicon P-cores, preloaded GPU libraries, available execution providers, and any recommended GPU command.
- `trackflow version` — Prints the installed package version.

## Environment variables

- `TRACKFLOW_DEFAULT_DEVICE`: Sets the default `--device` value when not passed on the command line (automatically set to `tensorrt` by the `trackflow-python-tensorrt` entrypoint).

## Output files in a run directory

Each `trackflow run` creates a clean folder (`runs/run-1`, `runs/run-2`, …):

| File | Produced when | Contents |
| :--- | :--- | :--- |
| `tracks.parquet` | `parquet` in `--export` | Snappy-compressed Arrow Parquet observations (DuckDB-ready) |
| `tracks.jsonl` | `jsonl` in `--export` | One JSON object per tracked observation |
| `tracks.csv` | `csv` in `--export` | CSV with the same 20 columns as Parquet |
| `tracks.txt` | `mot` in `--export` | 1-indexed MOTChallenge format (`frame,id,x,y,w,h,conf,-1,-1,-1`) for TrackEval |
| `report.html` | `html` in `--export` | Self-contained HTML summary report |
| `annotated.mp4` | `--save-video` (default) | Annotated video with boxes, track IDs, trails, lines, and optional anonymization |
| `summary.json` | Always | Run metadata, active provider, FPS, latency, track counts, `line_counts`, output paths |
| `events.json` | Always | Structured events (`track_recovered`, `line_crossed`, etc.) |
| `track_cards.json` | Always | Per-track summaries, thumbnails (omitted when anonymized), and VLM embeddings (when `--vlm` or `--query` is set) |

## Observation schema (`tracks.parquet`, `tracks.jsonl`, `tracks.csv`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `frame_idx` | `int64` | 0-based source frame index |
| `timestamp_s` | `float64` | Seconds from video start |
| `track_id` | `int64` | Persistent integer track identifier |
| `class_id` | `int64` | Integer detector class ID |
| `class_name` | `string` | Human-readable class label (e.g. `person`) |
| `confidence` | `float64` | Detection score (`0.0–1.0`) |
| `x1`, `y1`, `x2`, `y2` | `float64` | Bounding box corners in image pixels |
| `width`, `height` | `float64` | Bounding box dimensions in pixels |
| `age` | `int64` | Frames since track creation |
| `hits` | `int64` | Total matched detections for this track |
| `recovered` | `bool` | `true` on the frame a track re-linked after a missed-detection gap |
| `query_score` | `float64` | Cosine similarity to `--query` (or `null` when `--query` is unset) |
| `query_match` | `bool` | Whether `query_score` exceeded `--query-threshold` |
| `world_x`, `world_y` | `float64` | Floorplan coordinates in meters (populated when `--homography` is set) |
| `zone` | `string` | Active dwell zone name on the floorplan (empty when outside zones or uncalibrated) |
