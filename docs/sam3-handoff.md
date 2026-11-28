# Using Trackflow alongside SAM 3 (or heavy VLMs)

Trackflow and Meta's [SAM 3](https://github.com/facebookresearch/sam3) solve different layers of video understanding and pair cleanly when you need both high-throughput tracking and occasional pixel-accurate masks.

## Honest comparison

| Capability | Trackflow | SAM 3 / SAM 3.1 |
| :--- | :--- | :--- |
| **Core license** | **Apache-2.0** (OSI permissive) | Custom **Meta SAM License** (non-OSI; restricts military/espionage/reverse engineering, includes patent-retaliation terms) |
| **Model size & hardware** | ~15–126 MB ONNX detectors; runs on laptop CPU, Apple Silicon, CUDA, and browser WebGPU | ~848M parameters; official path requires Python >=3.12, PyTorch >=2.7, CUDA >=12.6 |
| **Long video & live RTSP** | Flat memory; streams hours of footage or live cameras | Memory bank scales with `frames × objects`; needs windowing on long clips |
| **Outputs** | Bounding boxes, track IDs, Parquet/DuckDB/CSV/MOT, line crossings, 4-point floorplan, anonymization, text search, local inspector | Open-vocabulary text/exemplar/box/point **pixel masks** |
| **Where it wins** | Fast local analytics, commercial Apache-2.0 pipelines, live cameras, filtering hours of video | Open-vocabulary concepts beyond COCO classes and pixel-exact segmentation on short GPU clips |

Trackflow never bundles SAM 3 or pulls non-OSI dependencies into core.

## Pattern: Track & filter locally, segment selectively

1. Run Trackflow locally across the full video with text search enabled (`--vlm siglip2` for pure ONNX, or `--vlm embeddinggemma` for highest attribute recall):

```bash
trackflow run lobby_2h.mp4 --tracker boosttrack --vlm siglip2 --no-video
```

2. Shortlist the exact matching tracks and export a prompt manifest (`trackflow.prompts.v1`):

```bash
trackflow search runs/run-1 "person carrying a red backpack" \
  --top-k 5 --pad-s 1.0 --export-prompts prompts.json
```

Each entry in `prompts.json` gives the track's time/frame window (`start_frame`, `end_frame`, `start_s`, `end_s`), the highest-confidence observation (`prompt_frame`), pixel `bbox_xyxy`, and SAM-3-ready normalized `[0, 1]` coordinates (`sam3_box_xywh_norm`, `sam3_point_xy_norm`).

3. Feed those short windows into SAM 3 in your own separate GPU environment (keeps CUDA 12.6 and Meta's SAM License completely outside your Trackflow deployment):

```python
import json
from sam3.model_builder import build_sam3_video_predictor

manifest = json.loads(open("prompts.json").read())
predictor = build_sam3_video_predictor()

for item in manifest["prompts"]:
    # Slice manifest["source"] to [item["start_frame"], item["end_frame"]] (or pass full path
    # with prompt_frame offset), then initialize a SAM 3 video session on the short clip:
    resp = predictor.handle_request({"type": "start_session", "resource_path": manifest["source"]})
    sid = resp["session_id"]
    predictor.handle_request({
        "type": "add_prompt",
        "session_id": sid,
        "frame_index": item["prompt_frame"],
        "bounding_boxes": [item["sam3_box_xywh_norm"]],
        "bounding_box_labels": [1],
    })
    # Propagate masks only over the shortlisted window instead of the entire 2-hour video.
```

## Live ambiguity-triggered mask fallback (SAM 2.1 Apache-2.0 or external SAM 3)

For crowded dance or sports scenes where two people cross behind an occluder and box trackers swap IDs, Trackflow exposes an online ambiguity monitor (`MaskFallbackResolver`) with a bounded in-RAM frame ring buffer. It runs the fast ONNX box tracker on normal frames and dispatches only short transient-occlusion windows to a `MaskPropagator`, reconciling identities with a collision-free bijective permutation. By default it uses the conservative transient-occlusion-recovery gate (`include_fragmentation=False`) for zero-regression safety on unseen crowds; pass `MaskFallbackConfig(include_fragmentation=True)` to also relink occlusion-split newborn tracks.

### Option A: Built-in Apache-2.0 SAM 2.1 / EdgeTAM (`[sam2]` extra)

Install the optional Apache-2.0 extra (Hugging Face Transformers + PyTorch; zero Meta SAM License):

```bash
pip install "trackflow-python[sam2]"
trackflow run dance.mp4 --tracker ocsort --mask-fallback sam2.1-tiny --mask-budget 0.10
```

Available presets (`trackflow models`): `sam2.1-tiny` (default), `edgetam`, `sam2.1-small`, `sam2.1-base-plus`, `sam2.1-large`. All five are pinned to exact Apache-2.0 Hugging Face revisions.

### Option B: External SAM 3 adapter (keeps Meta SAM License out of Trackflow)

If you have approved access to `facebook/sam3` in your own GPU environment, wrap your SAM 3 video predictor (`handle_request` + `handle_stream_request`) with `Sam3MaskPropagator` and pass it into `TrackflowPipeline`. Note that SAM 3's `start_session` reads JPEG frames from a temporary directory on disk, so `Sam3MaskPropagator` cannot be combined with `--anonymize` (use in-memory `Sam2MaskPropagator` when anonymization is active):

```python
from sam3.model_builder import build_sam3_video_predictor
from trackflow import MaskFallbackConfig, Sam3MaskPropagator, TrackflowPipeline

predictor = build_sam3_video_predictor()
pipe = TrackflowPipeline(
    tracker="ocsort",
    mask_propagator=Sam3MaskPropagator(predictor),
    mask_fallback_config=MaskFallbackConfig(max_frame_rate=0.10),
)
summary = pipe.run("match.mp4", save_video=False)
print(summary.mask_fallback)
```
