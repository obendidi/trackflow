# Trackers

Trackflow separates detection from association so you can swap trackers with `--tracker` without changing the detector or output format.

## Choosing a tracker

| `--tracker` | Algorithm | Best for | Speed / accuracy tradeoff |
| :--- | :--- | :--- | :--- |
| `bytetrack` *(default)* | [ByteTrack](https://arxiv.org/abs/2110.06864) (ECCV 2022) | Fixed cameras, low-CPU devices, browser WebGPU | Fastest matching (~2,900 tracker FPS); no camera compensation or Re-ID |
| `ocsort` | [OC-SORT](https://arxiv.org/abs/2203.14360) (CVPR 2023) | Fast motion, sports, nonlinear paths without Re-ID overhead | Strong no-Re-ID baseline (`38.4` mean HOTA, `5,501` ID switches across 4 datasets) |
| `botsort` | [BoT-SORT](https://arxiv.org/abs/2206.14651) (2022) | Moving/panning cameras and crowded scenes; pairs with `--reid osnet` | `38.8` mean HOTA when paired with `--reid osnet` |
| `boosttrack` | [BoostTrack++](https://arxiv.org/abs/2408.13003) (2024) | Crowded occlusions where boxes dip below threshold briefly | Strong MOT17 (`46.9`) and MOT20 (`24.0`) HOTA with camera compensation and optional `--reid osnet` |
| `nltrack` | NLTrack *(Apache-2.0, zero extra weights)* | Non-linear motion, crossings, uniform outfits, and longer occlusions | Highest accuracy across all 4 datasets: `42.9` mean HOTA (`5,193` ID switches) with `--reid osnet`, and `39.6` without Re-ID |

## Shared behavior across all five trackers

- **Per-class isolation**: A `person` and a `backpack` overlapping at the same coordinates stay separate tracks and never merge.
- **Multi-frame confirmation**: New detections only receive a confirmed `track_id` once seen across consecutive frames (`min_hits`), filtering single-frame false positives.
- **Occlusion recovery**: When an object disappears briefly and returns within the tracker's buffer window, it keeps its original ID and sets `recovered = true` on the re-matched frame. OC-SORT and BoostTrack wait for 3 consecutive matches after a miss before re-emitting the track (drawing a coasting box in the annotated video in between).
- **Skip-aware IoU relaxation**: If you pass `--skip N` to run detection every `N+1` frames, Trackflow widens matching tolerances automatically (`iou_buffer_for_interval`) so fast-moving objects still link across skipped frames.

## Camera motion compensation (CMC) and Re-ID

`botsort`, `boosttrack`, and `nltrack` estimate sparse optical flow between consecutive frames to compensate for panning or handheld camera movement. On a strictly fixed CCTV camera, CMC is unnecessary overhead; you can disable it from Python:

```python
from trackflow import BoostTracker, BotSortTracker, NonLinearTracker, TrackflowPipeline

pipeline = TrackflowPipeline(
    model="rtdetr-v2-r18",
    tracker_factory=lambda: NonLinearTracker(use_cmc=False),
)
```

To add appearance matching on top of motion matching, pass `--reid osnet` with `nltrack`, `botsort`, or `boosttrack`:

```bash
trackflow run crowd.mp4 --tracker nltrack --reid osnet
```

`osnet` extracts 512-dimensional person embeddings (`OSNet-AIN x1.0`, ~9 MB) for every detected person box. It reduces ID switches after crossings, at the cost of running one extra ONNX model per frame. If you want cross-exit re-linking after a person leaves the frame completely and walks back in later, add `--reid-merge` (see [Text Search & Re-ID](search.md#cross-exit-re-identification-reid-merge)).

## Tuning thresholds and parameters

### Detection threshold (`--threshold`)

The default high-confidence tracking threshold is `0.45`. Lower detections still feed ByteTrack/BoT-SORT's second matching stage so partially occluded people keep their ID.

- Raise `--threshold 0.6` on sports broadcasts or stadium footage if blurry background spectators are turning into noisy tracks.
- Lower `--threshold 0.3` on distant aerial or low-light cameras where true objects score lower.

### Custom tracker tuning in Python

Every built-in tracker class is exported directly from `trackflow` so you can customize thresholds, track buffers, or appearance weights through `tracker_factory`:

```python
from trackflow import BotSortTracker, ByteTracker, OcSortTracker, TrackflowPipeline

# Example: longer occlusion buffer (60 frames) and tighter IoU gate
pipeline = TrackflowPipeline(
    model="dfine-m",
    tracker_factory=lambda: OcSortTracker(
        high_thresh=0.45,
        iou_threshold=0.3,
        max_age=60,
        min_hits=3,
    ),
)
summary = pipeline.run("court.mp4")
```

See the [Python API Reference](../reference/python-api.md#trackers) for every constructor parameter on `ByteTracker`, `BotSortTracker`, `OcSortTracker`, `BoostTracker`, and `NonLinearTracker`.

## Comparing trackers on your own footage

Benchmark numbers shift across camera angles and crowd density. Run two trackers side by side on a clip from your own camera and inspect the differences frame by frame:

```bash
trackflow compare footage.mp4 --trackers bytetrack,botsort --inspect
```

For quantitative accuracy on labeled sequences, export MOTChallenge text files (`--export mot`) and score them with [TrackEval](https://github.com/JonathonLuiten/TrackEval). See [Benchmarks](../benchmarks.md) for Trackflow's 4-dataset evaluation.
