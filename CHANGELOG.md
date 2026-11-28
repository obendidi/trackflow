# Changelog

Notable changes in each release. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Each GitHub release uses its version's section from this file as its notes.

## [Unreleased]

### Added

- Opt-in `--postprocess` offline pass (`TrackflowPipeline(..., postprocess=True)` and `postprocess_records` / `postprocess_tracks_array`) for recorded videos and image directories: splits suspect track switches, links fragmented tracklets with camera-motion-compensated kinematics and self-calibrated OSNet appearance, fills short supported occlusion gaps, and smooths jitter with motion-adaptive two-scale Gaussian filtering. Lifts 4-dataset mean HOTA across all 81 official validation videos by **+3.2 HOTA** on `botsort --reid osnet` (`38.8 → 42.0`: DanceTrack `+3.4`, SportsMOT `+5.3`, MOT17 `+2.8`, MOT20 `+1.2`) and **+0.7 to +1.1 HOTA** without Re-ID, in `~64–265 ms/video` (max `397 ms`), while leaving live `stream()` and `ByteTracker` web parity untouched.
- `--postprocess` flag in `scripts/eval_benchmarks.py` and `last_cmc_warp` exposure on `BotSortTracker`, `OcSortTracker`, and `BoostTracker`.

## [0.6.1] - 2026-11-27

### Added

- Dedicated one-command GPU packages published alongside `trackflow-python`: `trackflow-python-cuda` (bundles `onnxruntime-gpu[cuda,cudnn]`) and `trackflow-python-tensorrt` (bundles `onnxruntime-gpu[cuda,cudnn]>=1.28` + `tensorrt-cu13-libs>=10,<11` and defaults `--device` to `tensorrt`), plus a multi-target `Dockerfile` (`cpu`, `cuda`, `tensorrt`).
- Automatic CUDA, cuDNN, and TensorRT shared-library preloading (`preload_gpu_libraries`) so pip/uv installs work without installing system CUDA Toolkit/TensorRT or setting `LD_LIBRARY_PATH`.
- Persistent TensorRT FP16 engine and timing caches under `~/.cache/trackflow/trt_cache/` (cutting warm startup from ~3–4 minutes to <1 s, with `2.6–3.2x` higher FPS over CUDA on an NVIDIA T4) and automatic FP32 fallback on `rfdetr-*` presets whose DINOv2 activations overflow FP16.
- Hardware diagnostics in `trackflow info [--json]` (NVIDIA GPU detection, Apple Silicon P-core count, preloaded TensorRT libs, and copy-paste GPU commands), `--device` flag on `trackflow compare`, `trackflow search --json`, and `web/llms.txt`.

### Changed

- Pinned ONNX Runtime CPU threads on Apple Silicon to performance cores (`hw.perflevel0.physicalcpu`), speeding up CPU DETR detectors by 17–27% without setting `OMP_NUM_THREADS`.
- Enabled static-batch (`8`) OSNet Re-ID on both CoreML and TensorRT (1.8x faster than CUDA on T4, 5–24x faster than CPU on Apple Silicon with 0.99996–1.00000 cosine parity).
- Overlapped local frame decoding and MP4 rendering/encoding on bounded worker threads in `TrackflowPipeline.run()` (`1.29–1.41x` faster end-to-end 1080p tracking on Apple M4 Pro), resized BGR before RGB conversion in `OnnxDetrDetector.preprocess()`, and removed temporary allocations in `XywhKalmanFilter` and `ocsort._direction`.
- Removed the broken `[cuda]` optional extra from `trackflow-python` (which installed both `onnxruntime` and `onnxruntime-gpu` in the same environment and silently broke GPU providers).

## [0.6.0] - 2026-11-26

### Added

- Apache-2.0 ONNX detector presets `dfine-s` and `dfine-m` (`dfine-m` reaches 46.5 HOTA on MOT17 with ByteTrack and 46.9 with BoT-SORT).
- Pure-ONNX INT8 `--vlm siglip2` preset for natural-language track search without PyTorch (0.442 mAP on PA-100K / 0.488 on UPAR at ~45 crops/s on CPU, outperforming SigLIP 1 in both speed and accuracy).
- `trackflow search <run> "<query>" --export-prompts prompts.json [--pad-s 1.0]` to export top search hits as time windows and clipped normalized `[0, 1]` boxes/points for downstream mask models like SAM 3 ([docs/sam3-handoff.md](docs/sam3-handoff.md)).
- Bundled `onnx-vision-deploy` coding-agent skill (`skills/onnx-vision-deploy/SKILL.md`).
- Full 4-dataset benchmark harnesses (`scripts/eval_benchmarks.py` and `scripts/eval_text_search.py`) and speed-vs-quality Pareto charts in the README across DanceTrack (25 videos), SportsMOT (45), MOT17 (7), MOT20 (4), PA-100K (10,000 crops), and UPAR (13,287 crops).

### Changed

- Unified foreground-masked RANSAC camera-motion compensation (`SparseOpticalFlowCMC`), direct `[cx, cy, w, h]` noise-scaled Kalman filtering (`XywhKalmanFilter`), and height-modulated IoU (`compute_hmiou_matrix`) across BoT-SORT, OC-SORT, and BoostTrack++, while keeping ByteTrack bit-exact with the browser playground. Across all 4 official MOT validation sets (81 videos, identical `rtdetr-v2-r18` detections), Trackflow outperforms matched-threshold Roboflow `trackers==2.7.0` on every dataset (37.9–38.8 vs 34.3–37.5 mean HOTA).

### Fixed

- Enabled ImageNet normalization on `rfdetr-nano`, `rfdetr-small`, `rfdetr-medium`, and `rfdetr-base` (overriding `do_normalize: false` in the upstream ONNX preprocessor sidecar), lifting `rfdetr-medium` on MOT17 from 41.2 to 42.9 HOTA (+3.3 IDF1).
- Routed `ReidEmbedder(backend="osnet")` to CPU when `--device coreml` or `mps` is passed, avoiding CoreML MLProgram recompilation stalls on dynamic per-frame crop batch sizes.
- Preserved matched active track IDs during BoostTrack duplicate pruning and forwarded `new_track_thresh` through `create_tracker`.

## [0.5.1] - 2026-11-25

### Added

- `trackflow inspect` renders the actual video or image-folder frames behind Track Debugger boxes/trails (with a `Video` toggle) and behind the 4-Point Floorplan calibration cloud via `GET /api/frame/<frame_idx>`. Anonymized runs never serve raw source frames and only read from `annotated.mp4` when present.
- Migration guides for [Google Cloud Video Intelligence (`OBJECT_TRACKING`)](docs/migrate-google-video-intelligence.md) and [AWS Rekognition (`StartPersonTracking` / People Pathing)](docs/migrate-aws-rekognition.md), plus an animated `trackflow inspect` walkthrough in the README.

## [0.5.0] - 2026-11-25

### Added

- `--vlm embeddinggemma` scores `--query` and powers `trackflow search` with Google's EmbeddingGemma 2 (install `trackflow-python[transformers]`). On 4,000 human-labelled pedestrian crops it ranked people matching clothing and accessory queries above SigLIP on 33 of 34 queries, and runs at 6.3 FPS against SigLIP's 6.9 on a crowded street clip. Its `--query-threshold` defaults to 0.71 to match its score scale.

### Changed

- The `transformers` extra now requires transformers 5.18 or later and installs sentence-transformers.
- Runs with `--query` or `--vlm` embed each track when it first appears and then at ages 1, 2, 4, 8 seconds and so on, and average those crops, instead of embedding every track every 5 frames. Runs with them are 2.3 to 2.5 times faster on busy scenes (3.0 to 6.9 FPS on a crowded street with SigLIP on an Apple M4 Pro), with search results about the same. `TrackCardIndex` no longer takes `embed_every_n_frames`.

### Fixed

- `--query` with `--vlm siglip` flags matching tracks instead of none, and `--query` with CLIP, the default, no longer flags nearly every track. `--query-threshold` now defaults to 0.08 for SigLIP and 0.27 for CLIP instead of 0.22 for both, and `TrackCardIndex` uses the same per-model default.
- Ctrl-C stops a run and still writes its track cards, events, summary and report, so a camera or stream run can be searched and inspected afterwards. Before, Ctrl-C kept only the track tables. A second Ctrl-C aborts.

## [0.4.3] - 2026-11-24

### Changed

- Requires ONNX Runtime 1.20 or later (was 1.17).

### Fixed

- `--device coreml` runs detectors and Re-ID on the Apple GPU with the same boxes as the CPU, using about a third of the CPU. YOLO26 presets no longer return wrong boxes on it.

## [0.4.2] - 2026-11-24

### Fixed

- YOLO26 presets keep each frame's aspect ratio and pad it with gray, as YOLO is trained, instead of stretching it to 640×640. With ByteTrack on MOT17, YOLO26 Nano goes from 29.9 to 37.4 HOTA.
- The browser playground letterboxes YOLO26 Nano frames the same way.

## [0.4.1] - 2026-11-23

### Fixed

- `--line` counts a crossing only once the track has stayed on the new side for 0.3 seconds, so boxes that jitter across the line in crowds no longer inflate the counts.

## [0.4.0] - 2026-11-23

### Added

- `--line name=x1,y1,x2,y2` counts tracks crossing a line in each direction, with totals in the run summary, `summary.json` and the HTML report, and a `line_crossed` event per crossing.

### Changed

- The `--anonymize` help and the playground label say that `face-blur` blurs the top 30% of each person box, so nobody mistakes it for a face detector.

### Fixed

- `LICENSE` now has the official Apache-2.0 text word for word, so GitHub and license scanners recognize it.

## [0.3.0] - 2026-11-23

### Added

- `--tracker ocsort`: [OC-SORT](https://arxiv.org/abs/2203.14360) (CVPR 2023). After an occlusion it re-matches a track to its last observed box and corrects the motion model along the gap.
- `--tracker boosttrack`: [BoostTrack++](https://arxiv.org/abs/2408.13003) (2024). It raises the confidence of weak detections that fit a track or look like a new object, compensates for camera motion, and matches on appearance with `--reid osnet`.
- `OcSortTracker`, `BoostTracker` and `TRACKER_NAMES` in the Python API.
- GitHub releases, with notes from this changelog and the wheel and source archive attached.

### Changed

- The pipeline computes Re-ID embeddings for the detections the chosen tracker uses, instead of only for BoT-SORT. A custom tracker opts in with `embedding_min_confidence`.

### Fixed

- An unknown `--tracker` name fails right away instead of after the model downloads.
- A run that fails before writing anything, such as one with a missing source file, no longer leaves an empty `runs/run-N` folder.

## [0.2.0] - 2026-11-23

### Added

- Detector presets for RF-DETR Small, Medium and Large, YOLO26 n/s/m/l/x, and DEIMv2 Atto and S, for 14 in total. Each preset is pinned to a Hugging Face revision.
- Any Transformers object-detection checkpoint with `-m hf:owner/repo` and the `transformers` extra, on CPU, CUDA or Apple's MPS.
- `TrackflowPipeline.stream(frames)`, which returns tracks frame by frame without writing files.
- `Detector` and `Tracker` protocols, a `tracker_factory` hook, `create_detector`, and type hints (`py.typed`).
- `--export mot`, which writes MOTChallenge files that TrackEval scores directly.
- RTSP/HTTP streams and image folders as sources, `--fps`, `trackflow models --json`, and `trackflow inspect` with no argument to open the latest run.
- CI on macOS and Windows.

### Changed

- Every `run`, `compare` and `benchmark` writes to a new `runs/<name>-N` folder, and `-o` must point to an empty folder, so a new run never mixes with an old one's files.
- `--reid-merge` only links people who have been gone longer than the tracker's own recovery window, so it merges less often.
- The benchmark's split screen shows the speed published for the 2017 version instead of a simulated replay.

### Fixed

- ByteTrack and BoT-SORT matching is class-aware, so a person and a backpack in the same box stay separate tracks.
- D-FINE INT8 no longer crashes on Apple Silicon.
- An unknown model name or an unavailable `--device` is a clear error instead of a traceback or a silent fallback to CPU.
- The browser playground gets the same tracker fixes, releases the old model when you switch, and builds its privacy backdrop from the frame.

## [0.1.1] - 2026-11-23

### Added

- The `rfdetr-nano` preset.
- `--reid osnet`, an OSNet person Re-ID model for BoT-SORT appearance matching, and opt-in `--reid-merge` to give people who leave and come back their earlier ID.
- `--vlm clip|siglip` for `--query` and `trackflow search`.

### Changed

- CLIP and SigLIP are no longer used for identity. `--reid clip` still works, with a warning, as `--vlm clip`.
- A new track needs a second detection before it gets an ID, and a box mostly inside an already tracked person no longer starts a track.
- Box matching tolerates more movement when frames are processed further apart, which keeps IDs stable with `--skip` and in slow browsers.
- Anonymization blurs every detected person, including low-confidence detections and people the tracker briefly missed, and anonymized runs write no person crops or thumbnails.

### Fixed

- `trackflow run` no longer crashes with an `AttributeError` from the progress bar.
- In the browser playground, confidence scores no longer let background boxes through, and IDs no longer hop between people.
- BoT-SORT camera-motion compensation no longer compares against a stale frame.
- Without CUDA, ONNX Runtime falls back to CPU automatically.

## [0.1.0] - 2026-11-23

First release of Trackflow, a rewrite of the 2017 Tracking-with-darkflow project under Apache-2.0. The 2017 code stays on the `legacy-2017` branch under GPL-3.0.

### Added

- ONNX Runtime detectors (RT-DETRv2, D-FINE, RF-DETR) with ByteTrack and BoT-SORT.
- The `trackflow` CLI, with Parquet, JSONL, CSV, HTML and annotated MP4 output.
- Text queries (`--query`) and `trackflow search`, with CLIP or SigLIP.
- `trackflow inspect`, a local web UI to review runs, compare trackers and project tracks onto a floorplan.
- Anonymization by face blur, blur, pixelation or stick figures.
- A browser playground on GitHub Pages.
- A benchmark against the 2017 version.

### Known issues

- `trackflow run` crashes with an `AttributeError` from the progress bar. Use 0.1.1 or later.

[Unreleased]: https://github.com/obendidi/trackflow/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/obendidi/trackflow/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/obendidi/trackflow/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/obendidi/trackflow/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/obendidi/trackflow/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/obendidi/trackflow/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/obendidi/trackflow/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/obendidi/trackflow/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/obendidi/trackflow/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/obendidi/trackflow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/obendidi/trackflow/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/obendidi/trackflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/obendidi/trackflow/releases/tag/v0.1.0
