# Changelog

Notable changes in each release. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Each GitHub release uses its version's section from this file as its notes.

## [Unreleased]

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

[Unreleased]: https://github.com/obendidi/trackflow/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/obendidi/trackflow/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/obendidi/trackflow/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/obendidi/trackflow/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/obendidi/trackflow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/obendidi/trackflow/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/obendidi/trackflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/obendidi/trackflow/releases/tag/v0.1.0
