"""End-to-end Trackflow video/webcam/directory tracking pipeline."""

from __future__ import annotations

import contextlib
import json
import re
import signal
import tempfile
import threading
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from itertools import count
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from trackflow.detector import Detector, _allowed_classes, _class_key, create_detector
from trackflow.homography import FloorplanHomography
from trackflow.lines import CountingLine, LineCounter
from trackflow.models import Precision
from trackflow.privacy import AnonymizeMode, PrivacyAnonymizer
from trackflow.reid import ReidBackend, ReidEmbedder, TrackCardIndex, VlmBackend, VlmEmbedder
from trackflow.sinks import ObservationWriter, draw_tracks, write_run_report
from trackflow.tracker import Tracker, create_tracker, iou_buffer_for_interval
from trackflow.types import Detection, FrameResult, RunSummary

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fresh_run_dir(stem: str = "run", parent: str | Path = "runs") -> Path:
    """Create and return the first unused ``<parent>/<stem>-N`` directory."""
    for number in count(1):
        path = Path(parent) / f"{stem}-{number}"
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            continue
        return path
    raise AssertionError("unreachable")


@contextlib.contextmanager
def _stop_on_interrupt() -> Iterator[threading.Event]:
    """Turn the first Ctrl-C into a stop request that ``run`` checks between frames.

    After that first Ctrl-C the previous handler is back, so a second one interrupts as usual.
    Signal handlers can only be set from the main thread; elsewhere Ctrl-C keeps its default.
    """
    stop = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    if threading.current_thread() is not threading.main_thread() or previous is None:
        yield stop
        return

    def request_stop(signum: int, frame: Any) -> None:
        stop.set()
        signal.signal(signal.SIGINT, previous)

    signal.signal(signal.SIGINT, request_stop)
    try:
        yield stop
    finally:
        signal.signal(signal.SIGINT, previous)


@contextlib.contextmanager
def _run_directory(output_dir: str | Path | None) -> Iterator[Path]:
    if output_dir is None:
        path = fresh_run_dir()
        created = True
    else:
        path = Path(output_dir)
        created = not path.exists()
        path.mkdir(parents=True, exist_ok=True)

    # Two callers can arrive at an existing empty directory simultaneously.
    lock = path / ".trackflow-running"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Output {path} is in use. Choose a fresh output directory.") from exc
    try:
        if any(p != lock for p in path.iterdir()):
            raise FileExistsError(
                f"Output {path} is not empty. Choose a fresh output directory (-o)."
            )
        yield path
    except BaseException:
        # A run that fails before writing anything leaves no empty folder behind.
        lock.unlink(missing_ok=True)
        if created and not any(path.iterdir()):
            path.rmdir()
        raise
    finally:
        lock.unlink(missing_ok=True)


def _is_stream_url(source: str | Path | int) -> bool:
    return isinstance(source, str) and source.lower().startswith(
        ("rtsp://", "rtsps://", "http://", "https://")
    )


def _source_label(source: str | Path | int) -> str:
    if not _is_stream_url(source):
        return str(source)
    url = urlsplit(str(source))
    return urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))


def _iter_frames(
    source: str | Path | int,
    *,
    max_frames: int | None = None,
    fps: float | None = None,
) -> tuple[Generator[tuple[int, float, np.ndarray], None, None], float, int, int, int]:
    """Open files, image sequences, OpenCV network streams or cameras."""
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be positive and finite.")
    is_camera = isinstance(source, int) or (
        isinstance(source, str)
        and (source.isdigit() or source.lower() == "camera")
        and not Path(source).exists()
    )
    if is_camera:
        capture_source = 0 if str(source).lower() == "camera" else int(source)
    elif _is_stream_url(source):
        capture_source = str(source)
    else:
        path = Path(source)
        if path.is_dir():

            def natural_order(p: Path) -> list[int | str]:
                return [
                    int(part) if part.isdecimal() else part
                    for part in re.split(r"(\d+)", p.name.casefold())
                ]

            images = sorted(
                (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                key=natural_order,
            )
            if not images:
                raise FileNotFoundError(f"No images found in directory: {path}")
            first = cv2.imread(str(images[0]))
            if first is None:
                raise RuntimeError(f"Failed to read image: {images[0]}")
            h, w = first.shape[:2]
            rate = fps or 25.0
            total = min(len(images), max_frames) if max_frames is not None else len(images)

            def images_iter() -> Generator[tuple[int, float, np.ndarray], None, None]:
                for idx, p in enumerate(images[:total]):
                    frame = first if idx == 0 else cv2.imread(str(p))
                    if frame is None or frame.shape[:2] != (h, w):
                        raise ValueError(f"Expected a readable {w}x{h} image at {p}")
                    yield idx, idx / rate, frame

            return images_iter(), rate, w, h, total
        if not path.is_file():
            raise FileNotFoundError(f"Video source not found: {path}")
        capture_source = str(path)

    cap = cv2.VideoCapture(capture_source)
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError(f"Could not read video source: {_source_label(source)}")
    h, w = first.shape[:2]
    rate = fps or cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(rate) or rate <= 0:
        rate = 30.0
    reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total = int(reported) if np.isfinite(reported) and reported > 0 else 0
    if max_frames is not None:
        total = min(total, max_frames) if total else max_frames

    def capture_iter() -> Generator[tuple[int, float, np.ndarray], None, None]:
        try:
            idx, frame = 0, first
            while True:
                if frame.shape[:2] != (h, w):
                    raise ValueError("Video dimensions changed during tracking.")
                yield idx, idx / rate, frame
                idx += 1
                if max_frames is not None and idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
        finally:
            cap.release()

    return capture_iter(), rate, w, h, total


def _crop(frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, int(round(bbox[0]))), max(0, int(round(bbox[1])))
    x2, y2 = min(w, int(round(bbox[2]))), min(h, int(round(bbox[3])))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame_bgr[y1:y2, x1:x2]


class TrackflowPipeline:
    """Unified real-time tracking pipeline combining detection, MOT, Re-ID/NL query, privacy & floorplan."""

    def __init__(
        self,
        *,
        model: str | Path | None = None,
        precision: Precision = "fp32",
        tracker: str = "bytetrack",
        device: str = "auto",
        threshold: float = 0.45,
        classes: Sequence[str] | None = ("person",),
        reid: ReidBackend = "none",
        vlm: VlmBackend | None = None,
        reid_merge: bool = False,
        query: str | None = None,
        query_threshold: float = 0.22,
        anonymize: AnonymizeMode = "none",
        homography: FloorplanHomography | str | Path | None = None,
        lines: Sequence[CountingLine] = (),
        detector: Detector | None = None,
        tracker_factory: Callable[[], Tracker] | None = None,
        reid_embedder: ReidEmbedder | None = None,
        vlm_embedder: VlmEmbedder | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = (
            str(model)
            if model is not None
            else (
                getattr(detector, "model_name", type(detector).__name__)
                if detector is not None
                else "rtdetr-v2-r18"
            )
        )
        self.tracker_name = tracker
        self.tracker_factory = tracker_factory
        if tracker_factory is None:
            create_tracker(tracker)  # reject an unknown name before any model loads
        self.threshold = float(threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        self._allowed = _allowed_classes(classes)
        self.classes = sorted(self._allowed) if self._allowed is not None else ["all"]
        self.query = query.strip() if query else None
        self.query_threshold = float(query_threshold)
        self.anonymize_mode: AnonymizeMode = anonymize

        if reid in ("clip", "siglip"):
            # Pre-0.1 runs used `--reid clip` for queries. CLIP can't tell people apart, so it is
            # now only a vision-language model for --query / search, never an identity model.
            warnings.warn(
                f"--reid {reid} is now `--vlm {reid}` (text queries & search only). Use "
                "`--reid osnet` for person re-identification.",
                stacklevel=2,
            )
            vlm = vlm if vlm not in (None, "none") else reid  # type: ignore[assignment]
            reid = "none"
        if vlm is None:
            vlm = "clip" if self.query else "none"
        if reid_merge and reid == "none" and reid_embedder is None:
            raise ValueError("--reid-merge needs a person Re-ID model: add `--reid osnet`.")

        self.reid_backend: ReidBackend = reid_embedder.backend if reid_embedder else reid
        self.vlm_backend: VlmBackend = vlm_embedder.backend if vlm_embedder else vlm
        self.reid_merge = bool(reid_merge)
        if self.query and self.vlm_backend == "none":
            raise ValueError("A text query needs a vision-language model. Choose clip or siglip.")

        detect_classes = self._allowed
        if detect_classes is not None and anonymize in ("blur", "pixelate", "face-blur"):
            detect_classes = detect_classes | {"person"}
        self.detector = (
            detector
            if detector is not None
            else create_detector(
                model=self.model_name,
                precision=precision,
                device=device,
                threshold=min(0.15, threshold, max(0.10, threshold * 0.35)),
                classes=sorted(detect_classes) if detect_classes else None,
                cache_dir=cache_dir,
            )
        )
        self.reid_embedder = reid_embedder or (
            ReidEmbedder(
                backend=self.reid_backend,
                device="cpu" if device == "mps" else device,
                cache_dir=cache_dir,
            )
            if self.reid_backend != "none"
            else None
        )
        self.vlm_embedder = vlm_embedder or (
            VlmEmbedder(
                backend=self.vlm_backend,
                device="cpu" if device == "mps" else device,
                cache_dir=cache_dir,
            )
            if self.vlm_backend != "none"
            else None
        )
        self.anonymizer = PrivacyAnonymizer(mode=anonymize)
        if isinstance(homography, (str, Path)):
            self.homography: FloorplanHomography | None = FloorplanHomography.from_json(homography)
        else:
            self.homography = homography
        self.lines = list(lines)
        LineCounter(self.lines)  # reject duplicate names before any frame is read

    def _new_counter(self) -> LineCounter | None:
        return LineCounter(self.lines) if self.lines else None

    def _new_tracker(self, frame_dt: float) -> Tracker:
        if self.tracker_factory is not None:
            return self.tracker_factory()
        return create_tracker(
            self.tracker_name,
            track_high_thresh=self.threshold,
            track_low_thresh=min(self.threshold, max(0.10, self.threshold * 0.35)),
            new_track_thresh=max(self.threshold, min(0.95, self.threshold + 0.05)),
            iou_buffer=iou_buffer_for_interval(frame_dt),
        )

    def _new_cards(
        self,
        tracker: Tracker,
        stride: int,
        crops_dir: Path | None = None,
    ) -> TrackCardIndex:
        buffer = getattr(tracker, "track_buffer", None)
        if self.reid_merge and buffer is None:
            raise ValueError(
                "A custom tracker using reid_merge must expose its track_buffer in processed frames."
            )
        return TrackCardIndex(
            vlm=self.vlm_embedder,
            reid=self.reid_embedder,
            query=self.query,
            query_threshold=self.query_threshold,
            reid_merge=self.reid_merge,
            # A lost raw ID can still be recovered during the track buffer. Merging a
            # different raw ID into it during that window can give two objects the same ID.
            min_exit_gap_frames=max(8, ((buffer or 0) + 1) * stride),
            crops_dir=crops_dir,
        )

    def _process_frame(
        self,
        frame: np.ndarray,
        idx: int,
        timestamp: float,
        dt: float,
        tracker: Tracker,
        cards: TrackCardIndex | None,
        floorplan: FloorplanHomography | None,
        counter: LineCounter | None = None,
        *,
        record_floorplan: bool = False,
    ) -> tuple[FrameResult, list[Detection]]:
        start = time.perf_counter()
        raw = self.detector.detect(frame)
        detected = time.perf_counter()
        targets = (
            raw
            if self._allowed is None
            else [d for d in raw if _class_key(d.class_name) in self._allowed]
        )
        min_conf = getattr(tracker, "embedding_min_confidence", None)
        if min_conf is not None and self.reid_embedder is not None:
            people = [
                d for d in targets if d.confidence >= min_conf and d.class_name.lower() == "person"
            ]
            for det, emb in zip(
                people,
                self.reid_embedder.encode_images([_crop(frame, d.bbox) for d in people]),
                strict=True,
            ):
                det.embedding = emb
        tracks = tracker.update(targets, frame_bgr=frame)
        tracked = time.perf_counter()
        if floorplan is not None:
            for tr in tracks:
                if record_floorplan:
                    tr.world_xy, tr.zone = floorplan.observe(tr.track_id, tr.foot_point, dt_s=dt)
                else:
                    tr.world_xy = floorplan.project(tr.foot_point)
                    tr.zone = floorplan.locate_zone(tr.world_xy)
        events = list(getattr(tracker, "last_events", ()))
        if counter is not None:
            # Raw tracker IDs: a Re-ID merge would join a track to a point from before it left.
            events.extend(counter.update(tracks, timestamp))
        if cards is not None:
            events.extend(cards.update(idx, timestamp, frame, tracks, frame_dt_s=dt))
        return FrameResult(
            frame_idx=idx,
            timestamp_s=timestamp,
            tracks=tracks,
            num_detections=len(targets),
            inference_ms=(detected - start) * 1000,
            tracking_ms=(tracked - detected) * 1000,
            events=events,
        ), raw

    def stream(self, frames: Iterable[np.ndarray], *, fps: float = 30.0) -> Iterator[FrameResult]:
        """Yield observations for consecutive BGR uint8 frames without writing any files.

        Each call starts a fresh tracker. With re-entry merging enabled, a merge event
        reports the new canonical ID; observations already yielded keep their original ID.
        """
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite.")
        dt = 1.0 / fps
        tracker = self._new_tracker(dt)
        cards = self._new_cards(tracker, 1) if self.vlm_embedder or self.reid_merge else None
        counter = self._new_counter()
        shape = None
        for idx, frame in enumerate(frames):
            if (
                not isinstance(frame, np.ndarray)
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.dtype != np.uint8
                or not frame.size
            ):
                raise ValueError(f"Frame {idx} must be a nonempty H×W×3 uint8 BGR array.")
            if shape is not None and frame.shape != shape:
                raise ValueError(
                    f"Frame {idx} changed size. Start a new stream when geometry changes."
                )
            shape = frame.shape
            result, _ = self._process_frame(
                frame, idx, idx / fps, dt, tracker, cards, self.homography, counter
            )
            yield result

    def run(
        self,
        source: str | Path | int,
        *,
        output_dir: str | Path | None = None,
        skip: int = 0,
        fps: float | None = None,
        max_frames: int | None = None,
        save_video: bool = True,
        exports: Sequence[str] = ("parquet", "jsonl", "html"),
        display: bool = False,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> RunSummary:
        """Execute tracking on ``source`` and write requested artifacts into ``output_dir``.

        Ctrl-C stops reading frames and still writes every artifact for the frames processed so
        far, which is how a camera or stream run ends. A second Ctrl-C aborts.
        """
        if skip < 0 or (max_frames is not None and max_frames < 1):
            raise ValueError("skip must be nonnegative and max_frames must be positive.")
        if isinstance(exports, str):
            exports = exports.split(",")
        export_set = {e.lower().strip() for e in exports}
        unknown = export_set - {"parquet", "jsonl", "csv", "mot", "html"}
        if unknown:
            raise ValueError(f"Unknown export format(s): {', '.join(sorted(unknown))}")
        stride = skip + 1
        with contextlib.ExitStack() as stack:
            out_dir = stack.enter_context(_run_directory(output_dir))
            frame_iter, src_fps, width, height, total_frames = _iter_frames(
                source, max_frames=max_frames, fps=fps
            )

            capture_stack = stack.enter_context(contextlib.ExitStack())
            capture_stack.callback(frame_iter.close)
            frame_dt_s = stride / src_fps
            tracker = self._new_tracker(frame_dt_s)
            card_index = self._new_cards(
                tracker,
                stride,
                out_dir / "crops" if self.anonymize_mode == "none" else None,
            )
            floorplan = self.homography.fresh() if self.homography else None
            counter = self._new_counter()

            video_writer: cv2.VideoWriter | None = None
            video_path = out_dir / "annotated.mp4"
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_fps = src_fps / max(1, skip + 1)
                video_writer = cv2.VideoWriter(str(video_path), fourcc, out_fps, (width, height))
                if not video_writer.isOpened():
                    video_writer.release()
                    raise RuntimeError(f"Could not open MP4 video writer at {video_path}")
                capture_stack.callback(video_writer.release)

            counts: list[int] = []
            extensions = {"parquet": "parquet", "jsonl": "jsonl", "csv": "csv", "mot": "txt"}
            paths = {
                fmt: out_dir / f"tracks.{ext}"
                for fmt, ext in extensions.items()
                if fmt in export_set
            }
            writer_stack = stack.enter_context(contextlib.ExitStack())
            sink = ObservationWriter(paths, writer_stack)
            # Re-entry IDs can change retrospectively. Only that mode needs a disk spool.
            spool = (
                stack.enter_context(
                    tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir=out_dir)
                )
                if self.reid_merge and paths
                else None
            )
            trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
            all_events: list[dict[str, Any]] = []

            total_dets = 0
            recovered_count = 0
            total_inf_ms = total_trk_ms = 0.0
            observations = 0

            t_start = time.perf_counter()
            processed = 0
            seen_frames = 0

            stop = stack.enter_context(_stop_on_interrupt())
            try:
                for frame_idx, timestamp_s, frame_bgr in frame_iter:
                    if stop.is_set():
                        break
                    seen_frames += 1
                    if skip > 0 and (frame_idx % (skip + 1) != 0):
                        continue

                    fr, dets = self._process_frame(
                        frame_bgr,
                        frame_idx,
                        timestamp_s,
                        frame_dt_s,
                        tracker,
                        card_index,
                        floorplan,
                        counter,
                        record_floorplan=True,
                    )
                    tracks = fr.tracks
                    observations += len(tracks)
                    total_dets += fr.num_detections
                    total_inf_ms += fr.inference_ms
                    total_trk_ms += fr.tracking_ms
                    for ev in fr.events:
                        all_events.append(
                            {"frame_idx": frame_idx, "timestamp_s": round(timestamp_s, 3), **ev}
                        )

                    for tr in tracks:
                        if tr.recovered:
                            recovered_count += 1
                        if paths:
                            record = tr.to_record(frame_idx, timestamp_s)
                            if spool is None:
                                sink.append(record)
                            else:
                                spool.write(json.dumps(record) + "\n")
                        if save_video or display:
                            fx, fy = tr.foot_point
                            trail = trails[tr.track_id]
                            trail.append((int(round(fx)), int(round(fy))))
                            del trail[:-48]
                    if "html" in export_set:
                        counts.append(len(tracks))
                    processed += 1

                    elapsed_now = max(time.perf_counter() - t_start, 1e-6)
                    live_fps = processed / elapsed_now

                    if save_video or display:
                        # Anonymize every box that may hold a person, not just confirmed tracks.
                        coasting = (
                            tracker.coasting_tracks(3)
                            if hasattr(tracker, "coasting_tracks")
                            else []
                        )
                        anon_frame = self.anonymizer.apply(
                            frame_bgr, tracks, detections=dets, coasting=coasting
                        )
                        for c in coasting:
                            c.track_id = card_index.canonical_id(c.track_id)
                        rendered = draw_tracks(
                            anon_frame,
                            tracks,
                            coasting=coasting,
                            trails=trails,
                            fps=live_fps,
                            query=self.query,
                            anonymize_mode=self.anonymize_mode,
                            lines=self.lines,
                            line_counts=counter.counts if counter else None,
                        )
                        if video_writer is not None:
                            video_writer.write(rendered)
                        if display:
                            try:
                                cv2.imshow("Trackflow", rendered)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    break
                            except cv2.error:
                                display = False

                    if progress_callback is not None:
                        progress_callback(processed, (total_frames + skip) // stride, live_fps)
            finally:
                capture_stack.close()
                if display:
                    with contextlib.suppress(cv2.error):
                        cv2.destroyAllWindows()

            # Merges are decided a few frames into a returning person's new track; relabel the
            # observations recorded before that so exports use one ID per person.
            if processed == 0:
                if stop.is_set():
                    raise KeyboardInterrupt
                raise RuntimeError(f"No readable frames in {source}")
            if spool is not None:
                spool.seek(0)
                for line in spool:
                    rec = json.loads(line)
                    rec["track_id"] = card_index.canonical_id(rec["track_id"])
                    sink.append(rec)
            writer_stack.close()

            elapsed_s = max(time.perf_counter() - t_start, 1e-6)
            avg_fps = processed / elapsed_s
            avg_inf = total_inf_ms / processed
            avg_trk = total_trk_ms / processed

            device_label = getattr(self.detector, "active_provider", "custom")
            output_files = {fmt: str(path) for fmt, path in paths.items()}
            if save_video and video_path.is_file():
                output_files["video"] = str(video_path)

            # Persist Track Cards & diagnostic events (always written for `trackflow inspect` / `search`)
            cards_path = out_dir / "track_cards.json"
            card_index.save_json(
                cards_path, reid_backend=self.reid_backend, vlm_backend=self.vlm_backend
            )
            output_files["track_cards"] = str(cards_path)

            events_path = out_dir / "events.json"
            events_path.write_text(json.dumps(all_events, indent=2), encoding="utf-8")
            output_files["events"] = str(events_path)

            if floorplan is not None:
                fp_png = floorplan.render_floorplan(out_dir / "floorplan.png")
                fp_json = out_dir / "homography.json"
                floorplan.save_json(fp_json)
                output_files["floorplan"] = str(fp_png)
                output_files["homography"] = str(fp_json)

            summary = RunSummary(
                source=_source_label(source),
                model=self.model_name,
                model_revision=getattr(self.detector, "revision", None),
                precision=getattr(self.detector, "precision", None),
                tracking_threshold=self.threshold,
                tracker=getattr(tracker, "name", type(tracker).__name__)
                if self.tracker_factory
                else self.tracker_name,
                device=device_label,
                total_frames=seen_frames,
                processed_frames=processed,
                source_fps=src_fps,
                avg_fps=avg_fps,
                avg_inference_ms=avg_inf,
                avg_tracking_ms=avg_trk,
                total_detections=total_dets,
                total_track_observations=observations,
                frame_stride=stride,
                unique_tracks=len(card_index.cards),
                recovered_tracks=recovered_count,
                reid_merges=card_index.reid_merges,
                elapsed_s=elapsed_s,
                frame_width=width,
                frame_height=height,
                classes=self.classes,
                query=self.query,
                anonymize_mode=self.anonymize_mode,
                output_files=output_files,
                zone_dwell_totals_s=(
                    dict(floorplan.zone_dwell_totals_s) if floorplan is not None else {}
                ),
                line_counts=counter.counts if counter else {},
            )

            if "html" in export_set:
                html_path = write_run_report(
                    summary, counts, all_events, card_index.sorted_cards(), out_dir / "report.html"
                )
                summary.output_files["html"] = str(html_path)

            summary_path = out_dir / "summary.json"
            summary.output_files["summary"] = str(summary_path)
            tmp_summary = out_dir / ".summary.tmp"
            tmp_summary.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
            tmp_summary.replace(summary_path)
            return summary
