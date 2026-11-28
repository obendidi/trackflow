"""End-to-end Trackflow video/webcam/directory tracking pipeline."""

from __future__ import annotations

import contextlib
import json
import queue
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
from trackflow.mask_fallback import (
    MaskFallbackConfig,
    MaskFallbackResolver,
    MaskPropagator,
)
from trackflow.models import Precision
from trackflow.postprocess import postprocess_records
from trackflow.privacy import AnonymizeMode, PrivacyAnonymizer
from trackflow.reid import (
    QUERY_THRESHOLDS,
    ReidBackend,
    ReidEmbedder,
    TrackCardIndex,
    VlmBackend,
    VlmEmbedder,
    create_vlm_embedder,
)
from trackflow.sinks import ObservationWriter, draw_tracks, write_run_report
from trackflow.tracker import Tracker, compute_iou_matrix, create_tracker, iou_buffer_for_interval
from trackflow.types import Detection, FrameResult, RunSummary, Track

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


def _is_camera_source(source: str | Path | int) -> bool:
    return isinstance(source, int) or (
        isinstance(source, str)
        and (source.isdigit() or source.lower() == "camera")
        and not Path(source).exists()
    )


def _is_live_source(source: str | Path | int) -> bool:
    return _is_camera_source(source) or _is_stream_url(source)


_WRITER_STOP = object()


@contextlib.contextmanager
def _prefetch_frames(
    frame_iter: Generator[tuple[int, float, np.ndarray], None, None],
    *,
    maxsize: int = 2,
) -> Iterator[Iterator[tuple[int, float, np.ndarray]]]:
    """Decode frames one step ahead on a worker thread so file I/O overlaps inference."""
    q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=maxsize)
    cancel = threading.Event()

    def _put(item: tuple[str, Any]) -> bool:
        while not cancel.is_set():
            try:
                q.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _reader() -> None:
        try:
            for item in frame_iter:
                if not _put(("frame", item)):
                    return
            _put(("eof", None))
        except BaseException as exc:
            _put(("error", exc))
        finally:
            frame_iter.close()

    thread = threading.Thread(target=_reader, name="trackflow-frame-reader", daemon=True)
    thread.start()

    def _consume() -> Iterator[tuple[int, float, np.ndarray]]:
        while True:
            kind, payload = q.get()
            if kind == "frame":
                yield payload
            elif kind == "eof":
                return
            else:
                raise payload

    try:
        yield _consume()
    finally:
        cancel.set()
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        thread.join()


class _AsyncVideoWriter:
    """Render annotations and encode video frames on a worker thread with bounded backpressure."""

    def __init__(
        self,
        video_writer: cv2.VideoWriter,
        render_fn: Callable[..., np.ndarray],
        *,
        maxsize: int = 2,
    ) -> None:
        self._video_writer = video_writer
        self._render_fn = render_fn
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._abort = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker, name="trackflow-video-writer", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _WRITER_STOP:
                    return
                if self._abort.is_set():
                    continue
                rendered = self._render_fn(*item)
                self._video_writer.write(rendered)
        except BaseException as exc:
            self._error = exc
            self._abort.set()

    def submit(self, *args: Any) -> None:
        while True:
            if self._error is not None:
                raise self._error
            try:
                self._queue.put(args, timeout=0.05)
                return
            except queue.Full:
                continue

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if abort:
            self._abort.set()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        while self._thread.is_alive():
            try:
                self._queue.put(_WRITER_STOP, timeout=0.05)
                break
            except queue.Full:
                continue
        self._thread.join()
        if not abort and self._error is not None:
            raise self._error


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
    is_camera = _is_camera_source(source)
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


def _match_track_embeddings(
    tracks: list[Track],
    detections: list[Detection],
    frame_bgr: np.ndarray | None,
    reid_embedder: ReidEmbedder | None,
    *,
    seen_counts: dict[int, int] | None = None,
    sample_every: int = 3,
) -> list[np.ndarray | None]:
    """Attach a float16 Re-ID embedding to each track, reusing detection embeddings when present.

    When a tracker did not attach embeddings to detections (e.g. ByteTrack/OC-SORT with
    ``--reid osnet``), crops are encoded on track birth and every ``sample_every`` observations of
    each raw track to match ``TrackCardIndex.reid_sample_every`` and bound CPU overhead.
    """
    if not tracks:
        return []

    out: list[np.ndarray | None] = [None] * len(tracks)
    det_with_emb = [d for d in detections if d.embedding is not None]
    if det_with_emb:
        t_boxes = np.array([t.bbox for t in tracks], dtype=np.float64).reshape(-1, 4)
        d_boxes = np.array([d.bbox for d in det_with_emb], dtype=np.float64).reshape(-1, 4)
        iou = compute_iou_matrix(t_boxes, d_boxes)
        for i, tr in enumerate(tracks):
            j = int(np.argmax(iou[i]))
            if float(iou[i, j]) >= 0.5 and det_with_emb[j].class_id == tr.class_id:
                emb = det_with_emb[j].embedding
                if emb is not None:
                    out[i] = np.asarray(emb, dtype=np.float16)

    if reid_embedder is None or frame_bgr is None:
        return out

    missing: list[int] = []
    for i, tr in enumerate(tracks):
        obs_n = seen_counts.get(tr.track_id, 0) if seen_counts is not None else 0
        if seen_counts is not None:
            seen_counts[tr.track_id] = obs_n + 1
        if (
            out[i] is None
            and tr.class_name.lower() == "person"
            and (sample_every <= 1 or obs_n % sample_every == 0)
        ):
            missing.append(i)
    if missing:
        crops = [_crop(frame_bgr, tracks[i].bbox) for i in missing]
        for i, emb in zip(missing, reid_embedder.encode_images(crops), strict=True):
            if emb is not None:
                out[i] = np.asarray(emb, dtype=np.float16)
    return out


def _pack_postprocess_embeddings(
    embs: Sequence[np.ndarray | None],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pack per-observation optional embeddings into dense ``(N, D)`` float32 + bool mask."""
    dim = next((int(e.shape[0]) for e in embs if e is not None), 0)
    if dim <= 0:
        return None, None
    dense = np.zeros((len(embs), dim), dtype=np.float32)
    valid = np.zeros((len(embs),), dtype=bool)
    for i, emb in enumerate(embs):
        if emb is not None:
            dense[i] = np.asarray(emb, dtype=np.float32)
            valid[i] = True
    return dense, valid


def _remap_tracker_events(
    raw_events: list[dict[str, Any]],
    records: list[dict[str, Any]],
    source_row_indices: np.ndarray,
    raw_tids: list[int],
    raw_frames: list[int],
) -> list[dict[str, Any]]:
    """Remap Pass-1 tracker lifecycle events from raw IDs to finalized post-processed IDs."""
    exact: dict[tuple[int, int], int] = {}
    by_raw: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rec, si in zip(records, source_row_indices.tolist(), strict=True):
        if si < 0:
            continue
        fid = int(rec["track_id"])
        rid = raw_tids[si]
        f_i = raw_frames[si]
        exact[(f_i, rid)] = fid
        by_raw[rid].append((f_i, fid))

    def _resolve(frame_idx: int, raw_id: int) -> int | None:
        hit = exact.get((frame_idx, raw_id))
        if hit is not None:
            return hit
        hist = by_raw.get(raw_id)
        if not hist:
            return None
        return min(hist, key=lambda pair: abs(pair[0] - frame_idx))[1]

    remapped: list[dict[str, Any]] = []
    for ev in raw_events:
        f_i = int(ev.get("frame_idx", 0))
        et = ev.get("type")
        if et == "possible_id_switch":
            new_id = _resolve(f_i, int(ev["new_track_id"]))
            prev_id = _resolve(f_i, int(ev["previous_track_id"]))
            if new_id is None or prev_id is None or new_id == prev_id:
                continue
            remapped.append({**ev, "new_track_id": new_id, "previous_track_id": prev_id})
        elif "track_id" in ev:
            fid = _resolve(f_i, int(ev["track_id"]))
            if fid is None:
                continue
            remapped.append({**ev, "track_id": fid})
        else:
            remapped.append(ev)
    return remapped


class TrackflowPipeline:
    """Unified real-time tracking pipeline combining detection, MOT, Re-ID/NL query, privacy & floorplan."""

    def __init__(
        self,
        *,
        model: str | Path | None = None,
        precision: Precision = "fp32",
        tracker: str = "bytetrack",
        use_cmc: bool = True,
        device: str = "auto",
        threshold: float = 0.45,
        classes: Sequence[str] | None = ("person",),
        reid: ReidBackend = "none",
        vlm: VlmBackend | None = None,
        reid_merge: bool = False,
        postprocess: bool = False,
        query: str | None = None,
        query_threshold: float | None = None,
        anonymize: AnonymizeMode = "none",
        homography: FloorplanHomography | str | Path | None = None,
        lines: Sequence[CountingLine] = (),
        detector: Detector | None = None,
        tracker_factory: Callable[[], Tracker] | None = None,
        reid_embedder: ReidEmbedder | None = None,
        vlm_embedder: VlmEmbedder | None = None,
        mask_fallback: str | None = None,
        mask_propagator: MaskPropagator | None = None,
        mask_fallback_config: MaskFallbackConfig | None = None,
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
        self.use_cmc = bool(use_cmc)
        self.tracker_factory = tracker_factory
        if tracker_factory is None:
            create_tracker(tracker, use_cmc=self.use_cmc)  # reject an unknown name before any model loads
        self.threshold = float(threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        self._allowed = _allowed_classes(classes)
        self.classes = sorted(self._allowed) if self._allowed is not None else ["all"]
        self.query = query.strip() if query else None
        self.anonymize_mode: AnonymizeMode = anonymize

        # Validate mask fallback options early before loading detectors/embedders.
        mf_name = mask_fallback.strip() if isinstance(mask_fallback, str) else None
        if mf_name and mf_name.lower() == "none":
            mf_name = None
        if mask_fallback_config is not None and mf_name is None and mask_propagator is None:
            raise ValueError(
                "--mask-budget / mask_fallback_config requires --mask-fallback or mask_propagator."
            )
        if mask_propagator is not None:
            if self.anonymize_mode != "none" and getattr(
                mask_propagator, "writes_frames_to_disk", False
            ):
                raise ValueError(
                    "Cannot use a mask propagator that writes raw frames to disk "
                    "(such as Sam3MaskPropagator) when anonymize != 'none'. "
                    "Use an in-memory propagator such as Sam2MaskPropagator."
                )
            self.mask_propagator: MaskPropagator | None = mask_propagator
            self.mask_fallback_backend: str | None = getattr(
                mask_propagator, "model_name", type(mask_propagator).__name__
            )
        elif mf_name is not None:
            from trackflow.sam2_adapter import create_mask_propagator

            self.mask_propagator = create_mask_propagator(
                mf_name, device=device, cache_dir=cache_dir
            )
            self.mask_fallback_backend = mf_name
        else:
            self.mask_propagator = None
            self.mask_fallback_backend = None
        self.mask_fallback_config = mask_fallback_config

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
        self.query_threshold = float(
            QUERY_THRESHOLDS.get(self.vlm_backend, 0.22)
            if query_threshold is None
            else query_threshold
        )
        self.reid_merge = bool(reid_merge)
        self.postprocess = bool(postprocess)
        if self.query and self.vlm_backend == "none":
            raise ValueError(
                "A text query needs a vision-language model. Choose clip, siglip, siglip2 or embeddinggemma."
            )

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
                device=device,
                cache_dir=cache_dir,
            )
            if self.reid_backend != "none"
            else None
        )
        self.vlm_embedder = vlm_embedder or (
            create_vlm_embedder(self.vlm_backend, device=device, cache_dir=cache_dir)
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

    def _new_mask_resolver(self) -> MaskFallbackResolver | None:
        if self.mask_propagator is None:
            return None
        return MaskFallbackResolver(self.mask_propagator, self.mask_fallback_config)

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
            use_cmc=self.use_cmc,
        )

    def _new_cards(
        self,
        tracker: Tracker,
        stride: int,
        crops_dir: Path | None = None,
        *,
        enable_online_reid_merge: bool | None = None,
    ) -> TrackCardIndex:
        online_merge = self.reid_merge if enable_online_reid_merge is None else enable_online_reid_merge
        buffer = getattr(tracker, "track_buffer", None)
        if online_merge and buffer is None:
            raise ValueError(
                "A custom tracker using reid_merge must expose its track_buffer in processed frames."
            )
        return TrackCardIndex(
            vlm=self.vlm_embedder,
            reid=self.reid_embedder,
            query=self.query,
            query_threshold=self.query_threshold,
            reid_merge=online_merge,
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
        mask_resolver: MaskFallbackResolver | None = None,
        *,
        record_floorplan: bool = False,
    ) -> tuple[FrameResult, list[Detection], list[int]]:
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
        raw_track_ids = [int(tr.track_id) for tr in tracks]
        mask_events = (
            mask_resolver.observe(idx, frame, tracks) if mask_resolver is not None else []
        )
        tracked = time.perf_counter()
        if floorplan is not None:
            for tr in tracks:
                if record_floorplan:
                    tr.world_xy, tr.zone = floorplan.observe(tr.track_id, tr.foot_point, dt_s=dt)
                else:
                    tr.world_xy = floorplan.project(tr.foot_point)
                    tr.zone = floorplan.locate_zone(tr.world_xy)
        events = list(getattr(tracker, "last_events", ()))
        events.extend(mask_events)
        if counter is not None:
            # Raw tracker IDs: a Re-ID merge would join a track to a point from before it left.
            events.extend(counter.update(tracks, timestamp))
        if cards is not None:
            for ev in mask_events:
                for src_id, dst_id in ev.get("relink_pairs", ()):
                    cards.reconcile_relink(int(src_id), int(dst_id))
            events.extend(cards.update(idx, timestamp, frame, tracks, frame_dt_s=dt))
        return (
            FrameResult(
                frame_idx=idx,
                timestamp_s=timestamp,
                tracks=tracks,
                num_detections=len(targets),
                inference_ms=(detected - start) * 1000,
                tracking_ms=(tracked - detected) * 1000,
                events=events,
            ),
            raw,
            raw_track_ids,
        )

    def stream(self, frames: Iterable[np.ndarray], *, fps: float = 30.0) -> Iterator[FrameResult]:
        """Yield observations for consecutive BGR uint8 frames without writing any files.

        Each call starts a fresh tracker. With re-entry merging or mask fallback enabled,
        committed fixes apply to subsequent frames; observations already yielded keep their
        original ID.
        """
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite.")
        dt = 1.0 / fps
        tracker = self._new_tracker(dt)
        cards = self._new_cards(tracker, 1) if self.vlm_embedder or self.reid_merge else None
        counter = self._new_counter()
        mask_resolver = self._new_mask_resolver()
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
            result, _, _ = self._process_frame(
                frame, idx, idx / fps, dt, tracker, cards, self.homography, counter, mask_resolver
            )
            yield result

    def _finalize_postprocess(
        self,
        pp_records: list[dict[str, Any]],
        pp_embs: list[np.ndarray | None],
        pp_cmc_by_frame: dict[int, np.ndarray],
        pp_frame_indices: list[int],
        raw_events: list[dict[str, Any]],
        *,
        tracker: Tracker,
        card_index: TrackCardIndex,
        floorplan: FloorplanHomography | None,
        counter: LineCounter | None,
        src_fps: float,
        frame_dt_s: float,
        stride: int,
        width: int,
        height: int,
    ) -> tuple[
        list[dict[str, Any]],
        dict[int, list[dict[str, Any]]],
        list[dict[str, Any]],
        dict[int, dict[str, dict[str, int]]],
        int,
    ]:
        """Run offline post-processing and recompute floorplan, line counts, cards, and events."""
        raw_tids = [int(r["track_id"]) for r in pp_records]
        raw_frames = [int(r["frame_idx"]) for r in pp_records]
        emb_dense, emb_valid = _pack_postprocess_embeddings(pp_embs)

        cmc_warps: np.ndarray | None = None
        cmc_valid: np.ndarray | None = None
        if pp_cmc_by_frame:
            n_cmc = max(pp_frame_indices[-1] + 1, max(pp_cmc_by_frame) + 1)
            cmc_warps = np.broadcast_to(
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                (n_cmc, 2, 3),
            ).copy()
            cmc_valid = np.zeros((n_cmc,), dtype=bool)
            for f_i, w_mat in pp_cmc_by_frame.items():
                cmc_warps[f_i] = w_mat
                cmc_valid[f_i] = True

        buf = getattr(tracker, "track_buffer", None)
        min_exit_gap = max(8, ((buf or 0) + 1) * stride)
        pp_res = postprocess_records(
            pp_records,
            fps=src_fps,
            frame_width=width,
            frame_height=height,
            embeddings=emb_dense,
            emb_valid=emb_valid,
            cmc_warps=cmc_warps,
            cmc_valid=cmc_valid,
            frame_stride=stride,
            allow_long_reid_merge=self.reid_merge,
            min_exit_gap_frames=min_exit_gap,
        )
        final_records = pp_res.records

        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_frame_src: dict[int, list[int]] = defaultdict(list)
        for rec, si in zip(final_records, pp_res.source_row_indices.tolist(), strict=True):
            f_i = int(rec["frame_idx"])
            by_frame[f_i].append(rec)
            by_frame_src[f_i].append(int(si))

        if floorplan is not None:
            for f_i in pp_frame_indices:
                for rec in by_frame.get(f_i, ()):
                    foot = (0.5 * (rec["x1"] + rec["x2"]), rec["y2"])
                    wxy, zone = floorplan.observe(int(rec["track_id"]), foot, dt_s=frame_dt_s)
                    rec["world_x"] = round(float(wxy[0]), 3)
                    rec["world_y"] = round(float(wxy[1]), 3)
                    rec["zone"] = zone

        final_events = _remap_tracker_events(
            raw_events, final_records, pp_res.source_row_indices, raw_tids, raw_frames
        )
        for ev in pp_res.reid_merge_events:
            final_events.append(
                {**ev, "timestamp_s": round(int(ev["frame_idx"]) / max(src_fps, 1e-6), 3)}
            )

        line_counts_by_frame: dict[int, dict[str, dict[str, int]]] = {}
        if counter is not None:
            last_seen_frame: dict[int, int] = {}
            last_raw_id: dict[int, int] = {}
            for f_i in pp_frame_indices:
                f_recs = by_frame.get(f_i, ())
                f_srcs = by_frame_src.get(f_i, ())
                if f_recs:
                    f_tracks: list[Track] = []
                    ts = float(f_recs[0]["timestamp_s"])
                    for rec, si in zip(f_recs, f_srcs, strict=True):
                        tid = int(rec["track_id"])
                        prev_frame = last_seen_frame.get(tid)
                        if si >= 0:
                            cur_raw = raw_tids[si]
                            prev_raw = last_raw_id.get(tid)
                            # Reset line state only across an un-interpolated stitched exit gap
                            # between different raw IDs (never mid-trajectory after interpolation).
                            if (
                                prev_raw is not None
                                and prev_frame is not None
                                and cur_raw != prev_raw
                                and f_i - prev_frame >= min_exit_gap
                            ):
                                counter.forget_track(tid)
                            last_raw_id[tid] = cur_raw
                        last_seen_frame[tid] = f_i
                        f_tracks.append(
                            Track(
                                track_id=tid,
                                bbox=(rec["x1"], rec["y1"], rec["x2"], rec["y2"]),
                                confidence=rec["confidence"],
                                class_id=rec["class_id"],
                                class_name=rec["class_name"],
                            )
                        )
                    for ev in counter.update(f_tracks, ts):
                        final_events.append({"frame_idx": f_i, "timestamp_s": round(ts, 3), **ev})
                line_counts_by_frame[f_i] = {name: dict(c) for name, c in counter.counts.items()}

        card_index.rebuild_from_postprocess(
            final_records,
            pp_res.source_row_indices,
            raw_tids,
            frame_dt_s=frame_dt_s,
            reid_merges=pp_res.reid_merges,
        )
        final_events.append(
            {
                "frame_idx": pp_frame_indices[-1],
                "timestamp_s": round(pp_frame_indices[-1] / src_fps, 3),
                "type": "offline_postprocess",
                "unique_tracks_before": pp_res.unique_tracks_before,
                "unique_tracks_after": pp_res.unique_tracks_after,
                "interpolated_rows": pp_res.interpolated_rows,
                "merges": pp_res.merges,
                "reid_merges": pp_res.reid_merges,
                "splits": pp_res.splits,
                "elapsed_ms": round(pp_res.elapsed_ms, 2),
            }
        )
        final_events.sort(key=lambda e: int(e.get("frame_idx", 0)))
        return final_records, by_frame, final_events, line_counts_by_frame, min_exit_gap

    def _replay_postprocess_video(
        self,
        source: str | Path | int,
        video_path: Path,
        by_frame: dict[int, list[dict[str, Any]]],
        pp_privacy_masks: dict[int, tuple[list[Detection], list[Track]]],
        line_counts_by_frame: dict[int, dict[str, dict[str, int]]],
        *,
        seen_frames: int,
        src_fps: float,
        skip: int,
        width: int,
        height: int,
        gap_reset_frames: int,
    ) -> None:
        """Render ``annotated.mp4`` from finalized post-processed tracks while preserving masks."""
        replay_iter, _, _, _, _ = _iter_frames(source, max_frames=seen_frames, fps=src_fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = src_fps / max(1, skip + 1)
        v_writer = cv2.VideoWriter(str(video_path), fourcc, out_fps, (width, height))
        if not v_writer.isOpened():
            replay_iter.close()
            v_writer.release()
            video_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not open MP4 video writer at {video_path}")
        pp_trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
        last_trail_frame: dict[int, int] = {}
        written_frames = 0
        try:
            with _stop_on_interrupt() as replay_stop:
                for f_i, _ts, f_bgr in replay_iter:
                    if replay_stop.is_set():
                        break
                    if skip > 0 and (f_i % (skip + 1) != 0):
                        continue
                    f_recs = by_frame.get(f_i, ())
                    f_tracks = [
                        Track(
                            track_id=int(r["track_id"]),
                            bbox=(r["x1"], r["y1"], r["x2"], r["y2"]),
                            confidence=float(r["confidence"]),
                            class_id=int(r["class_id"]),
                            class_name=str(r["class_name"]),
                            query_score=r["query_score"],
                            query_match=bool(r["query_match"]),
                            zone=r["zone"],
                        )
                        for r in f_recs
                    ]
                    for tr in f_tracks:
                        prev_f = last_trail_frame.get(tr.track_id)
                        if prev_f is not None and f_i - prev_f >= gap_reset_frames:
                            pp_trails[tr.track_id].clear()
                        last_trail_frame[tr.track_id] = f_i
                        fx, fy = tr.foot_point
                        trl = pp_trails[tr.track_id]
                        trl.append((int(round(fx)), int(round(fy))))
                        del trl[:-48]
                    saved_dets, saved_coasting = pp_privacy_masks.get(f_i, ([], []))
                    # In skeleton mode, draw only finalized tracks (avoid double-drawing stale coasting).
                    mask_coasting = () if self.anonymize_mode == "skeleton" else saved_coasting
                    anon_frame = self.anonymizer.apply(
                        f_bgr,
                        f_tracks,
                        detections=saved_dets,
                        coasting=mask_coasting,
                    )
                    rendered = draw_tracks(
                        anon_frame,
                        f_tracks,
                        trails=pp_trails,
                        query=self.query,
                        anonymize_mode=self.anonymize_mode,
                        lines=self.lines,
                        line_counts=line_counts_by_frame.get(f_i) if self.lines else None,
                    )
                    v_writer.write(rendered)
                    written_frames += 1
        finally:
            replay_iter.close()
            v_writer.release()
            if written_frames == 0:
                video_path.unlink(missing_ok=True)

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
        if self.postprocess and _is_live_source(source):
            raise ValueError("--postprocess requires a recorded video file or image directory.")
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
            # In offline --postprocess mode, defer Re-ID merging to the global post-processor
            # instead of mutating raw IDs greedily mid-pass.
            card_index = self._new_cards(
                tracker,
                stride,
                out_dir / "crops" if self.anonymize_mode == "none" else None,
                enable_online_reid_merge=False if self.postprocess else None,
            )
            floorplan = self.homography.fresh() if self.homography else None
            counter = self._new_counter()
            mask_resolver = self._new_mask_resolver()

            defer_video = self.postprocess and save_video
            video_writer: cv2.VideoWriter | None = None
            video_path = out_dir / "annotated.mp4"
            if save_video and not defer_video:
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
            # Re-entry and mask-fallback IDs can change retrospectively. Only non-postprocess
            # modes need a disk spool; offline postprocess buffers full-precision records in memory.
            spool = (
                stack.enter_context(
                    tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir=out_dir)
                )
                if (self.reid_merge or mask_resolver is not None) and not self.postprocess and paths
                else None
            )
            pp_records: list[dict[str, Any]] = []
            pp_embs: list[np.ndarray | None] = []
            pp_seen_counts: dict[int, int] = {}
            pp_cmc_by_frame: dict[int, np.ndarray] = {}
            pp_frame_indices: list[int] = []
            pp_privacy_masks: dict[int, tuple[list[Detection], list[Track]]] = {}
            trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
            all_events: list[dict[str, Any]] = []

            total_dets = 0
            recovered_count = 0
            total_inf_ms = total_trk_ms = 0.0
            observations = 0

            t_start = time.perf_counter()
            processed = 0
            seen_frames = 0

            def _render_frame(
                frame_bgr: np.ndarray,
                tracks: list[Any],
                dets: list[Detection],
                coasting: list[Any],
                trails_snap: dict[int, list[tuple[int, int]]],
                live_fps: float,
                line_counts_snap: dict[str, dict[str, int]] | None,
            ) -> np.ndarray:
                anon_frame = self.anonymizer.apply(
                    frame_bgr, tracks, detections=dets, coasting=coasting
                )
                return draw_tracks(
                    anon_frame,
                    tracks,
                    coasting=coasting,
                    trails=trails_snap,
                    fps=live_fps,
                    query=self.query,
                    anonymize_mode=self.anonymize_mode,
                    lines=self.lines,
                    line_counts=line_counts_snap,
                )

            frames: Iterator[tuple[int, float, np.ndarray]] = (
                capture_stack.enter_context(_prefetch_frames(frame_iter))
                if not _is_live_source(source) and total_frames != 1
                else frame_iter
            )
            async_writer = (
                _AsyncVideoWriter(video_writer, _render_frame)
                if video_writer is not None and not display
                else None
            )

            interrupted_pass1 = False
            aborted = False
            with _stop_on_interrupt() as stop:
                try:
                    for frame_idx, timestamp_s, frame_bgr in frames:
                        if stop.is_set():
                            break
                        seen_frames += 1
                        if skip > 0 and (frame_idx % (skip + 1) != 0):
                            continue

                        fr, dets, raw_track_ids = self._process_frame(
                            frame_bgr,
                            frame_idx,
                            timestamp_s,
                            frame_dt_s,
                            tracker,
                            card_index,
                            floorplan if not self.postprocess else None,
                            counter if not self.postprocess else None,
                            mask_resolver,
                            record_floorplan=not self.postprocess,
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

                        if self.postprocess:
                            pp_frame_indices.append(frame_idx)
                            warp = getattr(tracker, "last_cmc_warp", None)
                            if warp is not None:
                                pp_cmc_by_frame[frame_idx] = np.asarray(warp, dtype=np.float32)
                            obs_embs = _match_track_embeddings(
                                tracks,
                                dets,
                                frame_bgr,
                                self.reid_embedder,
                                seen_counts=pp_seen_counts,
                                sample_every=card_index.reid_sample_every,
                            )
                            for tr, raw_tid, emb in zip(tracks, raw_track_ids, obs_embs, strict=True):
                                if tr.recovered:
                                    recovered_count += 1
                                rec = tr.to_record(frame_idx, timestamp_s)
                                # Keep full-precision coordinates for offline post-processing.
                                rec["x1"], rec["y1"], rec["x2"], rec["y2"] = tr.bbox
                                if mask_resolver is not None:
                                    rec["_raw_track_id"] = raw_tid
                                pp_records.append(rec)
                                pp_embs.append(emb)
                                if display:
                                    fx, fy = tr.foot_point
                                    trail = trails[tr.track_id]
                                    trail.append((int(round(fx)), int(round(fy))))
                                    del trail[:-48]
                        else:
                            for tr, raw_tid in zip(tracks, raw_track_ids, strict=True):
                                if tr.recovered:
                                    recovered_count += 1
                                if paths:
                                    record = tr.to_record(frame_idx, timestamp_s)
                                    if spool is None:
                                        sink.append(record)
                                    else:
                                        if mask_resolver is not None:
                                            record["_raw_track_id"] = raw_tid
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

                        if defer_video and self.anonymize_mode != "none":
                            coasting = (
                                tracker.coasting_tracks(3)
                                if hasattr(tracker, "coasting_tracks")
                                else []
                            )
                            # Preserve all raw detections, Pass-1 track boxes & coasting boxes so
                            # Pass-2 privacy masking covers every box even after smoothing shifts.
                            raw_track_masks = [
                                Track(tr.track_id, tr.bbox, tr.confidence, tr.class_id, tr.class_name)
                                for tr in tracks
                            ]
                            pp_privacy_masks[frame_idx] = (
                                [
                                    Detection(
                                        bbox=d.bbox,
                                        confidence=d.confidence,
                                        class_id=d.class_id,
                                        class_name=d.class_name,
                                    )
                                    for d in dets
                                ],
                                [*coasting, *raw_track_masks],
                            )

                        if (save_video and not defer_video) or display:
                            # Anonymize every box that may hold a person, not just confirmed tracks.
                            coasting = (
                                tracker.coasting_tracks(3)
                                if hasattr(tracker, "coasting_tracks")
                                else []
                            )
                            for c in coasting:
                                cid = (
                                    mask_resolver.resolve_id(frame_idx, c.track_id)
                                    if mask_resolver is not None
                                    else c.track_id
                                )
                                c.track_id = card_index.canonical_id(cid)
                            if async_writer is not None:
                                trails_snap = {
                                    tr.track_id: trails[tr.track_id][-32:] for tr in tracks
                                }
                                line_counts_snap = (
                                    {k: dict(v) for k, v in counter.counts.items()}
                                    if counter
                                    else None
                                )
                                async_writer.submit(
                                    frame_bgr,
                                    tracks,
                                    dets,
                                    coasting,
                                    trails_snap,
                                    live_fps,
                                    line_counts_snap,
                                )
                            else:
                                rendered = _render_frame(
                                    frame_bgr,
                                    tracks,
                                    dets,
                                    coasting,
                                    trails,
                                    live_fps,
                                    counter.counts if counter else None,
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
                except BaseException:
                    aborted = True
                    raise
                finally:
                    interrupted_pass1 = stop.is_set()
                    try:
                        if async_writer is not None:
                            async_writer.close(abort=aborted)
                    finally:
                        capture_stack.close()
                        if display:
                            with contextlib.suppress(cv2.error):
                                cv2.destroyAllWindows()

            # Merges are decided a few frames into a returning person's new track; relabel the
            # observations recorded before that so exports use one ID per person.
            if processed == 0:
                if interrupted_pass1:
                    raise KeyboardInterrupt
                raise RuntimeError(f"No readable frames in {source}")

            if self.postprocess:
                if mask_resolver is not None:
                    for rec in pp_records:
                        raw_tid = rec.pop("_raw_track_id", rec["track_id"])
                        rec["track_id"] = mask_resolver.resolve_id(rec["frame_idx"], raw_tid)
                (
                    final_records,
                    by_frame,
                    all_events,
                    line_counts_by_frame,
                    gap_reset_frames,
                ) = self._finalize_postprocess(
                    pp_records,
                    pp_embs,
                    pp_cmc_by_frame,
                    pp_frame_indices,
                    all_events,
                    tracker=tracker,
                    card_index=card_index,
                    floorplan=floorplan,
                    counter=counter,
                    src_fps=src_fps,
                    frame_dt_s=frame_dt_s,
                    stride=stride,
                    width=width,
                    height=height,
                )
                observations = len(final_records)
                if paths:
                    for rec in final_records:
                        sink.append(rec)
                if "html" in export_set:
                    counts = [len(by_frame.get(f_i, ())) for f_i in pp_frame_indices]
                if defer_video:
                    self._replay_postprocess_video(
                        source,
                        video_path,
                        by_frame,
                        pp_privacy_masks,
                        line_counts_by_frame,
                        seen_frames=seen_frames,
                        src_fps=src_fps,
                        skip=skip,
                        width=width,
                        height=height,
                        gap_reset_frames=gap_reset_frames,
                    )
            elif spool is not None:
                spool.seek(0)
                for line in spool:
                    rec = json.loads(line)
                    raw_tid = rec.pop("_raw_track_id", rec["track_id"])
                    tid = (
                        mask_resolver.resolve_id(rec["frame_idx"], raw_tid)
                        if mask_resolver is not None
                        else rec["track_id"]
                    )
                    rec["track_id"] = card_index.canonical_id(tid)
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
                postprocessed=self.postprocess,
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
                mask_fallback=(
                    {
                        "backend": self.mask_fallback_backend,
                        **mask_resolver.stats.to_dict(),
                    }
                    if mask_resolver is not None
                    else None
                ),
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
