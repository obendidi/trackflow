"""End-to-end Trackflow video/webcam/directory tracking pipeline."""

from __future__ import annotations

import contextlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from trackflow.detector import OnnxDetrDetector
from trackflow.homography import FloorplanHomography
from trackflow.models import Precision
from trackflow.privacy import AnonymizeMode, PrivacyAnonymizer
from trackflow.reid import AppearanceEmbedder, ReidBackend, TrackCardIndex
from trackflow.sinks import draw_tracks, write_csv, write_html_report, write_jsonl, write_parquet
from trackflow.tracker import create_tracker
from trackflow.types import Detection, FrameResult, RunSummary

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_frames(
    source: str | int,
    *,
    max_frames: int | None = None,
) -> tuple[Iterator[tuple[int, float, np.ndarray]], float, int, int, int]:
    """Open a video file, webcam index, or directory of images."""
    if isinstance(source, int) or str(source).isdigit() or str(source).lower() == "camera":
        cam_idx = 0 if str(source).lower() == "camera" else int(source)
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {cam_idx}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        def _gen_cam() -> Iterator[tuple[int, float, np.ndarray]]:
            idx = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield idx, idx / fps, frame
                    idx += 1
                    if max_frames is not None and idx >= max_frames:
                        break
            finally:
                cap.release()

        return _gen_cam(), float(fps), w, h, max_frames or 0

    src_path = Path(str(source))
    if src_path.is_dir():
        imgs = sorted(p for p in src_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not imgs:
            raise FileNotFoundError(f"No images found in directory: {src_path}")
        first = cv2.imread(str(imgs[0]))
        if first is None:
            raise RuntimeError(f"Failed to read image: {imgs[0]}")
        h, w = first.shape[:2]
        fps = 25.0
        total = min(len(imgs), max_frames) if max_frames is not None else len(imgs)

        def _gen_dir() -> Iterator[tuple[int, float, np.ndarray]]:
            for idx, p in enumerate(imgs[:total]):
                frame = cv2.imread(str(p))
                if frame is not None:
                    yield idx, idx / fps, frame

        return _gen_dir(), fps, w, h, total

    if not src_path.is_file():
        raise FileNotFoundError(f"Video source not found: {src_path}")

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and (total == 0 or max_frames < total):
        total = max_frames

    def _gen_vid() -> Iterator[tuple[int, float, np.ndarray]]:
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / fps, frame
                idx += 1
                if max_frames is not None and idx >= max_frames:
                    break
        finally:
            cap.release()

    return _gen_vid(), float(fps), w, h, total


class TrackflowPipeline:
    """Unified real-time tracking pipeline combining detection, MOT, Re-ID/NL query, privacy & floorplan."""

    def __init__(
        self,
        *,
        model: str = "rtdetr-v2-r18",
        precision: Precision = "fp32",
        tracker: str = "bytetrack",
        device: str = "auto",
        threshold: float = 0.45,
        classes: Sequence[str] | None = ("person",),
        reid: ReidBackend = "none",
        query: str | None = None,
        query_threshold: float = 0.22,
        anonymize: AnonymizeMode = "none",
        homography: FloorplanHomography | str | Path | None = None,
        detector: Any | None = None,
        embedder: AppearanceEmbedder | None = None,
    ) -> None:
        self.model_name = model
        self.tracker_name = tracker
        self.threshold = float(threshold)
        self.classes = list(classes) if classes else ["all"]
        self.query = query.strip() if query else None
        self.query_threshold = float(query_threshold)
        self.anonymize_mode: AnonymizeMode = anonymize

        # Automatically enable CLIP when a natural-language --query is passed without --reid
        effective_reid: ReidBackend = reid
        if self.query and effective_reid in ("none", "color") and embedder is None:
            effective_reid = "clip"
        self.reid_backend: ReidBackend = effective_reid

        self.detector = (
            detector
            if detector is not None
            else OnnxDetrDetector(
                model=model,
                precision=precision,
                device=device,
                threshold=min(0.15, threshold),  # keep low-conf tier for ByteTrack 2nd stage!
                classes=classes,
            )
        )
        self.embedder = (
            embedder
            if embedder is not None
            else (
                AppearanceEmbedder(backend=self.reid_backend, device=device)
                if self.reid_backend != "none"
                else None
            )
        )
        self.anonymizer = PrivacyAnonymizer(mode=anonymize)
        if isinstance(homography, (str, Path)):
            self.homography: FloorplanHomography | None = FloorplanHomography.from_json(homography)
        else:
            self.homography = homography

    def run(
        self,
        source: str | int,
        *,
        output_dir: str | Path = "runs/latest",
        skip: int = 0,
        max_frames: int | None = None,
        save_video: bool = True,
        exports: Sequence[str] = ("parquet", "jsonl", "html"),
        display: bool = False,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> RunSummary:
        """Execute tracking on ``source`` and write requested artifacts into ``output_dir``."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        export_set = {e.lower().strip() for e in exports}

        frame_iter, src_fps, width, height, total_frames = _iter_frames(
            source, max_frames=max_frames
        )

        tracker = create_tracker(
            self.tracker_name,
            track_high_thresh=self.threshold,
            track_low_thresh=max(0.10, self.threshold * 0.35),
            new_track_thresh=min(0.95, self.threshold + 0.05),
        )
        card_index = TrackCardIndex(
            embedder=self.embedder,
            query=self.query,
            query_threshold=self.query_threshold,
            crops_dir=out_dir / "crops",
        )

        video_writer: cv2.VideoWriter | None = None
        video_path = out_dir / "annotated.mp4"
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = src_fps / max(1, skip + 1)
            video_writer = cv2.VideoWriter(str(video_path), fourcc, out_fps, (width, height))

        frame_results: list[FrameResult] = []
        records: list[dict[str, Any]] = []
        trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
        all_events: list[dict[str, Any]] = []

        total_dets = 0
        recovered_count = 0
        inf_ms_list: list[float] = []
        trk_ms_list: list[float] = []
        frame_dt_s = (skip + 1) / max(src_fps, 1e-3)

        t_start = time.perf_counter()
        processed = 0
        seen_frames = 0

        try:
            for frame_idx, timestamp_s, frame_bgr in frame_iter:
                seen_frames += 1
                if skip > 0 and (frame_idx % (skip + 1) != 0):
                    continue

                t0 = time.perf_counter()
                dets: list[Detection] = self.detector.detect(frame_bgr)
                inf_ms = (time.perf_counter() - t0) * 1000.0
                inf_ms_list.append(inf_ms)
                total_dets += len(dets)

                # Attach per-detection embeddings when BoT-SORT uses Re-ID
                if (
                    self.tracker_name.lower().replace("-", "_") == "botsort"
                    and self.embedder is not None
                    and self.embedder.backend != "none"
                ):
                    for d in dets:
                        if d.confidence >= self.threshold:
                            x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            if x2 - x1 >= 4 and y2 - y1 >= 4:
                                d.embedding = self.embedder.encode_image(frame_bgr[y1:y2, x1:x2])

                t1 = time.perf_counter()
                tracks = tracker.update(dets, frame_bgr=frame_bgr)
                trk_ms = (time.perf_counter() - t1) * 1000.0
                trk_ms_list.append(trk_ms)

                # Spatial homography projection & dwell zones
                if self.homography is not None:
                    for tr in tracks:
                        wxy, zone = self.homography.observe(
                            tr.track_id, tr.foot_point, dt_s=frame_dt_s
                        )
                        tr.world_xy = wxy
                        tr.zone = zone

                # Persistent Track Cards, cross-exit Re-ID merges, and NL query scoring
                reid_events = card_index.update(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    frame_bgr=frame_bgr,
                    tracks=tracks,
                    frame_dt_s=frame_dt_s,
                )

                frame_events = list(tracker.last_events) + reid_events
                for ev in frame_events:
                    all_events.append(
                        {"frame_idx": frame_idx, "timestamp_s": round(timestamp_s, 3), **ev}
                    )

                for tr in tracks:
                    if tr.recovered:
                        recovered_count += 1
                    fx, fy = tr.foot_point
                    trails[tr.track_id].append((int(round(fx)), int(round(fy))))
                    if len(trails[tr.track_id]) > 48:
                        trails[tr.track_id] = trails[tr.track_id][-48:]
                    records.append(tr.to_record(frame_idx, timestamp_s))

                fr = FrameResult(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    tracks=tracks,
                    num_detections=len(dets),
                    inference_ms=inf_ms,
                    tracking_ms=trk_ms,
                    events=frame_events,
                )
                frame_results.append(fr)
                processed += 1

                elapsed_now = max(time.perf_counter() - t_start, 1e-6)
                live_fps = processed / elapsed_now

                if save_video or display:
                    anon_frame = self.anonymizer.apply(frame_bgr, tracks)
                    rendered = draw_tracks(
                        anon_frame,
                        tracks,
                        trails=trails,
                        fps=live_fps,
                        query=self.query,
                        anonymize_mode=self.anonymize_mode,
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
                    progress_callback(processed, total_frames, live_fps)
        finally:
            if video_writer is not None:
                video_writer.release()
            if display:
                with contextlib.suppress(cv2.error):
                    cv2.destroyAllWindows()

        elapsed_s = max(time.perf_counter() - t_start, 1e-6)
        avg_fps = processed / elapsed_s
        avg_inf = float(np.mean(inf_ms_list)) if inf_ms_list else 0.0
        avg_trk = float(np.mean(trk_ms_list)) if trk_ms_list else 0.0

        device_label = getattr(self.detector, "active_provider", "custom")
        output_files: dict[str, str] = {}
        if save_video and video_path.is_file():
            output_files["video"] = str(video_path)

        # Persist Track Cards & diagnostic events (always written for `trackflow inspect` / `search`)
        cards_path = out_dir / "track_cards.json"
        card_index.save_json(cards_path, reid_backend=self.reid_backend)
        output_files["track_cards"] = str(cards_path)

        events_path = out_dir / "events.json"
        events_path.write_text(json.dumps(all_events, indent=2), encoding="utf-8")
        output_files["events"] = str(events_path)

        if "parquet" in export_set:
            output_files["parquet"] = str(write_parquet(records, out_dir / "tracks.parquet"))
        if "jsonl" in export_set:
            output_files["jsonl"] = str(write_jsonl(records, out_dir / "tracks.jsonl"))
        if "csv" in export_set:
            output_files["csv"] = str(write_csv(records, out_dir / "tracks.csv"))

        if self.homography is not None:
            fp_png = self.homography.render_floorplan(out_dir / "floorplan.png")
            fp_json = out_dir / "homography.json"
            self.homography.save_json(fp_json)
            output_files["floorplan"] = str(fp_png)
            output_files["homography"] = str(fp_json)

        summary = RunSummary(
            source=str(source),
            model=self.model_name,
            tracker=self.tracker_name,
            device=device_label,
            total_frames=seen_frames,
            processed_frames=processed,
            source_fps=src_fps,
            avg_fps=avg_fps,
            avg_inference_ms=avg_inf,
            avg_tracking_ms=avg_trk,
            total_detections=total_dets,
            total_track_observations=len(records),
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
                dict(self.homography.zone_dwell_totals_s) if self.homography is not None else {}
            ),
        )

        if "html" in export_set:
            html_path = write_html_report(
                summary, frame_results, card_index.sorted_cards(), out_dir / "report.html"
            )
            summary.output_files["html"] = str(html_path)

        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
        summary.output_files["summary"] = str(summary_path)
        return summary
