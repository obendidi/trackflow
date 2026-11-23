"""Reproducible benchmark harness & 2017 vs. 2026 split-screen comparison generator."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

from trackflow.pipeline import TrackflowPipeline, fresh_run_dir
from trackflow.types import RunSummary

# Historical baseline recorded in the original July 2017 README (preserved on `legacy-2017`)
LEGACY_2017_BASELINE: dict[str, Any] = {
    "era": "2017 (Tracking-with-darkflow)",
    "detector": "YOLOv2 (darkflow / TF 1.0)",
    "tracker": "DeepSORT (manual TF frozen graph)",
    "hardware": "NVIDIA GTX 1080 (1024x1280)",
    "avg_fps": 14.0,
    "avg_fps_with_mog2": 10.5,
    "setup_steps": 6,
    "license": "GPL-3.0",
    "features": "Boxes + IDs + CSV only",
}


@dataclass(slots=True)
class BenchmarkRow:
    model: str
    tracker: str
    device: str
    avg_fps: float
    avg_inference_ms: float
    avg_tracking_ms: float
    unique_tracks: int
    processed_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tracker": self.tracker,
            "device": self.device,
            "avg_fps": round(self.avg_fps, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "avg_tracking_ms": round(self.avg_tracking_ms, 2),
            "unique_tracks": self.unique_tracks,
            "processed_frames": self.processed_frames,
        }


def generate_synthetic_crowd_video(
    out_path: str | Path,
    *,
    width: int = 640,
    height: int = 360,
    num_frames: int = 90,
    fps: float = 30.0,
) -> Path:
    """Create a deterministic synthetic multi-pedestrian test video for offline/CI benchmarks."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer at {path}")

    actors = [
        {"x": 60.0, "y": 140.0, "vx": 4.2, "vy": 0.4, "color": (60, 60, 220)},
        {"x": 540.0, "y": 150.0, "vx": -3.8, "vy": 0.3, "color": (220, 140, 50)},
        {"x": 180.0, "y": 120.0, "vx": 2.1, "vy": 0.8, "color": (70, 200, 110)},
        {"x": 420.0, "y": 180.0, "vx": -2.5, "vy": -0.3, "color": (200, 90, 200)},
    ]

    try:
        for _ in range(num_frames):
            frame = np.full((height, width, 3), (38, 42, 52), dtype=np.uint8)
            cv2.rectangle(frame, (0, int(height * 0.52)), (width, height), (54, 58, 68), -1)
            for a in actors:
                cx, cy = int(a["x"]), int(a["y"])
                # Simple stylized pedestrian silhouette
                cv2.circle(frame, (cx, cy - 36), 11, (205, 185, 170), -1)
                cv2.rectangle(frame, (cx - 14, cy - 24), (cx + 14, cy + 20), a["color"], -1)
                cv2.rectangle(frame, (cx - 12, cy + 20), (cx - 2, cy + 56), (40, 45, 60), -1)
                cv2.rectangle(frame, (cx + 2, cy + 20), (cx + 12, cy + 56), (40, 45, 60), -1)
                a["x"] = 20 + (a["x"] - 20 + a["vx"]) % (width - 40)
                a["y"] = float(np.clip(a["y"] + a["vy"], 90, height - 80))
            writer.write(frame)
    finally:
        writer.release()

    return path


def render_split_screen_2017_vs_2026(
    modern_video_path: str | Path,
    out_path: str | Path,
    *,
    modern_summary: RunSummary,
) -> Path:
    """Create a side-by-side comparison video against the 2017 baseline."""
    src = Path(modern_video_path)
    out = Path(out_path)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open annotated video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w * 2, h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer at {out}")

    # The 2017 README supplies a measurement, not frame-level predictions for this input.
    left_frame = np.full((h, w, 3), (24, 24, 30), dtype=np.uint8)
    lines = [
        "2017 historical measurement",
        "YOLOv2 + DeepSORT",
        "14 FPS on NVIDIA GTX 1080",
        "1280 x 1024, original README",
    ]
    scale = max(0.2, min(w / 600, h / 400, 1.0))
    for i, line in enumerate(lines):
        cv2.putText(
            left_frame,
            line,
            (int(12 * scale), int((40 + i * 38) * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 * scale,
            (220, 220, 225),
            1,
            cv2.LINE_AA,
        )
    try:
        while True:
            ok, right_frame = cap.read()
            if not ok:
                break

            cv2.rectangle(right_frame, (0, 0), (w, 42), (10, 38, 32), -1)
            cv2.putText(
                right_frame,
                f"2026: Trackflow ({modern_summary.model} + {modern_summary.tracker} · {modern_summary.avg_fps:.1f} FPS · 1 cmd)",
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (140, 255, 200),
                1,
                cv2.LINE_AA,
            )

            combined = np.hstack([left_frame, right_frame])
            cv2.line(combined, (w, 0), (w, h), (56, 189, 248), 2)
            writer.write(combined)
    finally:
        cap.release()
        writer.release()

    return out


def run_benchmark_suite(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    models: Sequence[str] = ("rtdetr-v2-r18",),
    trackers: Sequence[str] = ("bytetrack", "botsort"),
    device: str = "auto",
    max_frames: int | None = 120,
    split_screen: bool = False,
    detector_factory: Any | None = None,
) -> dict[str, Any]:
    """Run reproducible Trackflow benchmarks and write JSON + Markdown 2017 vs. 2026 report.

    Without ``output_dir``, results go to a fresh ``runs/benchmark-N`` directory.
    """
    out_dir = fresh_run_dir("benchmark") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []
    summaries: list[RunSummary] = []

    for model_name in models:
        for tracker_name in trackers:
            sub_dir = out_dir / f"{quote(str(model_name), safe='')}_{quote(tracker_name, safe='')}"
            det = detector_factory(model_name) if detector_factory is not None else None
            pipe = TrackflowPipeline(
                model=model_name,
                tracker=tracker_name,
                device=device,
                detector=det,
            )
            summary = pipe.run(
                str(video_path),
                output_dir=sub_dir,
                max_frames=max_frames,
                save_video=split_screen,
                exports=("parquet", "jsonl"),
            )
            summaries.append(summary)
            rows.append(
                BenchmarkRow(
                    model=model_name,
                    tracker=tracker_name,
                    device=summary.device,
                    avg_fps=summary.avg_fps,
                    avg_inference_ms=summary.avg_inference_ms,
                    avg_tracking_ms=summary.avg_tracking_ms,
                    unique_tracks=summary.unique_tracks,
                    processed_frames=summary.processed_frames,
                )
            )

    split_path: str | None = None
    if split_screen and summaries and "video" in summaries[0].output_files:
        split_file = render_split_screen_2017_vs_2026(
            summaries[0].output_files["video"],
            out_dir / "split_2017_vs_2026.mp4",
            modern_summary=summaries[0],
        )
        split_path = str(split_file)

    md_lines = [
        "# Trackflow Benchmark Report — 2017 vs. 2026",
        "",
        "## Historical 2017 Baseline (`legacy-2017`)",
        f"- **Stack:** {LEGACY_2017_BASELINE['detector']} + {LEGACY_2017_BASELINE['tracker']}",
        f"- **Recorded Speed:** {LEGACY_2017_BASELINE['avg_fps']} FPS on {LEGACY_2017_BASELINE['hardware']} ({LEGACY_2017_BASELINE['avg_fps_with_mog2']} FPS w/ MOG2)",
        f"- **Setup Friction:** {LEGACY_2017_BASELINE['setup_steps']} manual steps (3 git submodules, Cython build, Google Drive weights)",
        "",
        "## 2026 Trackflow Results",
        "",
        "| Model | Tracker | Provider | Avg FPS | Inference (ms) | Tracking (ms) | Unique Tracks | Frames |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r.model}` | `{r.tracker}` | `{r.device}` | **{r.avg_fps:.1f}** | "
            f"{r.avg_inference_ms:.1f} | {r.avg_tracking_ms:.2f} | {r.unique_tracks} | {r.processed_frames} |"
        )
    if split_path:
        md_lines.extend(["", f"- **Split-screen video:** `{split_path}`"])

    report_md = out_dir / "benchmark_report.md"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    payload = {
        "video": str(video_path),
        "legacy_2017": LEGACY_2017_BASELINE,
        "results": [r.to_dict() for r in rows],
        "report_md": str(report_md),
        "split_screen_video": split_path,
    }
    report_json = out_dir / "benchmark_results.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report_json"] = str(report_json)
    return payload
