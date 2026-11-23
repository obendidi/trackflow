"""Structured data sinks: Parquet, JSONL, CSV, Annotated MP4, and standalone HTML run report."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trackflow.privacy import _track_color
from trackflow.types import FrameResult, RunSummary, Track, TrackCard

PARQUET_SCHEMA = pa.schema(
    [
        ("frame_idx", pa.int64()),
        ("timestamp_s", pa.float64()),
        ("track_id", pa.int64()),
        ("class_id", pa.int64()),
        ("class_name", pa.string()),
        ("confidence", pa.float64()),
        ("x1", pa.float64()),
        ("y1", pa.float64()),
        ("x2", pa.float64()),
        ("y2", pa.float64()),
        ("width", pa.float64()),
        ("height", pa.float64()),
        ("age", pa.int64()),
        ("hits", pa.int64()),
        ("recovered", pa.bool_()),
        ("query_score", pa.float64()),
        ("query_match", pa.bool_()),
        ("world_x", pa.float64()),
        ("world_y", pa.float64()),
        ("zone", pa.string()),
    ]
)


def write_parquet(records: list[dict[str, Any]], path: Path) -> Path:
    """Write track observations to Apache Parquet (directly queryable by DuckDB / Polars / Pandas)."""
    if records:
        table = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)
    else:
        empty = {field.name: [] for field in PARQUET_SCHEMA}
        table = pa.Table.from_pydict(empty, schema=PARQUET_SCHEMA)
    pq.write_table(table, str(path), compression="snappy")
    return path


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    """Write line-delimited JSON track observations."""
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def write_csv(records: list[dict[str, Any]], path: Path) -> Path:
    """Write CSV track observations."""
    fieldnames = [f.name for f in PARQUET_SCHEMA]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return path


def draw_tracks(
    frame_bgr: np.ndarray,
    tracks: list[Track],
    *,
    trails: dict[int, list[tuple[int, int]]],
    coasting: list[Track] | None = None,
    fps: float | None = None,
    query: str | None = None,
    anonymize_mode: str = "none",
) -> np.ndarray:
    """Draw boxes, IDs, motion trails, query highlight badges, and HUD onto a frame."""
    canvas = frame_bgr.copy()

    # Briefly-missed tracks at their predicted position: thin dimmed box + ID, so boxes don't blink.
    for tr in coasting or []:
        x1, y1, x2, y2 = [int(round(v)) for v in tr.bbox]
        dim = tuple(int(c * 0.6) for c in _track_color(tr.track_id))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), dim, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"#{tr.track_id}",
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            dim,
            1,
            cv2.LINE_AA,
        )

    for tr in tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in tr.bbox]
        base_color = _track_color(tr.track_id)
        is_match = tr.query_match

        color = base_color if is_match else (100, 105, 115)
        thickness = 3 if (query and is_match) else (2 if is_match else 1)

        # Draw motion trail
        trail = trails.get(tr.track_id, [])
        if len(trail) >= 2:
            pts = np.array(trail[-32:], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, color, 2 if is_match else 1, cv2.LINE_AA)

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Label pill
        parts = [f"#{tr.track_id}", tr.class_name, f"{tr.confidence:.2f}"]
        if tr.query_score is not None:
            parts.append(f"q:{tr.query_score:.2f}")
        if tr.zone:
            parts.append(f"[{tr.zone}]")
        label = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(th + 8, y1)
        cv2.rectangle(canvas, (x1, ly - th - 8), (x1 + tw + 10, ly), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, ly - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 18, 24),
            1,
            cv2.LINE_AA,
        )

    # Top-left HUD
    hud_items = [f"Trackflow | {len(tracks)} active"]
    if fps is not None:
        hud_items.append(f"{fps:.1f} FPS")
    if query:
        hud_items.append(f'query: "{query}"')
    if anonymize_mode != "none":
        hud_items.append(f"privacy: {anonymize_mode}")
    hud_text = "  •  ".join(hud_items)
    (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(canvas, (10, 10), (24 + hw, 24 + hh), (15, 18, 26), -1)
    cv2.putText(
        canvas,
        hud_text,
        (17, 17 + hh),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 242, 250),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_html_report(
    summary: RunSummary,
    frame_results: list[FrameResult],
    cards: list[TrackCard],
    path: Path,
) -> Path:
    """Generate a self-contained HTML run report with inline SVG charts & Track Card gallery."""
    counts = [len(fr.tracks) for fr in frame_results]
    max_count = max(counts, default=1) or 1

    # Build inline SVG sparkline for active tracks per frame
    svg_w, svg_h = 720, 150
    if len(counts) >= 2:
        pts = [
            f"{(i / (len(counts) - 1)) * svg_w:.1f},{svg_h - 12 - (c / max_count) * (svg_h - 24):.1f}"
            for i, c in enumerate(counts)
        ]
        polyline = " ".join(pts)
        area = f"0,{svg_h} {polyline} {svg_w},{svg_h}"
    else:
        polyline = f"0,{svg_h - 12} {svg_w},{svg_h - 12}"
        area = f"0,{svg_h} {polyline} {svg_w},{svg_h}"

    # Diagnostic events summary
    event_counts: dict[str, int] = defaultdict(int)
    id_switch_rows: list[str] = []
    for fr in frame_results:
        for ev in fr.events:
            et = ev.get("type", "event")
            event_counts[et] += 1
            if et in ("possible_id_switch", "reid_cross_exit_merge", "track_recovered"):
                id_switch_rows.append(
                    f"<tr><td>{fr.frame_idx}</td><td>{fr.timestamp_s:.2f}s</td>"
                    f"<td><code>{html.escape(et)}</code></td>"
                    f"<td><code>{html.escape(json.dumps(ev))}</code></td></tr>"
                )

    card_html: list[str] = []
    for c in cards[:60]:
        q_badge = (
            f'<span class="badge {"match" if c.query_match else "muted"}">query {c.query_score:.2f}</span>'
            if c.query_score is not None
            else ""
        )
        merged_badge = (
            f'<span class="badge accent">Re-ID merged {len(c.merged_track_ids)}x</span>'
            if c.merged_track_ids
            else ""
        )
        thumb = (
            f'<img src="{html.escape(c.thumbnail_relpath)}" alt="Track #{c.track_id}" />'
            if c.thumbnail_relpath
            else '<div class="thumb-placeholder">No Crop</div>'
        )
        card_html.append(
            f"""
            <div class="card">
              <div class="thumb">{thumb}</div>
              <div class="card-body">
                <div class="card-head">
                  <strong>Track #{c.track_id}</strong>
                  <span class="badge">{html.escape(c.class_name)}</span>
                  {q_badge}{merged_badge}
                </div>
                <div class="meta">
                  Frames {c.first_frame}–{c.last_frame} ({c.duration_s:.1f}s · {c.frame_count} obs)<br/>
                  Peak conf: {c.best_confidence:.2f}
                  {f"<br/>Zones: {', '.join(c.zones_visited)}" if c.zones_visited else ""}
                </div>
              </div>
            </div>
            """
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Trackflow Run Report — {html.escape(Path(summary.source).name)}</title>
<style>
  :root {{
    --bg: #0b1020; --surface: #131b31; --border: #24304f;
    --text: #e6eefc; --muted: #93a4c7; --accent: #34d399; --sky: #38bdf8;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); padding: 28px; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ margin: 0 0 6px; font-size: 24px; }}
  .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 22px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 22px; }}
  .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
  .kpi .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .kpi .val {{ font-size: 26px; font-weight: 700; margin-top: 4px; color: var(--accent); }}
  .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 20px; }}
  .panel h2 {{ margin: 0 0 12px; font-size: 16px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }}
  .card {{ background: #0f1629; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; display: flex; gap: 10px; padding: 10px; }}
  .thumb {{ width: 64px; height: 84px; flex: 0 0 64px; border-radius: 6px; overflow: hidden; background: #1a2440; display: flex; align-items: center; justify-content: center; }}
  .thumb img {{ width: 100%; height: 100%; object-fit: cover; }}
  .thumb-placeholder {{ font-size: 11px; color: var(--muted); }}
  .card-head {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 6px; }}
  .badge {{ font-size: 11px; padding: 2px 7px; border-radius: 999px; background: #1d2b4f; color: var(--sky); }}
  .badge.match, .badge.accent {{ background: #0d3b2e; color: var(--accent); }}
  .badge.muted {{ background: #1e2433; color: var(--muted); }}
  .meta {{ font-size: 12px; color: var(--muted); line-height: 1.45; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; }}
  code {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; color: #bae6fd; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Trackflow Run Report</h1>
  <div class="sub">
    Source: <code>{html.escape(summary.source)}</code> ·
    Model: <code>{html.escape(summary.model)}</code> ·
    Tracker: <code>{html.escape(summary.tracker)}</code> ·
    Provider: <code>{html.escape(summary.device)}</code>
    {f" · Query: <code>{html.escape(summary.query)}</code>" if summary.query else ""}
    {f" · Privacy: <code>{html.escape(summary.anonymize_mode)}</code>" if summary.anonymize_mode != "none" else ""}
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">Unique Tracks</div><div class="val">{summary.unique_tracks}</div></div>
    <div class="kpi"><div class="label">Avg Speed</div><div class="val">{summary.avg_fps:.1f} FPS</div></div>
    <div class="kpi"><div class="label">Frames</div><div class="val">{summary.processed_frames}</div></div>
    <div class="kpi"><div class="label">Inference</div><div class="val">{summary.avg_inference_ms:.1f} ms</div></div>
    <div class="kpi"><div class="label">Recovered / Re-ID</div><div class="val">{summary.recovered_tracks} / {summary.reid_merges}</div></div>
  </div>

  <div class="panel">
    <h2>Active Tracked Objects Over Time (peak: {max_count})</h2>
    <svg viewBox="0 0 {svg_w} {svg_h}" width="100%" height="150" preserveAspectRatio="none">
      <polygon points="{area}" fill="rgba(56, 189, 248, 0.16)" />
      <polyline points="{polyline}" fill="none" stroke="#38bdf8" stroke-width="2.2" />
    </svg>
  </div>

  <div class="panel">
    <h2>Track Cards ({len(cards)} total)</h2>
    <div class="cards">{"".join(card_html) or '<div class="meta">No tracks recorded.</div>'}</div>
  </div>

  <div class="panel">
    <h2>Tracker Diagnostics (Possible ID Switches, Recoveries & Re-ID Merges)</h2>
    <table>
      <thead><tr><th>Frame</th><th>Time</th><th>Event</th><th>Details</th></tr></thead>
      <tbody>{"".join(id_switch_rows[:40]) or '<tr><td colspan="4" class="meta">No anomalies or switches detected.</td></tr>'}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path
