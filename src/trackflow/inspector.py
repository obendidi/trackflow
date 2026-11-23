"""Local HTTP server and API for the Trackflow Visual Track Debugger & 4-Point Floorplan Studio."""

from __future__ import annotations

import json
import mimetypes
import shutil
import webbrowser
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from trackflow.homography import DwellZone, FloorplanHomography
from trackflow.reid import search_track_cards
from trackflow.sinks import PARQUET_SCHEMA

UI_HTML_PATH = Path(__file__).with_name("inspector_ui.html")


def load_run_payload(run_dir: Path) -> dict[str, Any]:
    """Load summary, track observations, Track Cards, diagnostic events, and homography from a run directory."""
    summary_path = run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {"source": str(run_dir), "frame_width": 640, "frame_height": 480}
    )

    cards_path = run_dir / "track_cards.json"
    cards_payload = (
        json.loads(cards_path.read_text(encoding="utf-8"))
        if cards_path.is_file()
        else {"reid_backend": "none", "cards": []}
    )
    # Strip raw embeddings from browser payload to keep responses fast
    cards = [
        {k: v for k, v in c.items() if k != "embedding"} for c in cards_payload.get("cards", [])
    ]

    events_path = run_dir / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.is_file() else []

    homography_path = run_dir / "homography.json"
    homography = (
        json.loads(homography_path.read_text(encoding="utf-8"))
        if homography_path.is_file()
        else None
    )

    records: list[dict[str, Any]] = []
    jsonl_path = run_dir / "tracks.jsonl"
    parquet_path = run_dir / "tracks.parquet"
    outputs = summary.get("output_files")

    def present(key: str, path: Path) -> bool:
        return (outputs is None or key in outputs) and path.is_file()

    csv_path = run_dir / "tracks.csv"
    if present("jsonl", jsonl_path):
        with jsonl_path.open(encoding="utf-8") as rows:
            records = [json.loads(line) for line in rows if line.strip()]
    elif present("parquet", parquet_path):
        records = pq.read_table(str(parquet_path)).to_pylist()
    elif present("csv", csv_path):
        opts = pcsv.ConvertOptions(column_types=PARQUET_SCHEMA, strings_can_be_null=True)
        records = pcsv.read_csv(csv_path, convert_options=opts).to_pylist()

    frames_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        frames_map[int(rec["frame_idx"])].append(rec)
    if "frame_stride" in summary and outputs and {"jsonl", "parquet", "csv"}.intersection(outputs):
        for idx in range(0, summary["total_frames"], summary["frame_stride"]):
            frames_map.setdefault(idx, [])

    frames_sorted = [
        {"frame_idx": idx, "tracks": frames_map[idx]} for idx in sorted(frames_map.keys())
    ]

    id_switches = [e for e in events if e.get("type") == "possible_id_switch"]
    recoveries = [e for e in events if e.get("type") == "track_recovered"]
    reid_merges = [e for e in events if e.get("type") == "reid_cross_exit_merge"]

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "reid_backend": cards_payload.get("reid_backend", "none"),
        "cards": cards,
        "events": events,
        "diagnostics": {
            "possible_id_switches": len(id_switches),
            "recoveries": len(recoveries),
            "reid_merges": len(reid_merges),
            "total_observations": len(records),
        },
        "homography": homography,
        "frames": frames_sorted,
        "has_video": present("video", run_dir / "annotated.mp4"),
        "has_floorplan": (run_dir / "floorplan.png").is_file(),
    }


def compute_homography_projection(
    run_dir: Path,
    image_points: list[list[float]],
    width_m: float = 10.0,
    height_m: float = 10.0,
    zones_raw: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calibrate 4-point homography on a saved run and project all track foot points to bird's-eye."""
    world_points = [
        (0.0, 0.0),
        (float(width_m), 0.0),
        (float(width_m), float(height_m)),
        (0.0, float(height_m)),
    ]
    zones = [
        DwellZone(
            name=str(z["name"]),
            polygon=[(float(p[0]), float(p[1])) for p in z["polygon"]],
            color=str(z.get("color", "#38bdf8")),
        )
        for z in (zones_raw or [])
    ]
    hom = FloorplanHomography(
        image_points=[(float(p[0]), float(p[1])) for p in image_points],
        world_points=world_points,
        width_m=float(width_m),
        height_m=float(height_m),
        zones=zones,
    )

    run_data = load_run_payload(run_dir)
    src_fps = float(run_data["summary"].get("source_fps", 30.0) or 30.0)
    dt_s = run_data["summary"].get("frame_stride", 1) / max(src_fps, 1e-3)

    trajectories: dict[str, list[list[float]]] = defaultdict(list)
    for frame in run_data["frames"]:
        for tr in frame["tracks"]:
            fx = (float(tr["x1"]) + float(tr["x2"])) * 0.5
            fy = float(tr["y2"])
            (wx, wy), _ = hom.observe(int(tr["track_id"]), (fx, fy), dt_s=dt_s)
            trajectories[str(tr["track_id"])].append([round(wx, 3), round(wy, 3)])

    hom.save_json(run_dir / "homography.json")
    hom.render_floorplan(run_dir / "floorplan.png")

    return {
        "homography": hom.to_dict(),
        "trajectories": dict(trajectories),
        "zone_dwell_totals_s": {k: round(v, 2) for k, v in hom.zone_dwell_totals_s.items()},
    }


def make_inspector_handler(
    run_dir: Path,
    compare_dir: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler bound to ``run_dir`` (and optional ``compare_dir``)."""
    root_dir = run_dir.resolve()
    cmp_dir = compare_dir.resolve() if compare_dir is not None else None

    class InspectorHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

        def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            resolved = path.resolve()
            if not resolved.is_relative_to(root_dir) or not resolved.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime, _ = mimetypes.guess_type(str(resolved))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(resolved.stat().st_size))
            self.end_headers()
            with resolved.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path

            if route in ("/", "/index.html"):
                html_bytes = UI_HTML_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
                return

            if route == "/api/run":
                payload = load_run_payload(root_dir)
                if cmp_dir is not None and cmp_dir.is_dir():
                    payload["compare"] = load_run_payload(cmp_dir)
                self._send_json(payload)
                return

            if route.startswith("/crops/"):
                rel = route.removeprefix("/crops/")
                summary_path = root_dir / "summary.json"
                summary = (
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary_path.is_file()
                    else {}
                )
                if (
                    summary.get("anonymize_mode", "none") != "none"
                    or not Path(rel).match("track_*.jpg")
                    or Path(rel).name != rel
                ):
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_file(root_dir / "crops" / rel)
                return

            if route.startswith("/media/"):
                rel = route.removeprefix("/media/")
                if rel not in {"floorplan.png", "annotated.mp4"}:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if rel == "annotated.mp4":
                    summary_path = root_dir / "summary.json"
                    if summary_path.is_file():
                        outputs = json.loads(summary_path.read_text(encoding="utf-8")).get(
                            "output_files"
                        )
                        if outputs is not None and "video" not in outputs:
                            self.send_error(HTTPStatus.NOT_FOUND)
                            return
                self._send_file(root_dir / rel)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError("Invalid content length")
                if self.headers.get_content_type() != "application/json":
                    self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                    return
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
            except (ValueError, UnicodeError):
                self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/search":
                query = str(body.get("query", "")).strip()
                if not query:
                    self._send_json({"error": "Empty query"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    top_k = int(body.get("top_k", 10))
                    hits = search_track_cards(root_dir, query, top_k=top_k)
                    self._send_json({"query": query, "results": hits})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/homography":
                pts = body.get("image_points")
                if not isinstance(pts, list) or len(pts) != 4:
                    self._send_json(
                        {"error": "Provide exactly 4 image_points [[x,y], ...]"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    res = compute_homography_projection(
                        root_dir,
                        image_points=pts,
                        width_m=float(body.get("width_m", 10.0)),
                        height_m=float(body.get("height_m", 10.0)),
                        zones_raw=body.get("zones"),
                    )
                    self._send_json(res)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return InspectorHandler


def serve_inspector(
    run_dir: str | Path,
    *,
    compare_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Start the Trackflow Visual Track Debugger HTTP server."""
    handler = make_inspector_handler(
        Path(run_dir),
        Path(compare_dir) if compare_dir is not None else None,
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
