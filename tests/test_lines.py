"""Line counting: crossing geometry, CLI parsing, and pipeline outputs."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from trackflow import CountingLine, Detection, Track, TrackflowPipeline
from trackflow.cli import app
from trackflow.lines import LineCounter


def _at(track_id: int, foot_x: float, foot_y: float) -> Track:
    return Track(track_id, (foot_x - 10, foot_y - 40, foot_x + 10, foot_y), 0.9, 0, "person")


def _feed(counter, frames, step=1.0):
    """Feed one list of tracks per frame, ``step`` seconds apart, and return all events."""
    return [e for i, tracks in enumerate(frames) for e in counter.update(tracks, i * step)]


def test_counts_each_crossing_by_direction():
    # Drawn left to right, so "in" is moving down the frame.
    counter = LineCounter([CountingLine("door", (0, 100), (200, 100))])
    events = _feed(counter, [[_at(1, 50, y)] for y in (80, 95, 105, 120, 90, 85)])
    assert counter.counts == {"door": {"in": 1, "out": 1}}
    assert [(e["direction"], e["track_id"]) for e in events] == [("in", 1), ("out", 1)]
    assert events[0] == {
        "type": "line_crossed",
        "line": "door",
        "direction": "in",
        "track_id": 1,
        "class_name": "person",
    }


def test_waits_for_a_track_to_stay_on_the_new_side():
    counter = LineCounter([CountingLine("door", (0, 100), (200, 100))])
    # At 30 FPS, a box edge that pokes over the line for two frames is not a crossing.
    ys = [90] * 5 + [105] * 2 + [95] * 5 + [110] * 8 + [112] * 10
    events = _feed(counter, [[_at(1, 50, y)] for y in ys], step=1 / 30)
    assert counter.counts == {"door": {"in": 1, "out": 0}}
    assert [e["direction"] for e in events] == ["in"]


def test_ignores_moves_past_the_segment_end_and_points_on_the_line():
    counter = LineCounter([CountingLine("door", (0, 100), (200, 100))])
    _feed(counter, [[_at(1, 300, y)] for y in (80, 120, 120)])  # beside the segment
    _feed(counter, [[_at(2, 50, y)] for y in (80, 100, 100, 120, 120)])  # pauses on the line
    assert counter.counts == {"door": {"in": 1, "out": 0}}


def test_tracks_and_lines_are_counted_independently():
    counter = LineCounter(
        [CountingLine("top", (0, 100), (200, 100)), CountingLine("bottom", (0, 200), (200, 200))]
    )
    _feed(counter, [[_at(1, 50, y), _at(2, 150, 300 - y)] for y in (80, 150, 250, 260)])
    assert counter.counts == {"top": {"in": 1, "out": 1}, "bottom": {"in": 1, "out": 1}}


def test_parse_and_validation():
    assert CountingLine.parse("door=0,10,200,10.5", "line-1") == CountingLine(
        "door", (0, 10), (200, 10.5)
    )
    assert CountingLine.parse("1,2,3,4", "line-2").name == "line-2"
    for bad in ("door=1,2,3", "a,b,c,d", "1,1,1,1", "nan,0,1,1"):
        with pytest.raises(ValueError):
            CountingLine.parse(bad, "line-1")
    with pytest.raises(ValueError, match="unique"):
        TrackflowPipeline(
            detector=object(),
            lines=[CountingLine("a", (0, 0), (1, 1)), CountingLine("a", (0, 1), (1, 0))],
        )


class _TwoWalkers:
    """One person walking right along the top, one walking left along the bottom."""

    active_provider = "MockCPU"

    def __init__(self) -> None:
        self.step = 0

    def detect(self, frame_bgr):
        s, self.step = self.step, self.step + 1
        return [
            Detection((40 + 12 * s, 20, 80 + 12 * s, 120), 0.95, 0, "person"),
            Detection((400 - 12 * s, 200, 440 - 12 * s, 300), 0.95, 0, "person"),
        ]


# Drawn top to bottom, so "in" is moving left across x = 240.
DOOR = CountingLine("door", (240, 0), (240, 320))


def test_stream_reports_crossings():
    pipe = TrackflowPipeline(detector=_TwoWalkers(), lines=[DOOR])
    frames = (np.zeros((320, 480, 3), np.uint8) for _ in range(30))
    events = [e for fr in pipe.stream(frames, fps=10) for e in fr.events]
    crossings = [e for e in events if e["type"] == "line_crossed"]
    assert sorted(e["direction"] for e in crossings) == ["in", "out"]


def _video(path, frames=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 320))
    for _ in range(frames):
        writer.write(np.zeros((320, 480, 3), np.uint8))
    writer.release()
    return path


def test_run_writes_line_counts_to_summary_events_and_report(tmp_path):
    pipe = TrackflowPipeline(detector=_TwoWalkers(), lines=[DOOR])
    summary = pipe.run(
        _video(tmp_path / "walk.mp4"), output_dir=tmp_path / "run", exports=("jsonl", "html")
    )
    assert summary.line_counts == {"door": {"in": 1, "out": 1}}
    saved = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert saved["line_counts"] == {"door": {"in": 1, "out": 1}}
    events = json.loads((tmp_path / "run" / "events.json").read_text())
    assert sum(e["type"] == "line_crossed" for e in events) == 2
    assert "Line door (in / out)" in (tmp_path / "run" / "report.html").read_text()
    assert (tmp_path / "run" / "annotated.mp4").is_file()


def test_cli_line_option(tmp_path, monkeypatch):
    monkeypatch.setattr("trackflow.pipeline.create_detector", lambda **_kwargs: _TwoWalkers())
    vid = _video(tmp_path / "walk.mp4")
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["run", str(vid), "--no-video", "--export", "jsonl", "-o", str(tmp_path / "out"),
         "--line", "door=240,0,240,320", "--line", "0,160,480,160"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    counts = json.loads((tmp_path / "out" / "summary.json").read_text())["line_counts"]
    assert counts == {"door": {"in": 1, "out": 1}, "line-2": {"in": 0, "out": 0}}
    assert "Line door" in res.output

    res = runner.invoke(app, ["run", str(vid), "--line", "door=1,2,3"])
    assert res.exit_code == 2 and "x1,y1,x2,y2" in res.output
