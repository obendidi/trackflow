"""The browser playground (web/*.js) must behave exactly like the Python package.

Replays recorded real-video detections (tests/fixtures/vtest_detections.json) through both the
Python ``ByteTracker`` and ``web/tracker.js``, and random DETR outputs through both
post-processors. Requires Node.js; skipped when ``node`` is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from trackflow.detector import postprocess_detr_outputs
from trackflow.tracker import ByteTracker
from trackflow.types import Detection

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "vtest_detections.json"
REPLAY = ROOT / "tests" / "js" / "replay.mjs"
THRESHOLD = 0.45  # TrackflowPipeline / playground default

node = shutil.which("node")


def _python_replay(
    frames: list[list[list[float]]], iou_buffer: float = 0.0
) -> list[dict[str, list]]:
    tracker = ByteTracker(
        track_high_thresh=THRESHOLD,
        track_low_thresh=max(0.10, THRESHOLD * 0.35),
        new_track_thresh=min(0.95, THRESHOLD + 0.05),
        iou_buffer=iou_buffer,
    )
    out = []
    for dets in frames:
        tracks = tracker.update(
            [
                Detection(
                    bbox=tuple(d[:4]),
                    confidence=d[4],
                    class_id=int(d[5]) if len(d) > 5 else 0,
                    class_name="person" if len(d) <= 5 else str(int(d[5])),
                )
                for d in dets
            ]
        )
        fmt = lambda t: [t.track_id, *t.bbox]  # noqa: E731
        out.append(
            {
                "tracks": [fmt(t) for t in tracks],
                "coasting": [fmt(t) for t in tracker.coasting_tracks(3)],
            }
        )
    return out


def _run_node(*args: str) -> object:
    res = subprocess.run(
        [node, str(REPLAY), *args], capture_output=True, text=True, check=True, timeout=120
    )
    return json.loads(res.stdout)


@pytest.mark.skipif(node is None, reason="Node.js not installed")
@pytest.mark.parametrize("iou_buffer", [0.0, 0.3])
def test_web_tracker_matches_python_tracker(iou_buffer: float) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # Buffered case on every 3rd frame: the low-FPS situation it exists for.
    stride = 1 if iou_buffer == 0.0 else 3
    js = _run_node("tracker", str(FIXTURE), str(THRESHOLD), str(iou_buffer), str(stride))

    for name, frames in fixture["sequences"].items():
        py = _python_replay(frames[::stride], iou_buffer)
        assert len(js[name]) == len(py)
        for f, (p, j) in enumerate(zip(py, js[name], strict=True)):
            for key in ("tracks", "coasting"):
                assert [t[0] for t in p[key]] == [t[0] for t in j[key]], f"{name} frame {f} {key}"
                if p[key]:
                    np.testing.assert_allclose(
                        np.array(p[key])[:, 1:], np.array(j[key])[:, 1:], atol=1e-6
                    )


@pytest.mark.skipif(node is None, reason="Node.js not installed")
def test_web_postprocess_matches_python(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    nq, nc = 300, 91
    logits = rng.normal(-6.0, 1.5, size=(1, nq, nc)).astype(np.float32)
    boxes = np.concatenate(
        [rng.uniform(0.1, 0.9, size=(1, nq, 2)), rng.uniform(0.02, 0.3, size=(1, nq, 2))], axis=-1
    ).astype(np.float32)
    # A few confident people (class 1) incl. a near-duplicate pair, plus other classes.
    for q, c, v in [(3, 1, 4.0), (10, 1, 2.5), (11, 1, 2.4), (20, 3, 3.0), (42, 1, -0.5)]:
        logits[0, q, c] = v
    boxes[0, 11] = boxes[0, 10] + np.array([0.002, 0.0, 0.0, 0.0], dtype=np.float32)
    id2label = {i: f"class{i}" for i in range(nc)}
    id2label[1] = "person"

    for allowed in (None, {"person"}):
        case = {
            "logits": logits.ravel().tolist(),
            "dims": [1, nq, nc],
            "boxes": boxes.ravel().tolist(),
            "id2label": {str(k): v for k, v in id2label.items()},
            "width": 768,
            "height": 576,
            "threshold": 0.15,
            "allowed": sorted(allowed) if allowed else None,
        }
        path = tmp_path / "case.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        js = _run_node("postprocess", str(path))
        py = postprocess_detr_outputs(
            logits, boxes, (576, 768), id2label, threshold=0.15, allowed_classes=allowed
        )
        assert len(py) == len(js) and len(py) >= 3
        for d, j in zip(py, js, strict=True):
            assert d.class_id == j[0]
            assert abs(d.confidence - j[1]) < 1e-5
            np.testing.assert_allclose(d.bbox, j[2:], atol=1e-3)


def test_tracker_quality_on_recorded_detections() -> None:
    """Guardrail: fixed real-video detections must not regress into ghost IDs again."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for name, frames in fixture["sequences"].items():
        lengths: dict[int, int] = {}
        for frame in _python_replay(frames):
            for t in frame["tracks"]:
                lengths[t[0]] = lengths.get(t[0], 0) + 1
        short = sum(1 for n in lengths.values() if n < 10)
        # Before track confirmation: 75 IDs / 38 short on the full RT-DETR video.
        assert short <= 3, f"{name}: {short} IDs lived < 10 frames"
        assert len(lengths) <= 20, f"{name}: {len(lengths)} unique IDs in 250 frames"


@pytest.mark.skipif(node is None, reason="Node.js not installed")
def test_web_multiclass_tracks_match_python(tmp_path: Path):
    frames = [
        [[0, 0, 80, 100, 0.9, 0], [10, 10, 60, 70, 0.85, 24]],
        [[11, 10, 61, 70, 0.3, 24]],
        [[12, 10, 62, 70, 0.95, 24], [1, 0, 81, 100, 0.9, 0]],
    ]
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps({"sequences": {"mixed": frames}}))
    js = _run_node("tracker", str(path), str(THRESHOLD))["mixed"]
    py = _python_replay(frames)
    assert [[t[0] for t in f["tracks"]] for f in js] == [[1, 2], [2], [1, 2]]
    for a, b in zip(js, py, strict=True):
        np.testing.assert_allclose(a["tracks"], b["tracks"])
