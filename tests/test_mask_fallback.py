"""Tests for ambiguity-triggered selective mask fallback, protocol contracts, and license purity."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from trackflow import (
    MASK_REGISTRY,
    Detection,
    MaskFallbackConfig,
    MaskFallbackResolver,
    Sam3MaskPropagator,
    Track,
    TrackflowPipeline,
)
from trackflow.cli import app
from trackflow.mask_fallback import (
    _close_bijective_frame_map,
    compute_mask_box_affinities,
)
from trackflow.sam2_adapter import Sam2MaskPropagator, _resolve_torch_device, _select_dtype


def _box_mask(
    shape_hw: tuple[int, int],
    bbox_xyxy: tuple[float, float, float, float],
    orig_wh: tuple[int, int],
) -> np.ndarray:
    mh, mw = shape_hw
    ow, oh = orig_wh
    mask = np.zeros((mh, mw), dtype=bool)
    x1 = max(0, int(round(bbox_xyxy[0] * mw / ow)))
    y1 = max(0, int(round(bbox_xyxy[1] * mh / oh)))
    x2 = min(mw, int(round(bbox_xyxy[2] * mw / ow)))
    y2 = min(mh, int(round(bbox_xyxy[3] * mh / oh)))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _two_person_swap_sequence() -> list[list[Track]]:
    """Build 26 frames where person A (raw 1) and person B (raw 2) cross with transient occlusion and swap."""
    frames: list[list[Track]] = []
    # Person A moves left->right (x=10 -> 135), Person B moves right->left (x=135 -> 10).
    for f in range(26):
        ax = 10.0 + f * 5.0
        bx = 135.0 - f * 5.0
        box_a = (ax, 20.0, ax + 30.0, 90.0)
        box_b = (bx, 20.0, bx + 30.0, 90.0)
        if f < 12:
            frames.append(
                [
                    Track(1, box_a, 0.92, 0, "person"),
                    Track(2, box_b, 0.92, 0, "person"),
                ]
            )
        elif f in (12, 13, 14):
            # Person A vanishes behind Person B during peak overlap (ax=70, bx=75 at f=12)
            frames.append([Track(2, box_b, 0.92, 0, "person")])
        else:
            # Person A recovers at f=15, but box tracker swapped raw IDs: A gets 2, B gets 1!
            frames.append(
                [
                    Track(2, box_a, 0.92, 0, "person"),
                    Track(1, box_b, 0.92, 0, "person"),
                ]
            )
    return frames


class _OracleLinearPropagator:
    """Deterministic fake MaskPropagator returning native 256x256 true-identity masks."""

    def __init__(self, true_boxes_by_seed: dict[int, dict[int, tuple[float, float, float, float]]]):
        self.true_boxes_by_seed = true_boxes_by_seed
        self.calls: list[tuple[int, dict[int, int]]] = []

    def propagate_window(self, frames_rgb, seeds, frame_size_wh):
        # Identify window's first global frame from the encoded frame index in pixel [0, 0, 0]
        start_f = int(frames_rgb[0][0, 0, 0])
        self.calls.append((start_f, {tid: lf for tid, (lf, _) in seeds.items()}))
        out: dict[int, dict[int, np.ndarray]] = {}
        for lf in range(len(frames_rgb)):
            vid_f = start_f + lf
            f_masks: dict[int, np.ndarray] = {}
            for st in seeds:
                box = self.true_boxes_by_seed.get(st, {}).get(vid_f)
                if box is not None:
                    f_masks[st] = _box_mask((256, 256), box, frame_size_wh)
            out[lf] = f_masks
        return out


def test_core_import_never_pulls_torch_or_transformers() -> None:
    code = (
        "import sys, trackflow, trackflow.mask_fallback; "
        "assert 'torch' not in sys.modules, 'torch imported in core'; "
        "assert 'transformers' not in sys.modules, 'transformers imported in core'"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr

    assert MASK_REGISTRY
    for name, spec in MASK_REGISTRY.items():
        assert spec.license == "Apache-2.0"
        assert len(spec.revision) == 40
        assert "sam3" not in name.lower() and "sam3" not in spec.repo_id.lower()


def test_compute_mask_box_affinities_matches_across_native_and_full_res() -> None:
    box_a = (40.0, 20.0, 120.0, 180.0)
    box_b = (200.0, 20.0, 280.0, 180.0)
    m_full = _box_mask((360, 640), box_a, (640, 360))
    m_256 = _box_mask((256, 256), box_a, (640, 360))

    area_f, sc_f = compute_mask_box_affinities(m_full, [box_a, box_b], 640, 360)
    area_s, sc_s = compute_mask_box_affinities(m_256, [box_a, box_b], 640, 360)
    assert area_f > 0 and area_s > 0
    assert sc_f[0][0] > 0.90 and sc_s[0][0] > 0.90
    assert sc_f[1][0] == 0.0 and sc_s[1][0] == 0.0


def test_resolver_fixes_transient_occlusion_mutual_swap() -> None:
    seq = _two_person_swap_sequence()
    true_a = {f: (10.0 + f * 5.0, 20.0, 40.0 + f * 5.0, 90.0) for f in range(26)}
    true_b = {f: (135.0 - f * 5.0, 20.0, 165.0 - f * 5.0, 90.0) for f in range(26)}
    prop = _OracleLinearPropagator({1: true_a, 2: true_b})
    resolver = MaskFallbackResolver(prop)

    all_events: list[dict[str, Any]] = []
    live_ids_by_f: dict[int, list[tuple[int, float]]] = {}
    for f, tracks in enumerate(seq):
        frame = np.full((120, 200, 3), f, dtype=np.uint8)
        evs = resolver.observe(f, frame, tracks)
        all_events.extend(evs)
        live_ids_by_f[f] = [(t.track_id, t.bbox[0]) for t in tracks]

    assert resolver.stats.windows_dispatched == 1
    assert resolver.stats.overrides == 1
    assert resolver.stats.swaps == 2
    assert resolver.stats.dispatched_frames < 20
    assert any(ev["type"] == "mask_fallback_resolved" for ev in all_events)

    # After window closes, person A (larger x) has restored ID 1 and person B (smaller x) has ID 2.
    assert sorted(live_ids_by_f[25], key=lambda item: item[1]) == [(2, 10.0), (1, 135.0)]
    # Retrospective resolution maps swapped raw IDs inside and after the window back to true IDs.
    assert resolver.resolve_id(15, 2) == 1
    assert resolver.resolve_id(15, 1) == 2
    assert resolver.resolve_id(25, 2) == 1
    assert resolver.resolve_id(25, 1) == 2


def test_resolver_relinks_occlusion_fragmentation_and_blocks_occluder_latch() -> None:
    # Person 1 vanishes behind Person 2 at f=11..13; newborn track 3 emerges on Person 1 at f=14.
    seq: list[list[Track]] = []
    true_1: dict[int, tuple[float, float, float, float]] = {}
    true_2: dict[int, tuple[float, float, float, float]] = {}
    for f in range(24):
        ax = 20.0 + f * 4.0
        bx = 110.0 - f * 4.0
        box_a = (ax, 20.0, ax + 30.0, 90.0)
        box_b = (bx, 20.0, bx + 30.0, 90.0)
        true_1[f] = box_a
        true_2[f] = box_b
        if f < 11:
            seq.append([Track(1, box_a, 0.9, 0, "person"), Track(2, box_b, 0.9, 0, "person")])
        elif f in (11, 12, 13):
            seq.append([Track(2, box_b, 0.9, 0, "person")])
        else:
            seq.append([Track(3, box_a, 0.9, 0, "person"), Track(2, box_b, 0.9, 0, "person")])

    # Case 0: Conservative default (include_fragmentation=False) skips fragmentation-only windows
    seq_def = [
        [Track(t.track_id, t.bbox, t.confidence, t.class_id, t.class_name) for t in trs]
        for trs in seq
    ]
    prop_def = _OracleLinearPropagator({1: true_1, 2: true_2})
    res_def = MaskFallbackResolver(prop_def)
    assert res_def.config.include_fragmentation is False
    for f, tracks in enumerate(seq_def):
        res_def.observe(f, np.full((120, 200, 3), f, dtype=np.uint8), tracks)
    assert res_def.stats.windows_detected >= 1
    assert res_def.stats.windows_dispatched == 0
    assert res_def.stats.windows_skipped_gate >= 1

    # Case 1: Opting into include_fragmentation=True relinks newborn 3 -> 1 without touching 2
    prop_good = _OracleLinearPropagator({1: true_1, 2: true_2})
    res_good = MaskFallbackResolver(prop_good, MaskFallbackConfig(include_fragmentation=True))
    for f, tracks in enumerate(seq):
        res_good.observe(f, np.full((120, 200, 3), f, dtype=np.uint8), tracks)
    assert res_good.stats.overrides == 1
    assert res_good.stats.relinks == 1
    assert res_good.resolve_id(23, 3) == 1
    assert res_good.resolve_id(23, 2) == 2

    # Case 2: Buggy mask latches onto a huge 3x occluder box -> area guard rejects and abstains!
    seq2 = [
        [Track(t.track_id, t.bbox, t.confidence, t.class_id, t.class_name) for t in trs]
        for trs in seq
    ]
    bloated_1 = {
        f: (true_1[f] if f < 11 else (true_2[f][0] - 25.0, 5.0, true_2[f][2] + 25.0, 115.0))
        for f in range(24)
    }
    prop_bad = _OracleLinearPropagator({1: bloated_1, 2: true_2})
    res_bad = MaskFallbackResolver(prop_bad, MaskFallbackConfig(include_fragmentation=True))
    for f, tracks in enumerate(seq2):
        res_bad.observe(f, np.full((120, 200, 3), f, dtype=np.uint8), tracks)
    assert res_bad.stats.windows_dispatched == 1
    assert res_bad.stats.overrides == 0
    assert res_bad.stats.abstained == 1


def test_causal_frame_budget_caps_dispatch_rate() -> None:
    # Build two back-to-back occlusion events and enforce a tight bucket cap so the second is skipped.
    seq = _two_person_swap_sequence()
    true_a = {f: (10.0 + f * 5.0, 20.0, 40.0 + f * 5.0, 90.0) for f in range(26)}
    true_b = {f: (135.0 - f * 5.0, 20.0, 165.0 - f * 5.0, 90.0) for f in range(26)}
    prop = _OracleLinearPropagator({1: true_a, 2: true_b})
    cfg = MaskFallbackConfig(max_frame_rate=0.05, bucket_cap_frames=4.0)
    resolver = MaskFallbackResolver(prop, cfg)
    for f, tracks in enumerate(seq):
        resolver.observe(f, np.full((120, 200, 3), f, dtype=np.uint8), tracks)
    assert resolver.stats.windows_detected >= 1
    assert resolver.stats.windows_skipped_budget >= 1
    assert resolver.stats.windows_dispatched == 0


def test_pipeline_end_to_end_reconciles_all_exports_and_preserves_privacy(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for f in range(26):
        cv2.imwrite(str(img_dir / f"{f:03d}.png"), np.full((120, 200, 3), f, dtype=np.uint8))

    class _ReplayDetector:
        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            idx = int(frame_bgr[0, 0, 0])
            return [
                Detection(t.bbox, t.confidence, t.class_id, t.class_name)
                for t in _two_person_swap_sequence()[idx]
            ]

    class _ReplayTracker:
        def __init__(self) -> None:
            self._seq = _two_person_swap_sequence()
            self._idx = 0

        def update(self, detections: list[Detection], frame_bgr: np.ndarray | None = None) -> list[Track]:
            out = self._seq[self._idx]
            self._idx += 1
            return out

    true_a = {f: (10.0 + f * 5.0, 20.0, 40.0 + f * 5.0, 90.0) for f in range(26)}
    true_b = {f: (135.0 - f * 5.0, 20.0, 165.0 - f * 5.0, 90.0) for f in range(26)}
    prop = _OracleLinearPropagator({1: true_a, 2: true_b})

    pipe = TrackflowPipeline(
        detector=_ReplayDetector(),
        tracker_factory=_ReplayTracker,
        mask_propagator=prop,
        anonymize="blur",
    )
    run = pipe.run(
        img_dir,
        output_dir=tmp_path / "run",
        save_video=False,
        exports="parquet,jsonl,csv,mot",
    )

    assert not (tmp_path / "run" / "crops").exists()
    assert run.mask_fallback is not None
    assert run.mask_fallback["overrides"] == 1
    assert run.mask_fallback["swaps"] == 2
    assert json.loads(Path(run.output_files["summary"]).read_text()) == run.to_dict()

    # Check frame 25 across ALL four export formats: Person A (x1=135) must be track 1, Person B (x1=10) track 2!
    pq_rows = pq.read_table(run.output_files["parquet"]).to_pylist()
    f25_pq = sorted((r["track_id"], r["x1"]) for r in pq_rows if r["frame_idx"] == 25)
    assert f25_pq == [(1, 135.0), (2, 10.0)]

    jsonl_rows = [
        json.loads(line) for line in Path(run.output_files["jsonl"]).read_text().splitlines()
    ]
    assert all("_raw_track_id" not in r for r in jsonl_rows)
    f25_jl = sorted((r["track_id"], r["x1"]) for r in jsonl_rows if r["frame_idx"] == 25)
    assert f25_jl == [(1, 135.0), (2, 10.0)]

    csv_rows = list(csv.DictReader(Path(run.output_files["csv"]).read_text().splitlines()))
    f25_csv = sorted(
        (int(r["track_id"]), float(r["x1"])) for r in csv_rows if int(r["frame_idx"]) == 25
    )
    assert f25_csv == [(1, 135.0), (2, 10.0)]

    mot_rows = list(csv.reader(Path(run.output_files["mot"]).read_text().splitlines()))
    f25_mot = sorted((int(r[1]), float(r[2])) for r in mot_rows if int(r[0]) == 26)
    assert f25_mot == [(1, 136.0), (2, 11.0)]  # MOTChallenge uses 1-based frame & x1+1


def test_sam3_external_adapter_protocol_and_session_cleanup() -> None:
    calls: list[dict[str, Any]] = []

    class _FakeSam3Predictor:
        def __init__(self, fail_propagate: bool = False) -> None:
            self.fail_propagate = fail_propagate

        def handle_request(self, req: dict[str, Any]) -> Any:
            calls.append(req)
            rtype = req["type"]
            if rtype == "start_session":
                assert Path(req["resource_path"]).is_dir()
                return {"session_id": "sess-42"}
            if rtype == "add_prompt":
                return {"ok": True}
            if rtype == "propagate_in_video":
                if self.fail_propagate:
                    raise RuntimeError("boom")
                m = np.zeros((16, 16), dtype=np.float32)
                m[2:10, 2:10] = 1.0
                return {"outputs": [{"frame_index": 0, "masks": {1: m}}, {"frame_index": 1, "masks": {1: m}}]}
            if rtype == "close_session":
                return {"closed": True}
            raise AssertionError(f"Unexpected request {rtype}")

    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(2)]
    adapter = Sam3MaskPropagator(_FakeSam3Predictor())
    masks = adapter.propagate_window(frames, {1: (0, (20.0, 10.0, 120.0, 60.0))}, (200, 100))
    assert set(masks.keys()) == {0, 1}
    assert masks[0][1].dtype == np.bool_ and masks[0][1].sum() == 64

    add_reqs = [c for c in calls if c["type"] == "add_prompt"]
    assert len(add_reqs) == 1
    assert add_reqs[0]["bounding_boxes"] == [[pytest.approx(0.1), pytest.approx(0.1), pytest.approx(0.5), pytest.approx(0.5)]]
    assert calls[-1] == {"type": "close_session", "session_id": "sess-42"}

    # Cleanup runs even when propagate_in_video fails
    calls.clear()
    failing = Sam3MaskPropagator(_FakeSam3Predictor(fail_propagate=True))
    with pytest.raises(RuntimeError, match="boom"):
        failing.propagate_window(frames, {1: (0, (20.0, 10.0, 120.0, 60.0))}, (200, 100))
    assert calls[-1] == {"type": "close_session", "session_id": "sess-42"}


def test_cli_models_catalog_lists_pinned_apache_mask_models() -> None:
    res = CliRunner().invoke(app, ["models", "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.output)
    m_models = {row["name"]: row for row in payload["mask_fallback"]}
    assert "sam2.1-tiny" in m_models and "edgetam" in m_models
    assert all(row["license"] == "Apache-2.0" for row in m_models.values())
    assert all(len(row["revision"]) == 40 for row in m_models.values())


def test_unknown_mask_fallback_preset_raises_clean_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown mask fallback model"):
        TrackflowPipeline(detector=object(), mask_fallback="not-a-real-model")  # type: ignore[arg-type]


def test_pipeline_stream_applies_mask_fallback_fixes_live() -> None:
    class _ReplayDetector:
        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            idx = int(frame_bgr[0, 0, 0])
            return [
                Detection(t.bbox, t.confidence, t.class_id, t.class_name)
                for t in _two_person_swap_sequence()[idx]
            ]

    class _ReplayTracker:
        def __init__(self) -> None:
            self._seq = _two_person_swap_sequence()
            self._idx = 0

        def update(self, detections: list[Detection], frame_bgr: np.ndarray | None = None) -> list[Track]:
            out = self._seq[self._idx]
            self._idx += 1
            return out

    true_a = {f: (10.0 + f * 5.0, 20.0, 40.0 + f * 5.0, 90.0) for f in range(26)}
    true_b = {f: (135.0 - f * 5.0, 20.0, 165.0 - f * 5.0, 90.0) for f in range(26)}
    pipe = TrackflowPipeline(
        detector=_ReplayDetector(),
        tracker_factory=_ReplayTracker,
        mask_propagator=_OracleLinearPropagator({1: true_a, 2: true_b}),
    )
    frames = [np.full((120, 200, 3), f, dtype=np.uint8) for f in range(26)]
    results = list(pipe.stream(frames, fps=25.0))
    assert len(results) == 26
    assert any(ev["type"] == "mask_fallback_resolved" for fr in results for ev in fr.events)
    last_ids = sorted((t.track_id, t.bbox[0]) for t in results[-1].tracks)
    assert last_ids == [(1, 135.0), (2, 10.0)]


def test_in_window_frame_map_stays_strictly_bijective_on_3cycles_chains_and_coasting() -> None:
    # 3-cycle where only 1->2 is matched on frame f: orbit closure must map all 3 without duplicate IDs.
    perm_cycle = {1: 2, 2: 3, 3: 1}
    fmap_cycle = _close_bijective_frame_map({1: 2}, perm_cycle)
    mapped = [fmap_cycle.get(cid, cid) for cid in [1, 3, 2]]
    assert sorted(mapped) == [1, 2, 3] and len(set(mapped)) == 3

    # 3-chain ending in fresh ID: [1, 3, 2] must stay collision-free.
    perm_chain = {1: 2, 2: 3, 3: 1_000_000_000}
    fmap_chain = _close_bijective_frame_map({1: 2}, perm_chain)
    mapped_chain = [fmap_chain.get(cid, cid) for cid in [1, 3, 2]]
    assert len(set(mapped_chain)) == 3

    # Relink with coasting raw track 1: fmap must map 3->1 AND 1->fresh so active 3 & coasting 1 never collide.
    perm_relink = {3: 1, 1: 1_000_000_000}
    fmap_relink = _close_bijective_frame_map({3: 1}, perm_relink)
    assert fmap_relink[3] == 1 and fmap_relink[1] == 1_000_000_000


def test_end_frame_applies_suffix_perm_immediately_even_when_apply_in_window_false() -> None:
    seq = _two_person_swap_sequence()
    true_a = {f: (10.0 + f * 5.0, 20.0, 40.0 + f * 5.0, 90.0) for f in range(26)}
    true_b = {f: (135.0 - f * 5.0, 20.0, 165.0 - f * 5.0, 90.0) for f in range(26)}
    resolver = MaskFallbackResolver(
        _OracleLinearPropagator({1: true_a, 2: true_b}),
        MaskFallbackConfig(apply_in_window=False),
    )
    resolved_end_f = None
    live_on_end = None
    for f, tracks in enumerate(seq):
        evs = resolver.observe(f, np.full((120, 200, 3), f, dtype=np.uint8), tracks)
        if evs:
            resolved_end_f = evs[0]["window_end"]
            live_on_end = sorted(((t.track_id, t.bbox[0]) for t in tracks), key=lambda item: item[1])
    assert resolved_end_f is not None and live_on_end is not None
    # On the exact closing frame, Person B (smaller x) is ID 2 and Person A (larger x) is ID 1.
    assert live_on_end[0][0] == 2 and live_on_end[1][0] == 1


def test_sam3_adapter_matches_real_sam3_stream_schema_tensors_clipping_and_privacy() -> None:
    class _FakeCudaTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def detach(self) -> _FakeCudaTensor:
            return self

        def cpu(self) -> _FakeCudaTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    reqs: list[dict[str, Any]] = []

    class _RealStyleSam3Predictor:
        def __init__(self, fail_close: bool = False, fail_stream: bool = False) -> None:
            self.fail_close = fail_close
            self.fail_stream = fail_stream

        def handle_request(self, req: dict[str, Any]) -> dict[str, Any]:
            reqs.append(req)
            if req["type"] == "start_session":
                return {"session_id": "real-sess"}
            if req["type"] == "add_prompt":
                return {"ok": True}
            if req["type"] == "close_session":
                if self.fail_close:
                    raise OSError("close failed")
                return {"closed": True}
            raise AssertionError(f"Unexpected non-stream request {req['type']}")

        def handle_stream_request(self, req: dict[str, Any]):
            reqs.append(req)
            if self.fail_stream:
                raise RuntimeError("propagate failed")
            m = np.zeros((1, 16, 16), dtype=np.float32)
            m[0, 2:6, 2:6] = 1.0
            yield {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_binary_masks": [_FakeCudaTensor(m)],
                },
            }

    frames = [np.zeros((100, 200, 3), dtype=np.uint8)]
    adapter = Sam3MaskPropagator(_RealStyleSam3Predictor())
    # Out-of-bounds box must be clipped before normalizing so x + w <= 1.0
    masks = adapter.propagate_window(frames, {7: (0, (150.0, -10.0, 250.0, 120.0))}, (200, 100))
    assert masks[0][7].dtype == np.bool_ and masks[0][7].sum() == 16
    add_req = next(r for r in reqs if r["type"] == "add_prompt")
    bx, by, bw, bh = add_req["bounding_boxes"][0]
    assert bx == pytest.approx(0.75) and by == pytest.approx(0.0)
    assert bx + bw <= 1.0 + 1e-6 and by + bh <= 1.0 + 1e-6

    # close_session failure must not mask the primary propagation RuntimeError
    failing = Sam3MaskPropagator(_RealStyleSam3Predictor(fail_close=True, fail_stream=True))
    with pytest.raises(RuntimeError, match="propagate failed"):
        failing.propagate_window(frames, {7: (0, (10.0, 10.0, 50.0, 50.0))}, (200, 100))

    # Privacy guard: disk-writing Sam3MaskPropagator is rejected whenever anonymize != "none"
    with pytest.raises(ValueError, match="writes raw frames to disk"):
        TrackflowPipeline(
            detector=object(),  # type: ignore[arg-type]
            anonymize="skeleton",
            mask_propagator=adapter,
        )


def test_fragmentation_relink_folds_phantom_track_card_in_pipeline(tmp_path: Path) -> None:
    img_dir = tmp_path / "frag_imgs"
    img_dir.mkdir()
    for f in range(24):
        cv2.imwrite(str(img_dir / f"{f:03d}.png"), np.full((120, 200, 3), f, dtype=np.uint8))

    seq: list[list[Track]] = []
    true_1: dict[int, tuple[float, float, float, float]] = {}
    true_2: dict[int, tuple[float, float, float, float]] = {}
    for f in range(24):
        ax, bx = 20.0 + f * 4.0, 110.0 - f * 4.0
        ba, bb = (ax, 20.0, ax + 30.0, 90.0), (bx, 20.0, bx + 30.0, 90.0)
        true_1[f], true_2[f] = ba, bb
        if f < 11:
            seq.append([Track(1, ba, 0.9, 0, "person"), Track(2, bb, 0.9, 0, "person")])
        elif f in (11, 12, 13):
            seq.append([Track(2, bb, 0.9, 0, "person")])
        else:
            seq.append([Track(3, ba, 0.9, 0, "person"), Track(2, bb, 0.9, 0, "person")])

    class _Det:
        def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
            return [Detection(t.bbox, t.confidence, t.class_id, t.class_name) for t in seq[int(frame_bgr[0, 0, 0])]]

    class _Trk:
        def __init__(self) -> None:
            self._i = 0

        def update(self, detections: list[Detection], frame_bgr: np.ndarray | None = None) -> list[Track]:
            out = [Track(t.track_id, t.bbox, t.confidence, t.class_id, t.class_name) for t in seq[self._i]]
            self._i += 1
            return out

    pipe = TrackflowPipeline(
        detector=_Det(),
        tracker_factory=_Trk,
        mask_propagator=_OracleLinearPropagator({1: true_1, 2: true_2}),
        mask_fallback_config=MaskFallbackConfig(include_fragmentation=True),
    )
    run = pipe.run(img_dir, output_dir=tmp_path / "frag_run", save_video=False, exports="parquet")
    assert run.unique_tracks == 2
    assert set(pq.read_table(run.output_files["parquet"])["track_id"].to_pylist()) == {1, 2}


def test_mask_fallback_validation_and_sam2_adapter_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    # --mask-budget / mask_fallback_config without --mask-fallback fails loudly before loading detector
    with pytest.raises(ValueError, match="requires --mask-fallback"):
        TrackflowPipeline(
            detector=object(),  # type: ignore[arg-type]
            mask_fallback_config=MaskFallbackConfig(max_frame_rate=0.10),
        )
    with pytest.raises(ValueError, match="owner/repo"):
        Sam2MaskPropagator("hf:")

    # Test _resolve_torch_device and _select_dtype with fake torch module
    class _FakeCuda:
        def __init__(self, avail: bool, bf16: bool) -> None:
            self._avail, self._bf16 = avail, bf16

        def is_available(self) -> bool:
            return self._avail

        def is_bf16_supported(self) -> bool:
            return self._bf16

    class _FakeBackends:
        class mps:
            @staticmethod
            def is_available() -> bool:
                return False

    class _FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"
        backends = _FakeBackends()

        def __init__(self, cuda_avail: bool, bf16: bool) -> None:
            self.cuda = _FakeCuda(cuda_avail, bf16)

        def inference_mode(self):
            import contextlib

            return contextlib.nullcontext()

        def autocast(self, **_kw):
            import contextlib

            return contextlib.nullcontext()

    t_t4 = _FakeTorch(cuda_avail=True, bf16=False)
    t_l4 = _FakeTorch(cuda_avail=True, bf16=True)
    t_cpu = _FakeTorch(cuda_avail=False, bf16=False)

    assert _resolve_torch_device(t_l4, "tensorrt") == "cuda"
    assert _resolve_torch_device(t_l4, "cuda:1") == "cuda:1"
    assert _select_dtype(t_l4, "cuda") == "bf16"
    assert _select_dtype(t_t4, "cuda") == "fp16"
    assert _select_dtype(t_cpu, "cpu") == "fp32"
    with pytest.raises(RuntimeError, match="no CUDA"):
        _resolve_torch_device(t_cpu, "cuda")
    with pytest.raises(RuntimeError, match="MPS"):
        _resolve_torch_device(t_cpu, "mps")
    with pytest.raises(ValueError, match="Unsupported device"):
        _resolve_torch_device(t_cpu, "tpu")

    # Mocked Sam2MaskPropagator initialization & window propagation
    seen_kwargs: dict[str, Any] = {}

    class _FakePredTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def __gt__(self, thresh: float) -> _FakePredTensor:
            return _FakePredTensor(self._arr > thresh)

        @property
        def ndim(self) -> int:
            return self._arr.ndim

        def __getitem__(self, item: Any) -> _FakePredTensor:
            return _FakePredTensor(self._arr[item])

        def detach(self) -> _FakePredTensor:
            return self

        def cpu(self) -> _FakePredTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeSession:
        def __init__(self) -> None:
            self.obj_ids: list[int] = []

    class _FakeModel:
        def to(self, dev: str) -> _FakeModel:
            return self

        def eval(self) -> _FakeModel:
            return self

        def __call__(self, inference_session: Any, frame_idx: int) -> None:
            seen_kwargs.setdefault("cond_frames", []).append(frame_idx)

        def propagate_in_video_iterator(self, session: _FakeSession, start_frame_idx: int):
            arr = np.ones((len(session.obj_ids), 1, 256, 256), dtype=np.float32)

            class _Out:
                frame_idx = start_frame_idx
                pred_masks = _FakePredTensor(arr)

            yield _Out()

    class _FakeProc:
        def init_video_session(self, **_kw: Any) -> _FakeSession:
            return _FakeSession()

        def add_inputs_to_inference_session(self, inference_session: _FakeSession, frame_idx: int, obj_ids: Any, input_boxes: Any) -> None:
            ids = obj_ids if isinstance(obj_ids, list) else [obj_ids]
            inference_session.obj_ids.extend(ids)
            seen_kwargs["boxes"] = input_boxes

    class _FakeTransformers:
        class Sam2VideoModel:
            @staticmethod
            def from_pretrained(repo_id: str, **kw: Any) -> _FakeModel:
                seen_kwargs["model_kw"] = kw
                return _FakeModel()

        class Sam2VideoProcessor:
            @staticmethod
            def from_pretrained(repo_id: str, **kw: Any) -> _FakeProc:
                seen_kwargs["proc_kw"] = kw
                return _FakeProc()

    monkeypatch.setattr("trackflow.sam2_adapter._import_sam2_deps", lambda: (t_l4, _FakeTransformers))
    prop = Sam2MaskPropagator("sam2.1-tiny", device="cuda")
    assert seen_kwargs["model_kw"]["trust_remote_code"] is False
    assert seen_kwargs["proc_kw"]["trust_remote_code"] is False
    out = prop.propagate_window(
        [np.zeros((100, 200, 3), dtype=np.uint8)],
        {1: (0, (20.0, 10.0, 120.0, 60.0)), 2: (0, (100.0, 20.0, 180.0, 80.0))},
        (200, 100),
    )
    assert seen_kwargs["cond_frames"] == [0]
    assert set(out[0].keys()) == {1, 2}
    assert out[0][1].shape == (256, 256) and out[0][1].dtype == np.bool_


