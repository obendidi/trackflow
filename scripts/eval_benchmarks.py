"""Reproducible MOT benchmark harness for DanceTrack, SportsMOT, MOT17, and MOT20.

Separates two evaluation tracks:
- Track A (Tracker association only): replays identical cached detections across Trackflow
  trackers and Roboflow `trackers==2.7.0` controls (`rf:bytetrack`, `rf:botsort`, `rf:ocsort`,
  `rf:cbiou`) on the exact same frames.
- Track B (End-to-end Apache-2.0 stack): runs Trackflow detectors (`--models`) with `--device`,
  records detector FPS, tracker FPS, and end-to-end FPS alongside HOTA/AssA/DetA/IDF1/MOTA/IDSW.

Usage:
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python scripts/eval_benchmarks.py \
        --datasets dancetrack-dev,sportsmot-dev,mot17-val,mot20-dev \
        --models rtdetr-v2-r18 \
        --trackers bytetrack,botsort,botsort+osnet,ocsort,boosttrack,boosttrack+osnet \
        --device coreml
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from trackflow import Detection, TrackflowPipeline
from trackflow.reid import ReidEmbedder

TRACKERS_PKG = "trackers==2.7.0"
DEFAULT_DATA_ROOT = Path.home() / ".cache" / "trackflow" / "benchmarks"

# Deterministic stratified splits (zero video-ID overlap between dev and holdout).
SPLITS: dict[str, dict[str, Any]] = {
    "dancetrack-dev": {
        "base": "dancetrack/val",
        "class_config": "mot17",
        "seqs": [
            "dancetrack0018",
            "dancetrack0026",
            "dancetrack0035",
            "dancetrack0043",
            "dancetrack0073",
            "dancetrack0094",
        ],
    },
    "dancetrack-holdout": {
        "base": "dancetrack/val",
        "class_config": "mot17",
        "seqs": [
            "dancetrack0025",
            "dancetrack0041",
            "dancetrack0065",
            "dancetrack0081",
            "dancetrack0090",
        ],
    },
    "dancetrack-val": {
        "base": "dancetrack/val",
        "class_config": "mot17",
        "seqs": None,
    },
    "sportsmot-dev": {
        "base": "sportsmot/val",
        "class_config": "mot17",
        "seqs": [
            "v_0kUtTtmLaJA_c005",
            "v_2QhNRucNC7E_c017",
            "v_4-EmEtrturE_c009",
            "v_5ekaksddqrc_c001",
            "v_9MHDmAMxO5I_c004",
            "v_BgwzTUxJaeU_c008",
        ],
    },
    "sportsmot-holdout": {
        "base": "sportsmot/val",
        "class_config": "mot17",
        "seqs": [
            "v_00HRwkvvjtQ_c005",
            "v_G-vNjfx1GGc_c008",
            "v_ITo3sCnpw_k_c010",
            "v_cC2mHWqMcjk_c009",
            "v_dw7LOz17Omg_c067",
            "v_i2_L4qquVg0_c007",
        ],
    },
    "sportsmot-val": {
        "base": "sportsmot/val",
        "class_config": "mot17",
        "seqs": None,
    },
    "mot17-val": {
        "base": "mot17/mot17/val",
        "class_config": "mot17",
        "seqs": None,
    },
    "mot20-dev": {
        "base": "mot20/val",
        "class_config": "mot20",
        "seqs": ["MOT20-01", "MOT20-03"],
    },
    "mot20-val": {
        "base": "mot20/val",
        "class_config": "mot20",
        "seqs": None,
    },
}


def frame_rate(sequence: Path) -> float:
    info = configparser.ConfigParser()
    info.read(sequence / "seqinfo.ini")
    return float(info["Sequence"]["frameRate"])


def resolve_sequences(data_root: Path, dataset: str) -> tuple[Path, list[Path], str]:
    if dataset not in SPLITS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from: {', '.join(SPLITS)}")
    spec = SPLITS[dataset]
    base = data_root / spec["base"]
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {base}")
    if spec["seqs"] is None:
        seqs = sorted(p for p in base.iterdir() if (p / "img1").is_dir() and (p / "gt" / "gt.txt").is_file())
    else:
        seqs = [base / s for s in spec["seqs"]]
        missing = [s.name for s in seqs if not (s / "img1").is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing sequences in {base}: {missing}")
    if not seqs:
        raise FileNotFoundError(f"No valid sequences found in {base}")
    return base, seqs, spec["class_config"]


def _cache_key(model: str, device: str, threshold: float = 0.10) -> str:
    from trackflow.models import MODEL_REGISTRY

    spec = MODEL_REGISTRY.get(model)
    safe_model = model.replace("/", "__").replace(":", "_")
    # Include preprocessing/revision suffix when normalization or threshold differs from legacy cache.
    suffix = ""
    if spec is not None and spec.normalize:
        suffix += f"_norm1_{(spec.revision or 'none')[:8]}"
    if abs(threshold - 0.10) > 1e-6:
        suffix += f"_t{threshold:.2f}"
    return f"{safe_model}__{device}{suffix}"


def load_or_compute_detections(
    data_root: Path,
    dataset_base_name: str,
    seq: Path,
    model: str,
    device: str,
    detector_holder: list[Any],
    threshold: float = 0.10,
) -> tuple[list[list[Detection]], float]:
    """Return per-frame detections and detector FPS (cached to disk per model+device+seq)."""
    cache_path = (
        data_root / "det_cache" / _cache_key(model, device) / dataset_base_name / f"{seq.name}.npz"
    )
    meta_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(cache_path)
        counts = data["counts"]
        boxes = data["boxes"]
        scores = data["scores"]
        class_ids = data["class_ids"]
        frames_dets: list[list[Detection]] = []
        off = 0
        for c in counts.tolist():
            f_dets = []
            for i in range(off, off + c):
                b = boxes[i]
                f_dets.append(
                    Detection(
                        bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        confidence=float(scores[i]),
                        class_id=int(class_ids[i]),
                        class_name="person",
                    )
                )
            frames_dets.append(f_dets)
            off += c
        return frames_dets, float(meta["det_fps"])

    if not detector_holder:
        # Use TrackflowPipeline's detector initialization with low threshold so all trackers
        # (including BoostTrack boosting and ByteTrack low-score stage) see identical boxes.
        pipe = TrackflowPipeline(model=model, device=device, threshold=0.45)
        if hasattr(pipe.detector, "threshold"):
            pipe.detector.threshold = min(pipe.detector.threshold, threshold)
        detector_holder.append(pipe.detector)
    detector = detector_holder[0]

    img_paths = sorted((seq / "img1").glob("*.jpg"))
    print(f"[detect] {model} ({device}) on {dataset_base_name}/{seq.name} ({len(img_paths)} frames)...", file=sys.stderr)
    frames_dets = []
    counts_list: list[int] = []
    boxes_list: list[list[float]] = []
    scores_list: list[float] = []
    cids_list: list[int] = []

    t0 = time.perf_counter()
    for p in img_paths:
        frame = cv2.imread(str(p))
        dets = detector.detect(frame)
        # Keep only person detections (MOT benchmarks evaluate pedestrians/players/dancers).
        person_dets = [d for d in dets if d.class_name.lower() == "person"]
        frames_dets.append(person_dets)
        counts_list.append(len(person_dets))
        for d in person_dets:
            boxes_list.append(list(d.bbox))
            scores_list.append(d.confidence)
            cids_list.append(d.class_id)
    elapsed = max(time.perf_counter() - t0, 1e-6)
    det_fps = len(img_paths) / elapsed

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        counts=np.array(counts_list, dtype=np.int32),
        boxes=np.array(boxes_list, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores_list, dtype=np.float32),
        class_ids=np.array(cids_list, dtype=np.int32),
    )
    meta_path.write_text(
        json.dumps({"model": model, "device": device, "frames": len(img_paths), "det_fps": det_fps}),
        encoding="utf-8",
    )
    return frames_dets, det_fps


def load_or_compute_reid_embeddings(
    data_root: Path,
    dataset_base_name: str,
    seq: Path,
    model: str,
    device: str,
    reid_name: str,
    frames_dets: list[list[Detection]],
    reid_holder: list[Any],
    min_conf: float = 0.10,
) -> tuple[list[list[np.ndarray | None]], float]:
    """Cache per-detection Re-ID embeddings for fast, deterministic Track A evaluation."""
    cache_path = (
        data_root
        / "reid_cache"
        / _cache_key(model, device)
        / f"{reid_name}_c{min_conf:.2f}"
        / dataset_base_name
        / f"{seq.name}.npz"
    )
    meta_path = cache_path.with_suffix(".json")
    total_dets = sum(len(f) for f in frames_dets)

    if cache_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("total_dets") == total_dets:
            data = np.load(cache_path)
            valid_mask = data["valid_mask"]
            embs_flat = data["embeddings"]
            out: list[list[np.ndarray | None]] = []
            idx = 0
            e_idx = 0
            for f_dets in frames_dets:
                f_embs: list[np.ndarray | None] = []
                for _ in f_dets:
                    if valid_mask[idx]:
                        f_embs.append(embs_flat[e_idx])
                        e_idx += 1
                    else:
                        f_embs.append(None)
                    idx += 1
                out.append(f_embs)
            return out, float(meta["reid_fps"])

    if not reid_holder:
        reid_holder.append(ReidEmbedder(backend=reid_name, device=device))  # type: ignore[arg-type]
    embedder: ReidEmbedder = reid_holder[0]

    img_paths = sorted((seq / "img1").glob("*.jpg"))
    print(f"[reid] {reid_name} ({device}) on {dataset_base_name}/{seq.name}...", file=sys.stderr)
    valid_mask_list: list[bool] = []
    embs_collected: list[np.ndarray] = []
    out = []

    t0 = time.perf_counter()
    for p, f_dets in zip(img_paths, frames_dets, strict=True):
        frame = cv2.imread(str(p))
        h, w = frame.shape[:2]
        crops: list[np.ndarray | None] = []
        for d in f_dets:
            if d.confidence < min_conf:
                crops.append(None)
                continue
            x1, y1 = max(0, int(round(d.bbox[0]))), max(0, int(round(d.bbox[1])))
            x2, y2 = min(w, int(round(d.bbox[2]))), min(h, int(round(d.bbox[3])))
            crops.append(frame[y1:y2, x1:x2] if (x2 - x1 >= 4 and y2 - y1 >= 4) else None)
        f_embs = embedder.encode_images(crops)
        out.append(f_embs)
        for e in f_embs:
            if e is not None:
                valid_mask_list.append(True)
                embs_collected.append(e.astype(np.float16))
            else:
                valid_mask_list.append(False)
    elapsed = max(time.perf_counter() - t0, 1e-6)
    reid_fps = len(img_paths) / elapsed

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emb_arr = (
        np.stack(embs_collected, axis=0)
        if embs_collected
        else np.zeros((0, 512), dtype=np.float16)
    )
    np.savez_compressed(
        cache_path,
        valid_mask=np.array(valid_mask_list, dtype=bool),
        embeddings=emb_arr,
    )
    meta_path.write_text(
        json.dumps({"total_dets": total_dets, "reid_fps": reid_fps}),
        encoding="utf-8",
    )
    return out, reid_fps


def write_mot_txt(path: Path, rows: list[tuple[int, int, float, float, float, float, float]]) -> None:
    lines = [
        f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1,-1\n"
        for frame_id, track_id, x1, y1, x2, y2, conf in rows
    ]
    path.write_text("".join(lines), encoding="utf-8")


def run_trackflow_tracker(
    tracker_spec: str,
    seqs: list[Path],
    dets_by_seq: dict[str, list[list[Detection]]],
    embs_by_seq: dict[str, list[list[np.ndarray | None]]] | None,
    pred_dir: Path,
    threshold: float = 0.45,
    postprocess: bool = False,
) -> float:
    """Run a Trackflow tracker across sequences and return tracker-only FPS."""
    from trackflow.pipeline import _match_track_embeddings, _pack_postprocess_embeddings
    from trackflow.postprocess import postprocess_records
    from trackflow.tracker import create_tracker

    tracker_name, _, reid_name = tracker_spec.partition("+")
    pred_dir.mkdir(parents=True, exist_ok=True)
    need_frames = tracker_name in ("botsort", "boosttrack", "ocsort", "nltrack")

    total_frames = 0
    total_trk_s = 0.0

    for seq in seqs:
        info = configparser.ConfigParser()
        info.read(seq / "seqinfo.ini")
        fps = float(info["Sequence"]["frameRate"])
        seq_w = int(info["Sequence"]["imWidth"])
        seq_h = int(info["Sequence"]["imHeight"])
        tracker = create_tracker(
            tracker_name,
            track_high_thresh=threshold,
            track_low_thresh=min(threshold, max(0.10, threshold * 0.35)),
            new_track_thresh=max(threshold, min(0.95, threshold + 0.05)),
        )
        min_conf = getattr(tracker, "embedding_min_confidence", None)
        img_paths = sorted((seq / "img1").glob("*.jpg"))
        seq_dets = dets_by_seq[seq.name]
        seq_embs = embs_by_seq[seq.name] if (embs_by_seq and reid_name) else None

        mot_rows: list[tuple[int, int, float, float, float, float, float]] = []
        pp_records: list[dict] = []
        obs_embs: list[np.ndarray | None] = []
        cmc_warps = np.broadcast_to(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            (len(img_paths), 2, 3),
        ).copy()
        cmc_valid = np.zeros((len(img_paths),), dtype=bool)
        for f_idx, (img_p, raw_dets) in enumerate(zip(img_paths, seq_dets, strict=True), start=1):
            frame_bgr = cv2.imread(str(img_p)) if need_frames else None
            # Fresh Detection objects with optional cached embeddings attached
            frame_dets: list[Detection] = []
            emb_dets: list[Detection] = []
            f_embs = seq_embs[f_idx - 1] if seq_embs is not None else None
            for d_i, d in enumerate(raw_dets):
                raw_e = f_embs[d_i] if f_embs is not None else None
                emb = (
                    raw_e.astype(np.float32)
                    if (raw_e is not None and min_conf is not None and d.confidence >= min_conf)
                    else None
                )
                det_obj = Detection(
                    bbox=d.bbox,
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    embedding=emb,
                )
                frame_dets.append(det_obj)
                if postprocess and raw_e is not None:
                    emb_dets.append(
                        det_obj
                        if emb is not None
                        else Detection(
                            bbox=d.bbox,
                            confidence=d.confidence,
                            class_id=d.class_id,
                            class_name=d.class_name,
                            embedding=raw_e.astype(np.float32),
                        )
                    )
            t0 = time.perf_counter()
            tracks = tracker.update(frame_dets, frame_bgr=frame_bgr)
            total_trk_s += time.perf_counter() - t0
            total_frames += 1
            warp = getattr(tracker, "last_cmc_warp", None)
            if warp is not None:
                cmc_warps[f_idx - 1] = np.asarray(warp, dtype=np.float32)
                cmc_valid[f_idx - 1] = True
            matched_embs = (
                _match_track_embeddings(tracks, emb_dets, None, None)
                if postprocess
                else [None] * len(tracks)
            )
            for tr, m_emb in zip(tracks, matched_embs, strict=True):
                x1, y1, x2, y2 = tr.bbox
                mot_rows.append((f_idx, tr.track_id, x1, y1, x2, y2, tr.confidence))
                if postprocess:
                    rec = tr.to_record(f_idx - 1, (f_idx - 1) / fps)
                    rec["x1"], rec["y1"], rec["x2"], rec["y2"] = tr.bbox
                    pp_records.append(rec)
                    obs_embs.append(m_emb)
        if postprocess and pp_records:
            emb_dense, emb_valid = _pack_postprocess_embeddings(obs_embs)
            t0 = time.perf_counter()
            pp_res = postprocess_records(
                pp_records,
                fps=fps,
                frame_width=seq_w,
                frame_height=seq_h,
                embeddings=emb_dense,
                emb_valid=emb_valid,
                cmc_warps=cmc_warps if cmc_valid.any() else None,
                cmc_valid=cmc_valid if cmc_valid.any() else None,
                frame_stride=1,
            )
            total_trk_s += time.perf_counter() - t0
            mot_rows = [
                (
                    int(r["frame_idx"]) + 1,
                    int(r["track_id"]),
                    float(r["x1"]),
                    float(r["y1"]),
                    float(r["x2"]),
                    float(r["y2"]),
                    float(r["confidence"]),
                )
                for r in pp_res.records
            ]
        write_mot_txt(pred_dir / f"{seq.name}.txt", mot_rows)

    return total_frames / max(total_trk_s, 1e-6)


def run_roboflow_tracker(
    rf_name: str,
    seqs: list[Path],
    dets_by_seq: dict[str, list[list[Detection]]],
    pred_dir: Path,
    threshold: float = 0.45,
) -> float:
    """Run Roboflow trackers==2.7.0 in its uvx env on identical detections and frames."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rf-track-") as td:
        spec_path = Path(td) / "job.json"
        dets_dir = Path(td) / "dets"
        dets_dir.mkdir()
        seq_meta = []
        for seq in seqs:
            seq_dets = dets_by_seq[seq.name]
            counts = [len(f) for f in seq_dets]
            boxes = [list(d.bbox) for f in seq_dets for d in f]
            scores = [d.confidence for f in seq_dets for d in f]
            cids = [d.class_id for f in seq_dets for d in f]
            npz = dets_dir / f"{seq.name}.npz"
            np.savez(
                npz,
                counts=np.array(counts, dtype=np.int32),
                boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                scores=np.array(scores, dtype=np.float32),
                class_ids=np.array(cids, dtype=np.int32),
            )
            seq_meta.append(
                {
                    "name": seq.name,
                    "img_dir": str(seq / "img1"),
                    "fps": frame_rate(seq),
                    "npz": str(npz),
                }
            )
        spec_path.write_text(
            json.dumps(
                {
                    "tracker": rf_name,
                    "threshold": threshold,
                    "pred_dir": str(pred_dir),
                    "seqs": seq_meta,
                }
            ),
            encoding="utf-8",
        )
        runner = r"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, supervision as sv, trackers

job = json.loads(Path(sys.argv[1]).read_text())
tname = job["tracker"]
pred_dir = Path(job["pred_dir"])
need_frames = tname in ("botsort", "botsort_cmc", "ocsort_cmc", "botsort_aligned", "ocsort_cmc_aligned")

total_frames = 0
total_s = 0.0
thresh = float(job["threshold"])
act = min(0.95, thresh + 0.05)

for sm in job["seqs"]:
    fps = float(sm["fps"])
    if tname == "bytetrack":
        tr = trackers.ByteTrackTracker(frame_rate=fps)
    elif tname == "bytetrack_aligned":
        tr = trackers.ByteTrackTracker(frame_rate=fps, high_conf_det_threshold=thresh, track_activation_threshold=act)
    elif tname == "botsort":
        tr = trackers.BoTSORTTracker(frame_rate=fps, enable_cmc=True)
    elif tname == "botsort_aligned":
        tr = trackers.BoTSORTTracker(frame_rate=fps, enable_cmc=True, high_conf_det_threshold=thresh, track_activation_threshold=act)
    elif tname == "ocsort":
        tr = trackers.OCSORTTracker(frame_rate=fps, enable_cmc=False)
    elif tname == "ocsort_aligned":
        tr = trackers.OCSORTTracker(frame_rate=fps, enable_cmc=False, high_conf_det_threshold=thresh)
    elif tname == "ocsort_cmc":
        tr = trackers.OCSORTTracker(frame_rate=fps, enable_cmc=True)
    elif tname == "ocsort_cmc_aligned":
        tr = trackers.OCSORTTracker(frame_rate=fps, enable_cmc=True, high_conf_det_threshold=thresh)
    elif tname == "cbiou":
        tr = trackers.CBIoUTracker(frame_rate=fps)
    elif tname == "cbiou_aligned":
        tr = trackers.CBIoUTracker(frame_rate=fps, high_conf_det_threshold=thresh, track_activation_threshold=act)
    else:
        raise ValueError(f"Unknown rf tracker: {tname}")

    data = np.load(sm["npz"])
    counts, boxes, scores, cids = data["counts"], data["boxes"], data["scores"], data["class_ids"]
    imgs = sorted(Path(sm["img_dir"]).glob("*.jpg"))
    off = 0
    lines = []
    for f_idx, (img_p, c) in enumerate(zip(imgs, counts.tolist()), start=1):
        frame = cv2.imread(str(img_p)) if need_frames else None
        if c > 0:
            det = sv.Detections(
                xyxy=boxes[off:off+c].astype(np.float32),
                confidence=scores[off:off+c].astype(np.float32),
                class_id=cids[off:off+c].astype(np.int32),
            )
        else:
            det = sv.Detections.empty()
        off += c
        t0 = time.perf_counter()
        out = tr.update(det, frame=frame)
        total_s += time.perf_counter() - t0
        total_frames += 1
        if len(out) > 0 and out.tracker_id is not None:
            for xyxy, tid, conf in zip(out.xyxy, out.tracker_id, out.confidence):
                if tid is None or int(tid) < 0:
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                lines.append(f"{f_idx},{int(tid)},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{float(conf):.4f},-1,-1,-1\n")
    (pred_dir / f"{sm['name']}.txt").write_text("".join(lines), encoding="utf-8")

print(json.dumps({"trk_fps": total_frames / max(total_s, 1e-6)}))
"""
        res = subprocess.run(
            ["uvx", "--from", TRACKERS_PKG, "python", "-c", runner, str(spec_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in reversed(res.stdout.strip().splitlines()):
            if line.startswith("{"):
                return float(json.loads(line)["trk_fps"])
        raise RuntimeError(f"Unexpected Roboflow runner output: {res.stdout}\n{res.stderr}")


def eval_mot_predictions(
    gt_dir: Path,
    pred_dir: Path,
    seqs: list[Path],
    class_config: str,
    scores_path: Path,
) -> dict[str, float]:
    """Score predictions with Roboflow trackers eval (TrackEval-equivalent HOTA/AssA/DetA/IDF1/MOTA/IDSW)."""
    if shutil.which("uvx") is None:
        raise RuntimeError("uvx is required to run trackers eval.")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as smap:
        smap.write("name\n" + "\n".join(s.name for s in seqs) + "\n")
        seqmap_path = Path(smap.name)
    try:
        subprocess.run(
            [
                "uvx",
                "--from",
                TRACKERS_PKG,
                "trackers",
                "eval",
                "--gt_dir",
                str(gt_dir),
                "--predictions_dir",
                str(pred_dir),
                "--seqmap",
                str(seqmap_path),
                "--class_config",
                class_config,
                "--metrics",
                "[CLEAR,HOTA,Identity]",
                "--output",
                str(scores_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        seqmap_path.unlink(missing_ok=True)

    raw = json.loads(scores_path.read_text(encoding="utf-8"))
    agg = raw["aggregate"]
    hota = agg["HOTA"]
    clear = agg["CLEAR"]
    ident = agg["Identity"]
    return {
        "HOTA": 100.0 * float(hota["HOTA"]),
        "AssA": 100.0 * float(hota["AssA"]),
        "DetA": 100.0 * float(hota["DetA"]),
        "IDF1": 100.0 * float(ident["IDF1"]),
        "MOTA": 100.0 * float(clear["MOTA"]),
        "IDSW": int(clear["IDSW"]),
    }


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--datasets",
        default="dancetrack-dev,sportsmot-dev,mot17-val",
        help="comma-separated dataset splits",
    )
    parser.add_argument("--models", default="rtdetr-v2-r18", help="comma-separated detector presets")
    parser.add_argument(
        "--trackers",
        default="bytetrack,botsort,botsort+osnet,ocsort,boosttrack",
        help="comma-separated trackers (prefix `rf:` for Roboflow trackers==2.7.0)",
    )
    parser.add_argument("--device", default="coreml", help="detector/reid device")
    parser.add_argument("--threshold", type=float, default=0.45, help="tracking high threshold")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="apply Trackflow offline post-processing pass after tracking",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="write raw JSON results here")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tracker_list = [t.strip() for t in args.trackers.split(",") if t.strip()]

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="trackflow-bench-") as tmp_root:
        tmp = Path(tmp_root)
        for model in models:
            det_holder: list[Any] = []
            reid_holders: dict[str, list[Any]] = {}

            for ds in datasets:
                gt_dir, seqs, class_cfg = resolve_sequences(args.data, ds)
                ds_base = SPLITS[ds]["base"].replace("/", "_")
                dets_by_seq: dict[str, list[list[Detection]]] = {}
                det_fps_vals: list[float] = []
                for seq in seqs:
                    f_dets, d_fps = load_or_compute_detections(
                        args.data, ds_base, seq, model, args.device, det_holder
                    )
                    dets_by_seq[seq.name] = f_dets
                    det_fps_vals.append(d_fps)
                mean_det_fps = float(np.mean(det_fps_vals))

                # Precompute Re-ID embeddings only for requested +reid suffixes at required min_conf
                embs_cache: dict[tuple[str, float], tuple[dict[str, list[list[np.ndarray | None]]], float]] = {}
                for tspec in tracker_list:
                    if tspec.startswith("rf:"):
                        continue
                    tname, _, rname = tspec.partition("+")
                    if not rname:
                        continue
                    req_min_conf = 0.10 if tname == "boosttrack" else args.threshold
                    ckey = (rname, req_min_conf)
                    if ckey not in embs_cache:
                        holder = reid_holders.setdefault(rname, [])
                        seq_embs: dict[str, list[list[np.ndarray | None]]] = {}
                        rfps_vals: list[float] = []
                        for seq in seqs:
                            embs, rfps = load_or_compute_reid_embeddings(
                                args.data,
                                ds_base,
                                seq,
                                model,
                                args.device,
                                rname,
                                dets_by_seq[seq.name],
                                holder,
                                min_conf=req_min_conf,
                            )
                            seq_embs[seq.name] = embs
                            rfps_vals.append(rfps)
                        embs_cache[ckey] = (seq_embs, float(np.mean(rfps_vals)))

                for tspec in tracker_list:
                    run_dir = tmp / f"{model}__{ds}__{tspec.replace(':', '_').replace('+', '_')}"
                    pred_dir = run_dir / "mot"
                    if tspec.startswith("rf:"):
                        rf_name = tspec.removeprefix("rf:")
                        trk_fps = run_roboflow_tracker(
                            rf_name, seqs, dets_by_seq, pred_dir, threshold=args.threshold
                        )
                        reid_fps = None
                    else:
                        tname, _, rname = tspec.partition("+")
                        req_min_conf = 0.10 if tname == "boosttrack" else args.threshold
                        seq_embs, reid_fps = (
                            embs_cache[(rname, req_min_conf)] if rname else (None, None)
                        )
                        trk_fps = run_trackflow_tracker(
                            tspec,
                            seqs,
                            dets_by_seq,
                            seq_embs,
                            pred_dir,
                            threshold=args.threshold,
                            postprocess=args.postprocess,
                        )
                    metrics = eval_mot_predictions(
                        gt_dir, pred_dir, seqs, class_cfg, run_dir / "scores.json"
                    )
                    inv_e2e = (1.0 / mean_det_fps) + (1.0 / trk_fps) + (
                        (1.0 / reid_fps) if reid_fps else 0.0
                    )
                    e2e_fps = 1.0 / inv_e2e
                    row = {
                        "dataset": ds,
                        "model": model,
                        "device": args.device,
                        "tracker": tspec,
                        **metrics,
                        "det_fps": round(mean_det_fps, 1),
                        "trk_fps": round(trk_fps, 1),
                        "e2e_fps": round(e2e_fps, 1),
                    }
                    results.append(row)
                    print(
                        f"| {ds} | `{model}` | `{tspec}` | {row['HOTA']:.2f} | {row['AssA']:.2f} "
                        f"| {row['DetA']:.2f} | {row['IDF1']:.2f} | {row['MOTA']:.2f} "
                        f"| {row['IDSW']} | {row['trk_fps']:.1f} | {row['e2e_fps']:.1f} |",
                        flush=True,
                    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
