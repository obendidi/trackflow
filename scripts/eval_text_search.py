"""Benchmark vision-language encoders (`--vlm`) on pedestrian attribute retrieval (PA-100K + UPAR/PETA).

Evaluates mAP across natural-language pedestrian clothing & accessory queries plus crop encoding
throughput (crops/sec) on Apple Silicon.

Usage:
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python scripts/eval_text_search.py \
        --models clip,siglip,siglip2-base,pe-core-b16 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from trackflow.detector import create_onnx_session
from trackflow.reid import _l2_normalize, create_vlm_embedder

DATA_ROOT = Path.home() / ".cache" / "trackflow" / "benchmarks" / "text_search"

PA100K_QUERIES: list[tuple[str, str]] = [
    ("Hat", "person wearing a hat"),
    ("Glasses", "person wearing glasses"),
    ("HandBag", "person carrying a handbag"),
    ("ShoulderBag", "person carrying a shoulder bag"),
    ("Backpack", "person wearing a backpack"),
    ("HoldObjectsInFront", "person holding something in front"),
    ("ShortSleeve", "person in a short sleeve shirt"),
    ("LongSleeve", "person in a long sleeve shirt"),
    ("UpperStride", "person wearing a striped shirt"),
    ("UpperLogo", "person wearing a shirt with a logo"),
    ("UpperPlaid", "person wearing a plaid shirt"),
    ("LongCoat", "person wearing a long coat"),
    ("Trousers", "person wearing long pants"),
    ("Shorts", "person wearing shorts"),
    ("Skirt&Dress", "person wearing a skirt or dress"),
    ("boots", "person wearing boots"),
    ("Female", "woman walking"),
    ("AgeOver60", "elderly person"),
]

UPAR_QUERIES: list[tuple[str, str]] = [
    ("UpperBody-Color-Red", "person wearing a red top"),
    ("UpperBody-Color-Blue", "person wearing a blue top"),
    ("UpperBody-Color-White", "person wearing a white top"),
    ("UpperBody-Color-Black", "person wearing a black top"),
    ("UpperBody-Color-Green", "person wearing a green top"),
    ("UpperBody-Color-Yellow", "person wearing a yellow top"),
    ("UpperBody-Color-Pink", "person wearing a pink top"),
    ("UpperBody-Color-Grey", "person wearing a grey top"),
    ("LowerBody-Color-Blue", "person wearing blue pants"),
    ("LowerBody-Color-Black", "person wearing black pants"),
    ("LowerBody-Color-White", "person wearing white pants"),
    ("LowerBody-Type-Skirt&Dress", "person wearing a skirt or dress"),
    ("Accessory-Backpack", "person wearing a backpack"),
    ("Accessory-Bag", "person carrying a bag"),
    ("Accessory-Hat", "person wearing a hat"),
    ("Accessory-Glasses-Normal", "person wearing eyeglasses"),
    ("Accessory-Glasses-Sun", "person wearing sunglasses"),
    ("Hair-Length-Long", "person with long hair"),
    ("Gender-Female", "woman"),
    ("UpperBody-Length-Short", "person in short sleeves"),
]

CANDIDATE_ONNX_VLMS: dict[str, dict[str, Any]] = {
    "siglip2-base": {
        "repo_id": "onnx-community/siglip2-base-patch16-224-ONNX",
        "revision": "ba1f3b0843f24bc5417d38e19c37b287d719b2f4",
        "vision_file": "onnx/vision_model_int8.onnx",
        "text_file": "onnx/text_model_int8.onnx",
        "pad_id": 0,
        "pad_to_max": True,
        "lowercase": True,
    },
    "siglip2-base-fp16": {
        "repo_id": "onnx-community/siglip2-base-patch16-224-ONNX",
        "revision": "ba1f3b0843f24bc5417d38e19c37b287d719b2f4",
        "vision_file": "onnx/vision_model_fp16.onnx",
        "text_file": "onnx/text_model_fp16.onnx",
        "pad_id": 0,
        "pad_to_max": True,
        "lowercase": True,
    },
    "siglip2-so400m": {
        "repo_id": "onnx-community/siglip2-so400m-patch16-256-ONNX",
        "revision": "7881d8f282cbd7cdeac1aca6bed493a36d58e713",
        "vision_file": "onnx/vision_model_int8.onnx",
        "text_file": "onnx/text_model_int8.onnx",
        "pad_id": 0,
        "pad_to_max": True,
        "lowercase": True,
    },
    "pe-core-b16": {
        "repo_id": "onnx-community/PE-Core-B16-224-ONNX",
        "revision": "7668f1df203c393f95310563ff999bfc8fd627b4",
        "vision_file": "onnx/vision_model_int8.onnx",
        "text_file": "onnx/text_model_int8.onnx",
        "pad_id": 0,
        "pad_to_max": True,
        "lowercase": False,
    },
}


class CandidateOnnxVlm:
    """Generic ONNX vision+text encoder wrapper for evaluating new Apache-2.0 VLM candidates."""

    def __init__(self, spec: dict[str, Any], device: str = "cpu") -> None:
        repo = spec["repo_id"]
        rev = spec["revision"]
        vpath = Path(hf_hub_download(repo, spec["vision_file"], revision=rev))
        tpath = Path(hf_hub_download(repo, spec["text_file"], revision=rev))
        tok_path = Path(hf_hub_download(repo, "tokenizer.json", revision=rev))
        prep_path = Path(hf_hub_download(repo, "preprocessor_config.json", revision=rev))
        cfg_path = Path(hf_hub_download(repo, "config.json", revision=rev))

        self._vision, _ = create_onnx_session(vpath, device=device, coreml_format=None)
        self._text, _ = create_onnx_session(tpath, device=device, coreml_format=None)
        self._tok = Tokenizer.from_file(str(tok_path))
        self._spec = spec

        prep = json.loads(prep_path.read_text(encoding="utf-8"))
        size = prep.get("crop_size") or prep.get("size") or 224
        if isinstance(size, int):
            self._hw = (size, size)
        else:
            self._hw = (
                int(size.get("height", size.get("shortest_edge", 224))),
                int(size.get("width", size.get("shortest_edge", 224))),
            )
        self._mean = np.array(prep.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(prep.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tcfg = cfg.get("text_config", {})
        self._max_len = int(tcfg.get("max_position_embeddings", 64))

    def encode_image(self, crop_bgr: np.ndarray) -> np.ndarray:
        th, tw = self._hw
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_CUBIC)
        chw = ((resized.astype(np.float32) / 255.0).transpose(2, 0, 1) - self._mean) / self._std
        inp = self._vision.get_inputs()[0]
        tensor = chw[None].astype(np.float16 if "float16" in inp.type else np.float32)
        outs = self._vision.run(None, {inp.name: tensor})
        names = [o.name for o in self._vision.get_outputs()]
        by_name = dict(zip(names, outs, strict=False))
        emb = by_name.get("image_embeds")
        if emb is None:
            emb = by_name.get("pooler_output", outs[0])
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        return _l2_normalize(arr[0])

    def encode_text(self, query: str) -> np.ndarray:
        text = query.strip()
        if self._spec.get("lowercase"):
            text = text.lower()
        self._tok.enable_truncation(max_length=self._max_len)
        if self._spec.get("pad_to_max"):
            self._tok.enable_padding(length=self._max_len, pad_id=int(self._spec["pad_id"]))
        else:
            self._tok.no_padding()
        enc = self._tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {}
        for inp in self._text.get_inputs():
            if inp.name == "input_ids":
                feed[inp.name] = ids
            elif inp.name == "attention_mask":
                feed[inp.name] = mask
        outs = self._text.run(None, feed)
        names = [o.name for o in self._text.get_outputs()]
        by_name = dict(zip(names, outs, strict=False))
        emb = by_name.get("text_embeds")
        if emb is None:
            emb = by_name.get("pooler_output", outs[0])
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        return _l2_normalize(arr[0])


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    return float((tp[y == 1] / ranks[y == 1]).mean())


def load_parquet_subset(
    path: Path, queries: list[tuple[str, str]], limit: int
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    table = pq.read_table(path)
    n = table.num_rows
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=min(limit, n), replace=False))
    sub = table.take(idx)
    imgs: list[np.ndarray] = []
    for item in sub["image"].to_pylist():
        raw = item["bytes"] if isinstance(item, dict) else item
        arr = np.frombuffer(raw, dtype=np.uint8)
        imgs.append(cv2.imdecode(arr, cv2.IMREAD_COLOR))
    labels: dict[str, np.ndarray] = {}
    for col, _ in queries:
        vals = sub[col].to_pylist()
        labels[col] = np.array([1 if v and int(v) > 0 else 0 for v in vals], dtype=np.int32)
    return imgs, labels


def load_model(name: str, device: str) -> Any:
    if name in ("clip", "siglip", "siglip2", "embeddinggemma"):
        return create_vlm_embedder(name, device=device)
    if name in CANDIDATE_ONNX_VLMS:
        return CandidateOnnxVlm(CANDIDATE_ONNX_VLMS[name], device=device)
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="clip,siglip,siglip2-base,pe-core-b16")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    datasets = [
        ("PA-100K", DATA_ROOT / "pa100k_val.parquet", PA100K_QUERIES),
        ("UPAR", DATA_ROOT / "upar_val.parquet", UPAR_QUERIES),
    ]
    loaded = {}
    for ds_name, path, qs in datasets:
        if path.is_file():
            loaded[ds_name] = (load_parquet_subset(path, qs, args.limit), qs)

    summary: dict[str, Any] = {}
    for mname in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            model = load_model(mname, args.device)
        except Exception as e:
            print(f"[SKIP] {mname}: {e}")
            continue
        m_res: dict[str, Any] = {}
        for ds_name, ((imgs, labels), qs) in loaded.items():
            t0 = time.perf_counter()
            embs = np.stack([model.encode_image(im) for im in imgs], axis=0)
            elapsed = max(time.perf_counter() - t0, 1e-6)
            crops_per_s = len(imgs) / elapsed
            aps: dict[str, float] = {}
            for col, prompt in qs:
                if int(labels[col].sum()) < 5:
                    continue
                qvec = model.encode_text(prompt)
                scores = embs @ qvec
                aps[col] = average_precision(scores, labels[col])
            mean_ap = float(np.nanmean(list(aps.values())))
            m_res[ds_name] = {
                "mAP": round(mean_ap, 4),
                "crops_per_s": round(crops_per_s, 1),
                "num_queries": len(aps),
                "per_query": {k: round(v, 4) for k, v in aps.items()},
            }
            print(
                f"| `{mname}` | {ds_name} ({len(imgs)}) | {mean_ap:.4f} | {len(aps)} | {crops_per_s:.1f} crops/s |",
                flush=True,
            )
        summary[mname] = m_res

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
