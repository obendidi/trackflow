"""Apache-2.0 ONNX Runtime detector supporting D-FINE, RT-DETRv2, and RF-DETR."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from trackflow.models import Precision, resolve_detector_artifacts
from trackflow.types import Detection

DEFAULT_COCO_LABELS: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    4: "aeroplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "sofa",
    58: "pottedplant",
    59: "bed",
    60: "diningtable",
    61: "toilet",
    62: "tvmonitor",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def select_providers(device: str = "auto") -> list[str]:
    """Choose ONNX Runtime execution providers based on requested device."""
    available = ort.get_available_providers()
    req = device.lower().strip()

    if req == "cpu":
        return ["CPUExecutionProvider"]
    if req == "cuda":
        return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if req == "coreml":
        return [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
    if req == "tensorrt":
        return [
            p
            for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ]

    # Auto-selection: prefer NVIDIA CUDA if present, else multi-threaded CPU.
    # (DETR/RF-DETR graphs split into 75–135 partitions on CoreML EP, making CPU 1.4x–3x faster on Apple Silicon;
    # users can still explicitly opt in with --device coreml.)
    preferred = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [p for p in preferred if p in available]
    return selected or ["CPUExecutionProvider"]


def create_onnx_session(
    onnx_path: Path,
    device: str = "auto",
) -> tuple[ort.InferenceSession, str]:
    """Create an ONNX Runtime session with safe fallback to CPU if an EP fails."""
    providers = select_providers(device)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    except Exception:
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

    active = session.get_providers()
    active_name = active[0] if active else "CPUExecutionProvider"
    return session, active_name


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clip = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _nms_class_aware(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Deduplicate overlapping same-class DETR queries so duplicate boxes don't spawn ghost tracks."""
    if len(detections) <= 1 or iou_threshold <= 0.0 or iou_threshold >= 1.0:
        return detections

    kept: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate = False
        for prev in kept:
            if prev.class_id != det.class_id:
                continue
            px1, py1, px2, py2 = prev.bbox
            iw = max(0.0, min(x2, px2) - max(x1, px1))
            ih = max(0.0, min(y2, py2) - max(y1, py1))
            inter = iw * ih
            if inter <= 0.0:
                continue
            area_b = max(0.0, px2 - px1) * max(0.0, py2 - py1)
            iou = inter / max(area_a + area_b - inter, 1e-6)
            if iou >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def postprocess_detr_outputs(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_hw: tuple[int, int],
    id2label: dict[int, str],
    threshold: float = 0.45,
    allowed_classes: set[str] | None = None,
    max_detections: int = 300,
    nms_iou: float = 0.65,
) -> list[Detection]:
    """Convert DETR/RT-DETR/D-FINE/RF-DETR raw logits + normalized cxcywh boxes into Detections."""
    # Squeeze batch dimension [1, N, C] -> [N, C]
    if logits.ndim == 3:
        logits = logits[0]
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes[0]

    probs = sigmoid(logits)
    class_ids = np.argmax(probs, axis=-1)
    scores = probs[np.arange(probs.shape[0]), class_ids]

    keep = scores >= threshold
    if not np.any(keep):
        return []

    kept_scores = scores[keep]
    kept_classes = class_ids[keep]
    kept_boxes = pred_boxes[keep]

    order = np.argsort(-kept_scores)[:max_detections]
    orig_h, orig_w = orig_hw

    detections: list[Detection] = []
    for idx in order:
        cid = int(kept_classes[idx])
        cname = id2label.get(cid, str(cid))
        if allowed_classes is not None and cname.lower() not in allowed_classes:
            continue

        cx, cy, w, h = kept_boxes[idx].tolist()
        # Handle both normalized [0, 1] and already-pixel coordinates safely.
        if max(cx, cy, w, h) <= 2.0:
            x1 = (cx - 0.5 * w) * orig_w
            y1 = (cy - 0.5 * h) * orig_h
            x2 = (cx + 0.5 * w) * orig_w
            y2 = (cy + 0.5 * h) * orig_h
        else:
            x1, y1, x2, y2 = cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h

        x1 = float(np.clip(x1, 0.0, orig_w))
        y1 = float(np.clip(y1, 0.0, orig_h))
        x2 = float(np.clip(x2, 0.0, orig_w))
        y2 = float(np.clip(y2, 0.0, orig_h))

        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue

        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(kept_scores[idx]),
                class_id=cid,
                class_name=cname,
            )
        )

    return _nms_class_aware(detections, iou_threshold=nms_iou)


class OnnxDetrDetector:
    """End-to-end transformer object detector powered by ONNX Runtime."""

    def __init__(
        self,
        model: str = "rtdetr-v2-r18",
        *,
        precision: Precision = "fp32",
        device: str = "auto",
        threshold: float = 0.45,
        classes: Sequence[str] | None = ("person",),
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model
        self.precision = precision
        self.threshold = float(threshold)
        self.allowed_classes = (
            None
            if not classes or any(c.lower() in ("*", "all") for c in classes)
            else {c.lower().strip() for c in classes}
        )

        onnx_path, config_path, prep_path = resolve_detector_artifacts(
            model=model,
            precision=precision,
            cache_dir=cache_dir,
        )
        self.onnx_path = onnx_path
        self.id2label = self._load_labels(config_path)
        self.input_hw, self.do_rescale, self.do_normalize, self.mean, self.std = (
            self._load_preprocess_config(prep_path)
        )

        # CoreML EP on int8 quantized DETR graphs can occasionally fail or degrade; prefer CPU for int8.
        effective_device = "cpu" if (device == "auto" and precision == "int8") else device
        self.session, self.active_provider = create_onnx_session(onnx_path, device=effective_device)

        self.input_name = self.session.get_inputs()[0].name
        self.input_dtype = (
            np.float16 if "float16" in self.session.get_inputs()[0].type else np.float32
        )
        self.output_names = [o.name for o in self.session.get_outputs()]

    @staticmethod
    def _load_labels(config_path: Path | None) -> dict[int, str]:
        if config_path is None or not config_path.is_file():
            return dict(DEFAULT_COCO_LABELS)
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        id2label_raw = raw.get("id2label")
        if not isinstance(id2label_raw, dict):
            return dict(DEFAULT_COCO_LABELS)
        return {int(k): str(v) for k, v in id2label_raw.items()}

    @staticmethod
    def _load_preprocess_config(
        prep_path: Path | None,
    ) -> tuple[tuple[int, int], bool, bool, np.ndarray, np.ndarray]:
        default_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        default_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        if prep_path is None or not prep_path.is_file():
            return (640, 640), True, False, default_mean, default_std

        cfg: dict[str, Any] = json.loads(prep_path.read_text(encoding="utf-8"))
        size = cfg.get("size", {})
        h = int(size.get("height", 640))
        w = int(size.get("width", 640))
        do_rescale = bool(cfg.get("do_rescale", True))
        do_normalize = bool(cfg.get("do_normalize", False))
        mean = np.array(cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(
            3, 1, 1
        )
        std = np.array(cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(
            3, 1, 1
        )
        return (h, w), do_rescale, do_normalize, mean, std

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert OpenCV BGR uint8 frame to NCHW float tensor."""
        target_h, target_w = self.input_hw
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32)
        if self.do_rescale:
            arr *= 1.0 / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        if self.do_normalize:
            chw = (chw - self.mean) / self.std
        return np.expand_dims(chw, axis=0).astype(self.input_dtype, copy=False)

    def detect(self, frame_bgr: np.ndarray, threshold: float | None = None) -> list[Detection]:
        """Run object detection on a single BGR frame."""
        orig_h, orig_w = frame_bgr.shape[:2]
        tensor = self.preprocess(frame_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})

        by_name = dict(zip(self.output_names, outputs, strict=False))
        logits = by_name.get("logits", outputs[0])
        pred_boxes = by_name.get("pred_boxes", outputs[1])

        return postprocess_detr_outputs(
            logits=np.asarray(logits, dtype=np.float32),
            pred_boxes=np.asarray(pred_boxes, dtype=np.float32),
            orig_hw=(orig_h, orig_w),
            id2label=self.id2label,
            threshold=self.threshold if threshold is None else threshold,
            allowed_classes=self.allowed_classes,
        )
