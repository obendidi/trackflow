"""Detector adapters for ONNX Runtime and optional Hugging Face Transformers models."""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import onnxruntime as ort

from trackflow.models import MODEL_REGISTRY, Precision, resolve_detector_artifacts
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


class Detector(Protocol):
    """A detector accepts one BGR uint8 frame and returns pixel-space xyxy boxes."""

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


_COCO_ALIASES = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "tvmonitor": "tv",
}


def _class_key(name: str) -> str:
    name = name.strip().lower()
    return _COCO_ALIASES.get(name, name)


def _allowed_classes(classes: Sequence[str] | None) -> set[str] | None:
    if isinstance(classes, str):
        classes = classes.split(",")
    names = {_class_key(c) for c in classes or () if c.strip()}
    if not names or names.intersection({"*", "all"}):
        return None
    return names


def select_providers(device: str = "auto") -> list[str]:
    """Choose ONNX Runtime execution providers based on requested device."""
    available = ort.get_available_providers()
    req = device.lower().strip()

    options = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    if req != "auto":
        if req not in options:
            raise ValueError(
                f"Unknown ONNX device {device!r}. Choose auto, cpu, coreml, cuda, or tensorrt."
            )
        if options[req][0] not in available:
            raise ValueError(
                f"{device} requires {options[req][0]}. Available providers: {available}"
            )
        return [p for p in options[req] if p in available]

    # Auto-selection: prefer NVIDIA CUDA if present, else multi-threaded CPU. On Apple Silicon,
    # CoreML runs part of each DETR graph on the CPU: it frees most CPU cores but is slower for
    # RF-DETR, so it stays opt-in with --device coreml.
    preferred = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [p for p in preferred if p in available]
    return selected or ["CPUExecutionProvider"]


def create_onnx_session(
    onnx_path: Path,
    device: str = "auto",
    *,
    optimization: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    coreml_format: str | None = "MLProgram",
) -> tuple[ort.InferenceSession, str]:
    """Create an ONNX Runtime session with safe fallback to CPU if an EP fails."""
    providers = select_providers(device)
    # CoreML runs on the GPU, not the Neural Engine, so results match the CPU's fp32.
    # ``coreml_format=None`` keeps ONNX Runtime's default CoreML setup.
    provider_options = [
        {"ModelFormat": coreml_format, "MLComputeUnits": "CPUAndGPU"}
        if p == "CoreMLExecutionProvider" and coreml_format is not None
        else {}
        for p in providers
    ]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = optimization

    try:
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options,
        )
    except Exception:
        if device.strip().lower() != "auto" or providers == ["CPUExecutionProvider"]:
            raise
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

    active = session.get_providers()
    active_name = active[0] if active else "CPUExecutionProvider"
    if device.strip().lower() != "auto" and active_name != providers[0]:
        raise RuntimeError(f"Could not start {providers[0]}. ONNX Runtime selected {active_name}.")
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


def _letterbox(
    frame_bgr: np.ndarray, input_hw: tuple[int, int], pad: int
) -> tuple[np.ndarray, tuple[int, int, float, float]]:
    """Resize keeping the aspect ratio, centered on a ``pad``-filled canvas of ``input_hw``.

    Returns the canvas and ``(left, top, scale_x, scale_y)`` for ``_unletterbox``.
    """
    h, w = input_hw
    orig_h, orig_w = frame_bgr.shape[:2]
    scale = min(w / orig_w, h / orig_h)
    rw, rh = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
    left, top = (w - rw) // 2, (h - rh) // 2
    canvas = np.full((h, w, 3), pad, dtype=np.uint8)
    canvas[top : top + rh, left : left + rw] = cv2.resize(frame_bgr, (rw, rh))
    return canvas, (left, top, rw / orig_w, rh / orig_h)


def _unletterbox(xyxy: np.ndarray, placement: tuple[int, int, float, float]) -> np.ndarray:
    """Map xyxy boxes on a ``_letterbox`` canvas back to the original frame's pixels."""
    left, top, sx, sy = placement
    return (np.asarray(xyxy, dtype=np.float64) - [left, top, left, top]) / [sx, sy, sx, sy]


def postprocess_detr_outputs(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_hw: tuple[int, int],
    id2label: dict[int, str],
    threshold: float = 0.45,
    allowed_classes: set[str] | None = None,
    max_detections: int = 300,
    nms_iou: float = 0.65,
    softmax: bool = False,
) -> list[Detection]:
    """Convert DETR/RT-DETR/D-FINE/RF-DETR raw logits + normalized cxcywh boxes into Detections."""
    # Squeeze batch dimension [1, N, C] -> [N, C]
    if logits.ndim == 3:
        logits = logits[0]
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes[0]

    if softmax:
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = (exp / exp.sum(axis=-1, keepdims=True))[:, :-1]  # trailing no-object class
    else:
        probs = sigmoid(logits)
    class_ids = np.argmax(probs, axis=-1)
    scores = probs[np.arange(probs.shape[0]), class_ids]

    keep = (scores >= threshold) & np.isfinite(pred_boxes).all(axis=-1)
    if not np.any(keep):
        return []

    kept_scores = scores[keep]
    kept_classes = class_ids[keep]
    kept_boxes = pred_boxes[keep]

    # Stable sort so ties keep query order (web/detector.js mirrors this for parity).
    order = np.argsort(-kept_scores, kind="stable")
    orig_h, orig_w = orig_hw

    detections: list[Detection] = []
    for idx in order:
        cid = int(kept_classes[idx])
        cname = id2label.get(cid, str(cid))
        if (
            allowed_classes is not None
            and cname.lower() not in allowed_classes
            and _class_key(cname) not in allowed_classes
        ):
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

    return _nms_class_aware(detections, iou_threshold=nms_iou)[:max_detections]


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
        spec = MODEL_REGISTRY.get(model)
        self.revision = spec.revision if spec else None
        self.threshold = float(threshold)
        self.allowed_classes = _allowed_classes(classes)

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
        if spec and spec.normalize:
            self.do_normalize = True

        # CoreML EP on int8 quantized DETR graphs can occasionally fail or degrade; prefer CPU for int8.
        effective_device = "cpu" if (device == "auto" and precision == "int8") else device
        cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
        optimization = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT's extended CPU fusions for this signed INT8 graph abort the process on arm64 macOS.
        if (
            cfg.get("model_type") == "d_fine"
            and (precision == "int8" or "int8" in onnx_path.name)
            and platform.system() == "Darwin"
            and platform.machine() == "arm64"
        ):
            optimization = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session, self.active_provider = create_onnx_session(
            onnx_path,
            device=effective_device,
            optimization=optimization,
            coreml_format=spec.coreml_format if spec else "MLProgram",
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        inputs = {i.name for i in self.session.get_inputs()}
        self.letterbox = bool(spec and spec.letterbox)
        self.deployed = inputs == {"images", "orig_target_sizes"} and set(self.output_names) == {
            "labels",
            "boxes",
            "scores",
        }
        if self.deployed:
            self.input_name = "images"
        elif len(inputs) == 1 and {"logits", "pred_boxes"}.issubset(self.output_names):
            self.raw_outputs = ["logits", "pred_boxes"]
        elif len(inputs) == 1 and len(self.output_names) == 2:
            # Legacy two-output exports can use other names or reverse the output order.
            first, second = self.session.get_outputs()
            if (
                len(first.shape) != 3
                or len(second.shape) != 3
                or 4 not in (first.shape[-1], second.shape[-1])
            ):
                raise ValueError(
                    "Expected class logits and cxcywh boxes with shape (1, queries, 4)."
                )
            self.raw_outputs = (
                [first.name, second.name] if second.shape[-1] == 4 else [second.name, first.name]
            )
        else:
            raise ValueError(
                "Expected an ONNX detector with logits/pred_boxes or images + orig_target_sizes and labels/boxes/scores. For other formats supply detector= to TrackflowPipeline."
            )
        image_input = next(i for i in self.session.get_inputs() if i.name == self.input_name)
        if len(image_input.shape) != 4 or image_input.type not in (
            "tensor(float)",
            "tensor(float16)",
        ):
            raise ValueError("Expected a float32 or float16 NCHW image input.")
        self.input_dtype = np.float16 if image_input.type == "tensor(float16)" else np.float32
        h, w = image_input.shape[-2:]
        if isinstance(h, int) and isinstance(w, int):
            self.input_hw = (h, w)
        # The converted YOLO26 graphs use the YOLOS config type but have no background logit.
        logits_out = next((o for o in self.session.get_outputs() if o.name == "logits"), None)
        self.softmax = cfg.get("model_type") in ("detr", "yolos", "table-transformer") and (
            logits_out is not None and logits_out.shape[-1] == max(self.id2label, default=-1) + 2
        )

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
        if isinstance(size, int):
            size = {"height": size, "width": size}
        h = int(size.get("height", size.get("shortest_edge", 640)))
        w = int(size.get("width", size.get("shortest_edge", 640)))
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
        cutoff = self.threshold if threshold is None else threshold
        if self.deployed:
            # Upstream DEIMv2 / RT-DETRv4 deployment graphs take a black-letterboxed RGB
            # image and emit xyxy in the coordinate system supplied via orig_target_sizes.
            h, w = self.input_hw
            padded, placement = _letterbox(frame_bgr, self.input_hw, 0)
            labels, boxes, scores = self.session.run(
                ["labels", "boxes", "scores"],
                {
                    "images": self.preprocess(padded),
                    "orig_target_sizes": np.array([[w, h]], dtype=np.int64),
                },
            )
            return _pixel_detections(
                labels[0],
                _unletterbox(boxes[0], placement),
                scores[0],
                (orig_h, orig_w),
                self.id2label,
                self.allowed_classes,
                cutoff,
            )
        image, placement = frame_bgr, None
        if self.letterbox:
            image, placement = _letterbox(frame_bgr, self.input_hw, 114)
        tensor = self.preprocess(image)
        logits, pred_boxes = self.session.run(self.raw_outputs, {self.input_name: tensor})

        detections = postprocess_detr_outputs(
            logits=np.asarray(logits, dtype=np.float32),
            pred_boxes=np.asarray(pred_boxes, dtype=np.float32),
            orig_hw=image.shape[:2],
            id2label=self.id2label,
            threshold=cutoff,
            allowed_classes=self.allowed_classes,
            softmax=self.softmax,
        )
        if placement is None:
            return detections
        restored = []
        for det in detections:
            x1, y1, x2, y2 = _unletterbox(det.bbox, placement)
            x1, x2 = np.clip([x1, x2], 0.0, orig_w)
            y1, y2 = np.clip([y1, y2], 0.0, orig_h)
            if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
                det.bbox = (float(x1), float(y1), float(x2), float(y2))
                restored.append(det)
        return restored


def _pixel_detections(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    orig_hw: tuple[int, int],
    id2label: dict[int, str],
    allowed: set[str] | None,
    threshold: float,
) -> list[Detection]:
    h, w = orig_hw
    detections = []
    for i in np.flatnonzero(scores >= threshold):
        cid = int(labels[i])
        name = id2label.get(cid, str(cid))
        if allowed is not None and _class_key(name) not in allowed:
            continue
        x1, y1, x2, y2 = np.clip(boxes[i], 0, [w, h, w, h]).tolist()
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            detections.append(Detection((x1, y1, x2, y2), float(scores[i]), cid, name))
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return _nms_class_aware(detections, 0.65)


class TransformersDetector:
    """Use a native Transformers object-detection checkpoint and its own image processor."""

    def __init__(
        self,
        repo_id: str,
        *,
        revision: str | None,
        device: str,
        precision: Precision,
        threshold: float,
        classes: Sequence[str] | None,
        cache_dir: str | Path | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise ImportError(
                "Install 'trackflow-python[transformers]' to load native Transformers checkpoints."
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("Transformers supports --device auto, cpu, cuda, or mps.")
        if precision not in ("fp32", "fp16"):
            raise ValueError("Transformers detectors support fp32 or fp16.")
        self.model_name = repo_id
        self.device = device
        self.precision = precision
        self.dtype = torch.float32 if precision == "fp32" else torch.float16
        kwargs = {"revision": revision, "cache_dir": cache_dir, "trust_remote_code": False}
        self.processor = AutoImageProcessor.from_pretrained(repo_id, **kwargs)
        self.model = (
            AutoModelForObjectDetection.from_pretrained(repo_id, use_safetensors=True, **kwargs)
            .to(device=device, dtype=self.dtype)
            .eval()
        )
        self.revision = revision or getattr(self.model.config, "_commit_hash", None)
        self.active_provider = f"torch:{device}"
        self.threshold = threshold
        self.allowed_classes = _allowed_classes(classes)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        import torch

        hw = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device, self.dtype)
            output = self.model(**inputs)
            result = self.processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=[hw]
            )[0]
        return _pixel_detections(
            result["labels"].cpu().numpy(),
            result["boxes"].float().cpu().numpy(),
            result["scores"].float().cpu().numpy(),
            hw,
            self.model.config.id2label,
            self.allowed_classes,
            self.threshold,
        )


def create_detector(
    model: str | Path = "rtdetr-v2-r18",
    *,
    precision: Precision = "fp32",
    device: str = "auto",
    threshold: float = 0.45,
    classes: Sequence[str] | None = ("person",),
    cache_dir: str | Path | None = None,
) -> Detector:
    """Load a preset, compatible ONNX graph/repository, or hf:owner/repo Transformers checkpoint."""
    model = str(model)
    spec = MODEL_REGISTRY.get(model)
    if model.startswith("hf:") or (spec and spec.backend == "transformers"):
        return TransformersDetector(
            spec.repo_id if spec else model.removeprefix("hf:"),
            revision=spec.revision if spec else None,
            device=device,
            precision=precision,
            threshold=threshold,
            classes=classes,
            cache_dir=cache_dir,
        )
    return OnnxDetrDetector(
        model,
        precision=precision,
        device=device,
        threshold=threshold,
        classes=classes,
        cache_dir=cache_dir,
    )
