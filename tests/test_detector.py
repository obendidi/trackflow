"""Tests for DETR / D-FINE / RT-DETRv2 / RF-DETR ONNX output post-processing."""

from __future__ import annotations

import numpy as np
import pytest

from trackflow.detector import postprocess_detr_outputs


def test_deployed_graph_reverses_padding_and_reports_native_scores():
    from trackflow import OnnxDetrDetector

    class Session:
        def run(self, names, inputs):
            assert names == ["labels", "boxes", "scores"]
            assert inputs["images"].shape == (1, 3, 320, 320)
            assert inputs["orig_target_sizes"].tolist() == [[320, 320]]
            assert np.all(inputs["images"][0, :, :80] == 0)
            assert np.all(inputs["images"][0, :, 80:240] == 1)
            return (
                np.array([[1, 0]]),
                np.array([[[40, 100, 200, 200], [0, 0, 30, 40]]]),
                np.array([[0.76, 0.9]]),
            )

    detector = OnnxDetrDetector.__new__(OnnxDetrDetector)
    detector.deployed = detector.do_rescale = True
    detector.do_normalize = False
    detector.input_hw = (320, 320)
    detector.id2label = {0: "person", 1: "bicycle"}
    detector.allowed_classes = None
    detector.threshold = 0.2
    detector.input_dtype = np.float32
    detector.session = Session()
    (det,) = detector.detect(np.full((100, 200, 3), 255, np.uint8))
    assert det.bbox == (25.0, 12.5, 125.0, 75.0)
    assert det.class_name == "bicycle" and det.confidence == 0.76


def test_letterboxed_preset_pads_with_gray_and_maps_boxes_back():
    from trackflow import OnnxDetrDetector

    class Session:
        def run(self, names, inputs):
            image = inputs["pixel_values"][0]
            assert image.shape == (3, 320, 320)
            # A 200x100 frame fills a 320x160 band; the rest is YOLO's gray (114).
            assert np.allclose(image[:, :80], 114 / 255) and np.allclose(image[:, 240:], 114 / 255)
            assert np.all(image[:, 80:240] == 1)
            # One person at (40, 100, 200, 200) on the 320x320 canvas, as normalized cxcywh.
            logits = np.full((1, 1, 2), -10.0, np.float32)
            logits[0, 0, 0] = 10.0
            return logits, np.array([[[0.375, 0.46875, 0.5, 0.3125]]], np.float32)

    detector = OnnxDetrDetector.__new__(OnnxDetrDetector)
    detector.deployed, detector.letterbox, detector.softmax = False, True, False
    detector.do_rescale, detector.do_normalize = True, False
    detector.input_hw, detector.input_dtype = (320, 320), np.float32
    detector.input_name, detector.raw_outputs = "pixel_values", ["logits", "pred_boxes"]
    detector.id2label, detector.allowed_classes, detector.threshold = {0: "person"}, None, 0.2
    detector.session = Session()
    (det,) = detector.detect(np.full((100, 200, 3), 255, np.uint8))
    assert det.bbox == pytest.approx((25.0, 12.5, 125.0, 75.0))
    assert det.class_name == "person"


def test_only_yolo_presets_are_letterboxed():
    from trackflow.models import MODEL_REGISTRY

    assert {n for n, s in MODEL_REGISTRY.items() if s.letterbox} == {
        "yolo26n",
        "yolo26s",
        "yolo26m",
        "yolo26l",
        "yolo26x",
    }


def test_coreml_sessions_run_mlprogram_on_the_gpu(monkeypatch, tmp_path):
    from trackflow import detector as detector_module

    seen = {}

    class Session:
        def __init__(self, path, sess_options=None, providers=None, provider_options=None):
            seen.update(providers=providers, provider_options=provider_options)

        def get_providers(self):
            return seen["providers"]

    monkeypatch.setattr(
        detector_module.ort,
        "get_available_providers",
        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(detector_module.ort, "InferenceSession", Session)
    _, active = detector_module.create_onnx_session(tmp_path / "model.onnx", device="coreml")
    assert active == "CoreMLExecutionProvider"
    assert seen["provider_options"] == [
        {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"},
        {},
    ]
    detector_module.create_onnx_session(
        tmp_path / "model.onnx", device="coreml", coreml_format="NeuralNetwork"
    )
    assert seen["provider_options"][0]["ModelFormat"] == "NeuralNetwork"
    detector_module.create_onnx_session(tmp_path / "model.onnx", device="cpu")
    assert seen["provider_options"] == [{}]


def test_precision_mistakes_and_missing_extra_are_actionable(monkeypatch):
    import sys

    from trackflow import create_detector

    with pytest.raises(ValueError, match="fp32"):
        create_detector("deimv2-s", precision="int8")
    with pytest.raises(ValueError, match="Unknown precision"):
        create_detector("rtdetr-v2-r18", precision="int9")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"trackflow-python\[transformers\]"):
        create_detector("rfdetr-large")


def test_softmax_background_and_nonfinite_boxes():
    logits = np.array([[[2.0, 3.0], [4.0, 0.0], [-2.0, 4.0]]])
    boxes = np.array([[[0.5, 0.5, 0.1, 0.1], [np.nan, 0.2, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]]])
    (det,) = postprocess_detr_outputs(
        logits, boxes, (100, 100), {0: "person"}, threshold=0.2, softmax=True
    )
    assert det.confidence == pytest.approx(1 / (1 + np.e))


def test_postprocess_detr_0_indexed_and_class_filter() -> None:
    # 2 queries, 3 classes: query 0 -> person (id 0), query 1 -> car (id 2)
    logits = np.array(
        [
            [
                [3.0, -4.0, -4.0],  # sigmoid(3.0) ~= 0.953 -> person
                [-4.0, -4.0, 2.5],  # sigmoid(2.5) ~= 0.924 -> car
            ]
        ],
        dtype=np.float32,
    )
    boxes = np.array(
        [
            [
                [0.5, 0.5, 0.2, 0.4],  # center (500, 250), size (200, 200) on 1000x500
                [0.2, 0.3, 0.1, 0.1],
            ]
        ],
        dtype=np.float32,
    )
    id2label = {0: "person", 1: "bicycle", 2: "car"}

    people = postprocess_detr_outputs(
        logits,
        boxes,
        orig_hw=(500, 1000),
        id2label=id2label,
        threshold=0.5,
        allowed_classes={"person"},
    )
    assert len(people) == 1
    assert people[0].class_name == "person"
    assert np.allclose(people[0].bbox, (400.0, 150.0, 600.0, 350.0), atol=1e-3)

    all_objs = postprocess_detr_outputs(
        logits, boxes, orig_hw=(500, 1000), id2label=id2label, threshold=0.5, allowed_classes=None
    )
    assert len(all_objs) == 2
    assert {d.class_name for d in all_objs} == {"person", "car"}


def test_postprocess_rfdetr_1_indexed_labels() -> None:
    logits = np.array([[[-5.0, 4.0, -5.0]]], dtype=np.float32)  # index 1 -> person in RF-DETR
    boxes = np.array([[[0.25, 0.5, 0.2, 0.4]]], dtype=np.float32)
    rfdetr_labels = {1: "person", 2: "bicycle"}

    dets = postprocess_detr_outputs(
        logits,
        boxes,
        orig_hw=(400, 800),
        id2label=rfdetr_labels,
        threshold=0.5,
        allowed_classes={"person"},
    )
    assert len(dets) == 1
    assert dets[0].class_id == 1
    assert dets[0].class_name == "person"


def test_postprocess_deduplicates_overlapping_queries() -> None:
    logits = np.array([[[4.0, -4.0], [3.0, -4.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4], [0.502, 0.501, 0.2, 0.4]]], dtype=np.float32)
    dets = postprocess_detr_outputs(
        logits,
        boxes,
        orig_hw=(500, 1000),
        id2label={0: "person", 1: "car"},
        threshold=0.5,
        allowed_classes={"person"},
        nms_iou=0.65,
    )
    assert len(dets) == 1
