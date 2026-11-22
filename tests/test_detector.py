"""Tests for DETR / D-FINE / RT-DETRv2 / RF-DETR ONNX output post-processing."""

from __future__ import annotations

import numpy as np

from trackflow.detector import postprocess_detr_outputs


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
