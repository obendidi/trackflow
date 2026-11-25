"""Opt-in CPU checks of the published checkpoints. See README for the invocation."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from trackflow import MODEL_REGISTRY, create_detector
from trackflow.tracker import compute_iou_matrix

pytestmark = pytest.mark.skipif(
    os.environ.get("TRACKFLOW_TEST_WEIGHTS") != "1",
    reason="Set TRACKFLOW_TEST_WEIGHTS=1 to download and test real model weights",
)


@pytest.fixture(scope="module")
def crowd_frame(tmp_path_factory):
    path = Path(tmp_path_factory.mktemp("opencv")) / "vtest.avi"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/4.12.0/samples/data/vtest.avi", path
    )
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf"
    )
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        assert ok
        return frame
    finally:
        cap.release()


@pytest.mark.parametrize(
    ("model", "precision"),
    [
        *((name, "fp32") for name in MODEL_REGISTRY),
        ("dfine-n", "int8"),  # previously aborted in native code on Apple Silicon
    ],
)
def test_checkpoint_detects_people_in_source_coordinates(crowd_frame, model, precision):
    detector = create_detector(model, device="cpu", precision=precision, threshold=0.2)
    people = detector.detect(crowd_frame)
    assert people and all(p.class_name == "person" for p in people)
    expected = np.array([[640, 238, 684, 322], [499, 156, 531, 231], [251, 221, 284, 309]])
    iou = compute_iou_matrix(expected, np.array([p.bbox for p in people]))
    assert sum(iou.max(axis=1) > 0.4) >= 2


def test_embeddinggemma_ranks_the_matching_crop_first():
    pytest.importorskip("sentence_transformers")
    from trackflow.reid import create_vlm_embedder

    vlm = create_vlm_embedder("embeddinggemma", device="cpu")
    red = np.zeros((160, 80, 3), np.uint8)
    red[:] = (30, 30, 220)  # BGR
    blue = red[:, :, ::-1].copy()
    query = vlm.encode_text("a red picture")
    red_emb, blue_emb = vlm.encode_image(red), vlm.encode_image(blue)
    assert red_emb.shape == (768,)
    assert abs(float(np.linalg.norm(red_emb)) - 1) < 1e-3
    assert float(red_emb @ query) > float(blue_emb @ query)
