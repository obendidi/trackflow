"""Tests for privacy-first anonymization and 4-point floorplan homography."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trackflow.homography import DwellZone, FloorplanHomography
from trackflow.privacy import (
    PrivacyAnonymizer,
    _cluster_head_boxes,
    _dedup_boxes_preserving_coverage,
)
from trackflow.types import Detection, Track


def test_privacy_modes_modify_or_remove_person_pixels() -> None:
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    track = Track(
        track_id=1, bbox=(40, 30, 120, 170), confidence=0.9, class_id=0, class_name="person"
    )

    for mode in ("blur", "pixelate", "face-blur"):
        anon = PrivacyAnonymizer(mode=mode)  # type: ignore[arg-type]
        out = anon.apply(frame, [track])
        assert out.shape == frame.shape
        # Inside head region must differ from raw frame; outside box must stay identical
        assert not np.array_equal(out[35:65, 50:110], frame[35:65, 50:110])
        assert np.array_equal(out[0:10, 0:10], frame[0:10, 0:10])

    # Skeleton mode removes ALL original background pixels (zero PII)
    skel = PrivacyAnonymizer(mode="skeleton").apply(frame, [track])
    assert not np.array_equal(skel[0:10, 0:10], frame[0:10, 0:10])


def test_privacy_blurs_unconfirmed_detections_too() -> None:
    """People who are detected but not (yet) confirmed tracks must still be anonymized."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    det = Detection(bbox=(40, 30, 120, 170), confidence=0.2, class_id=0, class_name="person")
    for mode in ("blur", "pixelate", "face-blur"):
        out = PrivacyAnonymizer(mode=mode).apply(frame, [], detections=[det])  # type: ignore[arg-type]
        assert not np.array_equal(out[35:65, 50:110], frame[35:65, 50:110]), mode
        assert np.array_equal(out[0:10, 0:10], frame[0:10, 0:10]), mode


def test_privacy_dedup_preserves_full_pixel_coverage() -> None:
    """Box deduplication must cut redundant passes while keeping 100% pixel mask coverage."""
    import cv2

    rng = np.random.default_rng(99)
    h, w = 260, 320
    frame = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    # Person 1: track + near-identical detection + strictly contained low-conf box + coasting shift
    p1_track = Track(
        track_id=1, bbox=(30.0, 25.0, 110.0, 210.0), confidence=0.92, class_id=0, class_name="person"
    )
    p1_det = Detection(
        bbox=(32.0, 24.0, 109.0, 212.0), confidence=0.90, class_id=0, class_name="person"
    )
    p1_inner = Detection(
        bbox=(40.0, 45.0, 98.0, 190.0), confidence=0.22, class_id=0, class_name="person"
    )
    p1_coast = Track(
        track_id=11,
        bbox=(31.0, 26.0, 111.0, 209.0),
        confidence=0.60,
        class_id=0,
        class_name="person",
        coasting=True,
    )
    # Person 2: separate person on the right with a track + slightly shifted detection
    p2_track = Track(
        track_id=2, bbox=(200.0, 40.0, 285.0, 230.0), confidence=0.88, class_id=0, class_name="person"
    )
    p2_det = Detection(
        bbox=(202.0, 39.0, 284.0, 231.0), confidence=0.86, class_id=0, class_name="person"
    )

    tracks = [p1_track, p2_track]
    coasting = [p1_coast]
    dets = [p1_det, p1_inner, p2_det]

    # Build expected pixel-union mask from all 6 expanded boxes (without any deduplication).
    margin = 0.08
    raw_rects: list[tuple[int, int, int, int]] = []
    for box in [t.bbox for t in tracks] + [t.bbox for t in coasting] + [d.bbox for d in dets]:
        bx1, by1, bx2, by2 = box
        mx, my = (bx2 - bx1) * margin, (by2 - by1) * margin
        raw_rects.append(
            (
                max(0, min(w - 1, int(round(bx1 - mx)))),
                max(0, min(h - 1, int(round(by1 - my)))),
                max(1, min(w, int(round(bx2 + mx)))),
                max(1, min(h, int(round(by2 + my)))),
            )
        )

    deduped = _dedup_boxes_preserving_coverage(raw_rects)
    assert len(deduped) == 2, f"Expected 6 overlapping boxes to collapse to 2, got {len(deduped)}"

    expected_union = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in raw_rects:
        expected_union[y1:y2, x1:x2] = True
    actual_union = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in deduped:
        actual_union[y1:y2, x1:x2] = True
    # Every single pixel of every original box must be covered by the deduplicated rectangles.
    assert not np.any(expected_union & ~actual_union)

    # Head-box clustering collapses overlapping head boxes per person while keeping every unique ellipse
    # (p1_inner's lower head box extends below p1_track's top-30% head rect, so it forms its own cluster).
    head_rects = [
        (x1, y1, x2, min(h, y1 + max(8, int((y2 - y1) * 0.30)))) for x1, y1, x2, y2 in raw_rects
    ]
    clusters = _cluster_head_boxes(head_rects)
    assert len(clusters) == 3
    expected_ellipse = np.zeros((h, w), dtype=np.uint8)
    for rx1, ry1, rx2, ry2 in head_rects:
        cv2.ellipse(
            expected_ellipse,
            ((rx1 + rx2) // 2, (ry1 + ry2) // 2),
            (max(2, int((rx2 - rx1) * 0.42)), max(2, int((ry2 - ry1) * 0.48))),
            0,
            0,
            360,
            255,
            -1,
        )
    actual_ellipse = np.zeros((h, w), dtype=np.uint8)
    for (ux1, uy1, ux2, uy2), members in clusters:
        sub = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.uint8)
        for rx1, ry1, rx2, ry2 in members:
            cv2.ellipse(
                sub,
                ((rx1 + rx2) // 2 - ux1, (ry1 + ry2) // 2 - uy1),
                (max(2, int((rx2 - rx1) * 0.42)), max(2, int((ry2 - ry1) * 0.48))),
                0,
                0,
                360,
                255,
                -1,
            )
        actual_ellipse[uy1:uy2, ux1:ux2] |= sub
    assert np.array_equal(expected_ellipse, actual_ellipse)

    # End-to-end apply check: every person region is anonymized and clean background is untouched.
    for mode in ("blur", "pixelate", "face-blur"):
        out = PrivacyAnonymizer(mode=mode, margin=margin).apply(  # type: ignore[arg-type]
            frame, tracks, detections=dets, coasting=coasting
        )
        assert np.array_equal(out[0:10, 0:10], frame[0:10, 0:10]), mode
        assert np.array_equal(out[100:150, 140:170], frame[100:150, 140:170]), mode
        # Both people's head regions must be heavily altered
        assert not np.array_equal(out[35:65, 45:95], frame[35:65, 45:95]), mode
        assert not np.array_equal(out[50:80, 215:270], frame[50:80, 215:270]), mode


def test_floorplan_homography_projection_dwell_and_render(tmp_path: Path) -> None:
    zone = DwellZone(name="entrance", polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    hom = FloorplanHomography(
        image_points=[(100, 100), (500, 100), (500, 400), (100, 400)],
        world_points=[(0, 0), (10, 0), (10, 8), (0, 8)],
        width_m=10.0,
        height_m=8.0,
        zones=[zone],
    )

    # Exact center of image quad maps to exact center of world rectangle (5.0, 4.0)
    wx, wy = hom.project((300.0, 250.0))
    assert np.isclose(wx, 5.0, atol=1e-2)
    assert np.isclose(wy, 4.0, atol=1e-2)

    # Observe inside entrance zone twice
    w1, z1 = hom.observe(track_id=1, foot_xy=(150.0, 150.0), dt_s=0.5)
    w2, z2 = hom.observe(track_id=1, foot_xy=(200.0, 180.0), dt_s=0.5)
    assert z1 == "entrance" and z2 == "entrance"
    assert np.isclose(hom.zone_dwell_totals_s["entrance"], 1.0)
    assert w1 != w2

    # Save/reload JSON and render floorplan PNG
    cfg_path = tmp_path / "homography.json"
    hom.save_json(cfg_path)
    reloaded = FloorplanHomography.from_json(cfg_path)
    assert np.allclose(reloaded.project((300.0, 250.0)), (5.0, 4.0), atol=1e-2)

    png_path = hom.render_floorplan(tmp_path / "floorplan.png")
    assert png_path.is_file() and png_path.stat().st_size > 1000
