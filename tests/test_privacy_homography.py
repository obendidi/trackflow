"""Tests for privacy-first anonymization and 4-point floorplan homography."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trackflow.homography import DwellZone, FloorplanHomography
from trackflow.privacy import PrivacyAnonymizer
from trackflow.types import Track


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
