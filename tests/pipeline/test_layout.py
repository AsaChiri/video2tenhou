# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Frame transforms, region crops and per-recording calibration."""

import itertools
import json

import cv2
import numpy as np
import pytest

from tests.paths import DATA
from video2tenhou.layout import (
    Calibration,
    Rect,
    apply,
    box_to_quad,
    contact_sheet,
    quad_to_box,
)

LABELS = DATA / "layout_boxes"


@pytest.mark.parametrize("channels", [(), (3,)])
@pytest.mark.parametrize(
    "bounds",
    [
        (1, 1, 3, 2),
        (-2, 1, 5, 3),
        (2, -2, 3, 5),
        (4, 1, 5, 3),
        (1, 3, 3, 5),
        (-8, -8, 3, 3),
        (9, 9, 3, 3),
    ],
)
def test_rectangle_crop_matches_frame_coordinates_at_image_edges(
    channels: "tuple[int, ...]", bounds: tuple[int, int, int, int]
) -> None:
    """Verify rectangle crop matches frame coordinates at image edges."""
    frame = np.arange(np.prod((5, 7, *channels)), dtype=np.uint8).reshape(
        5, 7, *channels
    )
    rect = Rect(*bounds)
    expected = cv2.warpAffine(
        frame,
        np.array([[1, 0, -rect.x], [0, 1, -rect.y]], dtype=np.float32),
        (rect.w, rect.h),
        flags=cv2.INTER_NEAREST,
    )
    assert np.array_equal(rect.crop(frame), expected)


@pytest.mark.parametrize("size", [(0, 2), (2, 0), (-1, 2)])
def test_rectangle_crop_rejects_nonpositive_dimensions(size: tuple[int, int]) -> None:
    """Verify rectangle crop rejects nonpositive dimensions."""
    with pytest.raises(ValueError, match="Crop dimensions must be positive"):
        Rect(0, 0, *size).crop(np.zeros((5, 7, 3), np.uint8))


@pytest.fixture(scope="module")
def cal() -> "Calibration":
    """Load the packaged reference calibration."""
    return Calibration.load("pml")


def test_transforms_invert(cal: "Calibration") -> None:
    """Verify transforms invert."""
    pts = np.array([[500.0, 400.0], [960.0, 540.0], [1300.0, 700.0]])
    for name in cal.regions():
        transform, (w, h) = cal.transform(name)
        assert w > 0
        assert h > 0
        back = apply(np.linalg.inv(transform), apply(transform, pts))
        assert np.allclose(back, pts, atol=1e-6)


def test_region_shapes(cal: "Calibration") -> None:
    """Verify region shapes."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    for name in cal.regions():
        img, _transform = cal.region(frame, name)
        _, (w, h) = cal.transform(name)
        assert img.shape[:2] == (h, w), name


def test_contact_sheet_renders_table_only_layout(cal: "Calibration") -> None:
    """Verify contact sheet renders table only layout."""
    assert "overlay" not in cal.data
    sheet = contact_sheet(np.zeros((1080, 1920, 3), np.uint8), cal)
    assert sheet.shape == (1620, 1920, 3)


def _labels(kind: "str") -> "list[dict]":
    out = []
    for p in sorted(LABELS.glob(f"{kind}_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["boxes"]:
            out.append(d)
    return out


@pytest.mark.parametrize("kind", ["pond", "hand", "meld"])
def test_labels_fall_inside_their_region(cal: "Calibration", kind: str) -> None:
    """Verify labels fall inside their region."""
    labels = _labels(kind)
    assert labels, "converted labels missing"
    outside = 0
    total = 0
    for d in labels:
        transform, (w, h) = cal.transform(f"{kind}:{d['corner']}")
        for b in d["boxes"]:
            x0, y0, x1, y1 = quad_to_box(transform, b["quad"])
            total += 1
            if x0 < -2 or y0 < -2 or x1 > w + 2 or y1 > h + 2:
                outside += 1
    assert outside / total < 0.01, (
        f"{outside}/{total} {kind} boxes fall outside the region"
    )


def test_pond_rows_grow_toward_player_and_cols_to_the_right(cal: "Calibration") -> None:
    """Verify pond rows grow toward player and cols to the right.

    In the upright pond region (owner at the bottom) row 0 is nearest the centre
    (smallest y) and columns increase with x (the owner's left is the region's left).
    """
    bad_rows = bad_cols = checked = 0
    for d in _labels("pond"):
        transform, _ = cal.transform(f"pond:{d['corner']}")
        rows: dict[int, list[tuple[int, float, float]]] = {}
        for b in d["boxes"]:
            if "row" not in b:
                continue
            x0, y0, x1, y1 = quad_to_box(transform, b["quad"])
            rows.setdefault(b["row"], []).append(
                (b["col"], (x0 + x1) / 2, (y0 + y1) / 2)
            )
        ys = [np.mean([c[2] for c in cells]) for r, cells in sorted(rows.items())]
        for a, b2 in itertools.pairwise(ys):
            checked += 1
            bad_rows += b2 <= a
        for cells in rows.values():
            cells.sort()
            for (_c0, x0, _), (_c1, x1, _) in itertools.pairwise(cells):
                checked += 1
                bad_cols += x1 <= x0
    assert checked > 100
    assert bad_rows == 0
    assert bad_cols == 0


def test_box_quad_roundtrip(cal: "Calibration") -> None:
    """Verify box quad roundtrip."""
    transform, _ = cal.transform("pond:TL")
    box = (100.0, 50.0, 180.0, 160.0)
    q = box_to_quad(transform, box)
    assert np.allclose(quad_to_box(transform, q), box, atol=0.2)
