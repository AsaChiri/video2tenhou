
from tests.paths import DATA
import json
from pathlib import Path

import numpy as np
import pytest

from video2tenhou.layout import Calibration, apply, box_to_quad, quad_to_box

LABELS = DATA / "layout_boxes"


@pytest.fixture(scope="module")
def cal():
    return Calibration.load("pml")


def test_transforms_invert(cal):
    pts = np.array([[500.0, 400.0], [960.0, 540.0], [1300.0, 700.0]])
    for name in cal.regions():
        M, (w, h) = cal.transform(name)
        assert w > 0 and h > 0
        back = apply(np.linalg.inv(M), apply(M, pts))
        assert np.allclose(back, pts, atol=1e-6)


def test_region_shapes(cal):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    for name in cal.regions():
        img, M = cal.region(frame, name)
        _, (w, h) = cal.transform(name)
        assert img.shape[:2] == (h, w), name


def _labels(kind):
    out = []
    for p in sorted(LABELS.glob(f"{kind}_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        if d["boxes"]:
            out.append(d)
    return out


@pytest.mark.parametrize("kind", ["pond", "hand", "meld"])
def test_labels_fall_inside_their_region(cal, kind):
    labels = _labels(kind)
    assert labels, "converted labels missing"
    outside = 0
    total = 0
    for d in labels:
        M, (w, h) = cal.transform(f"{kind}:{d['corner']}")
        for b in d["boxes"]:
            x0, y0, x1, y1 = quad_to_box(M, b["quad"])
            total += 1
            if x0 < -2 or y0 < -2 or x1 > w + 2 or y1 > h + 2:
                outside += 1
    assert outside / total < 0.01, f"{outside}/{total} {kind} boxes fall outside the region"


def test_pond_rows_grow_toward_player_and_cols_to_the_right(cal):
    """In the upright pond region (owner at the bottom) row 0 is nearest the centre (smallest y)
    and columns increase with x (the owner's left is the region's left)."""
    bad_rows = bad_cols = checked = 0
    for d in _labels("pond"):
        M, _ = cal.transform(f"pond:{d['corner']}")
        rows: dict[int, list[tuple[int, float, float]]] = {}
        for b in d["boxes"]:
            if "row" not in b:
                continue
            x0, y0, x1, y1 = quad_to_box(M, b["quad"])
            rows.setdefault(b["row"], []).append((b["col"], (x0 + x1) / 2, (y0 + y1) / 2))
        ys = [np.mean([c[2] for c in cells]) for r, cells in sorted(rows.items())]
        for a, b2 in zip(ys, ys[1:]):
            checked += 1
            bad_rows += b2 <= a
        for cells in rows.values():
            cells.sort()
            for (c0, x0, _), (c1, x1, _) in zip(cells, cells[1:]):
                checked += 1
                bad_cols += x1 <= x0
    assert checked > 100
    assert bad_rows == 0 and bad_cols == 0


def test_box_quad_roundtrip(cal):
    M, _ = cal.transform("pond:TL")
    box = (100.0, 50.0, 180.0, 160.0)
    q = box_to_quad(M, box)
    assert np.allclose(quad_to_box(M, q), box, atol=0.2)
