
from tests.paths import DATA
from pathlib import Path

import cv2
import pytest

from video2tenhou.overlay import read_overlay



@pytest.mark.parametrize(
    "name, kyoku, honba, riichi, winds, scores",
    [
        ("pml_e4_h0.jpg", 3, 0, 0, "SEWN", (24900, 20300, 29400, 25400)),
        ("pml_s4_h0_negative.jpg", 7, 0, 0, "SEWN", (-11500, 47000, 47000, 17500)),
    ],
)
def test_reference_frames(name, kyoku, honba, riichi, winds, scores):
    image = cv2.imread(str(DATA / name))
    st = read_overlay(image, names=False)
    assert read_overlay(image, names=False, complete_only=True).to_dict() == st.to_dict()
    assert st.kyoku_index == kyoku
    assert st.honba == honba and st.riichi_sticks == riichi
    assert "".join(st.corners[c].wind for c in ("TL", "TR", "BL", "BR")) == winds
    assert tuple(st.corners[c].score for c in ("TL", "TR", "BL", "BR")) == scores
    assert st.seat_order() == ["TL", "BL", "BR", "TR"]  # seat 0 = corner that was East in E1


def test_batched_digit_correlations_preserve_reference_decisions():
    """Compare the optimized matcher with scalar NCC on all templates and noisy edges."""
    import numpy as np
    from video2tenhou import overlay as o

    matcher = o.DigitMatcher()
    random = np.random.default_rng(24)
    masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(o.TEMPLATE_DIR.glob("digit_*.png"))]
    for original in masks:
        for mask in (original, cv2.GaussianBlur(original, (3, 3), .6),
                     np.where(random.random(original.shape) < .01, 255 - original, original).astype(np.uint8)):
            sig = o.digit_signature(mask)
            label, confidence = None, -1.0
            for digit, templates in matcher.templates.items():
                score = max(o._ncc(sig, template) for template in templates)
                if score > confidence:
                    label, confidence = digit, score
            actual, measured = matcher.classify(mask)
            assert actual == label
            assert measured == pytest.approx(confidence, abs=1e-7)
            assert (measured >= o.DIGIT_MIN_CONF) == (confidence >= o.DIGIT_MIN_CONF)
    assert matcher.classify(np.zeros((30, 20), dtype=np.uint8)) == (None, 0.0)


def test_incomplete_timeline_rows_do_not_read_unused_fields(monkeypatch):
    import numpy as np
    from video2tenhou import overlay as o, timeline

    image = np.zeros((1080, 1920, 3), np.uint8)
    monkeypatch.setattr(o._WIND, "match", lambda *a, **k: (None, 0.))
    def unused(*args, **kwargs):
        raise AssertionError("A missing round already excludes this frame")
    monkeypatch.setattr(o, "ocr_digit", unused)
    monkeypatch.setattr(o, "read_corner", unused)
    state = o.read_overlay(image, names=False, complete_only=True)
    assert not timeline.valid(timeline.reading_from_state(0., state))


def test_duplicate_seat_winds_cannot_become_a_valid_timeline_row(monkeypatch):
    import numpy as np
    from video2tenhou import overlay as o, timeline

    image = np.zeros((1080, 1920, 3), np.uint8)
    calls = []
    monkeypatch.setattr(o._WIND, "match", lambda *a, **k: ("E", 1.))
    monkeypatch.setattr(o, "ocr_digit", lambda *a, **k: 0)
    def corner(frame, name, **kwargs):
        calls.append(name)
        return o.CornerInfo(name, 25000, "", "E", 1.)
    monkeypatch.setattr(o, "read_corner", corner)
    state = o.read_overlay(image, names=False, complete_only=True)
    assert calls == ["TL", "TR"]
    assert not timeline.valid(timeline.reading_from_state(0., state))
    calls.clear()
    o.read_overlay(image, names=True)
    assert calls == ["TL", "TR", "BL", "BR"]  # diagnostics still read every corner


def test_ocr_reuses_only_exact_pixels_shape_and_configuration(monkeypatch):
    import numpy as np
    import pytesseract
    from video2tenhou import overlay as o

    calls = []
    def recognize(image, config):
        calls.append((image.shape, image.tobytes(), config))
        return f" {len(calls)} "
    monkeypatch.setattr(pytesseract, "image_to_string", recognize)
    o._tess_pixels.cache_clear()
    try:
        image = np.zeros((2, 4), np.uint8)
        assert o._tess(image, "--psm 6") == o._tess(image.copy(), "--psm 6") == "1"
        image[0, 0] = 1
        assert o._tess(image, "--psm 6") == "2"
        assert o._tess(image.reshape(4, 2), "--psm 6") == "3"
        assert o._tess(image, "--psm 8") == "4"
        assert len(calls) == 4
    finally:
        o._tess_pixels.cache_clear()
