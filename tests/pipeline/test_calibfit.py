"""Stage 0: the per-video fit and the border check (DESIGN.md 4.2a).

The tests use synthetic pictures for the geometry and the real reference plate, when it has been
built, for the fit that must come back as the calibration it was cut from.
"""

from tests.paths import DATA, ROOT
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from video2tenhou import calibfit
from video2tenhou.layout import Calibration, Rect, apply_fit, fit_path

REF_PLATE = ROOT / "work" / "full_1080p" / "plate.png"


def test_fresh_and_warm_geometry_gate_share_authenticated_play_windows(tmp_path, monkeypatch):
    from video2tenhou import timeline
    source = tmp_path / "broadcast.mp4"
    source.write_bytes(b"test source")
    cal = Calibration.load("pml")
    rows = [{"t": t, "kyoku": 0, "honba": 0, "sticks": 0,
             "scores": dict.fromkeys(("TL", "TR", "BL", "BR"), 25000),
             "winds": dict(zip(("TL", "TR", "BL", "BR"), "ESWN")), "ok": True}
            for t in range(100, 201)]
    scans = []
    def scan(*args, **kwargs):
        scans.append(True)
        yield from rows
    monkeypatch.setattr(timeline, "scan", scan)
    sampled = []
    blocks = object()
    monkeypatch.setattr(calibfit, "table_plate", lambda *a, **k: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(calibfit, "pond_blocks", lambda *a: blocks)
    def check(video, calibration, detector, hands, **kwargs):
        sampled.append(calibfit.check_times(video, hands, 8))
        assert kwargs["blocks"] is blocks
        return []
    monkeypatch.setattr(calibfit, "check_regions", check)
    calibfit.check_all(source, cal, object(), tmp_path)
    # A stale hands.json cannot influence a gate; only matching numeric evidence can.
    (tmp_path / "hands.json").write_text('[{"t_start":999,"t_end":1000}]')
    calibfit.check_all(source, cal, object(), tmp_path)
    pipeline = timeline.play_hands(rows)[0]
    start, end = pipeline.t_read
    assert sampled == [[start + .55 * (end - start)]] * 2
    assert len(scans) == 1


def test_geometry_gate_rejects_missing_play_and_retains_real_border_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from video2tenhou import cli, timeline
    from video2tenhou.perception import detector
    monkeypatch.setattr(timeline, "run_scan", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="No stable play"):
        calibfit.prepare_hands(Path("v.mp4"), Calibration.load("pml"), tmp_path)
    failure = calibfit.RegionCheck("pond:BR", held=43, cut=6)
    failure.verdict()
    assert not failure.ok
    monkeypatch.setattr(calibfit, "check_all", lambda *a: [failure])
    monkeypatch.setattr(detector, "Detector", lambda: object())
    cal = Calibration.load("pml")
    cal.fit = {"overhead": {}}
    with pytest.raises(SystemExit, match="pond:BR"):
        cli._gate(SimpleNamespace(skip_fit_check=False, video="v.mp4"), cal, tmp_path)


def test_plate_cache_tracks_source_and_sampling_recipe(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / "v.mp4"
    source.write_bytes(b"first")
    calls = []
    monkeypatch.setattr(calibfit.videomod, "probe", lambda path: SimpleNamespace(duration=10.))
    def frame(path, t):
        calls.append(t)
        return np.full((4, 4, 3), 10 if source.read_bytes() == b"first" else 20, np.uint8)
    monkeypatch.setattr(calibfit.videomod, "frame_at", frame)
    work = tmp_path / "work"
    assert np.all(calibfit.table_plate(source, work, n=3) == 10)
    calibfit.table_plate(source, work, n=3)
    assert len(calls) == 3
    source.write_bytes(b"replacement")
    assert np.all(calibfit.table_plate(source, work, n=3) == 20)
    assert len(calls) == 6
    calibfit.table_plate(source, work, n=4)
    assert len(calls) == 10


def test_meld_refinement_uses_crossing_boxes_not_neighbouring_tiles(monkeypatch):
    from video2tenhou.perception.detector import Det
    cal = Calibration.load("pml")
    data = json.loads(json.dumps(cal.data))
    data["meld"]["TL"] = {"rect": [100, 100, 100, 100], "scale": 1.}
    cal = Calibration(data)
    monkeypatch.setattr(calibfit, "check_times", lambda *args: [1.])
    monkeypatch.setattr(calibfit.videomod, "frame_at", lambda *args: np.zeros((300, 300, 3), np.uint8))
    class Detector:
        def predict(self, image):
            return [Det((60, 125, 100, 155), .9, False),  # cut by the bottom border
                    Det((0, 0, 20, 20), .99, False)]    # outside: must not expand top/left
    assert calibfit._expand_cut_meld(Path("v.mp4"), cal, "TL", Detector(), []) == [100, 100, 100, 119]


@pytest.mark.parametrize("improves", [True, False])
def test_automatic_meld_refinement_requires_a_better_border_check(tmp_path, monkeypatch, improves):
    cal = Calibration.load("pml")
    monkeypatch.setattr(calibfit, "table_plate", lambda *args, **kwargs: object())
    monkeypatch.setattr(calibfit, "fit_overhead", lambda *args: {"center": [960, 540], "angle": 45., "scale": 1., "iou": 1.})
    monkeypatch.setattr(calibfit, "prepare_hands", lambda *args: [{"t_start": 0., "t_end": 60.}])
    monkeypatch.setattr(calibfit, "fit_panel", lambda plate, rect: (100, 100, 100, 100) if rect == cal.meld["TL"][0] else None)
    monkeypatch.setattr(calibfit, "fit_path", lambda path: tmp_path / "calib.json")
    monkeypatch.setattr(calibfit, "_expand_cut_meld", lambda *args: [100, 100, 100, 119])
    def check(video, candidate, *args, **kwargs):
        rect = candidate.meld["TL"][0]
        good = rect.h == 119 and improves
        rc = calibfit.RegionCheck("meld:TL", held=3 if good else 0,
                                  cut=0 if good else (3 if rect.h == 100 else 6))
        rc.verdict()
        return [rc]
    monkeypatch.setattr(calibfit, "check_regions", check)
    fitted = calibfit.run_fit(Path("v.mp4"), cal, tmp_path, det=object(), keep={"hand"})
    assert fitted["meld"]["TL"]["rect"] == [100, 100, 100, 119 if improves else 100]


def test_human_meld_fit_is_never_expanded(tmp_path, monkeypatch):
    cal = Calibration.load("pml")
    cal.fit = {"overhead": {"center": [960, 540], "angle": 45., "scale": 1.},
               "meld": {c: {"rect": [100, 100, 100, 100], "source": "human"} for c in ("TL", "TR", "BL", "BR")}}
    monkeypatch.setattr(calibfit, "table_plate", lambda *args, **kwargs: object())
    monkeypatch.setattr(calibfit, "prepare_hands", lambda *args: [])
    monkeypatch.setattr(calibfit, "fit_panel", lambda *args: pytest.fail("Human rectangle must not be measured again"))
    monkeypatch.setattr(calibfit, "fit_path", lambda path: tmp_path / "calib.json")
    fitted = calibfit.run_fit(Path("v.mp4"), cal, tmp_path, det=object(), keep={"hand", "overhead"})
    assert fitted["meld"] == cal.fit["meld"]


@pytest.fixture(scope="module")
def cal():
    return Calibration.load("pml")


# -- the fit is applied, and only where it says something -------------------------------

def test_apply_fit_replaces_only_what_it_names(cal):
    fit = {"overhead": {"center": [1000.0, 500.0], "angle": 46.5, "scale": 0.97},
           "meld": {"TL": {"rect": [10, 20, 100, 110]}},
           "hand": {"BR": {"roll": 7.5}}}
    c2 = Calibration(apply_fit(cal.data, fit))
    assert c2.center == (1000.0, 500.0) and c2.angle == 46.5 and c2.scale == 0.97
    assert c2.meld["TL"][0] == Rect(10, 20, 100, 110)
    assert c2.meld["TR"][0] == cal.meld["TR"][0]          # untouched corners keep the layout
    assert c2.roll("BR") == 7.5 and c2.roll("TL") == cal.roll("TL")
    assert c2.hand["BR"][0] == cal.hand["BR"][0]          # a roll alone does not move the band
    assert cal.center == (960.0, 540.0)                   # the layout itself is never edited in place


def test_pond_rects_are_the_table_not_the_fit(cal):
    """Pond rectangles are a property of the table, so a fit may not carry them: they follow the overhead."""
    fit = {"overhead": {"center": [1010.0, 530.0], "angle": 46.0, "scale": 1.0}, "pond": {"TL": {"rect": [0, 0, 5, 5]}}}
    c2 = Calibration(apply_fit(cal.data, fit))
    assert c2.pond["TL"][0] == cal.pond["TL"][0]


def test_overhead_scale_moves_the_pond_in_the_frame(cal):
    """The whole point of the fit: a different overhead puts the same pond rectangle somewhere else."""
    a, _ = cal.transform("pond:TL")
    c2 = Calibration(apply_fit(cal.data, {"overhead": {"center": [1015.0, 537.0], "angle": 46.5, "scale": 0.99}}))
    b, _ = c2.transform("pond:TL")
    assert not np.allclose(a, b)


# -- the border check -------------------------------------------------------------------

def test_cut_classifies_a_box_against_a_border():
    inner = (100.0, 100.0, 300.0, 300.0)
    assert calibfit._cut((120, 120, 180, 200), inner) is False      # wholly inside
    assert calibfit._cut((10, 120, 60, 200), inner) is None         # wholly outside
    assert calibfit._cut((60, 120, 160, 200), inner) is True        # half on each side of the left border


def test_grown_region_holds_the_true_region_with_a_margin(cal):
    for name in ("pond:TL", "pond:BR", "meld:TR", "hand:BL"):
        kind = name.partition(":")[0]
        M, inner, size = calibfit.grown(cal, name, calibfit.GROW[kind])
        assert 0 < inner[0] < inner[2] < size[0] and 0 < inner[1] < inner[3] < size[1]
        base, (w, h) = cal.transform(name)
        assert abs((inner[2] - inner[0]) - w) <= 2 and abs((inner[3] - inner[1]) - h) <= 2


def test_verdicts_follow_the_counts():
    def rc(held, cut, region="pond:TL"):
        c = calibfit.RegionCheck(region, held=held, cut=cut, frames=8)
        c.verdict()
        return c.level
    assert rc(60, 0) == "ok"
    assert rc(60, 1) == "ok"                 # one stray box in a busy region is not a misplaced region
    assert rc(60, 5) == "warn"               # a few: worth a look, not a reason to stop the run
    assert rc(60, 10) == "fail"
    assert rc(3, 3) == "fail"                # every tile at the border cut: the region is in the wrong place
    assert rc(0, 0) == "fail"                # a pond that never holds a tile is not looking at the table
    assert rc(0, 0, "meld:TL") == "warn"     # a meld camera may see no call in eight frames: look at it
    c = calibfit.RegionCheck("pond:TL", held=60, cut=0, frames=8, foreign=2)
    c.verdict()
    assert c.level == "fail"                 # a pond holding a neighbour's discards reads the wrong pond


# -- the overhead fit -------------------------------------------------------------------

def test_unit_template_is_in_overhead_coordinates(cal):
    t = calibfit.unit_template(cal)
    assert t.shape == (cal.side, cal.side)
    ys, xs = np.nonzero(t)
    u = cal.unit
    assert abs(xs.mean() - (u.x + u.w / 2)) < 30 and abs(ys.mean() - (u.y + u.h / 2)) < 30


def test_unit_mask_finds_a_dark_square_on_felt(cal):
    plate = np.zeros((1080, 1920, 3), np.uint8)
    plate[:, :] = (120, 110, 40)                            # teal felt: saturated
    cv2.fillPoly(plate, [np.array([[960, 400], [1120, 540], [960, 680], [800, 540]])], (60, 60, 60))
    m = calibfit.unit_mask(plate)
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - 960) < 12 and abs(ys.mean() - 540) < 12


def test_unit_mask_without_a_unit_says_so(cal):
    plate = np.zeros((1080, 1920, 3), np.uint8)
    plate[:, :] = (120, 110, 40)
    with pytest.raises(RuntimeError):
        calibfit.unit_mask(plate)


@pytest.mark.skipif(not REF_PLATE.exists(), reason="the reference plate has not been built")
def test_reference_video_fits_back_to_its_own_calibration(cal):
    """The layout was drawn on the reference VOD, so fitting that video must return that layout."""
    fit = calibfit.fit_overhead(cv2.imread(str(REF_PLATE)), cal)
    assert fit["iou"] > 0.95
    assert abs(fit["center"][0] - cal.center[0]) < 1.5 and abs(fit["center"][1] - cal.center[1]) < 1.5
    assert abs(fit["angle"] - cal.angle) < 0.5 and abs(fit["scale"] - 1.0) < 0.02


def test_second_video_fit_is_not_the_layout():
    """A second broadcast of the same layout does not have the same table geometry: if it did, stage 0
    would be pointless. This guards against a fit that silently falls back to the layout's numbers."""
    fit = json.loads((DATA / "week_11_calib.json").read_text(encoding="utf-8"))
    cal = Calibration.load("pml")
    oh = fit["overhead"]
    assert abs(oh["center"][0] - cal.center[0]) > 10       # the overhead really did move
