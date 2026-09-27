"""The selected layout must reach the pixel reader, not just the tile crops."""

from tests.paths import DATA
import copy
from pathlib import Path

import cv2
import numpy as np

from video2tenhou.layout import Calibration
from video2tenhou.overlay import read_overlay
from video2tenhou import timeline


def test_shifted_overlay_layout_through_header_scan(monkeypatch):
    cal = Calibration.load("pml")
    data = copy.deepcopy(cal.data)
    shift = 30
    for key in ("score", "name", "wind", "strip"):
        for rect in data["overlay"][key].values():
            rect[1] += shift
    for key in ("round_wind", "round_num", "honba", "sticks"):
        data["overlay"][key][1] += shift
    frame = cv2.imread(str(DATA / "pml_e4_h0.jpg"))
    shifted = np.zeros_like(frame)
    shifted[shift:] = frame[:-shift]
    monkeypatch.setattr(timeline.video, "sample", lambda *args, **kw: iter([(12.0, shifted)]))
    rows = list(timeline.scan("synthetic.mp4", Calibration(data)))
    assert len(rows) == 1 and rows[0]["ok"]
    assert rows[0]["kyoku"] == 3 and rows[0]["scores"]["TL"] == 24900
