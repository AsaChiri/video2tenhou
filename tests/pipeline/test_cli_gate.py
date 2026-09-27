"""`convert` must not start on a video whose geometry has not been measured (DESIGN.md 4.2a)."""
import types

import pytest

from video2tenhou import cli
from video2tenhou.layout import Calibration


def args(**kw):
    a = types.SimpleNamespace(video="videos/nothing_here.mp4", calib="pml", work="work", skip_fit_check=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_gate_refuses_a_video_with_no_fit(tmp_path):
    cal = Calibration.load("pml")                      # no video: cal.fit is None
    with pytest.raises(SystemExit) as e:
        cli._gate(args(), cal, tmp_path)
    assert "calib fit" in str(e.value)


def test_gate_can_be_overridden_explicitly(tmp_path):
    cal = Calibration.load("pml")
    cli._gate(args(skip_fit_check=True), cal, tmp_path)   # returns without touching the models
