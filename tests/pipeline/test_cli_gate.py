# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Conversion requires measured geometry and a usable border check.

`convert` must not start on a video whose geometry has not been measured (DESIGN.md
4.2a).
"""

from argparse import Namespace
from typing import TYPE_CHECKING

import pytest

from video2tenhou import cli
from video2tenhou.layout import Calibration

if TYPE_CHECKING:
    from pathlib import Path


def args(**kw: "object") -> "Namespace":
    """Build conversion arguments with test-specific overrides."""
    a = Namespace(
        video="videos/nothing_here.mp4", calib="pml", work="work", skip_fit_check=False
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_gate_refuses_a_video_with_no_fit(tmp_path: "Path") -> None:
    """Verify gate refuses a video with no fit."""
    cal = Calibration.load("pml")  # no video: cal.fit is None
    with pytest.raises(SystemExit) as e:
        cli._gate(args(), cal, tmp_path)
    assert "calib fit" in str(e.value)


def test_gate_can_be_overridden_explicitly(tmp_path: "Path") -> None:
    """Verify gate can be overridden explicitly."""
    cal = Calibration.load("pml")
    cli._gate(
        args(skip_fit_check=True), cal, tmp_path
    )  # returns without touching the models
