# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Conversion requires measured geometry and a usable border check.

`convert` must not start on a video whose geometry has not been measured (DESIGN.md
4.2a).
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from tests.recognition import RecognitionStub
from video2tenhou import cli, read
from video2tenhou.layout import Calibration


def args(**kw: object) -> Namespace:
    """Build conversion arguments with test-specific overrides."""
    a = Namespace(
        video="videos/nothing_here.mp4", calib="pml", work="work", skip_fit_check=False
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def context(tmp_path: Path) -> read.ReadContext:
    """Use the layout without a fit and models that fail if inference starts."""
    model = RecognitionStub("unused")
    return read.ReadContext("v.mp4", Calibration.load("pml"), tmp_path, model, model)


def test_gate_refuses_a_video_with_no_fit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as e:
        cli._gate(args(), context(tmp_path))
    assert "Prepare the recording first" in str(e.value)


def test_gate_can_be_overridden_explicitly(tmp_path: Path) -> None:
    cli._gate(args(skip_fit_check=True), context(tmp_path))
