# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The studio verifies its device and models before any jobs can be started."""

from __future__ import annotations

import os
from argparse import Namespace

import pytest

from video2tenhou import cli
from video2tenhou.paths import DATA_DIR
from video2tenhou.perception import classifier, detector, device
from video2tenhou.tool import server


@pytest.fixture(autouse=True)
def isolated_model_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace model loading while retaining runtime startup behavior."""
    monkeypatch.setenv("VIDEO2TENHOU_DEVICE", "auto")
    monkeypatch.setattr(detector, "Detector", object)
    monkeypatch.setattr(classifier, "Classifier", object)


def test_checked_device_is_inherited_by_jobs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VIDEO2TENHOU_DEVICE", "auto")
    monkeypatch.setattr(device, "select_device", lambda: "cpu")
    calls = []
    monkeypatch.setattr(
        server,
        "serve_workspace",
        lambda *_unused_a, **_unused_k: calls.append(os.environ["VIDEO2TENHOU_DEVICE"]),
    )
    cli.main(["web", "--data", str(DATA_DIR), "--port", "8899", "--no-browser"])
    assert calls == ["cpu"]
    assert "CPU (processing will be slower)" in capsys.readouterr().err


def test_unusable_explicit_device_stops_before_studio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise RuntimeError("GPU kernel check failed")

    monkeypatch.setattr(device, "select_device", fail)
    monkeypatch.setattr(
        server,
        "serve_workspace",
        lambda *_unused_a, **_unused_k: pytest.fail(
            "server started before runtime check"
        ),
    )
    with pytest.raises(RuntimeError, match="GPU kernel"):
        cli.cmd_web(Namespace(data=str(DATA_DIR), port=8899, no_browser=True))
