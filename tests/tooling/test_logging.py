# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Command result streams remain parseable alongside progress and embedded callers."""

from __future__ import annotations

import json
import logging
from contextlib import closing
from io import StringIO
from typing import TYPE_CHECKING

import pytest

from video2tenhou.calibfit import RegionCheck
from video2tenhou.logging_setup import RESULT, command_logging
from video2tenhou.tool import check_calibration

if TYPE_CHECKING:
    from pathlib import Path


def test_command_results_survive_progress_and_repeated_invocations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each invocation emits one bare JSON document and sends progress to stderr."""

    @command_logging
    def report() -> None:
        logging.getLogger("video2tenhou.test").info("Progress: 100%% (%s)", "done")
        RESULT.info("%s", json.dumps({"value": "100% complete"}))

    for _ in range(2):
        report()
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {"value": "100% complete"}
        assert captured.err == "Progress: 100% (done)\n"


def test_failed_command_restores_embedding_apps_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed command cannot replace the caller's existing logging destination."""
    logger = logging.getLogger("video2tenhou")
    stream = StringIO()
    with closing(logging.StreamHandler(stream)) as handler:
        monkeypatch.setattr(logger, "handlers", [handler])
        monkeypatch.setattr(logger, "level", logging.WARNING)
        monkeypatch.setattr(logger, "propagate", False)

        @command_logging
        def fail() -> None:
            message = "Command failed"
            raise RuntimeError(message)

        with pytest.raises(RuntimeError, match="Command failed"):
            fail()
        logger.info("Hidden message")
        logger.warning("Caller still receives warnings")
        assert stream.getvalue() == "Caller still receives warnings\n"


def test_calibration_command_emits_one_json_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The review subprocess protocol tolerates model progress on stderr."""
    monkeypatch.setattr(check_calibration, "Detector", object)

    def checks(*_args: object) -> list[RegionCheck]:
        logging.getLogger("video2tenhou.calibfit").info("Checking table borders")
        return [RegionCheck("pond:TL", note="100% checked")]

    monkeypatch.setattr(check_calibration, "check_all", checks)
    check_calibration.main([str(tmp_path / "video.mp4"), "pml", str(tmp_path)])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "pond:TL": {"level": "ok", "held": 0, "cut": 0, "note": "100% checked"}
    }
    assert captured.err == "Checking table borders\n"
