# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Command result streams remain parseable alongside progress and embedded callers."""

from __future__ import annotations

import json
import logging
from contextlib import closing
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from video2tenhou import calibfit, cli
from video2tenhou.calibfit import RegionCheck
from video2tenhou.logging_setup import RESULT, command_logging
from video2tenhou.perception import detector


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
            raise RuntimeError("Command failed")

        with pytest.raises(RuntimeError, match="Command failed"):
            fail()
        logger.info("Hidden message")
        logger.warning("Caller still receives warnings")
        assert stream.getvalue() == "Caller still receives warnings\n"


@pytest.mark.parametrize("level", ["ok", "fail"])
def test_calibration_command_emits_one_json_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    level: str,
) -> None:
    """Studio child commands end with one outcome; progress stays on stderr."""
    monkeypatch.setattr(detector, "Detector", object)

    def checks(*_args: object) -> list[RegionCheck]:
        logging.getLogger("video2tenhou.calibfit").info("Checking table borders")
        return [RegionCheck("pond:TL", level=level, cut=3, note="100% checked")]

    monkeypatch.setattr(calibfit, "check_all", checks)
    monkeypatch.setattr(
        calibfit, "fit_sheet", lambda *_args: np.zeros((2, 2, 3), np.uint8)
    )
    video = str(tmp_path / "video.mp4")
    code = cli.main(
        ["calib", "check", video, "--work", str(tmp_path), "--out", str(tmp_path)]
    )
    captured = capsys.readouterr()
    result = {"pond:TL": {"level": level, "held": 0, "cut": 3, "note": "100% checked"}}
    if level == "ok":
        assert code == 0
        assert json.loads(captured.out) == {"result": result}
    else:
        assert code == 1
        assert json.loads(captured.out) == {
            "error": "Adjust the table borders in Calibration: pond:TL cuts tiles.",
            "result": result,
        }
    assert "Checking table borders\n" in captured.err
    assert "{" not in captured.err
