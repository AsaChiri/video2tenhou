"""Setup failures give actionable advice without mistaking optional warnings for errors."""
import pytest

from video2tenhou.tool.workflow import _failure_message


@pytest.mark.parametrize("line,expected", [
    ("RuntimeError: CUDA error: no kernel image is available for execution on the device", "relaunch Start.cmd"),
    ("RuntimeError: CUDA out of memory", "Close other GPU applications"),
    ("FileNotFoundError: models/detector/weights.pt", "complete model bundle"),
    ("ValueError: Detector metadata does not match checkpoint SHA-256", "complete model bundle"),
])
def test_setup_failures_have_specific_recovery(line, expected):
    assert expected in _failure_message([line])


def test_optional_warning_does_not_mask_other_error():
    assert "processing log" in _failure_message(["xFormers not available", "ValueError: other problem"])
