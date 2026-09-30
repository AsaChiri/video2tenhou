"""Unexpected storage and pipeline failures retain their original diagnostics."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from video2tenhou.tool.workflow import Workspace


def test_invalid_manifest_is_not_silently_hidden(tmp_path):
    manifest = tmp_path / "work/projects/broken.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        Workspace(tmp_path)
    assert manifest.read_text(encoding="utf-8") == "{broken"


@pytest.mark.parametrize("operation", ["source", "record", "calibration"])
def test_storage_denial_is_not_reported_as_missing_or_stale(
    tmp_path, monkeypatch, operation
):
    from video2tenhou import cache
    from video2tenhou.layout import Calibration

    workspace = Workspace(tmp_path)
    project = {
        "name": "recording",
        "video": "recording.mp4",
        "layout": "pml",
        "games": [1],
    }
    denial = PermissionError("recording storage is unavailable")

    def denied(*args, **kwargs):
        raise denial

    try:
        if operation == "source":
            monkeypatch.setattr(cache, "source_identity", denied)
            action = workspace._source
        elif operation == "record":
            monkeypatch.setattr(Path, "read_text", denied)
            action = workspace._record_matches
        else:
            monkeypatch.setattr(cache, "source_identity", lambda path: "digest")
            monkeypatch.setattr(Calibration, "load", denied)
            action = workspace._signature
        with pytest.raises(PermissionError) as error:
            action(project)
        assert error.value is denial
    finally:
        workspace.close()


def test_subprocess_failure_retains_exit_code_and_output(tmp_path):
    workspace = Workspace(tmp_path)
    project = {"job": {"log": []}}
    command = [
        sys.executable,
        "-c",
        "print('Original failure detail', flush=True); raise SystemExit(7)",
    ]
    try:
        with pytest.raises(subprocess.CalledProcessError) as error:
            workspace._run_command(command, project)
        assert error.value.returncode == 7
        assert error.value.cmd == command
        assert error.value.output == "Original failure detail"
        assert project["job"]["log"] == ["Original failure detail"]
    finally:
        workspace.close()


def test_review_reader_does_not_hide_or_retry_corrupt_data(tmp_path):
    from video2tenhou.tool.review_state import ReviewState

    path = tmp_path / "review-changes.json"
    assert ReviewState._read_json(path) is None
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        ReviewState._read_json(path)
    assert path.read_text(encoding="utf-8") == "{broken"
