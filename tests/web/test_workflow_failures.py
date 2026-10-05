# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Unexpected storage and pipeline failures are reported, never hidden."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.web.analysis import write_analysis
from video2tenhou import cache, layout
from video2tenhou.files import atomic_write_json
from video2tenhou.layout import Calibration
from video2tenhou.tool import review_state
from video2tenhou.tool.processes import ProcessOwner
from video2tenhou.tool.workflow import ChildError, Job, Workspace


def test_invalid_manifest_is_not_silently_hidden(tmp_path: Path) -> None:
    manifest = tmp_path / "work/projects/broken.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        Workspace(tmp_path)
    assert manifest.read_text(encoding="utf-8") == "{broken"


@pytest.fixture
def analyzed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Workspace, str]]:
    """Prepare and analyze one project with model-free child commands."""
    monkeypatch.setattr(review_state, "ROOT", tmp_path)
    monkeypatch.setattr(layout, "LABEL_DIR", tmp_path / "labels")

    def runner(args: list[str], _job: Job) -> None:
        if args[4] == "calib":
            atomic_write_json(
                tmp_path / "labels/recording/calib.json", {"overhead": {}}
            )
        elif args[4] == "convert":
            write_analysis(tmp_path, "recording", [1])

    video = tmp_path / "recording.mp4"
    video.write_bytes(b"local recording")
    workspace = Workspace(tmp_path, runner=runner)
    key = workspace.create({"source": str(video), "games": [1]})["id"]
    for action in ("prepare", "analyze"):
        workspace.start(key, action)
        for thread in workspace.threads:
            thread.join(timeout=5)
    assert workspace.snapshot(key)["artifacts"]
    yield workspace, key
    workspace.close()


@pytest.mark.parametrize("operation", ["source", "record", "calibration"])
def test_storage_denial_is_not_reported_as_missing_or_stale(
    analyzed: tuple[Workspace, str],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """A storage failure reaches the caller instead of hiding exports."""
    workspace, key = analyzed
    denial = PermissionError("recording storage is unavailable")

    def denied(*_unused_args: object, **_unused_kwargs: object) -> None:
        raise denial

    if operation == "source":
        monkeypatch.setattr(cache, "source_identity", denied)
        Path(workspace.project(key)["video"]).write_bytes(b"replaced recording")
    elif operation == "record":
        original = Path.read_text

        def read_text(path: Path, encoding: str | None = None) -> str:
            if path.name == "record.json":
                denied()
            return original(path, encoding)

        monkeypatch.setattr(Path, "read_text", read_text)
    else:
        monkeypatch.setattr(Calibration, "load", denied)
    with pytest.raises(PermissionError) as error:
        workspace.snapshot(key)
    assert error.value is denial


def test_child_failure_without_an_outcome_keeps_exit_code_and_output(
    tmp_path: Path,
) -> None:
    """A crash reports its exit code; its output stays in the developer log."""
    workspace = Workspace(tmp_path)
    job = Job(kind="analyze", project="key", stage="Working")
    command = [
        sys.executable,
        "-c",
        "print('Original failure detail', flush=True); raise SystemExit(7)",
    ]
    try:
        with pytest.raises(ChildError) as error:
            workspace.run_child(command, job)
        assert error.value.unexpected
        assert error.value.message == "the command stopped with exit code 7"
        assert list(job.log) == ["Original failure detail"]
    finally:
        workspace.close()


def test_child_outcome_carries_its_result_and_partial_results(
    tmp_path: Path,
) -> None:
    """A command's final JSON line is its result, even beside an error."""
    workspace = Workspace(tmp_path)
    job = Job(kind="check", project="key", stage="Checking")
    script = (
        "import json,sys; print('[0 fit] checking', file=sys.stderr);"
        " print(json.dumps({'result': {'pond:TL': {'level': 'ok'}}}))"
    )
    failing = (
        "import json; print(json.dumps({'error': 'Adjust the table borders in "
        "Calibration: pond:TL cuts tiles.', 'result': {'pond:TL': {'level': "
        "'fail'}}})); raise SystemExit(1)"
    )
    try:
        assert workspace.run_child([sys.executable, "-c", script], job) == {
            "pond:TL": {"level": "ok"}
        }
        assert job.stage == "Checking table borders"
        with pytest.raises(ChildError) as error:
            workspace.run_child([sys.executable, "-c", failing], job)
        assert not error.value.unexpected
        assert error.value.result == {"pond:TL": {"level": "fail"}}
        assert error.value.message.endswith("pond:TL cuts tiles.")
    finally:
        workspace.close()


def test_review_reader_does_not_hide_or_retry_corrupt_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt decode fails loudly and stays unchanged."""
    monkeypatch.setattr(review_state, "ROOT", tmp_path)
    write_analysis(tmp_path, "recording", [1])
    path = tmp_path / "work/recording/decode/00.json"
    path.write_text("{broken", encoding="utf-8")
    state = review_state.ReviewState(
        tmp_path / "recording.mp4",
        tmp_path / "work",
        "pml",
        tmp_path / "out",
        ProcessOwner(),
    )
    with pytest.raises(json.JSONDecodeError):
        state.hand_summary()
    assert path.read_text(encoding="utf-8") == "{broken"
