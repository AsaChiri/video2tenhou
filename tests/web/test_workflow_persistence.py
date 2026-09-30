# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Job admission and completion stay truthful when project storage fails."""

import ctypes
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from video2tenhou import files
from video2tenhou.tool import workflow

if TYPE_CHECKING:
    from collections.abc import Iterator

    from video2tenhou.tool.workflow import Workspace


@pytest.fixture
def project(
    tmp_path: "Path",
) -> "Iterator[tuple[Workspace, str, Path, list[list[str]]]]":
    """Create a persisted studio project with a controlled command runner."""
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"local recording identity")
    commands = []
    workspace = workflow.Workspace(
        tmp_path, runner=lambda args, _p: commands.append(args)
    )
    value = workspace.create({"source": str(video), "games": [22002]})
    key = value["id"]
    path = workspace.projects_dir / f"{key}.json"
    yield workspace, key, path, commands
    workspace.close()


def finish(workspace: "Workspace") -> None:
    """Join all workspace jobs and verify that each worker stopped."""
    for thread in workspace._threads:
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("boundary", ["temporary_file", "replacement"])
def test_failed_admission_keeps_manifest_and_allows_retry(
    project: "tuple[Workspace, str, Path, list[list[str]]]",
    monkeypatch: "pytest.MonkeyPatch",
    boundary: str,
) -> None:
    """Verify failed admission keeps manifest and allows retry."""
    workspace, key, path, commands = project
    before = path.read_bytes()
    attempts = []
    original_replace = Path.replace

    def denied(*_unused_args: object, **_unused_kwargs: object) -> None:
        attempts.append(True)
        msg = "data folder denies writing"
        raise PermissionError(msg)

    def replace(source: "Path", target: "Path") -> "Path | None":
        if target == path:
            return denied()
        return original_replace(source, target)

    with monkeypatch.context() as patch:
        if boundary == "temporary_file":
            patch.setattr(files.tempfile, "NamedTemporaryFile", denied)
        else:
            patch.setattr(Path, "replace", replace)
        with pytest.raises(ValueError, match="Could not save the project"):
            workspace.start(key, "prepare")
        assert attempts == [True]  # Ordinary permission failures are not retried.
        assert not commands
        value = workspace.snapshot(key)
        assert value["status"] == "failed"
        assert not value["job"]["running"]
        assert "retry" in value["job"]["error"]
        assert path.read_bytes() == before
        assert not list(path.parent.glob(".*.tmp"))
    workspace.start(key, "prepare")
    finish(workspace)
    assert len(commands) == 1
    assert json.loads(path.read_text())["status"] == "ready"
    reopened = workflow.Workspace(workspace.root)
    try:
        assert reopened.snapshot(key)["status"] == "ready"
    finally:
        reopened.close()


def test_persistent_windows_denial_is_bounded_and_retryable(
    project: "tuple[Workspace, str, Path, list[list[str]]]",
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify persistent windows denial is bounded and retryable."""
    workspace, key, path, commands = project
    before = path.read_bytes()
    original = Path.replace
    attempts = []

    def replace(source: "Path", target: "Path") -> "Path":
        if target == path:
            attempts.append(True)
            error = PermissionError("persistent replacement denial")
            error.winerror = 5
            raise error
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", replace)
        with pytest.raises(ValueError, match="Could not save"):
            workspace.start(key, "prepare")
    assert 1 <= len(attempts) <= 5
    assert not commands
    assert path.read_bytes() == before
    assert not workspace.snapshot(key)["job"]["running"]
    workspace.start(key, "prepare")
    finish(workspace)
    assert len(commands) == 1
    assert workspace.snapshot(key)["status"] == "ready"


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing semantics")
def test_real_windows_reader_contention_recovers_before_launch(
    project: "tuple[Workspace, str, Path, list[list[str]]]",
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify real windows reader contention recovers before launch."""
    workspace, key, path, commands = project
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    before = path.read_bytes()
    original = Path.replace
    denied = []

    def replace(source: "Path", target: "Path") -> "Path":
        nonlocal handle
        try:
            return original(source, target)
        except OSError as exc:
            if target == path:
                denied.append(exc.winerror)
                assert not commands
                assert path.read_bytes() == before
                if handle is not None:
                    kernel.CloseHandle(handle)
                    handle = None
            raise

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", replace)
            workspace.start(key, "prepare")
            finish(workspace)
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
    assert len(denied) == 1
    assert denied[0] in (5, 32, 33)
    assert len(commands) == 1
    assert json.loads(path.read_text())["status"] == "ready"


def test_completion_save_failure_is_visible_and_restart_recovers(
    project: "tuple[Workspace, str, Path, list[list[str]]]",
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify completion save failure is visible and restart recovers."""
    workspace, key, path, commands = project
    entered, release = threading.Event(), threading.Event()

    def runner(args: "list[str]", value: "dict") -> None:
        commands.append(args)
        entered.set()
        assert release.wait(5)

    workspace.runner = runner
    workspace.start(key, "prepare")
    assert entered.wait(5)
    admitted = path.read_bytes()
    assert json.loads(admitted)["job"]["running"]
    original = Path.replace

    def replace(source: "Path", target: "Path") -> "Path":
        if target == path:
            msg = "completion cannot be saved"
            raise PermissionError(msg)
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", replace)
        release.set()
        finish(workspace)
        value = workspace.snapshot(key)
        assert value["status"] == "failed"
        assert not value["job"]["running"]
        assert value["export_signature"] is None
        assert "Could not save" in value["job"]["error"]
        assert "completion cannot be saved" in value["job"]["log"][-1]
        assert path.read_bytes() == admitted
    reopened = workflow.Workspace(
        workspace.root, runner=lambda args, _p: commands.append(args)
    )
    try:
        assert reopened.snapshot(key)["status"] == "interrupted"
        assert not reopened.snapshot(key)["job"]["running"]
        reopened.start(key, "prepare")
        finish(reopened)
        assert reopened.snapshot(key)["status"] == "ready"
        assert json.loads(path.read_text())["status"] == "ready"
    finally:
        reopened.close()


def test_create_failure_does_not_leave_an_unopenable_project(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify create failure does not leave an unopenable project."""
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"recording")
    workspace = workflow.Workspace(tmp_path)

    def denied(*_unused_args: object, **_unused_kwargs: object) -> None:
        msg = "cannot create manifest"
        raise PermissionError(msg)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(files.tempfile, "NamedTemporaryFile", denied)
            with pytest.raises(PermissionError):
                workspace.create({"source": str(video), "games": [1]})
        assert workspace.projects == {}
        project = workspace.create({"source": str(video), "games": [1]})
        assert workspace.snapshot(project["id"])["status"] == "new"
    finally:
        workspace.close()
