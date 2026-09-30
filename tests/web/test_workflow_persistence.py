"""Job admission and completion stay truthful when project storage fails."""

import ctypes
import json
import os
import threading
from pathlib import Path

import pytest

from video2tenhou import files
from video2tenhou.tool import workflow


@pytest.fixture
def project(tmp_path):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"local recording identity")
    commands = []
    workspace = workflow.Workspace(
        tmp_path, runner=lambda args, p: commands.append(args)
    )
    value = workspace.create({"source": str(video), "games": [22002]})
    key = value["id"]
    path = workspace.projects_dir / f"{key}.json"
    yield workspace, key, path, commands
    workspace.close()


def finish(workspace):
    for thread in workspace._threads:
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("boundary", ["temporary_file", "replacement"])
def test_failed_admission_keeps_manifest_and_allows_retry(
    project, monkeypatch, boundary
):
    workspace, key, path, commands = project
    before = path.read_bytes()
    attempts = []
    original_replace = Path.replace

    def denied(*args, **kwargs):
        attempts.append(True)
        raise PermissionError("data folder denies writing")

    def replace(source, target):
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
        assert value["status"] == "failed" and not value["job"]["running"]
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


def test_persistent_windows_denial_is_bounded_and_retryable(project, monkeypatch):
    workspace, key, path, commands = project
    before = path.read_bytes()
    original = Path.replace
    attempts = []

    def replace(source, target):
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
    assert not commands and path.read_bytes() == before
    assert not workspace.snapshot(key)["job"]["running"]
    workspace.start(key, "prepare")
    finish(workspace)
    assert len(commands) == 1 and workspace.snapshot(key)["status"] == "ready"


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing semantics")
def test_real_windows_reader_contention_recovers_before_launch(project, monkeypatch):
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

    def replace(source, target):
        nonlocal handle
        try:
            return original(source, target)
        except OSError as exc:
            if target == path:
                denied.append(exc.winerror)
                assert not commands and path.read_bytes() == before
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
    assert len(denied) == 1 and denied[0] in (5, 32, 33)
    assert len(commands) == 1
    assert json.loads(path.read_text())["status"] == "ready"


def test_completion_save_failure_is_visible_and_restart_recovers(project, monkeypatch):
    workspace, key, path, commands = project
    entered, release = threading.Event(), threading.Event()

    def runner(args, value):
        commands.append(args)
        entered.set()
        assert release.wait(5)

    workspace.runner = runner
    workspace.start(key, "prepare")
    assert entered.wait(5)
    admitted = path.read_bytes()
    assert json.loads(admitted)["job"]["running"]
    original = Path.replace

    def replace(source, target):
        if target == path:
            raise PermissionError("completion cannot be saved")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", replace)
        release.set()
        finish(workspace)
        value = workspace.snapshot(key)
        assert value["status"] == "failed" and not value["job"]["running"]
        assert value["export_signature"] is None
        assert "Could not save" in value["job"]["error"]
        assert "completion cannot be saved" in value["job"]["log"][-1]
        assert path.read_bytes() == admitted
    reopened = workflow.Workspace(
        workspace.root, runner=lambda args, p: commands.append(args)
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


def test_create_failure_does_not_leave_an_unopenable_project(tmp_path, monkeypatch):
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"recording")
    workspace = workflow.Workspace(tmp_path)

    def denied(*args, **kwargs):
        raise PermissionError("cannot create manifest")

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
