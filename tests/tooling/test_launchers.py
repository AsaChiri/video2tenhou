# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise source launchers with local tools, without installing dependencies."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.paths import ROOT


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("success", 0),
        ("setup_failure", 7),
        ("venv_failure", 6),
        ("cpu", 0),
        ("custom_backend", 0),
        ("runtime_failure", 9),
        ("missing_tool", 1),
    ],
)
def test_launcher_paths_arguments_and_failure_boundaries(
    tmp_path: "Path", case: str, expected: int
) -> None:
    """Verify launcher paths arguments and failure boundaries."""
    checkout = tmp_path / "starter with spaces"
    tools = tmp_path / "stub tools"
    checkout.mkdir()
    tools.mkdir()
    log = tmp_path / "launch.log"
    data_home = tmp_path / "separate data directory"
    windows = os.name == "nt"
    launcher = "Start.cmd" if windows else "start.sh"
    shutil.copyfile(ROOT / launcher, checkout / launcher)
    env = {
        **os.environ,
        "LAUNCH_LOG": str(log),
        "VIDEO2TENHOU_HOME": str(data_home),
        "SETUP_EXIT": "7" if case == "setup_failure" else "0",
        "VENV_EXIT": "6" if case == "venv_failure" else "0",
        "RUN_EXIT": "9" if case == "runtime_failure" else "0",
    }
    env.pop("UV_TORCH_BACKEND", None)
    env.pop("VIDEO2TENHOU_DEVICE", None)
    if case == "cpu":
        env["VIDEO2TENHOU_DEVICE"] = "cpu"
    if case == "custom_backend":
        env["UV_TORCH_BACKEND"] = "chosen-backend"
    command = (
        _windows_tools(checkout, tools, case, env)
        if windows
        else _posix_tools(checkout, tools, case, env)
    )
    result = subprocess.run(  # noqa: S603
        command,
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    if case == "missing_tool":
        assert "ffprobe" in result.stdout + result.stderr
        assert "QUICKSTART.md" in result.stdout + result.stderr
        assert not log.exists()
        return
    calls = [line.split("|", 2) for line in log.read_text().splitlines()]
    assert all(
        Path(cwd).resolve() == checkout.resolve() and home == str(data_home)
        for cwd, home, _ in calls
    )
    assert len(calls) == (
        1 if case == "venv_failure" else 2 if case == "setup_failure" else 3
    )
    assert calls[0][2] == "venv --allow-existing --python 3.12 .venv-runtime"
    if len(calls) > 1:
        backend = (
            "cpu"
            if case == "cpu"
            else "chosen-backend"
            if case == "custom_backend"
            else "auto"
        )
        assert calls[1][2].replace('"', "") == (
            f"pip install --python .venv-runtime --torch-backend {backend} "
            "--upgrade-package torch --upgrade-package torchvision --editable ."
        )
    if len(calls) > 2:
        assert calls[2][2] == "run --no-sync video2tenhou web --port 8898 --no-browser"


def _windows_tools(
    checkout: Path, tools: Path, case: str, env: dict[str, str]
) -> list[str]:
    """Install platform tool fixtures and construct the launcher command."""
    launcher = "Start.cmd"
    tmp_path = tools.parent
    env["PATH"] = (
        str(tools) + os.pathsep + str(Path(os.environ["SYSTEMROOT"]) / "System32")
    )
    env["ProgramFiles"] = str(tmp_path / "no installed fallback")
    env["USERPROFILE"] = str(tmp_path / "no user fallback")
    env["LOCALAPPDATA"] = str(tmp_path / "no winget fallback")
    uv = tools / "uv.cmd"
    uv.write_text(
        '@echo off\n>>"%LAUNCH_LOG%" echo %CD%^|%VIDEO2TENHOU_HOME%^|%*\n'
        'if not "%UV_PROJECT_ENVIRONMENT%"==".venv-runtime" exit /b 81\n'
        'if "%1"=="venv" exit /b %VENV_EXIT%\n'
        'if "%1"=="pip" exit /b %SETUP_EXIT%\n'
        "exit /b %RUN_EXIT%\n"
    )
    for tool in ("ffmpeg", "ffprobe"):
        if tool == "ffprobe" and case == "missing_tool":
            continue
        (tools / (tool + ".cmd")).write_text("@exit /b 0\n")
    return [
        os.environ.get("COMSPEC", "cmd.exe"),
        "/d",
        "/c",
        str(checkout / launcher),
        "--port",
        "8898",
        "--no-browser",
    ]


def _posix_tools(
    checkout: Path, tools: Path, case: str, env: dict[str, str]
) -> list[str]:
    """Install platform tool fixtures and construct the launcher command."""
    launcher = "start.sh"
    env["PATH"] = str(tools)
    for tool in ("dirname",):
        executable = shutil.which(tool)
        assert executable, f"POSIX launcher requires the standard {tool} utility"
        (tools / tool).symlink_to(executable)
    uv = tools / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "%s|%s|%s\\n" "$PWD" "$VIDEO2TENHOU_HOME" "$*" '
        '>> "$LAUNCH_LOG"\nif [ "$UV_PROJECT_ENVIRONMENT" != .venv-runtime '
        ']; then exit 81; fi\nif [ "$1" = venv ]; then exit "$VENV_EXIT"; '
        'fi\nif [ "$1" = pip ]; then exit "$SETUP_EXIT"; fi\nexit '
        '"$RUN_EXIT"\n'
    )
    uv.chmod(0o755)
    for tool in ("ffmpeg", "ffprobe"):
        if tool == "ffprobe" and case == "missing_tool":
            continue
        executable = tools / tool
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    return [
        "/bin/sh",
        str(checkout / launcher),
        "--port",
        "8898",
        "--no-browser",
    ]
