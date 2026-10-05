# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise source launchers with stub tools, without installing dependencies."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.paths import ROOT

WINDOWS = os.name == "nt"
APP_ARGS = ("--port", "8898", "--no-browser")
VENV = "venv --allow-existing --python 3.12 .venv-runtime"
RUN = "run --no-sync video2tenhou web --port 8898 --no-browser"
UPGRADE = "--upgrade-package torch --upgrade-package torchvision"
WINDOWS_UV = """@echo off
>>"%LAUNCH_LOG%" echo %CD%^|%VIDEO2TENHOU_HOME%^|%*
if not "%UV_PROJECT_ENVIRONMENT%"==".venv-runtime" exit /b 81
if not "%1"=="venv" goto other
if not "%VENV_EXIT%"=="0" exit /b %VENV_EXIT%
if not exist .venv-runtime mkdir .venv-runtime
exit /b 0
:other
if "%1"=="pip" exit /b %SETUP_EXIT%
exit /b %RUN_EXIT%
"""
POSIX_UV = """#!/bin/sh
printf '%s|%s|%s\\n' "$PWD" "$VIDEO2TENHOU_HOME" "$*" >> "$LAUNCH_LOG"
[ "$UV_PROJECT_ENVIRONMENT" = .venv-runtime ] || exit 81
case "$1" in
venv) [ "$VENV_EXIT" = 0 ] || exit "$VENV_EXIT"; mkdir -p .venv-runtime ;;
pip) exit "$SETUP_EXIT" ;;
*) exit "$RUN_EXIT" ;;
esac
"""


def install(backend: str = "auto", *, upgrade: bool = False) -> str:
    """Return the expected runtime installation command."""
    options = f" {UPGRADE}" if upgrade else ""
    return (
        f"pip install --python .venv-runtime --torch-backend {backend}"
        f"{options} --editable ."
    )


@dataclass
class Launch:
    """Exit status, console output and the uv commands one launch issued."""

    returncode: int
    output: str
    calls: list[str]


class Starter:
    """A copied launcher beside stub uv/FFmpeg tools in paths containing spaces."""

    def __init__(self, root: Path, *, missing_tool: bool = False) -> None:
        """Copy the platform launcher and install stub tools on an isolated PATH."""
        self.checkout = root / "starter with spaces"
        self.tools = root / "stub tools"
        self.log = root / "launch.log"
        self.home = root / "separate data directory"
        self.checkout.mkdir()
        self.tools.mkdir()
        self.launcher = self.checkout / ("Start.cmd" if WINDOWS else "start.sh")
        shutil.copyfile(ROOT / self.launcher.name, self.launcher)
        self.project('[project]\nname = "video2tenhou"\n')
        self.env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {"UV_TORCH_BACKEND", "VIDEO2TENHOU_DEVICE", "VIDEO2TENHOU_UPDATE"}
        }
        self.env.update(LAUNCH_LOG=str(self.log), VIDEO2TENHOU_HOME=str(self.home))
        tools = ["ffmpeg"] if missing_tool else ["ffmpeg", "ffprobe"]
        if WINDOWS:
            self._windows_tools(tools)
        else:
            self._posix_tools(tools)

    def project(self, text: str) -> None:
        """Write the checkout's project file, which identifies the installed runtime."""
        (self.checkout / "pyproject.toml").write_text(text)

    @property
    def installed(self) -> bool:
        """Report whether the launcher recorded a completed runtime installation."""
        return (self.checkout / ".venv-runtime/video2tenhou-pyproject.toml").exists()

    def launch(self, *args: str, **env: str) -> Launch:
        """Run the launcher with application arguments and uv exit-code overrides."""
        self.log.unlink(missing_ok=True)
        environment = {
            **self.env,
            "SETUP_EXIT": "0",
            "VENV_EXIT": "0",
            "RUN_EXIT": "0",
            **env,
        }
        command = (
            [environment.get("COMSPEC", "cmd.exe"), "/d", "/c", str(self.launcher)]
            if WINDOWS
            else ["/bin/sh", str(self.launcher)]
        )
        result = subprocess.run(  # noqa: S603
            [*command, *args],
            cwd=self.checkout.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        rows = (
            [line.split("|", 2) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )
        assert all(
            Path(cwd).resolve() == self.checkout.resolve() and home == str(self.home)
            for cwd, home, _ in rows
        )
        return Launch(
            result.returncode,
            result.stdout + result.stderr,
            [" ".join(command.replace('"', "").split()) for _, _, command in rows],
        )

    def _windows_tools(self, tools: list[str]) -> None:
        root = self.checkout.parent
        system = Path(os.environ["SYSTEMROOT"], "System32")
        self.env.update(
            PATH=f"{self.tools}{os.pathsep}{system}",
            ProgramFiles=str(root / "no installed fallback"),
            USERPROFILE=str(root / "no user fallback"),
            LOCALAPPDATA=str(root / "no winget fallback"),
        )
        (self.tools / "uv.cmd").write_text(WINDOWS_UV)
        for tool in tools:
            (self.tools / f"{tool}.cmd").write_text("@exit /b 0\n")

    def _posix_tools(self, tools: list[str]) -> None:
        self.env["PATH"] = str(self.tools)
        for utility in ("dirname", "cmp", "cp", "rm", "mkdir"):
            executable = shutil.which(utility)
            assert executable, f"POSIX launcher tests require the {utility} utility"
            (self.tools / utility).symlink_to(executable)
        scripts = {"uv": POSIX_UV} | dict.fromkeys(tools, "#!/bin/sh\n")
        for name, text in scripts.items():
            (self.tools / name).write_text(text)
            (self.tools / name).chmod(0o755)


@pytest.fixture
def starter(tmp_path: Path) -> Starter:
    """Provide a fresh starter folder without a runtime environment."""
    return Starter(tmp_path)


def test_first_start_installs_and_later_starts_reuse_the_runtime(
    starter: Starter,
) -> None:
    """Later launches start offline without re-resolving PyTorch."""
    first = starter.launch(*APP_ARGS)
    assert first.returncode == 0, first.output
    assert first.calls == [VENV, install(), RUN]
    assert starter.installed
    second = starter.launch(*APP_ARGS)
    assert second.returncode == 0, second.output
    assert second.calls == [RUN]


def test_changed_project_reinstalls_without_upgrading_pytorch(
    starter: Starter,
) -> None:
    """New dependencies are installed while a satisfying PyTorch is kept."""
    assert starter.launch(*APP_ARGS).returncode == 0
    starter.project('[project]\nname = "video2tenhou"\nversion = "9"\n')
    assert starter.launch(*APP_ARGS).calls == [VENV, install(), RUN]


@pytest.mark.parametrize(
    ("args", "env"), [(("--update",), {}), ((), {"VIDEO2TENHOU_UPDATE": "1"})]
)
def test_explicit_update_refreshes_pytorch_and_keeps_application_arguments(
    starter: Starter, args: tuple[str, ...], env: dict[str, str]
) -> None:
    """Only a requested update upgrades PyTorch; the flag is not passed on."""
    assert starter.launch(*APP_ARGS).returncode == 0
    update = starter.launch(*args, *APP_ARGS, **env)
    assert update.returncode == 0, update.output
    assert update.calls == [VENV, install(upgrade=True), RUN]
    assert starter.launch(*APP_ARGS).calls == [RUN]


@pytest.mark.parametrize(
    ("env", "backend"),
    [
        ({"VIDEO2TENHOU_DEVICE": "cpu"}, "cpu"),
        ({"UV_TORCH_BACKEND": "chosen-backend"}, "chosen-backend"),
    ],
)
def test_installation_honors_backend_choice(
    starter: Starter, env: dict[str, str], backend: str
) -> None:
    """CPU mode and an explicit uv backend select the PyTorch build."""
    result = starter.launch(*APP_ARGS, **env)
    assert result.returncode == 0, result.output
    assert result.calls == [VENV, install(backend), RUN]


@pytest.mark.parametrize(
    ("env", "expected", "calls", "installed"),
    [
        ({"VENV_EXIT": "6"}, 6, [VENV], False),
        ({"SETUP_EXIT": "7"}, 7, [VENV, install()], False),
        ({"RUN_EXIT": "9"}, 9, [VENV, install(), RUN], True),
    ],
)
def test_failures_keep_exit_status_and_retry_incomplete_setup(
    starter: Starter,
    env: dict[str, str],
    expected: int,
    calls: list[str],
    *,
    installed: bool,
) -> None:
    """A failed setup is never recorded as installed, so relaunching retries it."""
    result = starter.launch(*APP_ARGS, **env)
    assert result.returncode == expected, result.output
    assert result.calls == calls
    assert starter.installed is installed
    retry = starter.launch(*APP_ARGS)
    assert retry.calls == ([RUN] if installed else [VENV, install(), RUN])


def test_missing_prerequisite_stops_before_setup(tmp_path: Path) -> None:
    """Missing tools are named with setup guidance before uv runs."""
    result = Starter(tmp_path, missing_tool=True).launch(*APP_ARGS)
    assert result.returncode == 1
    assert "ffprobe" in result.output
    assert "QUICKSTART.md" in result.output
    assert not result.calls
