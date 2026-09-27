"""Exercise source launchers with local tools, without installing dependencies."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.paths import ROOT


@pytest.mark.parametrize("case,expected", [("success", 0), ("sync_failure", 7),
                                            ("runtime_failure", 9), ("missing_tool", 1),
                                            ("missing_english", 1)])
def test_launcher_paths_arguments_and_failure_boundaries(tmp_path, case, expected):
    checkout = tmp_path / "starter with spaces"
    tools = tmp_path / "stub tools"
    checkout.mkdir()
    tools.mkdir()
    log = tmp_path / "launch.log"
    data_home = tmp_path / "separate data directory"
    windows = os.name == "nt"
    launcher = "Start.cmd" if windows else "start.sh"
    shutil.copyfile(ROOT / launcher, checkout / launcher)
    env = {**os.environ, "LAUNCH_LOG": str(log), "VIDEO2TENHOU_HOME": str(data_home),
           "SYNC_EXIT": "7" if case == "sync_failure" else "0",
           "RUN_EXIT": "9" if case == "runtime_failure" else "0"}
    if windows:
        env["PATH"] = str(tools) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32")
        env["ProgramFiles"] = str(tmp_path / "no installed fallback")
        uv = tools / "uv.cmd"
        uv.write_text('@echo off\n>>"%LAUNCH_LOG%" echo %CD%^|%VIDEO2TENHOU_HOME%^|%*\n'
                      'if "%1"=="sync" exit /b %SYNC_EXIT%\nexit /b %RUN_EXIT%\n')
        for tool in ("ffmpeg", "ffprobe", "tesseract"):
            if tool == "ffprobe" and case == "missing_tool":
                continue
            language = "@echo eng\n" if tool == "tesseract" and case != "missing_english" else ""
            (tools / (tool + ".cmd")).write_text(language + "@exit /b 0\n")
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(checkout / launcher),
                   "--port", "8898", "--no-browser"]
    else:
        env["PATH"] = str(tools)
        for tool in ("dirname", "grep"):
            executable = shutil.which(tool)
            assert executable, f"POSIX launcher requires the standard {tool} utility"
            (tools / tool).symlink_to(executable)
        uv = tools / "uv"
        uv.write_text('#!/bin/sh\nprintf "%s|%s|%s\\n" "$PWD" "$VIDEO2TENHOU_HOME" "$*" >> "$LAUNCH_LOG"\n'
                      'if [ "$1" = sync ]; then exit "$SYNC_EXIT"; fi\nexit "$RUN_EXIT"\n')
        uv.chmod(0o755)
        for tool in ("ffmpeg", "ffprobe", "tesseract"):
            if tool == "ffprobe" and case == "missing_tool":
                continue
            executable = tools / tool
            language = "printf 'eng\\n'\n" if tool == "tesseract" and case != "missing_english" else ""
            executable.write_text("#!/bin/sh\n" + language + "exit 0\n")
            executable.chmod(0o755)
        command = ["/bin/sh", str(checkout / launcher), "--port", "8898", "--no-browser"]
    result = subprocess.run(command, cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == expected, result.stdout + result.stderr
    if case in ("missing_tool", "missing_english"):
        assert ("ffprobe" if case == "missing_tool" else "English") in result.stdout + result.stderr
        assert "QUICKSTART.md" in result.stdout + result.stderr
        assert not log.exists()
        return
    calls = [line.split("|", 2) for line in log.read_text().splitlines()]
    assert all(Path(cwd).resolve() == checkout.resolve() and home == str(data_home)
               for cwd, home, _ in calls)
    assert calls[0][2] == "sync --frozen --no-dev"
    assert len(calls) == (1 if case == "sync_failure" else 2)
    if len(calls) == 2:
        assert calls[1][2] == "run --no-sync video2tenhou web --port 8898 --no-browser"
