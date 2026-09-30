@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed

if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"

set "missing=0"
for %%T in (uv ffmpeg ffprobe) do (
    where %%T >nul 2>nul
    if errorlevel 1 (
        echo Missing prerequisite: %%T
        set "missing=1"
    )
)
if "%missing%"=="1" goto prerequisites

echo Checking this computer and preparing video2tenhou. Leave this window open.
set "UV_PROJECT_ENVIRONMENT=.venv-runtime"
if not defined UV_TORCH_BACKEND set "UV_TORCH_BACKEND=auto"
if /i "%VIDEO2TENHOU_DEVICE%"=="cpu" set "UV_TORCH_BACKEND=cpu"
call uv venv --allow-existing --python 3.12 .venv-runtime
if errorlevel 1 goto failed
call uv pip install --python .venv-runtime --torch-backend "%UV_TORCH_BACKEND%" --upgrade-package torch --upgrade-package torchvision --editable .
if errorlevel 1 goto failed
call uv run --no-sync video2tenhou web %*
if errorlevel 1 goto failed
exit /b 0

:prerequisites
echo Install the missing prerequisites, then run Start.cmd again.
echo In PowerShell, install only the missing tools:
echo   winget install --id astral-sh.uv --exact
echo   winget install "FFmpeg (Essentials Build)"
echo See docs\QUICKSTART.md or:
echo https://github.com/AsaChiri/video2tenhou/blob/main/docs/QUICKSTART.md
pause
exit /b 1

:failed
set "launch_error=%errorlevel%"
echo.
echo The application could not start or stopped with an error.
echo See the message above and docs\QUICKSTART.md for setup instructions.
pause
exit /b %launch_error%
