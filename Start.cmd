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

rem Install only when the runtime is missing, incomplete or older than pyproject.toml,
rem or when asked to: a new PyTorch build invalidates every recognition cache.
set "app_args=%*"
set "update=0"
if "%VIDEO2TENHOU_UPDATE%"=="1" set "update=1"
if /i "%~1"=="--update" set "update=1"
if /i "%~1"=="--update" set "app_args=%app_args:*--update=%"
set "UV_PROJECT_ENVIRONMENT=.venv-runtime"
set "installed=.venv-runtime\video2tenhou-pyproject.toml"
if "%update%"=="0" if exist "%installed%" fc /b pyproject.toml "%installed%" >nul 2>nul && goto launch

echo Preparing video2tenhou for this computer. Leave this window open.
if not defined UV_TORCH_BACKEND set "UV_TORCH_BACKEND=auto"
if /i "%VIDEO2TENHOU_DEVICE%"=="cpu" set "UV_TORCH_BACKEND=cpu"
set "upgrade="
if "%update%"=="1" set "upgrade=--upgrade-package torch --upgrade-package torchvision"
if exist "%installed%" del "%installed%"
call uv venv --allow-existing --python 3.12 .venv-runtime
if errorlevel 1 goto failed
call uv pip install --python .venv-runtime --torch-backend "%UV_TORCH_BACKEND%" %upgrade% --editable .
if errorlevel 1 goto failed
copy /y pyproject.toml "%installed%" >nul
if errorlevel 1 goto failed

:launch
call uv run --no-sync video2tenhou web %app_args%
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
