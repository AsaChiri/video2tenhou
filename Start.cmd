@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed

where tesseract >nul 2>nul
if errorlevel 1 if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" set "PATH=%ProgramFiles%\Tesseract-OCR;%PATH%"

set "missing=0"
for %%T in (uv ffmpeg ffprobe tesseract) do (
    where %%T >nul 2>nul
    if errorlevel 1 (
        echo Missing prerequisite: %%T
        set "missing=1"
    )
)
if "%missing%"=="1" goto prerequisites
call tesseract --list-langs 2>nul | findstr /x /c:"eng" >nul
if errorlevel 1 (
    echo Missing Tesseract English language data: eng
    goto prerequisites
)

echo Preparing video2tenhou. The first launch downloads Python dependencies.
call uv sync --frozen --no-dev
if errorlevel 1 goto failed
echo Opening the studio. Leave this window open while using the application.
call uv run --no-sync video2tenhou web %*
if errorlevel 1 goto failed
exit /b 0

:prerequisites
echo Install the missing prerequisites, then run Start.cmd again.
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
