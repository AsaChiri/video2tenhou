# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Download recordings, sample video through an ffmpeg pipe and seek single frames.

Stages that read a whole video use `sample`, one sequential decode with no seeking; the
tool and evidence crops use `frame_at`. Every frame is returned as a 1080p BGR array,
whatever the source size.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from typing import IO, TYPE_CHECKING, cast

import numpy as np

from video2tenhou.commands import executable

if TYPE_CHECKING:
    from collections.abc import Generator

FRAME_W, FRAME_H = 1920, 1080
# Decoded frames buffered ahead of the consumer: pipes are small on Windows, so
# without a reader ffmpeg waits while each frame is processed.
READ_AHEAD = 16


@dataclass(frozen=True)
class VideoInfo:
    """Source dimensions, nominal frame rate and duration reported by ffprobe."""

    path: str
    width: int
    height: int
    fps: float
    duration: float


def probe(path: str | Path) -> VideoInfo:
    """Inspect a local video's first video stream; propagate invalid-file errors."""
    out = subprocess.run(  # noqa: S603
        [
            executable("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return VideoInfo(
        str(path),
        int(s["width"]),
        int(s["height"]),
        float(num) / float(den),
        float(j["format"]["duration"]),
    )


def sample(
    path: str | Path,
    fps: float = 2.0,
    start: float = 0.0,
    end: float | None = None,
    size: tuple[int, int] = (FRAME_W, FRAME_H),
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Yield (t, frame) at `fps` from `start` to `end`, decoded sequentially by ffmpeg.

    Frames are scaled to `size` by ffmpeg. `t` is a nominal sampling-grid label
    (start + k / fps), not the decoded frame's presentation timestamp. FFmpeg
    quantizes the requested duration; filtering these labels from a longer
    sample does not necessarily reproduce a separately decoded shorter window.
    A reader thread keeps up to ``READ_AHEAD`` frames decoded ahead. Closing the
    generator stops ffmpeg and joins the reader; decoder failures raise.
    """
    w, h = size
    if fps <= 0 or w <= 0 or h <= 0 or start < 0:
        raise ValueError(
            "fps and dimensions must be positive; start must be nonnegative"
        )
    if end is not None and end <= start:
        return
    cmd = [executable("ffmpeg"), "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{max(0.0, end - start):.3f}"]
    cmd += [
        "-vf",
        f"fps={fps},scale={w}:{h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    with (
        tempfile.TemporaryFile() as errors,
        subprocess.Popen(  # noqa: S603
            cmd, stdout=subprocess.PIPE, stderr=errors, bufsize=w * h * 3 * 4
        ) as proc,
        ThreadPoolExecutor(1, thread_name_prefix="video-read-ahead") as pool,
    ):
        stdout = cast("IO[bytes]", proc.stdout)
        frames: Queue[bytes | None] = Queue(READ_AHEAD)
        stop = Event()
        reader = pool.submit(_read_frames, stdout, w * h * 3, frames, stop)
        try:
            k = 0
            while (buf := frames.get()) is not None:
                yield start + k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                k += 1
            reader.result()
            returncode = proc.wait()
            if returncode:
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg could not sample {path}: {detail}")
        finally:
            stop.set()
            if proc.poll() is None:
                proc.terminate()
            while not reader.done():  # a full queue may block the reader's put
                with suppress(Empty):
                    frames.get(timeout=0.1)
            stdout.close()


def _read_frames(
    stdout: IO[bytes], size: int, frames: Queue[bytes | None], stop: Event
) -> None:
    """Queue complete raw frames until end of stream or ``stop``, then ``None``."""
    try:
        while not stop.is_set():
            buf = stdout.read(size)
            if len(buf) < size:
                return
            frames.put(buf)
    finally:
        frames.put(None)


def frame_at(
    path: str | Path, t: float, size: tuple[int, int] = (FRAME_W, FRAME_H)
) -> np.ndarray:
    """One frame at time t (seconds), scaled to `size`."""
    w, h = size
    out = subprocess.run(  # noqa: S603
        [
            executable("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={w}:{h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    if len(out) < w * h * 3:
        raise ValueError(f"no frame at t={t} in {path}")
    return np.frombuffer(out[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def time_range(
    start: str | float | None = None, end: str | float | None = None
) -> tuple[float, float | None]:
    """Normalize optional seconds or MM:SS/HH:MM:SS bounds with yt-dlp's parser."""
    from yt_dlp.utils import parse_duration  # noqa: PLC0415  a 0.3 s import

    values = []
    for label, supplied, default in (("Start", start, 0.0), ("End", end, None)):
        value = supplied
        if value is None or (isinstance(value, str) and not value.strip()):
            values.append(default)
            continue
        try:
            if isinstance(value, str):
                value = value.strip()
                # Restrict the UI format; parse_duration also accepts days and named
                # units.
                value = (
                    parse_duration(value)
                    if re.fullmatch(r"\d+(?::[0-5]?\d){0,2}(?:\.\d+)?", value)
                    else None
                )
            valid = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and value >= 0
            )
        except (ValueError, OverflowError):
            valid = False
        if not valid or not isinstance(value, (int, float)):
            raise ValueError(
                f"{label} time must be nonnegative seconds, MM:SS, or HH:MM:SS "
                "(fractional seconds are allowed)."
            )
        values.append(float(value))
    start, end = values[0] or 0.0, values[1]
    if end is not None and end <= start:
        raise ValueError("End time must be later than start time.")
    return start, end


def format_time(seconds: float) -> str:
    """Format normalized seconds without exponent notation or rounding precision."""
    return format(Decimal(str(seconds)), "f")


def trim(
    source: str | Path,
    out: str | Path,
    start: str | float | None = None,
    end: str | float | None = None,
) -> Path:
    """Create a separate accurately cut, browser-playable MP4 without resizing.

    High-quality H.264 re-encoding preserves source dimensions and frame timing;
    the original recording remains untouched. Publish only the completed clip.
    """
    start, end = time_range(start, end)
    source, out = Path(source).resolve(), Path(out)
    if source == out.resolve() or (out.exists() and source.samefile(out)):
        raise ValueError(
            "The trimmed recording must use a different file from the original."
        )
    info = probe(source)
    if not math.isfinite(info.duration) or info.duration <= 0:
        raise ValueError("The recording has no valid duration.")
    if start >= info.duration or (end is not None and end > info.duration):
        raise ValueError(
            f"Start and end times must be within the recording's {info.duration:g} "
            "seconds."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".trim-") as directory:
        pending = Path(directory) / "clip.mp4"
        cmd = [
            executable("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            format_time(start),
            "-i",
            str(source),
        ]
        if end is not None:
            cmd += ["-t", format_time(end - start)]
        cmd += [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(pending),
        ]
        subprocess.run(cmd, check=True)  # noqa: S603
        if probe(pending).duration <= 0:
            raise ValueError("The selected time range contains no video frames.")
        pending.replace(out)
    return out


def download(
    url: str,
    out: str | Path,
    start: str | float | None = None,
    end: str | float | None = None,
) -> Path:
    """Download a complete 1080p recording or accurately cut section."""
    start, end = time_range(start, end)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".download-") as directory:
        pending = Path(directory) / "video.mp4"
        # Direct video URLs may omit height metadata; let yt-dlp accept those too.
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "bestvideo[height<=?1080]+bestaudio/best[height<=?1080]",
            "--merge-output-format",
            "mp4",
            "--remux-video",
            "mp4",
            "-o",
            str(pending),
        ]
        if start or end is not None:
            sec = (
                f"*{format_time(start)}-"
                f"{(format_time(end) if end is not None else 'inf')}"
            )
            cmd += ["--download-sections", sec, "--force-keyframes-at-cuts"]
        cmd += ["--", url]
        subprocess.run(cmd, check=True)  # noqa: S603
        if not pending.is_file() or not pending.stat().st_size:
            raise ValueError("The download did not produce a complete video.")
        pending.replace(out)
    return out
