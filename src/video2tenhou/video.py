"""Frame access: sequential sampling through an ffmpeg pipe, seeking for single frames, download.

Stages that read a whole video use `sample`, one sequential decode with no
seeking; the tool and evidence crops use `frame_at`. Every frame is returned
as a 1080p BGR array, whatever the source size.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

FRAME_W, FRAME_H = 1920, 1080


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
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return VideoInfo(str(path), int(s["width"]), int(s["height"]), float(num) / float(den), float(j["format"]["duration"]))


def sample(path: str | Path, fps: float = 2.0, start: float = 0.0, end: Optional[float] = None,
           size: tuple[int, int] = (FRAME_W, FRAME_H)) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t, frame) at `fps` from `start` to `end`, decoded sequentially by ffmpeg.

    Frames are scaled to `size` by ffmpeg. `t` is a nominal sampling-grid label
    (start + k / fps), not the decoded frame's presentation timestamp. FFmpeg
    quantizes the requested duration; filtering these labels from a longer
    sample does not necessarily reproduce a separately decoded shorter window.
    """
    w, h = size
    if fps <= 0 or w <= 0 or h <= 0 or start < 0:
        raise ValueError("fps and dimensions must be positive; start must be nonnegative")
    if end is not None and end <= start:
        return
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{max(0.0, end - start):.3f}"]
    cmd += ["-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    errors = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors, bufsize=w * h * 3 * 4)
    assert proc.stdout is not None
    n = w * h * 3
    k = 0
    finished = False
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield start + k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
        returncode = proc.wait()
        finished = True
        if returncode:
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg could not sample {path}: {detail}")
    finally:
        proc.stdout.close()
        if not finished and proc.poll() is None:
            proc.terminate()
        proc.wait()
        errors.close()


def frame_at(path: str | Path, t: float, size: tuple[int, int] = (FRAME_W, FRAME_H)) -> np.ndarray:
    """One frame at time t (seconds), scaled to `size`."""
    w, h = size
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True).stdout
    if len(out) < w * h * 3:
        raise ValueError(f"no frame at t={t} in {path}")
    return np.frombuffer(out[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def normalize(frame: np.ndarray, size: tuple[int, int] = (FRAME_W, FRAME_H)) -> np.ndarray:
    """Scale an arbitrary frame to the calibration size."""
    if frame.shape[1] == size[0] and frame.shape[0] == size[1]:
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA if frame.shape[1] > size[0] else cv2.INTER_CUBIC)


def download(url: str, out: str | Path, start: Optional[str] = None, end: Optional[str] = None) -> Path:
    """Download a Twitch VOD (or a section of it) at the best 1080p quality with yt-dlp."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["uvx", "yt-dlp", "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]", "--merge-output-format", "mp4",
           "-o", str(out), url]
    if start or end:
        sec = f"*{start or '0'}-{end or 'inf'}"
        cmd[2:2] = ["--download-sections", sec]
    subprocess.run(cmd, check=True)
    return out
