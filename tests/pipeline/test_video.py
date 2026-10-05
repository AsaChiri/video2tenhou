# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Video metadata, frame sampling and bounded recording extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

from tests.spies import record_results
from video2tenhou import video
from video2tenhou.cli import main
from video2tenhou.commands import executable

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a small encoded video for real decoder integration tests."""
    p = tmp_path_factory.mktemp("v") / "t.mp4"
    subprocess.run(  # noqa: S603
        [
            executable("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=30",
            "-t",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(p),
        ],
        check=True,
    )
    return p


def test_probe(clip: Path) -> None:
    info = video.probe(clip)
    assert (info.width, info.height) == (320, 180)
    assert abs(info.fps - 30) < 1e-6
    assert abs(info.duration - 2.0) < 0.1


def test_sample_and_seek(clip: Path) -> None:
    frames = list(video.sample(clip, fps=2.0))
    assert len(frames) == 4
    t, f = frames[1]
    assert t == 0.5
    assert f.shape == (1080, 1920, 3)
    part = list(video.sample(clip, fps=2.0, start=1.0, end=2.0))
    assert [round(t, 2) for t, _ in part] == [1.0, 1.5]
    one = video.frame_at(clip, 1.0)
    assert one.shape == (1080, 1920, 3)


def test_failed_decode_is_reported_instead_of_an_empty_success(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="ffmpeg could not sample"):
        list(video.sample(tmp_path / "missing.mp4"))


def test_sample_can_be_closed_early_without_waiting_for_the_entire_clip(
    clip: Path,
) -> None:
    frames = video.sample(clip, fps=30)
    assert next(frames)[0] == 0
    frames.close()


def test_closing_a_sample_stops_the_decoder_and_joins_its_reader(
    clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frames decoded ahead never outlive the generator that owns them."""
    children = []
    monkeypatch.setattr(
        video.subprocess, "Popen", record_results(subprocess.Popen, children)
    )
    frames = video.sample(clip, fps=30)
    next(frames)
    frames.close()
    assert children[0].poll() is not None
    assert not any(t.name.startswith("video-read-ahead") for t in threading.enumerate())


def test_empty_window_does_not_start_a_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video.subprocess,
        "Popen",
        lambda *_unused_a, **_unused_k: pytest.fail("unexpected decode"),
    )
    assert list(video.sample("unused.mp4", start=1.0, end=1.0)) == []


@pytest.mark.parametrize(
    "source", ["https://www.youtube.com/watch?v=example&feature=share", "--version"]
)
def test_download_cli_passes_source_as_literal_to_ytdlp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    commands = []

    def download(cmd: list[str], **kwargs: object) -> None:
        commands.append((cmd, kwargs))
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"complete video")

    monkeypatch.setattr(video.subprocess, "run", download)
    output = tmp_path / "video.mp4"
    main(["download", "--", source, str(output)])
    command, kwargs = commands[0]
    assert command[:3] == [video.sys.executable, "-m", "yt_dlp"]
    assert command[-2:] == ["--", source]
    assert Path(command[command.index("-o") + 1]) != output
    assert output.read_bytes() == b"complete video"
    assert kwargs["check"] is True


def test_download_section_uses_installed_ytdlp_and_preserves_literal_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = []

    def download(command: list[str], **_unused_kwargs: object) -> None:
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"complete section")

    monkeypatch.setattr(video.subprocess, "run", download)
    video.download("--version", tmp_path / "clip.mp4", start="00:01:00", end="00:02:00")
    assert commands[0][:3] == [video.sys.executable, "-m", "yt_dlp"]
    assert commands[0][-5:] == [
        "--download-sections",
        "*60.0-120.0",
        "--force-keyframes-at-cuts",
        "--",
        "--version",
    ]


def test_download_propagates_ytdlp_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"previous complete video")

    def fail(cmd: list[str], **_unused_kwargs: object) -> None:
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"incomplete download")
        assert target.read_bytes() == b"previous complete video"
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(video.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        video.download("https://unsupported.test/video", target)
    assert target.read_bytes() == b"previous complete video"
    assert not list(tmp_path.glob(".download-*"))


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (None, None, (0.0, None)),
        (" ", "", (0.0, None)),
        ("12.25", None, (12.25, None)),
        ("01:02.5", "1:02:03.75", (62.5, 3723.75)),
        (0, 1.125, (0.0, 1.125)),
    ],
)
def test_time_range_formats(
    start: int | str | None,
    end: float | str | None,
    expected: tuple[float, None] | tuple[float, float],
) -> None:
    assert video.time_range(start, end) == expected


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, None),
        ("-1", None),
        ("00:-01", None),
        ("00:60", None),
        ("1:60:00", None),
        ("one minute", None),
        ("1:2:3:4", None),
        (float("nan"), None),
        (0, float("inf")),
        (True, None),
        ([], None),
        ("5", "5"),
        ("10", "9"),
        (None, "0"),
    ],
)
def test_invalid_ranges_are_rejected_before_any_processing(
    start: str | float | None, end: str | float | None
) -> None:
    with pytest.raises(ValueError, match="time"):
        video.time_range(start, end)


@pytest.mark.parametrize(
    "value", [10**400, "9" * 400], ids=["enormous-integer", "enormous-string"]
)
@pytest.mark.parametrize("bound", ["start", "end"])
def test_overflowing_times_are_validation_errors(value: str | int, bound: str) -> None:
    with pytest.raises(ValueError, match="time must be"):
        video.time_range(**{bound: value})


def test_local_trim_cuts_between_keyframes_without_resizing_or_changing_rate(
    clip: Path, tmp_path: Path
) -> None:
    original = clip.read_bytes()
    output = video.trim(clip, tmp_path / "trimmed.mp4", "0.7", "1.4")
    info = video.probe(output)
    assert (info.width, info.height, info.fps) == (320, 180, 30.0)
    assert abs(info.duration - 0.7) < 1 / 30 + 0.01
    for offset in (0.0, 0.3):
        actual = video.frame_at(output, offset, size=(320, 180)).astype(float)
        expected = video.frame_at(clip, 0.7 + offset, size=(320, 180)).astype(float)
        assert np.abs(actual - expected).mean() < 2
    profile = json.loads(
        subprocess.run(  # noqa: S603
            [
                executable("ffprobe"),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=profile,pix_fmt",
                "-of",
                "json",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert profile["streams"][0]["profile"] in ("High", "Main", "Constrained Baseline")
    assert profile["streams"][0]["pix_fmt"] == "yuv420p"
    assert clip.read_bytes() == original
    assert not list(tmp_path.glob(".trim-*"))


@pytest.mark.parametrize(
    ("start", "end", "duration"),
    [("0.5", None, 1.5), (None, "1.2", 1.2), (None, None, 2.0)],
)
def test_local_trim_optional_bounds(
    clip: Path, tmp_path: Path, start: str | None, end: str | None, duration: float
) -> None:
    output = video.trim(clip, tmp_path / "trimmed.mp4", start, end)
    assert abs(video.probe(output).duration - duration) < 0.05


@pytest.mark.parametrize(("start", "end"), [("2", None), (None, "2.1"), ("9", "10")])
def test_local_trim_rejects_bounds_outside_recording(
    clip: Path, tmp_path: Path, start: str | None, end: str | None
) -> None:
    output = tmp_path / "trimmed.mp4"
    with pytest.raises(ValueError, match="within the recording"):
        video.trim(clip, output, start, end)
    assert not output.exists()


def test_local_trim_never_overwrites_original(clip: Path) -> None:
    with pytest.raises(ValueError, match="different file"):
        video.trim(clip, clip, 0, 1)


def test_trim_failure_preserves_existing_output_and_cleans_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "trimmed.mp4"
    source.write_bytes(b"original")
    output.write_bytes(b"complete")
    monkeypatch.setattr(
        video, "probe", lambda _: video.VideoInfo(str(source), 320, 180, 30.0, 2.0)
    )

    def fail(command: list[str], **_unused_kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(video.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        video.trim(source, output, 0, 1)
    assert source.read_bytes() == b"original"
    assert output.read_bytes() == b"complete"
    assert not list(tmp_path.glob(".trim-*"))


def test_trim_cli_passes_literal_paths_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(video, "trim", lambda *args: calls.append(args))
    main(
        [
            "trim",
            "--start",
            "00:00.5",
            "--end",
            "1.5",
            "--",
            "--source.mp4",
            "out file.mp4",
        ]
    )
    assert calls == [("--source.mp4", "out file.mp4", "00:00.5", "1.5")]


def test_trim_cli_accepts_normalized_submillisecond_bounds(
    clip: Path, tmp_path: Path
) -> None:
    output = tmp_path / "tiny-start.mp4"
    assert video.format_time(1e-5) == "0.00001"
    main(
        [
            "trim",
            "--start",
            video.format_time(1e-5),
            "--end",
            video.format_time(1.00001),
            "--",
            str(clip),
            str(output),
        ]
    )
    assert abs(video.probe(output).duration - 1) < 0.05


@pytest.mark.parametrize(
    ("start", "end", "section"),
    [(None, "12.75", "*0.0-12.75"), ("1.23456789", None, "*1.23456789-inf")],
)
def test_download_optional_bounds_are_normalized_without_losing_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: str | None,
    end: str | None,
    section: str,
) -> None:
    def run(command: list[str], **_unused_kwargs: object) -> None:
        actual_start, actual_end = (
            command[command.index("--download-sections") + 1]
            .removeprefix("*")
            .split("-")
        )
        expected_start, expected_end = section.removeprefix("*").split("-")
        assert float(actual_start) == pytest.approx(float(expected_start), abs=1e-12)
        assert actual_end == expected_end
        assert "--force-keyframes-at-cuts" in command
        Path(command[command.index("-o") + 1]).write_bytes(b"video")

    monkeypatch.setattr(video.subprocess, "run", run)
    video.download("https://example.test/video", tmp_path / "clip.mp4", start, end)


def test_download_success_without_output_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        video.subprocess, "run", lambda *_unused_args, **_unused_kwargs: None
    )
    output = tmp_path / "clip.mp4"
    with pytest.raises(ValueError, match="complete video"):
        video.download("https://example.test/video", output)
    assert not output.exists()


def test_trim_keeps_optional_audio(clip: Path, tmp_path: Path) -> None:
    source = tmp_path / "with-audio.mp4"
    subprocess.run(  # noqa: S603
        [
            executable("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(clip),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    output = video.trim(source, tmp_path / "trimmed.mp4", "0.5", "1.5")
    streams = json.loads(
        subprocess.run(  # noqa: S603
            [
                executable("ffprobe"),
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["streams"]
    assert [stream["codec_type"] for stream in streams] == ["video", "audio"]
    assert abs(video.probe(output).duration - 1) < 0.1


def test_real_ytdlp_section_download_publishes_complete_accurate_clip(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "downloaded.mp4"
    output.write_bytes(b"previous complete recording")
    run = subprocess.run
    completed = []

    def observe_publication(
        command: list[str], *, check: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        if command[:3] != [video.sys.executable, "-m", "yt_dlp"]:
            return run(command, check=check)
        pending = Path(command[command.index("-o") + 1])
        assert pending != output
        assert pending.parent.parent == output.parent
        result = run(command, check=check, timeout=30)
        assert pending.is_file()
        assert pending.stat().st_size > 0
        assert output.read_bytes() == b"previous complete recording"
        completed.append(pending)
        return result

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    handler = partial(SimpleHTTPRequestHandler, directory=str(clip.parent))
    with (
        monkeypatch.context() as patch,
        ThreadingHTTPServer(("127.0.0.1", 0), handler) as server,
    ):
        patch.setattr(video.subprocess, "run", observe_publication)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert (
                video.download(
                    f"http://127.0.0.1:{server.server_port}/{clip.name}",
                    output,
                    start="00:00.7",
                    end="00:01.4",
                )
                == output
            )
        finally:
            server.shutdown()
            thread.join(timeout=2)
    assert len(completed) == 1
    assert not completed[0].exists()
    assert not list(tmp_path.glob(".download-*"))
    info = video.probe(output)
    assert (info.width, info.height, info.fps) == (320, 180, 30.0)
    assert abs(info.duration - 0.7) < 1 / 30 + 0.01
    for offset in (0.0, 0.3):
        actual = video.frame_at(output, offset, size=(320, 180)).astype(float)
        expected = video.frame_at(clip, 0.7 + offset, size=(320, 180)).astype(float)
        assert np.abs(actual - expected).mean() < 4
