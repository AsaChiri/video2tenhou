import shutil
import subprocess

import pytest

from video2tenhou import video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "t.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30", "-t", "2",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_probe(clip):
    info = video.probe(clip)
    assert (info.width, info.height) == (320, 180)
    assert abs(info.fps - 30) < 1e-6 and abs(info.duration - 2.0) < 0.1


def test_sample_and_seek(clip):
    frames = list(video.sample(clip, fps=2.0))
    assert len(frames) == 4
    t, f = frames[1]
    assert t == 0.5 and f.shape == (1080, 1920, 3)
    part = list(video.sample(clip, fps=2.0, start=1.0, end=2.0))
    assert [round(t, 2) for t, _ in part] == [1.0, 1.5]
    one = video.frame_at(clip, 1.0)
    assert one.shape == (1080, 1920, 3)


def test_failed_decode_is_reported_instead_of_an_empty_success(tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg could not sample"):
        list(video.sample(tmp_path / "missing.mp4"))


def test_sample_can_be_closed_early_without_waiting_for_the_entire_clip(clip):
    frames = video.sample(clip, fps=30)
    assert next(frames)[0] == 0
    frames.close()


def test_empty_window_does_not_start_a_decoder(monkeypatch):
    monkeypatch.setattr(video.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected decode"))
    assert list(video.sample("unused.mp4", start=1., end=1.)) == []


@pytest.mark.parametrize("source", ["https://www.youtube.com/watch?v=example&feature=share", "--version"])
def test_download_cli_passes_source_as_literal_to_ytdlp(tmp_path, monkeypatch, source):
    from video2tenhou.cli import main

    commands = []
    monkeypatch.setattr(video.subprocess, "run", lambda cmd, **kwargs: commands.append((cmd, kwargs)))
    output = tmp_path / "video.mp4"
    main(["download", "--", source, str(output)])
    command, kwargs = commands[0]
    assert command[:3] == [video.sys.executable, "-m", "yt_dlp"]
    assert command[-4:] == ["-o", str(output), "--", source]
    assert kwargs["check"] is True


def test_download_section_uses_installed_ytdlp_and_preserves_literal_url(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(video.subprocess, "run", lambda command, **kwargs: commands.append(command))
    video.download("--version", tmp_path / "clip.mp4", start="00:01:00", end="00:02:00")
    assert commands[0][:3] == [video.sys.executable, "-m", "yt_dlp"]
    assert commands[0][-4:] == ["--download-sections", "*00:01:00-00:02:00", "--", "--version"]


def test_download_propagates_ytdlp_failure(tmp_path, monkeypatch):
    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(video.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        video.download("https://unsupported.test/video", tmp_path / "video.mp4")
