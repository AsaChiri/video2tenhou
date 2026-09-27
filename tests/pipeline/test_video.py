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
