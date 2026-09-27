"""The streaming header benchmark forwards sampling inputs and detects changed evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from video2tenhou import benchmark, overlay, video
from video2tenhou.layout import CORNERS


@pytest.fixture
def streaming(monkeypatch):
    calls, closed = [], []
    mask = np.zeros((40, 24), np.uint8)
    mask[4:36, 5:19] = 255
    def sample(path, **kwargs):
        calls.append((path, kwargs))
        try:
            yield kwargs["start"], mask
            yield kwargs["start"] + 1 / kwargs["fps"], mask
        finally:
            closed.append(True)
    def read(frame, *, names, cal):
        assert names is False and cal == "custom geometry"
        label, confidence = overlay._DIGITS.classify(frame)
        return SimpleNamespace(kyoku_index=0, honba=int(label or 0), riichi_sticks=0,
                               corners={c: SimpleNamespace(score=25000, wind=w)
                                        for c, w in zip(CORNERS, "ESWN")})
    monkeypatch.setattr(video, "sample", sample)
    monkeypatch.setattr(overlay, "read_overlay", read)
    return calls, closed


def test_header_benchmark_streams_each_window_and_restores_matcher(streaming, tmp_path):
    original = overlay._DIGITS
    windows = [(2., 4.), (10., 12.)]
    result = benchmark.compare_header(tmp_path / "recording.mp4", windows, "custom geometry", fps=2)
    calls, closed = streaming
    assert result["rows_equal"] and not result["changed_rows"]
    assert result["metrics"]["scalar"]["frames"] == result["metrics"]["production"]["frames"] == 4
    assert [kwargs for _, kwargs in calls] == [
        {"fps": 2, "start": start, "end": end} for _ in range(2) for start, end in windows]
    assert len(closed) == 4 and overlay._DIGITS is original
    for metrics in result["metrics"].values():
        assert metrics["total_seconds"] >= metrics["recognition_seconds"] + metrics["sampling_seconds"] >= 0


def test_changed_header_values_fail_equivalence(streaming, monkeypatch, tmp_path):
    monkeypatch.setattr(overlay.DigitMatcher, "classify", lambda self, mask: ("99", 1.))
    result = benchmark.compare_header(tmp_path / "recording.mp4", [(0., 2.)], "custom geometry")
    assert not result["rows_equal"]
    assert len(result["changed_rows"]) == 2
    assert result["changed_rows"][0]["production"]["honba"] == 99
    assert result["changed_rows"][0]["scalar"]["honba"] != 99


def test_reader_failure_closes_sampler_and_restores_matcher(streaming, monkeypatch, tmp_path):
    original = overlay._DIGITS
    def fail(*args, **kwargs):
        raise RuntimeError("recognition failed")
    monkeypatch.setattr(overlay, "read_overlay", fail)
    with pytest.raises(RuntimeError, match="recognition failed"):
        benchmark.compare_header(tmp_path / "recording.mp4", [(0., 2.)], "custom geometry")
    assert overlay._DIGITS is original and streaming[1] == [True]


def test_video_cli_writes_report_and_propagates_mismatch(monkeypatch, tmp_path):
    from video2tenhou.layout import Calibration
    seen = []
    monkeypatch.setattr(Calibration, "load", lambda name, path: "custom geometry")
    def compare(path, windows, cal, fps):
        seen.append((path, windows, cal, fps))
        return {"rows_equal": False, "changed_rows": [{"index": 0}]}
    monkeypatch.setattr(benchmark, "compare_header", compare)
    output = tmp_path / "report.json"
    assert benchmark.main(["--video", "recording.mp4", "--window", "1", "3", "--window", "8", "10",
                           "--calib", "custom.json", "--fps", "2", "--output", str(output)]) == 1
    assert seen[0][1:] == ([[1., 3.], [8., 10.]], "custom geometry", 2.)
    assert '"rows_equal": false' in output.read_text()


def test_invalid_windows_and_empty_video_are_not_successful_comparisons(monkeypatch, tmp_path):
    path = tmp_path / "recording.mp4"
    with pytest.raises(ValueError, match="START < END"):
        benchmark.compare_header(path, [(4., 2.)], "custom geometry")
    def empty(*args, **kwargs):
        yield from ()
    monkeypatch.setattr(video, "sample", empty)
    with pytest.raises(ValueError, match="No frames"):
        benchmark.compare_header(path, [(0., 2.)], "custom geometry")
