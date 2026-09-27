"""Profiler accounting must not change results, exceptions or decoder cleanup."""

from tests.paths import ROOT
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("profile_conversion", ROOT / "tools/profile_conversion.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_threaded_accounting_and_stage_hand_context(tmp_path):
    recorder = profile.Recorder(tmp_path)
    recorder.active_stage = "decode"
    recorder.active_hand = 7
    def add(_):
        for _ in range(100):
            recorder.add("search", .25, hand=recorder.active_hand)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(add, range(4)))
    row, = recorder.totals.values()
    assert (row["calls"], row["seconds"], row["hand"], row["stage"]) == (400, 100, 7, "decode")
    recorder.close()


def test_generator_excludes_consumer_and_closes_early(tmp_path, monkeypatch):
    ticks = [0.]
    monkeypatch.setattr(profile.time, "perf_counter", lambda: ticks[0])
    recorder = profile.Recorder(tmp_path)
    closed = []
    def source():
        try:
            ticks[0] += 2
            yield "frame"
        finally:
            ticks[0] += 3
            closed.append(True)
    iterator = recorder.generator(source)()
    assert next(iterator) == "frame"
    ticks[0] += 100  # Consumer processing must not be charged to ffmpeg.
    iterator.close()
    assert closed == [True]
    rows = {row["name"]: row for row in recorder.totals.values()}
    assert rows["video.sample.next_wait"]["seconds"] == 2
    assert rows["video.sample.close"]["seconds"] == 3
    recorder.close()


def test_failure_preserves_exception_and_restores_hand_stage(tmp_path, monkeypatch):
    recorder = profile.Recorder(tmp_path)
    threads = {"opencv": 32}
    monkeypatch.setattr(profile, "runtime_threads", lambda: dict(threads))
    def failing(entry):
        threads["opencv"] = 1  # Model imports can change settings before a failure.
        raise ValueError("original")
    wrapped = recorder.wrap(failing, "hand", progress=True, hand_call=True, stage_call=True)
    with pytest.raises(ValueError, match="original"):
        wrapped({"hand": 3})
    assert recorder.active_hand is None and recorder.active_stage is None
    row, = recorder.totals.values()
    assert row["failures"] == 1 and row["hand"] == 3
    recorder.close()
    events = list(map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines()))
    assert [row["event"] for row in events] == ["start", "end"]
    assert [row["runtime_threads"]["opencv"] for row in events] == [32, 1]


def test_cp_solver_actual_method_is_timed_once_and_restored(tmp_path):
    from ortools.sat.python.cp_model import CpModel, CpSolver, OPTIMAL
    original = CpSolver.solve
    recorder = profile.Recorder(tmp_path)
    with profile.instrument(recorder):
        model = CpModel()
        variable = model.NewIntVar(0, 1, "x")
        model.Minimize(variable)
        assert CpSolver().Solve(model) == OPTIMAL
    assert CpSolver.solve is original
    rows = [row for row in recorder.totals.values() if row["name"] == "CpSolver.solve"]
    assert len(rows) == 1 and rows[0]["calls"] == 1
    recorder.close()
    event = next(row for row in map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines()) if row["event"] == "cp_search")
    assert event["status"] == "OPTIMAL" and event["objective"] == 0


def test_imported_overlay_alias_is_timed_once_and_restored(tmp_path, monkeypatch):
    from video2tenhou import overlay, timeline
    calls = []
    def fake(frame, **kwargs):
        calls.append((frame, kwargs))
        return "reading"
    monkeypatch.setattr(overlay, "read_overlay", fake)
    monkeypatch.setattr(timeline, "read_overlay", fake)
    recorder = profile.Recorder(tmp_path)
    with profile.instrument(recorder):
        assert timeline.read_overlay is overlay.read_overlay
        assert timeline.read_overlay("frame", names=False) == "reading"
    assert timeline.read_overlay is fake and overlay.read_overlay is fake
    rows = [row for row in recorder.totals.values() if row["name"] == "overlay.read_overlay"]
    assert len(rows) == 1 and rows[0]["calls"] == 1
    assert calls == [("frame", {"names": False})]
    recorder.close()


def test_tool_versions_keep_failures_bounded_and_report_resolved_paths(tmp_path, monkeypatch):
    """One broken optional probe must not prevent metadata for the other tools."""
    commands = []
    monkeypatch.setattr(profile.shutil, "which", lambda name: None if name == "ffmpeg" else str(tmp_path / name))

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 3 and kwargs["check"] is False
        assert kwargs["stderr"] == profile.subprocess.STDOUT
        assert kwargs["stdin"] == profile.subprocess.DEVNULL
        name = Path(command[0]).name
        if name == "uv":
            raise profile.subprocess.TimeoutExpired(command, kwargs["timeout"])
        if name == "ffprobe":
            kwargs["stdout"].write(b"broken executable\n" + b"x" * 10000)
            return profile.subprocess.CompletedProcess(command, 7)
        kwargs["stdout"].write(b"\ntesseract 5.5.3\nleptonica details\n")
        return profile.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(profile.subprocess, "run", run)
    rows = profile.tool_versions()
    assert rows["ffmpeg"] == {"path": None, "version": None, "status": "missing"}
    assert rows["ffprobe"]["status"] == "error"
    assert rows["ffprobe"]["error"] == "broken executable"
    assert rows["ffprobe"]["returncode"] == 7
    assert rows["tesseract"]["version"] == "tesseract 5.5.3"
    assert rows["tesseract"]["path"] == str((tmp_path / "tesseract").resolve())
    assert rows["tesseract"]["status"] == "ok"
    assert rows["uv"]["status"] == "timeout" and rows["uv"]["version"] is None
    assert [command[1] for command in commands] == ["-version", "--version", "--version"]

    monkeypatch.setattr(profile.shutil, "which", lambda name: str(tmp_path / name))
    def denied(*args, **kwargs):
        raise PermissionError("not executable")
    monkeypatch.setattr(profile.subprocess, "run", denied)
    assert all(row["status"] == "error" and row["version"] is None for row in profile.tool_versions().values())
