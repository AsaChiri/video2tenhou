# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Profiler accounting must not change results, exceptions or decoder cleanup."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from ortools.sat.python.cp_model import OPTIMAL, CpModel, CpSolver

from tools import profile_conversion as profile
from video2tenhou import timeline
from video2tenhou.layout import Calibration
from video2tenhou.perception.detector import Detector

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import BinaryIO


def test_threaded_accounting_and_stage_hand_context(tmp_path: "Path") -> None:
    """Verify threaded accounting and stage hand context."""
    recorder = profile.Recorder(tmp_path)
    recorder.active_stage = "decode"
    recorder.active_hand = 7

    def add(_: object) -> None:
        for _ in range(100):
            recorder.add("search", 0.25, hand=recorder.active_hand)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(add, range(4)))
    (row,) = recorder.totals.values()
    assert (row["calls"], row["seconds"], row["hand"], row["stage"]) == (
        400,
        100,
        7,
        "decode",
    )
    recorder.close()


def test_generator_excludes_consumer_and_closes_early(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify generator excludes consumer and closes early."""
    ticks = [0.0]
    monkeypatch.setattr(profile.time, "perf_counter", lambda: ticks[0])
    recorder = profile.Recorder(tmp_path)
    closed = []

    def source() -> "Generator[str]":
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


def test_failure_preserves_exception_and_restores_hand_stage(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify failure preserves exception and restores hand stage."""
    recorder = profile.Recorder(tmp_path)
    threads = {"opencv": 32}
    monkeypatch.setattr(profile, "runtime_threads", lambda: dict(threads))

    def failing(entry: "dict") -> None:
        threads["opencv"] = 1  # Model imports can change settings before a failure.
        msg = "original"
        raise ValueError(msg)

    wrapped = recorder.wrap(
        failing, "hand", progress=True, hand_call=True, stage_call=True
    )
    with pytest.raises(ValueError, match="original"):
        wrapped({"hand": 3})
    assert recorder.active_hand is None
    assert recorder.active_stage is None
    (row,) = recorder.totals.values()
    assert row["failures"] == 1
    assert row["hand"] == 3
    recorder.close()
    events = list(map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines()))
    assert [row["event"] for row in events] == ["start", "end"]
    assert [row["runtime_threads"]["opencv"] for row in events] == [32, 1]


def test_cp_solver_actual_method_is_timed_once_and_restored(tmp_path: "Path") -> None:
    """Verify cp solver actual method is timed once and restored."""
    original = CpSolver.solve
    recorder = profile.Recorder(tmp_path)
    with profile.instrument(recorder):
        model = CpModel()
        variable = model.new_int_var(0, 1, "x")
        model.minimize(variable)
        assert CpSolver().solve(model) == OPTIMAL
    assert CpSolver.solve is original
    rows = [row for row in recorder.totals.values() if row["name"] == "CpSolver.solve"]
    assert len(rows) == 1
    assert rows[0]["calls"] == 1
    recorder.close()
    event = next(
        row
        for row in map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines())
        if row["event"] == "cp_search"
    )
    assert event["status"] == "OPTIMAL"
    assert event["objective"] == 0


def test_table_counting_is_timed_once_and_restored(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify table counting is timed once and restored."""
    calls = []

    def fake(
        frame: "str",
        _cal: object,
        _intervals: object,
        _detector: object,
        **kwargs: "object",
    ) -> str:
        calls.append((frame, kwargs))
        return "reading"

    monkeypatch.setattr(timeline, "read_pond_counts", fake)

    recorder = profile.Recorder(tmp_path)
    with profile.instrument(recorder):
        assert (
            timeline.read_pond_counts(
                "frame",
                Calibration.load("pml"),
                [],
                Detector.__new__(Detector),
                log=print,
            )
            == "reading"
        )
    assert timeline.read_pond_counts is fake
    rows = [
        row
        for row in recorder.totals.values()
        if row["name"] == "timeline.read_pond_counts"
    ]
    assert len(rows) == 1
    assert rows[0]["calls"] == 1
    assert calls == [("frame", {"log": print})]
    recorder.close()


def test_tool_versions_keep_failures_bounded_and_report_resolved_paths(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """One broken optional probe must not prevent metadata for the other tools."""
    commands = []
    monkeypatch.setattr(
        profile.shutil,
        "which",
        lambda name: None if name == "ffmpeg" else str(tmp_path / name),
    )

    def run(
        command: "list[str]", *, timeout: float, stdout: "BinaryIO", **kwargs: "object"
    ) -> "profile.subprocess.CompletedProcess[bytes]":
        commands.append(command)
        assert timeout == 3
        assert kwargs["check"] is False
        assert kwargs["stderr"] == profile.subprocess.STDOUT
        assert kwargs["stdin"] == profile.subprocess.DEVNULL
        name = Path(command[0]).name
        if name == "uv":
            raise profile.subprocess.TimeoutExpired(command, timeout)
        if name == "ffprobe":
            stdout.write(b"broken executable\n" + b"x" * 10000)
            return profile.subprocess.CompletedProcess(command, 7)
        raise AssertionError(command)

    monkeypatch.setattr(profile.subprocess, "run", run)
    rows = profile.tool_versions()
    assert rows["ffmpeg"] == {"path": None, "version": None, "status": "missing"}
    assert rows["ffprobe"]["status"] == "error"
    assert rows["ffprobe"]["error"] == "broken executable"
    assert rows["ffprobe"]["returncode"] == 7
    assert rows["uv"]["status"] == "timeout"
    assert rows["uv"]["version"] is None
    assert [command[1] for command in commands] == [
        "-version",
        "--version",
    ]

    monkeypatch.setattr(profile.shutil, "which", lambda name: str(tmp_path / name))

    def denied(*_unused_args: object, **_unused_kwargs: object) -> None:
        msg = "not executable"
        raise PermissionError(msg)

    monkeypatch.setattr(profile.subprocess, "run", denied)
    assert all(
        row["status"] == "error" and row["version"] is None
        for row in profile.tool_versions().values()
    )
