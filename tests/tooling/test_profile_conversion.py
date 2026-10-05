# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Profiling must not change results, exceptions or the instrumented functions."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from ortools.sat.python.cp_model import OPTIMAL, CpModel, CpSolver

from tools import profile_conversion as profile
from video2tenhou import cli, timeline
from video2tenhou.engine import decode
from video2tenhou.layout import Calibration
from video2tenhou.perception.detector import Detector
from video2tenhou.record import HandResult


def call(name: str, start: float, wall: float, **fields: object) -> profile.Call:
    """Build a completed call with a controlled interval."""
    completed = profile.Call(name, stage=None, hand=None, start=start, wall=wall)
    return replace(completed, failed=False, **fields)


def test_hand_and_stage_labels_reach_worker_calls_and_survive_failures() -> None:
    """Worker calls inherit the active stage and hand; errors propagate unchanged."""
    recorder = profile.Recorder()
    worker = recorder.wrap(lambda: None, "worker")

    def decode_hand(entry: dict) -> None:
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _: worker(), range(3)))
        raise ValueError("original")

    stage = recorder.wrap(
        recorder.wrap(decode_hand, profile.HAND, hand=True), "decode", stage=True
    )
    with pytest.raises(ValueError, match="original"):
        stage({"hand": 3})
    assert (recorder.stage, recorder.hand) == (None, None)
    workers = [item for item in recorder.calls if item.name == "worker"]
    assert [(item.stage, item.hand) for item in workers] == [("decode", 3)] * 3
    hand = next(item for item in recorder.calls if item.name == profile.HAND)
    assert hand.failed
    assert hand.stage == "decode"


def test_instrumentation_times_actual_methods_once_and_restores_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every patched entry point exists, is timed once and is restored afterwards."""
    original = CpSolver.solve
    calls = []

    def counts(frame: str, *_args: object, **kwargs: object) -> str:
        calls.append((frame, kwargs))
        return "reading"

    monkeypatch.setattr(timeline, "read_pond_counts", counts)
    recorder = profile.Recorder()
    with profile.instrument(recorder):
        model = CpModel()
        model.minimize(model.new_int_var(0, 1, "x"))
        assert CpSolver().solve(model) == OPTIMAL
        detector = Detector.__new__(Detector)
        reading = timeline.read_pond_counts(
            "frame", Calibration.load("pml"), [], detector, log=print
        )
        assert reading == "reading"
    assert CpSolver.solve is original
    assert timeline.read_pond_counts is counts
    assert calls == [("frame", {"log": print})]
    search, pond = recorder.calls
    assert (search.name, search.status, search.failed) == (
        "CpSolver.solve",
        "OPTIMAL",
        False,
    )
    assert pond.name == "timeline.read_pond_counts"


def test_summary_separates_concurrent_searches_dense_reads_and_stages() -> None:
    """Hands report solve and certification time; stage shares give a ceiling."""
    hand = {"stage": "engine.decode.run_decode", "hand": 2}
    calls = [
        call("read.run_read", 0, 80, cpu=160),
        call("engine.decode.run_decode", 80, 20, cpu=20),
        call(profile.HAND, 80, 10, cpu=12, **hand),
        call("engine.dense.dense_reads", 81, 2, **hand),
        call("engine.dense.dense_pond_reads", 83, 3, **hand),
        call(profile.SOLVE, 84, 3, **hand),
        call(profile.SEARCH, 84, 3, status="OPTIMAL", budget=60, **hand),
        call(profile.CERTIFY, 87, 9, **hand),
        call(profile.SEARCH, 87, 4, status="FEASIBLE", budget=4, **hand),
        call(profile.SEARCH, 91, 4, status="UNKNOWN", budget=4, **hand),
    ]
    summary = profile.summarize(calls)
    assert [(row["name"], row["wall"]) for row in summary["stages"]] == [
        ("read.run_read", 80),
        ("engine.decode.run_decode", 20),
    ]
    (row,) = summary["hands"]
    assert row["wall"] == 10
    assert row["cpu"] == 12
    assert row["dense_reads"] == 5
    assert row["solve"] == 3
    assert row["certify"] == 9
    assert row["near_budget"] == 2
    assert row["statuses"] == {"OPTIMAL": 1, "FEASIBLE": 1, "UNKNOWN": 1}
    summary.update(
        failure=None,
        wall_seconds=100,
        process_cpu_seconds=200,
        metadata={"logical_cpus": 8, "initial_files": {"work": 0, "out": 0}},
    )
    report = profile.render(summary)
    assert "| read.run_read | 1 | 80.0 | 160.0 | 80.0% | 5.00x |" in report
    assert "2.00 logical cores of 8" in report


def test_main_writes_summary_and_report_even_when_conversion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed conversion keeps its timings and fails the command."""

    def convert(args: object) -> None:
        decode.decode_hand({"hand": 5}, {}, cast("HandResult", None))

    def failing_hand(entry: dict, *_args: object) -> None:
        raise RuntimeError("hand failed")

    monkeypatch.setattr(cli, "cmd_convert", convert)
    monkeypatch.setattr(decode, "decode_hand", failing_hand)
    monkeypatch.setattr(
        profile,
        "run_metadata",
        lambda _args: {"logical_cpus": 1, "initial_files": {"work": 3, "out": 0}},
    )
    output = tmp_path / "profile"
    assert (
        profile.main(["--profile-output", str(output), "video.mp4", "--game", "1"]) == 1
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["failure"] == "RuntimeError: hand failed"
    assert summary["hands"][0]["failed"]
    assert (
        "Files before conversion: work 3, output 0."
        in (output / "report.md").read_text()
    )
    with pytest.raises(FileExistsError):
        profile.main(["--profile-output", str(output), "video.mp4", "--game", "1"])
