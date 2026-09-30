# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Reports distinguish overlapping search time from conversion wall time."""

import json
from typing import TYPE_CHECKING

import pytest

from tools import report_profile as reporter

if TYPE_CHECKING:
    from pathlib import Path


def test_overlap_status_and_amdahl_accounting(tmp_path: "Path") -> None:
    """Verify overlap status and amdahl accounting."""
    events = [
        {"event": "start", "thread": 1, "name": "cli.cmd_convert", "stage": None},
        {
            "event": "end",
            "thread": 1,
            "name": "read.run_read",
            "stage": "read.run_read",
            "seconds": 80,
        },
        {
            "event": "cp_search",
            "thread": 1,
            "hand": 2,
            "stage": "engine.decode.run_decode",
            "status": "OPTIMAL",
            "time_budget": 60,
            "seconds": 3,
            "elapsed": 83,
        },
        {
            "event": "cp_search",
            "thread": 2,
            "hand": 2,
            "stage": "engine.decode.run_decode",
            "status": "FEASIBLE",
            "time_budget": 4,
            "seconds": 4,
            "elapsed": 88,
        },
        {
            "event": "cp_search",
            "thread": 3,
            "hand": 2,
            "stage": "engine.decode.run_decode",
            "status": "UNKNOWN",
            "time_budget": 4,
            "seconds": 4,
            "elapsed": 89,
        },
        {
            "event": "end",
            "thread": 1,
            "hand": 2,
            "name": "read.dense_reads",
            "stage": "engine.decode.run_decode",
            "seconds": 2,
        },
        {
            "event": "end",
            "thread": 1,
            "hand": 2,
            "name": "read.dense_reads",
            "stage": "engine.decode.run_decode",
            "seconds": 3,
        },
        {
            "event": "end",
            "thread": 1,
            "name": "decode.decode_hand",
            "stage": "decode.run_decode",
            "hand": 2,
            "seconds": 10,
            "failed": False,
        },
        {
            "event": "end",
            "thread": 1,
            "name": "decode.run_decode",
            "stage": "decode.run_decode",
            "seconds": 20,
        },
        {
            "event": "end",
            "thread": 1,
            "name": "cli.cmd_convert",
            "stage": None,
            "seconds": 100,
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "conversion_wall_seconds": 100,
                "conversion_process_cpu_seconds": 200,
                "timings": [],
            }
        )
    )
    (tmp_path / "metadata.json").write_text(json.dumps({"logical_cpus": 8}))
    result = reporter.summarize(tmp_path, ["read.run_read"], 2)
    assert result["amdahl"]["predicted_total_speedup"] == pytest.approx(1 / 0.6)
    assert result["amdahl"]["infinite_stage_speedup_ceiling"] == pytest.approx(5)
    assert result["machine_cpu_fraction"] == 0.25
    alternative = next(
        row for row in result["cp_searches"] if row["role"] == "alternative"
    )
    assert alternative["seconds"] == 8
    assert alternative["near_budget"] == 2
    assert alternative["active_wall_seconds"] == 5
    assert result["cp_active_wall_seconds_by_role"] == {
        "main_or_diagnostic": 3,
        "alternative": 5,
    }
    assert result["dense_read_wall_seconds"] == 5
    assert result["dense_read_wall_seconds_by_hand"] == {"2": 5}
    assert alternative["statuses"] == {"FEASIBLE": 1, "UNKNOWN": 1}
    assert result["per_hand"]["2"]["seconds"] == 10
    assert not result["active_spans"]


def test_running_profile_ignores_only_incomplete_last_line(tmp_path: "Path") -> None:
    """Verify running profile ignores only incomplete last line."""
    start = {"event": "start", "thread": 1, "name": "cli.cmd_convert", "stage": None}
    (tmp_path / "events.jsonl").write_text(json.dumps(start) + "\n" + '{"event":')
    result = reporter.summarize(tmp_path, ["read.run_read"])
    assert not result["finished"]
    assert result["active_spans"] == [start]
    assert "unavailable" in result["amdahl"]
    (tmp_path / "events.jsonl").write_text('{"event":\n')
    with pytest.raises(json.JSONDecodeError):
        reporter.summarize(tmp_path)


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([(4, 8), (1, 5), (7, 10)], 9),  # Overlap chain in unsorted arrival order.
        (
            [(0, 10), (2, 4), (0, 10), (5, 5)],
            10,
        ),  # Nesting, duplicate and zero-length span.
        ([(0, 2), (2, 3), (7, 9)], 5),  # Touching endpoints plus a genuine idle gap.
    ],
)
def test_interval_union_counts_only_occupied_wall_time(
    intervals: list[tuple[int, int]], expected: int
) -> None:
    """Verify interval union counts only occupied wall time."""
    assert reporter.interval_union_seconds(intervals) == expected
