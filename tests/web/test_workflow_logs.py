# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Job status keeps changing after the bounded developer log fills up."""

from __future__ import annotations

from video2tenhou.tool.workflow import Job, Workspace


def test_log_count_keeps_advancing_while_only_the_latest_80_lines_are_retained():
    job = Job(kind="analyze", project="key", stage="Working")
    assert job.status()["log_lines"] == 0
    for count in range(1, 162):
        Workspace._follow([f"line {count}\n"], job, None)
        assert job.status()["log_lines"] == count
        assert list(job.log) == [
            f"line {number}" for number in range(max(1, count - 79), count + 1)
        ]
    assert job.record()["log"] == list(job.log)
    assert "log" not in job.status()
    retry = Job(kind="analyze", project="key", stage="Working")
    assert retry.status()["log_lines"] == 0


def test_log_count_excludes_blank_lines_and_stdout_outcomes():
    job = Job(kind="check", project="key", stage="Working")
    outcomes = []
    Workspace._follow(
        ["\n", "[0 fit] checking\n", '{"result": {"ok": true}}\n'],
        job,
        outcomes,
    )
    Workspace._follow(["  \n", "border detail\n"], job, None)
    assert job.status()["log_lines"] == 2
    assert list(job.log) == ["[0 fit] checking", "border detail"]
    assert job.stage == "Checking table borders"
    assert outcomes == [{"result": {"ok": True}}]
