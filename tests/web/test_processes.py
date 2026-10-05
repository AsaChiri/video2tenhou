# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Real process lifecycle coverage without models, videos or external services."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import psutil
import pytest

from tests.spies import record_results
from video2tenhou.tool import processes
from video2tenhou.tool.processes import ProcessCancelledError, ProcessOwner


def command(source: str) -> list:
    """Build an unbuffered Python child command for process lifecycle tests."""
    return [sys.executable, "-u", "-c", source]


def line(process: subprocess.Popen[str]) -> str:
    """Read one child output line with a bounded background wait."""
    result = queue.Queue()
    stdout = process.stdout
    assert stdout is not None
    threading.Thread(target=lambda: result.put(stdout.readline()), daemon=True).start()
    return result.get(timeout=5).strip()


def alive(pid: int) -> bool:
    """Check whether the child PID is still running and is not a zombie."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def wait_dead(pid: int) -> None:
    """Wait briefly for a child to exit, failing if it remains alive."""
    deadline = time.monotonic() + 5
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not alive(pid), f"Owned process {pid} remained alive"


CHILD_TREE = (
    "import subprocess,sys,time; "
    "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
    "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
    "print(child.pid,flush=True); time.sleep(30)"
)


def run(owner: ProcessOwner, source: str) -> tuple[str, str, int | None]:
    """Run an owned child to completion and return its text output and status."""
    with owner.spawn(
        command(source), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        stdout, stderr = process.communicate(timeout=5)
    return stdout, stderr, process.returncode


def test_spawn_keeps_text_output_and_exit_status() -> None:
    """Owned commands report UTF-8 text and their exit status."""
    owner = ProcessOwner()
    assert run(owner, "print('done \\u00e9')") == ("done é\n", "", 0)
    _, stderr, code = run(
        owner, "import sys; print('diagnostic',file=sys.stderr); sys.exit(7)"
    )
    assert code == 7
    assert "diagnostic" in stderr
    owner.shutdown()


def test_shutdown_stops_parent_and_grandchild_and_rejects_new_work() -> None:
    owner = ProcessOwner()
    with ExitStack() as scope:
        process = scope.enter_context(
            owner.spawn(command(CHILD_TREE), stdout=subprocess.PIPE)
        )
        grandchild = int(line(process))
        assert alive(process.pid)
        assert alive(grandchild)
        owner.shutdown(timeout=5)
        assert process.poll() is not None
        wait_dead(grandchild)
        with pytest.raises(ProcessCancelledError):
            scope.close()
    assert owner.closing
    owner.shutdown()  # no stale PID is acted on during repeat shutdown
    with pytest.raises(ProcessCancelledError):
        run(owner, "raise AssertionError('must never start')")


def test_leaving_stream_context_early_cleans_the_whole_tree() -> None:
    owner = ProcessOwner()
    with owner.spawn(command(CHILD_TREE), stdout=subprocess.PIPE) as process:
        grandchild = int(line(process))
    assert process.poll() is not None
    wait_dead(grandchild)
    owner.shutdown()


def test_successful_parent_cannot_leave_a_worker_behind() -> None:
    owner = ProcessOwner()
    parent_exits = CHILD_TREE.rsplit("; time.sleep(30)", 1)[0]
    stdout, _, code = run(owner, parent_exits)
    assert code == 0
    wait_dead(int(stdout.strip()))
    owner.shutdown()


def test_shutdown_unblocks_another_threads_capture() -> None:
    owner = ProcessOwner()
    started = threading.Event()
    failures = []

    def work() -> None:
        try:
            with owner.spawn(
                command("import time; print('ready',flush=True); time.sleep(30)"),
                stdout=subprocess.PIPE,
            ) as process:
                assert line(process) == "ready"
                started.set()
                process.communicate(timeout=10)
        except ProcessCancelledError as exc:
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert started.wait(5)
        owner.shutdown()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], ProcessCancelledError)
    finally:
        owner.shutdown()
        worker.join(timeout=5)


def test_assignment_failure_stops_suspended_child_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows job assignment")
    owner = ProcessOwner()
    started = []
    original_popen = subprocess.Popen
    marker = tmp_path / "child-ran"
    failure = PermissionError("job assignment failed")

    def reject_assignment(*_unused_args: object) -> None:
        raise failure

    monkeypatch.setattr(subprocess, "Popen", record_results(original_popen, started))
    monkeypatch.setattr(
        processes.win32job, "AssignProcessToJobObject", reject_assignment
    )
    with pytest.raises(PermissionError) as caught:
        run(owner, f"from pathlib import Path; Path({str(marker)!r}).touch()")
    assert caught.value is failure
    assert len(started) == 1
    assert started[0].poll() is not None
    assert started[0].stdout.closed
    assert started[0].stderr.closed
    assert not marker.exists()
    owner.shutdown()


def test_shutdown_leaves_other_owners_running() -> None:
    one, two = ProcessOwner(), ProcessOwner()
    try:
        with two.spawn(command(CHILD_TREE), stdout=subprocess.PIPE) as other:
            other_child = int(line(other))
            with ExitStack() as scope:
                process = scope.enter_context(
                    one.spawn(command(CHILD_TREE), stdout=subprocess.PIPE)
                )
                child = int(line(process))
                one.shutdown()
                wait_dead(child)
                assert alive(other.pid)
                assert alive(other_child)
                with pytest.raises(ProcessCancelledError):
                    scope.close()
            assert alive(other.pid)
            assert alive(other_child)
        wait_dead(other_child)
    finally:
        one.shutdown()
        two.shutdown()
