"""Real process lifecycle coverage without models, videos or external services."""

import os
import queue
import subprocess
import sys
import threading
import time

import pytest

from video2tenhou.tool.processes import ProcessCancelled, ProcessOwner


def command(source):
    return [sys.executable, "-u", "-c", source]


def line(process):
    result = queue.Queue()
    threading.Thread(
        target=lambda: result.put(process.stdout.readline()), daemon=True
    ).start()
    return result.get(timeout=5).strip()


def alive(pid):
    import psutil

    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def wait_dead(pid):
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


def test_run_keeps_output_and_nonzero_exit_errors():
    owner = ProcessOwner()
    result = owner.run(
        command("print('done')"), capture_output=True, text=True, check=True, timeout=5
    )
    assert result.stdout.strip() == "done" and result.returncode == 0
    with pytest.raises(subprocess.CalledProcessError) as caught:
        owner.run(
            command("import sys; print('diagnostic',file=sys.stderr); sys.exit(7)"),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    assert caught.value.returncode == 7 and "diagnostic" in caught.value.stderr
    owner.shutdown()


def test_shutdown_stops_parent_and_grandchild_and_rejects_new_work():
    owner = ProcessOwner()
    with (
        pytest.raises(ProcessCancelled),
        owner.spawn(command(CHILD_TREE), stdout=subprocess.PIPE, text=True) as process,
    ):
        grandchild = int(line(process))
        assert alive(process.pid) and alive(grandchild)
        owner.shutdown(timeout=5)
        assert process.poll() is not None
        wait_dead(grandchild)
    assert owner.closing
    owner.shutdown()  # no stale PID is acted on during repeat shutdown
    with pytest.raises(ProcessCancelled):
        owner.run(command("raise AssertionError('must never start')"))


def test_leaving_stream_context_early_cleans_the_whole_tree():
    owner = ProcessOwner()
    with owner.spawn(command(CHILD_TREE), stdout=subprocess.PIPE, text=True) as process:
        grandchild = int(line(process))
    assert process.poll() is not None
    wait_dead(grandchild)
    owner.shutdown()


def test_successful_parent_cannot_leave_a_worker_behind():
    owner = ProcessOwner()
    parent_exits = CHILD_TREE.rsplit("; time.sleep(30)", 1)[0]
    result = owner.run(
        command(parent_exits), capture_output=True, text=True, check=True, timeout=5
    )
    assert result.returncode == 0
    wait_dead(int(result.stdout.strip()))
    owner.shutdown()


def test_timeout_retains_exception_and_cleans_descendants():
    owner = ProcessOwner()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        owner.run(command(CHILD_TREE), capture_output=True, timeout=0.8)
    assert caught.value.output
    grandchild = int(caught.value.output.strip())
    wait_dead(grandchild)
    owner.shutdown()


def test_shutdown_unblocks_another_threads_capture():
    owner = ProcessOwner()
    started = threading.Event()
    failures = []

    def work():
        try:
            with owner.spawn(
                command("import time; print('ready',flush=True); time.sleep(30)"),
                stdout=subprocess.PIPE,
                text=True,
            ) as process:
                assert line(process) == "ready"
                started.set()
                process.communicate(timeout=10)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert started.wait(5)
        owner.shutdown()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(failures) == 1 and isinstance(failures[0], ProcessCancelled)
    finally:
        owner.shutdown()
        worker.join(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows job assignment")
def test_assignment_failure_stops_suspended_child_before_it_runs(tmp_path, monkeypatch):
    from video2tenhou.tool import processes

    owner = ProcessOwner()
    started = []
    original_popen = subprocess.Popen
    marker = tmp_path / "child-ran"
    failure = PermissionError("job assignment failed")

    def record_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        started.append(process)
        return process

    def reject_assignment(*args):
        raise failure

    monkeypatch.setattr(subprocess, "Popen", record_process)
    monkeypatch.setattr(
        processes.win32job, "AssignProcessToJobObject", reject_assignment
    )
    with pytest.raises(PermissionError) as caught:
        owner.run(
            command(f"from pathlib import Path; Path({str(marker)!r}).touch()"),
            capture_output=True,
        )
    assert caught.value is failure
    assert len(started) == 1 and started[0].poll() is not None
    assert started[0].stdout.closed and started[0].stderr.closed
    assert not marker.exists()
    owner.shutdown()


def test_shutdown_leaves_other_owners_running():
    one, two = ProcessOwner(), ProcessOwner()
    try:
        with two.spawn(command(CHILD_TREE), stdout=subprocess.PIPE, text=True) as other:
            other_child = int(line(other))
            with (
                pytest.raises(ProcessCancelled),
                one.spawn(
                    command(CHILD_TREE), stdout=subprocess.PIPE, text=True
                ) as process,
            ):
                child = int(line(process))
                one.shutdown()
                wait_dead(child)
                assert alive(other.pid) and alive(other_child)
            assert alive(other.pid) and alive(other_child)
        wait_dead(other_child)
    finally:
        one.shutdown()
        two.shutdown()
