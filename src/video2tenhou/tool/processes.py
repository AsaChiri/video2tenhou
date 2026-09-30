"""Coordinate application shutdown using subprocess, pywin32 and psutil.

Windows Job Objects and POSIX process groups include descendants after their
parent exits. The registry only contains processes launched by this workspace.
"""

import os
import signal
import subprocess
import threading
import time
from contextlib import ExitStack, closing, contextmanager, suppress

if os.name == "nt":
    import psutil
    import win32api
    import win32con
    import win32job
    import win32process


class ProcessCancelled(RuntimeError):
    """The application closed before the command completed."""


class ProcessOwner:
    """Register commands atomically with shutdown; cancel their whole process trees."""

    def __init__(self):
        self._lock = threading.RLock()
        self._children = {}
        self._closing = False

    @property
    def closing(self) -> bool:
        with self._lock:
            return self._closing

    @contextmanager
    def spawn(self, args, **kwargs):
        """Stream a subprocess; leaving the context stops remaining descendants."""
        if (
            kwargs.get("shell")
            or kwargs.get("preexec_fn")
            or kwargs.get("start_new_session")
            or kwargs.get("process_group") is not None
            or kwargs.get("creationflags")
        ):
            raise ValueError(
                "ProcessOwner controls process groups and accepts direct commands only."
            )
        with self._lock, ExitStack() as setup:
            if self._closing:
                raise ProcessCancelled(
                    "The app is closing; this operation was interrupted."
                )
            job = None
            if os.name == "nt":
                job = setup.enter_context(closing(win32job.CreateJobObject(None, "")))
                limits = win32job.QueryInformationJobObject(
                    job, win32job.JobObjectExtendedLimitInformation
                )
                limits["BasicLimitInformation"]["LimitFlags"] |= (
                    win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                )
                win32job.SetInformationJobObject(
                    job, win32job.JobObjectExtendedLimitInformation, limits
                )
                kwargs["creationflags"] = (
                    subprocess.CREATE_NO_WINDOW | win32process.CREATE_SUSPENDED
                )
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(args, **kwargs)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    setup.enter_context(closing(stream))
            setup.callback(process.wait, timeout=5)
            setup.callback(process.kill)
            if os.name == "nt":
                with closing(
                    win32api.OpenProcess(
                        win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
                        False,
                        process.pid,
                    )
                ) as handle:
                    win32job.AssignProcessToJobObject(job, handle)
                psutil.Process(process.pid).resume()
            self._children[process] = job
            setup.pop_all()
        try:
            yield process
            if self.closing:
                raise ProcessCancelled("The app closed before the operation finished.")
        finally:
            try:
                self._stop(process)
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    def _stop(self, process, *, deadline=None):
        with self._lock:
            if process not in self._children:
                return
            job = self._children[process]
            if job is not None:
                job.close()  # The OS terminates every member on last-handle close.
            else:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(
                timeout=5 if deadline is None else max(0.0, deadline - time.monotonic())
            )
            del self._children[process]

    def run(self, args, *, capture_output=False, timeout=None, check=False, **kwargs):
        """Capture command output while retaining application shutdown ownership."""
        if capture_output:
            if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
                raise ValueError("capture_output cannot be combined with stdout/stderr")
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with self.spawn(args, **kwargs) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                self._stop(process)
                exc.output, exc.stderr = process.communicate(timeout=5)
                raise
            if self.closing:
                raise ProcessCancelled("The app closed before the operation finished.")
            result = subprocess.CompletedProcess(
                args, process.returncode, stdout, stderr
            )
            if check:
                result.check_returncode()
            return result

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop every registered command and reject further work."""
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = time.monotonic() + timeout
        with self._lock, ExitStack() as pending:
            self._closing = True
            for process in self._children:
                pending.callback(self._stop, process, deadline=deadline)
