# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Coordinate application shutdown using subprocess, pywin32 and psutil.

Windows Job Objects and POSIX process groups include descendants after their
parent exits. The registry only contains processes launched by this workspace.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, closing, contextmanager, suppress
from typing import IO, TYPE_CHECKING, TypedDict, Unpack, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from _win32typing import PyHANDLE

if os.name == "nt":
    import psutil
    import win32api
    import win32con
    import win32job
    import win32process


class ProcessCancelledError(RuntimeError):
    """The application closed before the command completed."""


class LaunchOptions(TypedDict, total=False):
    """Standard subprocess options used by owned application commands."""

    stdin: int | IO[str] | None
    stdout: int | IO[str] | None
    stderr: int | IO[str] | None
    cwd: str | os.PathLike[str] | None
    env: Mapping[str, str] | None
    shell: bool
    preexec_fn: Callable[[], None] | None
    start_new_session: bool
    process_group: int | None
    creationflags: int


class ProcessOwner:
    """Register commands atomically with shutdown; cancel their whole process trees."""

    def __init__(self) -> None:
        """Initialize ownership tracking before any child processes are started."""
        self._lock = threading.RLock()
        self._children = {}
        self._closing = False

    @property
    def closing(self) -> bool:
        """Report whether shutdown has started under the ownership lock."""
        with self._lock:
            return self._closing

    @contextmanager
    def spawn(
        self, args: Sequence[str], **kwargs: Unpack[LaunchOptions]
    ) -> Iterator[subprocess.Popen[str]]:
        """Start a UTF-8 text subprocess; leaving the context stops its descendants.

        Raises ProcessCancelledError when the application closes before or while
        the command runs.
        """
        self._validate_options(kwargs)
        with self._lock, ExitStack() as setup:
            if self._closing:
                raise ProcessCancelledError(
                    "The app is closing; this operation was interrupted."
                )
            job = None
            if os.name == "nt":
                # types-pywin32 declares None here; pywin32 returns an owned handle.
                job = setup.enter_context(
                    closing(cast("PyHANDLE", win32job.CreateJobObject(None, "")))
                )
                limits = win32job.QueryInformationJobObject(
                    int(job), win32job.JobObjectExtendedLimitInformation
                )
                limits["BasicLimitInformation"]["LimitFlags"] |= (
                    win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                )
                win32job.SetInformationJobObject(
                    int(job), win32job.JobObjectExtendedLimitInformation, limits
                )
                kwargs["creationflags"] = (
                    subprocess.CREATE_NO_WINDOW | win32process.CREATE_SUSPENDED
                )
            else:
                kwargs["start_new_session"] = True
            # Python children must write UTF-8 regardless of the console code page
            # (cp936 cannot encode Japanese player names).
            kwargs["env"] = {
                **(kwargs.get("env") or os.environ),
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            }
            process = subprocess.Popen(  # noqa: S603
                args, text=True, encoding="utf-8", errors="replace", **kwargs
            )
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    setup.enter_context(closing(stream))
            setup.callback(process.wait, timeout=5)
            setup.callback(process.kill)
            if os.name == "nt":
                inherit_handle = False
                with closing(
                    cast(
                        "PyHANDLE",
                        win32api.OpenProcess(
                            win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
                            inherit_handle,
                            process.pid,
                        ),
                    )
                ) as handle:
                    if job is None:
                        raise RuntimeError(
                            "Windows process launch requires an owned job handle"
                        )
                    win32job.AssignProcessToJobObject(int(job), int(handle))
                psutil.Process(process.pid).resume()
            self._children[process] = job
            setup.pop_all()
        try:
            yield process
            if self.closing:
                raise ProcessCancelledError(
                    "The app closed before the operation finished."
                )
        finally:
            try:
                self._stop(process)
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    @staticmethod
    def _validate_options(options: LaunchOptions) -> None:
        """Reject caller options that would bypass process-tree ownership."""
        if (
            options.get("shell")
            or options.get("preexec_fn")
            or options.get("start_new_session")
            or options.get("process_group") is not None
            or options.get("creationflags")
        ):
            raise ValueError(
                "ProcessOwner controls process groups and accepts direct commands only."
            )

    def _stop(
        self, process: subprocess.Popen[str], *, deadline: float | None = None
    ) -> None:
        with self._lock:
            if process not in self._children:
                return
            job = self._children[process]
            if job is not None:
                job.close()  # The OS terminates every member on last-handle close.
            elif sys.platform != "win32":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(
                timeout=5 if deadline is None else max(0.0, deadline - time.monotonic())
            )
            del self._children[process]

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop every registered command and reject further work."""
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = time.monotonic() + timeout
        with self._lock, ExitStack() as pending:
            self._closing = True
            for process in self._children:
                pending.callback(self._stop, process, deadline=deadline)
