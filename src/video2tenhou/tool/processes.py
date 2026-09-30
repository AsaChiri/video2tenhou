# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Coordinate application shutdown using subprocess, pywin32 and psutil.

Windows Job Objects and POSIX process groups include descendants after their
parent exits. The registry only contains processes launched by this workspace.
"""

from __future__ import annotations

import locale
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, closing, contextmanager, suppress
from typing import IO, TYPE_CHECKING, Literal, TypedDict, Unpack, cast, overload

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

    stdin: int | IO[str] | IO[bytes] | None
    stdout: int | IO[str] | IO[bytes] | None
    stderr: int | IO[str] | IO[bytes] | None
    cwd: str | os.PathLike[str] | None
    env: Mapping[str, str] | None
    encoding: str | None
    errors: str | None
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

    @overload
    @contextmanager
    def spawn(
        self,
        args: Sequence[str],
        *,
        text: Literal[True],
        **kwargs: Unpack[LaunchOptions],
    ) -> Iterator[subprocess.Popen[str]]: ...

    @overload
    @contextmanager
    def spawn(
        self,
        args: Sequence[str],
        *,
        text: Literal[False] = False,
        **kwargs: Unpack[LaunchOptions],
    ) -> Iterator[subprocess.Popen[bytes]]: ...

    @overload
    @contextmanager
    def spawn(
        self, args: Sequence[str], *, text: bool, **kwargs: Unpack[LaunchOptions]
    ) -> Iterator[subprocess.Popen[str] | subprocess.Popen[bytes]]: ...

    @contextmanager
    def spawn(
        self,
        args: Sequence[str],
        *,
        text: bool = False,
        **kwargs: Unpack[LaunchOptions],
    ) -> Iterator[subprocess.Popen[str] | subprocess.Popen[bytes]]:
        """Stream a subprocess; leaving the context stops remaining descendants."""
        self._validate_options(kwargs)
        with self._lock, ExitStack() as setup:
            if self._closing:
                msg = "The app is closing; this operation was interrupted."
                raise ProcessCancelledError(msg)
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
            process = subprocess.Popen(args, text=text, **kwargs)  # noqa: S603
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
                        msg = "Windows process launch requires an owned job handle"
                        raise RuntimeError(msg)
                    win32job.AssignProcessToJobObject(int(job), int(handle))
                psutil.Process(process.pid).resume()
            self._children[process] = job
            setup.pop_all()
        try:
            yield process
            if self.closing:
                msg = "The app closed before the operation finished."
                raise ProcessCancelledError(msg)
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
            msg = (
                "ProcessOwner controls process groups and accepts direct commands only."
            )
            raise ValueError(msg)

    def _stop(
        self,
        process: subprocess.Popen[str] | subprocess.Popen[bytes],
        *,
        deadline: float | None = None,
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

    @overload
    def run(
        self,
        args: Sequence[str],
        *,
        text: Literal[True],
        capture_output: bool = False,
        timeout: float | None = None,
        check: bool = False,
        **kwargs: Unpack[LaunchOptions],
    ) -> subprocess.CompletedProcess[str]: ...

    @overload
    def run(
        self,
        args: Sequence[str],
        *,
        text: Literal[False] = False,
        capture_output: bool = False,
        timeout: float | None = None,
        check: bool = False,
        **kwargs: Unpack[LaunchOptions],
    ) -> subprocess.CompletedProcess[bytes]: ...

    def run(
        self,
        args: Sequence[str],
        *,
        capture_output: bool = False,
        text: bool = False,
        timeout: float | None = None,
        check: bool = False,
        **kwargs: Unpack[LaunchOptions],
    ) -> (
        subprocess.CompletedProcess[str]
        | subprocess.CompletedProcess[bytes]
        | subprocess.CompletedProcess[str | bytes]
    ):
        """Capture command output while retaining application shutdown ownership."""
        if capture_output:
            if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
                msg = "capture_output cannot be combined with stdout/stderr"
                raise ValueError(msg)
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with self.spawn(args, text=text, **kwargs) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                self._stop(process)
                stdout, stderr = process.communicate(timeout=5)
                # TimeoutExpired output is bytes even when normal output is text.
                encoding = kwargs.get("encoding") or locale.getpreferredencoding(
                    do_setlocale=False
                )
                exc.output = (
                    stdout.encode(encoding) if isinstance(stdout, str) else stdout
                )
                exc.stderr = (
                    stderr.encode(encoding) if isinstance(stderr, str) else stderr
                )
                raise
            if self.closing:
                msg = "The app closed before the operation finished."
                raise ProcessCancelledError(msg)
            result = subprocess.CompletedProcess(
                args, process.returncode, stdout, stderr
            )
            if check:
                result.check_returncode()
            return result

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop every registered command and reject further work."""
        if timeout < 0:
            msg = "timeout must be nonnegative"
            raise ValueError(msg)
        deadline = time.monotonic() + timeout
        with self._lock, ExitStack() as pending:
            self._closing = True
            for process in self._children:
                pending.callback(self._stop, process, deadline=deadline)
