"""Own pipeline subprocess trees for bounded application shutdown.

Windows children start suspended, join a kill-on-close Job Object, then resume;
their ffmpeg/yt-dlp descendants inherit that job. POSIX children start a new
session and are stopped as a process group. Only trees created by this owner
are controlled: adopting an arbitrary existing PID would not establish safe
ownership. Processes must not deliberately detach from their owned group.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import signal
import subprocess
import sys
import threading
import time


class ProcessCancelled(RuntimeError):
    """The application is closing; a pipeline command cannot start or finish."""


class _WindowsJob:
    """Native ownership boundary, including descendants whose parent has exited."""

    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w

        class Limits(c.Structure):
            _fields_ = [("process_time", c.c_longlong), ("job_time", c.c_longlong),
                        ("flags", w.DWORD), ("min_working", c.c_size_t), ("max_working", c.c_size_t),
                        ("active", w.DWORD), ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Extended(c.Structure):
            _fields_ = [("basic", Limits), ("io", c.c_ulonglong * 6),
                        ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                        ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]

        class ThreadEntry(c.Structure):
            _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("tid", w.DWORD), ("pid", w.DWORD),
                        ("base_priority", w.LONG), ("delta_priority", w.LONG), ("flags", w.DWORD)]

        self.c, self.w, self.ThreadEntry = c, w, ThreadEntry
        self.api = c.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, c.POINTER(ThreadEntry)], w.BOOL),
            "Thread32Next": ([w.HANDLE, c.POINTER(ThreadEntry)], w.BOOL),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "ResumeThread": ([w.HANDLE], w.DWORD),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.api.CloseHandle(self.handle)
            self.handle = None
            raise error

    def attach_and_resume(self, pid):
        process = self.api.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise self.c.WinError(self.c.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.handle, process):
                raise self.c.WinError(self.c.get_last_error())
        finally:
            self.api.CloseHandle(process)
        # Popen closes the primary thread handle. Enumerate that still-suspended
        # process's thread and reopen it using the documented Win32 thread API.
        snapshot = self.api.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == self.c.c_void_p(-1).value:
            raise self.c.WinError(self.c.get_last_error())
        resumed = False
        try:
            entry = self.ThreadEntry()
            entry.size = self.c.sizeof(entry)
            found = self.api.Thread32First(snapshot, self.c.byref(entry))
            while found:
                if entry.pid == pid:
                    thread = self.api.OpenThread(2, False, entry.tid)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise self.c.WinError(self.c.get_last_error())
                    try:
                        if self.api.ResumeThread(thread) == 0xFFFFFFFF:
                            raise self.c.WinError(self.c.get_last_error())
                        resumed = True
                    finally:
                        self.api.CloseHandle(thread)
                found = self.api.Thread32Next(snapshot, self.c.byref(entry))
        finally:
            self.api.CloseHandle(snapshot)
        if not resumed:
            raise RuntimeError("Could not resume the owned pipeline process.")

    def stop(self):
        if self.handle:
            handle, self.handle = self.handle, None
            try:
                if not self.api.TerminateJobObject(handle, 1):
                    raise self.c.WinError(self.c.get_last_error())
            finally:
                self.api.CloseHandle(handle)


class _Owned:
    def __init__(self, process, job):
        self.process, self.job = process, job
        self.lock = threading.Lock()
        self.stopped = False

    def stop(self, timeout):
        with self.lock:
            if self.stopped:
                return
            if self.job:
                self.job.stop()
            else:
                # The group also includes ffmpeg children after their immediate
                # parent exits. SIGKILL makes shutdown bounded and noninteractive.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait(timeout=max(0., timeout))
            self.stopped = True


class ProcessOwner:
    """Serialize process registration against shutdown and own every child tree.

    Use ``with owner.spawn(args, ...) as process`` for streamed output, or
    ``owner.run(args, ...)`` for captured output. ``shutdown`` is permanent and
    idempotent. Application code must also stop/join its worker threads; this
    owner prevents a worker from launching another phase after shutdown.
    """

    def __init__(self):
        """Create an idle registry; no system processes or handles are allocated."""
        self._lock = threading.RLock()
        self._children: set[_Owned] = set()
        self._closing = False

    @property
    def closing(self) -> bool:
        """Whether shutdown has begun; use this to label interrupted UI jobs."""
        with self._lock:
            return self._closing

    @contextmanager
    def spawn(self, args, **kwargs):
        """Yield an owned Popen and clean up its entire tree when the block ends.

        Arguments otherwise follow Popen. Shell execution and caller-controlled
        session/group flags are rejected to preserve a verifiable ownership
        boundary. Windows uses hidden windows and suspended job registration.
        Breaking out of a streamed read cancels the child instead of hanging.
        """
        if kwargs.get("shell") or kwargs.get("preexec_fn") or kwargs.get("start_new_session") or kwargs.get("process_group") is not None:
            raise ValueError("ProcessOwner controls process groups and accepts direct commands only.")
        with self._lock:
            if self._closing:
                raise ProcessCancelled("The app is closing; this operation was interrupted.")
            job = _WindowsJob() if os.name == "nt" else None
            if os.name == "nt":
                if kwargs.get("creationflags", 0):
                    if job:
                        job.stop()
                    raise ValueError("ProcessOwner controls Windows process creation flags.")
                kwargs["creationflags"] = 0x08000000 | 0x00000004  # NO_WINDOW | SUSPENDED
            else:
                kwargs["start_new_session"] = True
            process = None
            try:
                process = subprocess.Popen(args, **kwargs)
                if job:
                    job.attach_and_resume(process.pid)
                owned = _Owned(process, job)
                self._children.add(owned)
            except BaseException:
                if job:
                    job.stop()
                if process is not None:
                    process.kill()
                    process.wait(timeout=5)
                raise
        try:
            yield process
            if self.closing:
                raise ProcessCancelled("The app closed before the operation finished.")
        finally:
            original = sys.exc_info()[1]
            try:
                owned.stop(5)
            except Exception as exc:
                if original is None:
                    raise
                original.add_note(f"Process cleanup also failed: {exc}")
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream:
                        stream.close()
                with self._lock:
                    if owned.stopped:
                        self._children.discard(owned)

    def run(self, args, *, input=None, capture_output=False, timeout=None, check=False, **kwargs):
        """Run a command with subprocess.run semantics and owned-tree cleanup.

        Nonzero exits and timeouts retain their standard subprocess exceptions
        and captured output. Shutdown raises ProcessCancelled. Output buffering
        is appropriate for review rebuilds; use spawn for long progress logs.
        """
        if input is not None:
            if "stdin" in kwargs:
                raise ValueError("stdin and input cannot be combined")
            kwargs["stdin"] = subprocess.PIPE
        if capture_output:
            if "stdout" in kwargs or "stderr" in kwargs:
                raise ValueError("capture_output cannot be combined with stdout/stderr")
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with self.spawn(args, **kwargs) as process:
            try:
                stdout, stderr = process.communicate(input, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                with self._lock:
                    owned = next((item for item in self._children if item.process is process), None)
                if owned is not None:  # shutdown may already have reaped it
                    owned.stop(5)
                # Windows communicate uses reader threads and only exposes
                # their buffered diagnostics after the pipes reach EOF.
                exc.output, exc.stderr = process.communicate(timeout=5)
                raise
            if self.closing:
                raise ProcessCancelled("The app closed before the operation finished.")
            result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
            if check:
                result.check_returncode()
            return result

    def shutdown(self, timeout: float = 5.) -> None:
        """Reject future launches, stop every owned tree and reap children.

        Wait time is shared across children. Cleanup failures are raised rather
        than reported as success; repeated calls may retry an incomplete stop.
        The caller should invoke this before joining application worker threads.
        """
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        with self._lock:
            self._closing = True
            children = list(self._children)
        deadline = time.monotonic() + timeout
        errors = []
        for child in children:
            try:
                child.stop(max(0., deadline - time.monotonic()))
                with self._lock:
                    self._children.discard(child)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Could not stop {len(errors)} owned process tree(s): {errors[0]}") from errors[0]
