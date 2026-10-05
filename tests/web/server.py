# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the production ASGI app on a real isolated loopback listener for tests."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from video2tenhou.tool.server import create_app
from video2tenhou.tool.workflow import Job, Workspace


@dataclass(frozen=True)
class LiveStudio:
    """Connection address and workspace belonging to a running test application."""

    url: str
    workspace: Workspace


@contextmanager
def studio_server(
    root: Path, *, runner: Callable[[list[str], Job], object] | None = None
) -> Iterator[LiveStudio]:
    """Wait for ASGI startup and complete application shutdown after each test."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        workspace = Workspace(root, runner=runner)
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(workspace),
                host="127.0.0.1",
                port=port,
                http="h11",
                ws="none",
                proxy_headers=False,
                access_log=False,
                log_config=None,
                log_level="error",
                timeout_graceful_shutdown=2,
            )
        )
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError("ASGI test server failed to start")
                time.sleep(0.01)
            yield LiveStudio(f"http://127.0.0.1:{port}", workspace)
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError("ASGI test server failed to stop")
            workspace.close()
