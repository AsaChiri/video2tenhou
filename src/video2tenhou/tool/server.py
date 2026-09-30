# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Starlette workspace API and loopback-only Uvicorn application lifecycle."""

from __future__ import annotations

import logging
import re
import socket
import threading
import uuid
import webbrowser
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect, Request
from starlette.responses import FileResponse, Response
from starlette.routing import Route

from video2tenhou.logging_setup import command_logging

from .http import (
    PROJECT_BODY_LIMIT,
    STATIC,
    UPLOAD_BODY_LIMIT,
    LocalAccess,
    json_response,
    read_json_body,
)
from .review_routes import review_routes
from .workflow import VIDEO_SUFFIXES, Workspace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from starlette.types import ASGIApp

LOGGER = logging.getLogger(__name__)


def request_error(request: Request, error: Exception) -> Response:
    """Keep failures structured and preserve the existing read/write status contract."""
    if isinstance(error, HTTPException):
        response = json_response({"error": str(error.detail)}, error.status_code)
        if error.headers:
            response.headers.update(error.headers)
        return response
    if isinstance(error, (KeyError, FileNotFoundError)):
        status = HTTPStatus.NOT_FOUND
    elif isinstance(error, (ValueError, ClientDisconnect)) or (
        request.method == "POST" and isinstance(error, (TypeError, OSError))
    ):
        status = HTTPStatus.BAD_REQUEST
    else:
        LOGGER.error("Workspace request failed", exc_info=error)
        status = HTTPStatus.INTERNAL_SERVER_ERROR
    return json_response({"error": str(error)}, status)


class WorkspaceRoutes:
    """Bind workspace endpoints and keep blocking work off the event loop."""

    def __init__(self, workspace: Workspace) -> None:
        """Retain the application-owned workspace for these routes."""
        self.workspace = workspace

    def status(self, _request: Request) -> Response:
        """Return setup status and current project snapshots."""
        with self.workspace.lock:
            return json_response(
                {
                    "setup": self.workspace.setup(),
                    "projects": [
                        self.workspace.snapshot(key) for key in self.workspace.projects
                    ],
                }
            )

    def project(self, request: Request) -> Response:
        """Return one recording's current workflow state."""
        return json_response(self.workspace.snapshot(request.path_params["key"]))

    def results(self, request: Request) -> Response:
        """List current validated exports and pending corrections."""
        return json_response(self.workspace.results(request.path_params["key"]))

    def export(self, request: Request) -> Response:
        """Stream an allowlisted export with bounded memory use."""
        file = self.workspace.artifact(
            request.path_params["key"], request.path_params["name"]
        )
        return FileResponse(
            file, filename=None if file.suffix == ".html" else file.name
        )

    async def create(self, request: Request) -> Response:
        """Validate a bounded project request and persist its recording metadata."""
        body = await read_json_body(request, PROJECT_BODY_LIMIT)
        project = await run_in_threadpool(self.workspace.create, body)
        return json_response(project, HTTPStatus.CREATED)

    async def action(self, request: Request) -> Response:
        """Apply one named project action with bounded input."""
        body = await read_json_body(request, PROJECT_BODY_LIMIT)
        return await run_in_threadpool(
            self.apply_action,
            request.path_params["key"],
            request.path_params["action"],
            body,
        )

    def apply_action(self, key: str, action: str, body: dict) -> Response:
        """Dispatch explicit project actions in a worker thread."""
        actions: dict[str, Callable[[], dict]] = {
            "rename": lambda: self.workspace.rename(key, body),
            "delete": lambda: self.workspace.delete(key),
            "settings": lambda: self.workspace.update(key, body),
            "prepare": lambda: self.workspace.start(key, "prepare"),
            "analyze": lambda: self.workspace.start(key, "analyze"),
        }
        if action not in actions:
            return json_response({"error": "Unknown action."}, HTTPStatus.NOT_FOUND)
        status = (
            HTTPStatus.ACCEPTED if action in ("prepare", "analyze") else HTTPStatus.OK
        )
        return json_response(actions[action](), status)

    async def upload(self, request: Request) -> Response:
        """Stream a recording to a temporary file and publish only a complete upload."""
        size = int(request.headers.get("content-length", "0"))
        filename = unquote(request.headers.get("x-filename", ""))
        path = await run_in_threadpool(self.upload_path, filename, size)
        partial = path.with_suffix(path.suffix + ".upload")
        try:
            await receive_video(request, partial, size)
            await run_in_threadpool(partial.replace, path)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(partial.unlink, missing_ok=True)
            raise
        return json_response({"path": str(path), "name": path.name}, HTTPStatus.CREATED)

    def upload_path(self, filename: str, size: int) -> Path:
        """Validate an upload and choose a unique path in the sample directory."""
        if not 0 < size <= UPLOAD_BODY_LIMIT:
            raise HTTPException(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "Choose a nonempty recording smaller than 100 GB.",
            )
        if Path(filename).name != filename or "\\" in filename or "/" in filename:
            message = "Invalid upload filename."
            raise ValueError(message)
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            message = "Choose an MP4, MKV, MOV, WebM, AVI or M4V recording."
            raise ValueError(message)
        stem = (
            re.sub(r"[^\w .-]", "_", Path(filename).stem).strip(" .")[:80]
            or "recording"
        )
        folder = self.workspace.root / "samples"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


async def receive_video(request: Request, partial: Path, remaining: int) -> None:
    """Write a bounded upload asynchronously and reject truncated body streams."""
    try:
        async with await anyio.open_file(partial, "xb") as output:
            async for chunk in request.stream():
                remaining -= len(chunk)
                if remaining < 0:
                    raise HTTPException(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        "Upload exceeds its declared length.",
                    )
                await output.write(chunk)
    except ClientDisconnect as error:
        message = "The upload was interrupted. Choose the file again to retry."
        raise ValueError(message) from error
    if remaining:
        message = "The upload was interrupted. Choose the file again to retry."
        raise ValueError(message)


def static_file(request: Request) -> Response:
    """Serve only the packaged application and explicitly supported static assets."""
    path = request.url.path
    if path in ("/", "/index.html"):
        path = "/index.html"
    elif not re.fullmatch(
        r"/(?:tiles/[A-Za-z0-9_-]+\.(?:svg|md)|assets/[A-Za-z0-9_-]+\.(?:js|css))", path
    ):
        return json_response({"error": "Page not found."}, HTTPStatus.NOT_FOUND)
    file = STATIC / path.lstrip("/")
    if not file.is_file():
        return json_response({"error": "Page not found."}, HTTPStatus.NOT_FOUND)
    return FileResponse(file)


def create_app(workspace: Workspace) -> ASGIApp:
    """Build the ASGI app; its lifespan owns shutdown of all workspace child jobs."""
    endpoints = WorkspaceRoutes(workspace)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(workspace.close)

    routes = [
        Route("/api/workspace", endpoints.status),
        Route("/api/projects", endpoints.create, methods=["POST"]),
        Route("/api/projects/{key}", endpoints.project),
        Route("/api/projects/{key}/results", endpoints.results),
        Route("/api/projects/{key}/{action}", endpoints.action, methods=["POST"]),
        Route("/api/upload", endpoints.upload, methods=["POST"]),
        Route("/exports/{key}/{name:path}", endpoints.export),
        review_routes(workspace),
        Route("/{path:path}", static_file),
    ]
    errors = dict.fromkeys(
        (
            HTTPException,
            KeyError,
            ValueError,
            TypeError,
            OSError,
            ClientDisconnect,
            Exception,
        ),
        request_error,
    )
    return LocalAccess(
        Starlette(routes=routes, lifespan=lifespan, exception_handlers=errors)
    )


@command_logging
def serve_workspace(root: Path, port: int = 8765, *, open_browser: bool = True) -> None:
    """Bind loopback explicitly and let Uvicorn manage the ASGI server lifecycle."""
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            message = (
                f"Cannot start the app on port {port}: {error}. "
                "Try --port with another number."
            )
            raise SystemExit(message) from error
        workspace = Workspace(root)
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(workspace),
                host="127.0.0.1",
                port=listener.getsockname()[1],
                access_log=False,
                proxy_headers=False,
                ws="none",
                http="h11",
                timeout_graceful_shutdown=5,
            )
        )
        url = f"http://localhost:{listener.getsockname()[1]}"
        LOGGER.info(
            "video2tenhou: %s\nKeep this window open while a recording is running.", url
        )
        if open_browser:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.run(sockets=[listener])
        finally:
            workspace.close()
