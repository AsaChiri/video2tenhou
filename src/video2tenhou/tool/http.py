# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""ASGI transport policy for bounded, same-origin local workspace requests."""

from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from video2tenhou.files import sanitize

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

STATIC = Path(__file__).resolve().parent / "static"
PROJECT_BODY_LIMIT = 65536
REVIEW_BODY_LIMIT = 4 * 1024 * 1024
UPLOAD_BODY_LIMIT = 100 * 1024**3
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": "frame-ancestors 'self'",
}
IMMUTABLE = "public, max-age=31536000, immutable"


def json_response(obj: object, code: int = HTTPStatus.OK) -> JSONResponse:
    """Render finite browser-compatible JSON with the shared response policy."""
    return JSONResponse(sanitize(obj), status_code=code)


async def read_json_body(request: Request, limit: int) -> dict:
    """Read one bounded JSON object, checking streamed bytes as well as headers."""
    size = int(request.headers.get("content-length", "0"))
    if not 0 <= size <= limit:
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large."
        )
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise HTTPException(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large."
            )
        data.extend(chunk)
    body = json.loads(data or b"{}")
    if not isinstance(body, dict):
        raise TypeError("Expected a JSON object.")
    return body


class LocalAccess:
    """Enforce loopback host and write-origin policy around the entire ASGI app."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap framework responses, including failures, with security headers."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate browser access before consuming any request body."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.update(SECURITY_HEADERS)
                # API data, evidence and exports change in place; only static
                # assets choose to be cached.
                headers.setdefault("Cache-Control", "no-store")
            await send(message)

        error = self.access_error(scope)
        if error is not None:
            await json_response({"error": error}, HTTPStatus.FORBIDDEN)(
                scope, receive, send_headers
            )
            return
        await self.app(scope, receive, send_headers)

    @staticmethod
    def access_error(scope: Scope) -> str | None:
        """Require the listener's exact local authority and same-origin mutations."""
        headers = Headers(scope=scope)
        server = scope.get("server")
        port = server[1] if server is not None else None
        host = headers.get("host", "")
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return "Use the localhost address shown when starting the app."
        if scope["method"] not in ("GET", "HEAD") and (
            headers.get("origin") not in (None, f"http://{host}")
            or headers.get("x-video2tenhou") != "1"
        ):
            return "Only requests from this local app are accepted."
        return None
