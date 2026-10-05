# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming limits, request isolation and shutdown through a real ASGI listener."""

from __future__ import annotations

import json
import threading
import time
from contextlib import closing
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.web.server import studio_server
from video2tenhou.tool.http import PROJECT_BODY_LIMIT
from video2tenhou.tool.review_routes import ReviewRoutes
from video2tenhou.tool.workflow import Job


def connection(url: str) -> HTTPConnection:
    """Connect directly to the test listener with a bounded response wait."""
    address = urlparse(url)
    assert address.hostname is not None
    return HTTPConnection(address.hostname, address.port, timeout=2)


def test_chunked_json_cannot_bypass_body_limit(tmp_path: Path) -> None:
    """Count streamed bytes even when no Content-Length was supplied."""
    with studio_server(tmp_path) as studio, closing(connection(studio.url)) as client:
        client.request(
            "POST",
            "/api/projects",
            body=iter([b" " * PROJECT_BODY_LIMIT, b" "]),
            headers={"X-Video2Tenhou": "1", "Content-Type": "application/json"},
            encode_chunked=True,
        )
        response = client.getresponse()
        assert response.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert "too large" in json.loads(response.read())["error"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert studio.workspace.projects == {}


def test_incomplete_json_does_not_block_workspace_reads(tmp_path: Path) -> None:
    """An awaiting request body must leave the ASGI event loop responsive."""
    with studio_server(tmp_path) as studio, closing(connection(studio.url)) as slow:
        slow.putrequest("POST", "/api/projects")
        slow.putheader("Content-Length", "32")
        slow.putheader("X-Video2Tenhou", "1")
        slow.endheaders(b"{")
        with closing(connection(studio.url)) as other:
            other.request("GET", "/api/workspace")
            response = other.getresponse()
            assert response.status == HTTPStatus.OK
            assert json.loads(response.read())["projects"] == []


def test_disconnected_upload_removes_only_its_partial_file(tmp_path: Path) -> None:
    """A client disconnect must not publish a video or delete an existing recording."""
    samples = tmp_path / "samples"
    samples.mkdir()
    existing = samples / "existing.mp4"
    existing.write_bytes(b"preserved recording")
    with studio_server(tmp_path) as studio:
        with closing(connection(studio.url)) as client:
            client.putrequest("POST", "/api/upload")
            client.putheader("Content-Length", "32")
            client.putheader("X-Video2Tenhou", "1")
            client.putheader("X-Filename", "interrupted.mp4")
            client.endheaders(b"part")
            deadline = time.monotonic() + 2
            while not list(samples.glob("*.upload")) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert list(samples.glob("*.upload")), "Upload never started"
        deadline = time.monotonic() + 2
        while list(samples.glob("*.upload")) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert list(samples.iterdir()) == [existing]
        assert existing.read_bytes() == b"preserved recording"
    assert studio.workspace.processes.closing


@pytest.mark.parametrize(
    "endpoint",
    [
        "hand/invalid",
        "hand/-1",
        "hand/0/extra",
        "decode/0/extra",
        "read/extra",
        "unknown",
    ],
)
def test_review_router_rejects_unmatched_paths(tmp_path: Path, endpoint: str) -> None:
    """Malformed review paths never reach project lookup or prefix dispatch."""
    with studio_server(tmp_path) as studio, closing(connection(studio.url)) as client:
        client.request("GET", f"/review/missing/api/{endpoint}")
        response = client.getresponse()
        assert response.status == HTTPStatus.NOT_FOUND
        assert json.loads(response.read()) == {"error": "Not Found"}
        assert studio.workspace.states == {}


@pytest.mark.parametrize(
    ("method", "endpoint", "allowed"),
    [("POST", "hands", {"GET", "HEAD"}), ("GET", "label", {"POST"})],
)
def test_review_router_rejects_methods_before_reading_body(
    tmp_path: Path, method: str, endpoint: str, allowed: set[str]
) -> None:
    """Starlette rejects unsupported methods without waiting for an unread body."""
    with studio_server(tmp_path) as studio, closing(connection(studio.url)) as client:
        client.putrequest(method, f"/review/missing/api/{endpoint}")
        client.putheader("X-Video2Tenhou", "1")
        client.putheader("Content-Length", "32")
        client.endheaders()
        response = client.getresponse()
        assert response.status == HTTPStatus.METHOD_NOT_ALLOWED
        assert set(response.headers["Allow"].split(", ")) == allowed
        assert json.loads(response.read()) == {"error": "Method Not Allowed"}
        assert studio.workspace.states == {}


def test_review_write_rechecks_jobs_after_receiving_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipeline starting during a slow upload must still prevent the mutation."""
    checked = threading.Event()
    original = ReviewRoutes.check_write

    def check_write(routes: ReviewRoutes, *, allow_review_job: bool) -> None:
        original(routes, allow_review_job=allow_review_job)
        checked.set()

    monkeypatch.setattr(ReviewRoutes, "check_write", check_write)
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"recording")
    with studio_server(tmp_path) as studio, closing(connection(studio.url)) as client:
        project = studio.workspace.create({"source": str(video), "games": [21938]})
        key = project["id"]
        client.putrequest("POST", f"/review/{key}/api/facts")
        client.putheader("X-Video2Tenhou", "1")
        client.putheader("Content-Length", "2")
        client.endheaders(b"{")
        assert checked.wait(timeout=2), "Initial job exclusion check never ran"
        job = Job(kind="analyze", project=key, stage="Reading tiles")
        with studio.workspace.lock:
            studio.workspace.job = job
        try:
            client.send(b"}")
            response = client.getresponse()
            assert response.status == HTTPStatus.CONFLICT
            assert json.loads(response.read())["error"] == (
                "Analysis is running. Wait for it to finish."
            )
            assert studio.workspace.states == {}
        finally:
            job.running = False
