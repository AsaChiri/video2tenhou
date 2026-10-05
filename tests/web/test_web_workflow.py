# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""HTTP integration for import → prepare → analysis → review → downloadable output.

Only the expensive child commands are replaced. Real TCP requests exercise
routing, body streaming, validation, persistent manifests and project isolation.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import threading
import time
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import cv2
import httpx
import numpy as np
import pytest

from tests.web.analysis import publish, write_analysis
from tests.web.server import LiveStudio, studio_server
from video2tenhou import cache, calibfit, cli, layout, tenhou6, video
from video2tenhou.calm import REGIONS, region_key
from video2tenhou.engine.decode import DECODER_VERSION
from video2tenhou.engine.review import load_facts
from video2tenhou.files import atomic_write_json
from video2tenhou.layout import CALIB_DIR, Calibration, fit_path
from video2tenhou.perception import classifier
from video2tenhou.perception.detector_metadata import InferenceOptions
from video2tenhou.tool import review_state as review
from video2tenhou.tool.workflow import ChildError, Job, JobKind, Workspace

# One TLS context for every request: building one per request loads the CA bundle.
TLS = ssl.create_default_context()


class Client:
    """HTTP client for the isolated local studio test server."""

    def __init__(self, server: LiveStudio) -> None:
        """Bind requests to the test server's loopback port."""
        self.base = server.url

    def request(
        self,
        path: str,
        body: dict | bytes | None = None,
        *,
        headers: dict[str, str] | None = None,
        raw: bool = False,
    ) -> tuple:
        """Send one request on a fresh connection; return status, data and headers."""
        data = (
            body
            if isinstance(body, bytes)
            else json.dumps(body).encode()
            if body is not None
            else None
        )
        request_headers = {
            "X-Video2Tenhou": "1",
            "Content-Type": "application/json",
            **(headers or {}),
        }
        response = httpx.request(
            "POST" if data is not None else "GET",
            self.base + path,
            content=data,
            headers=request_headers,
            timeout=10,
            trust_env=False,
            verify=TLS,
        )
        return (
            response.status_code,
            response.content if raw else response.json(),
            response.headers,
        )


def option(args: list[str], name: str) -> list[str]:
    """Return the values following every occurrence of an option."""
    values, taking = [], False
    for value in args:
        if value.startswith("--"):
            taking = value == name
        elif taking:
            values.append(value)
    return values


@pytest.fixture
def web(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Client, Workspace, list[list[str]]]]:
    """Run an isolated studio server whose child commands write small outputs."""
    monkeypatch.setattr(review, "ROOT", tmp_path)
    monkeypatch.setattr(layout, "LABEL_DIR", tmp_path / "labels")
    commands = []

    def runner(args: list[str], _job: Job) -> object:
        commands.append(args)
        command = args[4]
        if command in ("download", "trim"):
            Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(args[-1]).write_bytes(b"downloaded video")
        elif command == "calib":
            name = Path(args[6]).stem
            if args[5] == "fit":
                atomic_write_json(
                    tmp_path / "labels" / name / "calib.json",
                    {"video": name, "layout": "pml", "overhead": {}},
                )
            return {"pond:TL": {"level": "ok", "held": 3, "cut": 0, "note": ""}}
        elif command == "convert":
            name = Path(args[5]).stem
            write_analysis(tmp_path, name, [int(g) for g in option(args, "--game")])
            for marker in ("calibration.changed", "inputs.changed"):
                (tmp_path / "work" / name / marker).unlink(missing_ok=True)
        elif command == "rebuild":
            hands = [int(h) for h in option(args, "--hands")]
            publish(tmp_path, Path(args[5]).stem, hands)
        return None

    with studio_server(tmp_path, runner=runner) as server:
        yield Client(server), server.workspace, commands


def wait_for_job(client: Client, key: str) -> dict:
    """Poll a project until its preparation or analysis finishes."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status, result, _ = client.request(f"/api/projects/{key}")
        assert status == HTTPStatus.OK
        if not result["job"]["running"]:
            return result
        time.sleep(0.01)
    pytest.fail("The background job did not finish.")


def wait_for_workspace_job(client: Client) -> dict:
    """Poll the workspace job until it finishes."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.request("/api/job")[1]["job"]
        if job and not job["running"]:
            return job
        time.sleep(0.01)
    pytest.fail("The workspace job did not finish.")


def create_local(
    client: Client, workspace: Workspace, name: str = "recording.mp4"
) -> dict:
    """Create a local recording and import it into the studio workspace."""
    path = workspace.root / "samples" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"recording")
    status, project, _ = client.request(
        "/api/projects", {"source": str(path), "games": [21938, 21939]}
    )
    assert status == HTTPStatus.CREATED
    return project


def analyzed(client: Client, workspace: Workspace, name: str = "a.mp4") -> dict:
    """Create, prepare and analyze a project with two one-hand games."""
    project = create_local(client, workspace, name)
    for action in ("prepare", "analyze"):
        assert (
            client.request(f"/api/projects/{project['id']}/{action}", {})[0]
            == HTTPStatus.ACCEPTED
        )
        assert not wait_for_job(client, project["id"])["job"].get("error")
    return project


@contextmanager
def running(workspace: Workspace, kind: JobKind, key: str) -> Iterator[Job]:
    """Occupy the workspace job slot for the duration of a block."""
    job = Job(kind=kind, project=key, stage="Working")
    workspace.job = job
    try:
        yield job
    finally:
        job.running = False


def test_review_items_carry_ids_and_certified_confidence_without_changing_evidence(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Review items gain stable ids and margins from their confidence rows."""
    client, workspace, _ = web
    project = analyzed(client, workspace)
    state = workspace.review_state(project["id"])
    items = [
        {"kind": "draw", "seat": "N", "j": 0, "tile": "1m"},
        {"kind": "discard", "seat": "N", "j": 0, "tile": "2p"},
        {"kind": "result", "seat": "N", "t": 80.6, "tiles": ["1m"] * 13},
        {"kind": "draw", "seat": "N", "j": 1, "margin": 0.25},
        {"kind": "discard", "seat": "N"},  # No turn identity: borrow no row.
        {"kind": "call", "seat": "N", "alternative_gap": 0.01},
    ]
    confidence = [
        {"field": "draw", "seat": "E", "turn": 0, "margin": 0.9, "lost": False},
        {"field": "draw", "seat": "N", "turn": 0, "margin": 0, "lost": True},
        {"field": "discard", "seat": "N", "turn": 0, "margin": 0.1, "lost": False},
        {"field": "draw", "seat": "N", "turn": 1, "margin": 0.4, "lost": False},
        {"field": "discard", "seat": "N", "turn": None, "margin": 0.2, "lost": True},
    ]
    path = state.decode_path(0)
    original = json.dumps(
        {"decoder_version": DECODER_VERSION, "items": items, "confidence": confidence}
    ).encode()
    path.write_bytes(original)
    status, rows, _ = client.request(f"/review/{project['id']}/api/items")
    assert status == HTTPStatus.OK
    assert path.read_bytes() == original
    assert [row["id"] for row in rows] == [
        "draw:N:0",
        "discard:N:0",
        "result:N:81",
        "draw:N:1",
        "discard:N:",
        "call:N:",
    ]
    assert all(row["hand"] == 0 for row in rows)
    assert [(row.get("margin"), row.get("lost")) for row in rows[:4]] == [
        (0, True),
        (0.1, False),
        (None, None),
        (0.25, False),
    ]
    assert "margin" not in rows[4]
    assert "margin" not in rows[5]


def test_prepare_storage_failure_is_retryable_through_http(
    web: tuple[Client, Workspace, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest that cannot be saved leaves a retryable, non-running job."""
    client, workspace, commands = web
    key = create_local(client, workspace)["id"]
    manifest = workspace.projects_dir / f"{key}.json"
    before = manifest.read_bytes()
    original = Path.replace

    def denied(source: Path, target: Path) -> Path:
        if target == manifest:
            raise PermissionError("manifest temporarily unavailable")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", denied)
        status, error, _ = client.request(f"/api/projects/{key}/prepare", {})
        assert status == HTTPStatus.BAD_REQUEST
        assert "Could not save the project" in error["error"]
        current = client.request(f"/api/projects/{key}")[1]
        assert current["status"] == "failed"
        assert not current["job"]["running"]
        assert not commands
        assert manifest.read_bytes() == before
    assert client.request(f"/api/projects/{key}/prepare", {})[0] == HTTPStatus.ACCEPTED
    assert wait_for_job(client, key)["status"] == "ready"
    assert len(commands) == 1


def test_pending_update_selects_changed_hands_and_preserves_others(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Additions and deletions select their hands; a failure keeps them pending."""
    client, workspace, commands = web
    project = create_local(client, workspace)
    write_analysis(workspace.root, project["name"], [21938], hands_per_game=3)
    workspace.project(project["id"])["export_signature"] = None
    prefix = f"/review/{project['id']}/api"
    state = workspace.review_state(project["id"])
    untouched = state.decode_path(1).read_bytes()
    answer = {"hand": 0, "kind": "draw", "seat": "N", "tile": "2p", "j": 0}
    deleted = client.request(prefix + "/facts", {**answer, "hand": 2})[1]
    publish(workspace.root, project["name"], [2])  # Hand 2 used that answer.
    assert client.request(prefix + "/facts", answer)[0] == HTTPStatus.OK
    assert client.request(prefix + "/facts/delete", {"ts": deleted["ts"]})[1] == {
        "deleted": 1
    }
    rows = client.request(prefix + "/hands")[1]
    assert [h["hand"] for h in rows if h["pending"]] == [0, 2]
    # Undoing a change needs no update: the decode already matches.
    undone = client.request(prefix + "/facts", {**answer, "hand": 1})[1]
    client.request(prefix + "/facts/delete", {"ts": undone["ts"]})
    rows = client.request(prefix + "/hands")[1]
    assert [h["hand"] for h in rows if h["pending"]] == [0, 2]
    facts_before = (state.labels / "facts.jsonl").read_bytes()
    entered, release = threading.Event(), threading.Event()
    original = workspace.runner

    def blocking(args: list[str], job: Job) -> object:
        entered.set()
        assert release.wait(5)
        return original(args, job)

    workspace.runner = blocking
    try:
        status, started, _ = client.request(prefix + "/rebuild", {"hands": "pending"})
        assert status == HTTPStatus.OK
        assert started["job"]["hands"] == [0, 2]
        assert started["job"]["running"]
        assert entered.wait(5)
        assert client.request(prefix + "/calib", {})[0] == HTTPStatus.CONFLICT
        assert client.request("/api/job")[1]["job"]["kind"] == "rebuild"
    finally:
        release.set()
    job = wait_for_workspace_job(client)
    assert job["error"] is None
    assert option(commands[-1], "--hands") == ["0", "2"]
    assert "--force" not in commands[-1]
    assert not any(h["pending"] for h in client.request(prefix + "/hands")[1])
    assert state.decode_path(1).read_bytes() == untouched
    assert (state.labels / "facts.jsonl").read_bytes() == facts_before
    count = len(commands)
    idle = client.request(prefix + "/rebuild", {"hands": "pending"})[1]
    assert not idle["job"]["running"]
    assert len(commands) == count  # Nothing pending: no child.


def test_failed_update_keeps_answers_pending_and_listed_hands_are_forced(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """A failed child shows its own message; explicit rebuilds decode afresh."""
    client, workspace, commands = web
    project = create_local(client, workspace)
    write_analysis(workspace.root, project["name"], [21938], hands_per_game=3)
    prefix = f"/review/{project['id']}/api"
    answer = {"hand": 2, "kind": "draw", "seat": "N", "tile": "2p", "j": 0}
    assert client.request(prefix + "/facts", answer)[0] == HTTPStatus.OK
    message = "Hand 3 has no tile readings. Choose Analyze recording to read it."
    original = workspace.runner

    def failing(_args: list[str], _job: Job) -> object:
        raise ChildError(message)

    workspace.runner = failing
    assert client.request(prefix + "/rebuild", {"hands": "pending"})[0] == 200
    job = wait_for_workspace_job(client)
    assert job["error"] == message
    assert job["hands"] == [2]
    pending = [h["hand"] for h in client.request(prefix + "/hands")[1] if h["pending"]]
    assert pending == [2]
    workspace.runner = original
    assert client.request(prefix + "/rebuild", {"hands": [1]})[0] == HTTPStatus.OK
    wait_for_workspace_job(client)
    assert option(commands[-1], "--hands") == ["1"]
    assert "--force" in commands[-1]
    assert client.request(prefix + "/rebuild", {"hands": [7]})[0] == 400
    assert client.request(prefix + "/rebuild", {"hands": "some"})[0] == 400


def test_project_library_rename_delete_and_restart(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Renaming and deleting a project never touches its recordings or evidence."""
    client, workspace, _ = web
    project = create_local(client, workspace)
    key = project["id"]
    paths = [workspace.root / "samples" / "recording.mp4"]
    for folder in ("work", "labels", "out"):
        path = workspace.root / folder / project["name"] / "keep.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep this evidence", encoding="utf-8")
        paths.append(path)
    before = [path.read_bytes() for path in paths]
    stored = workspace.project(key)
    status, renamed, _ = client.request(
        f"/api/projects/{key}/rename", {"display_name": "  Final table  "}
    )
    assert status == HTTPStatus.OK
    assert renamed["display_name"] == "Final table"
    assert renamed["name"] == project["name"]
    assert workspace.project(key) is stored  # Running workers retain this object.
    assert Workspace(workspace.root).snapshot(key)["display_name"] == "Final table"
    assert (
        client.request("/api/workspace")[1]["projects"][0]["display_name"]
        == "Final table"
    )
    assert client.request(f"/api/projects/{key}/delete", {})[0] == HTTPStatus.OK
    assert client.request(f"/api/projects/{key}")[0] == HTTPStatus.NOT_FOUND
    assert client.request("/api/workspace")[1]["projects"] == []
    assert Workspace(workspace.root).projects == {}
    assert [path.read_bytes() for path in paths] == before
    assert client.request(f"/api/projects/{key}/delete", {})[0] == HTTPStatus.NOT_FOUND


@pytest.mark.parametrize("name", ["", "   ", "x" * 121, None, 123])
def test_project_rename_validates_name(
    web: tuple[Client, Workspace, list[list[str]]], name: str
) -> None:
    client, workspace, _ = web
    project = create_local(client, workspace)
    status, _, _ = client.request(
        f"/api/projects/{project['id']}/rename", {"display_name": name}
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert workspace.snapshot(project["id"])["display_name"] == project["name"]


@pytest.mark.parametrize("kind", ["analyze", "rebuild"])
def test_project_delete_rejects_its_running_job(
    web: tuple[Client, Workspace, list[list[str]]], kind: JobKind
) -> None:
    """A project whose job runs cannot be deleted."""
    client, workspace, _ = web
    key = create_local(client, workspace)["id"]
    with running(workspace, kind, key):
        status, error, _ = client.request(f"/api/projects/{key}/delete", {})
        assert status == HTTPStatus.BAD_REQUEST
        assert "running job" in error["error"]
    assert (workspace.projects_dir / f"{key}.json").exists()


def test_project_mutation_storage_failure_retains_manifest_and_memory(
    web: tuple[Client, Workspace, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed manifest writes change neither the file nor the served project."""
    client, workspace, _ = web
    key = create_local(client, workspace)["id"]
    manifest = workspace.projects_dir / f"{key}.json"
    before = manifest.read_bytes()
    original_name = workspace.snapshot(key)["display_name"]

    def denied(*_unused_args: object, **_unused_kwargs: object) -> None:
        raise PermissionError("Storage unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", denied)
        assert (
            client.request(f"/api/projects/{key}/rename", {"display_name": "New"})[0]
            == HTTPStatus.BAD_REQUEST
        )
        assert (
            client.request(f"/api/projects/{key}/settings", {"games": [7]})[0]
            == HTTPStatus.BAD_REQUEST
        )
    assert workspace.snapshot(key)["display_name"] == original_name
    assert workspace.snapshot(key)["games"] == [21938, 21939]
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        assert (
            client.request(f"/api/projects/{key}/delete", {})[0]
            == HTTPStatus.BAD_REQUEST
        )
    assert manifest.read_bytes() == before
    assert key in workspace.projects


def test_import_prepare_analyze_review_and_export(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, commands = web
    data = b"video bytes" * 200000  # crosses the streaming chunk boundary
    status, upload, _ = client.request(
        "/api/upload", data, headers={"X-Filename": "broadcast.mp4"}
    )
    assert status == HTTPStatus.CREATED
    assert Path(upload["path"]).read_bytes() == data
    status, project, _ = client.request(
        "/api/projects", {"source": upload["path"], "games": [21938, 21939]}
    )
    assert status == HTTPStatus.CREATED
    key = project["id"]
    assert (
        client.request(f"/api/projects/{key}/analyze", {})[0] == HTTPStatus.BAD_REQUEST
    )
    assert client.request(f"/api/projects/{key}/prepare", {})[0] == HTTPStatus.ACCEPTED
    ready = wait_for_job(client, key)
    assert ready["status"] == "ready"
    assert ready["has_fit"]
    assert client.request(f"/api/projects/{key}/analyze", {})[0] == HTTPStatus.ACCEPTED
    done = wait_for_job(client, key)
    assert done["status"] == "complete"
    assert done["open_items"] == 0
    assert "g0.json" in done["artifacts"]
    assert "log" not in done["job"]
    assert client.request(f"/review/{key}/api/hands")[1][0]["hand"] == 0
    status, log, headers = client.request(f"/exports/{key}/g0.json")
    assert status == HTTPStatus.OK
    assert log == {"log": []}
    assert headers["Content-Disposition"] == 'attachment; filename="g0.json"'
    assert headers["Cache-Control"] == "no-store"
    assert (
        client.request(f"/exports/{key}/g0.html", raw=True)[1]
        == b"<h1>Replay links</h1>"
    )
    assert "--skip-fit-check" not in commands[-1]
    assert commands[-1][-4:] == ["--game", "21938", "--game", "21939"]
    assert commands[0][4:6] == ["calib", "fit"]
    reopened = Workspace(workspace.root)
    assert reopened.snapshot(key)["status"] == "complete"


@pytest.mark.parametrize(
    "source",
    [
        "https://www.twitch.tv/videos/123",
        "https://www.twitch.tv/videos/123?foo=bar#fragment",
        "https://www.youtube.com/watch?v=example",
        "https://vimeo.com/123",
        "http://example.test/video.mp4?token=abc",
    ],
)
def test_url_preparation_delegates_to_downloader(
    web: tuple[Client, Workspace, list[list[str]]], source: str
) -> None:
    client, workspace, commands = web
    status, p, _ = client.request(
        "/api/projects", {"source": source, "kind": "url", "games": [1]}
    )
    assert status == HTTPStatus.CREATED
    status, response, _ = client.request(f"/api/projects/{p['id']}/prepare", {})
    assert status == HTTPStatus.ACCEPTED, response
    assert wait_for_job(client, p["id"])["status"] == "ready"
    stored = workspace.project(p["id"])
    assert commands[0][-4:] == ["download", "--", source, stored["video"]]
    assert Path(stored["video"]).parent == workspace.root / "samples"
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    wait_for_job(client, p["id"])
    assert sum("download" in command for command in commands) == 1


def test_url_recording_identity_includes_query(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web

    def create(source: str) -> tuple:
        return client.request(
            "/api/projects", {"source": source, "kind": "url", "games": [1]}
        )

    first = "https://www.youtube.com/watch?v=first"
    status, p, _ = create(first)
    assert status == HTTPStatus.CREATED
    status, other, _ = create("https://www.youtube.com/watch?v=second")
    assert status == HTTPStatus.CREATED
    assert (
        workspace.project(p["id"])["video"] != workspace.project(other["id"])["video"]
    )
    assert create(first)[0] == 400
    assert create(" ")[0] == 400


@pytest.mark.parametrize("kind", ["local", "url"])
@pytest.mark.parametrize(
    ("start", "end"), [("01:30", "02:00"), ("01:30", ""), ("", "02:00")]
)
def test_selected_range_is_prepared_once_and_persisted(
    web: tuple[Client, Workspace, list[list[str]]], kind: str, start: str, end: str
) -> None:
    client, workspace, commands = web
    original = workspace.root / "original.mp4"
    original.write_bytes(b"original remains intact")
    source = str(original) if kind == "local" else "https://example.test/video?id=7"
    status, p, _ = client.request(
        "/api/projects",
        {"source": source, "kind": kind, "games": [1], "start": start, "end": end},
    )
    assert status == HTTPStatus.CREATED
    assert (p["start"], p["end"]) == (90 if start else 0, 120 if end else None)
    video = workspace.project(p["id"])["video"]
    assert video != source
    assert not p["has_fit"]
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    ready = wait_for_job(client, p["id"])
    assert ready["status"] == "ready"
    assert ready["has_fit"]
    operation = "trim" if kind == "local" else "download"
    cmd = commands[0]
    assert operation in cmd
    assert cmd[-3:] == ["--", source, video]
    assert ("--start" in cmd) == bool(start)
    assert ("--end" in cmd) == bool(end)
    if start:
        assert float(cmd[cmd.index("--start") + 1]) == 90
    if end:
        assert float(cmd[cmd.index("--end") + 1]) == 120
    assert original.read_bytes() == b"original remains intact"
    reloaded = Workspace(workspace.root).snapshot(p["id"])
    assert (reloaded["start"], reloaded["end"]) == (p["start"], p["end"])
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    prepared_again = wait_for_job(client, p["id"])
    assert prepared_again["status"] == "ready", prepared_again
    assert sum(operation in command for command in commands) == 1


@pytest.mark.parametrize("kind", ["local", "url"])
def test_ranges_have_distinct_projects_but_equivalent_time_formats_do_not(
    web: tuple[Client, Workspace, list[list[str]]], kind: str
) -> None:
    client, workspace, _ = web
    source = workspace.root / "original.mp4"
    source.write_bytes(b"original")
    body = {
        "source": str(source) if kind == "local" else "https://example.test/video",
        "kind": kind,
        "games": [1],
    }
    full = client.request("/api/projects", body)[1]
    first = client.request("/api/projects", {**body, "start": "60", "end": "120"})[1]
    second = client.request("/api/projects", {**body, "start": "120", "end": "180"})[1]
    assert len({p["name"] for p in (full, first, second)}) == 3
    assert (
        client.request("/api/projects", {**body, "start": "01:00", "end": "00:02:00"})[
            0
        ]
        == HTTPStatus.BAD_REQUEST
    )


@pytest.mark.parametrize(
    "bounds",
    [
        {"start": "-1"},
        {"end": "0"},
        {"start": "02:00", "end": "01:00"},
        {"start": "bad"},
        {"end": "inf"},
        {"start": True},
        {"end": []},
    ],
)
def test_invalid_ranges_do_not_create_projects(
    web: tuple[Client, Workspace, list[list[str]]], bounds: dict
) -> None:
    client, workspace, commands = web
    status, data, _ = client.request(
        "/api/projects",
        {"source": "https://example.test/video", "kind": "url", "games": [1], **bounds},
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert data["error"]
    assert not workspace.projects
    assert not commands


def test_failed_range_preparation_can_retry_without_changing_original(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, commands = web
    original = workspace.root / "original.mp4"
    original.write_bytes(b"original")
    status, p, _ = client.request(
        "/api/projects",
        {"source": str(original), "games": [1], "start": "10", "end": "20"},
    )
    assert status == HTTPStatus.CREATED
    runner = workspace.runner

    def interrupted(_args: list[str], _job: Job) -> None:
        raise ChildError("The selected time range could not be cut from the recording.")

    workspace.runner = interrupted
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    failed = wait_for_job(client, p["id"])
    assert failed["status"] == "failed"
    assert failed["job"]["error"] == (
        "The selected time range could not be cut from the recording."
    )
    assert not failed["has_fit"]
    assert not Path(workspace.project(p["id"])["video"]).exists()
    assert not commands
    workspace.runner = runner
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    assert wait_for_job(client, p["id"])["status"] == "ready"
    assert "trim" in commands[0]
    assert original.read_bytes() == b"original"


@pytest.mark.parametrize("kind", ["local", "url"])
def test_fractional_bounds_round_trip_through_preparation_cli(
    web: tuple[Client, Workspace, list[list[str]]],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    client, workspace, _ = web
    original = workspace.root / "original.mp4"
    original.write_bytes(b"original")
    source = str(original) if kind == "local" else "https://example.test/video"
    parsed = []

    def record_bounds(source: Path, out: Path, start: float, end: float) -> None:
        parsed.append(video.time_range(start, end))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"clip")

    operation = "trim" if kind == "local" else "download"
    monkeypatch.setattr(video, operation, record_bounds)
    runner = workspace.runner

    def run(args: list[str], job: Job) -> object:
        if operation in args:
            assert cli.main(args[4:]) == 0
            return None
        return runner(args, job)

    workspace.runner = run
    status, p, _ = client.request(
        "/api/projects",
        {
            "source": source,
            "kind": kind,
            "games": [1],
            "start": "0.00001",
            "end": "1.00001",
        },
    )
    assert status == HTTPStatus.CREATED
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    assert wait_for_job(client, p["id"])["status"] == "ready"
    assert parsed == [(0.00001, 1.00001)]


def test_native_results_use_current_json_and_shared_replay_links(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Serve whole-game and single-hand links without the CLI HTML artifact.

    Rebuilding a JSON export changes the project's results revision, and changing
    project inputs hides both the file download and native results.
    """
    client, workspace, _ = web
    p = analyzed(client, workspace)
    key = p["id"]
    output = workspace.root / "out" / p["name"]
    game = tenhou6.Game(
        names=['A<&"', "B", "C", "D"],
        title=["Fixture", ""],
        kyokus=[
            tenhou6.Kyoku(3, 1, 0, [25000] * 4, result=tenhou6.Ryukyoku()),
            tenhou6.Kyoku(4, 0, 0, [25000] * 4, result=tenhou6.Ryukyoku()),
        ],
    )
    (output / "g0.json").write_text(game.dumps(), encoding="utf-8")
    (output / "g0.html").unlink()
    revision = client.request(f"/api/projects/{key}")[1]["results_revision"]
    status, result, _ = client.request(f"/api/projects/{key}/results")
    assert status == HTTPStatus.OK
    assert result["pending_games"] == []
    row = result["games"][0]
    assert row["names"] == game.names
    assert row["record_id"] == p["games"][0]
    assert row["viewer_url"] == game.viewer_url()
    assert [(h["round"], h["honba"]) for h in row["hands"]] == [
        ("East 4", 1),
        ("South 1", 0),
    ]
    for i, hand in enumerate(row["hands"]):
        assert hand["editor_url"] == game.editor_url(i)
        payload = json.loads(unquote(hand["editor_url"].split("#json=", 1)[1]))
        assert payload["log"] == [game.kyokus[i].dump()]
    assert client.request(row["download"])[1] == game.to_dict()
    game.kyokus.append(tenhou6.Kyoku(4, 1, 0, [25000] * 4, result=tenhou6.Ryukyoku()))
    (output / "g0.json").write_text(game.dumps(), encoding="utf-8")
    assert client.request(f"/api/projects/{key}")[1]["results_revision"] != revision
    assert (
        len(client.request(f"/api/projects/{key}/results")[1]["games"][0]["hands"]) == 3
    )
    status, _, _ = client.request(
        f"/review/{key}/api/facts",
        {"hand": 1, "kind": "draw", "seat": "N", "j": 0, "t": 10, "tile": "2p"},
    )
    assert status == HTTPStatus.OK
    assert client.request(f"/api/projects/{key}/results")[1]["pending_games"] == [1]
    other = create_local(client, workspace, "other.mp4")
    assert client.request(f"/api/projects/{other['id']}/results")[1]["games"] == []
    assert (
        client.request(f"/api/projects/{key}/settings", {"games": [42]})[0]
        == HTTPStatus.OK
    )
    assert client.request(f"/api/projects/{key}/results")[1]["games"] == []
    assert client.request(row["download"])[0] == HTTPStatus.NOT_FOUND


def test_review_facts_are_project_scoped(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    one, two = [analyzed(client, workspace, name)["id"] for name in ("a.mp4", "b.mp4")]
    status, fact, _ = client.request(
        f"/review/{one}/api/facts",
        {"hand": 0, "kind": "draw", "seat": "N", "t": 20, "tile": "2p"},
    )
    assert status == HTTPStatus.OK
    assert fact["corner"] == "BR"
    assert client.request(f"/review/{one}/api/facts")[1][0]["tile"] == "2p"
    assert client.request(f"/review/{two}/api/facts")[1] == []
    assert client.request(f"/review/{one}/api/facts/delete", {"ts": fact["ts"]})[1] == {
        "deleted": 1
    }
    assert client.request(f"/review/{one}/api/facts")[1] == []
    status, error, _ = client.request(
        f"/review/{one}/api/facts", {"hand": 0, "kind": "note", "text": "OK"}
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert error["error"] == "Unsupported answer."


def test_cross_origin_traversal_and_upload_validation(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    assert (
        client.request("/api/projects", {}, headers={"Origin": "https://evil.test"})[0]
        == HTTPStatus.FORBIDDEN
    )
    assert (
        client.request("/api/projects", {}, headers={"X-Video2Tenhou": ""})[0]
        == HTTPStatus.FORBIDDEN
    )
    assert (
        client.request("/api/workspace", headers={"Host": "evil.test"})[0]
        == HTTPStatus.FORBIDDEN
    )
    assert (
        client.request("/api/upload", b"x", headers={"X-Filename": "../secret.mp4"})[0]
        == HTTPStatus.BAD_REQUEST
    )
    assert (
        client.request("/api/upload", b"x", headers={"X-Filename": "run.exe"})[0]
        == HTTPStatus.BAD_REQUEST
    )
    assert (
        client.request("/api/upload", b"", headers={"X-Filename": "empty.mp4"})[0]
        == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    )
    p = create_local(client, workspace)
    assert (
        client.request(f"/exports/{p['id']}/%2e%2e%2fsecret.txt")[0]
        == HTTPStatus.BAD_REQUEST
    )
    assert client.request("/../pyproject.toml")[0] == HTTPStatus.NOT_FOUND
    assert (
        client.request(f"/review/{p['id']}/../../studio.html")[0]
        == HTTPStatus.NOT_FOUND
    )
    assert client.request("/api/projects/" + "0" * 32)[0] == HTTPStatus.NOT_FOUND
    assert (
        client.request(
            "/api/projects",
            {"source": str(workspace.root / "samples" / "recording.mp4"), "games": [1]},
        )[0]
        == HTTPStatus.BAD_REQUEST
    )


def test_failed_and_interrupted_jobs_remain_actionable(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """A failure survives a restart; an unfinished job is reported interrupted."""
    client, workspace, _ = web
    p = create_local(client, workspace)

    def fail(_args: object, _job: object) -> None:
        raise RuntimeError("Detector weights missing")

    workspace.runner = fail
    client.request(f"/api/projects/{p['id']}/prepare", {})
    failed = wait_for_job(client, p["id"])
    assert failed["status"] == "failed"
    assert failed["job"]["error"] == (
        "Preparation failed unexpectedly: Detector weights missing. The processing "
        "log has details."
    )
    assert (
        Workspace(workspace.root).snapshot(p["id"])["job"]["error"]
        == failed["job"]["error"]
    )
    manifest = workspace.projects_dir / f"{p['id']}.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    saved["job"]["running"] = True
    manifest.write_text(json.dumps(saved), encoding="utf-8")
    restarted = Workspace(workspace.root).snapshot(p["id"])
    assert restarted["status"] == "interrupted"
    assert not restarted["job"]["running"]


def child(script: str) -> list[str]:
    """Build a child command that runs a Python script with the CLI's outcome line."""
    return [sys.executable, "-u", "-c", script]


def test_child_failure_shows_its_reported_message_and_keeps_the_log(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """The banner shows the command's own message; progress stays in the log."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    message = "Adjust the table borders in Calibration: meld:TL cuts tiles."
    script = (
        "import json, sys; print('FAIL meld:TL 0 tiles held, 3 cut', file=sys.stderr);"
        f" print(json.dumps({{'error': {message!r}}})); raise SystemExit(1)"
    )
    workspace.runner = lambda _args, job: workspace.run_child(child(script), job)
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    failed = wait_for_job(client, p["id"])
    assert failed["status"] == "failed"
    assert failed["job"]["error"] == message
    assert client.request(f"/api/projects/{p['id']}/log")[1] == {
        "log": ["FAIL meld:TL 0 tiles held, 3 cut"]
    }
    workspace.runner = lambda _args, job: workspace.run_child(
        child("raise SystemExit(7)"), job
    )
    client.request(f"/api/projects/{p['id']}/prepare", {})
    crashed = wait_for_job(client, p["id"])
    assert crashed["job"]["error"] == (
        "Preparation failed unexpectedly: the command stopped with exit code 7. "
        "The processing log has details."
    )


def test_cli_reports_one_outcome_and_studio_phrased_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every command ends with one JSON document; tracebacks stay on stderr."""
    work = tmp_path / "work"
    (work / "recording").mkdir(parents=True)
    (work / "recording" / "inputs.changed").write_text("Analyze again.\n")
    code = cli.main(["rebuild", "recording.mp4", "--work", str(work)])
    captured = capsys.readouterr()
    assert code == 1
    assert json.loads(captured.out) == {
        "error": "The table geometry or project settings changed. Choose Analyze "
        "recording to refresh the readings before updating hands."
    }
    (work / "recording" / "inputs.changed").unlink()
    assert cli.main(["rebuild", "recording.mp4", "--work", str(work)]) == 1
    captured = capsys.readouterr()
    outcome = json.loads(captured.out)
    assert outcome["unexpected"] is True
    assert "hands.json" in outcome["error"]
    assert "Traceback" in captured.err


def test_jobs_are_serialized_and_review_writes_wait(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    p = create_local(client, workspace)
    started, release = threading.Event(), threading.Event()

    def slow(_args: object, _job: object) -> None:
        started.set()
        assert release.wait(5)

    workspace.runner = slow
    try:
        assert (
            client.request(f"/api/projects/{p['id']}/prepare", {})[0]
            == HTTPStatus.ACCEPTED
        )
        assert started.wait(2)
        status, error, _ = client.request(f"/api/projects/{p['id']}/prepare", {})
        assert status == HTTPStatus.BAD_REQUEST
        assert error["error"] == "A recording is being prepared. Wait for it to finish."
        assert (
            client.request(f"/review/{p['id']}/api/facts", {})[0] == HTTPStatus.CONFLICT
        )
        job = client.request("/api/job")[1]["job"]
        assert job["running"]
        assert job["stage"] == "Measuring table layout"
        assert "log" not in job
    finally:
        release.set()
        wait_for_job(client, p["id"])


def test_studio_assets_are_cacheable_and_api_responses_are_not(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Vendored tiles may be cached; pages and data are always fetched again."""
    client, workspace, _ = web
    status, html, headers = client.request("/", raw=True)
    assert status == HTTPStatus.OK
    assert b'id="app"' in html
    assert headers["Cache-Control"] == "no-store"
    status, svg, headers = client.request("/tiles/Pin2.svg", raw=True)
    assert status == HTTPStatus.OK
    assert b"<svg" in svg
    assert "image/svg+xml" in headers["Content-Type"]
    assert headers["Cache-Control"] == "public, max-age=31536000, immutable"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert client.request("/tiles/LICENSE.md", raw=True)[0] == HTTPStatus.OK
    assert client.request("/api/job")[2]["Cache-Control"] == "no-store"
    p = create_local(client, workspace)
    assert (
        client.request(f"/review/{p['id']}/?embedded=1", raw=True)[0]
        == HTTPStatus.NOT_FOUND
    )


def test_calibration_save_and_checks_follow_the_geometry(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Checks belong to the geometry they measured.

    Editing an analyzed recording's geometry blocks review updates until analysis.
    """
    client, workspace, _ = web
    p = analyzed(client, workspace)
    prefix = f"/review/{p['id']}/api"
    assert client.request(prefix + "/calib")[1]["checks"] == {
        "pond:TL": {"level": "ok", "held": 3, "cut": 0, "note": ""}
    }
    status, saved, _ = client.request(
        prefix + "/calib", {"overhead": {"center": [981, 543], "angle": 46}}
    )
    assert status == HTTPStatus.OK
    assert saved["fit"]["overhead"]["source"] == "human"
    geometry = client.request(prefix + "/calib")[1]
    assert geometry["fit"]["overhead"]["center"] == [981, 543]
    assert geometry["checks"] == {}
    state = workspace.review_state(p["id"])
    assert state.cal.center == (981, 543)
    assert (state.work / "calibration.changed").exists()
    assert client.request(prefix + "/calib/check", {})[0] == HTTPStatus.OK
    assert wait_for_workspace_job(client)["kind"] == "check"
    assert client.request(prefix + "/calib")[1]["checks"]["pond:TL"]["level"] == "ok"
    assert not client.request(f"/api/projects/{p['id']}")[1]["artifacts"]


def test_rebuild_command_refuses_changed_geometry(tmp_path: Path) -> None:
    """Review updates never mix new geometry with old readings."""
    work = tmp_path / "recording"
    work.mkdir()
    (work / "calibration.changed").write_text("Analyze.\n")
    args = Namespace(video="recording.mp4", work=str(tmp_path), hands=None)
    with pytest.raises(SystemExit, match="Choose Analyze recording"):
        cli.cmd_rebuild(args)


@pytest.mark.parametrize("partial_fit", [False, True])
def test_failed_prepare_can_save_table_calibration_then_retry(
    *, web: tuple[Client, Workspace, list[list[str]]], partial_fit: bool
) -> None:
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    video_path = workspace.project(key)["video"]
    attempts = []

    def runner(args: list[str], _job: Job) -> None:
        attempts.append(args)
        path = fit_path(video_path)
        if len(attempts) == 1:
            if partial_fit:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"overhead": {}}')
            raise ChildError(
                "Adjust the table borders in Calibration: hand:TL cuts tiles.",
                {"hand:TL": {"level": "fail"}},
            )
        cal = Calibration.load("pml", video_path)
        assert cal.center == (981, 543)
        assert cal.fit is not None
        assert cal.fit["overhead"]["source"] == "human"
        assert cal.hand["TL"][0].x == 15

    workspace.runner = runner
    assert client.request(f"/api/projects/{key}/prepare", {})[0] == HTTPStatus.ACCEPTED
    failed = wait_for_job(client, key)
    assert failed["can_calibrate"]
    assert not failed["has_fit"]
    assert failed["job"]["error"].endswith("hand:TL cuts tiles.")
    status, geometry, _ = client.request(f"/review/{key}/api/calib")
    assert status == HTTPStatus.OK
    assert geometry["overhead"]["center"] == [960, 540]
    assert geometry["checks"] == {"hand:TL": {"level": "fail"}}
    assert (
        client.request(
            f"/review/{key}/api/calib",
            {
                "overhead": {"center": [981, 543], "angle": 46, "scale": 1},
                "hand": {"TL": {"rect": [15, 140, 700, 340]}},
            },
        )[0]
        == HTTPStatus.OK
    )
    assert len(attempts) == 1
    assert (
        client.request(f"/api/projects/{key}/analyze", {})[0] == HTTPStatus.BAD_REQUEST
    )
    assert client.request(f"/api/projects/{key}/prepare", {})[0] == HTTPStatus.ACCEPTED
    ready = wait_for_job(client, key)
    assert ready["has_fit"]
    assert ready["status"] == "ready"
    assert len(attempts) == 2


def test_plate_requests_never_build_a_plate(
    web: tuple[Client, Workspace, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or stale preview asks for preparation instead of writing files."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    work = workspace.root / "work" / p["name"]
    status, error, _ = client.request(f"/review/{p['id']}/api/plate")
    assert status == HTTPStatus.CONFLICT
    assert error["error"] == "Prepare the recording to show the table preview."
    assert not work.exists()
    source = Path(workspace.project(p["id"])["video"])
    monkeypatch.setattr(
        calibfit.videomod, "probe", lambda _path: SimpleNamespace(duration=60.0)
    )
    monkeypatch.setattr(
        calibfit.videomod, "frame_at", lambda *_args: np.full((4, 4, 3), 90, np.uint8)
    )
    calibfit.table_plate(source, work)  # as preparation does
    status, jpeg, headers = client.request(f"/review/{p['id']}/api/plate", raw=True)
    assert status == HTTPStatus.OK
    assert headers["Content-Type"] == "image/jpeg"
    assert jpeg.startswith(b"\xff\xd8")
    source.write_bytes(b"replaced recording")
    assert client.request(f"/review/{p['id']}/api/plate")[0] == HTTPStatus.CONFLICT


def test_label_prefill_loads_an_eager_detector_once(
    web: tuple[Client, Workspace, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graph capture would keep a model per preview shape until the project closes."""
    client, workspace, _ = web
    state = workspace.review_state(create_local(client, workspace)["id"])
    loaded = []
    monkeypatch.setattr(
        review.detector, "Detector", lambda **kw: loaded.append(kw) or object()
    )
    monkeypatch.setattr(classifier, "Classifier", object)
    assert state.models() is state.models()
    assert loaded == [{"settings": InferenceOptions(cuda_graph=False)}]


def test_scoped_frame_and_clip_routes_return_this_projects_evidence(
    web: tuple[Client, Workspace, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, workspace, _ = web
    a, b = [create_local(client, workspace, name) for name in ("red.mp4", "blue.mp4")]
    for project, color in ((a, (0, 0, 255)), (b, (255, 0, 0))):
        state = workspace.review_state(project["id"])
        monkeypatch.setattr(
            state, "frame", lambda _t, c=color: np.full((24, 32, 3), c, np.uint8)
        )
    for project, channel in ((a, 2), (b, 0)):
        status, jpeg, headers = client.request(
            f"/review/{project['id']}/api/frame?t=5&region=frame", raw=True
        )
        assert status == HTTPStatus.OK
        assert headers["Content-Type"] == "image/jpeg"
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert frame is not None
        assert frame[:, :, channel].mean() > 250
    clip = workspace.root / "test.mp4"
    clip.write_bytes(b"test clip bytes")
    monkeypatch.setattr(
        workspace.review_state(a["id"]), "clip", lambda _start, _end, _region: clip
    )
    status, content, headers = client.request(
        f"/review/{a['id']}/api/clip?t0=1&t1=4&region=frame", raw=True
    )
    assert status == HTTPStatus.OK
    assert content == b"test clip bytes"
    assert headers["Content-Type"] == "video/mp4"


def test_review_rebuild_child_reports_its_own_error_and_refreshes_exports(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Run the real `video2tenhou rebuild` child: exports and its structured error.

    No vision is needed: the hands have no readings. This catches stale exports
    after review, wrong process arguments, wrong output roots and cached names.
    """
    client, workspace, _ = web
    p = analyzed(client, workspace)
    work = workspace.root / "work" / p["name"]
    rows = json.loads((work / "record.json").read_text())
    rows[0]["players"] = {"EAST": "Reviewed player"}
    (work / "record.json").write_text(json.dumps(rows))
    for decode in (work / "decode").glob("*.json"):
        decode.unlink()
    out = workspace.root / "out" / p["name"]
    (out / "review.json").write_text('[{"kind":"stale"}]')
    workspace.runner = workspace.run_child
    status, started, _ = client.request(
        f"/review/{p['id']}/api/rebuild", {"hands": "all"}
    )
    assert status == HTTPStatus.OK
    assert started["job"]["hands"] == [0, 1]
    job = wait_for_workspace_job(client)
    assert job["error"] == (
        "Hands 1, 2 have no tile readings. Choose Analyze recording to read the "
        "video again."
    )
    status, output, _ = client.request(f"/exports/{p['id']}/g0.json")
    assert status == HTTPStatus.OK
    assert output["name"][0] == "Reviewed player"
    review_items = client.request(f"/exports/{p['id']}/review.json")[1]
    assert [item["kind"] for item in review_items] == ["conflict", "conflict"]
    assert (out / "g0.html").exists()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "hanchan 1: 0 of 1 hands written" in report
    log = client.request(f"/api/projects/{p['id']}/log")[1]["log"]
    assert not any(line.startswith('{"error"') for line in log)


def test_cli_uses_selected_data_directory_for_stage_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.paths, "DATA_DIR", tmp_path)
    captured = []
    monkeypatch.setattr(cli, "cmd_convert", captured.append)
    assert cli.main(["convert", "recording.mp4", "--game", "21938"]) == 0
    assert captured[0].work == str(tmp_path / "work")
    assert captured[0].out == str(tmp_path / "out")


def test_settings_corrections_hide_old_exports_and_preserve_answers(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    p = analyzed(client, workspace)
    key = p["id"]
    facts = workspace.root / "labels" / p["name"] / "facts.jsonl"
    facts.write_text('{"kind":"note","text":"keep me"}\n')
    status, changed, _ = client.request(
        f"/api/projects/{key}/settings", {"games": [22002, 22003], "layout": "pml"}
    )
    assert status == HTTPStatus.OK
    assert changed["has_fit"]
    assert changed["stale_exports"]
    assert changed["artifacts"] == []
    assert client.request(f"/exports/{key}/g0.json")[0] == HTTPStatus.NOT_FOUND
    assert client.request(f"/review/{key}/api/hands")[1] == []
    assert "keep me" in facts.read_text()
    client.request(f"/api/projects/{key}/analyze", {})
    refreshed = wait_for_job(client, key)
    assert refreshed["artifacts"]
    assert not refreshed["stale_exports"]
    assert "keep me" in facts.read_text()


def test_layout_change_requires_preparation(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    custom = workspace.root / "custom.json"
    custom.write_bytes((CALIB_DIR / "pml.json").read_bytes())
    status, changed, _ = client.request(
        f"/api/projects/{key}/settings", {"games": p["games"], "layout": str(custom)}
    )
    assert status == HTTPStatus.OK
    assert not changed["has_fit"]
    assert (
        client.request(f"/api/projects/{key}/analyze", {})[0] == HTTPStatus.BAD_REQUEST
    )


def test_changed_record_provenance_blocks_manual_export_urls(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    p = analyzed(client, workspace)
    key = p["id"]
    (workspace.root / "work" / p["name"] / "record.json").write_text('[{"id":999}]')
    assert client.request(f"/api/projects/{key}")[1]["artifacts"] == []
    assert client.request(f"/exports/{key}/g0.json")[0] == HTTPStatus.NOT_FOUND


def test_replaced_source_hides_exports_and_retires_evidence_without_rehashing_polls(
    web: tuple[Client, Workspace, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, workspace, _ = web
    p = analyzed(client, workspace)
    key = p["id"]
    state = workspace.review_state(key)
    facts = state.labels / "facts.jsonl"
    facts.write_bytes(b'{"kind":"note","text":"preserve answer"}\n')
    saved_facts = facts.read_bytes()
    old_revision = state.revision()
    calls = []
    digest = cache.sha256_file

    def counted(path: Path) -> str:
        calls.append(path)
        return digest(path)

    monkeypatch.setattr(cache, "sha256_file", counted)
    for _ in range(3):
        assert client.request(f"/api/projects/{key}")[1]["artifacts"]
    assert calls == []
    source = Path(workspace.project(key)["video"])
    stat = source.stat()
    replacement = source.with_suffix(".new")
    replacement.write_bytes(b"different")  # same length and mtime, new identity
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(source)
    status, changed, _ = client.request(f"/api/projects/{key}")
    assert status == HTTPStatus.OK
    assert changed["artifacts"] == []
    assert changed["stale_exports"]
    assert not changed["has_fit"]
    assert client.request(f"/exports/{key}/g0.json")[0] == HTTPStatus.NOT_FOUND
    assert client.request(f"/review/{key}/api/hands")[1] == []
    assert state.revision() != old_revision
    assert facts.read_bytes() == saved_facts
    assert len(calls) == 1
    assert (
        client.request(f"/api/projects/{key}/analyze", {})[0] == HTTPStatus.BAD_REQUEST
    )


def test_workspace_list_never_waits_for_a_recording_digest(
    web: tuple[Client, Workspace, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow first digest shows "checking" and blocks neither polls nor writes."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    release = threading.Event()
    identity = cache.source_identity

    def slow(path: str | Path) -> str:
        assert release.wait(10)
        return identity(path)

    source = Path(workspace.project(p["id"])["video"])
    source.write_bytes(b"recording")  # A new timestamp needs a new digest.
    os.utime(source, ns=(1, 1))
    monkeypatch.setattr(cache, "source_identity", slow)
    try:
        started = time.monotonic()
        projects = client.request("/api/workspace")[1]["projects"]
        assert time.monotonic() - started < 2
        assert projects[0]["checking"]
        assert projects[0]["artifacts"] == []
        assert (
            client.request(f"/api/projects/{p['id']}/rename", {"display_name": "B"})[0]
            == 200
        )
        assert client.request("/api/job")[0] == HTTPStatus.OK
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while client.request(f"/api/projects/{p['id']}")[1]["checking"]:
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_missing_source_is_a_recoverable_project_state(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    p = analyzed(client, workspace)
    key = p["id"]
    source = Path(workspace.project(key)["video"])
    source.unlink()
    status, missing, _ = client.request(f"/api/projects/{key}")
    assert status == HTTPStatus.OK
    assert missing["artifacts"] == []
    assert not missing["has_fit"]
    status, error, _ = client.request(f"/api/projects/{key}/prepare", {})
    assert status == HTTPStatus.BAD_REQUEST
    assert "Restore the local video" in error["error"]
    source.write_bytes(b"restored recording")
    for action in ("prepare", "analyze"):
        assert (
            client.request(f"/api/projects/{key}/{action}", {})[0]
            == HTTPStatus.ACCEPTED
        )
        result = wait_for_job(client, key)
    assert result["artifacts"]
    assert not result["stale_exports"]


@pytest.mark.parametrize(
    "missing",
    [("source_sha256",), ("export_signature",), ("source_sha256", "export_signature")],
)
def test_missing_project_provenance_is_not_silently_upgraded(
    web: tuple[Client, Workspace, list[list[str]]],
    missing: tuple[str, str] | tuple[str],
) -> None:
    client, workspace, _ = web
    p = analyzed(client, workspace)
    key = p["id"]
    video_path = workspace.project(key)["video"]
    cal = Calibration.load("pml", video_path)
    done = workspace.root / "work" / p["name"] / "reads/00/done.json"
    done.parent.mkdir(parents=True)
    manifest = {
        "geometry": {r: region_key(cal, r) for r in REGIONS},
        "identity": {"source": cache.source_identity(video_path)},
    }
    done.write_text(json.dumps(manifest))
    facts = workspace.root / "labels" / p["name"] / "facts.jsonl"
    answers = (
        json.dumps({"game": 0, "kyoku": 0, "honba": 0, "kind": "draw", "tile": "1m"})
        + "\n"
    )
    facts.write_text(answers)
    stored = workspace.project(key)
    for field in missing:
        del stored[field]
    with pytest.raises(KeyError, match="|".join(missing)):
        workspace.snapshot(key)
    assert all(field not in stored for field in missing)
    assert facts.read_text() == answers
    assert done.read_text() == json.dumps(manifest)


def test_new_project_does_not_adopt_untracked_outputs(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    path = workspace.root / "recording.mp4"
    path.write_bytes(b"video")
    write_analysis(workspace.root, path.stem, [21938])
    labels = workspace.root / "labels" / path.stem
    (labels / "calib.json").parent.mkdir(parents=True, exist_ok=True)
    (labels / "calib.json").write_text('{"layout":"pml"}')
    answers = '{"kind":"note","text":"human answer"}\n'
    (labels / "facts.jsonl").write_text(answers)
    status, project, _ = client.request(
        "/api/projects", {"source": str(path), "games": [21938]}
    )
    assert status == HTTPStatus.CREATED
    assert project["artifacts"] == []
    assert not project["has_fit"]
    stored = workspace.project(project["id"])
    assert stored["export_signature"] is None
    assert stored["needs_prepare"]
    assert (workspace.root / "work" / path.stem / "inputs.changed").exists()
    assert client.request(f"/review/{project['id']}/api/hands")[1] == []
    assert (labels / "facts.jsonl").read_text() == answers


class FakeEncoder:
    """Stand in for ffmpeg: write each requested clip, or fail on request."""

    def __init__(self) -> None:
        """Record the files each encode was asked to write."""
        self.made: list[Path] = []
        self.fail = False

    @contextmanager
    def spawn(
        self, args: list[str], **_unused_kwargs: object
    ) -> Iterator[SimpleNamespace]:
        """Write the output named last unless failing."""
        self.made.append(Path(args[-1]))
        if not self.fail:
            Path(args[-1]).write_bytes(b"clip")
        yield SimpleNamespace(
            communicate=lambda: ("", "broken input"), returncode=int(self.fail)
        )


def test_clips_publish_complete_files_keyed_by_source_and_geometry(
    web: tuple[Client, Workspace, list[list[str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip is encoded to a temporary file; source or geometry changes miss."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    encoder = FakeEncoder()
    monkeypatch.setattr(state, "processes", encoder)
    first = state.clip(0, 2, "frame")
    assert first.read_bytes() == b"clip"
    assert encoder.made[0] != first
    assert state.clip(0, 2, "frame") == first
    assert len(encoder.made) == 1
    Path(workspace.project(p["id"])["video"]).write_bytes(b"new source bytes")
    second = state.clip(0, 2, "frame")
    assert second != first
    client.request(f"/review/{p['id']}/api/calib", {"overhead": {"center": [981, 543]}})
    third = state.clip(0, 2, "frame")
    assert third not in (first, second)
    encoder.fail = True
    with pytest.raises(OSError, match="Could not encode"):
        state.clip(5, 9, "frame")
    assert sorted(path.name for path in first.parent.iterdir()) == sorted(
        path.name for path in (first, second, third)
    )


def test_source_changed_during_conversion_cannot_authenticate_new_exports(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    original = workspace.runner
    source = Path(workspace.project(key)["video"])

    def replace_during_run(args: list[str], job: Job) -> object:
        original(args, job)
        source.write_bytes(b"changed while converting")
        return None

    workspace.runner = replace_during_run
    client.request(f"/api/projects/{key}/analyze", {})
    result = wait_for_job(client, key)
    assert result["status"] == "failed"
    assert result["job"]["error"] == (
        "The recording changed during analysis. Prepare it again, then retry."
    )
    assert result["artifacts"] == []
    assert workspace.project(key)["needs_prepare"]


@pytest.mark.parametrize("edit_running_hand", [False, True])
def test_http_answers_saved_during_an_update_are_applied_by_the_next_job(
    *,
    web: tuple[Client, Workspace, list[list[str]]],
    edit_running_hand: bool,
) -> None:
    """Answers saved while hands update stay pending until the next update."""
    client, workspace, commands = web
    project = create_local(client, workspace)
    write_analysis(workspace.root, project["name"], [21938], hands_per_game=2)
    prefix = f"/review/{project['id']}/api"
    entered, release = threading.Event(), threading.Event()

    def blocking(args: list[str], _job: Job) -> object:
        # Like the real command, read the answers when starting.
        facts = load_facts(workspace.root / "labels" / project["name"])
        commands.append(args)
        if len(commands) == 1:
            entered.set()
            assert release.wait(5)
        hands = [int(hand) for hand in option(args, "--hands")]
        publish(workspace.root, project["name"], hands, facts)
        return None

    workspace.runner = blocking
    first = {"hand": 0, "kind": "draw", "seat": "E", "j": 0, "tile": "2p"}
    assert client.request(prefix + "/facts", first)[0] == HTTPStatus.OK
    assert client.request(prefix + "/rebuild", {"hands": "pending"})[0] == 200
    target = 0 if edit_running_hand else 1
    second = {**first, "hand": target, "j": 1, "tile": "3p"}
    try:
        assert entered.wait(5)
        status, saved, _ = client.request(prefix + "/facts", second)
        assert status == HTTPStatus.OK
        # Removal and replacement also remain safe while the child runs.
        assert client.request(prefix + "/facts/delete", {"ts": saved["ts"]})[1] == {
            "deleted": 1
        }
        assert (
            client.request(prefix + "/facts", {**second, "tile": "4p"})[0]
            == HTTPStatus.OK
        )
        assert client.request(prefix + "/calib", {})[0] == HTTPStatus.CONFLICT
    finally:
        release.set()
    assert wait_for_workspace_job(client)["error"] is None
    pending = [h["hand"] for h in client.request(prefix + "/hands")[1] if h["pending"]]
    assert pending == [target]
    assert [fact["tile"] for fact in client.request(prefix + "/facts")[1]] == [
        "2p",
        "4p",
    ]
    assert client.request(prefix + "/rebuild", {"hands": "pending"})[0] == 200
    wait_for_workspace_job(client)
    assert not any(h["pending"] for h in client.request(prefix + "/hands")[1])
    assert [option(c, "--hands") for c in commands] == [["0"], [str(target)]]


def test_a_running_update_blocks_calibration_and_other_jobs(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Every write except answers waits for the running update."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    with running(workspace, "rebuild", p["id"]):
        # Repeat actual requests without retrying failures: closing an unread
        # JSON body used to intermittently replace the 409 with WinError 10053.
        for _ in range(25):
            for endpoint in ("calib", "label", "rebuild", "calib/check", "calib/fit"):
                status, error, _headers = client.request(
                    f"/review/{p['id']}/api/{endpoint}", {}
                )
                assert status == HTTPStatus.CONFLICT
                assert (
                    error["error"] == "Hands are being updated. Wait for it to finish."
                )
        assert (
            client.request(f"/api/projects/{p['id']}/settings", {"games": [42]})[0]
            == HTTPStatus.BAD_REQUEST
        )


def test_job_status_reports_the_review_revision_of_the_open_project(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """The one status poll notices results rebuilt by another process."""
    client, workspace, _ = web
    p = analyzed(client, workspace)
    first = client.request(f"/api/job?project={p['id']}")[1]
    assert first["job"]["kind"] == "analyze"
    assert not first["job"]["running"]
    assert client.request("/api/job")[1]["revision"] is None
    publish(workspace.root, p["name"], [1])
    second = client.request(f"/api/job?project={p['id']}")[1]
    assert first["revision"] != second["revision"]


def test_busy_rejection_drain_is_bounded_for_incomplete_or_large_bodies(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    p = create_local(client, workspace)
    address = urlparse(client.base)
    with running(workspace, "rebuild", p["id"]):
        for size in (32, 2 * 1024 * 1024):
            with socket.create_connection(
                (address.hostname, address.port), timeout=2
            ) as connection:
                request = (
                    f"POST /review/{p['id']}/api/calib HTTP/1.1\r\n"
                    f"Host: {address.netloc}\r\nX-Video2Tenhou: 1\r\n"
                    f"Content-Type: application/json\r\nContent-Length: {size}\r\n\r\n"
                )
                started = time.monotonic()
                # Deliberately do not deliver the advertised body.
                connection.sendall(request.encode())
                response = connection.recv(4096)
                assert response.startswith(b"HTTP/1.1 409")
                assert time.monotonic() - started < 1.5


def test_project_request_limit_survives_rejection_drain(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, _, _ = web
    status, error, _ = client.request("/api/projects", b" " * 65537)
    assert status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert error["error"] == "Request body is too large."


def test_hand_view_shows_notes_and_ignored_answers_but_no_diagnostics(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    """Reviewers see notes and unapplied answers; developer reasoning stays out."""
    client, workspace, _ = web
    p = analyzed(client, workspace)
    prefix = f"/review/{p['id']}/api"
    payload = '<img src=x onerror="alert(1)">'
    saved = client.request(
        prefix + "/facts",
        {"hand": 0, "kind": "draw", "seat": "N", "t": 30.25, "tile": "2p"},
    )[1]
    state = workspace.review_state(p["id"])
    data = json.loads(state.decode_path(0).read_text(encoding="utf-8"))
    data.update(
        notes=[payload],
        diagnostics=["solver: 3 restarts"],
        ignored_facts=[
            {"kind": "draw", "seat": "N", "t": 30.25, "reason": "no such turn"},
            {"kind": "meld", "seat": None, "t": None, "reason": "no call near it"},
        ],
    )
    state.decode_path(0).write_text(json.dumps(data), encoding="utf-8")
    status, result, headers = client.request(prefix + "/hand/0")
    assert status == HTTPStatus.OK
    assert "application/json" in headers["Content-Type"]
    assert result["decode"]["notes"] == [payload]
    assert not {"diagnostics", "decode_context", "problems"} & set(result["decode"])
    assert result["ignored"] == [
        {"ts": saved["ts"], "kind": "draw", "reason": "no such turn"},
        {"ts": None, "kind": "meld", "reason": "no call near it"},
    ]
    assert client.request(prefix + "/hand/9")[0] == HTTPStatus.NOT_FOUND


def test_closing_workspace_interrupts_real_job_and_prevents_next_phase(
    web: tuple[Client, Workspace, list[list[str]]],
) -> None:
    client, workspace, _ = web
    p = create_local(client, workspace)
    workspace.runner = lambda _args, job: workspace.run_child(
        child("import time; print('[running] test job', flush=True); time.sleep(60)"),
        job,
    )
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    deadline = time.monotonic() + 5
    while not workspace.log(p["id"]) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert workspace.log(p["id"]) == ["[running] test job"]
    workspace.close()
    stopped = workspace.snapshot(p["id"])
    assert stopped["status"] == "interrupted"
    assert not stopped["job"]["running"]
    assert stopped["job"]["error"].startswith("Interrupted when the app closed")
    assert Workspace(workspace.root).snapshot(p["id"])["status"] == "interrupted"
    with pytest.raises(ValueError, match="closing"):
        workspace.start(p["id"], "prepare")
