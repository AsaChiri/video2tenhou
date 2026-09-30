# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""HTTP integration for import → prepare → analysis → review → downloadable output.

Only the expensive subprocess boundary is replaced. Real TCP requests exercise
routing, body streaming, validation, persistent manifests and project isolation.
"""

import json
import os
import socket
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import cv2
import httpx
import numpy as np
import pytest

from tests.web.server import LiveStudio, studio_server
from video2tenhou import cache, cli, layout, paths, record, tenhou6, video
from video2tenhou.calm import REGIONS, region_key
from video2tenhou.engine import decode
from video2tenhou.files import atomic_write_json
from video2tenhou.layout import CALIB_DIR, Calibration, fit_path
from video2tenhou.tool import rebuild as rebuild_command
from video2tenhou.tool import review_state as review
from video2tenhou.tool.workflow import Workspace

if TYPE_CHECKING:
    from video2tenhou.record import Game


if TYPE_CHECKING:
    from collections.abc import Iterator


class Client:
    """HTTP client for the isolated local studio test server."""

    def __init__(self, server: LiveStudio) -> None:
        """Bind requests to the test server's loopback port."""
        self.base = server.url

    def request(
        self,
        path: "str",
        body: "dict | bytes | None" = None,
        *,
        headers: "dict[str, str] | None" = None,
        raw: bool = False,
    ) -> "tuple":
        """Send a studio request and return status, decoded data and headers."""
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
        )
        return (
            response.status_code,
            response.content if raw else response.json(),
            response.headers,
        )


def test_review_items_project_certified_confidence_without_changing_evidence(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Review items gain certified coverage/margins from their confidence rows."""
    client, workspace, _ = web
    project = create_local(client, workspace)
    state = workspace.review_state(project["id"])
    state.hands = [
        {"hand": 0, "game": 0, "kyoku": 0, "honba": 0, "corner_wind": {"BR": "N"}}
    ]
    items = [
        {"kind": "draw", "seat": "N", "j": 0, "tile": "1m"},
        {"kind": "discard", "seat": "N", "j": 0, "tile": "2p"},
        {"kind": "result", "seat": "N", "tiles": ["1m"] * 13},
        {"kind": "draw", "seat": "N", "j": 1, "margin": 0.25},
        {
            "kind": "discard",
            "seat": "N",
        },  # No turn identity: do not borrow another row.
        {"kind": "call", "seat": "N", "alternative_gap": 0.01},
    ]
    confidence = [
        {"field": "draw", "seat": "E", "turn": 0, "margin": 0.9, "lost": False},
        {
            "field": "draw",
            "seat": "N",
            "turn": 0,
            "margin": 0,
            "alternative_gap": 80,
            "lost": True,
        },
        {
            "field": "discard",
            "seat": "N",
            "turn": 0,
            "margin": 0.1,
            "conf": 0.99,
            "lost": False,
        },
        {"field": "haipai", "seat": "N", "turn": -1, "margin": None, "lost": False},
        {"field": "draw", "seat": "N", "turn": 1, "margin": 0.4, "lost": False},
        {"field": "discard", "seat": "N", "turn": None, "margin": 0.2, "lost": True},
    ]
    path = state.decode_path(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"items": items, "confidence": confidence}).encode()
    path.write_bytes(original)
    status, rows, _ = client.request(f"/review/{project['id']}/api/items")
    assert status == HTTPStatus.OK
    assert path.read_bytes() == original
    assert [row["idx"] for row in rows] == list(range(6))
    assert all(row["hand"] == 0 for row in rows)
    assert [(row.get("margin"), row.get("lost")) for row in rows[:4]] == [
        (0, True),
        (0.1, False),
        (None, None),
        (0.25, False),
    ]
    assert rows[2]["tiles"] == ["1m"] * 13
    assert "margin" not in rows[4]
    assert "lost" not in rows[4]
    assert "margin" not in rows[5]


def test_prepare_storage_failure_is_retryable_through_http(
    web: "tuple[Client, Workspace, list[list[str]]]", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify prepare storage failure is retryable through http."""
    client, workspace, commands = web
    project = create_local(client, workspace)
    key = project["id"]
    manifest = workspace.projects_dir / f"{key}.json"
    before = manifest.read_bytes()
    original = Path.replace

    def denied(source: "Path", target: "Path") -> "Path":
        if target == manifest:
            msg = "manifest temporarily unavailable"
            raise PermissionError(msg)
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


def test_pending_rebuild_selects_server_changes_and_preserves_other_hands(
    web: "tuple[Client, Workspace, list[list[str]]]", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Real HTTP/job/child-program flow; only expensive reconstruction is doubled."""
    client, workspace, _ = web
    state, prefix = _seed_review_hands(client, workspace)
    untouched = (
        state.decode_path(1).read_bytes(),
        state.decode_path(1).stat().st_mtime_ns,
    )
    _change_review_facts(client, prefix)
    facts_before = (state.labels / "facts.jsonl").read_bytes()
    assert client.request(prefix + "/decode_pending")[1]["pending"] == [0, 2]
    assert [
        h["hand"] for h in client.request(prefix + "/hands")[1] if h["pending_rebuild"]
    ] == [0, 2]
    entered, release = threading.Event(), threading.Event()
    children = []
    fail = False

    decoded, written = _review_rebuild_doubles(state, monkeypatch)

    def child(args: "list[str]", **_unused_kwargs: object) -> "SimpleNamespace":
        children.append(args)
        entered.set()
        assert release.wait(5)
        if fail:
            return SimpleNamespace(
                returncode=1, stderr="recognition cache changed; Analyze recording"
            )
        # Execute the child entry point with the same model-free boundary.

        assert args[1:3] == ["-m", "video2tenhou.tool.rebuild"]
        rebuild_command.main(args[3:])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(state.processes, "run", child)
    try:
        status, job, _ = client.request(prefix + "/decode_pending", {"hands": [1]})
        assert status == HTTPStatus.OK
        assert job["hands"] == [0, 2]
        assert entered.wait(5)
        assert client.request(prefix + "/calib", {})[0] == HTTPStatus.CONFLICT
        assert client.request(prefix + "/decode_pending")[1]["running"]
    finally:
        release.set()
    job = _wait_for_rebuild(client, prefix)
    assert not job["running"]
    assert job["error"] is None
    assert job["pending"] == []
    assert decoded == [{0, 2}]
    assert written == [[0, 1, 2]]
    assert job["hands_done"] == job["hands_total"] == 2
    assert (
        state.decode_path(1).read_bytes(),
        state.decode_path(1).stat().st_mtime_ns,
    ) == untouched
    assert (state.labels / "facts.jsonl").read_bytes() == facts_before
    _assert_rebuild_receipts(state)
    assert client.request(prefix + "/decode_pending", {})[1]["pending"] == []
    assert (
        len(children) == 1
    )  # No pending changes means no child, even after a prior job.
    # Failure remains retryable, without acknowledging the newly saved answer.
    assert (
        client.request(
            prefix + "/facts",
            {"hand": 2, "kind": "draw", "seat": "N", "tile": "4z", "j": 0},
        )[0]
        == HTTPStatus.OK
    )
    fail = True
    assert client.request(prefix + "/decode_pending", {})[0] == HTTPStatus.OK
    job = _wait_for_rebuild(client, prefix)
    assert job["pending"] == [2]
    assert "Analyze recording" in job["error"]
    assert job["hands"] == [2]
    assert len(children) == 2


@pytest.fixture
def web(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> "Iterator[tuple[Client, Workspace, list[list[str]]]]":
    """Run an isolated studio server with controlled pipeline execution."""
    monkeypatch.setattr(review, "ROOT", tmp_path)
    monkeypatch.setattr(layout, "LABEL_DIR", tmp_path / "labels")
    commands = []

    def runner(args: "list[str]", project: "dict") -> None:
        commands.append(args)
        name = Path(project["video"]).stem
        if "download" in args or "trim" in args:
            Path(project["video"]).parent.mkdir(parents=True, exist_ok=True)
            Path(project["video"]).write_bytes(b"downloaded video")
        elif "calib" in args:
            path = tmp_path / "labels" / name / "calib.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, {"video": name, "layout": "pml", "overhead": {}})
        elif "convert" in args:
            work = tmp_path / "work" / name
            work.mkdir(parents=True, exist_ok=True)
            (work / "hands.json").write_text(
                json.dumps(
                    [
                        {
                            "hand": 0,
                            "game": 0,
                            "kyoku": 0,
                            "honba": 0,
                            "corner_wind": {"TL": "E", "TR": "S", "BL": "W", "BR": "N"},
                        }
                    ]
                )
            )
            (work / "record.json").write_text(
                json.dumps(
                    [
                        {"id": game, "players": {}, "final": {}, "hands": []}
                        for game in project["games"]
                    ]
                )
            )
            output = tmp_path / "out" / name
            output.mkdir(parents=True, exist_ok=True)
            (output / "g0.json").write_text('{"log": []}')
            (output / "g0.html").write_text("<h1>Replay links</h1>")
            (output / "review.json").write_text("[]")

    with studio_server(tmp_path, runner=runner) as server:
        yield Client(server), server.workspace, commands


def wait_for_job(client: "Client", key: "str") -> "dict":
    """Poll a project until its job finishes or the test deadline expires."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status, result, _ = client.request(f"/api/projects/{key}")
        assert status == HTTPStatus.OK
        if not result["job"]["running"]:
            return result
        time.sleep(0.01)
    pytest.fail("The background job did not finish.")


def create_local(
    client: "Client", workspace: "Workspace", name: str = "recording.mp4"
) -> "dict":
    """Create a local recording and import it into the studio workspace."""
    video = workspace.root / "samples" / name
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"recording")
    status, project, _ = client.request(
        "/api/projects", {"source": str(video), "games": [21938, 21939]}
    )
    assert status == HTTPStatus.CREATED
    return project


def test_project_library_rename_delete_and_restart(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify project library rename delete and restart."""
    client, workspace, _ = web
    project = create_local(client, workspace)
    key = project["id"]
    paths = [Path(project["video"])]
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
    assert renamed["video"] == project["video"]
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
    web: "tuple[Client, Workspace, list[list[str]]]", name: "str"
) -> None:
    """Verify project rename validates name."""
    client, workspace, _ = web
    project = create_local(client, workspace)
    status, _, _ = client.request(
        f"/api/projects/{project['id']}/rename", {"display_name": name}
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert workspace.snapshot(project["id"])["display_name"] == project["name"]


def test_project_delete_rejects_processing_and_review_jobs(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify project delete rejects processing and review jobs."""
    client, workspace, _ = web
    key = create_local(client, workspace)["id"]
    workspace.project(key)["job"]["running"] = True
    assert (
        client.request(f"/api/projects/{key}/delete", {})[0] == HTTPStatus.BAD_REQUEST
    )
    workspace.project(key)["job"]["running"] = False
    state = workspace.review_state(key)
    state.jobs["decode"] = {"running": True}
    try:
        assert (
            client.request(f"/api/projects/{key}/delete", {})[0]
            == HTTPStatus.BAD_REQUEST
        )
        assert workspace.snapshot(key)["review_running"] is True
    finally:
        state.jobs.clear()
    assert (workspace.projects_dir / f"{key}.json").exists()


def test_project_mutation_storage_failure_retains_manifest_and_memory(
    web: "tuple[Client, Workspace, list[list[str]]]", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify project mutation storage failure retains manifest and memory."""
    client, workspace, _ = web
    key = create_local(client, workspace)["id"]
    manifest = workspace.projects_dir / f"{key}.json"
    before = manifest.read_bytes()
    original_name = workspace.snapshot(key)["display_name"]

    def denied(*_unused_args: object, **_unused_kwargs: object) -> None:
        msg = "Storage unavailable"
        raise PermissionError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(workspace, "_save", denied)
        assert (
            client.request(f"/api/projects/{key}/rename", {"display_name": "New"})[0]
            == HTTPStatus.BAD_REQUEST
        )
    assert workspace.snapshot(key)["display_name"] == original_name
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        assert (
            client.request(f"/api/projects/{key}/delete", {})[0]
            == HTTPStatus.BAD_REQUEST
        )
    assert manifest.read_bytes() == before
    assert key in workspace.projects


def test_import_prepare_analyze_review_and_export(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify import prepare analyze review and export."""
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
    assert client.request(f"/review/{key}/api/hands")[1][0]["hand"] == 0
    status, log, headers = client.request(f"/exports/{key}/g0.json")
    assert status == HTTPStatus.OK
    assert log == {"log": []}
    assert headers["Content-Disposition"] == 'attachment; filename="g0.json"'
    assert (
        client.request(f"/exports/{key}/g0.html", raw=True)[1]
        == b"<h1>Replay links</h1>"
    )
    assert "--skip-fit-check" not in commands[-1]
    assert commands[-1][-4:] == ["--game", "21938", "--game", "21939"]
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
    web: "tuple[Client, Workspace, list[list[str]]]", source: str
) -> None:
    """Verify url preparation delegates to downloader."""
    client, workspace, commands = web
    status, p, _ = client.request(
        "/api/projects", {"source": source, "kind": "url", "games": [1]}
    )
    assert status == HTTPStatus.CREATED
    status, response, _ = client.request(f"/api/projects/{p['id']}/prepare", {})
    assert status == HTTPStatus.ACCEPTED, response
    assert wait_for_job(client, p["id"])["status"] == "ready"
    assert commands[0][-4:] == ["download", "--", source, p["video"]]
    assert Path(p["video"]).parent == workspace.root / "samples"
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    wait_for_job(client, p["id"])
    assert sum("download" in command for command in commands) == 1


def test_url_recording_identity_includes_query(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify url recording identity includes query."""
    client, _, _ = web

    def create(source: "str") -> "tuple":
        return client.request(
            "/api/projects", {"source": source, "kind": "url", "games": [1]}
        )

    first = "https://www.youtube.com/watch?v=first"
    status, p, _ = create(first)
    assert status == HTTPStatus.CREATED
    status, other, _ = create("https://www.youtube.com/watch?v=second")
    assert status == HTTPStatus.CREATED
    assert p["video"] != other["video"]
    assert create(first)[0] == 400
    assert create(" ")[0] == 400


@pytest.mark.parametrize("kind", ["local", "url"])
@pytest.mark.parametrize(
    ("start", "end"), [("01:30", "02:00"), ("01:30", ""), ("", "02:00")]
)
def test_selected_range_is_prepared_once_and_persisted(
    web: "tuple[Client, Workspace, list[list[str]]]", kind: str, start: str, end: str
) -> None:
    """Verify selected range is prepared once and persisted."""
    client, workspace, commands = web
    original = workspace.root / "original.mp4"
    original.write_bytes(b"original remains intact")
    source = str(original) if kind == "local" else "https://example.test/video?id=7"
    status, p, _ = client.request(
        "/api/projects",
        {
            "source": source,
            "kind": kind,
            "games": [1],
            "start": start,
            "end": end,
        },
    )
    assert status == HTTPStatus.CREATED
    assert (p["start"], p["end"]) == (90 if start else 0, 120 if end else None)
    assert p["video"] != source
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
    assert cmd[-3:] == ["--", source, p["video"]]
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
    web: "tuple[Client, Workspace, list[list[str]]]", kind: str
) -> None:
    """Verify ranges have distinct projects but equivalent time formats do not."""
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
    assert len({p["video"] for p in (full, first, second)}) == 3
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
    web: "tuple[Client, Workspace, list[list[str]]]", bounds: "dict"
) -> None:
    """Verify invalid ranges do not create projects."""
    client, workspace, commands = web
    status, data, _ = client.request(
        "/api/projects",
        {
            "source": "https://example.test/video",
            "kind": "url",
            "games": [1],
            **bounds,
        },
    )
    assert status == HTTPStatus.BAD_REQUEST
    assert data["error"]
    assert not workspace.projects
    assert not commands


def test_failed_range_preparation_can_retry_without_changing_original(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify failed range preparation can retry without changing original."""
    client, workspace, commands = web
    original = workspace.root / "original.mp4"
    original.write_bytes(b"original")
    status, p, _ = client.request(
        "/api/projects",
        {
            "source": str(original),
            "games": [1],
            "start": "10",
            "end": "20",
        },
    )
    assert status == HTTPStatus.CREATED
    runner = workspace.runner

    def interrupted(args: "list[str]", project: "dict") -> None:
        msg = "Clip creation was interrupted."
        raise RuntimeError(msg)

    workspace.runner = interrupted
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    failed = wait_for_job(client, p["id"])
    assert failed["status"] == "failed"
    assert not failed["has_fit"]
    assert not Path(p["video"]).exists()
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
    web: "tuple[Client, Workspace, list[list[str]]]",
    monkeypatch: "pytest.MonkeyPatch",
    kind: str,
) -> None:
    """Verify fractional bounds round trip through preparation cli."""
    client, workspace, _ = web
    original = workspace.root / "original.mp4"
    original.write_bytes(b"original")
    source = str(original) if kind == "local" else "https://example.test/video"
    parsed = []

    def record_bounds(
        source: "Path", out: "Path", start: "float", end: "float"
    ) -> None:
        parsed.append(video.time_range(start, end))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"clip")

    operation = "trim" if kind == "local" else "download"
    monkeypatch.setattr(video, operation, record_bounds)
    runner = workspace.runner

    def run(args: "list[str]", project: "dict") -> None:
        if operation in args:
            cli.main(args[4:])
        else:
            runner(args, project)

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
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Serve whole-game and single-hand links without the CLI HTML artifact.

    Rebuilding a JSON export changes its revision, and changing project inputs
    hides both the file download and native results until analysis completes.
    """
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    for action in ("prepare", "analyze"):
        assert (
            client.request(f"/api/projects/{key}/{action}", {})[0]
            == HTTPStatus.ACCEPTED
        )
        wait_for_job(client, key)
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
    status, result, _ = client.request(f"/api/projects/{key}/results")
    assert status == HTTPStatus.OK
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
    fresh = client.request(f"/api/projects/{key}/results")[1]
    assert fresh["revision"] != result["revision"]
    assert len(fresh["games"][0]["hands"]) == 3
    status, _, _ = client.request(
        f"/review/{key}/api/facts",
        {"hand": 0, "kind": "draw", "seat": "N", "j": 0, "t": 10, "tile": "2p"},
    )
    assert status == HTTPStatus.OK
    pending = client.request(f"/api/projects/{key}/results")[1]
    assert pending["pending_rebuilds"] == [0]
    assert pending["pending_games"] == [0]
    assert client.request(f"/api/projects/{key}")[1]["pending_rebuilds"] == [0]
    other = create_local(client, workspace, "other.mp4")
    assert client.request(f"/api/projects/{other['id']}/results")[1]["games"] == []
    assert (
        client.request(f"/api/projects/{key}/settings", {"games": [42]})[0]
        == HTTPStatus.OK
    )
    assert client.request(f"/api/projects/{key}/results")[1]["games"] == []
    assert client.request(row["download"])[0] == HTTPStatus.NOT_FOUND


def test_review_facts_are_project_scoped(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify review facts are project scoped."""
    client, workspace, _ = web
    projects = [create_local(client, workspace, name) for name in ("a.mp4", "b.mp4")]
    for project in projects:
        state = workspace.review_state(project["id"])
        state.hands = [
            {"hand": 0, "game": 0, "kyoku": 3, "honba": 1, "corner_wind": {"BR": "N"}}
        ]
    one, two = [p["id"] for p in projects]
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


def test_cross_origin_traversal_and_upload_validation(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify cross origin traversal and upload validation."""
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
        client.request("/api/projects", {"source": p["video"], "games": [1]})[0]
        == HTTPStatus.BAD_REQUEST
    )


def test_failed_and_interrupted_jobs_remain_actionable(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify failed and interrupted jobs remain actionable."""
    client, workspace, _ = web
    p = create_local(client, workspace)

    def fail(_args: object, _project: object) -> None:
        msg = "Detector weights missing"
        raise RuntimeError(msg)

    workspace.runner = fail
    client.request(f"/api/projects/{p['id']}/prepare", {})
    failed = wait_for_job(client, p["id"])
    assert failed["status"] == "failed"
    assert "Detector weights" in failed["job"]["error"]
    assert (
        Workspace(workspace.root).snapshot(p["id"])["job"]["error"]
        == failed["job"]["error"]
    )
    project = workspace.project(p["id"])
    project["job"]["running"] = True
    workspace._save(project)
    restarted = Workspace(workspace.root).snapshot(p["id"])
    assert restarted["status"] == "interrupted"
    assert not restarted["job"]["running"]


def test_failed_calibration_keeps_original_exit_status_and_cli_output(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify failed calibration keeps original exit status and cli output."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    script = (
        "print('FAIL meld:TL 0 tiles held, 3 cut by the border'); print('Check "
        "calibration in Settings'); raise SystemExit(1)"
    )
    workspace.runner = lambda _args, project: workspace._run_command(
        [sys.executable, "-u", "-c", script], project
    )
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    failed = wait_for_job(client, p["id"])
    assert failed["status"] == "failed"
    assert "exit status 1" in failed["job"]["error"]
    assert failed["job"]["log"] == [
        "FAIL meld:TL 0 tiles held, 3 cut by the border",
        "Check calibration in Settings",
    ]


def test_jobs_are_serialized_and_review_writes_wait(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify jobs are serialized and review writes wait."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    started, release = threading.Event(), threading.Event()

    def slow(_args: object, _project: object) -> None:
        started.set()
        assert release.wait(5)

    workspace.runner = slow
    try:
        assert (
            client.request(f"/api/projects/{p['id']}/prepare", {})[0]
            == HTTPStatus.ACCEPTED
        )
        assert started.wait(2)
        assert (
            client.request(f"/api/projects/{p['id']}/prepare", {})[0]
            == HTTPStatus.BAD_REQUEST
        )
        assert (
            client.request(f"/review/{p['id']}/api/facts", {})[0] == HTTPStatus.CONFLICT
        )
    finally:
        release.set()
        wait_for_job(client, p["id"])


def test_studio_and_vendored_tiles_are_served_offline(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify studio and vendored tiles are served offline."""
    client, workspace, _ = web
    status, html, _ = client.request("/", raw=True)
    assert status == HTTPStatus.OK
    assert b'id="app"' in html
    status, svg, headers = client.request("/tiles/Pin2.svg", raw=True)
    assert status == HTTPStatus.OK
    assert b"<svg" in svg
    assert "image/svg+xml" in headers["Content-Type"]
    assert client.request("/tiles/LICENSE.md", raw=True)[0] == HTTPStatus.OK
    p = create_local(client, workspace)
    assert (
        client.request(f"/review/{p['id']}/?embedded=1", raw=True)[0]
        == HTTPStatus.NOT_FOUND
    )


def test_calibration_save_reloads_geometry_and_evidence_cache(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify calibration save reloads geometry and evidence cache."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    state._frame_cache[10.0] = np.zeros((2, 2, 3), np.uint8)
    status, saved, _ = client.request(
        f"/review/{p['id']}/api/calib",
        {"overhead": {"center": [981, 543], "angle": 46}},
    )
    assert status == HTTPStatus.OK
    assert saved["fit"]["overhead"]["source"] == "human"
    assert not state._frame_cache
    status, geometry, _ = client.request(f"/review/{p['id']}/api/calib")
    assert status == HTTPStatus.OK
    assert geometry["fit"]["overhead"]["center"] == [981, 543]
    assert state.cal.center in ((981, 543), [981, 543])
    assert (workspace.root / "labels" / p["name"] / "calib.json").is_file()


@pytest.mark.parametrize("partial_fit", [False, True])
def test_failed_prepare_can_save_table_calibration_then_retry(
    *, web: "tuple[Client, Workspace, list[list[str]]]", partial_fit: bool
) -> None:
    """Verify failed prepare can save table calibration then retry."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    attempts = []

    def runner(args: "list[str]", project: "dict") -> None:
        attempts.append(args)
        path = fit_path(project["video"])
        if len(attempts) == 1:
            if partial_fit:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"overhead": {}}')
            msg = "No table tiles found."
            raise RuntimeError(msg)
        cal = Calibration.load(project["layout"], project["video"])
        assert cal.center == (981, 543)
        assert cal.fit is not None
        assert cal.fit["overhead"]["source"] == "human"
        assert cal.hand["TL"][0].x == 15

    workspace.runner = runner
    assert client.request(f"/api/projects/{key}/prepare", {})[0] == HTTPStatus.ACCEPTED
    failed = wait_for_job(client, key)
    assert failed["can_calibrate"]
    assert not failed["has_fit"]
    status, geometry, _ = client.request(f"/review/{key}/api/calib")
    assert status == HTTPStatus.OK
    assert geometry["overhead"]["center"] == [960, 540]
    assert not any(name.startswith("overlay") for name in geometry["regions"])
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


def test_scoped_frame_and_clip_routes_return_this_projects_evidence(
    web: "tuple[Client, Workspace, list[list[str]]]", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify scoped frame and clip routes return this projects evidence."""
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


def test_review_rebuild_subprocess_refreshes_existing_exports(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Execute the actual review child process and writer with an empty site game.

    No vision is needed: this specifically catches stale exports after review,
    incorrect process arguments, wrong output roots and cached game names.
    """
    client, workspace, _ = web
    p = create_local(client, workspace)
    work = workspace.root / "work" / p["name"]
    work.mkdir(parents=True)
    game = record.Game(21938, {"EAST": "Reviewed player"}, {}, [])
    (work / "hands.json").write_text("[]")
    (work / "record.json").write_text(
        json.dumps(
            [record.to_dict(game), record.to_dict(record.Game(21939, {}, {}, []))]
        )
    )
    workspace.project(p["id"])["export_signature"] = workspace._signature(p)
    out = workspace.root / "out" / p["name"]
    out.mkdir(parents=True)
    (out / "g0.json").write_text('{"name": ["stale"]}')
    (out / "review.json").write_text('[{"kind":"stale"}]')
    assert client.request(f"/review/{p['id']}/api/decode_all", {})[0] == HTTPStatus.OK
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status, job, _ = client.request(f"/review/{p['id']}/api/decode_all")
        assert status == HTTPStatus.OK
        if not job["running"]:
            break
        time.sleep(0.05)
    assert not job["running"], job
    assert not job["error"], job
    status, output, _ = client.request(f"/exports/{p['id']}/g0.json")
    assert status == HTTPStatus.OK
    assert output["name"][0] == "Reviewed player"
    assert client.request(f"/exports/{p['id']}/review.json")[1] == []
    assert (out / "g0.html").exists()
    assert (out / "report.md").exists()


def test_cli_uses_selected_data_directory_for_stage_outputs(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify cli uses selected data directory for stage outputs."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    captured = []
    monkeypatch.setattr(cli, "cmd_convert", captured.append)
    cli.main(["convert", "recording.mp4", "--game", "21938"])
    assert captured[0].work == str(tmp_path / "work")
    assert captured[0].out == str(tmp_path / "out")


def test_settings_corrections_hide_old_exports_and_preserve_answers(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify settings corrections hide old exports and preserve answers."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    client.request(f"/api/projects/{key}/analyze", {})
    assert wait_for_job(client, key)["artifacts"]
    facts = workspace.root / "labels" / p["name"] / "facts.jsonl"
    facts.write_text('{"kind":"note","text":"keep me"}\n')
    status, changed, _ = client.request(
        f"/api/projects/{key}/settings", {"games": [22002, 22003], "layout": "pml"}
    )
    assert status == HTTPStatus.OK
    assert changed["has_fit"]
    assert changed["stale_exports"]
    assert changed["artifacts"] == []
    assert not changed["has_hands"]
    assert client.request(f"/exports/{key}/g0.json")[0] == HTTPStatus.NOT_FOUND
    assert client.request(f"/review/{key}/api/hands")[1] == []
    assert "keep me" in facts.read_text()
    client.request(f"/api/projects/{key}/analyze", {})
    refreshed = wait_for_job(client, key)
    assert refreshed["artifacts"]
    assert not refreshed["stale_exports"]
    assert "keep me" in facts.read_text()


def test_layout_change_requires_preparation(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify layout change requires preparation."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    work = workspace.root / "work" / p["name"]
    work.mkdir(parents=True, exist_ok=True)
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
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify changed record provenance blocks manual export urls."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    client.request(f"/api/projects/{key}/analyze", {})
    assert wait_for_job(client, key)["artifacts"]
    record = workspace.root / "work" / p["name"] / "record.json"
    record.write_text('[{"id":999}]')
    assert client.request(f"/api/projects/{key}")[1]["artifacts"] == []
    assert client.request(f"/exports/{key}/g0.json")[0] == HTTPStatus.NOT_FOUND


def test_replaced_source_hides_exports_and_retires_evidence_without_rehashing_polls(
    web: "tuple[Client, Workspace, list[list[str]]]", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify source replacement retires exports and cached evidence."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    for action in ("prepare", "analyze"):
        client.request(f"/api/projects/{key}/{action}", {})
        assert not wait_for_job(client, key)["job"].get("error")
    old_state = workspace.review_state(key)
    old_revision = old_state.revision()
    facts = old_state.labels / "facts.jsonl"
    facts.write_bytes(b'{"kind":"note","text":"preserve answer"}\n')
    saved_facts = facts.read_bytes()
    old_revision = old_state.revision()
    calls = []
    digest = cache.sha256_file

    def counted(path: "Path") -> "str":
        calls.append(path)
        return digest(path)

    monkeypatch.setattr(cache, "sha256_file", counted)
    for _ in range(3):
        assert client.request(f"/api/projects/{key}")[1]["artifacts"]
    assert calls == []
    source = Path(p["video"])
    stat = source.stat()
    replacement = source.with_suffix(".new")
    replacement.write_bytes(
        b"different"
    )  # same length, same mtime, different file identity
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(source)
    status, changed, _ = client.request(f"/api/projects/{key}")
    assert status == HTTPStatus.OK
    assert changed["artifacts"] == []
    assert changed["stale_exports"]
    assert not changed["has_hands"]
    assert not changed["has_fit"]
    assert client.request(f"/exports/{key}/g0.json")[0] == HTTPStatus.NOT_FOUND
    assert client.request(f"/review/{key}/api/hands")[1] == []
    fresh_state = workspace.review_state(key)
    assert fresh_state is not old_state
    assert old_state.processes.closing
    assert fresh_state.revision() != old_revision
    assert facts.read_bytes() == saved_facts
    assert len(calls) == 1
    assert (
        client.request(f"/api/projects/{key}/analyze", {})[0] == HTTPStatus.BAD_REQUEST
    )


def test_missing_source_is_a_recoverable_project_state(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify missing source is a recoverable project state."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    for action in ("prepare", "analyze"):
        client.request(f"/api/projects/{key}/{action}", {})
        wait_for_job(client, key)
    source = Path(p["video"])
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
    web: "tuple[Client, Workspace, list[list[str]]]",
    missing: tuple[str, str] | tuple[str],
) -> None:
    """Verify missing project provenance is not silently upgraded."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    for action in ("prepare", "analyze"):
        client.request(f"/api/projects/{key}/{action}", {})
        wait_for_job(client, key)
    cal = Calibration.load("pml", p["video"])
    done = workspace.root / "work" / p["name"] / "reads/00/done.json"
    done.parent.mkdir(parents=True)
    manifest = {
        "geometry": {r: region_key(cal, r) for r in REGIONS},
        "identity": {"source": workspace._source(p)},
    }
    done.write_text(json.dumps(manifest))
    facts = workspace.root / "labels" / p["name"] / "facts.jsonl"
    answers = (
        json.dumps(
            {
                "hand": 0,
                "game": 0,
                "kyoku": 0,
                "honba": 0,
                "kind": "note",
                "text": "human answer",
            }
        )
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
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify new project does not adopt untracked outputs."""
    client, workspace, _ = web
    video = workspace.root / "recording.mp4"
    video.write_bytes(b"video")
    work = workspace.root / "work" / video.stem
    work.mkdir(parents=True)
    (work / "hands.json").write_text("[]")
    (work / "record.json").write_text('[{"id":21938}]')
    labels = workspace.root / "labels" / video.stem
    labels.mkdir(parents=True)
    (labels / "calib.json").write_text('{"layout":"pml"}')
    answers = '{"kind":"note","text":"human answer"}\n'
    (labels / "facts.jsonl").write_text(answers)
    out = workspace.root / "out" / video.stem
    out.mkdir(parents=True)
    (out / "g0.json").write_text('{"log":[]}')
    status, project, _ = client.request(
        "/api/projects", {"source": str(video), "games": [21938]}
    )
    assert status == HTTPStatus.CREATED
    assert project["export_signature"] is None
    assert project["artifacts"] == []
    assert not project["has_fit"]
    assert not project["has_hands"]
    assert project["needs_prepare"]
    assert project["inputs_changed"]
    assert (labels / "facts.jsonl").read_text() == answers


def test_clip_cache_does_not_reuse_video_from_replaced_source(
    web: "tuple[Client, Workspace, list[list[str]]]", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify clip cache does not reuse video from replaced source."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    made = []

    def fake_ffmpeg(command: "list[str]", **_unused_kwargs: object) -> None:
        made.append(Path(command[-1]))
        made[-1].write_bytes(b"clip")

    monkeypatch.setattr(state.processes, "run", fake_ffmpeg)
    before = state.clip(0, 2, "frame")
    Path(p["video"]).write_bytes(b"new source bytes")
    after = state.clip(0, 2, "frame")
    assert before != after
    assert before.exists()
    assert after.exists()
    assert len(made) == 2


def test_source_changed_during_conversion_cannot_authenticate_new_exports(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify source changed during conversion cannot authenticate new exports."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    original = workspace.runner

    def replace_during_run(args: "list[str]", project: "dict") -> None:
        original(args, project)
        Path(project["video"]).write_bytes(b"changed while converting")

    workspace.runner = replace_during_run
    client.request(f"/api/projects/{key}/analyze", {})
    result = wait_for_job(client, key)
    assert result["status"] == "failed"
    assert "changed during analysis" in result["job"]["error"]
    assert result["artifacts"] == []
    assert result["needs_prepare"]


def test_calibration_edits_clear_checks_and_block_stale_rebuild(
    web: "tuple[Client, Workspace, list[list[str]]]", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify calibration edits clear checks and block stale rebuild."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    state.hands = [{"hand": 0}]
    state._checks = {"pond:TL": {"level": "ok"}}
    state.work.mkdir(parents=True)
    old_clip = state.clip
    made = []

    def fake_ffmpeg(command: "list[str]", **_unused_kwargs: object) -> None:
        made.append(Path(command[-1]))
        made[-1].write_bytes(b"clip")

    monkeypatch.setattr(state.processes, "run", fake_ffmpeg)
    before = old_clip(0, 2, "frame")
    client.request(f"/review/{p['id']}/api/calib", {"overhead": {"center": [981, 543]}})
    assert not state._checks
    assert (state.work / "calibration.changed").exists()
    after = old_clip(0, 2, "frame")
    assert before != after
    assert len(made) == 2
    with pytest.raises(ValueError, match="Analyze recording"):
        state._run_decode("all")


@pytest.mark.parametrize("edit_running_hand", [False, True])
def test_http_answers_saved_during_rebuild_are_applied_by_the_next_job(
    *,
    web: "tuple[Client, Workspace, list[list[str]]]",
    monkeypatch: "pytest.MonkeyPatch",
    edit_running_hand: bool,
) -> None:
    """Exercise the real HTTP admission guard, journal and freshness receipts."""
    client, workspace, _ = web
    project = create_local(client, workspace)
    state = workspace.review_state(project["id"])
    prefix = f"/review/{project['id']}/api"
    state.hands = [
        {"hand": i, "game": 0, "kyoku": i, "honba": 0, "corner_wind": {"TL": "E"}}
        for i in range(2)
    ]
    for entry in state.hands:
        target = state.decode_path(entry["hand"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**entry, "items": []}))
        os.utime(target, (100, 100))
    state.out.mkdir(parents=True, exist_ok=True)
    (state.out / "g0.json").write_text('{"log":[]}')
    entered, release = threading.Event(), threading.Event()
    batches = []

    def child(args: "list[str]", **_unused_kwargs: object) -> "SimpleNamespace":
        selected = [int(value) for value in args[4].split(",")]
        batches.append(selected)
        if len(batches) == 1:
            entered.set()
            assert release.wait(5)
        for hand in selected:
            state.decode_path(hand).write_text(json.dumps({"hand": hand, "items": []}))
        (state.out / "g0.json").write_text('{"log":[],"updated":true}')
        return SimpleNamespace(returncode=0, stderr="")

    def completed() -> "dict":
        for worker in state._threads:
            worker.join(timeout=5)
        job = client.request(prefix + "/decode_pending")[1]
        assert not job["running"]
        assert not job["error"]
        return job

    monkeypatch.setattr(state.processes, "run", child)
    first = {"hand": 0, "kind": "draw", "seat": "E", "j": 0, "tile": "2p"}
    assert client.request(prefix + "/facts", first)[0] == HTTPStatus.OK
    assert client.request(prefix + "/decode_pending", {})[0] == HTTPStatus.OK
    target_hand = 0 if edit_running_hand else 1
    second = {**first, "hand": target_hand, "j": 1, "tile": "3p"}
    try:
        assert entered.wait(5)
        status, saved, _ = client.request(prefix + "/facts", second)
        assert status == HTTPStatus.OK
        assert saved["tile"] == "3p"
        # Removal and replacement also remain safe while the child owns its
        # older input snapshot. All three requests cross the real HTTP guard.
        assert client.request(prefix + "/facts/delete", {"ts": saved["ts"]})[1] == {
            "deleted": 1
        }
        assert (
            client.request(prefix + "/facts", {**second, "tile": "4p"})[0]
            == HTTPStatus.OK
        )
        assert client.request(prefix + "/decode_pending")[1]["running"]
        assert client.request(prefix + "/calib", {})[0] == HTTPStatus.CONFLICT
    finally:
        release.set()
    assert completed()["pending"] == [target_hand]
    assert [fact["tile"] for fact in client.request(prefix + "/facts")[1]] == [
        "2p",
        "4p",
    ]
    assert client.request(prefix + "/decode_pending", {})[0] == HTTPStatus.OK
    assert completed()["pending"] == []
    assert batches == [[0], [target_hand]]


def test_active_review_job_blocks_calibration_and_additional_jobs(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify active review job blocks calibration and additional jobs."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    state.jobs["decode_all"] = {"running": True}
    try:
        # Repeat actual requests without retrying failures: closing an unread
        # JSON body used to intermittently replace the 409 with WinError 10053.
        for _ in range(25):
            for endpoint in (
                "calib",
                "label",
                "decode_pending",
                "decode_all",
                "decode/0",
            ):
                status, error, _headers = client.request(
                    f"/review/{p['id']}/api/{endpoint}", {}
                )
                assert status == HTTPStatus.CONFLICT
                assert "review job is running" in error["error"]
        assert (
            client.request(f"/api/projects/{p['id']}/settings", {"games": [42]})[0]
            == HTTPStatus.BAD_REQUEST
        )
    finally:
        state.jobs["decode_all"]["running"] = False


def test_review_revision_changes_when_another_process_updates_results(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify review revision changes when another process updates results."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    first = client.request(f"/review/{p['id']}/api/revision")[1]
    path = state.work / "decode" / "00.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"items":[]}')
    second = client.request(f"/review/{p['id']}/api/revision")[1]
    assert first != second


def test_busy_rejection_drain_is_bounded_for_incomplete_or_large_bodies(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify busy rejection drain is bounded for incomplete or large bodies."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    state.jobs["decode_all"] = {"running": True}
    address = urlparse(client.base)
    try:
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
                connection.sendall(
                    request.encode()
                )  # Deliberately do not deliver the advertised body.
                response = connection.recv(4096)
                assert response.startswith(b"HTTP/1.1 409")
                assert time.monotonic() - started < 1.5
    finally:
        state.jobs["decode_all"]["running"] = False


def test_project_request_limit_survives_rejection_drain(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify project request limit survives rejection drain."""
    client, _, _ = web
    status, error, _ = client.request("/api/projects", b" " * 65537)
    assert status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert error["error"] == "Request body is too large."


def test_review_text_round_trips_as_json(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Escaping belongs to Vue rendering; the API preserves the original evidence."""
    client, workspace, _ = web
    project = create_local(client, workspace)
    state = workspace.review_state(project["id"])
    payload = '<img src=x onerror="alert(1)">'
    state.hands = [
        {
            "hand": 0,
            "game": 0,
            "kyoku": 0,
            "honba": 0,
            "corner_wind": {"TL": "E"},
            "nicks": {"TL": payload},
        }
    ]
    status, result, headers = client.request(f"/review/{project['id']}/api/hand/0")
    assert status == HTTPStatus.OK
    assert result["entry"]["nicks"]["TL"] == payload
    assert "application/json" in headers["Content-Type"]


def test_closing_workspace_interrupts_real_job_and_prevents_next_phase(
    web: "tuple[Client, Workspace, list[list[str]]]",
) -> None:
    """Verify closing workspace interrupts real job and prevents next phase."""
    client, workspace, _ = web
    p = create_local(client, workspace)
    workspace.runner = lambda _args, project: workspace._run_command(
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('[running] test job', flush=True); time.sleep(60)",
        ],
        project,
    )
    assert (
        client.request(f"/api/projects/{p['id']}/prepare", {})[0] == HTTPStatus.ACCEPTED
    )
    deadline = time.monotonic() + 5
    while not workspace.snapshot(p["id"])["job"]["log"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert workspace.snapshot(p["id"])["job"]["log"]
    workspace.close()
    stopped = workspace.snapshot(p["id"])
    assert stopped["status"] == "interrupted"
    assert not stopped["job"]["running"]
    assert Workspace(workspace.root).snapshot(p["id"])["status"] == "interrupted"
    with pytest.raises(ValueError, match="closing"):
        workspace.start(p["id"], "prepare")


def _seed_review_hands(
    client: Client, workspace: Workspace
) -> tuple[review.ReviewState, str]:
    """Publish three independent hand results for pending-rebuild assertions."""
    project = create_local(client, workspace)
    prefix = f"/review/{project['id']}/api"
    state = workspace.review_state(project["id"])
    state.hands = [
        {"hand": i, "game": 0, "kyoku": i, "honba": 0, "corner_wind": {"BR": "N"}}
        for i in range(3)
    ]
    state.work.mkdir(parents=True, exist_ok=True)
    (state.work / "hands.json").write_text(json.dumps(state.hands))
    (state.work / "record.json").write_text(
        json.dumps([record.to_dict(record.Game(21938, {}, {}, []))])
    )
    state.out.mkdir(parents=True, exist_ok=True)
    (state.out / "g0.json").write_text('{"log":[]}')
    state.decode_path(0).parent.mkdir(parents=True, exist_ok=True)
    for entry in state.hands:
        state.decode_path(entry["hand"]).write_text(
            json.dumps({**entry, "items": [], "score": None, "stats": {"turns": 0}})
        )
    return state, prefix


def _review_rebuild_doubles(
    state: review.ReviewState, monkeypatch: pytest.MonkeyPatch
) -> tuple[list, list]:
    """Record selected reconstructions and simulate their published exports."""
    decoded, written = [], []

    def rebuild(
        work: "Path",
        hands: "list[dict]",
        games: "list[Game]",
        *,
        options: decode.DecodeRunOptions,
        **_unused_kwargs: object,
    ) -> None:
        only = options.only
        assert options.force
        assert only == {0, 2}
        decoded.append(only)
        for entry in hands:
            if only is not None and entry["hand"] in only:
                path = state.decode_path(entry["hand"])
                value = json.loads(path.read_text())
                value["rebuilt"] = True
                path.write_text(json.dumps(value))

    def outputs(
        out: "Path",
        games: "list[Game]",
        decodes: "list[dict]",
        hands: "list[dict]",
        name: "str",
    ) -> None:
        written.append([d["hand"] for d in decodes])
        (out / "g0.json").write_text('{"log":[],"rebuilt":true}')

    monkeypatch.setattr(decode, "run_decode", rebuild)
    monkeypatch.setattr(cli, "write_outputs", outputs)

    return decoded, written


def _change_review_facts(client: Client, prefix: str) -> None:
    """Make an addition and deletion that must both schedule reconstruction."""
    assert (
        client.request(
            prefix + "/facts",
            {"hand": 0, "kind": "draw", "seat": "N", "tile": "2p", "j": 0},
        )[0]
        == HTTPStatus.OK
    )
    deleted = client.request(
        prefix + "/facts",
        {"hand": 2, "kind": "draw", "seat": "N", "tile": "1z", "j": 0},
    )[1]
    assert (
        client.request(prefix + "/facts/delete", {"ts": deleted["ts"]})[1]["deleted"]
        == 1
    )


def _assert_rebuild_receipts(state: review.ReviewState) -> None:
    """Check that only the selected hands received successful receipts."""
    receipts = json.loads((state.work / "review-changes.json").read_text())["rebuilds"]
    assert set(receipts) == {"0", "2"}
    assert all(r["success"] for r in receipts.values())


def _wait_for_rebuild(client: Client, prefix: str) -> dict:
    """Wait for the review child to publish success or a retryable failure."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.request(prefix + "/decode_pending")[1]
        if not job["running"]:
            return job
        time.sleep(0.01)
    pytest.fail("Review rebuild did not finish")
