"""HTTP integration for import → prepare → analysis → review → downloadable output.

Only the expensive subprocess boundary is replaced. Real TCP requests exercise
routing, body streaming, validation, persistent manifests and project isolation.
"""

from tests.paths import ROOT
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from video2tenhou.tool.app import make_server
from video2tenhou.tool.workflow import Workspace


class Client:
    def __init__(self, server):
        self.base = f"http://127.0.0.1:{server.server_port}"

    def request(self, path, body=None, *, headers=None, raw=False):
        data = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        request_headers = {"X-Video2Tenhou": "1", "Content-Type": "application/json", **(headers or {})}
        request = Request(self.base + path, data=data, headers=request_headers)
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            content = response.read()
            return response.status, content if raw else json.loads(content), response.headers


def test_review_items_project_certified_confidence_without_changing_evidence(web):
    """Review items gain certified coverage/margins from their confidence rows."""
    client, workspace, _ = web
    project = create_local(client, workspace)
    state = workspace.review_state(project["id"])
    state.hands = [dict(hand=0, game=0, kyoku=0, honba=0, corner_wind={"BR": "N"})]
    items = [
        dict(kind="draw", seat="N", j=0, tile="1m"),
        dict(kind="discard", seat="N", j=0, tile="2p"),
        dict(kind="result", seat="N", tiles=["1m"] * 13),
        dict(kind="draw", seat="N", j=1, margin=0.25),
        dict(kind="discard", seat="N"),  # No turn identity: do not borrow another row.
        dict(kind="call", seat="N", alternative_gap=0.01),
    ]
    confidence = [
        dict(field="draw", seat="E", turn=0, margin=0.9, lost=False),
        dict(field="draw", seat="N", turn=0, margin=0, alternative_gap=80, lost=True),
        dict(field="discard", seat="N", turn=0, margin=0.1, conf=0.99, lost=False),
        dict(field="haipai", seat="N", turn=-1, margin=None, lost=False),
        dict(field="draw", seat="N", turn=1, margin=0.4, lost=False),
        dict(field="discard", seat="N", turn=None, margin=0.2, lost=True),
    ]
    path = state.decode_path(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"items": items, "confidence": confidence}).encode()
    path.write_bytes(original)
    status, rows, _ = client.request(f"/review/{project['id']}/api/items")
    assert status == 200 and path.read_bytes() == original
    assert [row["idx"] for row in rows] == list(range(6))
    assert all(row["hand"] == 0 for row in rows)
    assert [(row.get("margin"), row.get("lost")) for row in rows[:4]] == [
        (0, True), (0.1, False), (None, None), (0.25, False)
    ]
    assert rows[2]["tiles"] == ["1m"] * 13
    assert "margin" not in rows[4] and "lost" not in rows[4]
    assert "margin" not in rows[5]


def test_prepare_storage_failure_is_retryable_through_http(web, monkeypatch):
    client, workspace, commands = web
    project = create_local(client, workspace)
    key = project["id"]
    manifest = workspace.projects_dir / f"{key}.json"
    before = manifest.read_bytes()
    original = Path.replace
    def denied(source, target):
        if target == manifest:
            raise PermissionError("manifest temporarily unavailable")
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", denied)
        status, error, _ = client.request(f"/api/projects/{key}/prepare", {})
        assert status == 400 and "Could not save the project" in error["error"]
        current = client.request(f"/api/projects/{key}")[1]
        assert current["status"] == "failed" and not current["job"]["running"]
        assert not commands and manifest.read_bytes() == before
    assert client.request(f"/api/projects/{key}/prepare", {})[0] == 202
    assert wait_for_job(client, key)["status"] == "ready"
    assert len(commands) == 1


def test_pending_rebuild_selects_server_changes_and_preserves_other_hands(web, monkeypatch):
    """Real HTTP/job/child-program flow; only expensive reconstruction is doubled."""
    import sys
    from types import SimpleNamespace
    from video2tenhou import cli
    from video2tenhou.engine import decode
    from video2tenhou.record import Game, to_dict
    client, workspace, _ = web
    project = create_local(client, workspace)
    prefix = f"/review/{project['id']}/api"
    state = workspace.review_state(project["id"])
    state.hands = [dict(hand=i, game=0, kyoku=i, honba=0, corner_wind={"BR": "N"}) for i in range(3)]
    state.work.mkdir(parents=True, exist_ok=True)
    (state.work / "hands.json").write_text(json.dumps(state.hands))
    (state.work / "record.json").write_text(json.dumps([to_dict(Game(21938, {}, {}, []))]))
    state.out.mkdir(parents=True, exist_ok=True)
    (state.out / "g0.json").write_text('{"log":[]}')
    state.decode_path(0).parent.mkdir(parents=True, exist_ok=True)
    for entry in state.hands:
        state.decode_path(entry["hand"]).write_text(json.dumps({**entry, "items": [], "score": None, "stats": {"turns": 0}}))
    untouched = state.decode_path(1).read_bytes(), state.decode_path(1).stat().st_mtime_ns
    assert client.request(prefix + "/facts", {"hand": 0, "kind": "draw", "seat": "N", "tile": "2p", "j": 0})[0] == 200
    deleted = client.request(prefix + "/facts", {"hand": 2, "kind": "draw", "seat": "N", "tile": "1z", "j": 0})[1]
    assert client.request(prefix + "/facts/delete", {"ts": deleted["ts"]})[1]["deleted"] == 1
    facts_before = (state.labels / "facts.jsonl").read_bytes()
    assert client.request(prefix + "/decode_pending")[1]["pending"] == [0, 2]
    assert [h["hand"] for h in client.request(prefix + "/hands")[1] if h["pending_rebuild"]] == [0, 2]
    entered, release = threading.Event(), threading.Event()
    decoded, written, children = [], [], []
    fail = False
    def rebuild(work, hands, games, *, force, only, **kwargs):
        assert force and only == {0, 2}
        decoded.append(only)
        for entry in hands:
            if entry["hand"] in only:
                path = state.decode_path(entry["hand"])
                value = json.loads(path.read_text())
                value["rebuilt"] = True
                path.write_text(json.dumps(value))
    def outputs(out, games, decodes, hands, name):
        written.append([d["hand"] for d in decodes])
        (out / "g0.json").write_text('{"log":[],"rebuilt":true}')
    monkeypatch.setattr(decode, "run_decode", rebuild)
    monkeypatch.setattr(cli, "write_outputs", outputs)
    def child(args, **kwargs):
        children.append(args)
        entered.set()
        assert release.wait(5)
        if fail:
            return SimpleNamespace(returncode=1, stderr="recognition cache changed; Analyze recording")
        # Execute the actual child program with a model-free decoder and writer.
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", ["-c", *args[3:]])
            exec(args[2], {})
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(state.processes, "run", child)
    try:
        status, job, _ = client.request(prefix + "/decode_pending", {"hands": [1]})
        assert status == 200 and job["hands"] == [0, 2] and entered.wait(5)
        assert client.request(prefix + "/facts", {"hand": 1})[0] == 409
        assert client.request(prefix + "/decode_pending")[1]["running"]
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.request(prefix + "/decode_pending")[1]
        if not job["running"]:
            break
        time.sleep(.01)
    assert not job["running"] and job["error"] is None and job["pending"] == []
    assert decoded == [{0, 2}] and written == [[0, 1, 2]]
    assert job["hands_done"] == job["hands_total"] == 2
    assert (state.decode_path(1).read_bytes(), state.decode_path(1).stat().st_mtime_ns) == untouched
    assert (state.labels / "facts.jsonl").read_bytes() == facts_before
    receipts = json.loads((state.work / "review-changes.json").read_text())["rebuilds"]
    assert set(receipts) == {"0", "2"} and all(r["success"] for r in receipts.values())
    assert client.request(prefix + "/decode_pending", {})[1]["pending"] == []
    assert len(children) == 1  # No pending changes means no child, even after a prior job.
    # Failure remains retryable, without acknowledging the newly saved answer.
    assert client.request(prefix + "/facts", {"hand": 2, "kind": "draw", "seat": "N", "tile": "4z", "j": 0})[0] == 200
    fail = True
    assert client.request(prefix + "/decode_pending", {})[0] == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.request(prefix + "/decode_pending")[1]
        if not job["running"]:
            break
        time.sleep(.01)
    assert job["pending"] == [2] and "Analyze recording" in job["error"]
    assert job["hands"] == [2] and len(children) == 2


@pytest.fixture
def web(tmp_path, monkeypatch):
    from video2tenhou import layout
    from video2tenhou.tool import server as review
    monkeypatch.setattr(review, "ROOT", tmp_path)
    monkeypatch.setattr(layout, "LABEL_DIR", tmp_path / "labels")
    commands = []

    def runner(args, project):
        commands.append(args)
        name = Path(project["video"]).stem
        if "download" in args:
            Path(project["video"]).parent.mkdir(parents=True, exist_ok=True)
            Path(project["video"]).write_bytes(b"downloaded video")
        elif "calib" in args:
            path = tmp_path / "labels" / name / "calib.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"video": name, "layout": "pml"}))
        elif "convert" in args:
            work = tmp_path / "work" / name
            work.mkdir(parents=True, exist_ok=True)
            (work / "hands.json").write_text(json.dumps([{"hand": 0, "game": 0, "kyoku": 0, "honba": 0,
                                                         "corner_wind": {"TL": "E", "TR": "S", "BL": "W", "BR": "N"}}]))
            (work / "record.json").write_text(json.dumps([{"id": game, "players": {}, "final": {}, "hands": []} for game in project["games"]]))
            output = tmp_path / "out" / name
            output.mkdir(parents=True, exist_ok=True)
            (output / "g0.json").write_text('{"log": []}')
            (output / "g0.html").write_text("<h1>Replay links</h1>")
            (output / "review.json").write_text("[]")
    server = make_server(tmp_path, 0, runner=runner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Client(server), server.workspace, commands
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def wait_for_job(client, key):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status, result, _ = client.request(f"/api/projects/{key}")
        assert status == 200
        if not result["job"]["running"]:
            return result
        time.sleep(0.01)
    pytest.fail("The background job did not finish.")


def create_local(client, workspace, name="recording.mp4"):
    video = workspace.root / "samples" / name
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"recording")
    status, project, _ = client.request("/api/projects", {"source": str(video), "games": [21938, 21939]})
    assert status == 201
    return project


def test_import_prepare_analyze_review_and_export(web):
    client, workspace, commands = web
    data = b"video bytes" * 200000  # crosses the streaming chunk boundary
    status, upload, _ = client.request("/api/upload", data, headers={"X-Filename": "broadcast.mp4"})
    assert status == 201
    assert Path(upload["path"]).read_bytes() == data
    status, project, _ = client.request("/api/projects", {"source": upload["path"], "games": [21938, 21939]})
    assert status == 201
    key = project["id"]
    assert client.request(f"/api/projects/{key}/analyze", {})[0] == 400
    assert client.request(f"/api/projects/{key}/prepare", {})[0] == 202
    ready = wait_for_job(client, key)
    assert ready["status"] == "ready" and ready["has_fit"]
    assert client.request(f"/api/projects/{key}/analyze", {})[0] == 202
    done = wait_for_job(client, key)
    assert done["status"] == "complete" and done["open_items"] == 0
    assert "g0.json" in done["artifacts"]
    assert client.request(f"/review/{key}/api/hands")[1][0]["hand"] == 0
    status, log, headers = client.request(f"/exports/{key}/g0.json")
    assert status == 200 and log == {"log": []}
    assert headers["Content-Disposition"] == 'attachment; filename="g0.json"'
    assert client.request(f"/exports/{key}/g0.html", raw=True)[1] == b"<h1>Replay links</h1>"
    assert "--skip-fit-check" not in commands[-1]
    assert commands[-1][-4:] == ["--game", "21938", "--game", "21939"]
    reopened = Workspace(workspace.root)
    assert reopened.snapshot(key)["status"] == "complete"


@pytest.mark.parametrize("source", [
    "https://www.twitch.tv/videos/123",
    "https://www.twitch.tv/videos/123?foo=bar#fragment",
    "https://www.youtube.com/watch?v=example",
    "https://vimeo.com/123",
    "http://example.test/video.mp4?token=abc",
])
def test_url_preparation_delegates_to_downloader(web, source):
    client, workspace, commands = web
    status, p, _ = client.request("/api/projects", {"source": source, "kind": "url", "games": [1]})
    assert status == 201
    assert client.request(f"/api/projects/{p['id']}/prepare", {})[0] == 202
    assert wait_for_job(client, p["id"])["status"] == "ready"
    assert commands[0][-4:] == ["download", "--", source, p["video"]]
    assert Path(p["video"]).parent == workspace.root / "samples"
    assert client.request(f"/api/projects/{p['id']}/prepare", {})[0] == 202
    wait_for_job(client, p["id"])
    assert sum("download" in command for command in commands) == 1


def test_url_recording_identity_includes_query(web):
    client, _, _ = web
    def create(source):
        return client.request("/api/projects", {"source": source, "kind": "url", "games": [1]})
    first = "https://www.youtube.com/watch?v=first"
    status, p, _ = create(first)
    assert status == 201
    status, other, _ = create("https://www.youtube.com/watch?v=second")
    assert status == 201 and p["video"] != other["video"]
    assert create(first)[0] == 400
    assert create(" ")[0] == 400


def test_native_results_use_current_json_and_shared_replay_links(web):
    """Serve whole-game and single-hand links without the CLI HTML artifact.

    Rebuilding a JSON export changes its revision, and changing project inputs
    hides both the file download and native results until analysis completes.
    """
    from urllib.parse import unquote
    from video2tenhou.tenhou6 import Game, Kyoku, Ryukyoku

    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    for action in ("prepare", "analyze"):
        assert client.request(f"/api/projects/{key}/{action}", {})[0] == 202
        wait_for_job(client, key)
    output = workspace.root / "out" / p["name"]
    game = Game(names=['A<&"', "B", "C", "D"], title=["Fixture", ""],
                kyokus=[Kyoku(3, 1, 0, [25000] * 4, result=Ryukyoku()),
                        Kyoku(4, 0, 0, [25000] * 4, result=Ryukyoku())])
    (output / "g0.json").write_text(game.dumps(), encoding="utf-8")
    (output / "g0.html").unlink()
    status, result, _ = client.request(f"/api/projects/{key}/results")
    assert status == 200
    row = result["games"][0]
    assert row["names"] == game.names and row["record_id"] == p["games"][0]
    assert row["viewer_url"] == game.viewer_url()
    assert [(h["round"], h["honba"]) for h in row["hands"]] == [("East 4", 1), ("South 1", 0)]
    for i, hand in enumerate(row["hands"]):
        assert hand["editor_url"] == game.editor_url(i)
        payload = json.loads(unquote(hand["editor_url"].split("#json=", 1)[1]))
        assert payload["log"] == [game.kyokus[i].dump()]
    assert client.request(row["download"])[1] == game.to_dict()
    game.kyokus.append(Kyoku(4, 1, 0, [25000] * 4, result=Ryukyoku()))
    (output / "g0.json").write_text(game.dumps(), encoding="utf-8")
    fresh = client.request(f"/api/projects/{key}/results")[1]
    assert fresh["revision"] != result["revision"] and len(fresh["games"][0]["hands"]) == 3
    status, _, _ = client.request(f"/review/{key}/api/facts", {"hand": 0, "kind": "draw", "seat": "N", "j": 0, "t": 10, "tile": "2p"})
    assert status == 200
    pending = client.request(f"/api/projects/{key}/results")[1]
    assert pending["pending_rebuilds"] == [0] and pending["pending_games"] == [0]
    assert client.request(f"/api/projects/{key}")[1]["pending_rebuilds"] == [0]
    other = create_local(client, workspace, "other.mp4")
    assert client.request(f"/api/projects/{other['id']}/results")[1]["games"] == []
    assert client.request(f"/api/projects/{key}/settings", {"games": [42]})[0] == 200
    assert client.request(f"/api/projects/{key}/results")[1]["games"] == []
    assert client.request(row["download"])[0] == 404


def test_review_facts_are_project_scoped(web):
    client, workspace, _ = web
    projects = [create_local(client, workspace, name) for name in ("a.mp4", "b.mp4")]
    for project in projects:
        state = workspace.review_state(project["id"])
        state.hands = [{"hand": 0, "game": 0, "kyoku": 3, "honba": 1, "corner_wind": {"BR": "N"}}]
    one, two = [p["id"] for p in projects]
    status, fact, _ = client.request(f"/review/{one}/api/facts", {"hand": 0, "kind": "draw", "seat": "N", "t": 20, "tile": "2p"})
    assert status == 200 and fact["corner"] == "BR"
    assert client.request(f"/review/{one}/api/facts")[1][0]["tile"] == "2p"
    assert client.request(f"/review/{two}/api/facts")[1] == []
    assert client.request(f"/review/{one}/api/facts/delete", {"ts": fact["ts"]})[1] == {"deleted": 1}
    assert client.request(f"/review/{one}/api/facts")[1] == []


def test_cross_origin_traversal_and_upload_validation(web):
    client, workspace, _ = web
    assert client.request("/api/projects", {}, headers={"Origin": "https://evil.test"})[0] == 403
    assert client.request("/api/projects", {}, headers={"X-Video2Tenhou": ""})[0] == 403
    assert client.request("/api/workspace", headers={"Host": "evil.test"})[0] == 403
    assert client.request("/api/upload", b"x", headers={"X-Filename": "../secret.mp4"})[0] == 400
    assert client.request("/api/upload", b"x", headers={"X-Filename": "run.exe"})[0] == 400
    assert client.request("/api/upload", b"", headers={"X-Filename": "empty.mp4"})[0] == 413
    p = create_local(client, workspace)
    assert client.request(f"/exports/{p['id']}/%2e%2e%2fsecret.txt")[0] == 400
    assert client.request("/../pyproject.toml")[0] == 404
    assert client.request(f"/review/{p['id']}/../../studio.html")[0] == 404
    assert client.request("/api/projects/" + "0" * 32)[0] == 404
    assert client.request("/api/projects", {"source": p["video"], "games": [1]})[0] == 400


def test_failed_and_interrupted_jobs_remain_actionable(web):
    client, workspace, _ = web
    p = create_local(client, workspace)
    def fail(_args, _project):
        raise RuntimeError("Detector weights missing")
    workspace.runner = fail
    client.request(f"/api/projects/{p['id']}/prepare", {})
    failed = wait_for_job(client, p["id"])
    assert failed["status"] == "failed" and "Detector weights" in failed["job"]["error"]
    assert Workspace(workspace.root).snapshot(p["id"])["job"]["error"] == failed["job"]["error"]
    project = workspace.project(p["id"])
    project["job"]["running"] = True
    workspace._save(project)
    restarted = Workspace(workspace.root).snapshot(p["id"])
    assert restarted["status"] == "interrupted" and not restarted["job"]["running"]


def test_failed_calibration_has_short_action_and_keeps_cli_details_in_log(web):
    import sys
    client, workspace, _ = web
    p = create_local(client, workspace)
    script = "print('FAIL meld:TL 0 tiles held, 3 cut by the border'); print('video2tenhou review video.mp4 -> Calibrate'); raise SystemExit(1)"
    workspace.runner = lambda args, project: workspace._run_command([sys.executable, "-u", "-c", script], project)
    assert client.request(f"/api/projects/{p['id']}/prepare", {})[0] == 202
    failed = wait_for_job(client, p["id"])
    assert failed["status"] == "failed"
    assert "Calibration" in failed["job"]["error"] and "\n" not in failed["job"]["error"]
    assert "video2tenhou" not in failed["job"]["error"]
    assert any("video2tenhou review" in line for line in failed["job"]["log"])


def test_jobs_are_serialized_and_review_writes_wait(web):
    client, workspace, _ = web
    p = create_local(client, workspace)
    started, release = threading.Event(), threading.Event()
    def slow(_args, _project):
        started.set()
        assert release.wait(5)
    workspace.runner = slow
    try:
        assert client.request(f"/api/projects/{p['id']}/prepare", {})[0] == 202
        assert started.wait(2)
        assert client.request(f"/api/projects/{p['id']}/prepare", {})[0] == 400
        assert client.request(f"/review/{p['id']}/api/facts", {})[0] == 409
    finally:
        release.set()
        wait_for_job(client, p["id"])


def test_studio_and_vendored_tiles_are_served_offline(web):
    client, workspace, _ = web
    status, html, _ = client.request("/", raw=True)
    assert status == 200 and b"Prepare recording" in html
    status, svg, headers = client.request("/tiles/Pin2.svg", raw=True)
    assert status == 200 and b"<svg" in svg
    assert "image/svg+xml" in headers["Content-Type"]
    assert client.request("/tiles/LICENSE.md", raw=True)[0] == 200
    p = create_local(client, workspace)
    status, embedded, _ = client.request(f"/review/{p['id']}/?embedded=1", raw=True)
    assert status == 200 and b"<!doctype html>" in embedded


def test_calibration_save_reloads_geometry_and_evidence_cache(web):
    import numpy as np
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    state._frame_cache[10.0] = np.zeros((2, 2, 3), np.uint8)
    status, saved, _ = client.request(f"/review/{p['id']}/api/calib", {"overhead": {"center": [981, 543], "angle": 46}})
    assert status == 200 and saved["fit"]["overhead"]["source"] == "human"
    assert not state._frame_cache
    status, geometry, _ = client.request(f"/review/{p['id']}/api/calib")
    assert status == 200 and geometry["fit"]["overhead"]["center"] == [981, 543]
    assert state.cal.center == (981, 543) or state.cal.center == [981, 543]
    assert (workspace.root / "labels" / p["name"] / "calib.json").is_file()


def test_scoped_frame_and_clip_routes_return_this_projects_evidence(web, monkeypatch):
    import cv2
    import numpy as np
    client, workspace, _ = web
    a, b = [create_local(client, workspace, name) for name in ("red.mp4", "blue.mp4")]
    for project, color in ((a, (0, 0, 255)), (b, (255, 0, 0))):
        state = workspace.review_state(project["id"])
        monkeypatch.setattr(state, "frame", lambda t, c=color: np.full((24, 32, 3), c, np.uint8))
    for project, channel in ((a, 2), (b, 0)):
        status, jpeg, headers = client.request(f"/review/{project['id']}/api/frame?t=5&region=frame", raw=True)
        assert status == 200 and headers["Content-Type"] == "image/jpeg"
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert frame[:, :, channel].mean() > 250
    clip = workspace.root / "test.mp4"
    clip.write_bytes(b"test clip bytes")
    monkeypatch.setattr(workspace.review_state(a["id"]), "clip", lambda start, end, region: clip)
    status, content, headers = client.request(f"/review/{a['id']}/api/clip?t0=1&t1=4&region=frame", raw=True)
    assert status == 200 and content == b"test clip bytes" and headers["Content-Type"] == "video/mp4"


def test_review_rebuild_subprocess_refreshes_existing_exports(web):
    """Execute the actual review child process and writer with an empty site game.

    No vision is needed: this specifically catches stale exports after review,
    incorrect process arguments, wrong output roots and cached game names.
    """
    from video2tenhou.record import Game, to_dict
    client, workspace, _ = web
    p = create_local(client, workspace)
    work = workspace.root / "work" / p["name"]
    work.mkdir(parents=True)
    game = Game(21938, {"EAST": "Reviewed player"}, {}, [])
    (work / "hands.json").write_text("[]")
    (work / "record.json").write_text(json.dumps([to_dict(game), to_dict(Game(21939, {}, {}, []))]))
    workspace.project(p["id"])["export_signature"] = workspace._signature(p)
    out = workspace.root / "out" / p["name"]
    out.mkdir(parents=True)
    (out / "g0.json").write_text('{"name": ["stale"]}')
    (out / "review.json").write_text('[{"kind":"stale"}]')
    assert client.request(f"/review/{p['id']}/api/decode_all", {})[0] == 200
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status, job, _ = client.request(f"/review/{p['id']}/api/decode_all")
        assert status == 200
        if not job["running"]:
            break
        time.sleep(0.05)
    assert not job["running"] and not job["error"], job
    status, output, _ = client.request(f"/exports/{p['id']}/g0.json")
    assert status == 200 and output["name"][0] == "Reviewed player"
    assert client.request(f"/exports/{p['id']}/review.json")[1] == []
    assert (out / "g0.html").exists() and (out / "report.md").exists()


def test_cli_uses_selected_data_directory_for_stage_outputs(tmp_path, monkeypatch):
    from video2tenhou import cli, paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    captured = []
    monkeypatch.setattr(cli, "cmd_convert", captured.append)
    cli.main(["convert", "recording.mp4", "--game", "21938"])
    assert captured[0].work == str(tmp_path / "work")
    assert captured[0].out == str(tmp_path / "out")


def test_existing_video_folder_is_available_without_copy(web):
    client, workspace, _ = web
    path = workspace.root / "videos" / "week_11.mp4"
    path.parent.mkdir()
    path.write_bytes(b"existing recording")
    status, data, _ = client.request("/api/workspace")
    assert status == 200
    assert {"path": str(path), "name": "videos/week_11.mp4"} in data["sources"]


def test_settings_corrections_hide_old_exports_and_preserve_answers(web):
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    client.request(f"/api/projects/{key}/analyze", {})
    assert wait_for_job(client, key)["artifacts"]
    facts = workspace.root / "labels" / p["name"] / "facts.jsonl"
    facts.write_text('{"kind":"note","text":"keep me"}\n')
    status, changed, _ = client.request(f"/api/projects/{key}/settings", {"games": [22002, 22003], "layout": "pml"})
    assert status == 200 and changed["has_fit"] and changed["stale_exports"]
    assert changed["artifacts"] == [] and not changed["has_hands"]
    assert client.request(f"/exports/{key}/g0.json")[0] == 404
    assert client.request(f"/review/{key}/api/hands")[1] == []
    assert "keep me" in facts.read_text()
    client.request(f"/api/projects/{key}/analyze", {})
    refreshed = wait_for_job(client, key)
    assert refreshed["artifacts"] and not refreshed["stale_exports"]
    assert "keep me" in facts.read_text()


def test_layout_change_requires_preparation_and_rescans_overlay(web):
    from video2tenhou.layout import CALIB_DIR
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    work = workspace.root / "work" / p["name"]
    work.mkdir(parents=True, exist_ok=True)
    overlay = work / "overlay.jsonl"
    overlay.write_text("old layout readings")
    custom = workspace.root / "custom.json"
    custom.write_bytes((CALIB_DIR / "pml.json").read_bytes())
    status, changed, _ = client.request(f"/api/projects/{key}/settings", {"games": p["games"], "layout": str(custom)})
    assert status == 200 and not changed["has_fit"]
    assert not overlay.exists()
    assert client.request(f"/api/projects/{key}/analyze", {})[0] == 400


def test_changed_record_provenance_blocks_manual_export_urls(web):
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
    assert client.request(f"/exports/{key}/g0.json")[0] == 404


def test_replaced_source_hides_exports_and_retires_evidence_without_rehashing_polls(web, monkeypatch):
    import os
    from video2tenhou import cache

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
    def counted(path):
        calls.append(path)
        return digest(path)
    monkeypatch.setattr(cache, "sha256_file", counted)
    for _ in range(3):
        assert client.request(f"/api/projects/{key}")[1]["artifacts"]
    assert calls == []
    source = Path(p["video"])
    stat = source.stat()
    replacement = source.with_suffix(".new")
    replacement.write_bytes(b"different")  # same length, same mtime, different file identity
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(source)
    status, changed, _ = client.request(f"/api/projects/{key}")
    assert status == 200 and changed["artifacts"] == [] and changed["stale_exports"]
    assert not changed["has_hands"] and not changed["has_fit"]
    assert client.request(f"/exports/{key}/g0.json")[0] == 404
    assert client.request(f"/review/{key}/api/hands")[1] == []
    fresh_state = workspace.review_state(key)
    assert fresh_state is not old_state and old_state.processes.closing
    assert fresh_state.revision() != old_revision
    assert facts.read_bytes() == saved_facts
    assert len(calls) == 1
    assert client.request(f"/api/projects/{key}/analyze", {})[0] == 400


def test_missing_source_is_a_recoverable_project_state(web):
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    for action in ("prepare", "analyze"):
        client.request(f"/api/projects/{key}/{action}", {})
        wait_for_job(client, key)
    source = Path(p["video"])
    source.unlink()
    status, missing, _ = client.request(f"/api/projects/{key}")
    assert status == 200 and missing["artifacts"] == [] and not missing["has_fit"]
    status, error, _ = client.request(f"/api/projects/{key}/prepare", {})
    assert status == 400 and "Restore the local video" in error["error"]
    source.write_bytes(b"restored recording")
    for action in ("prepare", "analyze"):
        assert client.request(f"/api/projects/{key}/{action}", {})[0] == 202
        result = wait_for_job(client, key)
    assert result["artifacts"] and not result["stale_exports"]


@pytest.mark.parametrize("missing", [("source_sha256",), ("export_signature",), ("source_sha256", "export_signature")])
def test_missing_project_provenance_requires_regeneration_and_preserves_answers(web, missing):
    from video2tenhou.calm import REGIONS, region_key
    from video2tenhou.layout import Calibration

    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    for action in ("prepare", "analyze"):
        client.request(f"/api/projects/{key}/{action}", {})
        wait_for_job(client, key)
    cal = Calibration.load("pml", p["video"])
    done = workspace.root / "work" / p["name"] / "reads/00/done.json"
    done.parent.mkdir(parents=True)
    manifest = {"geometry": {r: region_key(cal, r) for r in REGIONS},
                "identity": {"source": workspace._source(p)}}
    done.write_text(json.dumps(manifest))
    facts = workspace.root / "labels" / p["name"] / "facts.jsonl"
    answers = json.dumps({"hand": 0, "game": 0, "kyoku": 0, "honba": 0, "kind": "note", "text": "human answer"}) + "\n"
    facts.write_text(answers)
    stored = workspace.project(key)
    for field in missing:
        del stored[field]
    status, result, _ = client.request(f"/api/projects/{key}")
    assert status == 200
    assert result["artifacts"] == [] and not result["has_hands"] and not result["has_fit"]
    assert result["needs_prepare"] and result["inputs_changed"]
    assert result["export_signature"] is None
    assert facts.read_text() == answers and done.read_text() == json.dumps(manifest)
    assert client.request(f"/exports/{key}/g0.json")[0] == 404
    assert client.request(f"/api/projects/{key}/analyze", {})[0] == 400
    restarted = Workspace(workspace.root)
    try:
        assert restarted.snapshot(key)["artifacts"] == []
    finally:
        restarted.close()
    for action in ("prepare", "analyze"):
        assert client.request(f"/api/projects/{key}/{action}", {})[0] == 202
        result = wait_for_job(client, key)
    assert result["artifacts"] and result["has_hands"] and result["has_fit"]
    assert facts.read_text() == answers


def test_new_project_does_not_adopt_untracked_outputs(web):
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
    status, project, _ = client.request("/api/projects", {"source": str(video), "games": [21938]})
    assert status == 201
    assert project["export_signature"] is None and project["artifacts"] == []
    assert not project["has_fit"] and not project["has_hands"]
    assert project["needs_prepare"] and project["inputs_changed"]
    assert (labels / "facts.jsonl").read_text() == answers


def test_clip_cache_does_not_reuse_video_from_replaced_source(web, monkeypatch):
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    made = []
    def fake_ffmpeg(command, **kwargs):
        made.append(Path(command[-1]))
        made[-1].write_bytes(b"clip")
    monkeypatch.setattr(state.processes, "run", fake_ffmpeg)
    before = state.clip(0, 2, "frame")
    Path(p["video"]).write_bytes(b"new source bytes")
    after = state.clip(0, 2, "frame")
    assert before != after and before.exists() and after.exists() and len(made) == 2


def test_source_changed_during_conversion_cannot_authenticate_new_exports(web):
    client, workspace, _ = web
    p = create_local(client, workspace)
    key = p["id"]
    client.request(f"/api/projects/{key}/prepare", {})
    wait_for_job(client, key)
    original = workspace.runner
    def replace_during_run(args, project):
        original(args, project)
        Path(project["video"]).write_bytes(b"changed while converting")
    workspace.runner = replace_during_run
    client.request(f"/api/projects/{key}/analyze", {})
    result = wait_for_job(client, key)
    assert result["status"] == "failed" and "changed during analysis" in result["job"]["error"]
    assert result["artifacts"] == [] and result["needs_prepare"]


def test_calibration_edits_clear_checks_and_block_stale_rebuild(web, monkeypatch):
    import subprocess
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    state.hands = [{"hand": 0}]
    state._checks = {"pond:TL": {"level": "ok"}}
    state.work.mkdir(parents=True)
    old_clip = state.clip
    made = []
    def fake_ffmpeg(command, **kwargs):
        made.append(Path(command[-1]))
        made[-1].write_bytes(b"clip")
    monkeypatch.setattr(state.processes, "run", fake_ffmpeg)
    before = old_clip(0, 2, "frame")
    client.request(f"/review/{p['id']}/api/calib", {"overhead": {"center": [981, 543]}})
    assert not state._checks
    assert (state.work / "calibration.changed").exists()
    after = old_clip(0, 2, "frame")
    assert before != after and len(made) == 2
    with pytest.raises(ValueError, match="Analyze recording"):
        state._run_decode("all")


def test_active_review_job_blocks_fact_and_calibration_writes(web):
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    state.jobs["decode_all"] = {"running": True}
    try:
        # Repeat actual requests without retrying failures: closing an unread
        # JSON body used to intermittently replace the 409 with WinError 10053.
        for _ in range(25):
            for endpoint in ("facts", "facts/delete", "calib", "label"):
                status, error, _headers = client.request(f"/review/{p['id']}/api/{endpoint}", {})
                assert status == 409 and "review job is running" in error["error"]
        assert client.request(f"/api/projects/{p['id']}/settings", {"games": [42]})[0] == 400
    finally:
        state.jobs["decode_all"]["running"] = False


def test_review_revision_changes_when_another_process_updates_results(web):
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    first = client.request(f"/review/{p['id']}/api/revision")[1]
    path = state.work / "decode" / "00.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"items":[]}')
    second = client.request(f"/review/{p['id']}/api/revision")[1]
    assert first != second


def test_busy_rejection_drain_is_bounded_for_incomplete_or_large_bodies(web):
    import socket
    from urllib.parse import urlparse
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    state.jobs["decode_all"] = {"running": True}
    address = urlparse(client.base)
    try:
        for size in (32, 2 * 1024 * 1024):
            with socket.create_connection((address.hostname, address.port), timeout=2) as connection:
                request = (f"POST /review/{p['id']}/api/facts HTTP/1.1\r\n"
                           f"Host: {address.netloc}\r\nX-Video2Tenhou: 1\r\n"
                           f"Content-Type: application/json\r\nContent-Length: {size}\r\n\r\n")
                started = time.monotonic()
                connection.sendall(request.encode())  # Deliberately do not deliver the advertised body.
                response = connection.recv(4096)
                assert response.startswith(b"HTTP/1.0 409")
                assert time.monotonic() - started < 1.5
    finally:
        state.jobs["decode_all"]["running"] = False


def test_project_request_limit_survives_rejection_drain(web):
    client, _, _ = web
    status, error, _ = client.request("/api/projects", b" " * 65537)
    assert status == 413 and error["error"] == "Request body is too large."


def test_untrusted_review_text_is_escaped_before_html_insertion(web, tmp_path):
    import re
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("node not on PATH")
    client, workspace, _ = web
    p = create_local(client, workspace)
    state = workspace.review_state(p["id"])
    payload = '<img src=x onerror="alert(1)">'
    state.hands = [{"hand": 0, "game": 0, "kyoku": 0, "honba": 0, "corner_wind": {"TL": "E"}, "nicks": {"TL": payload}}]
    entry = client.request(f"/review/{p['id']}/api/hand/0")[1]["entry"]
    page = ROOT / "src/video2tenhou/tool/static/index.html"
    script = re.search(r"<script>([\s\S]*)</script>", page.read_text(encoding="utf-8"))[1]
    helpers = script.split("const GLYPH =")[0] + "\nconst GLYPH = {};\n"
    helpers += re.search(r"const tileHtml = .*", script)[0] + "\n"
    helpers += re.search(r"function factDesc\(f\) \{[\s\S]*?\n\}", script)[0] + "\n"
    helpers += "CUR_E = " + json.dumps(entry) + ";\n"
    helpers += "console.log(JSON.stringify([seatName(CUR_E, 'E'), human(CUR_E, " + json.dumps(payload) + "), factDesc({kind:'note',text:" + json.dumps(payload) + "})]));"
    file = tmp_path / "escape-check.js"
    file.write_text(helpers, encoding="utf-8")
    result = subprocess.run(["node", str(file)], capture_output=True, text=True, check=True)
    for rendered in json.loads(result.stdout):
        assert payload not in rendered
        assert "&lt;img" in rendered and "&quot;" in rendered


def test_closing_workspace_interrupts_real_job_and_prevents_next_phase(web):
    import sys
    client, workspace, _ = web
    p = create_local(client, workspace)
    workspace.runner = lambda args, project: workspace._run_command(
        [sys.executable, "-u", "-c", "import time; print('[running] test job', flush=True); time.sleep(60)"], project)
    assert client.request(f"/api/projects/{p['id']}/prepare", {})[0] == 202
    deadline = time.monotonic() + 5
    while not workspace.snapshot(p["id"])["job"]["log"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert workspace.snapshot(p["id"])["job"]["log"]
    workspace.close()
    stopped = workspace.snapshot(p["id"])
    assert stopped["status"] == "interrupted" and not stopped["job"]["running"]
    assert Workspace(workspace.root).snapshot(p["id"])["status"] == "interrupted"
    with pytest.raises(ValueError, match="closing"):
        workspace.start(p["id"], "prepare")
