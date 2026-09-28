"""Correction freshness across hand rebuilds, exported games and server restarts."""
import json
import os
import threading
from types import SimpleNamespace

import pytest

from video2tenhou.tool import server


@pytest.fixture
def review(tmp_path, monkeypatch):
    from video2tenhou import layout
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(layout, "LABEL_DIR", tmp_path / "labels")
    work = tmp_path / "work" / "recording"
    work.mkdir(parents=True)
    hands = [{"hand": i, "game": i // 2, "kyoku": i % 2, "honba": 0,
              "corner_wind": {"TL": "E"}} for i in range(3)]
    (work / "hands.json").write_text(json.dumps(hands))
    state = server.State(tmp_path / "recording.mp4", work.parent, "pml", tmp_path / "out")
    clock = [200.0]
    monkeypatch.setattr(server.time, "time", lambda: clock[0])
    for hand in hands:
        outputs(state, hand["hand"], 100)
    yield state, clock
    state.close()


def write_at(path, timestamp):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    os.utime(path, ns=(int(timestamp * 1e9), int(timestamp * 1e9)))


def outputs(state, hand, timestamp):
    write_at(state.decode_path(hand), timestamp)
    write_at(state.out / f"g{state.hands[hand]['game']}.json", timestamp)


def test_additions_require_both_decode_and_game_export(review):
    state, clock = review
    assert state.pending_rebuilds() == []
    state.add_fact({"hand": 0, "kind": "draw", "seat": "E", "tile": "2p"})
    assert state.pending_rebuilds() == [0]
    write_at(state.decode_path(0), 210)
    assert state.pending_rebuilds() == [0]
    write_at(state.out / "g0.json", 220)
    assert state.pending_rebuilds() == []


def test_deletion_survives_restart_even_when_no_facts_remain(review):
    state, clock = review
    fact = state.add_fact({"hand": 1, "kind": "draw", "tile": "1z"})
    outputs(state, 1, 210)
    clock[0] = 220
    assert state.delete_fact(fact["ts"]) == 1
    assert state.all_facts() == []
    reopened = server.State(state.video, state.work.parent, "pml", state.out.parent)
    try:
        assert reopened.pending_rebuilds() == [1]
        outputs(reopened, 1, 230)
        assert reopened.pending_rebuilds() == []
    finally:
        reopened.close()


def test_rebuilding_one_hand_does_not_clear_other_hand_in_same_game(review, monkeypatch):
    state, clock = review
    for i in (0, 1):
        state.add_fact({"hand": i, "kind": "draw", "tile": "2p"})
    clock[0] = 210
    def child(*args, **kwargs):
        outputs(state, 0, 220)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(state.processes, "run", child)
    state._run_decode("0")
    assert state.pending_rebuilds() == [1]


@pytest.mark.parametrize("remove", [False, True])
def test_edit_during_rebuild_stays_pending_after_completion_and_restart(review, monkeypatch, remove):
    state, clock = review
    fact = state.add_fact({"hand": 0, "kind": "draw", "tile": "2p"})
    clock[0] = 210
    state.jobs["decode_all"] = {"running": True, "started": 205}
    def child(*args, **kwargs):
        clock[0] = 215
        if remove:
            state.delete_fact(fact["ts"])
        else:
            state.add_fact({"hand": 0, "kind": "draw", "tile": "1z"})
        outputs(state, 0, 220)
        assert state.pending_rebuilds() == [0]  # Output exists while the old job is still active.
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(state.processes, "run", child)
    state._run_decode("all")
    state.jobs["decode_all"]["running"] = False
    reopened = server.State(state.video, state.work.parent, "pml", state.out.parent)
    try:
        assert reopened.pending_rebuilds() == [0]
        write_at(reopened.out / "g0.json", 225)  # Another hand's export does not clear this one.
        assert reopened.pending_rebuilds() == [0]
        outputs(reopened, 0, 230)  # A later external full rebuild supersedes the receipt.
        assert reopened.pending_rebuilds() == []
    finally:
        reopened.close()


def test_failed_rebuild_cannot_acknowledge_partially_written_outputs(review, monkeypatch):
    state, clock = review
    state.add_fact({"hand": 0, "kind": "draw", "tile": "2p"})
    clock[0] = 210
    def child(*args, **kwargs):
        outputs(state, 0, 220)
        return SimpleNamespace(returncode=1, stderr="export interrupted")
    monkeypatch.setattr(state.processes, "run", child)
    with pytest.raises(RuntimeError, match="export interrupted"):
        state._run_decode("0")
    assert state.pending_rebuilds() == [0]
    outputs(state, 0, 230)
    assert state.pending_rebuilds() == []


def test_existing_dated_facts_are_detected_without_ledger(review):
    state, clock = review
    (state.labels / "facts.jsonl").write_text(json.dumps({
        "hand": 99, "game": 1, "kyoku": 0, "honba": 0, "ts": 200, "kind": "note"}) + "\n")
    assert state.pending_rebuilds() == [2]  # Current game/round identity wins over a stale global hand index.


@pytest.mark.parametrize("single_hand", [True, False])
def test_review_jobs_share_exclusion_failure_and_shutdown(review, monkeypatch, single_hand):
    state, clock = review
    started, release = threading.Event(), threading.Event()

    def fail(*args):
        started.set()
        assert release.wait(5)
        raise RuntimeError("rebuild interrupted")

    monkeypatch.setattr(state, "_run_decode", fail)
    launch = state.start_redecode if single_hand else lambda _: state.start_job("calib", fail)
    status = launch(0)
    try:
        assert started.wait(5)
        assert status["key"] == (0 if single_hand else "calib")
        assert launch(0) is status  # Retrying a running job never starts another worker.
        with pytest.raises(ValueError, match="Another review job"):
            state.start_redecode(1)
        with pytest.raises(ValueError, match="Another review job"):
            state.start_job("check", lambda: None)
    finally:
        release.set()
        for worker in state._threads:
            worker.join(timeout=5)
    assert status["running"] is False
    assert status["error"] == "rebuild interrupted"
    assert status["done"] == clock[0]
    assert set(status) == {"key", "running", "started", "done", "error", "result"}
    state.close()
    with pytest.raises(ValueError, match="app is closing"):
        launch(0)
