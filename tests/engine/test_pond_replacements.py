"""Contradictory terminal readings remain reviewable without invented turns."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from tests.engine.helpers import obs, row
from video2tenhou.engine.pond_evidence import (
    consume_replacement,
    replacement_question,
    replacement_requests,
)
from video2tenhou.engine.ponds import track_pond


def sequence():
    return [obs(0, 2, row(["2p", "7m"])), obs(5, 6, row(["2p", "1m"]))]


def requests(log, *, call=None, facts=None, end=6):
    turns = [SimpleNamespace(seat="N", slot=s, t=s.t_first, call=call) for s in log]
    return replacement_requests(
        turns, {"corner_wind": {"BR": "N"}}, facts or {}, 0, end
    )


def test_terminal_replacement_survives_serialization_without_extra_discard():
    observations = sequence()
    log = track_pond(observations)
    assert [slot.tile for slot in log] == ["2p", "7m"]
    assert all(slot.t_removed is None for slot in log)
    payload = log[-1].to_dict()["pending_replacement"]
    assert payload["views"][0]["tile"]["p"] == observations[-1]["slots"][-1]["p"]
    assert payload["prefix_complete"] and payload["t0"] == 5
    payload["views"][0]["tile"]["p"][0] = -1
    assert log[-1].pending_replacement["views"][0]["tile"]["p"][0] >= 0
    (request,) = requests(log)
    assert request["window"] == [0, 6] and request["j"] == 1
    item = replacement_question(request)
    assert item["kind"] == "discard" and item["runner_up"] == "1m"
    assert item["tracking_uncertain"] and item["tile"] == "7m"
    assert requests(log)[0]["signature"] == request["signature"]
    log[-1].pending_replacement["views"][0]["tile"]["seen"] += 1
    assert requests(log)[0]["signature"] != request["signature"]
    assert request["pending"]["views"][0]["tile"]["seen"] == 3


def test_returning_old_tile_resolves_transient_candidate():
    log = track_pond(sequence() + [obs(8, 10, row(["2p", "7m"]))])
    assert [slot.tile for slot in log] == ["2p", "7m"]
    assert log[-1].pending_replacement is None and not requests(log)


def test_repeated_refill_preserves_removal_but_requires_independent_call():
    log = track_pond(sequence() + [obs(8, 10, row(["2p", "1m"]))])
    assert [slot.tile for slot in log] == ["2p", "7m", "1m"]
    assert log[1].t_removed == 5
    hidden = SimpleNamespace(human=False, anchor="hidden", seen=10, partial_only=False)
    assert len(requests(log, call=hidden, end=10)) == 1
    independent = SimpleNamespace(
        human=False, anchor="discard", seen=2, partial_only=False
    )
    assert not requests(log, call=independent, end=10)


def test_partial_or_long_gap_stays_uncertain_and_fact_suppresses_identity():
    observations = sequence()
    observations[-1]["partial"] = True
    log = track_pond(observations)
    assert log[1].t_removed is None
    assert requests(log)[0]["pending"]["views"][0]["partial"]
    log[1].pending_replacement["t1"] = 100
    (request,) = requests(log, end=100)
    assert request["window"] is None
    assert replacement_question(request)["tracking_uncertain"]
    assert not requests(log, facts={"discard": [{"seat": "N", "t": 0, "tile": "7m"}]})


def dense_frames():
    frames = []
    for i in range(31):
        tiles = row(["2p", "7m" if i < 10 else "1m"])
        boxes = [
            {**tile, "row": tile["key"][0], "col": tile["key"][1], "role": "tile"}
            for tile in tiles
        ]
        frames.append({"t": i / 5, "boxes": boxes})
    return frames


def test_continuous_evidence_substitutes_once_without_changing_turns():
    log = track_pond(sequence())
    (request,) = requests(log)
    before = log[-1].p.copy()
    frames = dense_frames()
    sparse = np.asarray(request["pending"]["stable_view"]["tile"]["p"]) * 3
    expected = (
        before - sparse + sum(np.asarray(frame["boxes"][-1]["p"]) for frame in frames)
    )
    assert consume_replacement(log[-1], request, frames)
    np.testing.assert_allclose(log[-1].p, expected)
    assert len(log) == 2 and log[-1].t_removed is None
    (fresh,) = requests(log)
    assert fresh["signature"] == request["signature"] and fresh["acquired"]
    assert not replacement_question(fresh)["tracking_uncertain"]
    assert not consume_replacement(log[-1], fresh, frames)
    np.testing.assert_allclose(log[-1].p, expected)


def test_occlusion_changed_prefix_or_possible_call_cannot_fuse():
    for defect in ("count", "gap", "geometry", "prefix", "call"):
        log = track_pond(sequence())
        (request,) = requests(log)
        before = log[-1].p.copy()
        frames = deepcopy(dense_frames())
        if defect == "count":
            frames[15]["boxes"].pop()
        elif defect == "gap":
            del frames[10:15]
        elif defect == "geometry":
            frames[15]["boxes"][-1]["xyxy"] = [1000, 0, 1038, 58]
        elif defect == "prefix":
            frames[15]["boxes"][0]["p"] = frames[15]["boxes"][-1]["p"]
        else:
            request["call_ambiguous"] = True
        assert not consume_replacement(log[-1], request, frames), defect
        np.testing.assert_array_equal(log[-1].p, before)
