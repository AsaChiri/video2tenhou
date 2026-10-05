# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Tile conservation, result constraints and solver confidence proofs."""

from __future__ import annotations

import numpy as np
import pytest

from tests.builders import kinds, observation, posterior, slot, tile_counts
from video2tenhou.engine import rules
from video2tenhou.engine.solver import (
    TI,
    TILES,
    DrawEvidence,
    Facts,
    HandEvidence,
    HandModel,
    HandRole,
    SeatTurn,
    drawn_end,
    hand_evidence,
    state_of,
)


def test_variable_discard_is_certified_even_when_raw_reading_is_kept() -> None:
    turns = {s: [] for s in rules.SEATS}
    p = kinds({"1m": 0.51, "2m": 0.49})
    turns["S"] = [SeatTurn(0, "draw", "1m", 0, 10, discard_p=p)]
    model = HandModel("E", turns, [])
    model.facts = Facts(draws={("S", 0): "3m"})
    tiles = [
        "1m",
        "2m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "9s",
        "1z",
        "1z",
        "3z",
        "6z",
    ]
    model.facts.haipai["S"] = tiles
    after_second = [*tiles, "3m"]
    after_second.remove("2m")
    # The hand view almost offsets the penalty for choosing the second pond
    # reading: raw1m remains best, but alternative2m is only .01 more costly.
    model.hand_ev = [
        HandEvidence(
            "S",
            0,
            after_draw=False,
            e=tile_counts(after_second),
            weight=0.76,
            t0=11,
            t1=12,
        )
    ]
    first = model.solve(workers=1)
    assert first.ok
    model.facts.haipai.update(first.haipai)
    sol = model.solve(workers=1)
    model.certify(sol, workers=1)
    assert ("S", 0) not in sol.discards
    certificate = sol.certificates["discard", "S", 0]
    assert certificate.state == "ambiguous"
    assert certificate.gap == pytest.approx(0.01)
    assert certificate.gap is not None
    assert certificate.margin <= certificate.gap
    assert certificate.runner_up == "2m"


def slot_dict(tile: str) -> dict:
    """Create a hand slot whose posterior is concentrated on one tile."""
    return slot(
        tile,
        key=[0],
        xyxy=[0, 0, 1, 1],
        p=posterior(tile, 0.95, floor=0.001),
        conf=0.95,
    )


def obs(t0: float, t1: float, tiles: list[str]) -> dict:
    """Create a hand-row observation of the supplied tiles."""
    return observation("hand:TL", t0, t1, [slot_dict(t) for t in tiles])


def test_solver_recovers_draws_from_perfect_states() -> None:
    # one non-dealer seat, three turns; the other seats have no turns and no evidence
    haipai = [
        "1m",
        "2m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "9s",
        "1z",
        "1z",
        "3z",
        "6z",
    ]
    draws = ["9m", "2z", "3z"]
    discards = ["6z", "9m", "1z"]  # turn 2 is a tsumogiri
    turns = {s: [] for s in rules.SEATS}
    hand = list(haipai)
    states = []
    for j, (dr, di) in enumerate(zip(draws, discards, strict=False)):
        turns["S"].append(SeatTurn(j, "draw", di, 100 * j + 10, 100 * j + 20))
        hand.append(dr)
        hand.remove(di)
        states.append(list(hand))
    model = HandModel("E", turns, ["5z"])
    model.hand_ev.append(
        HandEvidence(
            "S", -1, after_draw=False, e=tile_counts(haipai), weight=1.0, t0=0, t1=5
        )
    )
    for j, st in enumerate(states):
        model.hand_ev.append(
            HandEvidence(
                "S",
                j,
                after_draw=False,
                e=tile_counts(st),
                weight=1.0,
                t0=100 * j + 30,
                t1=100 * j + 90,
            )
        )
    sol = model.solve(time_limit=10)
    model.certify(sol)
    assert sol.status in ("optimal", "feasible")
    assert sorted(sol.haipai["S"]) == sorted(haipai)
    assert [sol.draws[("S", j)] for j in range(3)] == draws
    assert all(sol.certificates["draw", "S", j].state == "resolved" for j in range(3))
    assert sol.certificates["haipai", "S", -1].state == "resolved"


def test_state_mapping_and_direct_draw_evidence() -> None:
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 110, 120)]
    assert state_of(st, 0, 5) == -1
    assert state_of(st, 25, 60) == 0
    assert state_of(st, 15, 60) is None  # straddles the discard of turn 0
    # ends before the discard of turn 1 (its disturbed window started at 110)
    assert state_of(st, 25, 115) == 0
    assert state_of(st, 130, 200) == 1
    o13 = obs(
        25,
        60,
        ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"],
    )
    o14 = obs(
        100,
        108,
        [
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "7s",
            "8s",
            "9s",
            "1z",
            "1z",
            "3z",
            "9m",
            "2z",
        ],
    )
    hev, dev, habit = hand_evidence(
        "S", st, [o13, o14], {}, role=HandRole(dealer=False)
    )
    assert [(e.j, e.after_draw) for e in hev] == [(0, False), (0, True)]
    assert len(dev) == 1
    assert dev[0].j == 1
    assert TILES[int(np.argmax(dev[0].p))] == "2z"
    assert dev[0].weight > 1.0
    assert habit == (0, 1)


def test_oversized_row_cannot_draw_before_the_preceding_players_discard() -> None:
    """Keep extra detections before a legal draw as resting-state evidence."""
    tiles = [
        "1m",
        "2m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "9s",
        "1z",
        "1z",
        "3z",
        "9m",
    ]
    turns = [
        SeatTurn(0, "draw", "6z", 10, 20),
        SeatTurn(1, "draw", "9m", 110, 120, t_draw_min=100),
    ]
    early = obs(80, 85, [*tiles, "2z"])
    oversized = obs(90, 95, [*tiles, "2z", "3z"])
    timely = obs(105, 108, [*tiles, "2z"])
    hev, dev, _ = hand_evidence(
        "S", turns, [early, oversized, timely], {}, role=HandRole(dealer=False)
    )
    assert len(hev) == 2  # do not admit impossible rows the existing mapper excluded
    assert hev[0].j == 0
    assert not hev[0].after_draw
    assert hev[0].subset
    assert hev[1].j == 0
    assert hev[1].after_draw
    assert not hev[1].subset
    assert dev
    assert all(d.j == 1 for d in dev)
    assert sum(d.weight for d in dev) < 2.0  # only the timely row names an end


def test_drawn_end_needs_the_rest_to_match_and_learns_the_habit() -> None:
    thirteen = [
        "1m",
        "2m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "9s",
        "1z",
        "1z",
        "3z",
        "9m",
    ]
    ref = [obs(0, 1, thirteen)["slots"][i]["p"] for i in range(13)]
    # the player did not sort: the rest matches as a multiset, not by position
    row = obs(0, 1, ["2z", *thirteen[::-1]])["slots"]
    assert drawn_end(row, ref, None) == [("L", 2.0)]
    # a reference row of another state (four tiles differ) names nothing: the habit
    # weighs the ends
    other = obs(0, 1, ["5z", "6z", "7z", "2p", *thirteen[4:], "2z"])["slots"]
    weights = dict(drawn_end(other, ref, (0, 9)))
    assert weights["R"] > 5 * weights["L"]


def test_repair_re_reads_a_discard_but_not_one_a_call_took() -> None:
    """A discard the hand cannot hold is re-read by repair, at the cost of its
    posterior; a discard a call took was read by the meld camera too, so repair leaves
    it and the hand stays illegal (a contradiction elsewhere).
    """
    haipai = [
        "1m",
        "2m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "9s",
        "1z",
        "1z",
        "3z",
        "6z",
    ]
    for taken in (False, True):
        p = np.full(len(TILES), 0.001)
        p[TI["5z"]], p[TI["6z"]] = 0.8, 0.15
        turns = {s: [] for s in rules.SEATS}
        turns["S"].append(
            SeatTurn(0, "draw", "5z", 10, 20, discard_p=p / p.sum(), taken=taken)
        )
        model = HandModel("E", turns, ["2z"])
        model.facts.haipai["S"] = list(haipai)
        model.facts.draws[("S", 0)] = "9m"
        model.repair = True
        sol = model.solve(time_limit=10)
        if taken:
            assert not sol.ok
        else:
            assert sol.ok
            assert sol.discards[("S", 0)] in [*haipai, "9m"]


def test_a_forbidden_hand_gives_the_next_best_reconstruction() -> None:
    """The second VOD's hand 2: the views split one tile between 5s and 8s; the cheaper
    reading scored below the site, so the winner's hand is forbidden and the next best,
    as the reveal read it, is found.
    """
    haipai = [
        "1m",
        "2m",
        "3m",
        "6p",
        "7p",
        "4s",
        "5s",
        "6s",
        "6s",
        "7s",
        "8s",
        "8s",
        "9m",
    ]
    turns = {s: [] for s in rules.SEATS}
    turns["S"].append(SeatTurn(0, "draw", "9m", 10, 20))
    model = HandModel("E", turns, ["2z"])
    model.facts.haipai["S"] = list(haipai)
    # the draw: 8s read a little better than 5s
    ambiguous = kinds({"8s": 0.6, "5s": 0.4})
    model.draw_ev.append(DrawEvidence("S", 0, ambiguous, 1.0))
    best = model.solve(time_limit=10)
    assert best.draws[("S", 0)] == "8s"
    model.forbidden_hands.append(("S", 0, sorted(best.hands[("S", 0)])))
    nxt = model.solve(time_limit=10)
    assert nxt.ok
    assert nxt.draws[("S", 0)] == "5s"
    assert nxt.objective >= best.objective


def test_a_bound_hand_holds_through_the_solve() -> None:
    """Week 11 hand 3: the hand the site's score required must survive the solve that
    follows it (a time-limited solve returned another of equal cost). A bound state
    fixes the draws that bring it.
    """
    haipai = [
        "1m",
        "2m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "5s",
        "6s",
        "6s",
        "8s",
        "9m",
    ]
    turns = {s: [] for s in rules.SEATS}
    turns["S"].append(SeatTurn(0, "draw", "9m", 10, 20))
    model = HandModel("E", turns, ["2z"])
    model.facts.haipai["S"] = list(haipai)
    ambiguous = kinds({"8s": 0.6, "5s": 0.4})

    model.draw_ev.append(DrawEvidence("S", 0, ambiguous, 1.0))
    want = sorted([t for t in haipai if t != "9m"] + ["5s"])
    model.bound_hands = [("S", 0, want)]
    sol = model.solve(time_limit=10)
    assert sol.ok
    assert sol.draws[("S", 0)] == "5s"
    assert sorted(sol.hands[("S", 0)]) == want
