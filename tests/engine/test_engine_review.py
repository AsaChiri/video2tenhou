# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for engine evidence, reconstruction and export fixes."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from tests.engine import factories
from tests.recognition import models_stub
from video2tenhou.engine import dense, rules
from video2tenhou.engine.assemble import call_string, kyoku_from_decode
from video2tenhou.engine.confidence import (
    FIXED,
    Certificate,
    confidence_state,
    low_margin,
)
from video2tenhou.engine.decode import DecodeOptions, decode_hand
from video2tenhou.engine.dense import DenseContext, still_runs
from video2tenhou.engine.events import DeadWall, hand_of
from video2tenhou.engine.indicators import reconcile_kans
from video2tenhou.engine.melds import Call, fragments, read_views, track_melds
from video2tenhou.engine.ponds import PondSlot
from video2tenhou.engine.questions import (
    Report,
    changed_discard,
    site_corrected,
    uncertain_discards,
    uncertain_tiles,
)
from video2tenhou.engine.reconstruct import (
    apply_choices,
    apply_hand_facts,
    kan_indicators,
    seat_turns_of,
)
from video2tenhou.engine.review import (
    ReviewContext,
    confidence_rows,
    draws_to_reread,
    facts_for_hand,
    unseen_draw,
)
from video2tenhou.engine.score_reconcile import concealed_size, scoring_melds
from video2tenhou.engine.scoring import payment
from video2tenhou.engine.solver import (
    TI,
    TILES,
    DrawEvidence,
    HandModel,
    HandRole,
    SeatTurn,
    Solution,
    hand_evidence,
    open_turn,
)
from video2tenhou.engine.turns import Turn, assign_calls, merge
from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES
from video2tenhou.record import HandResult
from video2tenhou.tenhou6 import Agari


def test_confidence_boundary_requests_more_evidence() -> None:
    assert low_margin(0.5)
    assert low_margin(0.50000000000001)
    assert low_margin(0.0)
    assert not low_margin(0.5001)
    assert not low_margin(None)
    assert not low_margin(float("inf"))


def test_covered_ambiguity_has_no_duplicate_or_answered_questions() -> None:
    def draw(j: int, **changes: object) -> dict:
        return {
            "field": "draw",
            "seat": "W",
            "turn": j,
            "value": "4m",
            "margin": 0.0,
            "alternative_gap": 0.0,
            "human": False,
            "lost": False,
            "evidence": [{"region": "hand:TL", "t": 100 + j}],
            **changes,
        }

    rows = [
        draw(4),
        draw(5, value="7m"),
        draw(6, human=True),
        draw(7, lost=True),
        draw(8),
        draw(9, margin=2.0),
        {
            "field": "haipai",
            "seat": "E",
            "turn": -1,
            "value": ["1m"] * 14,
            "margin": 0.5,
            "alternative_gap": 0.5,
            "human": False,
            "lost": False,
            "evidence": [],
        },
    ]
    item = uncertain_tiles(rows, [{"kind": "draw", "seat": "W", "j": 8}])
    assert item is not None
    assert item["kind"] == "uncertain_tiles"
    assert item["count"] == 3
    assert [(c["field"], c["seat"], c["j"]) for c in item["choices"]] == [
        ("draw", "W", 4),
        ("draw", "W", 5),
        ("haipai", "E", -1),
    ]
    assert uncertain_tiles([draw(6, human=True), draw(7, lost=True)], []) is None


def test_cant_tell_starting_hand_does_not_answer_a_draw() -> None:
    entry = {
        "game": 0,
        "kyoku": 0,
        "honba": 0,
        "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"},
    }
    facts = facts_for_hand(
        [
            {
                "game": 0,
                "kyoku": 0,
                "honba": 0,
                "corner": "TR",
                "kind": "lost",
                "field": "haipai",
            }
        ],
        entry,
    )
    assert facts["lost_haipai"] == ["S"]
    assert facts["lost"] == []
    model = HandModel("E", {s: [] for s in rules.SEATS}, [])
    model.turns["S"] = [SeatTurn(0, "draw", "1m", 10, 20)]
    sol = Solution(
        "optimal",
        0,
        {"S": ["1m"] * 13},
        {("S", 0): "2m"},
        {},
        certificates={
            ("draw", "S", 0): Certificate(0),
            ("haipai", "S", -1): Certificate(0),
        },
    )
    rows = confidence_rows(
        model,
        sol,
        context=ReviewContext(
            turns=[], calls=[], inds=[], entry=entry, facts=facts, lost_keys=set(), t0=0
        ),
    )
    haipai = next(r for r in rows if r["field"] == "haipai")
    assert haipai["lost"]
    assert not haipai["human"]
    assert uncertain_tiles([haipai], []) is None
    # The first draw is still unreviewed and receives its own unseen question;
    # the starting-hand answer never enters the set of answered draw keys.
    assert facts["lost"] == []
    assert not next(r for r in rows if r["field"] == "draw")["human"]


def test_solver_changed_discard_needs_specific_review_unless_human_fixed() -> None:
    item = changed_discard(("S", 17), 3097.5, "1m", "2m", [])
    assert item is not None
    assert item["kind"] == "discard"
    assert item["observed"] == "1m"
    assert item["tile"] == "2m"
    assert changed_discard(("S", 17), 3097.5, "2m", "2m", []) is None
    assert (
        changed_discard(
            ("S", 17), 3097.5, "1m", "2m", [{"seat": "S", "t": 3097.5, "tile": "2m"}]
        )
        is None
    )
    assert (
        changed_discard(
            ("S", 17), 3097.5, "1m", "2m", [{"seat": "S", "t": 3097.5, "tile": "1m"}]
        )
        is not None
    )


def test_search_uncertainty_does_not_alone_request_more_video() -> None:
    sol = Solution(
        "optimal",
        0,
        {},
        {("S", j): "2m" for j in range(4)},
        {},
        certificates={
            ("draw", "S", j): Certificate(0.0, gap)
            for j, gap in enumerate([5.0, 0.5, None, float("inf")])
        },
    )
    assert draws_to_reread(sol) == [("S", 1)]  # only the close candidate
    rows = [
        {
            "field": "draw",
            "seat": "S",
            "turn": 0,
            "value": "2m",
            "margin": sol.certificates["draw", "S", 0].margin,
            "alternative_gap": 5.0,
            "human": False,
            "lost": False,
            "evidence": [],
        }
    ]
    assert uncertain_tiles(rows, []) is None
    assert (
        confidence_state(rows[0]["margin"], rows[0]["alternative_gap"])
        == "unresolvable"
    )


def test_serialized_confidence_keeps_threshold_precision() -> None:
    entry = {"corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}}
    model = HandModel("E", {s: [] for s in rules.SEATS}, [])
    model.turns["S"] = [
        SeatTurn(0, "draw", "1m", 10, 20),
        SeatTurn(1, "draw", "2m", 30, 40),
    ]
    sol = Solution(
        "optimal",
        0,
        {},
        {("S", 0): "3m", ("S", 1): "4m"},
        {},
        certificates={
            ("draw", "S", 0): Certificate(0.5004, 0.5004),
            ("draw", "S", 1): Certificate(0.5, 0.5),
        },
    )
    rows = confidence_rows(
        model,
        sol,
        context=ReviewContext(
            turns=[], calls=[], inds=[], entry=entry, facts={}, lost_keys=set(), t0=0
        ),
    )
    assert rows[0]["margin"] == rows[0]["alternative_gap"] == 0.5004
    # Supply image coverage: these choices belong to the grouped review policy.
    for row in rows:
        row["lost"] = False
    item = uncertain_tiles(rows, [])
    assert item is not None
    assert [c["j"] for c in item["choices"]] == [1]


def pslot(
    i: int,
    tile: str,
    t: float,
    removed: float | None = None,
) -> PondSlot:
    """Create a pond slot with controlled tile support, timing and removal."""
    p = np.full(len(CLASSES), 0.002)
    p[CLASS_INDEX[tile]] = 0.9
    s = PondSlot(
        i,
        0,
        i,
        p,
        t,
        (t - 3, t),
        t + 4,
        3,
        0.0,
    )
    s.t_removed = removed
    return s


def mslot(tile: str, group: int, i: int, sideways: float = 0.0) -> dict:
    """Create a meld tile observation with controlled group and orientation."""
    p = np.full(len(CLASSES), 0.002)
    p[CLASS_INDEX[tile]] = 0.8
    p /= p.sum()
    return {
        "key": [group, i],
        "tile": tile,
        "conf": 0.9,
        "seen": 3,
        "sideways": sideways,
        "disagree": False,
        "xyxy": [i * 40, group * 70, i * 40 + 38, group * 70 + 58],
        "p": p.tolist(),
    }


def mobs(
    t0: float,
    t1: float,
    slots: list[dict],
    n_used: int = 3,
    *,
    partial: bool = False,
) -> dict:
    """Create a meld observation interval with explicit support counts."""
    return {
        "region": "meld:TL",
        "t0": t0,
        "t1": t1,
        "n_readings": n_used,
        "n_used": n_used,
        "count": len(slots),
        "quality": 0.9,
        "slots": slots,
        "indicators": [],
        "partial": partial,
    }


# ---------------------------------------------------------------- turns


def test_unknown_source_prefers_the_pond_that_shows_the_removal() -> None:
    # W pons 3p; its turned tile was not read. S (kamicha of W) has no removed 3p, N
    # (shimocha) does: N is the source
    logs = {
        "E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
        "S": [pslot(2, "3p", 20), pslot(3, "4p", 60)],
        "W": [pslot(4, "5s", 30), pslot(5, "6s", 49)],
        "N": [pslot(6, "3p", 40, removed=47), pslot(7, "9m", 52)],
    }
    pon = Call(
        seat="W",
        t_first=48,
        t_window=(44, 48),
        type="pon",
        tiles=["3p", "3p", "3p"],
        conf=0.9,
    )
    merged = merge(logs, [pon], "E")
    turns = merged.turns
    assert pon.source is None  # the caller's call is not changed ...
    (bound,) = merged.calls  # ... the result carries it bound to its source
    assert bound.source == "shimocha"
    assert bound.called_pos == 2
    assert turns[3].call is bound
    assert merged.findings == []
    assert not any(t.virtual for t in turns)
    # after W's call and discard the turn passes to N, who discards; a seat never passes
    # for free
    assert [t.seat for t in turns] == ["E", "S", "W", "N", "W", "N", "E", "S"]


def test_daiminkan_from_shimocha_has_the_turned_tile_last() -> None:
    logs = {
        "E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
        "S": [pslot(2, "3p", 20), pslot(3, "4p", 60)],
        "W": [pslot(4, "5s", 30), pslot(5, "6s", 70)],
        "N": [pslot(6, "7z", 40, removed=47)],
    }
    kan = Call(
        seat="W",
        t_first=48,
        t_window=(44, 48),
        type="kan",
        tiles=["7z", "7z", "7z", "7z"],
        conf=0.9,
    )
    (kan,) = merge(logs, [kan], "E").calls
    assert kan.source == "shimocha"
    assert kan.called_pos == 3


def test_one_call_takes_one_removed_slot() -> None:
    # S's pond lost two 3p; one pon by W: only the removal nearest the call gets it
    seq = {
        "E": [],
        "S": [pslot(0, "3p", 20, removed=27), pslot(1, "3p", 40, removed=60)],
        "W": [],
        "N": [],
    }
    pon = Call(
        seat="W",
        t_first=62,
        t_window=(58, 62),
        type="pon",
        tiles=["3p", "3p", "3p"],
        called_pos=0,
        source="kamicha",
        called_tile="3p",
        conf=0.9,
    )
    callers = assign_calls([pon], seq)
    assert callers["S"] == [None, pon]


# ---------------------------------------------------------------- melds


def test_two_identical_chis_are_two_calls() -> None:
    def g(k: int) -> list:
        return [
            mslot("2m", k, 0, sideways=1.0),
            mslot("3m", k, 1),
            mslot("4m", k, 2),
        ]

    seq = [
        mobs(0, 5, []),
        mobs(10, 14, g(0)),
        mobs(20, 24, g(0)),
        mobs(30, 34, g(0) + g(1)),
        mobs(40, 44, g(0) + g(1)),
    ]
    calls = track_melds("E", read_views(seq))
    assert [c.type for c in calls] == ["chi", "chi"]
    assert [c.t_first for c in calls] == [
        10,
        30,
    ]


def test_a_meld_seen_once_and_then_absent_is_not_real() -> None:
    pon = [mslot("7z", 0, 0), mslot("7z", 0, 1), mslot("7z", 0, 2, sideways=1.0)]
    # seen once at 10, then two full views without it: a misread; the chi seen once in
    # the last view stays
    chi = [mslot("2m", 1, 0, sideways=1.0), mslot("3m", 1, 1), mslot("4m", 1, 2)]
    seq = [
        mobs(0, 5, []),
        mobs(10, 14, pon),
        mobs(20, 24, []),
        mobs(30, 34, []),
        mobs(40, 44, chi),
    ]
    calls = track_melds("E", read_views(seq))
    # kept only as a disputed hypothesis that a taken discard must establish
    assert [(c.type, c.contradicted) for c in calls] == [("pon", True), ("chi", False)]


def test_observation_without_readings_keeps_the_call_window() -> None:
    pon = [mslot("7z", 0, 0), mslot("7z", 0, 1), mslot("7z", 0, 2, sideways=1.0)]
    seq = [
        mobs(0, 5, []),
        mobs(8, 9, [], n_used=0),
        mobs(10, 14, pon),
        mobs(20, 24, pon),
    ]
    calls = track_melds("E", read_views(seq))
    assert calls[0].t_window == (5, 10)


# ---------------------------------------------------------------- solver


def hobs(t0: float, t1: float, tiles: list[str]) -> dict:
    """Create a full hand observation interval with peaked tile posteriors."""

    def sd(tile: str) -> dict:
        p = np.full(len(CLASSES), 0.001)
        p[CLASS_INDEX[tile]] = 0.95
        p /= p.sum()
        return {
            "key": [0],
            "tile": tile,
            "conf": 0.95,
            "seen": 3,
            "sideways": 0.0,
            "disagree": False,
            "xyxy": [0, 0, 1, 1],
            "p": p.tolist(),
        }

    return {
        "region": "hand:TL",
        "t0": t0,
        "t1": t1,
        "n_readings": 3,
        "n_used": 3,
        "count": len(tiles),
        "quality": 0.9,
        "slots": [sd(t) for t in tiles],
        "indicators": [],
    }


def test_hand_row_inside_a_turn_window_counts_only_after_a_draw() -> None:
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 100, 120)]
    # a row ending within 8 s of the discard's first sighting is ambiguous; one ending
    # long before it is not
    assert open_turn(st, 60) is None
    assert open_turn(st, 115) is st[1]
    assert open_turn(st, 110) is None
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
    o13 = hobs(108, 115, thirteen)  # the discard may already have happened: ambiguous
    # one more tile: after the draw, before the discard
    o14 = hobs(108, 115, [*thirteen, "2z"])
    # the 13 tiles are in the hand at some moment of the turn: all of them are in the
    # hand after its draw
    hev, dev, _ = hand_evidence("S", st, [o13], {}, role=HandRole(dealer=False))
    assert [(e.j, e.after_draw, e.subset) for e in hev] == [(0, True, True)]
    assert dev == []
    hev, dev, _ = hand_evidence("S", st, [o14], {}, role=HandRole(dealer=False))
    assert [(e.j, e.after_draw, e.subset) for e in hev] == [(0, True, False)]
    assert len(dev) == 2


def test_ura_indicators_count_against_the_four() -> None:
    turns = {s: [] for s in rules.SEATS}
    turns["S"].append(SeatTurn(0, "draw", "9p", 10, 20))
    hand = [
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
        "9p",
    ]
    with_ura = HandModel("E", turns, ["1m", "1m", "1m"], ura=["1m"])
    with_ura.facts.haipai["S"] = hand
    assert with_ura.solve(time_limit=5).status == "infeasible"
    without = HandModel("E", turns, ["1m", "1m", "1m"])
    without.facts.haipai["S"] = hand
    assert without.solve(time_limit=5).status != "infeasible"


def test_kakan_of_fives_reports_which_five_was_added() -> None:
    turns = {s: [] for s in rules.SEATS}
    turns["S"] = [
        SeatTurn(0, "call", "1z", 10, 20, removed=["5p", "5p"]),
        SeatTurn(1, "kan", "2z", 100, 120, kan="kakan", kan_tile="5p", two_draws=True),
    ]
    model = HandModel("E", turns, ["9s"])
    model.facts.haipai["S"] = [
        "1m",
        "2m",
        "3m",
        "5p",
        "5p",
        "0p",
        "7s",
        "8s",
        "9s",
        "1z",
        "2z",
        "3z",
        "6z",
    ]
    sol = model.solve(time_limit=5)
    assert sol.status != "infeasible"
    assert sol.kan_added[("S", 1)] == "0p"


def test_riichi_ankan_uses_drawn_tile_and_discards_rinshan_draw() -> None:
    turns = {s: [] for s in rules.SEATS}
    turns["S"] = [
        SeatTurn(0, "draw", "1z", 10, 20, riichi=True),
        SeatTurn(1, "kan", "9p", 100, 120, kan="ankan", kan_tile="3m", two_draws=True),
    ]
    model = HandModel("E", turns, ["9s"])
    model.facts.haipai["S"] = [
        "3m",
        "3m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "9s",
        "1z",
        "1z",
        "2z",
        "6z",
    ]
    sol = model.solve(time_limit=5)
    model.certify(sol, timeout=0)
    assert sol.status != "infeasible"
    assert sol.draws[("S", 1)] == "3m"
    assert sol.draws2[("S", 1)] == "9p"
    assert sol.draw_sources[("S", 1)] == "ankan"
    assert sol.certificates["draw", "S", 1] is FIXED


@pytest.mark.parametrize("ambiguous_pond", [False, True])
@pytest.mark.parametrize("budget", [0, 10])
def test_post_riichi_draw_uses_discard_without_a_search_or_question(
    *, ambiguous_pond: bool, budget: int
) -> None:
    turns = {s: [] for s in rules.SEATS}
    p = np.zeros(len(TILES))
    p[TI["1m"]], p[TI["2m"]] = 0.51, 0.49
    turns["S"] = [
        SeatTurn(0, "draw", "1z", 10, 20, riichi=True),
        SeatTurn(1, "draw", "1m", 30, 40, discard_p=p if ambiguous_pond else None),
    ]
    model = HandModel("E", turns, [])
    model.facts.draws[("S", 0)] = "1z"
    if ambiguous_pond:
        # A competing hand-camera reading offsets the pond's alternative
        # penalty (1.53), leaving a genuine .03-cost competing reconstruction.
        draw_p = np.zeros(len(TILES))
        draw_p[TI["2m"]] = 1
        model.draw_ev.append(DrawEvidence("S", 1, draw_p, 1.5))
    sol = model.solve(time_limit=5, workers=1)
    model.certify(sol, timeout=budget, workers=1)
    assert sol.ok
    assert sol.draws[("S", 1)] == sol.discards.get(("S", 1), "1m")
    assert sol.draw_sources[("S", 1)] == "discard"
    assert not unseen_draw(model, sol, "S", 1)
    assert ("S", 1) not in draws_to_reread(sol, model)
    # This draw is the discard: it shares that certificate instead of a search.
    assert sol.certificates["draw", "S", 1] is (
        sol.certificates["discard", "S", 1] if ambiguous_pond else FIXED
    )
    if ambiguous_pond and budget:
        assert sol.certificates["discard", "S", 1].state == "ambiguous"
    entry = {"corner_wind": {"TL": "E", "TR": "N", "BL": "S", "BR": "W"}}
    merged = [
        Turn(0, "S", "draw", pslot(0, "1z", 20), 20),
        Turn(1, "S", "draw", pslot(1, "1m", 40), 40),
    ]
    rows = confidence_rows(
        model,
        sol,
        context=ReviewContext(
            turns=merged,
            calls=[],
            inds=[],
            entry=entry,
            facts={},
            lost_keys=set(),
            t0=0,
        ),
    )
    row = next(
        r for r in rows if r["field"] == "draw" and r["seat"] == "S" and r["turn"] == 1
    )
    assert row["evidence"] == [{"region": "pond:BL", "t": 40}]
    assert uncertain_tiles([row], []) is None
    if ambiguous_pond and budget:
        questions = uncertain_discards(rows, [])
        assert [(q["seat"], q["j"]) for q in questions] == [("S", 1)]


def test_riichi_declaration_winning_draw_and_red_kan_are_not_assumed_discards() -> None:
    turns = {s: [] for s in rules.SEATS}
    turns["S"] = [
        SeatTurn(0, "draw", "1z", 10, 20, riichi=True),
        SeatTurn(1, "kan", "9p", 30, 40, kan="ankan", kan_tile="5m", two_draws=True),
        SeatTurn(2, "draw", None, 50, 60),
    ]
    model = HandModel("E", turns, [], tsumo_winner="S")
    assert model._riichi_draw_sources() == {}


# ---------------------------------------------------------------- assemble


def test_kans_of_fives_hold_the_red_one() -> None:
    assert (
        call_string({"type": "ankan", "tiles": ["5p", "5p", "X", "X"]}) == "522525a25"
    )
    assert (
        call_string(
            {
                "type": "kakan",
                "tiles": ["5s", "5s", "5s", "0s"],
                "called_pos": 0,
                "source": "kamicha",
            },
        )
        == "k53353535"
    )
    assert (
        call_string(
            {
                "type": "kakan",
                "tiles": ["5s", "5s", "5s", "5s"],
                "called_pos": 0,
                "source": "kamicha",
            },
        )
        == "k35533535"
    )
    assert (
        call_string(
            {
                "type": "kan",
                "tiles": ["5m", "5m", "5m", "5m"],
                "called_pos": 0,
                "source": "kamicha",
            },
        )
        == "m15511515"
    )


def test_ura_and_the_dealer_split_reach_the_log() -> None:
    d = {
        "dealer": "E",
        "haipai": {
            "E": [
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
                "9p",
            ],
            "S": [],
            "W": [],
            "N": [],
        },
        "turns": [
            {
                "i": 0,
                "seat": "E",
                "j": 0,
                "kind": "draw",
                "t": 10,
                "riichi": False,
                "draw": None,
                "draw2": None,
                "discard": "9p",
                "tsumogiri": False,
                "margin": None,
                "call": None,
                "own_call": None,
            }
        ],
        "draws": {},
        "dora": ["1s"],
        "ura": ["2s"],
        "result": {"winner": None, "loser": None},
        "score": None,
    }
    entry = {
        "kyoku": 0,
        "honba": 1,
        "sticks": 0,
        "scores": {"E": 25000, "S": 25000, "W": 25000, "N": 25000},
    }
    result = HandResult(0, 1, 0, {"EAST": 0, "SOUTH": 0, "WEST": 0, "NORTH": 0}, "draw")
    k, conf = kyoku_from_decode(d, entry, result)
    assert k.ura == [32]
    assert k.draws[0] == [29]
    assert k.discards[0] == [60]
    assert not any(c["lost"] for c in conf)


# ---------------------------------------------------------------- decode: kans, melds,
# facts


def test_scoring_melds_fill_kans_and_reds() -> None:
    calls = [
        Call(
            seat="E",
            t_first=10,
            t_window=(5, 10),
            type="ankan",
            tiles=["5p", "5p", "X", "X"],
            conf=0.9,
        ),
        Call(
            seat="E",
            t_first=20,
            t_window=(15, 20),
            type="kakan",
            tiles=["3s", "3s", "3s", "3s"],
            called_pos=0,
            source="kamicha",
            called_tile="3s",
            conf=0.9,
        ),
        Call(
            seat="E",
            t_first=30,
            t_window=(25, 30),
            type="pon",
            tiles=["5m", "5m", "5m"],
            called_pos=0,
            source="kamicha",
            called_tile="5m",
            conf=0.9,
        ),
        Call(
            seat="S",
            t_first=30,
            t_window=(25, 30),
            type="pon",
            tiles=["7z", "7z", "7z"],
            called_pos=0,
            source="kamicha",
            called_tile="7z",
            conf=0.9,
        ),
    ]
    assert scoring_melds(calls, "E") == [
        {"type": "ankan", "tiles": ["5p", "5p", "5p", "0p"]},
        {"type": "kakan", "tiles": ["3s", "3s", "3s", "3s"]},
        {"type": "pon", "tiles": ["5m", "5m", "5m"]},
    ]
    assert rules.count_ok([t for m in scoring_melds(calls, "E") for t in m["tiles"]])


def test_solver_kan_choices_reach_the_calls() -> None:
    kan = Call(
        seat="E",
        t_first=50,
        t_window=(42, 50),
        type="ankan",
        tiles=["?", "?", "X", "X"],
        conf=0.3,
    )
    kakan = Call(
        seat="S",
        t_first=80,
        t_window=(72, 80),
        type="kakan",
        tiles=["5p", "5p", "5p", "5p"],
        called_pos=0,
        source="kamicha",
        called_tile="5p",
        conf=0.9,
    )
    turns = [
        Turn(0, "E", "kan", pslot(0, "1z", 55), 55, own_call=kan),
        Turn(1, "S", "kan", pslot(1, "2z", 85), 85, own_call=kakan),
    ]
    model = HandModel("E", {s: [] for s in rules.SEATS}, ["9s"])
    for s in rules.SEATS:
        model.turns[s], _ = seat_turns_of(turns, s, "E")

    solution = Solution(
        "optimal", 0, {}, {}, {}, kans={("E", 0): "3m"}, kan_added={("S", 0): "0p"}
    )
    apply_choices(solution, model, turns, {id(kan)})
    assert kan.tiles == ["3m"] * 4
    assert kakan.tiles == ["5p", "5p", "5p", "0p"]


def test_dealer_first_turn_ankan_has_no_normal_draw() -> None:
    kan = Call(
        seat="E",
        t_first=12,
        t_window=(5, 12),
        type="ankan",
        tiles=["3m", "3m", "X", "X"],
        conf=0.9,
    )
    turns = [
        Turn(0, "E", "kan", pslot(0, "1z", 15), 15, own_call=kan),
        Turn(1, "S", "draw", pslot(1, "2z", 25), 25),
    ]
    st, _ = seat_turns_of(turns, "E", "E")
    assert st[0].kind == "first"
    assert st[0].kan == "ankan"
    assert st[0].two_draws
    model = HandModel("E", {"E": st, "S": [], "W": [], "N": []}, ["9s"])
    program = model.build()
    assert ("E", 0) not in program.draws
    assert ("E", 0) in program.rinshan


def test_a_kan_after_the_winners_last_discard_is_its_last_turn() -> None:
    kan = Call(
        seat="S",
        t_first=100,
        t_window=(92, 100),
        type="ankan",
        tiles=["3m", "3m", "X", "X"],
        conf=0.9,
    )
    turns = [
        Turn(0, "E", "draw", pslot(0, "1z", 15), 15),
        Turn(1, "S", "draw", pslot(1, "2z", 25), 25),
        Turn(2, "S", "kan", None, 100, own_call=kan),
    ]
    st, _melds_before = seat_turns_of(turns, "S", "E")
    assert [x.kind for x in st] == ["draw", "kan"]
    assert not st[1].two_draws
    assert st[1].discard is None
    model = HandModel(
        "E", {"E": [], "S": st, "W": [], "N": []}, ["9s"], tsumo_winner="S"
    )
    # the normal draw of the kan turn, then the winning rinshan draw
    assert model.draw_turns("S") == [0, 1, 2]
    model.facts.haipai["S"] = [
        "3m",
        "3m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "9s",
        "1z",
        "1z",
        "2z",
        "6z",
    ]
    sol = model.solve(time_limit=5)
    assert sol.status != "infeasible"
    assert len(sol.hands[("S", 2)]) == 11


def test_indicator_after_the_last_discard_is_the_tsumo_winners_kan() -> None:
    logs = {"E": [pslot(0, "1z", 15)], "S": [pslot(1, "2z", 25)], "W": [], "N": []}
    inds = [{"tile": "1s", "t_first": 0}, {"tile": "4p", "t_first": 40}]
    entry = {
        "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"},
        "corner_site": {"TL": "EAST", "TR": "SOUTH", "BR": "WEST", "BL": "NORTH"},
    }
    out = reconcile_kans(
        inds,
        [],
        logs=logs,
        obs={},
        entry=entry,
        t0=0,
        diagnostics=[],
    )
    assert out.calls == []
    assert out.unplaced == [inds[1]]  # no discard follows: no turn holds the kan
    out = reconcile_kans(
        inds,
        [],
        logs=logs,
        obs={},
        entry=entry,
        t0=0,
        diagnostics=[],
        tsumo_winner="S",
    )
    (kan,) = out.calls
    assert kan.seat == "S"
    assert kan.type == "ankan"
    assert kan.anchor == "indicator"
    assert kan.t_window[0] > 25
    assert out.revealed == {id(kan)}


def test_camera_kan_explains_the_indicator_inside_its_window() -> None:
    # the meld camera first shows the ankan 60 s after the indicator, but its window
    # (last view without it) opens before
    kan = Call(
        seat="S",
        t_first=100,
        t_window=(30, 100),
        type="ankan",
        tiles=["3m", "3m", "X", "X"],
        conf=0.9,
    )
    logs = {
        "E": [pslot(0, "1z", 15), pslot(2, "3z", 50)],
        "S": [pslot(1, "2z", 25), pslot(3, "4z", 45)],
        "W": [],
        "N": [],
    }
    inds = [{"tile": "1s", "t_first": 0}, {"tile": "4p", "t_first": 40}]
    entry = {
        "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"},
        "corner_site": {"TL": "EAST", "TR": "SOUTH", "BR": "WEST", "BL": "NORTH"},
    }
    out = reconcile_kans(
        inds,
        [kan],
        logs=logs,
        obs={},
        entry=entry,
        t0=0,
        diagnostics=[],
    )
    assert out.calls == [kan]
    assert out.revealed == {id(kan)}
    assert kan.t_window == (32, 40)


def test_an_anchored_kan_stands_without_an_indicator() -> None:
    """A kan the call anchor established stands; an unexplained indicator adds a kan.

    The second VOD's hand 3: the dora lies at the crop's edge and the ankan's
    indicator outside every region.
    """
    logs = {"E": [pslot(0, "1z", 15)], "S": [], "W": [], "N": []}
    entry = {
        "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"},
        "corner_site": {"TL": "EAST", "TR": "SOUTH", "BR": "WEST", "BL": "NORTH"},
    }
    kan = Call(
        seat="S",
        t_first=30,
        t_window=(20, 30),
        type="ankan",
        tiles=["1s", "1s", "X", "X"],
        conf=0.9,
        seen=5,
        anchor="kan",
    )
    out = reconcile_kans(
        [{"tile": "4s", "t_first": 0}],
        [kan],
        logs=logs,
        obs={},
        entry=entry,
        t0=0,
        diagnostics=[],
    )
    assert out.calls == [kan]
    assert not out.revealed
    assert not out.unplaced


def test_facts_for_hand_passes_the_new_kinds() -> None:
    entry = {
        "game": 0,
        "kyoku": 1,
        "honba": 0,
        "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"},
        "corner_site": {"TL": "SOUTH", "TR": "WEST", "BR": "NORTH", "BL": "EAST"},
    }
    facts = [
        {
            "game": 0,
            "kyoku": 1,
            "honba": 0,
            "kind": "draw",
            "corner": "TR",
            "j": 7,
            "t": None,
            "tile": "3m",
        },
        {
            "game": 0,
            "kyoku": 1,
            "honba": 0,
            "kind": "lost",
            "corner": "TR",
            "j": 2,
            "t": 300.0,
        },
        {
            "game": 0,
            "kyoku": 1,
            "honba": 0,
            "kind": "riichi_turn",
            "corner": "TL",
            "t": 250.0,
        },
        {"game": 0, "kyoku": 1, "honba": 0, "kind": "kan_time", "t": 260.0},
        {
            "game": 0,
            "kyoku": 1,
            "honba": 0,
            "kind": "meld_remove",
            "corner": "BL",
            "t": 270.0,
            "type": "kakan",
        },
        {
            "game": 0,
            "kyoku": 2,
            "honba": 0,
            "kind": "draw",
            "corner": "TR",
            "j": 1,
            "tile": "9m",
        },
    ]
    out = facts_for_hand(facts, entry)
    assert out["draw"] == [{"seat": "S", "t": None, "tile": "3m", "j": 7}]
    assert out["lost"] == [{"seat": "S", "t": 300.0, "j": 2}]
    assert out["riichi_turn"] == [{"seat": "E", "t": 250.0}]
    assert out["kan_time"] == [{"seat": None, "t": 260.0}]
    assert out["meld_remove"] == [{"seat": "N", "t": 270.0, "type": "kakan"}]


# ---------------------------------------------------------------- decode_hand end to
# end (synthetic ponds)


def pond_obs(
    region: str,
    span: tuple[float, float],
    tiles: list[str],
    sideways: Iterable[int] = (),
    indicators: Iterable[str] = (),
) -> dict:
    """Create a pond interval with explicit sideways tiles and indicators."""
    t0, t1 = span

    def sd(tile: str, r: int, c: int, *, side: bool) -> dict:
        p = np.full(len(CLASSES), 0.002)
        p[CLASS_INDEX[tile]] = 0.9
        p /= p.sum()
        return {
            "key": [r, c],
            "tile": tile,
            "conf": 0.9,
            "seen": 3,
            "sideways": 1.0 if side else 0.0,
            "disagree": False,
            "xyxy": [c * 40, r * 60, c * 40 + 38, r * 60 + 58],
            "p": p.tolist(),
        }

    slots = [sd(t, i // 6, i % 6, side=i in sideways) for i, t in enumerate(tiles)]
    ind = [{**sd(t, 0, 0, side=False), "xyxy": [400, 10, 438, 68]} for t in indicators]
    return {
        "region": region,
        "t0": t0,
        "t1": t1,
        "n_readings": 3,
        "n_used": 3,
        "count": len(slots),
        "quality": 0.9,
        "slots": slots,
        "indicators": ind,
    }


def synthetic_hand(
    riichi_seat: str | None = None,
) -> tuple[dict, dict[str, list[dict]], HandResult]:
    """Create four rotating ponds with one dora indicator and no calls.

    Four ponds, three discards each in rotation from 20 s on, one dora indicator; no
    hands or melds seen.
    """
    corners = {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}
    discards = {
        "E": ["1z", "2z", "3z"],
        "S": ["4z", "5z", "6z"],
        "W": ["1m", "2m", "3m"],
        "N": ["9p", "8p", "7p"],
    }
    obs = {}
    for corner, seat in corners.items():
        k = "ESWN".index(seat)
        seq = [
            pond_obs(
                f"pond:{corner}",
                (0, 5),
                [],
                indicators=["1s"] if corner == "TL" else [],
            )
        ]
        for n in range(1, 4):
            t = 20 + 40 * (n - 1) + 10 * k + 2
            seq.append(
                pond_obs(
                    f"pond:{corner}",
                    (t, t + 6),
                    discards[seat][:n],
                    indicators=["1s"] if corner == "TL" else [],
                )
            )
        seq.append(
            pond_obs(
                f"pond:{corner}",
                (150, 155),
                discards[seat],
                indicators=["1s"] if corner == "TL" else [],
            )
        )
        seq.append(pond_obs(f"pond:{corner}", (170, 175), []))
        obs[f"pond:{corner}"] = seq
    entry = {
        "hand": 0,
        "game": 0,
        "kyoku": 0,
        "honba": 0,
        "sticks": 0,
        "t_start": 0,
        "t_end": 180,
        "corner_wind": corners,
        "corner_site": {
            c: {"E": "EAST", "S": "SOUTH", "W": "WEST", "N": "NORTH"}[w]
            for c, w in corners.items()
        },
        "scores": dict.fromkeys("ESWN", 25000),
    }
    result = HandResult(
        0,
        0,
        0,
        {"EAST": 0, "SOUTH": 0, "WEST": 0, "NORTH": 0},
        "draw",
        riichi=[{"E": "EAST", "S": "SOUTH", "W": "WEST", "N": "NORTH"}[riichi_seat]]
        if riichi_seat
        else [],
    )
    return entry, obs, result


def test_decode_hand_returns_ura_and_places_a_riichi_turn_fact() -> None:
    entry, obs, result = synthetic_hand(riichi_seat="S")
    d = decode_hand(
        entry, obs, result, {"ura": ["2s"]}, options=DecodeOptions(time_limit=5)
    )
    assert d["ura"] == ["2s"]
    assert d["dora"] == ["1s"]
    # no turned tile: the last discard of S is the guess and the reviewer is asked
    assert [t["seat"] for t in d["turns"]] == list("ESWN") * 3
    assert [t["i"] for t in d["turns"] if t["riichi"]] == [9]
    assert any(it["kind"] == "riichi" for it in d["items"])
    # the reviewer names S's second discard (at 72 s): that turn alone is the riichi,
    # nothing is asked
    t_second = next(t["t"] for t in d["turns"] if t["seat"] == "S" and t["j"] == 1)
    d2 = decode_hand(
        entry,
        obs,
        result,
        {"riichi_turn": [{"seat": "S", "t": t_second + 1.0}]},
        options=DecodeOptions(time_limit=5),
    )
    assert [t["i"] for t in d2["turns"] if t["riichi"]] == [5]
    assert not any(it["kind"] == "riichi" for it in d2["items"])


def test_decode_hand_places_a_named_indicator_with_a_kan_time_fact() -> None:
    entry, obs, result = synthetic_hand()
    # the reviewer names a second indicator no frame shows: its kan has no time and is
    # asked for
    d = decode_hand(
        entry, obs, result, {"dora": ["1s", "4p"]}, options=DecodeOptions(time_limit=5)
    )
    assert d["dora"] == ["1s", "4p"]
    assert [it["tile"] for it in d["items"] if it["kind"] == "kan"] == ["4p"]
    assert d["stats"]["calls"] == 0
    # the reviewer clicks W's second discard: a kan by W just before it, its tile chosen
    # by the solver and written
    t_w = next(t["t"] for t in d["turns"] if t["seat"] == "W" and t["j"] == 1)
    d2 = decode_hand(
        entry,
        obs,
        result,
        {"dora": ["1s", "4p"], "kan_time": [{"seat": None, "t": t_w}]},
        options=DecodeOptions(time_limit=5),
    )
    assert not any(it["kind"] == "kan" for it in d2["items"])
    assert d2["ignored_facts"] == []
    kan = next(t for t in d2["turns"] if t["seat"] == "W" and t["j"] == 1)
    assert kan["kind"] == "kan"
    assert kan["own_call"]["type"] == "ankan"
    assert "?" not in kan["own_call"]["tiles"]
    assert d2["calls"][0]["tiles"] == kan["own_call"]["tiles"]
    assert kan["draw2"] is not None


def test_concealed_size_counts_melds_the_way_the_hand_does() -> None:
    """13 tiles, three out per meld set; the winning tile is never in the count.

    A kan's fourth tile is the one the kan adds, and a kakan is the pon it grew
    from, not a second set.
    """
    assert concealed_size([]) == 13
    assert concealed_size([{"type": "pon", "tiles": ["1m"] * 3}]) == 10
    assert (
        concealed_size(
            [
                {"type": "chi", "tiles": ["1m", "2m", "3m"]},
                {"type": "pon", "tiles": ["5p"] * 3},
            ]
        )
        == 7
    )
    assert concealed_size([{"type": "kan", "tiles": ["9s"] * 4}]) == 10
    assert concealed_size([{"type": "ankan", "tiles": ["5z"] * 4}]) == 10
    # a kakan is the pon it grew from: the two entries are one set
    assert (
        concealed_size(
            [
                {"type": "pon", "tiles": ["2p"] * 3},
                {"type": "kakan", "tiles": ["2p"] * 4},
            ]
        )
        == 10
    )
    # a tsumo whose winning tile was never named leaves the draw in the list
    assert concealed_size([], drawn_left_in=True) == 14
    assert (
        concealed_size([{"type": "pon", "tiles": ["1m"] * 3}], drawn_left_in=True) == 11
    )


def test_a_kan_whose_indicator_no_view_shows_is_asked_and_the_log_keeps_a_guess() -> (
    None
):
    ankan = Call(
        seat="N",
        t_first=200.0,
        t_window=(195.0, 200.0),
        type="ankan",
        tiles=["1s", "1s", "X", "X"],
        conf=0.5,
    )
    wall = DeadWall([{"tile": "4s", "t_first": 30.0, "region": "pond:TL"}], [], set())
    # solved after the kan: the hand holds three 1s and a 2s
    sol = Solution("unsolved", 0, {"E": ["1s"] * 3 + ["2s"]}, {}, {})
    seq = factories.sequence(live_calls=[ankan])
    report = Report()
    dora, inds = kan_indicators(factories.hand(), seq, wall, sol, 300.0, report)
    assert len(dora) == 2
    assert dora[0] == "4s"
    guess = dora[1]
    assert guess != "1s"
    assert inds[-1]["lost"]
    assert inds[-1]["tile"] == guess
    assert wall.dora == ["4s"]  # the observed row is unchanged
    (item,) = report.items
    assert item["kind"] == "dora"
    assert item["tiles"] == ["4s"]
    assert item["guess"] == [guess]
    assert item["t"] == 200.0
    # the reviewer cannot tell: the guess stays, nothing is asked
    report = Report()
    lost = factories.hand(facts={"lost_dora": True})
    dora, inds = kan_indicators(lost, seq, wall, sol, 300.0, report)
    assert len(dora) == 2
    assert not report.items
    assert inds[-1]["human"]


def test_the_first_indicator_unseen_is_asked_without_kans() -> None:
    """With no kan and no indicator seen, the one indicator is the question."""
    wall = DeadWall([], [], set())
    sol = Solution("unsolved", 0, {}, {}, {})
    report = Report()
    dora, _ = kan_indicators(
        factories.hand(), factories.sequence(), wall, sol, 300.0, report
    )
    (item,) = report.items
    assert (item["tiles"], item["guess"], item["t"]) == ([], dora, 300.0)
    assert "kan" not in item["text"]


def test_a_lost_dora_fact_reaches_the_decoder() -> None:
    entry = {
        "game": 0,
        "kyoku": 2,
        "honba": 0,
        "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"},
    }
    f = facts_for_hand(
        [
            {
                "game": 0,
                "kyoku": 2,
                "honba": 0,
                "kind": "lost",
                "field": "dora",
                "t": 200.0,
            }
        ],
        entry,
    )
    assert f["lost_dora"]
    assert not f["lost"]


def test_a_draw_nothing_covers_is_unseen() -> None:
    sol = Solution(
        "optimal",
        0,
        {},
        {},
        {},
        certificates={
            ("draw", "E", 1): Certificate(0.1),
            ("draw", "E", 2): Certificate(3.0),
            ("draw", "E", 3): Certificate(0.1, 0.1),
        },
    )
    model = HandModel("E", {s: [] for s in rules.SEATS}, [])
    assert unseen_draw(model, sol, "E", 1)  # unresolvable
    assert not unseen_draw(model, sol, "E", 2)  # pinned by certification
    assert unseen_draw(model, sol, "E", 3)  # ambiguous
    assert unseen_draw(model, sol, "E", 4)  # never certified
    model.draw_ev = [DrawEvidence("E", 1, np.zeros(37), 1)]
    assert not unseen_draw(model, sol, "E", 1)


def test_a_read_floor_row_showing_the_whole_hand_between_turns_is_the_state() -> None:
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 100, 120)]
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
    o = {**hobs(40, 60, thirteen), "partial": True}
    hev, _, _ = hand_evidence("S", st, [o], {}, role=HandRole(dealer=False))
    assert [(e.j, e.after_draw, e.subset, e.hidden) for e in hev] == [
        (0, False, False, 0)
    ]


def test_a_calm_row_one_tile_short_between_turns_holds_all_but_one() -> None:
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 100, 120)]
    twelve = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z"]
    hev, _, _ = hand_evidence(
        "S", st, [hobs(40, 60, twelve)], {}, role=HandRole(dealer=False)
    )
    assert [(e.j, e.subset, e.hidden) for e in hev] == [(0, False, 1)]
    # a read-floor view that short is a sub-multiset only (an arm may hide more than the
    # one)
    hev, _, _ = hand_evidence(
        "S",
        st,
        [{**hobs(40, 60, twelve), "partial": True}],
        {},
        role=HandRole(dealer=False),
    )
    assert [(e.subset, e.hidden) for e in hev] == [(True, 0)]
    # and it does not cover the draw by itself: the hidden tile may be the drawn one

    model = HandModel("E", {s: [] for s in rules.SEATS}, [])
    model.hand_ev = hand_evidence(
        "S", st, [hobs(40, 60, twelve)], {}, role=HandRole(dealer=False)
    )[0]
    assert unseen_draw(
        model,
        Solution(
            "optimal",
            0,
            {},
            {},
            {},
            certificates={("draw", "S", 1): Certificate(0.1)},
        ),
        "S",
        1,
    )


def test_a_misread_run_of_three_is_a_fragment_the_discard_can_anchor() -> None:
    # 4p 3p 0p laid, the 4p read 2p (no legal meld as read): the surest two of the run
    # make the fragment

    def g() -> list:
        return [mslot("2p", 0, 0), mslot("3p", 0, 1), mslot("0p", 0, 2)]

    seq = [mobs(0, 5, []), mobs(10, 14, g()), mobs(20, 24, g())]
    views = read_views(seq)
    (f,) = fragments("E", views, track_melds("E", views))
    assert f.t_first == 10
    assert len(f.ps) == 3
    assert sorted(f.tiles) in (["2p", "3p"], ["3p", "5p"])


# ---------------------------------------------------------------- dense still runs


def _dframe(t: float, tiles: list[str], flicker: tuple[int, str] | None = None) -> dict:
    def box(x: int, tile: str) -> dict:
        p = np.full(len(CLASSES), 0.001)
        p[CLASS_INDEX[tile]] = 0.95
        return {
            "role": "tile",
            "xyxy": [x * 40, 0, x * 40 + 38, 58],
            "p": (p / p.sum()).tolist(),
            "conf": 0.95,
        }

    tiles = list(tiles)
    if flicker is not None:
        tiles[flicker[0]] = flicker[1]
    return {"t": t, "boxes": [box(x, tl) for x, tl in enumerate(tiles)]}


def test_still_runs_are_the_frames_whose_reading_holds() -> None:
    row = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "9m"]
    # one flicker is still a run
    still = [
        _dframe(10 + 0.2 * k, row, flicker=(4, "6p") if k == 2 else None)
        for k in range(6)
    ]
    moving = [_dframe(11.2 + 0.2 * k, row[: k % 5 + 8]) for k in range(4)]  # no run
    drawn = [_dframe(12.0 + 0.2 * k, [*row, "2z"]) for k in range(5)]
    frames = still + moving + drawn
    runs = still_runs(frames, "hand:TL")
    assert [(r["count"], r["n_used"], r["t0"]) for r in runs] == [
        (13, 6, 10.0),
        (14, 5, 12.0),
    ]


def test_dense_draws_read_the_hand_before_and_after_the_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    before = [
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
    after = [
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
        "2z",
    ]
    # at rest before the draw, then after drawing 2z and discarding 9m
    frames = [_dframe(22 + 0.2 * k, before) for k in range(10)] + [
        _dframe(55 + 0.2 * k, after) for k in range(10)
    ]
    monkeypatch.setattr(
        dense,
        "dense_reads",
        lambda *a, **_unused_k: {
            "hand:TL": [f for f in frames if a[1] <= f["t"] <= a[2]]
        },
    )
    st = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 40, 50)]
    model = HandModel("E", {"S": st}, [])
    turns = [
        Turn(0, "S", "draw", None, 20.0),
        Turn(1, "W", "draw", None, 30.0),
        Turn(2, "S", "draw", None, 50.0),
        Turn(3, "W", "draw", None, 60.0),
    ]
    entry = {"corner_wind": {"TL": "S", "TR": "W", "BL": "E", "BR": "N"}}
    diagnostics: list[str] = []
    dense.draws(
        [("S", 1)],
        model,
        turns,
        {"S": {}},
        context=DenseContext(
            entry=entry,
            models=models_stub(),
            work_dir=tmp_path,
            t0=0.0,
            diagnostics=diagnostics,
        ),
    )
    assert diagnostics == [
        "dense hand reads for 1 uncertain draws: still views pin 1 of them"
    ]
    assert sorted((e.j, e.subset, e.hidden) for e in model.hand_ev) == [
        (0, False, 0),
        (1, False, 0),
    ]


# ---------------------------------------------------------------- the site's han/fu can
# be wrong


def test_han_fu_that_pay_the_same_differ_only_on_paper() -> None:
    # hand 19 of the second VOD: 10/40 against the site's 9/70, a baiman either way
    assert payment(10, 40, dealer=False, tsumo=False) == payment(
        9, 70, dealer=False, tsumo=False
    )
    # both mangan
    assert payment(4, 40, dealer=True, tsumo=True) == payment(
        5, 30, dealer=True, tsumo=True
    )
    assert payment(3, 30, dealer=False, tsumo=False) != payment(
        3, 40, dealer=False, tsumo=False
    )


def test_a_site_wrong_fact_replaces_the_sites_han_fu_and_keeps_the_deltas() -> None:
    entry = {
        "game": 1,
        "kyoku": 3,
        "honba": 1,
        "hand": 19,
        "corner_wind": {"TL": "S", "TR": "E", "BL": "W", "BR": "N"},
        "corner_site": {"TL": "EAST", "TR": "NORTH", "BL": "SOUTH", "BR": "WEST"},
    }
    fact = {
        "game": 1,
        "kyoku": 3,
        "honba": 1,
        "kind": "site_wrong",
        "corner": "BL",
        "han": 10,
        "fu": 40,
        "site": [9, 70],
    }
    facts = facts_for_hand([fact], entry)
    assert facts["site_score"] == {"han": 10, "fu": 40}
    deltas = {"EAST": 0, "SOUTH": 17300, "WEST": 0, "NORTH": -16300}
    result = HandResult(3, 1, 1, deltas, "ron", "SOUTH", "NORTH", 9, 70)
    hand = hand_of({**entry, "t_start": 0, "t_end": 60}, {}, result, facts)
    assert (hand.result.han, hand.result.fu, hand.result.deltas) == (10, 40, deltas)
    assert hand.site_han_fu == (9, 70)
    # the reviewer's correction is one short note on the hand
    entry, obs, result = synthetic_hand()
    d = decode_hand(entry, obs, result, facts, options=DecodeOptions(time_limit=5))
    assert d["result"]["site_wrong"]
    note = site_corrected(d["result"]["site"], (10, 40))
    assert d["notes"].count(note) == 1


def test_the_log_is_written_with_the_confirmed_han_fu_and_the_sites_deltas() -> None:
    d = {
        "dealer": "E",
        "haipai": {
            "E": [
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
                "9p",
            ],
            "S": [],
            "W": [],
            "N": [],
        },
        "turns": [
            {
                "i": 0,
                "seat": "E",
                "j": 0,
                "kind": "draw",
                "t": 10,
                "riichi": False,
                "draw": None,
                "draw2": None,
                "discard": "9p",
                "tsumogiri": False,
                "margin": None,
                "call": None,
                "own_call": None,
            }
        ],
        "draws": {},
        "dora": ["1s"],
        "ura": [],
        "result": {
            "winner": "S",
            "loser": "E",
            "han": 10,
            "fu": 40,
            "site": [9, 70],
            "site_wrong": True,
        },
        "score": {
            "match": True,
            "yaku": [{"name": "Chinitsu", "han": 6}, {"name": "Dora", "han": 4}],
        },
    }
    entry = {
        "kyoku": 0,
        "honba": 0,
        "sticks": 0,
        "scores": {"E": 25000, "S": 25000, "W": 25000, "N": 25000},
    }
    result = HandResult(
        0,
        0,
        0,
        {"EAST": -16000, "SOUTH": 16000, "WEST": 0, "NORTH": 0},
        "ron",
        "SOUTH",
        "EAST",
        9,
        70,
    )
    k, _ = kyoku_from_decode(d, entry, result)
    assert isinstance(k.result, Agari)
    (win,) = k.result.wins
    assert win.score_text == "倍満16000点"
    assert win.delta == [-16000, 16000, 0, 0]
    assert win.yaku == ["清一色(6飜)", "ドラ(4飜)"]


def test_uncertain_raw_discard_gets_question_without_duplicate_or_fact_override() -> (
    None
):
    row = {
        "field": "discard",
        "seat": "S",
        "turn": 17,
        "value": "2m",
        "runner_up": "1m",
        "margin": 0.5,
        "alternative_gap": 0.5,
        "human": False,
        "evidence": [{"region": "pond:BR", "t": 3097.5}],
    }
    items = uncertain_discards([row], [])
    assert len(items) == 1
    assert items[0]["kind"] == "discard"
    assert items[0]["tile"] == "2m"
    assert items[0]["runner_up"] == "1m"
    assert items[0]["t"] == 3097.5
    assert not uncertain_discards([row], items)
    assert not uncertain_discards([{**row, "human": True}], [])
    assert not uncertain_discards([{**row, "margin": 0.5001}], [])
    assert not uncertain_discards([{**row, "margin": None}], [])


@pytest.mark.parametrize("kind", ["discard", "missing_discard"])
def test_reviewed_discard_remains_fixed_during_repair(kind: str) -> None:
    turns = {s: [] for s in rules.SEATS}
    p = np.zeros(len(TILES))
    p[TI["1m"]] = 1
    turns["S"] = [SeatTurn(0, "draw", "1m", 0, 10, discard_p=p)]
    model = HandModel("E", turns, [])
    report = Report()
    facts = {kind: [{"seat": "S", "t": 10, "tile": "1m"}]}
    apply_hand_facts(model, facts, calls=[], window=(0, 20), report=report)
    assert report == Report()
    model.repair = True
    assert turns["S"][0].discard_p is None
    assert model.build().discards[("S", 0)] == {TI["1m"]: 1}
    # If no starting tile or draw can supply the reviewed discard, repair
    # must expose the contradiction rather than paying to change the fact.
    model.facts.haipai["S"] = [
        "2m",
        "3m",
        "4m",
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
    model.facts.draws[("S", 0)] = "9m"
    assert not model.solve(workers=1).ok


def test_confidence_uses_repaired_discard_without_final_override() -> None:
    turns = {s: [] for s in rules.SEATS}
    turns["S"] = [SeatTurn(0, "draw", "1m", 0, 10)]
    model = HandModel("E", turns, [])
    slot = pslot(0, "2m", 10)
    turn = Turn(0, "S", "draw", slot, 10)
    # The repaired identity is now fixed, so the final solve records no
    # variable-discard override. The original pond reading remains evidence.
    sol = Solution("repaired", 0, {}, {}, {})
    rows = confidence_rows(
        model,
        sol,
        context=ReviewContext(
            turns=[turn],
            calls=[],
            inds=[],
            entry={"corner_wind": {"BR": "S"}},
            facts={},
            lost_keys=set(),
            t0=0,
        ),
    )
    assert len(rows) == 1
    assert rows[0]["field"] == "discard"
    assert rows[0]["value"] == "1m"
    assert not rows[0]["human"]
    assert slot.tile == "2m"
    facts = {"missing_discard": [{"seat": "S", "t": 10, "tile": "1m"}]}
    reviewed = confidence_rows(
        model,
        sol,
        context=ReviewContext(
            turns=[turn],
            calls=[],
            inds=[],
            entry={"corner_wind": {"BR": "S"}},
            facts=facts,
            lost_keys=set(),
            t0=0,
        ),
    )
    assert reviewed[0]["human"]
    assert reviewed[0]["value"] == "1m"
