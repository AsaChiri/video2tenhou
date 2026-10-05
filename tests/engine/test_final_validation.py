# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Final construction and the review queue use the same legality checks."""

from __future__ import annotations

import pytest

from tests.engine import factories
from video2tenhou.engine import rules
from video2tenhou.engine.questions import Report
from video2tenhou.engine.score_reconcile import (
    Win,
    check_score,
    check_tenpai_and_ura,
    reconcile_score,
)
from video2tenhou.engine.solver import HandModel
from video2tenhou.engine.validation import review_artifact
from video2tenhou.record import HandResult


def snapshot() -> tuple:
    """Build a complete hand and its metadata for export validation."""
    entry = {
        "hand": 0,
        "game": 0,
        "kyoku": 0,
        "honba": 0,
        "sticks": 0,
        "scores": dict.fromkeys("ESWN", 25000),
        "corner_wind": {"TL": "E", "TR": "N", "BL": "S", "BR": "W"},
    }
    deck = [
        tile
        for tile in rules.KINDS + list(rules.PLAIN_OF)
        if tile != "0s"
        for _ in range(rules.max_count(tile))
    ]
    haipai = {"E": deck[:14], "S": deck[14:27], "W": deck[27:40], "N": deck[40:53]}
    decoded = {
        "hand": 0,
        "game": 0,
        "kyoku": 0,
        "honba": 0,
        "solver": {"status": "optimal"},
        "dealer": "E",
        "haipai": haipai,
        "draws": {},
        "turns": [],
        "dora": ["0s"],
        "ura": [],
        "result": {
            "outcome": "draw",
            "deltas": dict.fromkeys(("EAST", "SOUTH", "WEST", "NORTH"), 0),
            "site": [None, None],
            "riichi": [],
        },
        "score": None,
        "items": [],
        "stats": {"turns": 0},
        "t_last": 10,
        "play_window": [0, 10],
    }
    # A complete exhaustive draw: dealer's first tile plus 69 later draws.
    counts = dict.fromkeys("ESWN", 0)
    for i in range(70):
        seat = "ESWN"[i % 4]
        j = counts[seat]
        counts[seat] += 1
        drawn = deck[52 + i] if i else None
        discard = drawn if i else haipai["E"][-1]
        decoded["turns"].append(
            {
                "i": i,
                "j": j,
                "seat": seat,
                "kind": "draw",
                "t": i,
                "draw": drawn,
                "discard": discard,
                "tsumogiri": True,
                "riichi": False,
            }
        )
        if drawn:
            decoded["draws"][f"{seat}:{j}"] = drawn
    return decoded, entry


def test_export_rejection_becomes_a_located_conflict_and_clears_after_correction() -> (
    None
):
    decoded, entry = snapshot()
    decoded["haipai"]["E"][0] = "0s"
    view = review_artifact(decoded, entry)
    assert not view["validation"]["ok"]
    conflict = next(i for i in view["items"] if i.get("stage") == "export")
    over = next(o for o in conflict["over"] if o["tile"] == "0s")
    assert (over["count"], over["limit"]) == (2, 1)
    assert [s["kind"] for s in over["sources"]] == ["haipai", "indicator"]
    assert decoded["items"] == []  # projection must not mutate saved evidence
    view["haipai"]["E"][0] = "1m"
    corrected = review_artifact(view, entry)
    assert corrected["validation"]["ok"]
    assert not corrected["items"]


def test_export_violations_name_the_seat_and_the_time() -> None:
    decoded, entry = snapshot()
    south = next(t for t in decoded["turns"] if t["seat"] == "S" and t["j"] == 0)
    south["discard"], south["tsumogiri"] = "0p", False  # a tile South never held
    conflict = next(
        i
        for i in review_artifact(decoded, entry)["items"]
        if i.get("stage") == "export"
    )
    missing = next(v for v in conflict["violations"] if v["kind"] == "missing_tile")
    assert (missing["seat"], missing["tile"], missing["t"]) == ("S", "0p", south["t"])
    assert conflict["t"] == south["t"]


# South's closed hand waits on 2s-5s: pinfu tsumo, 2 han and 20 fu, plus a dora.
CONCEALED = [
    "2m",
    "3m",
    "4m",
    "6p",
    "7p",
    "8p",
    "6s",
    "7s",
    "8s",
    "3s",
    "4s",
    "9p",
    "9p",
]


@pytest.mark.parametrize("riichi", [True, False])
def test_score_mismatch_keeps_indicators_and_requests_hand_or_ura(
    *, riichi: bool
) -> None:
    """A score the indicators do not reach is asked, never matched by a new indicator.

    The indicator reads 1m (dora 2m, one han); read as 8p (dora 9p, the pair) the hand
    would score exactly the site's han.
    """
    site = 5 if riichi else 4
    hand = factories.hand(
        result=HandResult(0, 0, 0, {}, "tsumo", winner="SOUTH", han=site, fu=20)
    )
    model = HandModel("E", {s: [] for s in rules.SEATS}, ["1m"], tsumo_winner="S")
    model.facts.haipai["S"] = CONCEALED
    model.facts.draws["S", 0] = "2s"
    search = factories.search(model)
    win = Win(
        hand=hand,
        turns=[],
        live_calls=[],
        logs={},
        riichi=factories.riichi(*(["S"] if riichi else [])),
        dora=["1m"],
        wall_tiles=0,
    )
    report = Report()
    sol = reconcile_score(win, search, search.solve(), report)
    score = check_score(win, search, sol, report)
    check_tenpai_and_ura(hand, model, sol, [], win.riichi, report)
    assert win.dora == model.indicators == ["1m"]
    assert score is not None
    assert (score["han"], score["fu"], score["match"]) == (site - 1, 20, False)
    assert [item["kind"] for item in report.items] == ["ura" if riichi else "result"]
    assert report.notes == []
