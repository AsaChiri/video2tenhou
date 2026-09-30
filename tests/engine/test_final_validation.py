"""Final construction and the review queue use the same legality checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from video2tenhou.engine import rules
from video2tenhou.engine.decode import HandDecoder
from video2tenhou.engine.solver import HandModel
from video2tenhou.engine.validation import review_artifact
from video2tenhou.train.data import CLASSES


def snapshot():
    entry = dict(
        hand=0,
        game=0,
        kyoku=0,
        honba=0,
        sticks=0,
        scores=dict.fromkeys("ESWN", 25000),
        corner_wind=dict(TL="E", TR="N", BL="S", BR="W"),
    )
    deck = [
        tile
        for tile in rules.KINDS + list(rules.REDS)
        if tile != "0s"
        for _ in range(rules.max_count(tile))
    ]
    haipai = {"E": deck[:14], "S": deck[14:27], "W": deck[27:40], "N": deck[40:53]}
    decoded = dict(
        hand=0,
        game=0,
        kyoku=0,
        honba=0,
        solver={"status": "optimal"},
        dealer="E",
        haipai=haipai,
        draws={},
        turns=[],
        dora=["0s"],
        ura=[],
        result=dict(
            outcome="draw", deltas=dict.fromkeys(("EAST", "SOUTH", "WEST", "NORTH"), 0)
        ),
        score=None,
        items=[],
        stats={"turns": 0},
        t_last=10,
        play_window=[0, 10],
    )
    # A complete exhaustive draw: dealer's first tile plus 69 later draws.
    counts = dict.fromkeys("ESWN", 0)
    for i in range(70):
        seat = "ESWN"[i % 4]
        j = counts[seat]
        counts[seat] += 1
        drawn = deck[52 + i] if i else None
        discard = drawn if i else haipai["E"][-1]
        decoded["turns"].append(
            dict(
                i=i,
                j=j,
                seat=seat,
                kind="draw",
                t=i,
                draw=drawn,
                discard=discard,
                tsumogiri=True,
                riichi=False,
            )
        )
        if drawn:
            decoded["draws"][f"{seat}:{j}"] = drawn
    return decoded, entry


def test_export_rejection_becomes_a_located_conflict_and_clears_after_correction():
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


def test_processing_limit_is_only_a_hand_note_in_saved_results():
    decoded, entry = snapshot()
    decoded["items"] = [
        dict(kind="solver_incomplete", stage="confidence", text="Retry processing")
    ]
    view = review_artifact(decoded, entry)
    assert view["items"] == []
    assert view["notes"] == ["Some automatic checks reached their time limit."]
    assert review_artifact(view, entry)["notes"] == view["notes"]


@pytest.mark.parametrize("riichi", [True, False])
def test_score_mismatch_keeps_indicators_and_requests_hand_or_ura(riichi):
    decoder = HandDecoder.__new__(HandDecoder)
    model = HandModel("E", {s: [] for s in rules.SEATS}, ["1m"])
    tiles = [
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
        "2z",
        "0s",
    ]
    model.facts.haipai["S"] = tiles
    original = model.solve(margins=False, workers=1)
    assert original.ok
    decoder.model, decoder.sol = model, original
    decoder.winner, decoder.dora, decoder.ura = "S", ["1m"], []
    decoder.turns, decoder.live_calls, decoder.unknown_kans = [], [], set()
    decoder.result = SimpleNamespace(outcome="ron", han=2, fu=30)
    decoder.problems, decoder.items, decoder.facts = [], [], {}
    decoder.dealer, decoder.site_riichi, decoder.context, decoder.t1 = (
        "E",
        {"S"} if riichi else set(),
        {},
        10,
    )
    p = np.zeros(len(CLASSES))
    p[CLASSES.index("1m")], p[CLASSES.index("0s")] = 0.99, 0.01
    decoder.inds = [{"tile": "1m", "p": p}]
    decoder._winning_hand = lambda sol: (tiles, "2z", False, -1)
    matched = SimpleNamespace(ok=True, han=2, fu=30, yaku=[], error=None)
    previous = SimpleNamespace(ok=True, han=1, fu=30, yaku=[], error=None)
    score_of = lambda concealed, win, dora: matched if dora == ["0s"] else previous
    decoder._scorer = lambda melds: score_of
    decoder._next_hand_from_the_site = lambda conc, win, sc, melds, j: (
        conc,
        win,
        sc,
        melds,
    )
    decoder._red_five_from_the_site = lambda conc, win, sc, melds, scorer, j: (conc, sc)
    decoder.check_score()
    decoder.check_draw()
    assert decoder.dora == model.indicators == ["1m"]
    assert decoder.inds[0]["tile"] == "1m"
    assert not decoder.score["match"]
    assert [item["kind"] for item in decoder.items] == (
        ["ura"] if riichi else ["result"]
    )
