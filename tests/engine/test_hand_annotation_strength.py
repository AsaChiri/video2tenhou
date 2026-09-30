"""Uncertain hand annotations guide evidence without becoming human constraints."""

from collections import Counter
from types import SimpleNamespace

import pytest

from video2tenhou.engine import rules
from video2tenhou.engine.decode import HandDecoder
from video2tenhou.engine.review import facts_for_hand
from video2tenhou.engine.solver import TILES, HandModel

ENTRY = {
    "game": 0,
    "kyoku": 0,
    "honba": 0,
    "corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"},
}
HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "3z", "6z"]


def annotation(kind, **fields):
    return {
        "game": 0,
        "kyoku": 0,
        "honba": 0,
        "corner": "TR",
        "kind": kind,
        "tiles": HAND,
        **fields,
    }


@pytest.mark.parametrize("kind", ["haipai", "final_hand"])
@pytest.mark.parametrize("source", ["legacy-import", "camera-import"])
def test_sourced_hand_requires_explicit_strength(kind, source):
    with pytest.raises(ValueError, match="must explicitly set soft"):
        facts_for_hand([annotation(kind, source=source)], ENTRY)


@pytest.mark.parametrize("soft", ["false", "true", 0, 1, None])
def test_strength_rejects_non_boolean_values(soft):
    with pytest.raises(ValueError, match="soft must be a boolean"):
        facts_for_hand([annotation("final_hand", soft=soft)], ENTRY)


@pytest.mark.parametrize("kind", ["haipai", "final_hand"])
@pytest.mark.parametrize(
    "fields,soft",
    [
        ({}, False),
        ({"source": "reviewed-import", "soft": False}, False),
        ({"source": "camera-import", "soft": True}, True),
    ],
)
def test_explicit_hand_strength_controls_constraint_and_evidence(kind, fields, soft):
    facts = facts_for_hand([annotation(kind, **fields)], ENTRY)
    model = HandModel("E", {seat: [] for seat in rules.SEATS}, [])
    decoder = SimpleNamespace(
        facts=facts, live_calls=[], tsumo_winner=None, t0=10, t1=20, problems=[]
    )
    HandDecoder._apply_hand_facts(decoder, model)
    constraints = model.facts.haipai if kind == "haipai" else model.facts.final
    assert decoder.problems == []
    if soft:
        assert "S" not in constraints
        (evidence,) = model.hand_ev
        assert evidence.seat == "S" and evidence.j == -1
        assert evidence.t0 == evidence.t1 == (10 if kind == "haipai" else 20)
        assert evidence.e.tolist() == [HAND.count(tile) for tile in TILES]
    else:
        assert constraints["S"] == HAND
        assert model.hand_ev == []


def test_soft_winning_hand_without_winning_tile_targets_pre_draw_state():
    facts = facts_for_hand([annotation("final_hand", soft=True)], ENTRY)
    model = HandModel("E", {seat: [] for seat in rules.SEATS}, [], tsumo_winner="S")
    decoder = SimpleNamespace(
        facts=facts, live_calls=[], tsumo_winner="S", t0=10, t1=20, problems=[]
    )
    HandDecoder._apply_hand_facts(decoder, model)
    assert model.facts.final == model.facts.final_excl == {}
    assert model.hand_ev[0].j == -1


@pytest.mark.parametrize("kind", ["haipai", "final_hand"])
def test_incorrect_soft_hand_cannot_override_confirmed_tiles(kind):
    fact = annotation(kind, soft=True)
    fact["tiles"] = [
        "1m"
    ] * 13  # Impossible as a hard constraint, but a permissible mistaken observation.
    facts = facts_for_hand([fact], ENTRY)
    model = HandModel("E", {seat: [] for seat in rules.SEATS}, [])
    model.facts.haipai["S"] = HAND
    decoder = SimpleNamespace(
        facts=facts, live_calls=[], tsumo_winner=None, t0=10, t1=20, problems=[]
    )
    HandDecoder._apply_hand_facts(decoder, model)
    solution = model.solve(margins=False, workers=1, time_limit=3)
    assert solution.ok
    assert Counter(solution.haipai["S"]) == Counter(HAND)
