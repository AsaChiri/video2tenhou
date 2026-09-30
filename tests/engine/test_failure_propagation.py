"""Broken acquisition and dependencies must not become uncertainty or illegal hands."""

from collections import Counter
from types import SimpleNamespace

import pytest
from mahjong.shanten import Shanten

from video2tenhou import tenhou6
from video2tenhou.engine import decode, dense, pond_evidence, scoring
from video2tenhou.engine.hand import site_seat, site_seat_name
from video2tenhou.engine.review import facts_for_hand
from video2tenhou.engine.solver import SeatTurn
from video2tenhou.engine.turns import Turn

ENTRY = {"corner_wind": {"TL": "E", "TR": "S", "BL": "W", "BR": "N"}}
CONCEALED = [f"{n}m" for n in range(1, 10)] + ["1p", "2p", "3p", "5p"]
SCORE_OPTIONS = dict(
    tsumo=False, riichi=True, seat="S", round_wind="E", dora=[], ura=[]
)


@pytest.mark.parametrize(
    "operation", ["taken", "skipped", "draw", "riichi", "replacement"]
)
@pytest.mark.parametrize("failure", [OSError, RuntimeError, KeyError])
def test_acquisition_failure_stops_reconstruction(monkeypatch, operation, failure):
    def broken(*args, **kwargs):
        raise failure("acquisition failed")

    monkeypatch.setattr(dense, "dense_reads", broken)
    monkeypatch.setattr(dense, "dense_pond_reads", broken)
    monkeypatch.setattr(pond_evidence, "read_replacement", broken)
    slot = SimpleNamespace(id=1, xyxy=(0, 0, 10, 14), t_window=(10, 20))
    turns = [Turn(0, "S", "draw", slot, 20.0)]
    models, problems = (None, None, None, None), []
    model = SimpleNamespace(turns={"S": [SeatTurn(0, "draw", "1m", 10, 20)]})
    with pytest.raises(failure, match="acquisition failed"):
        if operation == "taken":
            dense.taken_discards("S", 10, 20, {}, ENTRY, models, None)
        elif operation == "skipped":
            dense.skipped_turns(
                ["turn 0: no discard of S"],
                turns,
                {},
                ENTRY,
                models,
                None,
                0,
                30,
                problems,
            )
        elif operation == "draw":
            dense.draws(
                [("S", 0)], model, turns, {}, ENTRY, {}, models, None, 0, problems
            )
        elif operation == "riichi":
            dense.turned_tile("S", turns, ENTRY, models, None, 0, problems)
        else:
            monkeypatch.setattr(
                pond_evidence,
                "replacement_requests",
                lambda *args: [
                    dict(acquired=False, window=[10, 20], seat="S", slot_id=1, t=20)
                ],
            )
            decoder = SimpleNamespace(
                turns=turns,
                entry=ENTRY,
                facts={},
                t0=0,
                t1=30,
                models=models,
                work_dir=None,
                problems=problems,
            )
            decode.HandDecoder.pond_replacements(decoder)
    assert not problems  # A failed read is not a review question about missing tiles.


@pytest.mark.parametrize("operation", ["score", "tenpai", "replay"])
def test_scoring_dependency_failures_propagate(monkeypatch, operation):
    def broken(*args, **kwargs):
        raise RuntimeError("scoring dependency failed")

    monkeypatch.setattr(scoring.HandCalculator, "estimate_hand_value", broken)
    monkeypatch.setattr(Shanten, "calculate_shanten", broken)
    with pytest.raises(RuntimeError, match="scoring dependency failed"):
        if operation == "score":
            scoring.score_hand(CONCEALED, "5p", [], **SCORE_OPTIONS)
        elif operation == "tenpai":
            scoring.is_tenpai(CONCEALED, [])
        else:
            tenhou6._wins(Counter(tenhou6.tile(t) for t in CONCEALED + ["5p"]))


def test_malformed_meld_schema_is_not_a_scoring_result():
    with pytest.raises(KeyError, match="unknown"):
        scoring.score_hand(
            CONCEALED, "5p", [{"type": "unknown", "tiles": ["1z"] * 3}], **SCORE_OPTIONS
        )


def test_domain_rejections_remain_explicit():
    assert not scoring.score_hand(["1m"] * 4, "1m", [], **SCORE_OPTIONS).ok
    result = scoring.score_hand(CONCEALED, "9p", [], **SCORE_OPTIONS)
    assert not result.ok and result.error
    assert not tenhou6._wins(Counter({11: 3}))
    assert not tenhou6._wins(Counter({11: 15}))


def test_missing_seat_mapping_cannot_assume_starting_winds():
    entry = {
        **ENTRY,
        "corner_site": {"TL": "SOUTH", "TR": "WEST", "BL": "NORTH", "BR": "EAST"},
    }
    assert site_seat("EAST", entry) == "N"
    assert site_seat_name("N", entry) == "EAST"
    del entry["corner_site"]["BR"]
    with pytest.raises(StopIteration):
        site_seat("EAST", entry)
    with pytest.raises(StopIteration):
        site_seat_name("invalid", entry)


def test_obsolete_result_annotation_requires_explicit_current_schema():
    entry = {**ENTRY, "game": 0, "kyoku": 0, "honba": 0}
    fact = dict(game=0, kyoku=0, honba=0, kind="result", ura=["1p"])
    with pytest.raises(ValueError, match="Unsupported result annotation"):
        facts_for_hand([fact], entry)
    assert fact["ura"] == ["1p"] and fact["kind"] == "result"
    current = dict(game=0, kyoku=0, honba=0, kind="ura", tiles=["1p"])
    assert facts_for_hand([current], entry)["ura"] == ["1p"]
