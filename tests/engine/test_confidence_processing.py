"""Machine search failures must never masquerade as tile questions or certainty."""

import time
from types import SimpleNamespace

import pytest

from video2tenhou.engine import rules, solver
from video2tenhou.engine.review import (
    confidence_state,
    uncertain_discards,
    uncertain_tiles,
)


@pytest.mark.parametrize("field", ["draw", "haipai", "discard"])
@pytest.mark.parametrize(
    "margin,gap,state",
    [
        (float("inf"), None, "resolved"),
        (0.5001, None, "resolved"),
        (0, None, "unresolvable"),
        (0, 80, "unresolvable"),
        (0, 0, "ambiguous"),
        (0, 0.5, "ambiguous"),
        (None, None, "unmeasured"),
    ],
)
def test_only_a_close_competing_witness_is_a_video_question(field, margin, gap, state):
    row = {
        "field": field,
        "seat": "S",
        "turn": 0,
        "value": "2p",
        "margin": margin,
        "alternative_gap": gap,
    }
    assert confidence_state(margin, gap) == state
    questions = (
        uncertain_discards([row], [])
        if field == "discard"
        else uncertain_tiles([row], [])
    )
    assert bool(questions) == (state == "ambiguous")


def empty_model():
    model = solver.HandModel("E", {s: [] for s in rules.SEATS}, [])
    hint = ({s: [0] * len(solver.TILES) for s in rules.SEATS}, {})
    return model, hint


def decoder_fixture():
    from video2tenhou.engine.decode import HandDecoder
    from video2tenhou.record import HandResult

    decoder = HandDecoder(
        {
            "corner_wind": {"TL": "E", "TR": "N", "BL": "S", "BR": "W"},
            "corner_site": {"TL": "EAST", "TR": "NORTH", "BL": "SOUTH", "BR": "WEST"},
        },
        {},
        HandResult(0, 0, 0, {}, "draw"),
        {},
        time_limit=60,
        models=None,
        work_dir=None,
    )
    decoder.turns = decoder.calls = decoder.dora = []
    decoder.t0, decoder.t1 = 0, 10
    return decoder


def test_confidence_budget_is_not_renewed_between_hand_solves(monkeypatch):
    decoder = decoder_fixture()
    decoder.unknown_kans = set()
    decoder.model, _ = empty_model()
    budgets = []

    def solve(**kw):
        budgets.append(kw["confidence_timeout"])
        return solver.Solution("optimal", 0, {}, {}, {}, confidence_seconds=6)

    monkeypatch.setattr(decoder.model, "solve", solve)
    for _ in range(3):
        decoder._solve()
    assert budgets == [10, 4, 0]


@pytest.mark.parametrize("status", ["unsolved", "infeasible"])
def test_only_proved_infeasibility_triggers_repair(monkeypatch, status):
    decoder = decoder_fixture()
    result = solver.Solution(status, 0, {}, {}, {})
    repairs = []
    monkeypatch.setattr(decoder, "_solve", lambda: result)
    monkeypatch.setattr(decoder, "_repair", lambda: repairs.append(True) or result)
    decoder._fit_evidence()
    assert bool(repairs) == (status == "infeasible")
    assert decoder.sol is result


def test_repair_timeout_does_not_trigger_conflict_diagnosis(monkeypatch):
    from video2tenhou.engine import decode

    decoder = decoder_fixture()
    decoder.model, _ = empty_model()
    result = solver.Solution("unsolved", 0, {}, {}, {})
    monkeypatch.setattr(decoder.model, "solve", lambda **kw: result)
    monkeypatch.setattr(
        decode, "diagnose", lambda *a: pytest.fail("Timeout is not infeasibility")
    )
    assert decoder._repair() is result
    assert decoder.items == []
    assert not decoder.model.repair


@pytest.mark.parametrize("first", ["UNKNOWN", "FEASIBLE"])
@pytest.mark.parametrize("budget", [10, 30])
def test_unfinished_check_runs_once_with_the_selected_budget(
    monkeypatch, first, budget
):
    calls = []

    class Search:
        parameters = SimpleNamespace()

        def Solve(self, model, callback):
            calls.append(self.parameters.max_time_in_seconds)
            assert len(model.proto.solution_hint.vars) == len(
                set(model.proto.solution_hint.vars)
            )
            return (
                getattr(solver.cp_model, first)
                if len(calls) == 1
                else solver.cp_model.INFEASIBLE
            )

        def ObjectiveValue(self):
            return 90 * solver.SCALE**2

        def BestObjectiveBound(self):
            return 0

        def Value(self, variable):
            return 0

    monkeypatch.setattr(solver.cp_model, "CpSolver", Search)
    model, hint = empty_model()
    monkeypatch.setattr(solver.time, "monotonic", lambda: 100.0)
    result = model._resolve(hint, 10, 1, deadline=100 + budget)
    assert calls == [budget]
    assert result == (0, None, None if first == "UNKNOWN" else 80)
    assert confidence_state(result[0], result[2]) == "unresolvable"


@pytest.mark.parametrize(
    "gap,stops", [(0, True), (0.5, True), (0.5001, False), (80, False)]
)
def test_witness_stopping_uses_the_inclusive_review_threshold(monkeypatch, gap, stops):
    stopped = []

    class Search:
        parameters = SimpleNamespace()

        def Solve(self, model, callback):
            callback.ObjectiveValue = self.ObjectiveValue
            callback.StopSearch = lambda: stopped.append(True)
            callback.on_solution_callback()
            return solver.cp_model.FEASIBLE

        def ObjectiveValue(self):
            return (10 + gap) * solver.SCALE**2

        def BestObjectiveBound(self):
            return 0

    monkeypatch.setattr(solver.cp_model, "CpSolver", Search)
    model, hint = empty_model()
    result = model._resolve(hint, 10, 1)
    assert bool(stopped) == stops
    assert confidence_state(result[0], result[2]) == (
        "ambiguous" if stops else "unresolvable"
    )


def test_exhausted_shared_budget_neither_starts_search_nor_invents_a_witness(
    monkeypatch,
):
    class Search:
        parameters = SimpleNamespace()

        def Solve(self, *args):
            pytest.fail("The hand's shared processing budget has already expired")

    monkeypatch.setattr(solver.cp_model, "CpSolver", Search)
    model, hint = empty_model()
    monkeypatch.setattr(
        model, "build", lambda **kw: pytest.fail("Expired checks must not build models")
    )
    result = model._resolve(hint, 10, 1, deadline=time.monotonic() - 1)
    assert result == (0, None, None)
    assert confidence_state(result[0], result[2]) == "unresolvable"


@pytest.mark.parametrize(
    "margin,gap,kinds,state",
    [
        (0, None, [], "unresolvable"),
        (0, 80, [], "unresolvable"),
        (0, 0.2, ["uncertain_tiles"], "ambiguous"),
        (float("inf"), None, [], "resolved"),
    ],
)
def test_decode_output_preserves_machine_failure_separately_from_questions(
    margin, gap, kinds, state
):
    from video2tenhou.engine.decode import HandDecoder, sanitize
    from video2tenhou.record import HandResult

    model, _ = empty_model()
    result = HandResult(0, 0, 0, {}, "draw")
    entry = {
        "hand": 0,
        "game": 0,
        "kyoku": 0,
        "honba": 0,
        "corner_wind": {"TL": "E", "TR": "N", "BL": "S", "BR": "W"},
    }
    decoder = HandDecoder(
        entry, {}, result, {}, time_limit=5, models=None, work_dir=None
    )
    decoder.sol = solver.Solution(
        "optimal",
        0,
        {"S": ["1m"] * 13},
        {},
        {},
        haipai_margins={"S": margin},
        haipai_alternative_gaps={"S": gap},
    )
    decoder.model = model
    decoder.turns = decoder.live_calls = decoder.calls = decoder.inds = (
        decoder.dora
    ) = []
    decoder.t0, decoder.t1 = 0, 10
    decoder.site_riichi, decoder.logs, decoder.score = set(), {}, None
    output = sanitize(decoder.output())
    assert [item["kind"] for item in output["items"]] == kinds
    assert output["confidence"][0]["state"] == state
