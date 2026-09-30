# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Machine search failures must never masquerade as tile questions or certainty."""

import time
from types import SimpleNamespace

import pytest

from video2tenhou.engine import decode, rules, solver
from video2tenhou.engine.confidence import confidence_state
from video2tenhou.engine.decode import DecodeOptions, HandDecoder
from video2tenhou.engine.review import (
    uncertain_discards,
    uncertain_tiles,
)
from video2tenhou.files import sanitize
from video2tenhou.record import HandResult


@pytest.mark.parametrize("field", ["draw", "haipai", "discard"])
@pytest.mark.parametrize(
    ("margin", "gap", "state"),
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
def test_only_a_close_competing_witness_is_a_video_question(
    field: str, margin: "float | None", gap: float | None, state: str
) -> None:
    """Verify only a close competing witness is a video question."""
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


def empty_model() -> "tuple[solver.HandModel, tuple[dict[str, list[int]], dict]]":
    """Create an empty reconstruction model and its matching warm-start hint."""
    model = solver.HandModel("E", {s: [] for s in rules.SEATS}, [])
    hint = ({s: [0] * len(solver.TILES) for s in rules.SEATS}, {})
    return model, hint


def decoder_fixture() -> "HandDecoder":
    """Create a draw decoder with no recognition evidence or human facts."""
    decoder = HandDecoder(
        {
            "corner_wind": {"TL": "E", "TR": "N", "BL": "S", "BR": "W"},
            "corner_site": {"TL": "EAST", "TR": "NORTH", "BL": "SOUTH", "BR": "WEST"},
        },
        {},
        HandResult(0, 0, 0, {}, "draw"),
        {},
        options=DecodeOptions(time_limit=60, models=None, work_dir=None),
    )
    decoder.turns = decoder.calls = decoder.dora = []
    decoder.t0, decoder.t1 = 0, 10
    return decoder


def test_confidence_budget_is_not_renewed_between_hand_solves(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify confidence budget is not renewed between hand solves."""
    decoder = decoder_fixture()
    decoder.unknown_kans = set()
    decoder.model, _ = empty_model()
    budgets = []

    def solve(**kw: "object") -> "solver.Solution":
        budgets.append(kw["confidence_timeout"])
        return solver.Solution("optimal", 0, {}, {}, {}, confidence_seconds=6)

    monkeypatch.setattr(decoder.model, "solve", solve)
    for _ in range(3):
        decoder._solve()
    assert budgets == [10, 4, 0]


@pytest.mark.parametrize("status", ["unsolved", "infeasible"])
def test_only_proved_infeasibility_triggers_repair(
    monkeypatch: "pytest.MonkeyPatch", status: str
) -> None:
    """Verify only proved infeasibility triggers repair."""
    decoder = decoder_fixture()
    result = solver.Solution(status, 0, {}, {}, {})
    repairs = []
    monkeypatch.setattr(decoder, "_solve", lambda: result)
    monkeypatch.setattr(decoder, "_repair", lambda: repairs.append(True) or result)
    decoder._fit_evidence()
    assert bool(repairs) == (status == "infeasible")
    assert decoder.sol is result


def test_repair_timeout_does_not_trigger_conflict_diagnosis(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify repair timeout does not trigger conflict diagnosis."""
    decoder = decoder_fixture()
    decoder.model, _ = empty_model()
    result = solver.Solution("unsolved", 0, {}, {}, {})
    monkeypatch.setattr(decoder.model, "solve", lambda **_unused_kw: result)
    monkeypatch.setattr(
        decode,
        "diagnose",
        lambda *_unused_a: pytest.fail("Timeout is not infeasibility"),
    )
    assert decoder._repair() is result
    assert decoder.items == []
    assert not decoder.model.repair


@pytest.mark.parametrize("first", ["UNKNOWN", "FEASIBLE"])
@pytest.mark.parametrize("budget", [10, 30])
def test_unfinished_check_runs_once_with_the_selected_budget(
    monkeypatch: "pytest.MonkeyPatch", first: str, budget: int
) -> None:
    """Verify unfinished check runs once with the selected budget."""
    calls = []

    class Search:
        parameters = SimpleNamespace()

        def solve(
            self,
            model: "solver.cp_model.CpModel",
            callback: "solver.cp_model.CpSolverSolutionCallback",
        ) -> "solver.cp_model.CpSolverStatus":
            calls.append(self.parameters.max_time_in_seconds)
            assert len(model.proto.solution_hint.vars) == len(
                set(model.proto.solution_hint.vars)
            )
            return (
                getattr(solver.cp_model, first)
                if len(calls) == 1
                else solver.cp_model.INFEASIBLE
            )

        @property
        def objective_value(self) -> "int":
            return 90 * solver.SCALE**2

        @property
        def best_objective_bound(self) -> int:
            return 0

        def value(self, variable: "solver.cp_model.IntVar") -> int:
            return 0

    monkeypatch.setattr(solver.cp_model, "CpSolver", Search)
    model, hint = empty_model()
    monkeypatch.setattr(solver.time, "monotonic", lambda: 100.0)
    result = model._resolve(
        hint, 10, 1, options=solver.ResolveOptions(deadline=100 + budget)
    )
    assert calls == [budget]
    assert result == (0, None, None if first == "UNKNOWN" else 80)
    assert confidence_state(result[0], result[2]) == "unresolvable"


@pytest.mark.parametrize(
    ("gap", "stops"), [(0, True), (0.5, True), (0.5001, False), (80, False)]
)
def test_witness_stopping_uses_the_inclusive_review_threshold(
    *, monkeypatch: "pytest.MonkeyPatch", gap: float, stops: bool
) -> None:
    """Verify witness stopping uses the inclusive review threshold."""
    stopped = []

    class Search:
        parameters = SimpleNamespace()

        def solve(
            self,
            model: "solver.cp_model.CpModel",
            callback: "solver.cp_model.CpSolverSolutionCallback",
        ) -> "solver.cp_model.CpSolverStatus":
            monkeypatch.setattr(
                type(callback),
                "objective_value",
                property(lambda _: self.objective_value),
            )
            monkeypatch.setattr(callback, "stop_search", lambda: stopped.append(True))
            callback.OnSolutionCallback()
            return solver.cp_model.FEASIBLE

        @property
        def objective_value(self) -> "float":
            return (10 + gap) * solver.SCALE**2

        @property
        def best_objective_bound(self) -> int:
            return 0

    monkeypatch.setattr(solver.cp_model, "CpSolver", Search)
    model, hint = empty_model()
    result = model._resolve(hint, 10, 1)
    assert bool(stopped) == stops
    assert confidence_state(result[0], result[2]) == (
        "ambiguous" if stops else "unresolvable"
    )


def test_exhausted_shared_budget_neither_starts_search_nor_invents_a_witness(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify exhausted shared budget neither starts search nor invents a witness."""

    class Search:
        parameters = SimpleNamespace()

        def solve(self, *_unused_args: object) -> None:
            pytest.fail("The hand's shared processing budget has already expired")

    monkeypatch.setattr(solver.cp_model, "CpSolver", Search)
    model, hint = empty_model()
    monkeypatch.setattr(
        model,
        "build",
        lambda **_unused_kw: pytest.fail("Expired checks must not build models"),
    )
    result = model._resolve(
        hint, 10, 1, options=solver.ResolveOptions(deadline=time.monotonic() - 1)
    )
    assert result == (0, None, None)
    assert confidence_state(result[0], result[2]) == "unresolvable"


@pytest.mark.parametrize(
    ("margin", "gap", "kinds", "state"),
    [
        (0, None, [], "unresolvable"),
        (0, 80, [], "unresolvable"),
        (0, 0.2, ["uncertain_tiles"], "ambiguous"),
        (float("inf"), None, [], "resolved"),
    ],
)
def test_decode_output_preserves_machine_failure_separately_from_questions(
    margin: "float | None", gap: float | None, kinds: "list[str]", state: str
) -> None:
    """Verify decode output preserves machine failure separately from questions."""
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
        entry,
        {},
        result,
        {},
        options=DecodeOptions(time_limit=5, models=None, work_dir=None),
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
