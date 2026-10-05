# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Certificates separate proofs, close witnesses and unfinished searches."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ortools.sat.python import cp_model

from tests.builders import kinds, tile_counts
from tests.engine import factories
from video2tenhou.engine import rules, solver
from video2tenhou.engine.confidence import FIXED, Certificate, confidence_state
from video2tenhou.engine.decode import Reconstruction, artifact
from video2tenhou.engine.dense import DenseContext
from video2tenhou.engine.events import Ponds
from video2tenhou.engine.questions import Report, uncertain_discards, uncertain_tiles
from video2tenhou.engine.reconstruct import Search, apply_hand_facts, fit_evidence
from video2tenhou.engine.review import draws_to_reread, unseen_draw
from video2tenhou.engine.solver import (
    TI,
    TILES,
    DrawEvidence,
    HandEvidence,
    HandModel,
    SeatTurn,
)
from video2tenhou.files import sanitize

# Legal starting hands without fives; South's 1z and 2z leave first.
WEST = ["1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "8p", "9p", "7m", "8m", "9m"]
STARTING = {
    "E": [*["1m", "2m", "3m", "4m"] * 3, "6m", "6m"],
    "S": ["1z", "2z", "3z", "4z", "5z", "6z", "7z", "1p", "2p", "3p", "4p", "6p", "7p"],
    "W": WEST,
    "N": list(WEST),
}


def model_with(turns: dict[str, list[SeatTurn]], fixed: set[str]) -> HandModel:
    """East deals; the seats in ``fixed`` have confirmed starting hands."""
    model = HandModel("E", {s: turns.get(s, []) for s in rules.SEATS}, ["9s"])
    model.facts.haipai = {seat: list(STARTING[seat]) for seat in fixed}
    return model


def tie_turns() -> list[SeatTurn]:
    """South draws 9p and 6s in an order no evidence decides."""
    return [SeatTurn(0, "draw", "1z", 10, 20), SeatTurn(1, "draw", "2z", 30, 40)]


def tie_row() -> HandEvidence:
    """South's full resting row after both draws."""
    after = [t for t in STARTING["S"] if t not in ("1z", "2z")] + ["9p", "6s"]
    return HandEvidence(
        "S", 1, after_draw=False, e=tile_counts(after), weight=1.0, t0=45, t1=50
    )


def certified(model: HandModel, timeout: float = 60.0) -> solver.Solution:
    """Solve and certify a small model."""
    sol = model.solve(time_limit=30, workers=2)
    assert sol.status == "optimal"
    model.certify(sol, timeout=timeout, workers=2)
    assert sol.certified
    return sol


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
    field: str, margin: float | None, gap: float | None, state: str
) -> None:
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


def test_a_genuine_tie_makes_both_draws_ambiguous() -> None:
    """Two draws whose order nothing decides each have the other as a witness."""
    model = model_with({"S": tie_turns()}, fixed=set(STARTING))
    model.hand_ev.append(tie_row())
    sol = certified(model)
    first, second = sol.draws["S", 0], sol.draws["S", 1]
    assert {first, second} == {"9p", "6s"}
    for j, runner_up in ((0, second), (1, first)):
        certificate = sol.certificates["draw", "S", j]
        assert certificate.state == "ambiguous"
        assert certificate.gap == pytest.approx(0)
        assert certificate.margin == pytest.approx(0)
        assert certificate.runner_up == runner_up
    assert sorted(draws_to_reread(sol, model)) == [("S", 0), ("S", 1)]
    assert all(sol.certificates["haipai", s, -1] is FIXED for s in rules.SEATS)


@pytest.mark.parametrize(("second", "state"), [(0.95, "ambiguous"), (0.94, "resolved")])
def test_the_review_threshold_is_inclusive(second: float, state: str) -> None:
    """A witness exactly 0.5 costlier is close; 0.6 is certified away."""
    model = model_with({"S": [SeatTurn(0, "draw", "1z", 10, 20)]}, set(STARTING))
    model.draw_ev.append(DrawEvidence("S", 0, kinds({"1p": 1.0, "2p": second}), 10.0))
    sol = certified(model)
    assert sol.draws["S", 0] == "1p"
    certificate = sol.certificates["draw", "S", 0]
    assert certificate.state == state
    if state == "ambiguous":
        assert certificate.gap == pytest.approx(0.5)
        assert certificate.runner_up == "2p"
    else:
        assert certificate.margin > 0.5  # a proof, whatever the candidate costs
    assert not unseen_draw(model, sol, "S", 0)  # the reading covers it


def test_pinned_and_confirmed_draws_are_resolved() -> None:
    """A strongly read draw is proven; a confirmed one is fixed."""
    turns = {
        "S": [SeatTurn(0, "draw", "1z", 10, 20)],
        "W": [SeatTurn(0, "draw", "1s", 20, 30)],
    }
    model = model_with(turns, set(STARTING))
    model.draw_ev.append(DrawEvidence("S", 0, kinds({"1p": 1.0}), 10.0))
    model.facts.draws["W", 0] = "5z"
    sol = certified(model)
    assert sol.certificates["draw", "S", 0].state == "resolved"
    assert sol.certificates["draw", "S", 0].margin > 0.5
    assert sol.certificates["draw", "W", 0] is FIXED
    assert draws_to_reread(sol, model) == []


def test_a_draw_no_reconstruction_can_change_has_an_infinite_margin() -> None:
    """The rules leave one draw: excluding it has no legal reconstruction."""
    model = model_with({"N": [SeatTurn(0, "draw", "1s", 30, 40)]}, set(STARTING))
    model.facts.final["N"] = [t for t in STARTING["N"] if t != "1s"] + ["5z"]
    sol = certified(model)
    assert sol.draws["N", 0] == "5z"
    assert sol.certificates["draw", "N", 0] == Certificate(math.inf, math.inf)


def test_an_expired_budget_certifies_nothing_and_finds_no_witness() -> None:
    """Unfinished certification is unresolvable: no question, no reread, no proof."""
    model = model_with({"S": tie_turns()}, fixed=set(STARTING) - {"E"})
    model.hand_ev.append(tie_row())
    model.facts.draws["S", 1] = "6s"
    sol = certified(model, timeout=0)
    assert sol.certificates["draw", "S", 0] == Certificate(0.0)
    assert sol.certificates["haipai", "E", -1] == Certificate(0.0)
    assert sol.certificates["draw", "S", 1] is FIXED
    assert {c.state for c in sol.certificates.values()} == {"resolved", "unresolvable"}
    assert draws_to_reread(sol, model) == []
    # Not pinned by certification: a draw nothing covers is still asked.
    assert unseen_draw(model, sol, "S", 0)


def reference_gap(sol: solver.Solution, key: tuple[str, str, int]) -> float:
    """Solve one exclusion to optimality: the exact cost of changing a decision."""
    program = sol.program
    assert program is not None
    trial = program.model.clone()
    trial.clear_hints()
    field, seat, j = key
    if field == "draw":
        trial.add(program.draws[seat, j][TI[sol.draws[seat, j]]] == 0)
    elif field == "discard":
        chosen = sol.discards.get((seat, j), "7m")
        trial.add(program.discards[seat, j][TI[chosen]] == 0)
    else:
        solver._differs(trial, program.starting[seat], sol.haipai[seat], "reference")
    search = cp_model.CpSolver()
    search.parameters.num_workers = 2
    status = search.solve(trial)
    if status == cp_model.INFEASIBLE:
        return math.inf
    assert status == cp_model.OPTIMAL
    return search.objective_value / solver.SCALE**2 - sol.objective


def test_grouped_certificates_agree_with_per_decision_searches() -> None:
    """Every certificate is consistent with the exact cost of changing its decision."""
    discard = kinds({"7m": 0.7, "8m": 0.3})
    turns = {
        "S": tie_turns(),
        "W": [SeatTurn(0, "draw", "1s", 20, 30)],
        "N": [SeatTurn(0, "draw", "7m", 30, 40, discard_p=discard)],
    }
    model = model_with(turns, fixed={"S", "N"})
    model.hand_ev += [
        tie_row(),
        HandEvidence(
            "W",
            -1,
            after_draw=False,
            e=tile_counts(STARTING["W"]),
            weight=1.0,
            t0=0,
            t1=5,
        ),
    ]
    model.draw_ev += [
        DrawEvidence("W", 0, kinds({"1p": 1.0, "2p": 0.97}), 10.0),
        DrawEvidence("N", 0, kinds({"5z": 1.0}), 10.0),
    ]
    sol = certified(model)
    expected = {
        ("draw", "S", 0): "ambiguous",
        ("draw", "S", 1): "ambiguous",
        ("draw", "W", 0): "ambiguous",
        ("draw", "N", 0): "resolved",
        ("discard", "N", 0): "resolved",
        ("haipai", "E", -1): "ambiguous",  # nothing shows the dealer's hand
        ("haipai", "W", -1): "resolved",
        ("haipai", "S", -1): "resolved",
        ("haipai", "N", -1): "resolved",
    }
    assert {key: c.state for key, c in sol.certificates.items()} == expected
    for key, certificate in sol.certificates.items():
        if certificate is FIXED:
            continue
        exact = reference_gap(sol, key)
        assert certificate.margin <= exact + 1e-6, key  # a lower bound
        assert (exact > 0.5 + 1e-9) == (certificate.state == "resolved"), key
        if certificate.state == "ambiguous":
            assert certificate.gap is not None
            assert exact <= certificate.gap + 1e-6 <= 0.5 + 1e-6, key
    assert sol.certificates["draw", "W", 0].runner_up == "2p"


def first_fit(time_limit: float) -> tuple[Search, solver.Solution, dict, Report]:
    """South's 5z discard is impossible as read; only repair may re-read it."""
    p = np.full(len(TILES), 0.0005)
    p[TI["5z"]], p[TI["6z"]] = 0.97, 0.015  # too weak for an ordinary alternative
    turns: dict[str, list[SeatTurn]] = {s: [] for s in rules.SEATS}
    turns["S"] = [SeatTurn(0, "draw", "5z", 10, 20, discard_p=p / p.sum())]
    model = HandModel("E", turns, ["2z"])
    hand = [t for t in STARTING["S"] if t != "5z"] + ["5p"]
    facts = {
        "haipai": [{"seat": "S", "tiles": hand, "soft": False}],
        "draw": [{"seat": "S", "t": 20, "tile": "9m", "j": 0}],
    }
    report = Report()
    apply_hand_facts(model, facts, calls=[], window=(0.0, 60.0), report=report)
    search = factories.search(model, time_limit=time_limit, workers=8)
    sol, repaired = fit_evidence(
        search,
        factories.hand(),
        factories.sequence(),
        factories.riichi(),
        melds_before={s: {} for s in rules.SEATS},
        context=DenseContext(entry=factories.ENTRY),
        report=report,
    )
    return search, sol, repaired, report


def test_proved_infeasibility_is_repaired() -> None:
    """A legal reconstruction exists only with the discard re-read."""
    search, sol, repaired, report = first_fit(time_limit=10)
    assert sol.status == "repaired"
    assert repaired == {("S", 0): "6z"}  # the pond's second reading
    assert sol.optimal  # both the repair and the final search were proven
    assert not sol.certified  # only the final reconstruction is certified
    assert not search.model.repair
    assert report.items == []  # a repaired hand is no conflict


def test_a_timeout_is_not_infeasibility() -> None:
    """A search stopped before any candidate neither repairs nor diagnoses."""
    search, sol, repaired, report = first_fit(time_limit=0)
    assert sol.status == "unsolved"
    assert repaired == {}
    assert report.items == []
    assert not search.model.repair


def test_final_certification_runs_once() -> None:
    """Solves leave certification to the final pass, which runs a single time."""
    search, sol, _, _ = first_fit(time_limit=10)
    search.certify(sol)
    certificates = dict(sol.certificates)
    assert certificates
    search.certify(sol)
    assert all(sol.certificates[key] is c for key, c in certificates.items())


@pytest.mark.parametrize(
    ("certificate", "kinds", "state"),
    [
        (Certificate(0), [], "unresolvable"),
        (Certificate(0, 80), [], "unresolvable"),
        (Certificate(0, 0.2), ["uncertain_tiles"], "ambiguous"),
        (FIXED, [], "resolved"),
    ],
)
def test_decode_output_preserves_machine_failure_separately_from_questions(
    certificate: Certificate, kinds: list[str], state: str
) -> None:
    """A processing limit is a diagnostic: never a question nor a note."""
    sol = solver.Solution(
        "optimal",
        0,
        {"S": ["1m"] * 13},
        {},
        {},
        certificates={("haipai", "S", -1): certificate},
    )
    model = HandModel("E", {s: [] for s in rules.SEATS}, [])
    rec = Reconstruction(
        hand=factories.hand(t1=10.0),
        ponds=Ponds({}, 0.0, None),
        seq=factories.sequence(),
        riichi=factories.riichi(),
        search=factories.search(model),
        sol=sol,
        repaired={},
        dora=[],
        indicators=[],
        score=None,
        lost=set(),
    )
    output = sanitize(artifact(rec, Report()))
    assert [item["kind"] for item in output["items"]] == kinds
    assert output["confidence"][0]["state"] == state
    assert output["notes"] == []
    assert bool(output["diagnostics"]) == (state == "unresolvable")
