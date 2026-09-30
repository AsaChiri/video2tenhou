"""Partial views cannot turn alternatives for one box into two visible tiles."""

import gzip
import json

import numpy as np
import pytest
from ortools.sat.python import cp_model

from tests.paths import DATA
from video2tenhou.engine import rules
from video2tenhou.engine.solver import (
    TI,
    TILES,
    HandModel,
    SeatTurn,
    _partial_cost,
    hand_evidence,
)


def posterior(**tiles):
    p = np.zeros(len(TILES))
    for tile, probability in tiles.items():
        p[TI[tile]] = probability
    return p


def match_cost(slots, tiles):
    counts = [tiles.count(t) for t in TILES]
    model = cp_model.CpModel()
    model.Minimize(_partial_cost(model, counts, slots, 100, 0))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    assert solver.Solve(model) == cp_model.OPTIMAL
    return solver.ObjectiveValue() / 10000


def test_hidden_copy_cannot_explain_second_alternative_of_same_box():
    slots = [posterior(**{"5s": 0.63, "8s": 0.37}), posterior(**{"8s": 1})]
    ordinary = match_cost(slots, ["5s", "8s", "7p"])
    phantom = match_cost(slots, ["5s", "8s", "8s"])
    assert ordinary == phantom == pytest.approx(0.37)


def test_two_visible_boxes_need_two_copies_and_keep_red_fives_distinct():
    slots = [posterior(**{"0p": 1}), posterior(**{"0p": 1})]
    assert match_cost(slots, ["0p", "5p"]) == 1
    assert match_cost(slots, ["0p", "0p"]) == 0


def test_assignment_can_use_second_choice_without_discarding_box():
    slots = [posterior(**{"5s": 0.63, "8s": 0.37}), posterior(**{"5s": 1})]
    assert match_cost(slots, ["5s", "8s"]) == pytest.approx(0.63)
    assert match_cost(slots, ["5s"]) == 1


def test_recorded_partial_rows_do_not_certify_an_unseen_extra_eight():
    fixture = DATA / "week11_partial_hand_slots.json.gz"
    data = json.loads(gzip.decompress(fixture.read_bytes()))
    turns = {s: [] for s in rules.SEATS}
    turns["S"] = [SeatTurn(0, "draw", "7p", 5547, 5557, t_draw_min=5547)]
    model = HandModel("E", turns, [])
    # A fixed test setup isolates the scoring defect. These starting counts
    # are not presented as independently adjudicated video ground truth.
    model.facts.haipai["S"] = data["test_starting_state"]
    model.hand_ev, model.draw_ev, _ = hand_evidence(
        "S", turns["S"], data["observations"], {}, False
    )
    assert model.hand_ev and all(ev.slots is not None for ev in model.hand_ev)
    model.facts.draws["S", 0] = "7p"
    confirmed = model.solve(margins=False, workers=1)
    model.facts.draws["S", 0] = "8s"
    previous_guess = model.solve(margins=False, workers=1)
    assert confirmed.ok and previous_guess.ok
    assert previous_guess.objective == pytest.approx(confirmed.objective)
