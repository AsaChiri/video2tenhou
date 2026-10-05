# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Conflict trials must preserve evidence even when the solver raises."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import pytest

from video2tenhou.engine import rules, solver


def fact_case(model: solver.HandModel) -> None:
    """Add one reviewed starting hand to diagnose."""
    model.facts.haipai["E"] = ["1m"] * 14


def result_case(model: solver.HandModel) -> None:
    """Require the dealer's starting state to be tenpai."""
    model.tenpai = [("E", -1, 4)]


def riichi_case(model: solver.HandModel) -> None:
    """Add one declaration whose removal the solver can try."""
    model.turns["E"] = [solver.SeatTurn(0, "draw", "1m", 0, 1, riichi=True)]


def meld_case(model: solver.HandModel) -> None:
    """Add one call whose removal the solver can try."""
    model.turns["E"] = [solver.SeatTurn(0, "call", "2m", 0, 1, removed=["1m", "1m"])]


@pytest.mark.parametrize("configure", [fact_case, result_case, riichi_case, meld_case])
@pytest.mark.parametrize("fail", [False, True])
def test_diagnostic_trial_preserves_original_evidence(
    monkeypatch: pytest.MonkeyPatch,
    configure: Callable[[solver.HandModel], None],
    *,
    fail: bool,
) -> None:
    """Successful and interrupted trials both leave the source model intact."""
    model = solver.HandModel("E", {seat: [] for seat in rules.SEATS}, [])
    configure(model)
    facts, turns = model.facts, model.turns
    before = (
        asdict(facts),
        {seat: [asdict(turn) for turn in rows] for seat, rows in turns.items()},
    )
    trials = []

    def build(trial: solver.HandModel) -> solver.Program:
        trials.append(trial)
        assert trial is not model
        if fail:
            raise RuntimeError("diagnostic solver interrupted")
        return solver.Program()

    monkeypatch.setattr(solver.HandModel, "build", build)
    if fail:
        with pytest.raises(RuntimeError, match="diagnostic solver interrupted"):
            solver.diagnose(model)
    else:
        assert solver.diagnose(model)
    assert trials
    assert model.facts is facts
    assert model.turns is turns
    assert model.result_constraints
    assert (
        asdict(model.facts),
        {seat: [asdict(turn) for turn in rows] for seat, rows in model.turns.items()},
    ) == before
