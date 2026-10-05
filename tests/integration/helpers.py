# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Restore recorded solver inputs without changing their evidence or constraints."""

from __future__ import annotations

import numpy as np

from video2tenhou.engine.solver import (
    DrawEvidence,
    Facts,
    HandEvidence,
    HandModel,
    SeatTurn,
    WinSpec,
)


def restore_model(data: dict) -> HandModel:
    """Load recorded evidence in the current solver schema."""
    turns: dict[str, list[SeatTurn]] = {}
    for seat, rows in data["turns"].items():
        turns[seat] = []
        for turn in rows:
            fields = dict(turn)
            if fields["discard_p"] is not None:
                fields["discard_p"] = np.array(fields["discard_p"])
            turns[seat].append(SeatTurn(**fields))
    model = HandModel(
        data["dealer"],
        turns,
        data["indicators"],
        tsumo_winner=data["tsumo_winner"],
        ura=data["ura"],
    )
    for row in data["hand_ev"]:
        fields = dict(row)
        fields["e"] = np.array(fields["e"])
        model.hand_ev.append(HandEvidence(**fields))
    for row in data["draw_ev"]:
        fields = dict(row)
        fields["p"] = np.array(fields["p"])
        model.draw_ev.append(DrawEvidence(**fields))
    facts = dict(data["facts"])
    facts["draws"] = {(row["seat"], row["j"]): row["tile"] for row in facts["draws"]}
    model.facts = Facts(**facts)
    for name in (
        "repair",
        "forbidden_hands",
        "bound_hands",
        "tenpai",
        "result_constraints",
    ):
        setattr(model, name, data[name])
    if data["win"]:
        win = dict(data["win"])
        if win["ron_from"] is not None:
            win["ron_from"] = tuple(win["ron_from"])
        model.win = WinSpec(**win)
    return model
