# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0
"""Small inputs for the decoder's stages."""

from __future__ import annotations

import dataclasses

from video2tenhou.engine.decode import DecodeOptions
from video2tenhou.engine.events import Hand, Riichi, TurnSequence
from video2tenhou.engine.melds import Call
from video2tenhou.engine.reconstruct import Search
from video2tenhou.engine.solver import HandModel
from video2tenhou.engine.turns import Turn
from video2tenhou.record import HandResult

SEATS_BY_CORNER = {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}
ENTRY = {
    "hand": 0,
    "game": 0,
    "kyoku": 0,
    "honba": 0,
    "corner_wind": SEATS_BY_CORNER,
    "corner_site": {
        c: {"E": "EAST", "S": "SOUTH", "W": "WEST", "N": "NORTH"}[w]
        for c, w in SEATS_BY_CORNER.items()
    },
}
DRAW = Hand(
    entry=ENTRY,
    obs={},
    result=HandResult(0, 0, 0, {}, "draw"),
    site_han_fu=(None, None),
    facts={},
    t0=0.0,
    t1=60.0,
)


def hand(**fields: object) -> Hand:
    """Build a hand's fixed inputs: a draw with no observations or facts by default."""
    return dataclasses.replace(DRAW, **fields)


def sequence(
    turns: list[Turn] | None = None, live_calls: list[Call] | None = None
) -> TurnSequence:
    """Build a merged turn sequence whose live calls are every call."""
    calls = list(live_calls or [])
    return TurnSequence(list(turns or []), calls, calls, 0)


def riichi(*seats: str) -> Riichi:
    """Build declarations with known turns and no called-tile alternatives."""
    return Riichi(frozenset(seats), frozenset(), [])


def search(
    model: HandModel,
    turns: list[Turn] | None = None,
    *,
    time_limit: float = 5.0,
    workers: int = 1,
) -> Search:
    """Wrap a model with small solver budgets."""
    options = DecodeOptions(time_limit=time_limit, workers=workers)
    return Search(model, list(turns or []), set(), options)
