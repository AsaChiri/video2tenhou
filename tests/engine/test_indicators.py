# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The dead-wall row: indicators tracked by order and identity, never by position."""

import numpy as np

from video2tenhou.engine.indicators import indicator_row
from video2tenhou.train.data import CLASS_INDEX, CLASSES


def ind(
    tile: "str", x: "float", conf: float = 0.95, second: "str | None" = None
) -> "dict":
    """Create a positioned indicator reading with an optional competing tile."""
    p = np.full(len(CLASSES), 0.001)
    p[CLASS_INDEX[tile]] = conf
    if second:
        p[CLASS_INDEX[second]] = 1 - conf
    p /= p.sum()
    return {
        "key": ["ind", 0],
        "tile": tile,
        "conf": conf,
        "seen": 3,
        "sideways": 0.0,
        "disagree": False,
        "xyxy": [x, 500, x + 100, 630],
        "p": p.tolist(),
    }


def view(*, t: "float", inds: "list[dict]", partial: bool = False) -> "dict":
    """Create an indicator view with a specified time and completeness."""
    return {
        "region": "pond:TR",
        "t0": t,
        "t1": t + 3,
        "n_readings": 3,
        "n_used": 3,
        "count": 0,
        "quality": 0.9,
        "slots": [],
        "indicators": inds,
        "partial": partial,
    }


def test_a_pushed_wall_read_differently_is_the_same_indicator() -> None:
    """Verify a pushed wall read differently is the same indicator.

    Hand 2 of the second VOD: the wall was pushed 110 px and the tile read 4p instead of
    4s. The count did not change, so it is the same tile, misread: no second indicator,
    no kan.
    """
    views = [view(t=t, inds=[ind("4s", 546)]) for t in range(0, 60, 6)] + [
        view(t=t, inds=[ind("4p", 439)]) for t in range(60, 90, 6)
    ]
    row = indicator_row({"pond:TR": views}, -1, 100)
    assert len(row) == 1
    assert row[0]["tile"] == "4s"
    assert row[0]["readings"] == {"4s": 10, "4p": 5}
    assert row[0]["alternative"] == "4p"


def test_a_kan_adds_one_tile_to_the_row_while_it_moves() -> None:
    """Verify a kan adds one tile to the row while it moves.

    Hand 11 of the second VOD: the dora 1z is pushed several times; the kan indicator 1p
    appears beside it.
    """
    views = [view(t=t, inds=[ind("1z", 550 - t)]) for t in range(0, 60, 6)]
    views += [
        view(t=t, inds=[ind("1z", 250), ind("1p", 350)]) for t in range(60, 100, 6)
    ]
    row = indicator_row({"pond:TR": views}, -1, 100)
    assert [(v["tile"], v["t_first"]) for v in row] == [("1z", 0), ("1p", 60)]
    assert row[1]["t_before"] == 57  # the region's last view without it ended there


def test_a_new_tile_reading_like_its_neighbour_goes_where_it_is() -> None:
    """Verify position separates neighbouring indicators with similar readings.

    The reference VOD's hand 0: the kan indicator first reads like the dora; nearness
    decides which is new.
    """
    views = [view(t=t, inds=[ind("6m", 300)]) for t in range(0, 60, 6)]
    views += [view(t=60, inds=[ind("6m", 200), ind("6m", 300)])]
    views += [
        view(t=t, inds=[ind("6z", 200), ind("6m", 300)]) for t in range(66, 90, 6)
    ]
    row = indicator_row({"pond:TR": views}, -1, 100)
    assert [v["tile"] for v in row] == ["6m", "6z"]
