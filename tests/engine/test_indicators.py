# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The dead-wall row: indicators tracked by order and identity, never by position."""

from __future__ import annotations

from tests.builders import observation, posterior, slot
from video2tenhou.engine.indicators import indicator_row


def ind(tile: str, x: float, conf: float = 0.95, second: str | None = None) -> dict:
    """Create a positioned indicator reading with an optional competing tile."""
    p = posterior(tile, conf, floor=0.001, second=second, second_peak=1 - conf)
    return slot(tile, key=["ind", 0], xyxy=[x, 500, x + 100, 630], p=p, conf=conf)


def view(*, t: float, inds: list[dict], partial: bool = False) -> dict:
    """Create an indicator view with a specified time and completeness."""
    return observation("pond:TR", t, t + 3, indicators=inds, partial=partial)


def test_a_pushed_wall_read_differently_is_the_same_indicator() -> None:
    """Hand 2 of the second VOD: the wall was pushed 110 px and the tile read 4p instead
    of 4s. The count did not change, so it is the same tile, misread: no second
    indicator, no kan.
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
    """Hand 11 of the second VOD: the dora 1z is pushed several times; the kan indicator
    1p appears beside it.
    """
    views = [view(t=t, inds=[ind("1z", 550 - t)]) for t in range(0, 60, 6)]
    views += [
        view(t=t, inds=[ind("1z", 250), ind("1p", 350)]) for t in range(60, 100, 6)
    ]
    row = indicator_row({"pond:TR": views}, -1, 100)
    assert [(v["tile"], v["t_first"]) for v in row] == [("1z", 0), ("1p", 60)]
    assert row[1]["t_before"] == 57  # the region's last view without it ended there


def test_a_new_tile_reading_like_its_neighbour_goes_where_it_is() -> None:
    """The reference VOD's hand 0: the kan indicator first reads like the dora; nearness
    decides which is new.
    """
    views = [view(t=t, inds=[ind("6m", 300)]) for t in range(0, 60, 6)]
    views += [view(t=60, inds=[ind("6m", 200), ind("6m", 300)])]
    views += [
        view(t=t, inds=[ind("6z", 200), ind("6m", 300)]) for t in range(66, 90, 6)
    ]
    row = indicator_row({"pond:TR": views}, -1, 100)
    assert [v["tile"] for v in row] == ["6m", "6z"]
