# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Region tile grouping, border rejection and row geometry."""

from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.perception.reader import Box, assign_hand, assign_meld, assign_pond
from video2tenhou.train.data import CLASS_INDEX, CLASSES

if TYPE_CHECKING:
    from collections.abc import Callable


def box(
    x: "float",
    y: "float",
    w: int = 40,
    *,
    sideways: bool = False,
    conf: float = 0.9,
) -> "Box":
    """Create a located reading box with configurable size and orientation."""
    h = 60
    if sideways:
        w, h = h, w
    return Box(
        (x, y, x + w, y + h), conf, sideways=sideways, p=np.zeros(39, np.float32)
    )


def face(
    x: "float",
    y: "float",
    size: tuple[int, int] = (105, 150),
    tile: str = "5m",
    *,
    sideways: bool = False,
) -> "Box":
    """Create a face-up box with a specified tile posterior."""
    w, h = size
    p = np.full(len(CLASSES), 0.001)
    p[CLASS_INDEX[tile]] = 0.9
    return Box((x, y, x + w, y + h), 0.9, sideways=sideways, p=p)


def back(x: "float", y: "float", w: int = 105, h: int = 90) -> "Box":
    """Create a face-down box with a certain back-class posterior."""
    p = np.zeros(len(CLASSES))
    p[CLASS_INDEX["X"]] = 1.0
    return Box((x, y, x + w, y + h), 0.9, sideways=False, p=p)


def test_pond_rows_cols_gaps_and_indicator() -> None:
    """Verify pond rows cols gaps and indicator."""
    # row 0: six tiles, row 1: three tiles laid unevenly (a called-away tile leaves no
    # gap), one sideways
    boxes = [box(10 + c * 42, 10) for c in range(6)]
    boxes.extend(box(12 + c * 42, 75) for c in (0, 1))
    boxes.append(box(12 + 3 * 42, 80, sideways=True))
    # dead wall indicator far below the rows
    boxes.append(box(120, 300))
    assert assign_pond(boxes) is False
    rows = [(b.row, b.col) for b in boxes if b.role == "tile"]
    assert rows == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert boxes[-1].role == "indicator"
    assert boxes[-1].row is None


def test_meld_groups() -> None:
    """Verify meld groups."""
    boxes = [
        box(0, 0),
        box(42, 0),
        box(84, 0, sideways=True),  # one meld: three tiles, the called one sideways
        box(200, 0),
        box(242, 0),
        box(284, 0),  # second meld further right on the same row
        box(0, 100),
        box(42, 100),
        box(84, 100),
    ]  # third meld on the next row
    assign_meld(boxes)
    assert [b.group for b in boxes] == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_wall_row_never_becomes_a_discard_row() -> None:
    """Verify wall row never becomes a discard row."""
    image_height = 630
    boxes = []
    for r in range(3):  # three full discard rows, the third one turned tile (riichi)
        boxes.extend(
            face(
                300 + c * 110,
                20 + r * 155,
                tile="1m" if r else "9p",
                sideways=r == 2 and c == 1,
            )
            if not (r == 2 and c == 1)
            else face(300 + c * 110, 20 + r * 155, sideways=True, size=(150, 105))
            for c in range(6)
        )
    # the wall row cut by the far edge: 14 backs and one face-up tile (the indicator),
    # all clipped
    boxes.extend(back(150 + c * 108, 540) for c in range(14))
    boxes.append(face(150 + 14 * 108, 540, tile="3p", size=(105, 90)))
    assign_pond(boxes, region_h=image_height)
    rows = [b for b in boxes if b.role == "tile"]
    assert len(rows) == 18
    assert {b.row for b in rows} == {0, 1, 2}
    assert any(b.sideways and b.row == 2 for b in rows)
    ind = [b for b in boxes if b.role == "indicator"]
    assert len(ind) == 15
    assert all(not b.sideways for b in ind)


def test_rule_of_six_drops_a_phantom_box_instead_of_the_sixth_tile() -> None:
    # six real tiles and a low-confidence second box on the third one: without the
    # repair the phantom takes a
    # column and the real sixth tile falls out of the row
    """Verify rule of six drops a phantom box instead of the sixth tile."""
    boxes = [box(100 + c * 42, 10) for c in range(6)]
    boxes.append(box(100 + 2 * 42 + 6, 12, conf=0.3))
    assert assign_pond(boxes, region_h=700, region_w=400) is False
    assert [(b.row, b.col) for b in boxes[:6]] == [(0, c) for c in range(6)]
    assert boxes[6].role == "other"
    assert boxes[6].row is None


def test_rule_of_six_rejects_a_row_of_seven_away_from_the_edges() -> None:
    """Verify rule of six rejects a row of seven away from the edges."""
    boxes = [
        box(50 + c * 42, 10) for c in range(7)
    ]  # seven distinct boxes, none overlapping, none at an edge
    assert assign_pond(boxes, region_h=700, region_w=400) is True
    assert all(b.role == "other" and b.row is None for b in boxes)


def test_rule_of_six_at_the_side_edge_is_the_neighbouring_pond() -> None:
    # the seventh box touches the right edge of the region: the neighbour's tile, the
    # row keeps its six
    """Verify rule of six at the side edge is the neighbouring pond."""
    boxes = [box(100 + c * 42, 10) for c in range(6)] + [box(362, 10, w=38)]
    assert assign_pond(boxes, region_h=700, region_w=400) is False
    assert [(b.row, b.col) for b in boxes[:6]] == [(0, c) for c in range(6)]
    assert boxes[6].role == "other"


def test_rule_of_six_at_the_left_edge_drops_the_left_box() -> None:
    # the reference VOD's hand 4 pond:BR: the neighbour's tile touches the LEFT edge;
    # the owner's six stay
    """Verify rule of six at the left edge drops the left box."""
    boxes = [box(0, 10, w=38)] + [box(60 + c * 42, 10) for c in range(6)]
    assert assign_pond(boxes, region_h=700, region_w=400) is False
    assert boxes[0].role == "other"
    assert [(b.row, b.col) for b in boxes[1:]] == [(0, c) for c in range(6)]


def test_meld_duplicate_box_and_overlapping_rows() -> None:
    # the second 4m of "2m 4m 4m 3m" is the same tile boxed twice: it goes, the chi
    # keeps three boxes
    """Verify meld duplicate box and overlapping rows."""
    boxes = [box(0, 0, sideways=True), box(62, 0), box(64, 2, conf=0.5), box(104, 0)]
    assign_meld(boxes)
    assert boxes[2].role == "other"
    assert [b.group for b in boxes if b.role != "other"] == [0, 0, 0]
    # two boxes over the same x at slightly different heights are two rows, not one run
    boxes = [box(0, 0), box(42, 0), box(84, 0), box(10, 30), box(52, 30), box(94, 30)]
    assign_meld(boxes)
    assert len({b.group for b in boxes[:3]}) == 1
    assert len({b.group for b in boxes[3:]}) == 1
    assert boxes[0].group != boxes[3].group


def test_side_sliver_uses_the_region_width() -> None:
    # three tiles ending at x = 374 and a 10 px box at 390-400: at the right edge of a
    # 400 px region it is a
    # sliver of the neighbouring pond; in a wider region the same box is a (narrow)
    # fourth tile of the row
    """Verify side sliver uses the region width."""
    boxes = [box(250 + c * 42, 10) for c in range(3)] + [box(390, 12, w=10)]
    assign_pond(boxes, region_h=700, region_w=400)
    assert boxes[3].role == "other"
    boxes = [box(250 + c * 42, 10) for c in range(3)] + [box(390, 12, w=10)]
    assign_pond(boxes, region_h=700, region_w=1080)
    assert boxes[3].role == "tile"
    assert (boxes[3].row, boxes[3].col) == (0, 3)


def test_fourth_row_in_the_wall_band_is_a_row_when_it_continues_the_rows() -> None:
    """Verify fourth row in the wall band is a row when it continues the rows."""
    image_height = 700  # band starts at 476; tiles 150 tall, row centres 95, 250, 405
    th = 150

    def pond(spacing: "float", fourth_row: "Callable[[int], list[Box]]") -> "list[Box]":
        boxes = [
            face(300 + c * 110, 20 + r * 155, tile="1m")
            for r in range(3)
            for c in range(6)
        ]
        boxes += fourth_row(405 + int(spacing * th))
        return boxes

    # a fourth discard row continues the rows at the usual spacing, even inside the band
    boxes = pond(
        1.1, lambda cy: [face(300 + c * 110, cy - 75, tile="7s") for c in range(3)]
    )
    assign_pond(boxes, region_h=image_height)
    assert [(b.row, b.col) for b in boxes if b.role == "tile"][-3:] == [
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    # one or two face-up tiles at that spacing are dora indicators, not the start of a
    # fourth row
    boxes = pond(
        1.1, lambda cy: [face(300 + c * 110, cy - 75, tile="7s") for c in range(2)]
    )
    assign_pond(boxes, region_h=image_height)
    assert [b.role for b in boxes[-2:]] == ["indicator", "indicator"]
    # a cluster clearly below the rows is the dead wall, whether or not its face-down
    # tiles were detected
    for fourth in (
        lambda cy: [face(300 + c * 110, cy - 75, tile="7s") for c in range(2)],
        lambda cy: [back(300, cy - 50, h=100), face(410, cy - 75, tile="7s")],
    ):
        boxes = pond(1.5, fourth)
        assign_pond(boxes, region_h=image_height)
        assert sum(1 for b in boxes if b.role == "tile") == 18
        assert [b.role for b in boxes[-2:]] == ["indicator", "indicator"]


def test_hand_row_separated_from_a_meld_beside_it() -> None:
    # a revealed hand: ten tiles in the row and a pon of three moved beside it, slightly
    # lower and to the right
    """Verify hand row separated from a meld beside it."""
    boxes = [box(20 + i * 44, 40) for i in range(10)] + [
        box(560 + i * 44, 52) for i in range(3)
    ]
    assign_hand(boxes)
    assert [b.role for b in boxes] == ["tile"] * 10 + ["extra"] * 3
    assert [b.group for b in boxes] == [0] * 10 + [1] * 3
    # a plain 13-tile row is one group
    boxes = [box(20 + i * 44, 40) for i in range(13)]
    assign_hand(boxes)
    assert all(b.role == "tile" and b.group == 0 for b in boxes)
    # the drawn tile set apart at the right end stays in the row: it is one tile, a meld
    # is three or four
    boxes = [box(20 + i * 44, 40) for i in range(13)] + [box(20 + 13 * 44 + 30, 40)]
    assign_hand(boxes)
    assert sum(1 for b in boxes if b.role == "tile") == 13
    assert boxes[-1].role == "extra"
