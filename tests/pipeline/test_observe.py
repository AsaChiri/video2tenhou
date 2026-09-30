from copy import deepcopy

import numpy as np

from video2tenhou.calm import Interval
from video2tenhou.observe import observe, observe_interval
from video2tenhou.train.data import CLASS_INDEX, CLASSES


def post(tile, noise=None):
    p = np.full(len(CLASSES), 0.001)
    p[CLASS_INDEX[tile]] = 0.9
    if noise:
        p[CLASS_INDEX[noise]] = 0.5
    p /= p.sum()
    return p.tolist()


def pbox(tile, row, col, conf=0.9, sideways=False, role="tile", noise=None):
    """A box at a realistic place: rows 65 px apart from the top, columns 42 px apart; an indicator
    (row 9) lies in the wall band at the bottom of a 700 px tall region.
    """
    x, y = 10 + col * 42, 10 + row * 65
    w, h = (60, 40) if sideways else (40, 60)
    d = {
        "xyxy": [x, y, x + w, y + h],
        "conf": conf,
        "sideways": sideways,
        "role": role,
        "p": post(tile, noise),
    }
    if role == "tile":
        d.update(row=row, col=col)
    return d


def hbox(tile, x, y=40, conf=0.9):
    return {
        "xyxy": [x, y, x + 40, y + 60],
        "conf": conf,
        "sideways": False,
        "role": "tile",
        "p": post(tile),
    }


def reading(t, boxes, region="pond:TL", size=(400, 700)):
    return {"t": t, "region": region, "size": list(size), "boxes": boxes}


def test_pond_observation_votes_and_flags_disagreement():
    iv = Interval("pond:TL", 0.0, 2.0, 5, True, 0.0, 0.0)
    rs = [
        reading(
            0.0,
            [
                pbox("1m", 0, 0),
                pbox("2p", 0, 1),
                pbox("7z", 0, 2, sideways=True),
                pbox("5z", 9, 9, role="indicator"),
            ],
        ),
        reading(
            0.5,
            [
                pbox("1m", 0, 0),
                pbox("2p", 0, 1, noise="3p"),
                pbox("7z", 0, 2, sideways=True),
                pbox("5z", 9, 9, role="indicator"),
            ],
        ),
        reading(
            1.0,
            [
                pbox("1m", 0, 0),
                pbox("3p", 0, 1),
                pbox("7z", 0, 2, sideways=True),
                pbox("5z", 9, 9, role="indicator"),
            ],
        ),
        reading(
            1.5, [pbox("1m", 0, 0), pbox("2p", 0, 1)]
        ),  # an arm hid a tile: count differs -> dropped
        reading(
            2.0,
            [
                pbox("1m", 0, 0),
                pbox("2p", 0, 1),
                pbox("7z", 0, 2, sideways=True),
                pbox("5z", 9, 9, role="indicator"),
            ],
        ),
    ]
    o = observe("pond:TL", rs, iv)
    assert o.n_readings == 5 and o.n_used == 4 and o.count == 3
    assert [s.key for s in o.slots] == [(0, 0), (0, 1), (0, 2)]
    assert o.slots[0].tile == "1m" and not o.slots[0].disagree
    assert (
        o.slots[1].tile == "2p" and o.slots[1].disagree
    )  # one reading said 3p: flagged, still 2p by vote
    assert o.slots[2].sideways == 1.0
    assert len(o.indicators) == 1 and o.indicators[0].tile == "5z"


def test_pond_reading_with_seven_in_a_row_is_rejected():
    iv = Interval("pond:TL", 0.0, 0.5, 2, True, 0.0, 0.0)
    seven = [pbox("1m", 0, c) for c in range(7)]
    six = [pbox("1m", 0, c) for c in range(6)]
    rs = [reading(0.0, seven), reading(0.5, six)]
    raw = deepcopy(rs)
    o = observe("pond:TL", rs, iv)
    assert rs == raw  # Retention/rejection must not mutate reusable raw evidence.
    assert o.count == 6 and o.n_used == 1 and len(o.slots) == 6
    # every reading rejected: an empty observation, never a truncated row
    o = observe("pond:TL", [reading(0.0, seven), reading(0.5, seven)], iv)
    assert o.n_readings == 2 and o.n_used == 0 and o.count == 0 and not o.slots


def test_observation_spans_the_readings_not_the_interval():
    # the calm interval runs 0-10 s but the hand window clipped the reads to 4-8 s
    iv = Interval("pond:TL", 0.0, 10.0, 21, True, 0.0, 0.0)
    rs = [reading(t, [pbox("1m", 0, 0)]) for t in (4.0, 6.0, 8.0)]
    o = observe("pond:TL", rs, iv)
    assert (o.t0, o.t1) == (4.0, 8.0) and (o.iv_t0, o.iv_t1) == (0.0, 10.0)
    d = o.to_dict()
    assert (
        d["t0"] == 4.0 and d["t1"] == 8.0 and d["iv_t0"] == 0.0 and d["iv_t1"] == 10.0
    )
    # with no reading the interval bounds stand
    o = observe("pond:TL", [], iv)
    assert (o.t0, o.t1) == (0.0, 10.0)


def test_hand_observation_keeps_a_meld_beside_the_row_apart():
    iv = Interval("hand:TL", 0.0, 1.0, 3, True, 0.0, 0.0)
    row = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z"]
    pon = ["5z", "5z", "5z"]

    def frame(t):
        boxes = [hbox(tile, 20 + i * 44) for i, tile in enumerate(row)]
        boxes += [hbox(tile, 560 + i * 44, y=52) for i, tile in enumerate(pon)]
        return reading(t, boxes, region="hand:TL", size=(720, 340))

    o = observe("hand:TL", [frame(0.0), frame(0.5), frame(1.0)], iv)
    assert o.count == 10 and o.n_used == 3
    assert [s.tile for s in o.slots] == row
    assert len(o.extra) == 1 and [s.tile for s in o.extra[0]] == pon
    d = o.to_dict()
    assert len(d["extra"]) == 1 and [s["key"] for s in d["extra"][0]] == [
        ["extra", 1, 0],
        ["extra", 1, 1],
        ["extra", 1, 2],
    ]


def test_a_quick_discard_splits_the_interval():
    iv = Interval("pond:TL", 0.0, 3.0, 7, True, 0.0, 0.0)
    two = [pbox("1m", 0, 0), pbox("2p", 0, 1)]
    three = two + [pbox("7z", 0, 2)]
    rs = [
        reading(0.0, two),
        reading(0.5, two),
        reading(1.0, two),
        reading(1.5, three),
        reading(2.0, three),
        reading(2.5, three),
    ]
    a, b = observe_interval("pond:TL", rs, iv)
    assert (a.count, a.t0, a.t1) == (2, 0.0, 1.0) and (b.count, b.t0, b.t1) == (
        3,
        1.5,
        2.5,
    )
    # a single odd reading is noise: one observation, the mode
    rs = [reading(0.0, two), reading(0.5, two), reading(1.0, three), reading(1.5, two)]
    (o,) = observe_interval("pond:TL", rs, iv)
    assert o.count == 2 and o.n_used == 3


def test_interval_split_keeps_rejected_counts_and_partial_provenance():
    """Reusing prepared evidence must preserve both vote and review metadata."""
    iv = Interval("pond:TL", 0.0, 3.0, 7, True, 0.0, 0.0, partial=True)
    two = [pbox("1m", 0, 0), pbox("2p", 0, 1)]
    three = two + [pbox("7z", 0, 2)]
    invalid = [pbox("1m", 0, c) for c in range(7)]
    rows = [
        reading(0.0, two),
        reading(0.5, two),
        reading(1.0, invalid),
        reading(1.5, three),
        reading(2.0, three),
    ]
    a, b = observe_interval("pond:TL", rows, iv)
    assert (a.count, b.count) == (2, 3)
    assert a.partial and b.partial
    assert (a.iv_t0, a.iv_t1, b.iv_t0, b.iv_t1) == (0.0, 0.5, 1.5, 3.0)
    assert (a.n_readings, a.n_used, b.n_readings, b.n_used) == (2, 2, 2, 2)
