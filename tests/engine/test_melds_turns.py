# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Legal meld decoding and chronological call/turn reconstruction."""

from __future__ import annotations

from tests import builders
from tests.builders import observation, posterior
from video2tenhou.engine import rules
from video2tenhou.engine.calls import CallAnchor
from video2tenhou.engine.dense import DenseContext
from video2tenhou.engine.melds import (
    Call,
    decode_group,
    fragments,
    meld_options,
    read_views,
    split_group,
    track_melds,
)
from video2tenhou.engine.ponds import PondSlot, insert_slot, next_slot_id, virtual_slot
from video2tenhou.engine.turns import Dropped, HandEnding, Skip, WrongEnd, merge


def slot(
    tile: str,
    group: int,
    i: int,
    sideways: float = 0.0,
    second: str | None = None,
) -> dict:
    """Create a meld box with group position and an optional competing tile."""
    return builders.slot(
        tile,
        key=[group, i],
        xyxy=[i * 40, group * 70, i * 40 + 38, group * 70 + 58],
        p=posterior(tile, 0.8, second=second, second_peak=0.3),
        sideways=sideways,
    )


def obs(t0: float, t1: float, slots: list[dict]) -> dict:
    """Create a meld-camera observation of the supplied boxes."""
    return observation("meld:TL", t0, t1, slots)


def test_rules() -> None:
    assert rules.relative("E", "N") == "kamicha"
    assert rules.relative("E", "S") == "shimocha"
    assert rules.relative("E", "W") == "toimen"
    assert all(
        rules.relative(me, rules.seat_at(me, rel)) == rel
        for me in rules.SEATS
        for rel in ("kamicha", "toimen", "shimocha")
    )
    assert rules.count_ok(["5m"] * 3 + ["0m"])
    assert not rules.count_ok(["5m"] * 4)
    assert rules.kan_tiles("0p") == ["5p", "5p", "5p", "0p"]
    assert {rules.OTHER_FIVE[rules.OTHER_FIVE[t]] for t in ("5s", "0s")} == {"5s", "0s"}


def test_decode_group_pon_rule_repairs_one_tile() -> None:
    g = [
        slot("9p", 0, 0),
        slot("9p", 0, 1, sideways=1.0),
        slot("5s", 0, 2, second="9p"),
    ]
    m = decode_group(g)
    assert m is not None
    assert m.type == "pon"
    assert m.tiles == ["9p", "9p", "9p"]
    assert m.called_pos == 1
    m = decode_group(
        [slot("3s", 0, 0, sideways=1.0), slot("4s", 0, 1), slot("5s", 0, 2)]
    )
    assert m is not None
    assert m.type == "chi"
    assert m.called_pos == 0
    assert decode_group([slot("1m", 0, 0), slot("5p", 0, 1), slot("9s", 0, 2)]) is None


def test_four_boxes_are_a_kan_only_when_all_four_read_so() -> None:
    """The second VOD's hand 22: the chi 2m-3m-4m with its 4m boxed twice was read as
    four 4m.
    """
    m = decode_group(
        [
            slot("2m", 0, 0, sideways=1.0),
            slot("4m", 0, 1),
            slot("4m", 0, 2),
            slot("3m", 0, 3),
        ]
    )
    assert m is not None
    assert m.type == "chi"
    assert sorted(m.tiles) == ["2m", "3m", "4m"]
    assert m.called_pos is not None
    assert m.tiles[m.called_pos] == "2m"
    # the reference VOD's hand 7: a pon of 9m beside the 7z of the next meld
    m = decode_group(
        [
            slot("9m", 0, 0),
            slot("9m", 0, 1),
            slot("7z", 0, 2),
            slot("9m", 0, 3, sideways=1.0),
        ]
    )
    assert m is not None
    assert m.type == "pon"
    assert m.tiles == ["9m", "9m", "9m"]
    m = decode_group(
        [
            slot("6z", 0, 0, sideways=1.0),
            slot("6z", 0, 1),
            slot("6z", 0, 2),
            slot("6z", 0, 3),
        ]
    )
    assert m is not None
    assert m.type == "kan"
    assert m.called_pos == 0


def test_a_long_run_of_boxes_is_several_melds() -> None:
    run = [
        slot("8p", 0, 0, second="9p"),
        slot("8p", 0, 1),
        slot("7p", 0, 2),
        slot("6s", 0, 3, sideways=1.0),
        slot("8s", 0, 4),
        slot("7s", 0, 5),
    ]
    melds = split_group(run)
    assert [m.type for m in melds] == ["chi", "chi"]
    second = melds[1]
    assert second.called_pos is not None
    assert second.tiles[second.called_pos] == "6s"


def test_a_known_meld_with_a_stray_box_is_the_same_meld() -> None:
    chi = [slot("2m", 0, 0, sideways=1.0), slot("4m", 0, 1), slot("3m", 0, 2)]
    seq = [
        obs(0, 5, []),
        obs(10, 14, chi),
        obs(20, 24, chi),
        obs(
            30,
            34,
            [
                slot("7z", 0, 0),
                slot("7z", 0, 1),
                slot("7z", 0, 2, sideways=1.0),
                slot("2m", 1, 0, sideways=1.0),
                slot("4m", 1, 1),
                slot("4m", 1, 2),
                slot("3m", 1, 3),
            ],
        ),
        obs(
            40,
            44,
            [
                slot("7z", 0, 0),
                slot("7z", 0, 1),
                slot("7z", 0, 2, sideways=1.0),
                slot("2m", 1, 0, sideways=1.0),
                slot("4m", 1, 1),
                slot("4m", 1, 2),
                slot("3m", 1, 3),
            ],
        ),
    ]
    calls = track_melds("S", read_views(seq))
    assert sorted(c.type for c in calls) == ["chi", "pon"]


def test_track_melds_sources_and_kakan() -> None:
    seq = [
        obs(0, 5, []),
        obs(
            10, 14, [slot("7z", 0, 0), slot("7z", 0, 1), slot("7z", 0, 2, sideways=1.0)]
        ),
        obs(
            20, 24, [slot("7z", 0, 0), slot("7z", 0, 1), slot("7z", 0, 2, sideways=1.0)]
        ),
        obs(
            30,
            34,
            [
                slot("7z", 0, 0),
                slot("7z", 0, 1),
                slot("7z", 0, 2, sideways=1.0),
                slot("7z", 0, 3),
                slot("2m", 1, 0, sideways=1.0),
                slot("3m", 1, 1),
                slot("4m", 1, 2),
            ],
        ),
        obs(
            40,
            44,
            [
                slot("7z", 0, 0),
                slot("7z", 0, 1),
                slot("7z", 0, 2, sideways=1.0),
                slot("7z", 0, 3),
                slot("2m", 1, 0, sideways=1.0),
                slot("3m", 1, 1),
                slot("4m", 1, 2),
            ],
        ),
    ]
    calls = track_melds("E", read_views(seq))
    # the pon stays (its called tile fixes a turn); the kakan is a second event timed by
    # its fourth tile
    assert sorted(c.type for c in calls) == ["chi", "kakan", "pon"]
    pon, chi, kakan = (
        next(c for c in calls if c.type == t) for t in ("pon", "chi", "kakan")
    )
    assert pon.source == "shimocha"
    assert pon.t_window == (5, 10)
    assert chi.source == "kamicha"
    assert chi.called_tile == "2m"
    assert chi.t_first == 30
    assert kakan.t_first == 30
    assert kakan.t_window == (24, 30)
    assert kakan.source == "shimocha"
    assert len(kakan.tiles) == 4


def pslot(
    i: int,
    tile: str,
    t: float,
    removed: float | None = None,
    *,
    side: bool = False,
) -> PondSlot:
    """Create a pond slot with controlled tile support, timing and removal."""
    p = posterior(tile, normalized=False)
    # sideways accumulates one vote per observation
    s = PondSlot(i, 0, i, p, t, (t - 3, t), t + 4, 3, 3.0 if side else 0.0)
    s.t_removed = removed
    return s


def test_merge_follows_rotation_and_calls() -> None:
    logs = {
        "E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
        "S": [pslot(2, "3p", 20, removed=27), pslot(3, "4p", 60)],
        "W": [pslot(4, "5s", 30, side=True), pslot(5, "6s", 70)],
        "N": [pslot(6, "7z", 40)],
    }
    # W pons S's 3p at ~27: W discards next (no draw), then N, E, S, W, ...
    pon = Call(
        seat="W",
        t_first=28,
        t_window=(24, 28),
        type="pon",
        tiles=["3p", "3p", "3p"],
        called_pos=0,
        source="kamicha",
        called_tile="3p",
    )
    merged = merge(logs, [pon], "E")
    turns = merged.turns
    assert merged.findings == []
    assert [(t.seat, t.kind) for t in turns] == [
        ("E", "draw"),
        ("S", "draw"),
        ("W", "call"),
        ("N", "draw"),
        ("E", "draw"),
        ("S", "draw"),
        ("W", "draw"),
    ]
    assert turns[1].call is pon
    assert turns[2].riichi


def test_merge_drops_a_phantom_instead_of_dragging_the_hand() -> None:
    """One slot at a time no discard could have (a tile of the next hand, a misread): it
    costs one drop, and every other turn keeps its place.
    """
    logs = {
        "E": [pslot(0, "1m", 10), pslot(1, "2m", 50), pslot(8, "9p", 400)],
        "S": [pslot(2, "3p", 20), pslot(3, "4p", 60)],
        "W": [pslot(4, "5s", 30), pslot(5, "6s", 70)],
        "N": [pslot(6, "7z", 40), pslot(7, "6z", 80)],
    }
    # a tile laid far out of time, second in E's log
    logs["E"].insert(1, pslot(9, "1z", 390))
    merged = merge(logs, [], "E")
    assert [t.seat for t in merged.turns] == list("ESWNESWNE")
    assert Dropped("E", "1z", 390, after_end=False) in merged.findings
    assert any(
        "1z seen at 390s is not a discard of this hand" in f.text
        for f in merged.findings
    )


def test_a_seat_that_runs_out_of_discards_is_a_skip_not_a_free_pass() -> None:
    logs = {
        "E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
        "S": [pslot(2, "3p", 20), pslot(3, "4p", 60)],
        "W": [pslot(4, "5s", 30)],  # W's second discard was never read
        "N": [pslot(6, "7z", 40), pslot(7, "6z", 80)],
    }
    merged = merge(logs, [], "E")
    assert [t.seat for t in merged.turns] == list("ESWNESN")
    # W's turn came between S's discard at 60 and N's at 80
    assert merged.skips == [Skip(6, "W", after=60, before=80)]
    assert merged.skips[0].text == "turn 6: no discard of W found in its pond (skipped)"


def test_the_result_fixes_where_the_sequence_ends() -> None:
    """After a tsumo the winner is next to play: a trailing tile that puts another seat
    there is dropped, and a sequence that cannot end at the winner says so.
    """
    logs = {
        "E": [pslot(0, "1m", 10), pslot(1, "2m", 50)],
        "S": [
            pslot(2, "3p", 20),
            pslot(3, "4p", 60),
            pslot(8, "9m", 75),  # after the win, the reveal
        ],
        "W": [pslot(4, "5s", 30)],
        "N": [pslot(6, "7z", 40)],
    }
    merged = merge(logs, [], "E", ending=HandEnding(end="W"))
    assert [t.seat for t in merged.turns] == list("ESWNES")
    assert Dropped("S", "9m", 75, after_end=True) in merged.findings
    logs["S"].pop()
    merged = merge(logs, [], "E", ending=HandEnding(end="N"))
    assert WrongEnd("W", "N") in merged.findings
    assert any(
        "ends with W to play, but the result says N" in f.text for f in merged.findings
    )


def test_a_removal_no_call_took_is_a_hidden_call() -> None:
    """Hand 0 of the second VOD: S discards twice with nobody between; W's 2z had left
    W's pond. Every removal is a call: the turn order says S took it (a pon from its
    shimocha), which the camera missed.
    """
    logs = {
        "E": [pslot(0, "1m", 10), pslot(1, "2m", 90)],
        "S": [pslot(2, "3p", 20), pslot(3, "4p", 34), pslot(9, "5p", 100)],
        "W": [pslot(4, "2z", 28, removed=31), pslot(5, "5s", 60)],
        "N": [pslot(6, "7z", 70)],
    }
    merged = merge(logs, [], "E")
    turns = merged.turns
    assert [t.seat for t in turns][:4] == ["E", "S", "W", "S"]
    hidden = turns[2].call
    assert hidden is not None
    assert hidden.anchor == "hidden"
    assert hidden.seat == "S"
    assert hidden.source == "shimocha"
    assert turns[3].own_call is hidden
    assert not merged.skips


def test_meld_options_come_from_the_called_tile() -> None:
    """The second VOD's hand 23: a chi on the red five from the kamicha, read by the
    camera as 0p 4p 3p.
    """
    ps = [posterior(t, 0.8) for t in ["0p", "4p", "3p"]]
    best = meld_options("0p", "kamicha", ps, four=False)[0]
    assert best.type == "chi"
    assert sorted(best.hand) == ["3p", "4p"]
    assert best.cost == 0
    # from the toimen only a pon (or, with four boxes, a daiminkan) is possible
    assert {o.type for o in meld_options("4p", "toimen", ps, four=False)} == {"pon"}
    assert {o.type for o in meld_options("4p", "toimen", ps, four=True)} == {"kan"}


def test_fragments_are_pairs_of_a_meld_the_camera_does_not_complete() -> None:
    """Hand 0 of the second VOD: S's camera shows "2z 2z 3z" (no legal meld: a tile
    misread) and hand 3's shows "1s 1s" (the turned tile not boxed): both are pairs of a
    pon for the pond or the turn order to complete.
    """
    views = [
        obs(t, t + 3, [slot("2z", 0, 0), slot("2z", 0, 1), slot("3z", 0, 2)])
        for t in (10, 20, 30)
    ]
    views += [
        obs(
            t,
            t + 3,
            [
                slot("2z", 0, 0),
                slot("2z", 0, 1),
                slot("3z", 0, 2),
                slot("1s", 1, 0),
                slot("1s", 1, 1),
            ],
        )
        for t in (40, 50)
    ]
    frags = {
        tuple(f.tiles): (f.t_first, f.seen)
        for f in fragments("S", read_views(views), [])
    }
    assert frags == {("2z", "2z"): (10, 5), ("1s", "1s"): (40, 2)}
    # a pair that is part of a meld the seat is known to have is that meld, not a
    # fragment

    known = Call(
        seat="S",
        t_first=5,
        t_window=(0, 5),
        type="pon",
        tiles=["1s"] * 3,
        called_pos=2,
        source="shimocha",
        called_tile="1s",
    )
    assert [f.tiles for f in fragments("S", read_views(views), [known])] == [
        ["2z", "2z"]
    ]


def test_an_ankan_is_all_four_of_its_kind() -> None:
    """The second VOD's hand 3: N chi'd W's 5m, then N's camera showed a 5m 5m pair with
    nothing taken. An ankan of 5m would need a fifth five: the pair was the chi re-read,
    not a kan. A kan the camera saw whole whose kind a pond shows keeps its place and
    loses its kind.
    """
    logs = {
        "E": [pslot(0, "1m", 10)],
        "S": [pslot(1, "9p", 20)],
        "W": [pslot(2, "5m", 30, removed=33)],
        "N": [pslot(3, "7z", 40)],
    }
    ps = [posterior(t, 0.8) for t in ["5m", "4m", "6m"]]
    chi = Call(
        seat="N",
        t_first=35,
        t_window=(30, 35),
        type="chi",
        tiles=["5m", "4m", "6m"],
        called_pos=0,
        source="kamicha",
        called_tile="5m",
        p=ps,
        seen=5,
    )
    pair = Call(
        seat="N",
        t_first=70,
        t_window=(60, 70),
        type="fragment",
        tiles=["5m", "5m"],
        p=ps[:2],
        seen=3,
    )
    whole = Call(
        seat="E",
        t_first=80,
        t_window=(75, 80),
        type="ankan",
        tiles=["9p", "9p", "X", "X"],
        seen=3,
    )
    diagnostics: list[str] = []
    calls = CallAnchor(
        logs,
        context=DenseContext(
            entry={}, models=None, work_dir=None, t0=0, diagnostics=diagnostics
        ),
    ).anchor([chi, pair, whole], 10, 90, None)
    assert [(c.seat, c.type) for c in calls] == [("N", "chi"), ("E", "ankan")]
    assert calls[0].anchor == "discard"
    assert calls[0].source == "kamicha"
    assert calls[0].called_tile == "5m"
    assert calls[1].tiles[:2] == ["?", "?"]
    assert any("misread" in p for p in diagnostics)
    assert any("no kan" in p for p in diagnostics)


def test_an_exhaustive_draw_ends_with_the_last_draw() -> None:
    """The second VOD's hand 7: the ponds hold one discard more than the live wall
    allows; the last, seen once during the reveal, was laid after the hand ended. With a
    wall of five draws, the fifth discard ends it.
    """
    logs = {
        "E": [pslot(0, "1m", 10), pslot(4, "5m", 50)],
        "S": [pslot(1, "2m", 20), pslot(5, "6m", 60)],
        "W": [pslot(2, "3m", 30)],
        "N": [pslot(3, "4m", 40)],
    }
    merged = merge(logs, [], "E", ending=HandEnding(end=None, wall=5, exhaustive=True))
    assert [t.seat for t in merged.turns] == ["E", "S", "W", "N", "E"]
    assert Dropped("S", "6m", 60, after_end=True) in merged.findings
    # a call turn draws nothing: with S's pon of E's first discard the same five draws
    # reach S's second discard

    logs["E"][0].t_removed = 12
    pon = Call(
        seat="S",
        t_first=14,
        t_window=(10, 14),
        type="pon",
        tiles=["1m", "1m", "1m"],
        called_pos=0,
        source="kamicha",
        called_tile="1m",
        seen=5,
    )
    logs["N"][0].t_first = 55
    turns = merge(
        logs, [pon], "E", ending=HandEnding(end=None, wall=4, exhaustive=True)
    ).turns
    assert [(t.seat, t.kind) for t in turns] == [
        ("E", "draw"),
        ("S", "call"),
        ("W", "draw"),
        ("N", "draw"),
        ("E", "draw"),
    ]


def test_a_meld_takes_a_discard_of_its_own_suit() -> None:
    """The second VOD's hand 22: S's camera read 7z 7z 7z; E's 0m had just left its
    pond, N's 7z a little earlier. A pon is of one suit and the inset confuses numbers,
    not suits: the 7z is the called tile, not the 0m.
    """
    logs = {
        "E": [pslot(0, "0m", 30, removed=32)],
        "S": [pslot(1, "6s", 26)],
        "W": [],
        "N": [pslot(2, "7z", 18, removed=21)],
    }
    for sl in logs["N"] + logs["E"]:
        sl.t_last = sl.t_first + 1
    ps = [posterior("7z", 0.8) for _ in range(3)]
    pon = Call(
        seat="S",
        t_first=41,
        t_window=(36, 41),
        type="pon",
        tiles=["7z", "7z", "7z"],
        called_pos=0,
        p=ps,
        seen=12,
    )
    calls = CallAnchor(
        logs,
        context=DenseContext(entry={}, models=None, work_dir=None, t0=0),
    ).anchor([pon], 10, 90, None)
    assert [(c.type, c.called_tile, c.source) for c in calls] == [
        ("pon", "7z", "toimen")
    ]


def test_a_meld_the_hand_does_not_confirm_is_a_reread() -> None:
    """The second VOD's hand 19: E's hand fell by three tiles at its chi and held that
    afterwards; 110 s later the inset regrouped the same chi and a 2p flickered in N's
    pond. The hand holds the tiles of the melds already known: the second chi is the
    camera re-reading the first, and it takes nothing.
    """
    entry = {"corner_wind": {"TL": "E", "TR": "S", "BR": "W", "BL": "N"}}
    hand = [
        {
            "region": "hand:TL",
            "t0": t,
            "t1": t + 2,
            "n_used": 3,
            "count": c,
            "slots": [{}] * c,
        }
        for t, c in ((50, 13), (60, 13), (75, 10), (90, 10), (190, 10), (200, 10))
    ]
    logs = {
        "E": [],
        "S": [],
        "W": [],
        "N": [pslot(0, "2p", 66, removed=68), pslot(1, "2p", 176, removed=178)],
    }
    for sl in logs["N"]:
        sl.t_last = sl.t_first + 1
    ps = [posterior(t, 0.8) for t in ["2p", "3p", "4p"]]
    first = Call(
        seat="E",
        t_first=72,
        t_window=(65, 72),
        type="chi",
        tiles=["2p", "3p", "4p"],
        called_pos=0,
        p=ps,
        seen=12,
    )
    again = Call(
        seat="E",
        t_first=183,
        t_window=(175, 183),
        type="chi",
        tiles=["2p", "3p", "4p"],
        called_pos=0,
        p=ps,
        seen=12,
    )
    diagnostics: list[str] = []
    anchor = CallAnchor(
        logs,
        {"hand:TL": hand},
        context=DenseContext(
            entry=entry, models=None, work_dir=None, t0=0, diagnostics=diagnostics
        ),
    )
    calls = anchor.anchor([first, again], 10, 300, None)
    assert [(c.seat, c.type, round(c.t_first)) for c in calls] == [("E", "chi", 72)]
    assert sum("re-read" in p for p in diagnostics) == 1
    # asking again (as the decoder does for unanchored events) adds no second note
    assert anchor.reread(again, calls)
    assert sum("re-read" in p for p in diagnostics) == 1


def test_a_virtual_discard_stays_virtual_when_its_log_is_renumbered() -> None:
    """A reviewer's missing discard keeps its flag when a dense read renumbers the log.

    Positions are reassigned whenever a slot found later enters the log; the flag, not
    the position, says that no pond view showed the discard.
    """
    logs = {
        "E": [pslot(0, "1m", 10), pslot(1, "2m", 50), pslot(2, "3m", 90)],
        "S": [],
        "W": [pslot(3, "5s", 30), pslot(4, "6s", 70)],
        "N": [pslot(5, "7z", 40), pslot(6, "6z", 80)],
    }
    missing = virtual_slot(logs, "9p", 60, seen=2)
    assert missing.id == 7
    logs["S"].append(missing)
    insert_slot(logs["S"], pslot(next_slot_id(logs), "4p", 20))
    assert missing.virtual
    assert (missing.row, missing.index) == (0, 1)
    turns = merge(logs, [], "E").turns
    assert [t.seat for t in turns] == list("ESWNESWNE")
    assert [t.virtual for t in turns if t.seat == "S"] == [False, True]
