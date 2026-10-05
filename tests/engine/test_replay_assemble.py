# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay actual turn interleaving and verify written result blocks."""

from __future__ import annotations

from collections.abc import Sequence

from video2tenhou import tenhou6 as t
from video2tenhou.engine.assemble import _yaku_text, score_text, tenhou_deltas
from video2tenhou.engine.scoring import (
    ScoreResult,
    WinContext,
    Yaku,
    matches_site,
    score_hand,
)
from video2tenhou.record import HandResult


def kyoku(
    haipai: Sequence[Sequence[int]],
    draws: Sequence[Sequence[int | str]],
    discards: Sequence[Sequence[int | str]],
    result: list,
    dora: tuple[int] = (21,),
) -> list:
    """Build a positional Tenhou hand from synthetic tile streams."""
    k = [[0, 0, 0], [25000] * 4, list(dora), []]
    for i in range(4):
        k += [list(haipai[i]), list(draws[i]), list(discards[i])]
    return [*k, result]


# seat 1 waits on 5p (tanki); seat 0 draws 5p and discards it at once; seat 1 rons
HAIPAI = [
    t.tiles("111222333444s1z"),
    t.tiles("123456789m123p5p"),
    t.tiles("555666777888s2z"),
    t.tiles("999s3334445556z"),
]
RON = ["和了", [-1000, 1000, 0, 0], [1, 0, 1, "30符1飜1000点"]]


def test_a_legal_ron_replays_clean() -> None:
    k = kyoku(HAIPAI, [[25], [], [], []], [[60], [], [], []], RON)
    assert t.replay_kyoku(k) == []


def test_the_replayer_catches_what_the_per_seat_check_missed() -> None:
    # a discard after the hand ended (the ron): never played
    k = kyoku(HAIPAI, [[25], [], [35], []], [[60], [], [60], []], RON)
    assert "unplayed" in {v.kind for v in t.replay_kyoku(k)}
    # after riichi every discard is the drawn tile
    k = kyoku(
        HAIPAI,
        [[31, 25], [41], [42], [43]],
        [["r60", 11], [60], [60], [60]],
        ["流局", [0, 0, 0, 0]],
    )
    assert "riichi_discard" in {v.kind for v in t.replay_kyoku(k)}
    # deltas that do not balance, and a winner who does not hold a winning hand
    k = kyoku(
        HAIPAI,
        [[25], [], [], []],
        [[60], [], [], []],
        ["和了", [-1000, 2000, 0, 0], [1, 0, 1, "30符1飜1000点"]],
    )
    assert "deltas" in {v.kind for v in t.replay_kyoku(k)}
    k = kyoku(HAIPAI, [[26], [], [], []], [[60], [], [], []], RON)
    assert "not_winning" in {v.kind for v in t.replay_kyoku(k)}
    # a fifth 1s among the hands, the draws and the indicators
    k = kyoku(
        HAIPAI,
        [[31], [], [], []],
        [[60], [], [], []],
        ["流局", [0, 0, 0, 0]],
        dora=(31,),
    )
    assert ("over_count", "1s") in {(v.kind, v.tile) for v in t.replay_kyoku(k)}


def test_a_chi_only_from_the_kamicha() -> None:
    # seat 1 chis seat 0's 3p (its kamicha): it discards next without drawing
    haipai = [
        t.tiles("111222333444s1z"),
        t.tiles("123456789m12p55p"),
        t.tiles("555666777888s2z"),
        t.tiles("999s3334445556z"),
    ]
    ok = kyoku(
        haipai,
        [[23], ["c232122"], [], []],
        [[60], [11], [], []],
        ["流局", [0, 0, 0, 0]],
    )
    assert not {"unplayed", "out_of_turn"} & {v.kind for v in t.replay_kyoku(ok)}
    # the same chi string in seat 2's list: seat 0 is not its kamicha, so it is no call
    # on that discard
    bad = kyoku(
        haipai,
        [[23], [], ["c232122"], []],
        [[60], [], [11], []],
        ["流局", [0, 0, 0, 0]],
    )
    assert {"unplayed", "out_of_turn"} & {v.kind for v in t.replay_kyoku(bad)}


def test_the_replayer_checks_the_shape_of_every_call() -> None:
    """A call whose tiles do not form its meld is rejected whatever the hand holds."""
    haipai = [
        t.tiles("111222333444s1z"),
        t.tiles("123456789m12p55p"),
        t.tiles("555666777888s2z"),
        t.tiles("999s3334445556z"),
    ]

    def replayed(chi: str) -> set[str]:
        violations = t.replay_kyoku(
            kyoku(
                haipai,
                [[23], [chi], [], []],
                [[60], [11], [], []],
                ["流局", [0, 0, 0, 0]],
            )
        )
        return {v.kind for v in violations}

    assert "bad_meld" in replayed("c232125")
    assert "bad_meld" not in replayed("c232122")
    # a red five is a five
    assert "bad_meld" not in replayed("c232452")
    assert t._shape_ok("p", [15, 51, 15])
    assert not t._shape_ok("p", [15, 16, 15])
    assert not t._shape_ok("c", [41, 42, 43])
    assert not t._shape_ok("c", [18, 19, 21])
    assert t._shape_ok("a", t._parse_call(t.ankan(15, has_aka=True))[1])
    assert not t._shape_ok("m", [15, 15, 15])


def test_value_text_in_tenhou_form() -> None:
    assert score_text(2, 30, dealer=False, tsumo=False) == "30符2飜2000点"
    assert score_text(2, 30, dealer=False, tsumo=True) == "30符2飜500-1000点"
    assert score_text(2, 30, dealer=True, tsumo=True) == "30符2飜1000点∀"
    # kiriage mangan
    assert score_text(4, 30, dealer=False, tsumo=False) == "満貫8000点"
    assert score_text(6, 30, dealer=True, tsumo=True) == "跳満6000点∀"


def test_the_deposits_leave_the_deltas() -> None:
    """The site charges a declarer's 1000 in the hand's deltas; the viewer charges it at
    the r discard.
    """
    r = HandResult(
        0,
        0,
        0,
        {"EAST": -2000, "SOUTH": 5000, "WEST": -2000, "NORTH": -1000},
        "tsumo",
        winner="SOUTH",
        han=3,
        fu=30,
        riichi=["WEST"],
    )
    assert tenhou_deltas(r) == [-2000, 5000, -1000, -1000]


def test_yaku_names_of_the_scoring_library() -> None:
    assert _yaku_text(
        [
            {"name": "Yakuhai (seat wind east)", "han": 1},
            {"name": "Riichi", "han": 1},
            {"name": "Dora", "han": 2},
            {"name": "Double Riichi", "han": 2},
            {"name": "Suu Ankou", "han": 13},
        ]
    ) == ["自風 東(1飜)", "立直(1飜)", "ドラ(2飜)", "両立直(2飜)", "四暗刻(役満)"]


def test_scored_yaku_keep_their_library_names() -> None:
    """Scored yaku carry the library's name and han, with no count in the name."""
    result = score_hand(
        ["2m", "3m", "4m", "5p", "6p", "7p", "3s", "4s", "5s", "6s", "7s", "8s", "9p"],
        "9p",
        [],
        WinContext(
            tsumo=False, riichi=True, seat="S", round_wind="E", dora=["2s"], ura=[]
        ),
    )
    assert result.ok
    assert {(y.name, y.han) for y in result.yaku} >= {("Riichi", 1), ("Dora", 1)}
    assert str(Yaku("Dora", 1)) == "Dora (1)"


def test_the_site_fu_counts_below_yakuman() -> None:
    """Reference hand 18: a 6/20 pinfu tsumo reconstructed as a 6/30 hand scores the
    same haneman, but the site records the fu, and a different fu means a different
    hand. Only a yakuman's fu means nothing.
    """
    assert matches_site(ScoreResult(ok=True, han=6, fu=20), 6, 20)
    assert not matches_site(ScoreResult(ok=True, han=6, fu=30), 6, 20)
    assert matches_site(ScoreResult(ok=True, han=13, fu=50), 13, 40)
