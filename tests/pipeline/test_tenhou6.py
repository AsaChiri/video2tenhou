# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Tenhou tile encoding, call notation and legal hand replay."""

from __future__ import annotations

import json
from urllib.parse import unquote

from video2tenhou import tenhou6 as t


def test_tile_parsing_and_roundtrip() -> None:
    assert t.tile("1m") == 11
    assert t.tile("9s") == 39
    assert t.tile("7z") == 47
    assert t.tile("0p") == 52
    assert t.tiles("123m0p77z") == [11, 12, 13, 52, 47, 47]
    for v in [11, 29, 35, 41, 47, 51, 52, 53]:
        assert t.tile(t.tile_str(v)) == v
    assert t.deaka(51) == 15
    assert t.deaka(47) == 47


def test_relative_seat() -> None:
    # seat 1's kamicha is seat 0, toimen seat 3, shimocha seat 2
    assert t.relative_seat(1, 0) == 0
    assert t.relative_seat(1, 3) == 1
    assert t.relative_seat(1, 2) == 2
    assert t.relative_seat(0, 3) == 0


def test_call_strings() -> None:
    assert t.chi(13, 11, 12) == "c131112"
    assert t.pon(25, 25, 52, rel=0) == "p252552"
    assert t.pon(25, 25, 52, rel=1) == "25p2552"
    assert t.pon(25, 25, 52, rel=2) == "2552p25"
    assert t.daiminkan(41, 41, 41, 41, rel=0) == "m41414141"
    assert t.daiminkan(41, 41, 41, 41, rel=1) == "41m414141"
    assert t.daiminkan(41, 41, 41, 41, rel=2) == "414141m41"
    assert t.ankan(33) == "333333a33"
    assert t.ankan(15, has_aka=True) == "511515a15"
    assert t.kakan(25, 25, 25, 25, rel=1) == "25k252525"
    assert t.discard(60) == 60
    assert t.discard(23, tsumogiri=True) == 60
    assert t.discard(23, riichi=True) == "r23"
    assert t.discard(23, tsumogiri=True, riichi=True) == "r60"


def test_game_dump_shape() -> None:
    k = t.Kyoku(
        kyoku=0, honba=0, riichi_sticks=0, scores=[25000] * 4, dora=[t.tile("3p")]
    )
    k.haipai[0] = t.tiles("1112345678999m")
    k.draws[0] = [t.tile("1s")]
    k.discards[0] = [t.discard(31, tsumogiri=True)]
    k.result = t.Agari(
        [
            t.Win(
                winner=0,
                from_seat=0,
                delta=[48000, -16000, -16000, -16000],
                score_text="役満48000点",
                yaku=["九蓮宝燈(役満)"],
            )
        ]
    )
    g = t.Game(names=["A", "B", "C", "D"], title=["PML", "test"], kyokus=[k])
    d = g.to_dict()
    assert {"title", "name", "rule", "log", "ver"} <= set(d)
    assert d["rule"] == {"disp": "南喰赤", "aka": 1}
    log = d["log"][0]
    assert log[0] == [0, 0, 0]
    assert log[1] == [25000] * 4
    assert log[2] == [23]
    assert log[3] == []
    assert len(log) == 4 + 12 + 1
    assert log[4] == t.tiles("1112345678999m")
    assert log[5] == [31]
    assert log[6] == [60]
    assert log[-1] == [
        "和了",
        [48000, -16000, -16000, -16000],
        [0, 0, 0, "役満48000点", "九蓮宝燈(役満)"],
    ]
    assert json.loads(g.dumps())["log"][0][-1][0] == "和了"
    assert k.name == "東1局"
    assert t.Kyoku(7, 0, 0, [0] * 4).name == "南4局"


def test_ryukyoku_dump() -> None:
    assert t.Ryukyoku([3000, -1000, -1000, -1000]).dump() == [
        "流局",
        [3000, -1000, -1000, -1000],
    ]


def test_viewer_url() -> None:
    g = t.Game(names=["A", "B", "C", "D"])
    assert g.viewer_url().startswith("https://tenhou.net/5/#json=")


def test_editor_url_carries_one_kyoku() -> None:
    g = t.Game(
        names=["A", "B", "C", "D"],
        title=["w", "g"],
        kyokus=[t.Kyoku(i, 0, 0, [25000] * 4, result=t.Ryukyoku()) for i in range(3)],
    )
    for i in range(3):
        url = g.editor_url(i)
        assert url.startswith("https://tenhou.net/6/#json=")
        assert "&" not in url
        d = json.loads(unquote(url.split("#json=", 1)[1]))
        assert d["log"] == [g.kyokus[i].dump()]
        assert d["name"] == ["A", "B", "C", "D"]
        assert d["title"] == ["w", "g"]
    # the whole hanchan in the viewer, then the editor URL of each kyoku
    page = g.links_html()
    assert page.count("https://tenhou.net/6/#json=") == 3
    assert page.count("https://tenhou.net/5/#json=") == 1
    assert "copy all 3 URLs" in page


def test_replay_needs_an_indicator_per_kan() -> None:
    k = t.Kyoku(0, 0, 0, [25000] * 4, dora=[33])
    k.haipai = [
        [11, 11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21],
        [22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26],
        [31, 31, 31, 32, 32, 32, 34, 34, 34, 35, 35, 35, 36],
        [41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45],
    ]
    # the dealer draws, kans 1m, draws its rinshan tile and discards it; the hand ends
    # there
    k.draws = [[27, 28], [], [], []]
    k.discards = [[t.ankan(11), 60], [], [], []]
    (needs,) = [v for v in t.replay_kyoku(k.dump()) if v.kind == "indicators"]
    assert needs.text.endswith("a log needs 2")
    k.dora = [33, 37]
    assert "indicators" not in {v.kind for v in t.replay_kyoku(k.dump())}
