# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 6: validated decoded hands as Tenhou games and score results.

Stage 6: decoded hands -> tenhou.net/6 Game objects (one per hanchan) plus the
confidence sidecar.

Seat index: tenhou seat 0 is the starting East (site seat E), 1 = S, 2 = W, 3 = N. The
dealer's 14-tile haipai is written as 13 tiles plus a first draw (the first discard when
it is among the 14, so that discard shows as tsumogiri).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from video2tenhou import tenhou6

from . import rules
from .scoring import score_text

if TYPE_CHECKING:
    from video2tenhou.record import Game as SiteGame
    from video2tenhou.record import HandResult

YAKUMAN_HAN = 13
FIRST_RED_TILE_ID = 51
DEALER_STARTING_TILES = 14


def player_index(seat: str, kyoku: int) -> int:
    """Map a hand's wind to the player's starting-seat index.

    Tenhou indexes the four players by the wind they held in the hanchan's first hand
    (player 0 is the site's EAST); our seats are the winds of this hand, so the index
    shifts with the kyoku.
    """
    return (rules.SEATS.index(seat) + kyoku) % 4


# the mahjong library's yaku names -> tenhou's
YAKU_JA = {
    "Riichi": "立直",
    "Double Riichi": "両立直",
    "Ippatsu": "一発",
    "Menzen Tsumo": "門前清自摸和",
    "Tanyao": "断幺九",
    "Pinfu": "平和",
    "Iipeiko": "一盃口",
    "Haitei Raoyue": "海底摸月",
    "Houtei Raoyui": "河底撈魚",
    "Rinshan Kaihou": "嶺上開花",
    "Chankan": "槍槓",
    "Chiitoitsu": "七対子",
    "Chantai": "混全帯幺九",
    "Ittsu": "一気通貫",
    "Sanshoku Doujun": "三色同順",
    "Sanshoku Doukou": "三色同刻",
    "Toitoi": "対々和",
    "San Ankou": "三暗刻",
    "San Kantsu": "三槓子",
    "Shou Sangen": "小三元",
    "Honroutou": "混老頭",
    "Ryanpeikou": "二盃口",
    "Junchan": "純全帯幺九",
    "Honitsu": "混一色",
    "Chinitsu": "清一色",
    "Dora": "ドラ",
    "Aka Dora": "赤ドラ",
    "Ura Dora": "裏ドラ",
    "Yakuhai (haku)": "役牌 白",
    "Yakuhai (hatsu)": "役牌 發",
    "Yakuhai (chun)": "役牌 中",
    "Yakuhai (seat wind east)": "自風 東",
    "Yakuhai (seat wind south)": "自風 南",
    "Yakuhai (seat wind west)": "自風 西",
    "Yakuhai (seat wind north)": "自風 北",
    "Yakuhai (round wind east)": "場風 東",
    "Yakuhai (round wind south)": "場風 南",
    "Yakuhai (round wind west)": "場風 西",
    "Yakuhai (round wind north)": "場風 北",
    "Suu Ankou": "四暗刻",
    "Suu Ankou Tanki": "四暗刻単騎",
    "Daisangen": "大三元",
    "Kokushi Musou": "国士無双",
    "Kokushi Musou Juusanmen Matchi": "国士無双１３面",
    "Shousuushii": "小四喜",
    "Dai Suushii": "大四喜",
    "Tsuu Iisou": "字一色",
    "Ryuuiisou": "緑一色",
    "Chinroutou": "清老頭",
    "Chuuren Poutou": "九蓮宝燈",
    "Daburu Chuuren Poutou": "純正九蓮宝燈",
    "Suu Kantsu": "四槓子",
    "Tenhou": "天和",
    "Chiihou": "地和",
}


def _yaku_text(yaku: list[str]) -> list[str]:
    """Format a scoring-library yaku name for Tenhou, including yakuman."""
    out = []
    for y in yaku:
        name, _, han = y.rpartition(" (")
        if not name:
            name, han = y, ""
        name = re.sub(r"\s*\d+$", "", name.strip())  # "Dora 2" -> "Dora"
        n = han.rstrip(")").strip()
        ja = YAKU_JA.get(name, name)
        out.append(
            f"{ja}(役満)"
            if n and int(n) >= YAKUMAN_HAN
            else (f"{ja}({n}飜)" if n else ja)
        )
    return out


def tenhou_deltas(result: HandResult) -> list[int]:
    """Map site deltas to Tenhou order and account for riichi deposits.

    The site's deltas in Tenhou order (EAST..NORTH is Tenhou's player 0..3) with
    riichi deposits taken out: the site charges a declarer's 1000 in the hand's deltas,
    the viewer charges it at the `r` discard.
    """
    return [
        result.deltas[n] + (1000 if n in result.riichi else 0)
        for n in ("EAST", "SOUTH", "WEST", "NORTH")
    ]


def _one_red(ids: list[int], protect: int = -1) -> list[int]:
    """Ensure a kan of fives contains exactly one red five.

    A kan of fives holds all four fives, so exactly one of its ids is red whatever the
    camera read; the tile at `protect` (the called or added one) keeps its reading when
    it can.
    """
    kind = tenhou6.deaka(ids[0])
    if kind not in (15, 25, 35):
        return ids
    red = {15: 51, 25: 52, 35: 53}[kind]
    keep = (
        protect
        if 0 <= protect < len(ids) and ids[protect] >= FIRST_RED_TILE_ID
        else next((i for i, x in enumerate(ids) if x >= FIRST_RED_TILE_ID), None)
    )
    if keep is None:
        keep = next((i for i in range(len(ids)) if i != protect), 0)
    return [red if i == keep else tenhou6.deaka(x) for i, x in enumerate(ids)]


def call_string(c: dict) -> str:
    """Tenhou call string for a decoded call (dict from Call.to_dict)."""
    known = [t for t in c["tiles"] if t not in ("X", "?")]
    if c["type"] == "ankan":
        base = rules.plain(known[0]) if known else None
        if base is None:
            msg = "ankan without a known tile"
            raise ValueError(msg)
        # a concealed kan of fives is all four fives, the red one among them
        return tenhou6.ankan(tenhou6.tile(base), has_aka=base in ("5m", "5p", "5s"))
    tiles = [tenhou6.tile(t) for t in c["tiles"]]
    pos = c.get("called_pos")
    pos = pos if pos is not None else 0
    typ = c["type"]
    if typ in ("chi", "pon", "kan"):
        if typ == "kan":
            tiles = _one_red(tiles, pos)
        called = tiles[pos]
        rest = [x for i, x in enumerate(tiles) if i != pos]
        if typ == "chi":
            return tenhou6.chi(called, rest[0], rest[1])
    src = c.get("source") or "kamicha"
    rel = {"kamicha": 0, "toimen": 1, "shimocha": 2}[src]
    if typ == "pon":
        return tenhou6.pon(called, rest[0], rest[1], rel=rel)
    if typ == "kan":
        return tenhou6.daiminkan(called, rest[0], rest[1], rest[2], rel=rel)
    if typ == "kakan":
        # the added tile is the last one (for fives, the plain or red one the solver
        # decided)
        tiles = _one_red(tiles, 3)
        return tenhou6.kakan(tiles[3], tiles[0], tiles[1], tiles[2], rel=rel)
    raise ValueError(typ)


def _call_stream(t: dict, draws: list, discards: list) -> None:
    """Append a call and any replacement draw in Tenhou event order."""
    c = t["own_call"]
    if c["type"] in ("chi", "pon", "kan"):
        draws.append(call_string(c))
        if c["type"] == "kan":
            discards.append(0)
            if t["draw"] is not None:
                draws.append(tenhou6.tile(t["draw"]))
    elif c["type"] in ("ankan", "kakan"):
        if t["draw"] is not None:
            draws.append(tenhou6.tile(t["draw"]))
        discards.append(call_string(c))
        if t.get("draw2") is not None:
            draws.append(tenhou6.tile(t["draw2"]))


def _turn_streams(
    turns: list[dict], *, is_dealer: bool, first_tsumogiri: bool
) -> tuple[list, list]:
    """Encode calls, replacement draws and discards in seat order."""
    draws: list = []
    discards: list = []
    for t in turns:
        if t["kind"] in ("call", "kan") and t.get("own_call"):
            _call_stream(t, draws, discards)
        elif t["kind"] == "draw" and t["draw"] is not None:
            draws.append(tenhou6.tile(t["draw"]))
        dealer_first = (
            is_dealer and t["j"] == 0 and t["kind"] == "draw"
        )  # no draw of its own: the 14th tile
        if t["discard"] is not None:
            tsumogiri = first_tsumogiri if dealer_first else bool(t["tsumogiri"])
            discards.append(
                tenhou6.discard(
                    tenhou6.tile(t["discard"]),
                    tsumogiri=tsumogiri,
                    riichi=bool(t["riichi"]),
                )
            )
    return draws, discards


def _hand_result(
    d: dict, entry: dict, result: HandResult
) -> tenhou6.Agari | tenhou6.Ryukyoku:
    """Encode the authoritative payments and accepted score explanation."""
    deltas = tenhou_deltas(result)
    if result.outcome in ("ron", "tsumo"):
        w = player_index(d["result"]["winner"], entry["kyoku"])
        frm = (
            player_index(d["result"]["loser"], entry["kyoku"])
            if result.outcome == "ron"
            else w
        )
        sc = d.get("score") or {}
        # the site's han/fu, or the reviewer's when they confirmed the site wrong (the
        # deltas stay the site's)
        han, fu = (
            (d["result"]["han"], d["result"]["fu"])
            if d["result"].get("site_wrong")
            else (result.han, result.fu)
        )
        text = score_text(
            han or 0,
            fu or 0,
            dealer=d["result"]["winner"] == d["dealer"],
            tsumo=result.outcome == "tsumo",
        )
        yaku = _yaku_text(sc.get("yaku") or []) if sc.get("match") else []
        return tenhou6.Agari([tenhou6.Win(w, frm, deltas, text, yaku)])
    return tenhou6.Ryukyoku(deltas)


def kyoku_from_decode(
    d: dict, entry: dict, result: HandResult
) -> tuple[tenhou6.Kyoku, list[dict]]:
    """Build the tenhou/6 kyoku. Returns (kyoku, confidence rows)."""
    scores = [0, 0, 0, 0]
    for s in rules.SEATS:
        scores[player_index(s, entry["kyoku"])] = entry["scores"][s]
    k = tenhou6.Kyoku(
        entry["kyoku"],
        entry["honba"],
        entry["sticks"],
        scores,
        dora=[tenhou6.tile(t) for t in d["dora"]],
        ura=[tenhou6.tile(t) for t in d.get("ura", [])],
    )
    conf: list[dict] = list(d.get("confidence", []))
    dealer = d["dealer"]
    turns_by_seat: dict[str, list[dict]] = {s: [] for s in rules.SEATS}
    for t in d["turns"]:
        turns_by_seat[t["seat"]].append(t)
    for s in rules.SEATS:
        i = player_index(s, entry["kyoku"])
        haipai = list(d["haipai"].get(s, []))
        first_draw: str | None = None
        first_tsumogiri = False
        if s == dealer and len(haipai) == DEALER_STARTING_TILES:
            # the split: the first discard when it is among the 14 (that discard then
            # shows as tsumogiri)
            mine = turns_by_seat[s]
            fd = (
                mine[0]["discard"]
                if mine and mine[0]["discard"] in haipai
                else haipai[-1]
            )
            haipai.remove(fd)
            first_draw = fd
            first_tsumogiri = (
                bool(mine) and mine[0]["discard"] == fd and mine[0]["kind"] == "draw"
            )
            conf.append(
                {
                    "seat": s,
                    "turn": 0,
                    "field": "first_draw",
                    "margin": None,
                    "human": False,
                    "lost": False,
                    "note": "dealer split is arbitrary",
                }
            )
        k.haipai[i] = sorted(tenhou6.tile(t) for t in haipai)
        draws: list = []
        discards: list = []
        if first_draw is not None:
            draws.append(tenhou6.tile(first_draw))
        turn_draws, discards = _turn_streams(
            turns_by_seat[s], is_dealer=s == dealer, first_tsumogiri=first_tsumogiri
        )
        draws.extend(turn_draws)
        # the winning tsumo draw
        if result.outcome == "tsumo" and d["result"]["winner"] == s:
            jw = len(turns_by_seat[s])
            wd = d["draws"].get(f"{s}:{jw}")
            if wd:
                draws.append(tenhou6.tile(wd))
        k.draws[i] = draws
        k.discards[i] = discards
    k.result = _hand_result(d, entry, result)
    return k, conf


def game_from_decodes(
    decodes: list[dict], entries: list[dict], site: SiteGame, title: str
) -> tuple[tenhou6.Game, dict[int, list[dict]], dict[int, list[str]]]:
    """Build one hanchan log from the hands accepted for export.

    The log of one hanchan: a kyoku per decoded hand, except the hands left out (section
    6: nothing is written for a conflict) — those with no legal reconstruction, and
    those the replayer rejects. Returns the game, the confidence rows by hand, and the
    reasons each left-out hand was left out.
    """
    names = [site.players.get(n, "") for n in ("EAST", "SOUTH", "WEST", "NORTH")]
    g = tenhou6.Game(names=names, title=[title, f"scoremj game {site.id}"])
    conf: dict[int, list[dict]] = {}
    left_out: dict[int, list[str]] = {}
    by_hand = {e["hand"]: e for e in entries}
    for d in sorted(decodes, key=lambda d: d["hand"]):
        e = by_hand[d["hand"]]
        if d["solver"]["status"] not in ("optimal", "feasible", "repaired"):
            left_out[d["hand"]] = ["no legal reconstruction (a conflict): not written"]
            continue
        k, rows = kyoku_from_decode(d, e, site.hands[e["site_index"]])
        violations = tenhou6.replay_kyoku(k.dump())
        if violations:
            left_out[d["hand"]] = violations
            continue
        g.kyokus.append(k)
        conf[d["hand"]] = rows
    return g, conf, left_out
