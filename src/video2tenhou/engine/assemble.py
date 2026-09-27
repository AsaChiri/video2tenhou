"""Stage 6: decoded hands -> tenhou.net/6 Game objects (one per hanchan) plus the confidence sidecar.

Seat index: tenhou seat 0 is the starting East (site seat E), 1 = S, 2 = W, 3 = N.
The dealer's 14-tile haipai is written as 13 tiles plus a first draw (the
first discard when it is among the 14, so that discard shows as tsumogiri).
"""
from __future__ import annotations

import re
from typing import Optional

from .. import tenhou6 as T
from ..record import Game as SiteGame, HandResult
from . import rules
from .scoring import score_text

def player_index(seat: str, kyoku: int) -> int:
    """tenhou indexes the four players by the wind they held in the hanchan's first hand (player 0 is the
    site's EAST); our seats are the winds of this hand, so the index shifts with the kyoku."""
    return (rules.SEATS.index(seat) + kyoku) % 4


# the mahjong library's yaku names -> tenhou's
YAKU_JA = {
    "Riichi": "立直", "Double Riichi": "両立直", "Ippatsu": "一発", "Menzen Tsumo": "門前清自摸和", "Tanyao": "断幺九",
    "Pinfu": "平和", "Iipeiko": "一盃口", "Haitei Raoyue": "海底摸月", "Houtei Raoyui": "河底撈魚",
    "Rinshan Kaihou": "嶺上開花", "Chankan": "槍槓", "Chiitoitsu": "七対子", "Chantai": "混全帯幺九", "Ittsu": "一気通貫",
    "Sanshoku Doujun": "三色同順", "Sanshoku Doukou": "三色同刻", "Toitoi": "対々和", "San Ankou": "三暗刻",
    "San Kantsu": "三槓子", "Shou Sangen": "小三元", "Honroutou": "混老頭", "Ryanpeikou": "二盃口", "Junchan": "純全帯幺九",
    "Honitsu": "混一色", "Chinitsu": "清一色", "Dora": "ドラ", "Aka Dora": "赤ドラ", "Ura Dora": "裏ドラ",
    "Yakuhai (haku)": "役牌 白", "Yakuhai (hatsu)": "役牌 發", "Yakuhai (chun)": "役牌 中",
    "Yakuhai (seat wind east)": "自風 東", "Yakuhai (seat wind south)": "自風 南", "Yakuhai (seat wind west)": "自風 西",
    "Yakuhai (seat wind north)": "自風 北", "Yakuhai (round wind east)": "場風 東", "Yakuhai (round wind south)": "場風 南",
    "Yakuhai (round wind west)": "場風 西", "Yakuhai (round wind north)": "場風 北",
    "Suu Ankou": "四暗刻", "Suu Ankou Tanki": "四暗刻単騎", "Daisangen": "大三元", "Kokushi Musou": "国士無双",
    "Kokushi Musou Juusanmen Matchi": "国士無双１３面", "Shousuushii": "小四喜", "Dai Suushii": "大四喜",
    "Tsuu Iisou": "字一色", "Ryuuiisou": "緑一色", "Chinroutou": "清老頭", "Chuuren Poutou": "九蓮宝燈",
    "Daburu Chuuren Poutou": "純正九蓮宝燈", "Suu Kantsu": "四槓子", "Tenhou": "天和", "Chiihou": "地和",
}


def tile_id(t: str) -> int:
    """Convert an internal tile token to tenhou's numeric identity, preserving red fives."""
    return T.tile(t)


def _yaku_text(yaku: list[str]) -> list[str]:
    """"Name (han)" from the scoring library -> tenhou's "名前(n飜)" (a yakuman: "名前(役満)")."""
    out = []
    for y in yaku:
        name, _, han = y.rpartition(" (")
        if not name:
            name, han = y, ""
        name = re.sub(r"\s*\d+$", "", name.strip())        # "Dora 2" -> "Dora"
        n = han.rstrip(")").strip()
        ja = YAKU_JA.get(name, name)
        out.append(f"{ja}(役満)" if n and int(n) >= 13 else (f"{ja}({n}飜)" if n else ja))
    return out


def tenhou_deltas(result: HandResult) -> list[int]:
    """The site's deltas in tenhou's order (EAST..NORTH is tenhou's player 0..3) with the riichi deposits taken
    out: the site charges a declarer's 1000 in the hand's deltas, the viewer charges it at the `r` discard."""
    return [result.deltas[n] + (1000 if n in result.riichi else 0) for n in ("EAST", "SOUTH", "WEST", "NORTH")]


def _one_red(ids: list[int], protect: int = -1) -> list[int]:
    """A kan of fives holds all four fives, so exactly one of its ids is red whatever the camera read; the
    tile at `protect` (the called or added one) keeps its reading when it can."""
    kind = T.deaka(ids[0])
    if kind not in (15, 25, 35):
        return ids
    red = {15: 51, 25: 52, 35: 53}[kind]
    keep = protect if 0 <= protect < len(ids) and ids[protect] >= 51 else next((i for i, x in enumerate(ids) if x >= 51), None)
    if keep is None:
        keep = next((i for i in range(len(ids)) if i != protect), 0)
    return [red if i == keep else T.deaka(x) for i, x in enumerate(ids)]


def call_string(c: dict, caller: str) -> str:
    """tenhou call string for a decoded call (dict from Call.to_dict)."""
    known = [t for t in c["tiles"] if t not in ("X", "?")]
    if c["type"] == "ankan":
        base = rules.plain(known[0]) if known else None
        if base is None:
            raise ValueError("ankan without a known tile")
        # a concealed kan of fives is all four fives, the red one among them
        return T.ankan(tile_id(base), has_aka=base in ("5m", "5p", "5s"))
    tiles = [tile_id(t) for t in c["tiles"]]
    pos = c.get("called_pos")
    typ = c["type"]
    if typ == "chi":
        called = tiles[pos if pos is not None else 0]
        rest = [x for i, x in enumerate(tiles) if i != (pos if pos is not None else 0)]
        return T.chi(called, rest[0], rest[1])
    src = c.get("source") or "kamicha"
    rel = {"kamicha": 0, "toimen": 1, "shimocha": 2}[src]
    if typ == "pon":
        called = tiles[pos] if pos is not None else tiles[0]
        rest = [x for i, x in enumerate(tiles) if i != (pos if pos is not None else 0)]
        return T.pon(called, rest[0], rest[1], rel=rel)
    if typ == "kan":
        p = pos if pos is not None else 0
        tiles = _one_red(tiles, p)
        rest = [x for i, x in enumerate(tiles) if i != p]
        return T.daiminkan(tiles[p], rest[0], rest[1], rest[2], rel=rel)
    if typ == "kakan":
        # the added tile is the last one (for fives, the plain or red one the solver decided)
        tiles = _one_red(tiles, 3)
        return T.kakan(tiles[3], tiles[0], tiles[1], tiles[2], rel=rel)
    raise ValueError(typ)


def kyoku_from_decode(d: dict, entry: dict, result: HandResult) -> tuple[T.Kyoku, list[dict]]:
    """Build the tenhou/6 kyoku. Returns (kyoku, confidence rows)."""
    scores = [0, 0, 0, 0]
    for s in rules.SEATS:
        scores[player_index(s, entry["kyoku"])] = entry["scores"][s]
    k = T.Kyoku(entry["kyoku"], entry["honba"], entry["sticks"], scores,
                dora=[tile_id(t) for t in d["dora"]], ura=[tile_id(t) for t in d.get("ura", [])])
    conf: list[dict] = list(d.get("confidence", []))
    dealer = d["dealer"]
    turns_by_seat: dict[str, list[dict]] = {s: [] for s in rules.SEATS}
    for t in d["turns"]:
        turns_by_seat[t["seat"]].append(t)
    for s in rules.SEATS:
        i = player_index(s, entry["kyoku"])
        haipai = list(d["haipai"].get(s, []))
        first_draw: Optional[str] = None
        first_tsumogiri = False
        if s == dealer and len(haipai) == 14:
            # the split: the first discard when it is among the 14 (that discard then shows as tsumogiri)
            mine = turns_by_seat[s]
            fd = mine[0]["discard"] if mine and mine[0]["discard"] in haipai else haipai[-1]
            haipai.remove(fd)
            first_draw = fd
            first_tsumogiri = bool(mine) and mine[0]["discard"] == fd and mine[0]["kind"] == "draw"
            conf.append({"seat": s, "turn": 0, "field": "first_draw", "margin": None, "human": False, "lost": False,
                         "note": "dealer split is arbitrary"})
        k.haipai[i] = sorted(tile_id(t) for t in haipai)
        draws: list = []
        discards: list = []
        if first_draw is not None:
            draws.append(tile_id(first_draw))
        for t in turns_by_seat[s]:
            if t["kind"] in ("call", "kan") and t.get("own_call"):
                c = t["own_call"]
                if c["type"] in ("chi", "pon", "kan"):
                    draws.append(call_string(c, s))
                    if c["type"] == "kan":
                        discards.append(0)
                        if t["draw"] is not None:
                            draws.append(tile_id(t["draw"]))
                elif c["type"] in ("ankan", "kakan"):
                    if t["draw"] is not None:
                        draws.append(tile_id(t["draw"]))
                    try:
                        discards.append(call_string(c, s))
                    except (ValueError, KeyError) as ex:
                        # an unresolved kan (tile unknown in a conflicting hand): the log keeps the turn without it
                        conf.append({"seat": s, "turn": t["j"], "field": "kan", "margin": None, "human": False, "lost": True, "note": str(ex)})
                        discards.append(0)
                    if t.get("draw2") is not None:
                        draws.append(tile_id(t["draw2"]))
            elif t["kind"] in ("draw",) and t["draw"] is not None:
                draws.append(tile_id(t["draw"]))
            dealer_first = s == dealer and t["j"] == 0 and t["kind"] == "draw"      # no draw of its own: the 14th tile
            if t["discard"] is not None:
                tsumogiri = first_tsumogiri if dealer_first else bool(t["tsumogiri"])
                discards.append(T.discard(tile_id(t["discard"]), tsumogiri=tsumogiri, riichi=bool(t["riichi"])))
        # the winning tsumo draw
        if result.outcome == "tsumo" and d["result"]["winner"] == s:
            jw = len(turns_by_seat[s])
            wd = d["draws"].get(f"{s}:{jw}")
            if wd:
                draws.append(tile_id(wd))
        k.draws[i] = draws
        k.discards[i] = discards
    # result
    deltas = tenhou_deltas(result)
    if result.outcome in ("ron", "tsumo"):
        w = player_index(d["result"]["winner"], entry["kyoku"])
        frm = player_index(d["result"]["loser"], entry["kyoku"]) if result.outcome == "ron" else w
        sc = d.get("score") or {}
        # the site's han/fu, or the reviewer's when they confirmed the site wrong (the deltas stay the site's)
        han, fu = (d["result"]["han"], d["result"]["fu"]) if d["result"].get("site_wrong") else (result.han, result.fu)
        text = score_text(han or 0, fu or 0, dealer=d["result"]["winner"] == dealer, tsumo=result.outcome == "tsumo")
        yaku = _yaku_text(sc.get("yaku") or []) if sc.get("match") else []
        k.result = T.Agari([T.Win(w, frm, deltas, text, yaku)])
    else:
        k.result = T.Ryukyoku(deltas)
    return k, conf


def game_from_decodes(decodes: list[dict], entries: list[dict], site: SiteGame,
                      title: str) -> tuple[T.Game, dict[int, list[dict]], dict[int, list[str]]]:
    """The log of one hanchan: a kyoku per decoded hand, except the hands left out (section 6: nothing is written
    for a conflict) — those with no legal reconstruction, and those the replayer rejects. Returns the game, the
    confidence rows by hand, and the reasons each left-out hand was left out."""
    names = [site.players.get(n, "") for n in ("EAST", "SOUTH", "WEST", "NORTH")]
    g = T.Game(names=names, title=[title, f"scoremj game {site.id}"])
    conf: dict[int, list[dict]] = {}
    left_out: dict[int, list[str]] = {}
    by_hand = {e["hand"]: e for e in entries}
    for d in sorted(decodes, key=lambda d: d["hand"]):
        e = by_hand[d["hand"]]
        if d["solver"]["status"] not in ("optimal", "feasible", "repaired"):
            left_out[d["hand"]] = ["no legal reconstruction (a conflict): not written"]
            continue
        k, rows = kyoku_from_decode(d, e, site.hands[e["site_index"]])
        violations = T.replay_kyoku(k.dump())
        if violations:
            left_out[d["hand"]] = violations
            continue
        g.kyokus.append(k)
        conf[d["hand"]] = rows
    return g, conf, left_out
