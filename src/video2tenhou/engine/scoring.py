"""Han / fu check of a winning hand against the site record, with the `mahjong` library.

M-League settings: aka dora, kiriage mangan, no abortive draws (irrelevant
here), open tanyao allowed. The site gives han and fu; we compute the yaku
list for the log and report whether the values agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mahjong.constants import EAST, NORTH, SOUTH, WEST
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.tile import TilesConverter

from . import rules

WIND_CONST = {"E": EAST, "S": SOUTH, "W": WEST, "N": NORTH}


def _t136(tiles: list[str]) -> list[int]:
    """Tile tokens -> 136-array ids (red fives as the '0' index of their suit)."""
    m = p = s = z = ""
    for t in tiles:
        n, suit = t[0], t[1]
        if suit == "m":
            m += "r" if n == "0" else n
        elif suit == "p":
            p += "r" if n == "0" else n
        elif suit == "s":
            s += "r" if n == "0" else n
        else:
            z += n
    return TilesConverter.string_to_136_array(man=m, pin=p, sou=s, honors=z, has_aka_dora=True)


def _pick(pool: list[int], tile: str, used: set[int]) -> int:
    """One 136-id of `tile` from the pool not used yet."""
    ids = _t136([tile])
    base = ids[0] // 4 * 4
    cands = [x for x in pool if x // 4 * 4 == base and x not in used]
    if tile in rules.REDS:
        red = [x for x in cands if x % 4 == 0]
        cands = red or cands
    elif tile in ("5m", "5p", "5s"):
        cands = [x for x in cands if x % 4 != 0] or cands
    x = cands[0]
    used.add(x)
    return x


@dataclass
class ScoreResult:
    """Scoring-library outcome; invalid hands carry an error instead of inferred points."""
    ok: bool
    han: Optional[int] = None
    fu: Optional[int] = None
    yaku: list = field(default_factory=list)
    error: Optional[str] = None
    cost: Optional[int] = None


def score_hand(concealed: list[str], win_tile: str, melds: list[dict], *, tsumo: bool, riichi: bool,
               seat: str, round_wind: str, dora: list[str], ura: list[str], ippatsu: bool = False,
               rinshan: bool = False, haitei: bool = False, chankan: bool = False,
               double_riichi: bool = False) -> ScoreResult:
    """concealed: the concealed tiles WITHOUT the winning tile; melds: [{type, tiles}] with
    type chi | pon | kan | kakan | ankan."""
    all_tiles = list(concealed) + [win_tile] + [t for m in melds for t in m["tiles"]]
    if not rules.count_ok(all_tiles):
        return ScoreResult(False, error="more than four of a kind in the winning hand")
    try:
        pool = _t136(all_tiles)
        used: set[int] = set()
        meld_objs = []
        for m in melds:
            tiles = sorted(m["tiles"], key=lambda t: (t[1], rules.number(t), t[0] == "0")) if m["type"] == "chi" else m["tiles"]
            ids = [_pick(pool, t, used) for t in tiles]
            kind = {"chi": Meld.CHI, "pon": Meld.PON, "kan": Meld.KAN, "kakan": Meld.SHOUMINKAN, "ankan": Meld.KAN}[m["type"]]
            meld_objs.append(Meld(meld_type=kind, tiles=ids, opened=m["type"] != "ankan"))
        win_id = _pick(pool, win_tile, used)
        hand_ids = [_pick(pool, t, used) for t in concealed] + [win_id] + [x for mo in meld_objs for x in mo.tiles]
    except Exception as e:  # noqa: BLE001
        return ScoreResult(False, error=f"cannot build the hand: {e} ({' '.join(all_tiles)})")
    cfg = HandConfig(is_tsumo=tsumo, is_riichi=riichi, is_ippatsu=ippatsu, is_rinshan=rinshan, is_haitei=haitei and tsumo,
                     is_houtei=haitei and not tsumo, is_chankan=chankan, is_daburu_riichi=double_riichi,
                     player_wind=WIND_CONST[seat], round_wind=WIND_CONST[round_wind],
                     options=OptionalRules(has_open_tanyao=True, has_aka_dora=True, kiriage=True, kazoe_limit=HandConfig.KAZOE_LIMITED))
    try:
        res = HandCalculator().estimate_hand_value(hand_ids, win_id, melds=meld_objs or None,
                                                   dora_indicators=_t136(dora + ura), config=cfg)
    except Exception as e:  # noqa: BLE001
        return ScoreResult(False, error=str(e))
    if res.error:
        return ScoreResult(False, error=str(res.error))
    opened = any(m["type"] != "ankan" for m in melds)
    yaku = []
    for y in (res.yaku or []):
        han = y.han_open if opened and y.han_open is not None else y.han_closed
        yaku.append(f"{y} ({han})")
    return ScoreResult(True, res.han, res.fu, yaku, cost=res.cost["main"] if res.cost else None)


def matches_site(r: ScoreResult, han: int, fu: int) -> bool:
    """Han and fu must match. The site records the true fu of a mangan or more too, though the payment ignores it
    (a 6/30 reconstruction of a 6/20 pinfu tsumo has the wrong hand); only a yakuman's fu means nothing."""
    return r.ok and r.han == han and (r.fu == fu or (r.han is not None and r.han >= 13))


def is_tenpai(concealed: list[str], melds: list[dict]) -> bool:
    """Shanten 0 for the concealed tiles (13 - 3 per meld)."""
    from mahjong.shanten import Shanten
    ids = _t136(concealed)
    arr = TilesConverter.to_34_array(ids)
    try:
        return Shanten().calculate_shanten(arr) <= 0
    except Exception:  # noqa: BLE001
        return False


LEVEL_JA = [("sanbaiman", "三倍満"), ("baiman", "倍満"), ("haneman", "跳満"), ("yakuman", "役満"), ("mangan", "満貫")]


def _cost(han: int, fu: int, dealer: bool, tsumo: bool) -> dict:
    from mahjong.hand_calculating.scores import ScoresCalculator
    cfg = HandConfig(is_tsumo=tsumo, player_wind=EAST if dealer else SOUTH, round_wind=EAST,
                     options=OptionalRules(kiriage=True, kazoe_limit=HandConfig.KAZOE_LIMITED))
    return ScoresCalculator().calculate_scores(han, fu, cfg, han >= 13)


def payment(han: int, fu: int, *, dealer: bool, tsumo: bool) -> tuple[int, int]:
    """What a win of han/fu pays under the ruleset (kiriage mangan), before honba: (main, additional) — a ron's
    payment, a dealer tsumo's per player, a non-dealer tsumo's dealer and non-dealer parts. Two han/fu that pay the
    same differ only on paper (10/40 and 9/70 are both a baiman)."""
    c = _cost(han, fu, dealer, tsumo)
    return int(c["main"]), int(c.get("additional") or 0)


def score_text(han: int, fu: int, dealer: bool, tsumo: bool) -> str:
    """tenhou's value text of a win from its han and fu (the site's, or the reviewer's correction) under the ruleset: the ron
    payment ("30符2飜2000点"), a non-dealer tsumo's two payments ("30符2飜500-1000点"), a dealer tsumo's one
    ("30符2飜1000点∀"); a limit hand by its name ("満貫8000点")."""
    cost = _cost(han, fu, dealer, tsumo)
    if not tsumo:
        pts = f"{cost['main']}点"
    elif dealer:
        pts = f"{cost['main']}点∀"
    else:
        pts = f"{cost['additional']}-{cost['main']}点"
    level = str(cost.get("yaku_level") or "")
    name = next((ja for key, ja in LEVEL_JA if key in level), None)
    return f"{name}{pts}" if name else f"{fu}符{han}飜{pts}"
