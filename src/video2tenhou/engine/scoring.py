# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Han / fu check of a winning hand against the site record, with the `mahjong` library.

M-League settings: aka dora, kiriage mangan, no abortive draws (irrelevant
here), open tanyao allowed. The site gives han and fu; we compute the yaku
list for the log and report whether the values agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mahjong.constants import EAST, NORTH, SOUTH, WEST
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.scores import ScoresCalculator
from mahjong.meld import Meld
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

from . import rules

if TYPE_CHECKING:
    from mahjong.hand_calculating.scores import ScoresResult

YAKUMAN_HAN = 13
WIND_CONST = {"E": EAST, "S": SOUTH, "W": WEST, "N": NORTH}


def _t136(tiles: list[str]) -> list[int]:
    """Tile tokens -> 136-array ids (a red five is the '0' of its suit)."""
    return TilesConverter.one_line_string_to_136_array(
        "".join(tiles), has_aka_dora=True
    )


def _pick(pool: list[int], tile: str, used: set[int]) -> int:
    """One 136-id of `tile` from the pool not used yet."""
    ids = _t136([tile])
    base = ids[0] // 4 * 4
    cands = [x for x in pool if x // 4 * 4 == base and x not in used]
    if tile in rules.PLAIN_OF:
        red = [x for x in cands if x % 4 == 0]
        cands = red or cands
    elif tile in rules.RED_OF:
        cands = [x for x in cands if x % 4 != 0] or cands
    x = cands[0]
    used.add(x)
    return x


@dataclass(frozen=True)
class Yaku:
    """One yaku of a scored hand, by the scoring library's name."""

    name: str
    han: int

    def __str__(self) -> str:
        """Name and han, as in "Riichi (1)"."""
        return f"{self.name} ({self.han})"

    def to_dict(self) -> dict:
        """Serialize for decode artifacts."""
        return {"name": self.name, "han": self.han}


@dataclass
class ScoreResult:
    """Scoring outcome; invalid hands carry an error instead of points."""

    ok: bool
    han: int | None = None
    fu: int | None = None
    yaku: list[Yaku] = field(default_factory=list)
    error: str | None = None
    cost: int | None = None


@dataclass(frozen=True, kw_only=True)
class WinContext:
    """Winning circumstances and visible indicators used for han/fu evaluation."""

    tsumo: bool
    riichi: bool
    seat: str
    round_wind: str
    dora: list[str]
    ura: list[str]
    ippatsu: bool = False
    rinshan: bool = False
    haitei: bool = False
    double_riichi: bool = False


def score_hand(
    concealed: list[str],
    win_tile: str,
    melds: list[dict],
    context: WinContext,
) -> ScoreResult:
    """Score concealed tiles, a winning tile and declared melds.

    concealed: the concealed tiles WITHOUT the winning tile; melds: [{type, tiles}] with
    type chi | pon | kan | kakan | ankan.
    """
    all_tiles = list(concealed) + [win_tile] + [t for m in melds for t in m["tiles"]]
    if not rules.count_ok(all_tiles):
        return ScoreResult(
            ok=False, error="more than four of a kind in the winning hand"
        )
    pool = _t136(all_tiles)
    used: set[int] = set()
    meld_objs = []
    for m in melds:
        tiles = (
            sorted(m["tiles"], key=lambda t: (t[1], rules.number(t), t[0] == "0"))
            if m["type"] == "chi"
            else m["tiles"]
        )
        ids = [_pick(pool, t, used) for t in tiles]
        kind = {
            "chi": Meld.CHI,
            "pon": Meld.PON,
            "kan": Meld.KAN,
            "kakan": Meld.SHOUMINKAN,
            "ankan": Meld.KAN,
        }[m["type"]]
        meld_objs.append(Meld(meld_type=kind, tiles=ids, opened=m["type"] != "ankan"))
    win_id = _pick(pool, win_tile, used)
    hand_ids = (
        [_pick(pool, t, used) for t in concealed]
        + [win_id]
        + [x for mo in meld_objs for x in mo.tiles]
    )
    cfg = HandConfig(
        is_tsumo=context.tsumo,
        is_riichi=context.riichi,
        is_ippatsu=context.ippatsu,
        is_rinshan=context.rinshan,
        is_haitei=context.haitei and context.tsumo,
        is_houtei=context.haitei and not context.tsumo,
        is_daburu_riichi=context.double_riichi,
        player_wind=WIND_CONST[context.seat],
        round_wind=WIND_CONST[context.round_wind],
        options=OptionalRules(
            has_open_tanyao=True,
            has_aka_dora=True,
            kiriage=True,
            kazoe_limit=HandConfig.KAZOE_LIMITED,
        ),
    )
    res = HandCalculator().estimate_hand_value(
        hand_ids,
        win_id,
        melds=meld_objs or None,
        dora_indicators=_t136(context.dora + context.ura),
        config=cfg,
    )
    if res.error:
        return ScoreResult(ok=False, error=str(res.error))
    opened = any(m["type"] != "ankan" for m in melds)
    yaku = [
        Yaku(y.name, y.han_open if opened and y.han_open is not None else y.han_closed)
        for y in res.yaku or []
    ]
    return ScoreResult(
        ok=True,
        han=res.han,
        fu=res.fu,
        yaku=yaku,
        cost=res.cost["main"] if res.cost else None,
    )


def matches_site(r: ScoreResult, han: int | None, fu: int | None) -> bool:
    """Check authoritative han and fu, including fu above mangan.

    The site records the true fu of a mangan or more too, though the payment ignores it
    (a 6/30 reconstruction of a 6/20 pinfu tsumo has the wrong hand); only a yakuman's
    fu means nothing.
    """
    return (
        han is not None
        and r.ok
        and r.han == han
        and ((fu is not None and r.fu == fu) or han >= YAKUMAN_HAN)
    )


def is_tenpai(concealed: list[str]) -> bool:
    """Shanten 0 for the concealed tiles (13 - 3 per meld)."""
    ids = _t136(concealed)
    arr = TilesConverter.to_34_array(ids)
    return Shanten().calculate_shanten(arr) <= 0


LEVEL_JA = [
    ("sanbaiman", "三倍満"),
    ("baiman", "倍満"),
    ("haneman", "跳満"),
    ("yakuman", "役満"),
    ("mangan", "満貫"),
]


def _cost(han: int, fu: int, *, dealer: bool, tsumo: bool) -> ScoresResult:

    cfg = HandConfig(
        is_tsumo=tsumo,
        player_wind=EAST if dealer else SOUTH,
        round_wind=EAST,
        options=OptionalRules(kiriage=True, kazoe_limit=HandConfig.KAZOE_LIMITED),
    )
    return ScoresCalculator().calculate_scores(han, fu, cfg, han >= YAKUMAN_HAN)


def payment(han: int, fu: int, *, dealer: bool, tsumo: bool) -> tuple[int, int]:
    """Compute what a win of han/fu pays under kiriage mangan, before honba.

    Returns (main, additional): a ron's payment, a dealer tsumo's per player, a
    non-dealer tsumo's dealer and non-dealer parts. Two han/fu that pay the same differ
    only on paper (10/40 and 9/70 are both a baiman).
    """
    c = _cost(han, fu, dealer=dealer, tsumo=tsumo)
    return int(c["main"]), int(c.get("additional") or 0)


def score_text(han: int, fu: int, *, dealer: bool, tsumo: bool) -> str:
    """Format Tenhou's value text of a win from its han and fu under the ruleset.

    The han and fu are the site's, or the reviewer's correction: the ron payment
    ("30符2飜2000点"), a non-dealer tsumo's two payments ("30符2飜500-1000点"), a dealer
    tsumo's one ("30符2飜1000点∀"); a limit hand by its name ("満貫8000点").
    """
    cost = _cost(han, fu, dealer=dealer, tsumo=tsumo)
    if not tsumo:
        pts = f"{cost['main']}点"
    elif dealer:
        pts = f"{cost['main']}点∀"
    else:
        pts = f"{cost['additional']}-{cost['main']}点"
    level = str(cost.get("yaku_level") or "")
    name = next((ja for key, ja in LEVEL_JA if key in level), None)
    return f"{name}{pts}" if name else f"{fu}符{han}飜{pts}"
