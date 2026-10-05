# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Data model and serializer for the tenhou.net/6 JSON log format.

The format (as consumed by tenhou.net/6, mjai-reviewer, Mortal, tensoul, ...):

    {
      "title": ["<title>", "<subtitle>"],
      "name":  ["<seat0>", "<seat1>", "<seat2>", "<seat3>"],
      "rule":  {"disp": "<rule text>", "aka": 1},
      "log":   [ <kyoku>, <kyoku>, ... ]
    }

    <kyoku> = [
      [kyoku_index, honba, riichi_sticks],   # kyoku_index: 0=E1 .. 3=E4, 4=S1 ..
      [score0, score1, score2, score3],      # scores at the start of the hand
      [dora indicators...],
      [ura dora indicators...],              # empty unless someone won with riichi
      haipai0, draws0, discards0,            # seat 0 (the seat that is East in E1)
      haipai1, draws1, discards1,
      haipai2, draws2, discards2,
      haipai3, draws3, discards3,
      result                                  # optional
    ]

Tile encoding: 11-19 = 1-9m, 21-29 = 1-9p, 31-39 = 1-9s, 41-47 = E S W N Wh G R,
51/52/53 = red 5m/5p/5s. The dealer's haipai has 13 tiles and their first draw is the
14th tile. A discard of 60 means tsumogiri (discarded the tile just drawn). A riichi
declaration prefixes the discard with "r" (e.g. "r60", "r23").

Calls are strings placed in *draws* (chi / pon / daiminkan, since they replace a draw)
or in *discards* (ankan / kakan, since they replace a discard; a daiminkan also appends
a literal 0 to discards). The called tile's position in the string marks who it came
from (left = kamicha, middle = toimen, right = shimocha):

    chi    "c<called><a><b>"                  always from kamicha
    pon    "p<called><a><b>" | "<a>p<called><b>" | "<a><b>p<called>"
    daiminkan  "m<called><a><b><c>" | "<a>m<called><b><c>" | "<a><b><c>m<called>"
    ankan  "<t><t><t>a<t>"
    kakan  "k<t><a><b><c>" | "<a>k<t><b><c>" | "<a><b>k<t><c>"   (the pon's position)

Result (the ruleset has no abortive draws):
    ["和了", [delta...],
     [winner, from, pao, "<fu>符<han>飜<pts>点", "yaku(1飜)", ...], ...]
    ["流局", [delta...]]
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING
from urllib.parse import quote

from mahjong.shanten import Shanten

if TYPE_CHECKING:
    from collections.abc import Sequence

TILE_TOKEN_LENGTH = 2
MAX_HONOR_RANK = 7
MAX_SUIT_RANK = 9
SHIMOCHA_SOURCE_OFFSET = 2
RESULT_FIELD_INDEX = 16
STARTING_HAND_SIZE = 13
MELD_SIZE = 3  # tiles of a chi or pon
KAN_SIZE = 4


TSUMOGIRI = 60
WINDS = "東南西北"
AGARI = "和了"
RYUKYOKU = "流局"

# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------

_SUIT_BASE = {"m": 10, "p": 20, "s": 30, "z": 40}
RED_ID = {15: 51, 25: 52, 35: 53}  # plain five -> the red five of its suit
PLAIN_ID = {red: five for five, red in RED_ID.items()}  # red five -> plain five


def tile(s: str) -> int:
    """Parse a tile in short notation: '5m', '0p' (red 5p), '1z' (E) ... '7z' (R)."""
    if len(s) != TILE_TOKEN_LENGTH or s[1] not in _SUIT_BASE or not s[0].isdigit():
        raise ValueError(f"bad tile {s!r}")
    n, suit = int(s[0]), s[1]
    if n == 0:
        if suit == "z":
            raise ValueError("no red honor tile")
        return RED_ID[_SUIT_BASE[suit] + 5]
    if suit == "z" and not 1 <= n <= MAX_HONOR_RANK:
        raise ValueError(f"bad honor {s!r}")
    if suit != "z" and not 1 <= n <= MAX_SUIT_RANK:
        raise ValueError(f"bad number {s!r}")
    return _SUIT_BASE[suit] + n


def tile_str(t: int) -> str:
    """Inverse of :func:`tile`."""
    suit = {1: "m", 2: "p", 3: "s", 4: "z"}[deaka(t) // 10]
    return f"0{suit}" if t in PLAIN_ID else f"{t % 10}{suit}"


def tiles(spec: str) -> list[int]:
    """'123m0p77z' -> [11, 12, 13, 52, 47, 47]."""
    out, digits = [], ""
    for ch in spec:
        if ch.isdigit():
            digits += ch
        elif ch in _SUIT_BASE:
            out.extend(tile(d + ch) for d in digits)
            digits = ""
        elif not ch.isspace():
            raise ValueError(f"bad char {ch!r} in {spec!r}")
    if digits:
        raise ValueError(f"trailing digits in {spec!r}")
    return out


def deaka(t: int) -> int:
    """Map a red-five ID to its ordinary kind for shape checks, preserving other IDs."""
    return PLAIN_ID.get(t, t)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

# relative seat of the feeder: 0 = kamicha (left), 1 = toimen, 2 = shimocha (right)
Rel = int


def relative_seat(me: int, other: int) -> Rel:
    """0 if `other` is my kamicha (plays before me), 1 toimen, 2 shimocha."""
    d = (other - me) % 4
    return {3: 0, 2: 1, 1: 2}[d]


def _place(
    prefix: str, called: int, own: Sequence[int], rel: Rel, *, kan: bool = False
) -> str:
    parts = [str(t) for t in own]
    pos = rel
    if kan and rel == SHIMOCHA_SOURCE_OFFSET:
        pos = 3  # daiminkan from shimocha: called tile goes last of four
    parts.insert(pos, f"{prefix}{called}")
    return "".join(parts)


def chi(called: int, a: int, b: int) -> str:
    """Encode a sequence call with the claimed tile first; source must be kamicha."""
    return f"c{called}{a}{b}"


def pon(called: int, a: int, b: int, rel: Rel) -> str:
    """Encode a triplet with the called-tile position identifying its source."""
    return _place("p", called, (a, b), rel)


def daiminkan(called: int, a: int, b: int, c: int, rel: Rel) -> str:
    """Encode an open kan with the called-tile position identifying its source."""
    return _place("m", called, (a, b, c), rel, kan=True)


def ankan(t: int, *, has_aka: bool = False) -> str:
    """Encode a concealed kan with an optional red-five substitution."""
    n = deaka(t)
    first = RED_ID[n] if has_aka else n
    return f"{first}{n}{n}a{n}"


def kakan(added: int, a: int, b: int, c: int, rel: Rel) -> str:
    """Upgrade a pon into a kan.

    a, b, c are the three tiles already in the pon (any order) and `rel` is the
    pon's original feeder position; tenhou marks the added tile with 'k' at
    that slot.
    """
    return _place("k", added, (a, b, c), rel)


def discard(t: int, *, tsumogiri: bool = False, riichi: bool = False) -> int | str:
    """Encode a tile discard, using 60 for tsumogiri and an r prefix for riichi."""
    v: int | str = TSUMOGIRI if tsumogiri else t
    return f"r{v}" if riichi else v


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Win:
    """One winner of an agari result: starting-seat indices, point deltas and yaku."""

    winner: int
    from_seat: int  # == winner for tsumo
    delta: list[int]
    score_text: str = ""  # e.g. "30符4飜7700点" or "満貫8000点" / "跳満12000点"
    yaku: list[str] = field(default_factory=list)  # e.g. "立直(1飜)", "断幺九(1飜)"

    def dump(self) -> list:
        """Serialize as the [deltas, winner details] pair of an agari block.

        The ruleset has no pao: the responsible seat is the winner.
        """
        head = [self.winner, self.from_seat, self.winner]
        return [self.delta, [*head, self.score_text, *list(self.yaku)]]


@dataclass
class Agari:
    """One or more winners sharing the same hand-ending result block."""

    wins: list[Win]

    def dump(self) -> list:
        """Serialize all wins in the order expected by the Tenhou viewer."""
        out: list = [AGARI]
        for w in self.wins:
            out.extend(w.dump())
        return out


@dataclass
class Ryukyoku:
    """An exhaustive draw with its tenpai payments."""

    delta: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def dump(self) -> list:
        """Serialize as Tenhou's draw result."""
        return [RYUKYOKU, self.delta]


Result = Agari | Ryukyoku


# ---------------------------------------------------------------------------
# Kyoku / Game
# ---------------------------------------------------------------------------


@dataclass
class Kyoku:
    """A hand in starting-seat order, with tile streams and an optional result."""

    kyoku: int  # 0 = E1 ... 7 = S4 ...
    honba: int
    riichi_sticks: int
    scores: list[int]
    dora: list[int] = field(default_factory=list)
    ura: list[int] = field(default_factory=list)
    haipai: list[list[int]] = field(default_factory=lambda: [[] for _ in range(4)])
    draws: list[list[int | str]] = field(default_factory=lambda: [[] for _ in range(4)])
    discards: list[list[int | str]] = field(
        default_factory=lambda: [[] for _ in range(4)]
    )
    result: Result | None = None

    @property
    def dealer(self) -> int:
        """Starting-seat index of the dealer for this round."""
        return self.kyoku % 4

    @property
    def name(self) -> str:
        """Japanese round label for display; honba is stored separately."""
        return f"{WINDS[self.kyoku // 4]}{self.kyoku % 4 + 1}局"

    def dump(self) -> list:
        """Serialize the positional Tenhou hand array; replay it before export."""
        out: list = [
            [self.kyoku, self.honba, self.riichi_sticks],
            list(self.scores),
            list(self.dora),
            list(self.ura),
        ]
        for i in range(4):
            out += [list(self.haipai[i]), list(self.draws[i]), list(self.discards[i])]
        if self.result is not None:
            out.append(self.result.dump())
        return out


@dataclass
class Game:
    """Tenhou export container; names and every hand stream use starting-seat order."""

    names: list[str]
    title: list[str] = field(default_factory=lambda: ["", ""])
    rule_disp: str = "南喰赤"
    aka: int = 1
    kyokus: list[Kyoku] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Produce the Tenhou viewer envelope with conservative default metadata.

        These are the fields tenhou's own logs carry; the viewer (tenhou.net/5) reads
        title / name / rule / log.
        """
        return {
            "ver": "2.3",
            "ref": "",
            "log": [k.dump() for k in self.kyokus],
            "ratingc": "PF4",
            "rule": {"disp": self.rule_disp, "aka": self.aka},
            "lobby": 0,
            "dan": ["", "", "", ""],
            "rate": [1500.0, 1500.0, 1500.0, 1500.0],
            "sx": ["C", "C", "C", "C"],
            "name": list(self.names),
            "title": list(self.title),
        }

    def dumps(self) -> str:
        """Encode the game as compact Unicode JSON for Tenhou's viewer."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    def viewer_url(self, kyoku_index: int = 0) -> str:
        """URL opening the whole log in tenhou's viewer (tenhou.net/5) at a kyoku."""
        return viewer_url(self.to_dict(), kyoku_index)

    def editor_url(self, kyoku_index: int) -> str:
        """Build the tenhou.net/6 URL of one kyoku.

        tenhou.net/6 is the log editor: it loads only log[ts], and the tools that take
        its URLs take one kyoku per URL, so the URL carries that kyoku alone, in the
        editor's own export shape (title / name / rule / log).
        """
        return editor_url(self.to_dict(), kyoku_index)

    def links_html(self) -> str:
        """Build a page with the hanchan in the viewer and each kyoku's editor URL.

        The editor URLs can be copied one by one or all at once.
        """
        rows = [
            (
                f'<li><a class="k6" href="{escape(self.editor_url(i))}">'
                f'{escape(k.name)} {k.honba}本場</a> <button onclick="copy(this, '
                '[this.previousElementSibling])">copy</button></li>'
            )
            for i, k in enumerate(self.kyokus)
        ]
        title = escape(" ".join(t for t in self.title if t))
        return (
            f'<!doctype html>\n<meta charset="utf-8"><title>{title}</title>\n'
            "<style>body{font:15px/1.7 system-ui,sans-serif;margin:2em "
            "auto;max-width:44em;padding:0 16px}\nbutton{font-size:12px}</style>\n"
            f"<h1>{title}</h1>\n<p>{escape(' · '.join(self.names))}</p>\n<p><a "
            f'href="{escape(self.viewer_url())}">Replay the whole hanchan</a> '
            "(tenhou.net/5)</p>\n<p>One kyoku per URL (tenhou.net/6, the log "
            "editor; paste these into tools that take tenhou.net/6 URLs):\n<button "
            "onclick=\"copy(this, document.querySelectorAll('a.k6'))\">copy all "
            f"{len(self.kyokus)} URLs</button></p>\n<ul>\n{chr(10).join(rows)}\n"
            "</ul>\n<script>\nfunction copy(b, links) {\n  const text = "
            'Array.from(links, a => a.getAttribute("href")).join("\\n");\n  '
            "navigator.clipboard.writeText(text).then(() => b.textContent = "
            '"copied",\n    () => b.textContent = "copy failed: right-click a link,'
            ' Copy link address");\n}\n</script>\n'
        )


def viewer_url(data: dict, kyoku_index: int = 0) -> str:
    """Open a serialized Tenhou game in the viewer at a zero-based hand."""
    return "https://tenhou.net/5/#json=" + _url_json(data) + f"&ts={kyoku_index}"


def editor_url(data: dict, kyoku_index: int) -> str:
    """Build a single-hand editor link from a serialized game.

    Invalid indices raise IndexError.
    """
    hand = {key: data[key] for key in ("title", "name", "rule") if key in data}
    hand["log"] = [data["log"][kyoku_index]]
    return "https://tenhou.net/6/#json=" + _url_json(hand)


def _url_json(d: dict) -> str:
    """Encode JSON for a tenhou #json= URL as the editor does (commas kept)."""
    return quote(
        json.dumps(d, ensure_ascii=False, separators=(",", ":")), safe="-_.!~*'(),"
    )


# ---------------------------------------------------------------------------
# Replayer: legality check of a finished log
# ---------------------------------------------------------------------------

LIVE_WALL = 70  # 136 - 13 * 4 - 14


def _parse_call(s: str) -> tuple[str, list[int], int]:
    """Call string -> (kind, tiles, position of the marked tile). kind: c p m a k."""
    kind = next(ch for ch in s if ch.isalpha())
    pos = s.index(kind)
    digits = s.replace(kind, "")
    tiles = [int(digits[i : i + 2]) for i in range(0, len(digits), 2)]
    return kind, tiles, pos // 2


def _feeder(kind: str, pos: int) -> Rel:
    """Return the relative_seat a chi / pon / daiminkan took its tile from."""
    if kind == "c":
        return 0
    return {0: 0, 1: 1, 2: 2, 3: 2}[pos]


CALL_NAMES = {"c": "chi", "p": "pon", "m": "daiminkan", "a": "ankan", "k": "kakan"}


def _shape_ok(kind: str, tiles: list[int]) -> bool:
    """Whether a call's tiles form its meld.

    A chi is three consecutive tiles of one numbered suit, a pon three of one kind and
    every kan four of one kind (red fives count as fives).
    """
    kinds = sorted(deaka(t) for t in tiles)
    if kind == "c":
        return (
            len(kinds) == MELD_SIZE
            and kinds[-1] < _SUIT_BASE["z"]
            and kinds[0] // 10 == kinds[-1] // 10
            and kinds == list(range(kinds[0], kinds[0] + MELD_SIZE))
        )
    size = MELD_SIZE if kind == "p" else KAN_SIZE
    return len(kinds) == size and len(set(kinds)) == 1


def _limit(t: int) -> int:
    return 1 if t in PLAIN_ID else (3 if t in RED_ID else 4)


def _wins(concealed: Counter) -> bool:
    """Is the concealed part (melds taken out) a complete hand? Shanten -1."""
    arr = [0] * 34
    for t, n in concealed.items():
        if n > 0:
            b = deaka(t)
            arr[(b // 10 - 1) * 9 + b % 10 - 1] += n
    # Invalid replay states are domain violations, not shanten-library failures.
    if sum(arr) not in (2, 5, 8, 11, 14):
        return False
    return Shanten().calculate_shanten(arr) == -1


@dataclass(frozen=True)
class Violation:
    """One rule of play a replayed kyoku breaks.

    ``player`` is the tenhou player index (None for the whole kyoku); ``draw`` and
    ``discard`` index that player's streams where the replay found it. ``text``
    describes it without naming the player.
    """

    kyoku: str
    kind: str
    text: str
    player: int | None = None
    tile: str | None = None
    draw: int | None = None
    discard: int | None = None

    def __str__(self) -> str:
        """Describe the violation with its kyoku and player index."""
        who = "" if self.player is None else f"seat {self.player} "
        return f"{self.kyoku}: {who}{self.text}"


class _Replay:
    """One kyoku played turn by turn."""

    def __init__(self, k: list) -> None:
        head = k[0]
        self.name = f"kyoku {head[0]} ({head[0]}/{head[1]})"
        self.sticks = head[2]
        self.dealer = head[0] % 4
        self.dora, self.ura = k[2], k[3]
        self.haipai = [list(k[4 + 3 * i]) for i in range(4)]
        self.draws = [list(k[5 + 3 * i]) for i in range(4)]
        self.discards = [list(k[6 + 3 * i]) for i in range(4)]
        self.result = k[RESULT_FIELD_INDEX] if len(k) > RESULT_FIELD_INDEX else None
        self.violations: list[Violation] = []
        self.hands = [Counter(h) for h in self.haipai]
        self.di, self.ki = [0] * 4, [0] * 4  # next draw / discard entry of each seat
        # melds (a kakan is its pon), open melds
        self.sets, self.open = [0] * 4, [0] * 4
        self.riichi = [False] * 4
        self.last_draw: list[int | None] = [None] * 4
        self.wall = self.kans = self.declared = 0
        self.last: tuple[int, int, bool] | None = None  # seat, tile, a declaration
        # the seat whose draw ended the hand (a tsumo)
        self.ended_on_draw: int | None = None

    def bad(
        self,
        kind: str,
        text: str,
        player: int | None = None,
        *,
        draw: int | None = None,
        discard: int | None = None,
    ) -> None:
        """Record a violation at the player's draw or discard index, when known."""
        self.violations.append(
            Violation(self.name, kind, text, player, draw=draw, discard=discard)
        )

    def take(
        self,
        i: int,
        t: int,
        verb: str,
        *,
        draw: int | None = None,
        discard: int | None = None,
    ) -> None:
        if self.hands[i][t] <= 0:
            self.violations.append(
                Violation(
                    self.name,
                    "missing_tile",
                    f"{verb} {tile_str(t)} without it in hand",
                    i,
                    tile_str(t),
                    draw=draw,
                    discard=discard,
                )
            )
        self.hands[i][t] -= 1

    def check_shape(
        self, i: int, call: str, *, draw: int | None = None, discard: int | None = None
    ) -> tuple[str, list[int], int]:
        """Parse a call of seat i and report it when its tiles do not form its meld."""
        kind, tiles, pos = _parse_call(call)
        if not _shape_ok(kind, tiles):
            self.bad(
                "bad_meld",
                f"has a meld that is not a {CALL_NAMES[kind]}",
                i,
                draw=draw,
                discard=discard,
            )
        return kind, tiles, pos

    def draw(self, i: int, *, rinshan: bool = False) -> bool:
        """Consume seat i's next draw; False when there is none or it is a call.

        A call where a draw is due is out of turn.
        """
        if self.di[i] >= len(self.draws[i]):
            return False
        x = self.draws[i][self.di[i]]
        if isinstance(x, str):
            due = "replacement draw" if rinshan else "draw"
            self.bad(
                "out_of_turn",
                f"calls where a {due} is due (out of turn)",
                i,
                draw=self.di[i],
            )
            return False
        self.di[i] += 1
        self.hands[i][x] += 1
        self.last_draw[i] = x
        self.wall += 0 if rinshan else 1
        return True

    def _self_kan(self, i: int, dsc: str) -> None:
        """Consume a self-kan and require its replacement draw before proceeding."""
        at = self.ki[i] - 1  # the kan's entry in the discards
        kind, tiles, pos = self.check_shape(i, dsc, discard=at)
        if kind == "a":
            for t in tiles:
                self.take(i, t, "kans", discard=at)
            self.sets[i] += 1
        else:
            self.take(i, tiles[pos], "adds", discard=at)
            if self.riichi[i]:
                self.bad("riichi_kan", "adds a kan tile after riichi", i, discard=at)
        self.kans += 1
        if not self.draw(i, rinshan=True) and self.ki[i] < len(self.discards[i]):
            self.bad("no_rinshan", "has no replacement draw after a kan", i, discard=at)

    def discard(self, i: int) -> tuple[int, bool] | None:
        """Consume seat i's discard after any self-kans and their rinshan draws.

        None when the hand ended on its draw.
        """
        while self.ki[i] < len(self.discards[i]):
            dsc = self.discards[i][self.ki[i]]
            self.ki[i] += 1
            if isinstance(dsc, str) and ("a" in dsc or "k" in dsc):
                self._self_kan(i, dsc)
                continue
            is_r = isinstance(dsc, str) and dsc.startswith("r")
            v = int(str(dsc).lstrip("r"))
            at = self.ki[i] - 1
            if v == 0:
                self.bad(
                    "out_of_turn",
                    "has a kan placeholder where a discard is due",
                    i,
                    discard=at,
                )
                continue
            if v == TSUMOGIRI:
                tile = self.last_draw[i]
                if tile is None:
                    self.bad(
                        "no_draw", "discards a drawn tile without a draw", i, discard=at
                    )
                    continue
            else:
                tile = v
                if self.riichi[i]:
                    self.bad(
                        "riichi_discard",
                        f"discards {tile_str(v)} from the hand after riichi",
                        i,
                        discard=at,
                    )
            self.take(i, tile, "discards", discard=at)
            if is_r:
                if self.open[i]:
                    self.bad(
                        "open_riichi",
                        "declares riichi with an open hand",
                        i,
                        discard=at,
                    )
                if self.riichi[i]:
                    self.bad("riichi_twice", "declares riichi twice", i, discard=at)
                self.riichi[i] = True
                self.declared += 1
            self.last_draw[i] = None
            return tile, is_r
        return None

    def caller(self, cur: int, tile: int) -> int | None:
        """Find the seat whose next draw calls this discard (pon or kan before chi)."""
        found = []
        for j in ((cur + 1) % 4, (cur + 2) % 4, (cur + 3) % 4):
            if self.di[j] < len(self.draws[j]) and isinstance(
                self.draws[j][self.di[j]], str
            ):
                kind, tiles, pos = _parse_call(self.draws[j][self.di[j]])
                if (
                    kind in "cpm"
                    and deaka(tiles[pos]) == deaka(tile)
                    and relative_seat(j, cur) == _feeder(kind, pos)
                ):
                    found.append((kind == "c", j))
        return min(found)[1] if found else None

    def call(self, j: int) -> None:
        at = self.di[j]  # the call's entry in the draws
        call = str(self.draws[j][at])
        self.di[j] += 1
        kind, tiles, pos = self.check_shape(j, call, draw=at)
        for q, t in enumerate(tiles):
            if q != pos:
                self.take(j, t, "calls with", draw=at)
        self.sets[j] += 1
        self.open[j] += 1
        if self.riichi[j]:
            self.bad("riichi_call", "calls after riichi", j, draw=at)
        if kind == "m":
            self.kans += 1
            if self.ki[j] < len(self.discards[j]) and self.discards[j][self.ki[j]] == 0:
                self.ki[j] += 1
            else:
                self.bad(
                    "kan_placeholder",
                    "has an open kan without its placeholder discard",
                    j,
                    draw=at,
                )
            if not self.draw(j, rinshan=True):
                self.bad(
                    "no_rinshan", "has no replacement draw after a kan", j, draw=at
                )

    def play(self) -> None:
        cur, need_draw = self.dealer, True
        for _ in range(500):
            if need_draw and not self.draw(cur):
                return  # the hand ended before this draw (a ron, or the wall ran out)
            d = self.discard(cur)
            if d is None:
                # no discard after the draw: the hand ended on it (a tsumo)
                self.ended_on_draw = cur
                return
            tile, is_r = d
            self.last = (cur, tile, is_r)
            j = self.caller(cur, tile)
            if j is None:
                cur, need_draw = (cur + 1) % 4, True
            else:
                self.call(j)
                cur, need_draw = j, False

    def _check_inventory(self) -> None:
        """Validate starting hand sizes and global tile-copy limits."""
        for i in range(4):
            if len(self.haipai[i]) != STARTING_HAND_SIZE:
                self.bad("haipai_size", f"starts with {len(self.haipai[i])} tiles", i)
        seen = (
            Counter(t for h in self.haipai for t in h)
            + Counter(self.dora)
            + Counter(self.ura)
        )
        seen += Counter(t for d in self.draws for t in d if isinstance(t, int))
        for t, c in sorted(seen.items()):
            if c > _limit(t):
                self.violations.append(
                    Violation(
                        self.name,
                        "over_count",
                        f"{tile_str(t)} appears {c} times (hands, draws and "
                        "indicators)",
                        tile=tile_str(t),
                    )
                )

    def _check_progress(self) -> None:
        """Check consumed streams, wall accounting and indicator counts."""
        for i in range(4):
            if self.di[i] < len(self.draws[i]) or self.ki[i] < len(self.discards[i]):
                self.bad(
                    "unplayed",
                    f"has {len(self.draws[i]) - self.di[i]} draw(s) and "
                    f"{len(self.discards[i]) - self.ki[i]} discard(s) that never "
                    "came to be played (out of turn, or more discards than draws)",
                    i,
                    draw=self.di[i],
                    discard=self.ki[i],
                )
        if self.wall + self.kans > LIVE_WALL:
            self.bad(
                "wall",
                f"{self.wall} wall draws and {self.kans} kan(s) exceed the live"
                f" wall of {LIVE_WALL}",
            )
        # every kan reveals an indicator; ura, when given, lie under each of them
        if len(self.dora) != 1 + self.kans:
            self.bad(
                "indicators",
                f"{len(self.dora)} dora indicator(s) for {self.kans} kan(s): a "
                f"log needs {1 + self.kans}",
            )
        if self.ura and len(self.ura) != len(self.dora):
            self.bad(
                "ura",
                f"{len(self.ura)} ura indicator(s) under {len(self.dora)} dora "
                "indicator(s)",
            )

    def _check_result(self) -> None:
        """Compare the recorded outcome with the replayed ending and payments."""
        res: list = (
            self.result if isinstance(self.result, list) and self.result else [None]
        )
        if res[0] == AGARI:
            for w in range(1, len(res), 2):
                self._check_win(res[w], res[w + 1])
        elif res[0] == RYUKYOKU:
            if len(res) > 1 and sum(res[1]) != 0:
                self.bad("deltas", f"the deltas of the draw sum to {sum(res[1])}")
            if self.wall + self.kans < LIVE_WALL:
                self.bad(
                    "short_wall",
                    f"an exhaustive draw after {self.wall} wall draws and "
                    f"{self.kans} kan(s), not {LIVE_WALL}",
                )

    def _check_win(self, delta: list[int], info: list) -> None:
        """Check the winner, winning hand and riichi-stick payment of one win."""
        winner, frm = info[0], info[1]
        hand = Counter(self.hands[winner])
        if winner != frm:
            if self.last is None or self.last[0] != frm:
                self.bad(
                    "not_last_discard",
                    "wins by ron on a discard that was not the last",
                    winner,
                )
            else:
                hand[self.last[1]] += 1
                if self.last[2]:
                    self.declared -= 1  # a declaration ronned on: its stick is not paid
        elif self.ended_on_draw != winner:
            self.bad(
                "not_on_draw",
                "wins by tsumo, but the hand did not end on its draw",
                winner,
            )
        if not _wins(hand):
            self.bad("not_winning", "does not hold a winning hand", winner)
        if sum(delta) != 1000 * (self.sticks + self.declared):
            self.bad(
                "deltas",
                f"the deltas sum to {sum(delta)}, not the "
                f"{1000 * (self.sticks + self.declared)} of the riichi sticks",
            )

    def check(self) -> list[Violation]:
        self._check_inventory()
        self.play()
        self._check_progress()
        self._check_result()
        for i in range(4):
            want = 13 - 3 * self.sets[i] + (1 if self.ended_on_draw == i else 0)
            n = sum(self.hands[i].values())
            if n != want or any(c < 0 for c in self.hands[i].values()):
                self.bad("hand_size", f"ends with {n} tiles in hand, not {want}", i)
        return self.violations


def replay_kyoku(k: list) -> list[Violation]:
    """Replay one kyoku of a tenhou/6 log turn by turn; return its violations.

    The real interleaving is simulated (the dealer first; after each discard, a call in
    another seat's draw list naming that tile from that seat takes the turn). An empty
    list means the hand is legal.
    """
    return _Replay(k).check()


def replay(game_dict: dict) -> list[Violation]:
    """Violations of every kyoku of a tenhou/6 log dict (empty = legal)."""
    return [p for k in game_dict["log"] for p in replay_kyoku(k)]
