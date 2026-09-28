"""Data model and serializer for the tenhou.net/6 JSON log format.

The format (as consumed by tenhou.net/6, mjai-reviewer, Mortal, tensoul, ...):

    {
      "title": ["<title>", "<subtitle>"],
      "name":  ["<seat0>", "<seat1>", "<seat2>", "<seat3>"],
      "rule":  {"disp": "<rule text>", "aka": 1},
      "log":   [ <kyoku>, <kyoku>, ... ]
    }

    <kyoku> = [
      [kyoku_index, honba, riichi_sticks],   # kyoku_index: 0=E1 .. 3=E4, 4=S1 .. 7=S4, 8=W1 ..
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
51/52/53 = red 5m/5p/5s. The dealer's haipai has 13 tiles and their first
draw is the 14th tile. A discard of 60 means tsumogiri (discarded the tile just
drawn). A riichi declaration prefixes the discard with "r" (e.g. "r60", "r23").

Calls are strings placed in *draws* (chi / pon / daiminkan, since they replace a
draw) or in *discards* (ankan / kakan, since they replace a discard; a
daiminkan also appends a literal 0 to discards). The called tile's position in
the string marks who it came from (left = kamicha, middle = toimen,
right = shimocha):

    chi    "c<called><a><b>"                  always from kamicha
    pon    "p<called><a><b>" | "<a>p<called><b>" | "<a><b>p<called>"
    daiminkan  "m<called><a><b><c>" | "<a>m<called><b><c>" | "<a><b><c>m<called>"
    ankan  "<t><t><t>a<t>"
    kakan  "k<t><a><b><c>" | "<a>k<t><b><c>" | "<a><b>k<t><c>"   (position = original pon)

Result:
    ["和了", [delta...], [winner, from, pao, "<fu>符<han>飜<pts>点", "yaku(1飜)", ...], ...]
    ["流局", [delta...]]
    ["九種九牌"] / ["四風連打"] / ["四家立直"] / ["四開槓"] / ["三家和"] / ["流し満貫", [delta...]]
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

TSUMOGIRI = 60
WINDS = "東南西北"
AGARI = "和了"
RYUKYOKU = "流局"

# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------

_SUIT_BASE = {"m": 10, "p": 20, "s": 30, "z": 40}
_AKA = {"m": 51, "p": 52, "s": 53}


def tile(s: str) -> int:
    """Parse a tile in mjai/tenhou short notation: '5m', '0p' (red 5p), '1z' (E) ... '7z' (R)."""
    if len(s) != 2 or s[1] not in _SUIT_BASE or not s[0].isdigit():
        raise ValueError(f"bad tile {s!r}")
    n, suit = int(s[0]), s[1]
    if n == 0:
        if suit == "z":
            raise ValueError("no red honor tile")
        return _AKA[suit]
    if suit == "z" and not 1 <= n <= 7:
        raise ValueError(f"bad honor {s!r}")
    if suit != "z" and not 1 <= n <= 9:
        raise ValueError(f"bad number {s!r}")
    return _SUIT_BASE[suit] + n


def tile_str(t: int) -> str:
    """Inverse of :func:`tile`."""
    if t in (51, 52, 53):
        return "0" + "mps"[t - 51]
    suit = {1: "m", 2: "p", 3: "s", 4: "z"}[t // 10]
    return f"{t % 10}{suit}"


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
    return {51: 15, 52: 25, 53: 35}.get(t, t)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

Rel = int  # relative seat of the feeder: 0 = kamicha (left), 1 = toimen, 2 = shimocha (right)


def relative_seat(me: int, other: int) -> Rel:
    """0 if `other` is my kamicha (plays before me), 1 toimen, 2 shimocha."""
    d = (other - me) % 4
    return {3: 0, 2: 1, 1: 2}[d]


def _place(prefix: str, called: int, own: Sequence[int], rel: Rel, *, kan: bool = False) -> str:
    parts = [str(t) for t in own]
    pos = rel
    if kan and rel == 2:
        pos = 3  # daiminkan from shimocha: called tile goes last of four
    parts.insert(pos, f"{prefix}{called}")
    return "".join(parts)


def chi(called: int, a: int, b: int) -> str:
    """Encode a sequence call with the claimed tile first; source must be kamicha."""
    return f"c{called}{a}{b}"


def pon(called: int, a: int, b: int, rel: Rel) -> str:
    """Encode a triplet with the called-tile position identifying its relative source seat."""
    return _place("p", called, (a, b), rel)


def daiminkan(called: int, a: int, b: int, c: int, rel: Rel) -> str:
    """Encode an open kan with the called-tile position identifying its relative source seat."""
    return _place("m", called, (a, b, c), rel, kan=True)


def ankan(t: int, *, has_aka: bool = False) -> str:
    """Encode a concealed kan, optionally replacing one ordinary five with the red five."""
    n = deaka(t)
    first = {15: 51, 25: 52, 35: 53}[n] if has_aka else n
    return f"{first}{n}{n}a{n}"


def kakan(added: int, a: int, b: int, c: int, rel: Rel) -> str:
    """Upgrade a pon into a kan.

    a, b, c are the three tiles already in the pon (any order) and `rel` is the
    pon's original feeder position; tenhou marks the added tile with 'k' at
    that slot.
    """
    return _place("k", added, (a, b, c), rel)


def discard(t: int, *, tsumogiri: bool = False, riichi: bool = False) -> Union[int, str]:
    """Encode a tile discard, using 60 for tsumogiri and an r prefix for riichi."""
    v: Union[int, str] = TSUMOGIRI if tsumogiri else t
    return f"r{v}" if riichi else v


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Win:
    """One winner in an agari result, with starting-seat indices, point deltas and yaku text."""
    winner: int
    from_seat: int  # == winner for tsumo
    delta: list[int]
    score_text: str = ""  # e.g. "30符4飜7700点" or "満貫8000点" / "跳満12000点"
    yaku: list[str] = field(default_factory=list)  # e.g. "立直(1飜)", "断幺九(1飜)"
    pao: Optional[int] = None

    def dump(self) -> list:
        """Tenhou result pair [deltas, winner details], ready to append to an agari block."""
        head = [self.winner, self.from_seat, self.pao if self.pao is not None else self.winner]
        return [self.delta, head + [self.score_text] + list(self.yaku)]


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
    """Exhaustive or abortive draw; only exhaustive/nagashi results include point deltas."""
    delta: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    kind: str = RYUKYOKU  # 流局 / 流し満貫 / 九種九牌 / 四風連打 / 四家立直 / 四開槓 / 三家和

    def dump(self) -> list:
        """Serialize a draw result, omitting point deltas for abortive draw kinds."""
        if self.kind in (RYUKYOKU, "流し満貫"):
            return [self.kind, self.delta]
        return [self.kind]


Result = Union[Agari, Ryukyoku]


# ---------------------------------------------------------------------------
# Kyoku / Game
# ---------------------------------------------------------------------------


@dataclass
class Kyoku:
    """One hand in starting-seat order, with draw/call/discard streams and optional result."""
    kyoku: int  # 0 = E1 ... 7 = S4 ...
    honba: int
    riichi_sticks: int
    scores: list[int]
    dora: list[int] = field(default_factory=list)
    ura: list[int] = field(default_factory=list)
    haipai: list[list[int]] = field(default_factory=lambda: [[] for _ in range(4)])
    draws: list[list[Union[int, str]]] = field(default_factory=lambda: [[] for _ in range(4)])
    discards: list[list[Union[int, str]]] = field(default_factory=lambda: [[] for _ in range(4)])
    result: Optional[Result] = None

    @property
    def dealer(self) -> int:
        """Starting-seat index of the dealer for this round."""
        return self.kyoku % 4

    @property
    def name(self) -> str:
        """Japanese round label for display; honba is stored separately."""
        return f"{WINDS[self.kyoku // 4]}{self.kyoku % 4 + 1}局"

    def dump(self) -> list:
        """Serialize the positional Tenhou hand array; callers should replay-validate before export."""
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
    """Tenhou export container; names and every hand stream use the same starting-seat order."""
    names: list[str]
    title: list[str] = field(default_factory=lambda: ["", ""])
    rule_disp: str = "南喰赤"
    aka: int = 1
    kyokus: list[Kyoku] = field(default_factory=list)

    def to_dict(self) -> dict:
        # the fields tenhou's own logs carry; the viewer (tenhou.net/5) reads title/name/rule/log
        """Produce the Tenhou viewer envelope with conservative default metadata."""
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

    def dumps(self, **kw) -> str:
        """Encode compact Unicode JSON; keyword arguments override JSON encoder defaults."""
        kw.setdefault("ensure_ascii", False)
        kw.setdefault("separators", (",", ":"))
        return json.dumps(self.to_dict(), **kw)

    def viewer_url(self, kyoku_index: int = 0) -> str:
        """URL that opens the whole log in tenhou's viewer (tenhou.net/5), starting at kyoku `kyoku_index`."""
        return viewer_url(self.to_dict(), kyoku_index)

    def editor_url(self, kyoku_index: int) -> str:
        """tenhou.net/6 URL of one kyoku. tenhou.net/6 is the log editor: it loads only log[ts], and the tools
        that take its URLs take one kyoku per URL, so the URL carries that kyoku alone, in the editor's own
        export shape (title / name / rule / log)."""
        return editor_url(self.to_dict(), kyoku_index)

    def links_html(self) -> str:
        """A page with the whole hanchan in the viewer and the editor URL of each kyoku, one by one or all at once."""
        from html import escape

        rows = [f'<li><a class="k6" href="{escape(self.editor_url(i))}">{escape(k.name)} {k.honba}本場</a>'
                f' <button onclick="copy(this, [this.previousElementSibling])">copy</button></li>'
                for i, k in enumerate(self.kyokus)]
        title = escape(" ".join(t for t in self.title if t))
        return f"""<!doctype html>
<meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px/1.7 system-ui,sans-serif;margin:2em auto;max-width:44em;padding:0 16px}}
button{{font-size:12px}}</style>
<h1>{title}</h1>
<p>{escape(" · ".join(self.names))}</p>
<p><a href="{escape(self.viewer_url())}">Replay the whole hanchan</a> (tenhou.net/5)</p>
<p>One kyoku per URL (tenhou.net/6, the log editor; paste these into tools that take tenhou.net/6 URLs):
<button onclick="copy(this, document.querySelectorAll('a.k6'))">copy all {len(self.kyokus)} URLs</button></p>
<ul>
{chr(10).join(rows)}
</ul>
<script>
function copy(b, links) {{
  const text = Array.from(links, a => a.getAttribute("href")).join("\\n");
  navigator.clipboard.writeText(text).then(() => b.textContent = "copied",
    () => b.textContent = "copy failed: right-click a link, Copy link address");
}}
</script>
"""


def viewer_url(data: dict, kyoku_index: int = 0) -> str:
    """Open a serialized Tenhou game at a zero-based hand, preserving its metadata."""
    return "https://tenhou.net/5/#json=" + _url_json(data) + f"&ts={kyoku_index}"


def editor_url(data: dict, kyoku_index: int) -> str:
    """Build a single-hand editor link from a serialized game; invalid indices raise IndexError."""
    hand = {key: data[key] for key in ("title", "name", "rule") if key in data}
    hand["log"] = [data["log"][kyoku_index]]
    return "https://tenhou.net/6/#json=" + _url_json(hand)


def _url_json(d: dict) -> str:
    """JSON for a tenhou #json= URL, encoded as the editor does it (encodeURIComponent, commas kept)."""
    from urllib.parse import quote

    return quote(json.dumps(d, ensure_ascii=False, separators=(",", ":")), safe="-_.!~*'(),")


# ---------------------------------------------------------------------------
# Replayer: legality check of a finished log
# ---------------------------------------------------------------------------

LIVE_WALL = 70          # 136 - 13 * 4 - 14


def _parse_call(s: str) -> tuple[str, list[int], int]:
    """Call string -> (kind, tiles, position of the marked tile). kind: c p m a k."""
    kind = next(ch for ch in s if ch.isalpha())
    pos = s.index(kind)
    digits = s.replace(kind, "")
    tiles = [int(digits[i: i + 2]) for i in range(0, len(digits), 2)]
    return kind, tiles, pos // 2


def _feeder(kind: str, pos: int) -> Rel:
    """relative_seat of the player a chi / pon / daiminkan took its tile from, by the marked position."""
    if kind == "c":
        return 0
    return {0: 0, 1: 1, 2: 2, 3: 2}[pos]


def _limit(t: int) -> int:
    return 1 if t in (51, 52, 53) else (3 if t in (15, 25, 35) else 4)


def _wins(concealed: Counter) -> bool:
    """Is this concealed part (melds taken out) a complete hand? Shanten -1 with the mahjong library."""
    from mahjong.shanten import Shanten
    arr = [0] * 34
    for t, n in concealed.items():
        if n > 0:
            b = deaka(t)
            arr[(b // 10 - 1) * 9 + b % 10 - 1] += n
    try:
        return Shanten().calculate_shanten(arr) == -1
    except Exception:  # noqa: BLE001
        return False


class _Replay:
    """One kyoku played turn by turn."""

    def __init__(self, k: list):
        head = k[0]
        self.name = f"kyoku {head[0]} ({head[0]}/{head[1]})"
        self.sticks = head[2]
        self.dealer = head[0] % 4
        self.dora, self.ura = k[2], k[3]
        self.haipai = [list(k[4 + 3 * i]) for i in range(4)]
        self.draws = [list(k[5 + 3 * i]) for i in range(4)]
        self.discards = [list(k[6 + 3 * i]) for i in range(4)]
        self.result = k[16] if len(k) > 16 else None
        self.problems: list[str] = []
        self.hands = [Counter(h) for h in self.haipai]
        self.di, self.ki = [0] * 4, [0] * 4                    # next draw / discard entry of each seat
        self.sets, self.open = [0] * 4, [0] * 4                # melds (a kakan is its pon), open melds
        self.riichi = [False] * 4
        self.last_draw: list[Optional[int]] = [None] * 4
        self.wall = self.kans = self.declared = 0
        self.last: Optional[tuple[int, int, bool]] = None      # (seat, tile, a riichi declaration)
        self.ended_on_draw: Optional[int] = None               # the seat whose draw ended the hand (a tsumo)

    def bad(self, text: str) -> None:
        self.problems.append(f"{self.name}: {text}")

    def take(self, i: int, t: int, what: str) -> None:
        if self.hands[i][t] <= 0:
            self.bad(f"seat {i} {what} without {tile_str(t)} in hand")
        self.hands[i][t] -= 1

    def draw(self, i: int, rinshan: bool = False) -> bool:
        """Seat i draws its next entry; False when there is none (or the entry is a call: out of turn)."""
        if self.di[i] >= len(self.draws[i]):
            return False
        x = self.draws[i][self.di[i]]
        if isinstance(x, str):
            self.bad(f"seat {i} has the call {x} where a {'rinshan ' if rinshan else ''}draw is due (out of turn)")
            return False
        self.di[i] += 1
        self.hands[i][x] += 1
        self.last_draw[i] = x
        self.wall += 0 if rinshan else 1
        return True

    def discard(self, i: int) -> Optional[tuple[int, bool]]:
        """Seat i's discard after any ankan / kakan (each followed by its rinshan draw); None when the hand ended
        on its draw."""
        while self.ki[i] < len(self.discards[i]):
            dsc = self.discards[i][self.ki[i]]
            self.ki[i] += 1
            if isinstance(dsc, str) and ("a" in dsc or "k" in dsc):
                kind, tiles, pos = _parse_call(dsc)
                if kind == "a":
                    for t in tiles:
                        self.take(i, t, f"ankan {dsc}")
                    self.sets[i] += 1
                else:
                    self.take(i, tiles[pos], f"kakan {dsc}")
                    if self.riichi[i]:
                        self.bad(f"seat {i} kakan after riichi")
                self.kans += 1
                if not self.draw(i, rinshan=True) and self.ki[i] < len(self.discards[i]):
                    self.bad(f"seat {i} has no rinshan draw after {dsc}")
                continue
            is_r = isinstance(dsc, str) and dsc.startswith("r")
            v = int(str(dsc).lstrip("r"))
            if v == 0:
                self.bad(f"seat {i} has a daiminkan placeholder where a discard is due")
                continue
            if v == TSUMOGIRI:
                if self.last_draw[i] is None:
                    self.bad(f"seat {i} tsumogiri without a draw (discard {self.ki[i] - 1})")
                    continue
                tile = self.last_draw[i]
            else:
                tile = v
                if self.riichi[i]:
                    self.bad(f"seat {i} discards a hand tile {tile_str(v)} after riichi")
            self.take(i, tile, f"discards {tile_str(tile)} (discard {self.ki[i] - 1})")
            if is_r:
                if self.open[i]:
                    self.bad(f"seat {i} riichi with an open hand")
                if self.riichi[i]:
                    self.bad(f"seat {i} declares riichi twice")
                self.riichi[i] = True
                self.declared += 1
            self.last_draw[i] = None
            return tile, is_r
        return None

    def caller(self, cur: int, tile: int) -> Optional[int]:
        """The seat whose next draw entry is a call on this discard (a pon or kan before a chi)."""
        found = []
        for j in ((cur + 1) % 4, (cur + 2) % 4, (cur + 3) % 4):
            if self.di[j] < len(self.draws[j]) and isinstance(self.draws[j][self.di[j]], str):
                kind, tiles, pos = _parse_call(self.draws[j][self.di[j]])
                if kind in "cpm" and deaka(tiles[pos]) == deaka(tile) and relative_seat(j, cur) == _feeder(kind, pos):
                    found.append((kind == "c", j))
        return min(found)[1] if found else None

    def call(self, j: int) -> None:
        call = self.draws[j][self.di[j]]
        self.di[j] += 1
        kind, tiles, pos = _parse_call(call)
        for q, t in enumerate(tiles):
            if q != pos:
                self.take(j, t, f"calls {call}")
        self.sets[j] += 1
        self.open[j] += 1
        if self.riichi[j]:
            self.bad(f"seat {j} calls {call} after riichi")
        if kind == "m":
            self.kans += 1
            if self.ki[j] < len(self.discards[j]) and self.discards[j][self.ki[j]] == 0:
                self.ki[j] += 1
            else:
                self.bad(f"seat {j} daiminkan {call} without its 0 in the discards")
            if not self.draw(j, rinshan=True):
                self.bad(f"seat {j} has no rinshan draw after {call}")

    def play(self) -> None:
        cur, need_draw = self.dealer, True
        for _ in range(500):
            if need_draw and not self.draw(cur):
                return                                 # the hand ended before this draw (a ron, or the wall ran out)
            d = self.discard(cur)
            if d is None:
                self.ended_on_draw = cur               # no discard after the draw: the hand ended on it (a tsumo)
                return
            tile, is_r = d
            self.last = (cur, tile, is_r)
            j = self.caller(cur, tile)
            if j is None:
                cur, need_draw = (cur + 1) % 4, True
            else:
                self.call(j)
                cur, need_draw = j, False

    def check(self) -> list[str]:
        for i in range(4):
            if len(self.haipai[i]) != 13:
                self.bad(f"seat {i} haipai has {len(self.haipai[i])} tiles")
        seen = Counter(t for h in self.haipai for t in h) + Counter(self.dora) + Counter(self.ura)
        seen += Counter(t for d in self.draws for t in d if isinstance(t, int))
        for t, c in sorted(seen.items()):
            if c > _limit(t):
                self.bad(f"{tile_str(t)} appears {c} times (hands, draws and indicators)")
        self.play()
        for i in range(4):
            if self.di[i] < len(self.draws[i]) or self.ki[i] < len(self.discards[i]):
                self.bad(f"seat {i} has {len(self.draws[i]) - self.di[i]} draw(s) and {len(self.discards[i]) - self.ki[i]} "
                         f"discard(s) that never came to be played (out of turn, or more discards than draws)")
        if self.wall + self.kans > LIVE_WALL:
            self.bad(f"{self.wall} wall draws and {self.kans} kan(s) exceed the live wall of {LIVE_WALL}")
        # every kan reveals an indicator; ura, when given, lie under each of them
        if len(self.dora) != 1 + self.kans:
            self.bad(f"{len(self.dora)} dora indicator(s) for {self.kans} kan(s): a log needs {1 + self.kans}")
        if self.ura and len(self.ura) != len(self.dora):
            self.bad(f"{len(self.ura)} ura indicator(s) under {len(self.dora)} dora indicator(s)")
        res = self.result if isinstance(self.result, list) and self.result else [None]
        if res[0] == AGARI:
            for w in range(1, len(res), 2):
                delta, info = res[w], res[w + 1]
                winner, frm = info[0], info[1]
                hand = Counter(self.hands[winner])
                if winner != frm:
                    if self.last is None or self.last[0] != frm:
                        self.bad(f"ron by seat {winner} on seat {frm}, whose discard was not the last")
                    else:
                        hand[self.last[1]] += 1
                        if self.last[2]:
                            self.declared -= 1         # a declaration ronned on: its stick is not paid
                elif self.ended_on_draw != winner:
                    self.bad(f"tsumo by seat {winner}, but the hand did not end on its draw")
                if not _wins(hand):
                    self.bad(f"the winner, seat {winner}, does not hold a winning hand")
                if sum(delta) != 1000 * (self.sticks + self.declared):
                    self.bad(f"the deltas sum to {sum(delta)}, not the {1000 * (self.sticks + self.declared)} of the riichi sticks")
        elif res[0] == RYUKYOKU:
            if len(res) > 1 and sum(res[1]) != 0:
                self.bad(f"the deltas of the draw sum to {sum(res[1])}, not 0")
            if self.wall + self.kans < LIVE_WALL:
                self.bad(f"an exhaustive draw after {self.wall} wall draws and {self.kans} kan(s), not {LIVE_WALL}")
        for i in range(4):
            want = 13 - 3 * self.sets[i] + (1 if self.ended_on_draw == i else 0)
            n = sum(self.hands[i].values())
            if n != want or any(c < 0 for c in self.hands[i].values()):
                self.bad(f"seat {i} ends with {n} tiles in hand, not {want}")
        return self.problems


def replay_kyoku(k: list) -> list[str]:
    """Simulate one kyoku of a tenhou/6 log turn by turn, in the real interleaving (the dealer first; after each
    discard, a call in another seat's draw list naming that tile from that seat takes the turn), and return
    its violations (empty = legal)."""
    return _Replay(k).check()


def replay(game_dict: dict) -> list[str]:
    """Violations of every kyoku of a tenhou/6 log dict (empty = legal)."""
    return [p for k in game_dict["log"] for p in replay_kyoku(k)]
