# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Meld observations of one seat -> meld events, and the legal melds a call can be.

DESIGN.md 4.8 `melds.py`. A meld camera shows the player's melds as rows of three or
four tiles; a called tile lies sideways and its position (left / middle / right) names
the player it came from. A new group in the observations is a meld event; its type
follows from the tiles (chi: consecutive same suit; pon / kan: identical; ankan: two
face-down). A group is a meld only as far as its readings support it: four boxes are a
kan only when every one of them reads as that kind, otherwise the best meld of three of
them is taken and the fourth box is noise; a run of five or more boxes is several melds
laid close together and is split into melds of three or four. The camera is the weakest
witness of a call: an event becomes a call only when anchored on the discard it took
(calls.py), and its tiles are then a choice among `meld_options`.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES

from . import rules

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_SUIT_RANK = 9
CONFIRMED_SIDEWAYS_FRACTION = 0.5
MIN_SIDEWAYS_FRACTION = 0.2
OPEN_MELD_SIZE = 3
KAN_SIZE = 4
ANKAN_BACK_COUNT = 2
MIN_SHARED_MELD_TILES = 2
MIN_KAKAN_BASE_VIEWS = 2
MIN_FRAGMENT_REPEATS = 2
MIN_FRAGMENT_SIZE = 2
FLOOR = 0.15  # a meld hypothesis needs each tile's reading at this posterior on average
KAN_TYPES = ("kan", "ankan", "kakan")


@dataclass(kw_only=True)
class Call:
    """An observed or reviewed meld and its candidate legal interpretations.

    Camera order determines called_pos; source is relative to the caller.
    A camera hypothesis becomes a call only once anchored to a discard or kan.
    """

    seat: str
    t_first: float  # start of the first observation showing the meld
    t_window: tuple[float, float]  # when the call happened
    type: str  # chi | pon | kan | ankan | kakan (fragment: a camera event only)
    tiles: list[str]  # tiles left to right as laid
    called_pos: int | None = None  # index of the sideways tile (None for ankan)
    source: str | None = None  # kamicha | toimen | shimocha | None
    called_tile: str | None = None
    p: list = field(default_factory=list)  # per tile posterior
    conf: float = 0.0
    seen: int = 0
    partial_only: bool = False  # seen only in partial (read-floor) observations so far
    # kakan: later full views that showed the pon with three tiles again
    unseen: int = 0
    absent: int = 0  # later full views of the camera without this meld (melds stay)
    human: bool = False  # stated by the reviewer (a meld fact)
    # how the call is established: camera (an event, not yet a call) | discard | kan |
    # indicator | hidden | fact
    anchor: str = "camera"
    # MeldOption: the legal compositions the solver chooses among
    options: list = field(default_factory=list)
    # later incomplete views outnumbered sightings: needs independent discard evidence
    contradicted: bool = False

    def to_dict(self) -> dict:
        """Serialize rounded posteriors and all alternative meld compositions."""
        d = self.__dict__.copy()
        d["p"] = [[round(float(v), 4) for v in p] for p in self.p]
        d["t_window"] = list(self.t_window)
        d["options"] = [o.to_dict() for o in self.options]
        return d


@dataclass
class MeldOption:
    """One legal composition of a call on a known called tile."""

    type: str  # chi | pon | kan (a daiminkan)
    tiles: list[str]  # as laid, the called tile at called_pos
    called_pos: int
    hand: list[str]  # the tiles that leave the caller's hand
    cost: float  # how much less the camera's readings support it than the best option

    def to_dict(self) -> dict:
        """Serialize a candidate and the concealed tiles it removes."""
        return {
            "type": self.type,
            "tiles": self.tiles,
            "called_pos": self.called_pos,
            "hand": self.hand,
            "cost": round(self.cost, 3),
        }


POSITION = {
    "pon": {"kamicha": 0, "toimen": 1, "shimocha": 2},
    "kan": {"kamicha": 0, "toimen": 1, "shimocha": 3},
}
OPTION_WEIGHT = 0.5  # solver cost per unit of log-support below the best option ...
OPTION_CAP = 5.0  # ... at most this


def _variants(tiles: list[str]) -> list[list[str]]:
    """Every choice of plain or red for the fives among `tiles`, at most one red."""
    out: list[list[str]] = [[]]
    for t in tiles:
        red = rules.RED_OF.get(t)
        out = [[*o, v] for o in out for v in ([t, red] if red else [t])]
    return [o for o in out if sum(1 for x in o if x in rules.PLAIN_OF) <= 1]


def _support(tiles: list[str], ps: list[np.ndarray]) -> float:
    """Score how well boxes read as these tiles (log posterior).

    Each tile takes the unused box that reads it best; a tile with no box left (the
    camera showed fewer boxes) is neutral.
    """
    free = list(range(len(ps)))
    score = 0.0
    for t in tiles:
        if not free:
            score += math.log(FLOOR)
            continue
        i = max(free, key=lambda i: _lp(ps[i], t))
        score += _lp(ps[i], t)
        free.remove(i)
    return score


def meld_options(
    called: str, source: str, ps: list[np.ndarray], *, four: bool
) -> list[MeldOption]:
    """Enumerate the legal melds a seat can have laid with `called` from `source`.

    A daiminkan when the camera shows four boxes, else a pon, and a chi when the tile
    came from the kamicha (DESIGN.md 4.8, call anchor step 3). Each is costed by how
    well the camera's readings `ps` support its tiles; the solver decides with the hand.
    """
    k = rules.plain(called)
    raw: list[tuple[str, list[str]]] = []
    if four:
        raw += [("kan", h) for h in _variants([k, k, k])]
    else:
        raw += [("pon", h) for h in _variants([k, k])]
        if source == "kamicha" and k[1] != "z":
            n, suit = rules.number(k), k[1]
            for a, b in ((n - 2, n - 1), (n - 1, n + 1), (n + 1, n + 2)):
                if 1 <= a <= MAX_SUIT_RANK and 1 <= b <= MAX_SUIT_RANK:
                    raw += [("chi", h) for h in _variants([f"{a}{suit}", f"{b}{suit}"])]
    raw = [
        (typ, h)
        for typ, h in raw
        if sum(1 for x in [*h, called] if x in rules.PLAIN_OF) <= 1
    ]
    options = []
    for typ, hand in raw:
        pos = 0 if typ == "chi" else POSITION[typ][source]
        laid = list(hand)
        laid.insert(pos, called)
        options.append(MeldOption(typ, laid, pos, hand, _support([called, *hand], ps)))
    best = max((o.cost for o in options), default=0.0)
    for o in options:
        o.cost = min(OPTION_CAP, OPTION_WEIGHT * (best - o.cost))
    return sorted(options, key=lambda o: o.cost)


@dataclass
class Meld:
    """One group of boxes read as a legal meld."""

    type: str  # chi | pon | kan | ankan
    tiles: list[str]
    called_pos: int | None  # the sideways tile among `tiles`
    ps: list[np.ndarray]
    score: float  # Σ log(p / FLOOR) over the tiles: > 0 when the readings support it


def _lp(p: np.ndarray, tile: str) -> float:
    return math.log(max(float(p[CLASS_INDEX[tile]]), 1e-4))


def _best_five(p: np.ndarray, tile: str) -> str:
    """Pick the plain or the red five, whichever the reading prefers."""
    red = rules.RED_OF.get(tile)
    return red if red and _lp(p, red) > _lp(p, tile) else tile


def _sideways(slots: list[dict]) -> int | None:
    side = [
        i for i, s in enumerate(slots) if s["sideways"] >= CONFIRMED_SIDEWAYS_FRACTION
    ]
    if side:
        return side[0]
    # a weaker vote still names the turned tile: the reader's aspect test misses oblique
    # views
    best = max(range(len(slots)), key=lambda i: slots[i]["sideways"])
    return best if slots[best]["sideways"] >= MIN_SIDEWAYS_FRACTION else None


def _decode3(slots: list[dict]) -> Meld | None:
    ps = [np.asarray(s["p"], np.float64) for s in slots]
    floor = 3 * math.log(FLOOR)
    best: Meld | None = None
    for kind in rules.KINDS:  # pon: one kind for all three
        tiles = [_best_five(p, kind) for p in ps]
        sc = sum(_lp(p, t) for p, t in zip(ps, tiles, strict=False)) - floor
        if best is None or sc > best.score:
            best = Meld("pon", tiles, None, ps, sc)
    for suit in "mps":  # chi: three consecutive of one suit, any order
        for n in range(1, 8):
            run = [f"{n + k}{suit}" for k in range(3)]
            for perm in (
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            ):
                tiles = [_best_five(p, run[perm[i]]) for i, p in enumerate(ps)]
                sc = sum(_lp(p, t) for p, t in zip(ps, tiles, strict=False)) - floor
                if best is None or sc > best.score:
                    best = Meld("chi", tiles, None, ps, sc)
    if best is None or best.score < 0:
        return None
    side = _sideways(slots)
    best.called_pos = side if side is not None else (0 if best.type == "chi" else None)
    return best


def decode_group(slots: list[dict]) -> Meld | None:
    """Decode three or four boxes as a legal meld, or None when the readings allow none.

    Four boxes are an ankan when two are face down, a kan when every box reads as one
    kind (the majority never overwrites a box that reads otherwise), else the best meld
    of three of them (a stray box: the same tile boxed twice, or a tile of the next
    meld).
    """
    n = len(slots)
    if n == OPEN_MELD_SIZE:
        return _decode3(slots)
    if n != KAN_SIZE:
        return None
    ps = [np.asarray(s["p"], np.float64) for s in slots]
    down = [i for i, p in enumerate(ps) if CLASSES[int(np.argmax(p))] == "X"]
    faces = [i for i in range(4) if i not in down]
    kind = (
        rules.plain(CLASSES[int(np.argmax(sum(ps[i] for i in faces)))])
        if faces
        else None
    )
    if kind in rules.KINDS:
        red = rules.RED_OF.get(kind)
        support = [
            float(ps[i][CLASS_INDEX[kind]] + (ps[i][CLASS_INDEX[red]] if red else 0.0))
            for i in faces
        ]
        if all(s >= FLOOR for s in support):
            tiles = ["X" if i in down else _best_five(ps[i], kind) for i in range(4)]
            sc = sum(math.log(s / FLOOR) for s in support)
            if len(down) >= ANKAN_BACK_COUNT:
                return Meld("ankan", tiles, None, ps, sc)
            return Meld("kan", tiles, _sideways(slots), ps, sc)
    best = None
    for drop in range(4):
        m = _decode3([s for i, s in enumerate(slots) if i != drop])
        if m is not None and (best is None or m.score > best.score):
            best = m
    return best


def split_group(slots: list[dict]) -> list[Meld]:
    """Split a run of boxes into the consecutive legal melds its readings support best.

    Melds laid close together read as one run of boxes. Each meld has three or four
    boxes, and a single stray box may be skipped. Empty when no split is legal.
    """
    n = len(slots)
    best: list[tuple[float, list[Meld]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])
    for i in range(n):
        current = best[i]
        if current is None:
            continue
        score, melds = current
        skipped = best[i + 1]
        if skipped is None or score > skipped[0]:
            best[i + 1] = (score, melds)  # skip one stray box
        for size in (3, 4):
            if i + size > n:
                continue
            m = decode_group(slots[i : i + size])
            if m is None or (size == KAN_SIZE and m.type not in ("kan", "ankan")):
                continue  # four boxes that are not a kan: a 3 + stray split covers it
            cand = (score + m.score, [*melds, m])
            previous = best[i + size]
            if previous is None or cand[0] > previous[0]:
                best[i + size] = cand
    final = best[n]
    return final[1] if final is not None else []


def source_of(type_: str, called_pos: int | None, n: int) -> str | None:
    """Infer the relative discarder from the sideways tile; None for ankan / unknown."""
    if type_ == "ankan" or called_pos is None:
        return None
    if type_ == "chi":
        return "kamicha"
    if called_pos == 0:
        return "kamicha"
    if called_pos == n - 1:
        return "shimocha"
    return "toimen"


def _groups(obs: dict) -> list[list[dict]]:
    groups: dict[int, list[dict]] = {}
    for s in obs["slots"]:
        groups.setdefault(s["key"][0], []).append(s)
    return [sorted(groups[g], key=lambda s: s["key"][1]) for g in sorted(groups)]


@dataclass(frozen=True)
class MeldView:
    """One meld-camera observation, its box groups and the legal melds each reads as."""

    obs: dict
    groups: list[list[dict]]
    melds: list[list[Meld]]  # per group

    @property
    def partial(self) -> bool:
        """Whether the view may hide melds (a read-floor view)."""
        return bool(self.obs.get("partial"))


def _read(group: list[dict]) -> list[Meld]:
    """Every legal meld one group of boxes shows."""
    if len(group) <= KAN_SIZE:
        m = decode_group(group)
        return [m] if m is not None else []
    return split_group(group)


def read_views(observations: list[dict]) -> list[MeldView]:
    """Read every observation of a meld camera once, in time order."""
    views = []
    for obs in sorted(observations, key=lambda o: o["t0"]):
        groups = _groups(obs) if obs["n_used"] else []
        views.append(MeldView(obs, groups, [_read(g) for g in groups]))
    return views


def _same(a: list[str], b: list[str]) -> bool:
    return [rules.plain(t) for t in a[:3]] == [rules.plain(t) for t in b[:3]]


def _is_known(c: Call, m: Meld) -> bool:
    """Check whether meld `m` is another view of call `c`.

    A pon or kan of a kind is the only one of its kind a seat can have; a chi is matched
    by its tiles in the order laid.
    """
    if c.type in ("pon", "kan", "ankan") and m.type in ("pon", "kan", "ankan"):
        return rules.plain(c.tiles[0]) == rules.plain(
            next((t for t in m.tiles if t != "X"), "?")
        )
    return c.type == m.type == "chi" and _same(c.tiles, m.tiles)


def _tops(g: list[dict]) -> list[str]:
    return [rules.plain(CLASSES[int(np.argmax(s["p"]))]) for s in g]


def _shows_part(c: Call, groups: list[list[dict]]) -> bool:
    """Check whether some group shows at least two tiles of the call.

    A meld one of whose tiles the camera missed is still shown.
    """
    have = Counter(rules.plain(t) for t in c.tiles if t not in ("X", "?"))
    return any(
        sum((Counter(_tops(g)) & have).values()) >= MIN_SHARED_MELD_TILES
        for g in groups
    )


@dataclass
class MeldTracker:
    """Track persistent camera hypotheses for one seat."""

    seat: str
    calls: list[Call] = field(default_factory=list)
    prev_end: float | None = None

    def update(self, view: MeldView) -> None:
        """Consume a view without treating hidden tiles as absent."""
        obs = view.obs
        if obs["n_used"] == 0:
            # no reading at all: says nothing about the melds, narrows no window
            return
        partial = view.partial
        window = (self.prev_end if self.prev_end is not None else obs["t0"], obs["t0"])
        taken: set[int] = set()
        new: list[Meld] = []
        for m in (m for melds in view.melds for m in melds):
            known = next(
                (
                    c
                    for c in self.calls
                    if c.type != "kakan"
                    and _is_known(c, m)
                    and (c.type != "chi" or id(c) not in taken)
                ),
                None,
            )
            if known is None:
                new.append(m)
                continue
            if id(known) in taken:
                continue  # a second group read as the same pon: one meld, read twice
            taken.add(id(known))
            self._see_known(known, m, obs)
        if new:
            # a seat calls at most once between two views of its camera (a call is
            # followed by its discard and the others' turns): of several new melds in
            # one view, the one the readings support best
            m = max(new, key=lambda m: m.score)
            self.calls.append(
                Call(
                    seat=self.seat,
                    t_first=obs["t0"],
                    t_window=window,
                    type=m.type,
                    tiles=m.tiles,
                    called_pos=m.called_pos,
                    source=source_of(m.type, m.called_pos, len(m.tiles)),
                    called_tile=(
                        m.tiles[m.called_pos] if m.called_pos is not None else None
                    ),
                    p=m.ps,
                    conf=float(np.mean([p.max() for p in m.ps])),
                    seen=1,
                    partial_only=partial,
                )
            )
            taken.add(id(self.calls[-1]))
        if not partial:
            # a partial view may hide a meld: it never narrows the call window
            self.prev_end = obs["t1"]
            for c in self.calls:
                # a full view showing tiles but not a meld seen earlier (not even two of
                # its tiles: a tile of a meld is often missed) counts against it; an
                # empty view (the table cleared) says nothing
                if (
                    view.groups
                    and c.type != "kakan"
                    and c.t_first < obs["t0"]
                    and id(c) not in taken
                    and not _shows_part(c, view.groups)
                ):
                    c.absent += 1

    def _see_known(self, known: Call, m: Meld, obs: dict) -> None:
        """Accumulate repeated sightings and distinguish an added kan from a pon."""
        partial = bool(obs.get("partial"))
        window = (self.prev_end if self.prev_end is not None else obs["t0"], obs["t0"])
        known.seen += 1
        if m.type == "pon" and known.type == "pon" and not partial:
            # a full view of the pon with three tiles: evidence against a kakan read
            # on it
            for k in self.calls:
                if (
                    k.type == "kakan"
                    and k.t_first <= obs["t0"]
                    and _same(k.tiles, m.tiles)
                ):
                    k.unseen += 1
        if m.type == "kan" and known.type == "pon":
            if known.partial_only:
                # the "pon" was part of this kan
                known.type, known.tiles, known.p = "kan", m.tiles, m.ps
            else:
                # a pon grew into a kan: the pon stays (its called tile fixes a turn)
                # and the kakan is a second event of this seat, timed by the first view
                # of the fourth tile
                kk = next(
                    (
                        k
                        for k in self.calls
                        if k.type == "kakan" and _same(k.tiles, m.tiles)
                    ),
                    None,
                )
                if kk is None:
                    self.calls.append(
                        Call(
                            seat=self.seat,
                            t_first=obs["t0"],
                            t_window=window,
                            type="kakan",
                            tiles=m.tiles,
                            called_pos=known.called_pos,
                            source=known.source,
                            called_tile=known.called_tile,
                            p=m.ps,
                            conf=float(np.mean([p.max() for p in m.ps])),
                            seen=1,
                            partial_only=partial,
                        )
                    )
                else:
                    kk.seen += 1
                    kk.partial_only = kk.partial_only and partial
        if not partial:
            known.partial_only = False


def track_melds(seat: str, views: Sequence[MeldView]) -> list[Call]:
    """Track one seat's camera hypotheses over its views (from `read_views`).

    A chi, pon or kan that later views contradict is kept with ``contradicted=True``:
    only independent evidence of a compatible taken discard can establish it, and it
    must not establish a self-kan.
    """
    tracker = MeldTracker(seat)
    for view in views:
        tracker.update(view)
    calls = tracker.calls

    def solid(c: Call) -> bool:
        if c.type == "kakan":
            # melds never revert: a kakan seen in fewer views than the plain pon
            # afterwards was a misread; it stands on the evidence of its pon
            return c.unseen <= c.seen and any(
                k.type == "pon"
                and k.seen >= MIN_KAKAN_BASE_VIEWS
                and _same(k.tiles, c.tiles)
                for k in calls
            )
        # a meld never leaves the table: one that later full views lack as often as they
        # show it was never a meld
        return c.absent == 0 or c.seen > c.absent

    out = []
    for c in calls:
        if solid(c):
            out.append(c)
        elif c.type in ("chi", "pon", "kan"):
            # A calm camera view is not proof that every meld was detected.
            # Preserve the earlier positive evidence for the pond to assess.
            c.contradicted = True
            out.append(c)
    return out


@dataclass
class Fragment:
    """Two tiles of a meld the camera keeps showing without a legal third.

    A meld one of whose tiles is not boxed (often the turned one) or is misread. Only
    the discard it took says which meld it is (calls.py).
    """

    seat: str
    tiles: list[str]
    t_first: float
    seen: int
    # the readings of the group's boxes at the first sighting
    ps: list = field(default_factory=list)
    t_before: float | None = None  # the camera's last full view before it


def _pair(g: list[dict]) -> list[str] | None:
    """Select the two tiles of a group that belong to one meld.

    Two identical tiles, else the two surest boxes that are two of a run (same suit,
    numbers one or two apart) — in a group of three or four read as no legal meld, the
    third tile is the misread one (a 4p read 2p beside 3p 0p leaves 3p 0p, of the run
    3p 4p 5p).
    """
    tops = [t for t in _tops(g) if t not in ("X", "none")]
    same = [t for t, n in Counter(tops).items() if n >= MIN_FRAGMENT_REPEATS]
    if same:
        return [same[0]] * 2
    conf = [
        max(float(np.max(s["p"])), 0.0)
        for s in g
        if rules.plain(CLASSES[int(np.argmax(s["p"]))]) not in ("X", "none")
    ]
    runs = [
        (conf[a] + conf[b], sorted([tops[a], tops[b]]))
        for a in range(len(tops))
        for b in range(a + 1, len(tops))
        if tops[a][1] == tops[b][1] != "z"
        and abs(rules.number(tops[a]) - rules.number(tops[b])) in (1, 2)
    ]
    return max(runs)[1] if runs else None


def fragments(
    seat: str, views: Sequence[MeldView], calls: list[Call], min_seen: int = 2
) -> list[Fragment]:
    """Collect pairs of a meld seen in at least `min_seen` full views.

    A pair comes from a group of two, or from a group of three or four that is no legal
    meld as read, and is not part of a meld the seat is known to have.
    """
    found: dict[tuple, Fragment] = {}
    prev_end: float | None = None
    for view in views:
        obs = view.obs
        if view.partial or not obs["n_used"]:
            continue
        before, prev_end = prev_end, obs["t1"]
        for g, melds in zip(view.groups, view.melds, strict=True):
            if len(g) not in (2, 3, 4) or (len(g) > MIN_FRAGMENT_SIZE and melds):
                continue
            pair = _pair(g)
            if pair is None:
                continue
            key = tuple(pair)
            if key in found:
                found[key].seen += 1
            else:
                found[key] = Fragment(
                    seat,
                    pair,
                    obs["t0"],
                    1,
                    [np.asarray(sl["p"], np.float64) for sl in g],
                    before,
                )
    known = [Counter(rules.plain(t) for t in c.tiles) for c in calls if c.seat == seat]
    return [
        f
        for f in found.values()
        if f.seen >= min_seen and not any(not (Counter(f.tiles) - k) for k in known)
    ]
