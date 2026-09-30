"""Meld observations of one seat -> meld events, and the legal melds a call can be (DESIGN.md 4.8 `melds.py`).

A meld camera shows the player's melds as rows of three or four tiles; a called tile lies sideways and its
position (left / middle / right) names the player it came from. A new group in the observations is a meld
event; its type follows from the tiles (chi: consecutive same suit; pon / kan: identical; ankan: two
face-down). A group is a meld only as far as its readings support it: four boxes are a kan only when every
one of them reads as that kind, otherwise the best meld of three of them is taken and the fourth box is
noise; a run of five or more boxes is several melds laid close together and is split into melds of three or
four. The camera is the weakest witness of a call: an event becomes a call only when anchored on the
discard it took (calls.py), and its tiles are then a choice among `meld_options`.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..train.data import CLASS_INDEX, CLASSES
from . import rules

FLOOR = 0.15  # a meld hypothesis needs each tile's reading at this posterior on average
REDS = {"5m": "0m", "5p": "0p", "5s": "0s"}


@dataclass
class Call:
    """An observed or reviewed meld and its candidate legal interpretations.

    Camera order determines called_pos; source is relative to the caller.
    A camera hypothesis becomes a call only once anchored to a discard or kan.
    """

    seat: str
    t_first: float  # start of the first observation showing the meld
    t_window: tuple[float, float]  # when the call happened
    type: str  # chi | pon | kan | ankan | kakan
    tiles: list[str]  # tiles left to right as laid
    called_pos: int | None  # index of the sideways tile (None for ankan)
    source: str | None  # kamicha | toimen | shimocha | None
    called_tile: str | None
    p: list  # per tile posterior
    conf: float
    group: int = 0
    seen: int = 0
    partial_only: bool = False  # seen only in partial (read-floor) observations so far
    unseen: int = (
        0  # kakan: later full views that showed the pon with three tiles again
    )
    absent: int = (
        0  # later full views of the camera without this meld (melds never leave)
    )
    human: bool = False  # stated by the reviewer (a meld fact)
    anchor: str = "camera"  # how the call is established: camera (an event, not yet a call) | discard | kan |
    # indicator | hidden | fact
    options: list = field(
        default_factory=list
    )  # MeldOption: the legal compositions the solver chooses among
    contradicted: bool = False  # later incomplete views outnumbered sightings; requires independent discard evidence

    def to_dict(self) -> dict:
        """Serialize the meld with rounded posteriors and all alternative compositions."""
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
        """Serialize one candidate, including the concealed tiles it removes from the hand."""
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
        red = REDS.get(t)
        out = [o + [v] for o in out for v in ([t, red] if red else [t])]
    return [o for o in out if sum(1 for x in o if x in rules.REDS) <= 1]


def _support(tiles: list[str], ps: list[np.ndarray]) -> float:
    """How well boxes read as these tiles: each tile takes the unused box that reads it best (log posterior);
    a tile with no box left (the camera showed fewer boxes) is neutral.
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
    called: str, source: str, ps: list[np.ndarray], four: bool
) -> list[MeldOption]:
    """The legal melds a seat can have laid with `called` from `source` (DESIGN.md 4.8, call anchor step 3): a
    daiminkan when the camera shows four boxes, else a pon, and a chi when the tile came from the kamicha.
    Each is costed by how well the camera's readings `ps` support its tiles; the solver decides with the hand.
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
                if 1 <= a <= 9 and 1 <= b <= 9:
                    raw += [("chi", h) for h in _variants([f"{a}{suit}", f"{b}{suit}"])]
    raw = [
        (typ, h)
        for typ, h in raw
        if sum(1 for x in h + [called] if x in rules.REDS) <= 1
    ]
    options = []
    for typ, hand in raw:
        pos = 0 if typ == "chi" else POSITION[typ][source]
        laid = list(hand)
        laid.insert(pos, called)
        options.append(MeldOption(typ, laid, pos, hand, _support([called] + hand, ps)))
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
    """The plain or the red five, whichever the reading prefers (any other tile unchanged)."""
    red = REDS.get(tile)
    return red if red and _lp(p, red) > _lp(p, tile) else tile


def _sideways(slots: list[dict]) -> int | None:
    side = [i for i, s in enumerate(slots) if s["sideways"] >= 0.5]
    if side:
        return side[0]
    # a weaker vote still names the turned tile: the reader's aspect test misses oblique views
    best = max(range(len(slots)), key=lambda i: slots[i]["sideways"])
    return best if slots[best]["sideways"] >= 0.2 else None


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
                if sc > best.score:
                    best = Meld("chi", tiles, None, ps, sc)
    if best.score < 0:
        return None
    side = _sideways(slots)
    best.called_pos = side if side is not None else (0 if best.type == "chi" else None)
    return best


def decode_group(slots: list[dict]) -> Meld | None:
    """The meld one group of three or four boxes is, or None when its readings support no legal meld.

    Four boxes are an ankan when two are face down, a kan when every box reads as one kind (the majority
    never overwrites a box that reads otherwise), else the best meld of three of them (a stray box: the
    same tile boxed twice, or a tile of the next meld).
    """
    n = len(slots)
    if n == 3:
        return _decode3(slots)
    if n != 4:
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
        support = [
            float(
                ps[i][CLASS_INDEX[kind]]
                + (ps[i][CLASS_INDEX[REDS[kind]]] if kind in REDS else 0.0)
            )
            for i in faces
        ]
        if all(s >= FLOOR for s in support):
            tiles = ["X" if i in down else _best_five(ps[i], kind) for i in range(4)]
            sc = sum(math.log(s / FLOOR) for s in support)
            if len(down) >= 2:
                return Meld("ankan", tiles, None, ps, sc)
            return Meld("kan", tiles, _sideways(slots), ps, sc)
    best = None
    for drop in range(4):
        m = _decode3([s for i, s in enumerate(slots) if i != drop])
        if m is not None and (best is None or m.score > best.score):
            best = m
    return best


def split_group(slots: list[dict]) -> list[Meld]:
    """Melds laid close together read as one run of boxes: the split into consecutive melds of three or four
    (a single stray box may be skipped) whose readings support them best. Empty when none is legal.
    """
    n = len(slots)
    best: list[tuple[float, list[Meld]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])
    for i in range(n):
        if best[i] is None:
            continue
        score, melds = best[i]
        if best[i + 1] is None or score > best[i + 1][0]:
            best[i + 1] = (score, melds)  # skip one stray box
        for size in (3, 4):
            if i + size > n:
                continue
            m = decode_group(slots[i : i + size])
            if m is None or (size == 4 and m.type not in ("kan", "ankan")):
                continue  # four boxes that are not a kan: a 3 + stray split covers it
            cand = (score + m.score, melds + [m])
            if best[i + size] is None or cand[0] > best[i + size][0]:
                best[i + size] = cand
    return best[n][1] if best[n] else []


def source_of(type_: str, called_pos: int | None, n: int) -> str | None:
    """Infer the relative discarder from meld orientation; ambiguous/concealed forms return None."""
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


def read_melds(obs: dict) -> list[Meld]:
    """Every meld one observation of a meld camera shows."""
    out: list[Meld] = []
    for g in _groups(obs):
        if len(g) <= 4:
            m = decode_group(g)
            if m is not None:
                out.append(m)
        else:
            out += split_group(g)
    return out


def _same(a: list[str], b: list[str]) -> bool:
    return [rules.plain(t) for t in a[:3]] == [rules.plain(t) for t in b[:3]]


def _is_known(c: Call, m: Meld) -> bool:
    """Is meld `m` a view of call `c`? A pon or kan of a kind is the only one of its kind a seat can have;
    a chi is matched by its tiles in the order laid.
    """
    if c.type in ("pon", "kan", "ankan") and m.type in ("pon", "kan", "ankan"):
        return rules.plain(c.tiles[0]) == rules.plain(
            next((t for t in m.tiles if t != "X"), "?")
        )
    return c.type == m.type == "chi" and _same(c.tiles, m.tiles)


def _tops(g: list[dict]) -> list[str]:
    return [rules.plain(CLASSES[int(np.argmax(s["p"]))]) for s in g]


def _shows_part(c: Call, groups: list[list[dict]]) -> bool:
    """Does some group show at least two tiles of the call (a meld one of whose tiles the camera missed)?"""
    have = Counter(rules.plain(t) for t in c.tiles if t not in ("X", "?"))
    return any(sum((Counter(_tops(g)) & have).values()) >= 2 for g in groups)


def track_melds(
    seat: str, observations: list[dict], *, include_contradicted: bool = False
) -> list[Call]:
    """Track one seat's camera hypotheses, retaining established sightings by default.

    Call anchoring may request external-call hypotheses contradicted by later
    views. Those carry ``contradicted=True`` and require independent evidence
    of a compatible taken discard; they must not establish a self-kan.
    """
    calls: list[Call] = []
    prev_end: float | None = None
    for obs in sorted(observations, key=lambda o: o["t0"]):
        if obs["n_used"] == 0:
            continue  # no reading at all: says nothing about the melds, narrows no window
        partial = bool(obs.get("partial"))
        melds = read_melds(obs)
        window = (prev_end if prev_end is not None else obs["t0"], obs["t0"])
        taken: set[int] = set()
        new: list[tuple[int, Meld]] = []
        for g, m in enumerate(melds):
            known = next(
                (
                    c
                    for c in calls
                    if c.type != "kakan"
                    and _is_known(c, m)
                    and (c.type != "chi" or id(c) not in taken)
                ),
                None,
            )
            if known is None:
                new.append((g, m))
                continue
            if id(known) in taken:
                continue  # a second group read as the same pon: one meld, read twice
            taken.add(id(known))
            known.seen += 1
            if m.type == "pon" and known.type == "pon" and not partial:
                # a full view of the pon with three tiles: evidence against a kakan read on it
                for k in calls:
                    if (
                        k.type == "kakan"
                        and k.t_first <= obs["t0"]
                        and _same(k.tiles, m.tiles)
                    ):
                        k.unseen += 1
            if m.type == "kan" and known.type == "pon":
                if known.partial_only:
                    known.type, known.tiles, known.p = (
                        "kan",
                        m.tiles,
                        m.ps,
                    )  # the "pon" was part of this kan
                else:
                    # a pon grew into a kan: the pon stays (its called tile fixes a turn) and the kakan is a
                    # second event of this seat, timed by the first view of the fourth tile
                    kk = next(
                        (
                            k
                            for k in calls
                            if k.type == "kakan" and _same(k.tiles, m.tiles)
                        ),
                        None,
                    )
                    if kk is None:
                        calls.append(
                            Call(
                                seat,
                                obs["t0"],
                                window,
                                "kakan",
                                m.tiles,
                                known.called_pos,
                                known.source,
                                known.called_tile,
                                m.ps,
                                float(np.mean([p.max() for p in m.ps])),
                                g,
                                1,
                                partial_only=partial,
                            )
                        )
                    else:
                        kk.seen += 1
                        kk.partial_only = kk.partial_only and partial
            if not partial:
                known.partial_only = False
        if new:
            # a seat calls at most once between two views of its camera (a call is followed by its discard and
            # the others' turns): of several new melds in one view, the one the readings support best
            g, m = max(new, key=lambda x: x[1].score)
            calls.append(
                Call(
                    seat,
                    obs["t0"],
                    window,
                    m.type,
                    m.tiles,
                    m.called_pos,
                    source_of(m.type, m.called_pos, len(m.tiles)),
                    m.tiles[m.called_pos] if m.called_pos is not None else None,
                    m.ps,
                    float(np.mean([p.max() for p in m.ps])),
                    g,
                    1,
                    partial_only=partial,
                )
            )
            taken.add(id(calls[-1]))
        if not partial:
            prev_end = obs[
                "t1"
            ]  # a partial view may hide a meld: it never narrows the call window
            groups = _groups(obs)
            for c in calls:
                # a full view showing tiles but not a meld seen earlier (not even two of its tiles: a tile of a
                # meld is often missed) counts against it; an empty view (the table cleared) says nothing
                if (
                    groups
                    and c.type != "kakan"
                    and c.t_first < obs["t0"]
                    and id(c) not in taken
                    and not _shows_part(c, groups)
                ):
                    c.absent += 1

    def solid(c: Call) -> bool:
        if c.type == "kakan":
            # melds never revert: a kakan seen in fewer views than the plain pon afterwards was a misread; it
            # stands on the evidence of its pon
            return c.unseen <= c.seen and any(
                k.type == "pon" and k.seen >= 2 and _same(k.tiles, c.tiles)
                for k in calls
            )
        # a meld never leaves the table: one that later full views lack as often as they show it was never a
        # meld; one seen in a single partial frame is returned as `weak`, for the pond to confirm or not
        return c.absent == 0 or c.seen > c.absent

    out = []
    for c in calls:
        if solid(c):
            out.append(c)
        elif include_contradicted and c.type in ("chi", "pon", "kan"):
            # A calm camera view is not proof that every meld was detected.
            # Preserve the earlier positive evidence for the pond to assess.
            c.contradicted = True
            out.append(c)
    return out


@dataclass
class Fragment:
    """Two tiles of a meld the camera keeps showing without a legal third: a meld one of whose tiles is not boxed
    (often the turned one) or is misread. Only the discard it took says which meld it is (calls.py).
    """

    seat: str
    tiles: list[str]
    t_first: float
    seen: int
    ps: list = field(
        default_factory=list
    )  # the readings of the group's boxes at the first sighting
    t_before: float | None = None  # the camera's last full view before it


def _pair(g: list[dict]) -> list[str] | None:
    """The two tiles of a group that belong to one meld: two identical tiles, else the two surest boxes that are two
    of a run (same suit, numbers one or two apart) — in a group of three or four read as no legal meld, the
    third tile is the misread one (a 4p read 2p beside 3p 0p leaves 3p 0p, of the run 3p 4p 5p).
    """
    tops = [t for t in _tops(g) if t not in ("X", "none")]
    same = [t for t, n in Counter(tops).items() if n >= 2]
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
    seat: str, observations: list[dict], calls: list[Call], min_seen: int = 2
) -> list[Fragment]:
    """Pairs of a meld seen in at least `min_seen` full views — a group of two, or a group of three or four that
    is no legal meld as read — that are not part of a meld the seat is known to have.
    """
    found: dict[tuple, Fragment] = {}
    prev_end: float | None = None
    for obs in sorted(observations, key=lambda o: o["t0"]):
        if obs.get("partial") or not obs["n_used"]:
            continue
        before, prev_end = prev_end, obs["t1"]
        for g in _groups(obs):
            if len(g) not in (2, 3, 4) or (len(g) > 2 and decode_group(g) is not None):
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
