# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Merge the four discard logs into a hand's turn sequence (DESIGN.md 4.8 `turns.py`).

Each pond's order is fixed; the interleaving is found as the cheapest path through the
turn automaton (E S W N in order, the dealer first; a call on a discard passes the turn
to the caller, who discards next without drawing). The cost is local: a discard placed
after another costs the seconds by which it was sighted before the previous one could
have happened, capped, so one wrong time costs one inversion and never drags the rest of
the hand along. Three escape moves have fixed costs above any honest inversion: a seat
passing its turn with no visible discard (skip), a slot left out as not a discard of
this hand (drop), and the sequence ending at the wrong seat for the site's result. Every
removal is a call: a removed slot no known call took may pass the turn to a seat other
than its natural successor, as a hidden call by that seat (the camera missed the meld;
the solver chooses its tiles). A discard that was called away before any calm frame
showed it never reaches the pond log; such a discard is inserted as a virtual slot from
the call itself.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field, replace

from . import rules
from .melds import POSITION, Call
from .ponds import PondSlot, virtual_slot

SEATS = rules.SEATS
SOURCES = ("kamicha", "toimen", "shimocha")
SKIP_COST = 60.0  # a seat passing its turn without a visible discard (a missed discard)
# a slot left out of the sequence (a phantom: not a discard of this hand)
DROP_COST = 45.0
INV_CAP = 40.0  # an inversion costs its seconds, at most this
INV_TOL = 20.0  # an inversion over this is reported
# per second from one discard to the next: of two equal sequences, the one whose turns
# follow closely
GAP_COST = 0.01
CALL_TOL = 25.0  # seconds between a pond removal and the meld appearing
# s a meld may be on camera before its called tile is last seen in the pond (hand tiles
# first)
TAKE_SLACK = 8.0
MAX_SKIPS = 3  # passes in a row (a fourth would return to the same seat)
# a removed slot no known call took, called by a seat out of the natural order ...
HIDDEN_CALL = 8.0
# ... or passed to the natural successor as if nobody took it (a misread removal)
UNTAKEN = 12.0
# a self-kan's turn is the seat's first discard sighted from this many s before the
# kan's window opened ...
KAN_TURN_SLACK = 1.0
# ... to this many s after the camera first showed the kan
KAN_TO_DISCARD = 20.0

type SearchState = tuple[int, ...]
GOAL: SearchState = (-1,)


@dataclass
class Turn:
    """A merged table turn, with distinct calls made by this seat and on its discard."""

    i: int
    seat: str
    kind: str  # draw | call | kan
    slot: PondSlot | None
    t: float
    call: Call | None = None  # the call made ON this discard (by another seat)
    riichi: bool = False
    virtual: bool = False
    # the call this seat made just before this discard (kind call / kan)
    own_call: Call | None = None

    @property
    def discard_slot(self) -> PondSlot:
        """Return a required discard, rejecting a terminal draw with no discard."""
        if self.slot is None:
            raise ValueError("This turn has no discard slot")
        return self.slot

    def to_dict(self) -> dict:
        """Serialize turn ordering together with its pond and call evidence."""
        return {
            "i": self.i,
            "seat": self.seat,
            "kind": self.kind,
            "t": self.t,
            "riichi": self.riichi,
            "virtual": self.virtual,
            "discard": self.slot.to_dict() if self.slot else None,
            "call": self.call.to_dict() if self.call else None,
            "own_call": self.own_call.to_dict() if self.own_call else None,
        }


@dataclass
class Skip:
    """A seat whose turn came while no discard of it was found in its pond."""

    i: int  # the skipped turn comes before turns[i] of the merged sequence
    seat: str
    after: float | None = None  # time of the turn before it (None: the first turn)
    before: float | None = None  # time of the turn after it (None: the last turn)

    @property
    def text(self) -> str:
        """Describe the skip."""
        return f"turn {self.i}: no discard of {self.seat} found in its pond (skipped)"


@dataclass(frozen=True)
class Dropped:
    """A pond tile left out of the sequence: not a discard of this hand."""

    seat: str
    tile: str
    t: float  # its first sighting
    after_end: bool  # laid after the hand ended

    @property
    def text(self) -> str:
        """Describe the dropped tile."""
        why = "dropped: laid after the hand ended" if self.after_end else "dropped"
        return (
            f"{self.seat}'s {self.tile} seen at {self.t:.0f}s is not a discard of this "
            f"hand ({why})"
        )


@dataclass(frozen=True)
class WrongEnd:
    """The sequence ends with another seat to play than the result says."""

    seat: str
    end: str

    @property
    def text(self) -> str:
        """Describe the ending mismatch."""
        return (
            f"the sequence ends with {self.seat} to play, but the result says "
            f"{self.end} (the last discards are uncertain)"
        )


@dataclass(frozen=True)
class HiddenCall:
    """A removal no camera call explains, taken by the seat that plays next."""

    discarder: str
    tile: str
    t: float  # when it left the pond
    caller: str

    @property
    def text(self) -> str:
        """Describe the inferred call."""
        return (
            f"{self.discarder}'s {self.tile} left the pond at {self.t:.0f}s and "
            f"{self.caller} plays next: a call by {self.caller} its meld camera did "
            "not show"
        )


@dataclass(frozen=True)
class Inversion:
    """A discard placed more than INV_TOL s before the previous one could happen."""

    i: int
    seat: str
    t: float
    seconds: float

    @property
    def text(self) -> str:
        """Describe the inversion."""
        return (
            f"turn {self.i}: {self.seat}'s discard at {self.t:.0f}s is out of time "
            f"order ({self.seconds:.0f} s before the previous discard could have "
            "happened)"
        )


@dataclass(frozen=True)
class UnmatchedCall:
    """A chi, pon or daiminkan that took no discard of the sequence."""

    seat: str
    type: str
    tiles: tuple[str, ...]
    t: float

    @property
    def text(self) -> str:
        """Describe the unmatched call."""
        return (
            f"call {self.type} {''.join(self.tiles)} by {self.seat} at {self.t:.0f}s "
            "matches no discard"
        )


type Finding = Skip | Dropped | WrongEnd | HiddenCall | Inversion | UnmatchedCall


@dataclass(frozen=True)
class MergeResult:
    """A merged turn sequence and what the merge had to assume to build it."""

    turns: list[Turn]
    calls: list[Call]  # the input calls, an unknown source bound to the best one
    findings: list[Finding]  # in sequence order, unmatched calls last
    cost: float

    @property
    def skips(self) -> list[Skip]:
        """Turns that passed with no discard found."""
        return [f for f in self.findings if isinstance(f, Skip)]


def call_distance(c: Call, slot: PondSlot, discarder: str) -> float | None:
    """Seconds between a pond removal and a call, None when the call cannot take it."""
    if slot.t_removed is None or c.seat == discarder or c.type in ("ankan", "kakan"):
        return None
    if c.source and rules.relative(c.seat, discarder) != c.source:
        return None
    if c.called_tile and rules.plain(c.called_tile) != rules.plain(slot.tile):
        return None
    return removal_gap(c, slot)


def removal_gap(c: Call, slot: PondSlot) -> float | None:
    """Measure the time between a pond removal and a call's event window.

    The window runs from the meld camera's last calm view without the meld to its first
    with it. None when the call cannot have taken the slot: the tile was still in its
    pond after the meld holding it was on the table, or it left too long before or
    after.
    """
    lo, hi = c.t_window[0], c.t_first
    if slot.t_removed is None or slot.t_last > hi + TAKE_SLACK:
        return None
    d = (
        0.0
        if lo <= slot.t_removed <= hi
        else min(abs(lo - slot.t_removed), abs(hi - slot.t_removed))
    )
    return d if d <= CALL_TOL else None


def assign_calls(
    calls: list[Call], seq: dict[str, list[PondSlot]]
) -> dict[str, list[Call | None]]:
    """Match each call to the single pond removal it consumed.

    A call takes one tile, so every call is assigned to its single nearest compatible
    removal and every removal to at most one call (nearest pairs first).
    """
    pairs = []
    for s, sls in seq.items():
        for k, sl in enumerate(sls):
            for c in calls:
                d = call_distance(c, sl, s)
                if d is not None:
                    pairs.append((d, s, k, c))
    pairs.sort(key=lambda x: x[0])
    out: dict[str, list[Call | None]] = {s: [None] * len(sls) for s, sls in seq.items()}
    taken: set[int] = set()
    for _, s, k, c in pairs:
        if id(c) in taken or out[s][k] is not None:
            continue
        out[s][k] = c
        taken.add(id(c))
    return out


def virtual_discards(
    logs: dict[str, list[PondSlot]], calls: list[Call]
) -> dict[str, list[PondSlot]]:
    """Copy the logs with a slot for every call whose called tile no removal matches.

    A chi / pon / daiminkan whose discard no pond view showed took a virtual slot,
    inserted by time among the discarder's slots.
    """
    out = {s: list(v) for s, v in logs.items()}
    for c in calls:
        if c.type in ("ankan", "kakan") or not c.called_tile or not c.source:
            continue
        src = rules.seat_at(c.seat, c.source)
        if any(call_distance(c, sl, src) is not None for sl in out[src]):
            continue
        t = c.t_window[0]
        vs = virtual_slot(out, c.called_tile, t, seen=1)
        vs.t_removed = c.t_first
        out[src].insert(sum(1 for sl in out[src] if sl.t_first <= t), vs)
    return out


def _with_source(c: Call, src: str) -> Call:
    """Copy a call, bound to one relative source and its called tile position."""
    pos = POSITION[c.type][src]
    return replace(
        c,
        source=src,
        called_pos=pos,
        called_tile=c.tiles[pos] if pos < len(c.tiles) else c.tiles[0],
    )


@dataclass(frozen=True, kw_only=True)
class HandEnding:
    """End seat and live-wall constraints used by chronological reconstruction."""

    end: str | None = None
    wall: int | None = None
    exhaustive: bool = False


def merge(
    logs: dict[str, list[PondSlot]],
    calls: list[Call],
    dealer: str,
    *,
    ending: HandEnding | None = None,
) -> MergeResult:
    """Reconstruct a legal chronological turn sequence from pond logs.

    logs: seat -> discard log (laying order). `ending.end`: the seat whose turn it is
    when the hand ends (a tsumo winner draws the winning tile; after a ron the seat
    after the loser), None for an exhaustive draw. `ending.wall`: the most draws the
    sequence may hold (the live wall less the kans, less a tsumo's winning draw);
    `ending.exhaustive`: the hand ends with the discard of the last of them.

    A pon / daiminkan whose turned tile was not seen has no source; every source is
    tried and the one with the cheapest sequence wins (a source whose pond shows a
    matching removal first). The input calls are not changed: the result carries the
    calls with the chosen sources.
    """
    ending = ending or HandEnding()
    unknown = [
        k for k, c in enumerate(calls) if c.type in ("pon", "kan") and not c.source
    ]
    choices = []
    for k in unknown:
        every, shown = [], []
        for src in SOURCES:
            bound = _with_source(calls[k], src)
            seat = rules.seat_at(bound.seat, src)
            every.append(bound)
            if any(
                call_distance(bound, sl, seat) is not None for sl in logs.get(seat, [])
            ):
                shown.append(bound)
        choices.append(shown or every)
    best: tuple[float, MergeResult] | None = None
    for combo in itertools.product(*choices):
        trial = list(calls)
        for k, c in zip(unknown, combo, strict=True):
            trial[k] = c
        result = _merge(logs, trial, dealer, ending)
        # a discard nobody saw is the weaker explanation
        cost = result.cost + SKIP_COST * sum(1 for t in result.turns if t.virtual) / 4
        if best is None or cost < best[0]:
            best = (cost, result)
    if best is None:
        raise ValueError("No source assignment exists for the observed calls")
    return best[1]


def _inversion(prev: PondSlot | None, sl: PondSlot) -> float:
    """Seconds by which `sl` was sighted before `prev` could have happened."""
    if prev is None:
        return 0.0
    lo = prev.t_window[0]
    return max(0.0, lo - sl.t_first)


def _step_cost(prev: PondSlot | None, sl: PondSlot) -> float:
    """Score placing `sl` after `prev`: the capped inversion, and a closeness tie-break.

    Of two otherwise equal sequences, the one whose turns follow each other closely is
    cheaper.
    """
    gap = max(0.0, sl.t_first - prev.t_first) if prev is not None else 0.0
    return min(INV_CAP, _inversion(prev, sl)) + GAP_COST * gap


@dataclass
class TurnSearch:
    """Fixed evidence and termination constraints for a turn-order search."""

    seq: list[list[PondSlot]]
    callers: dict[str, list[Call | None]]
    end: str | None
    most: int
    exhaustive: bool
    n: tuple[int, ...] = field(init=False)  # observed positions in each seat's pond

    def __post_init__(self) -> None:
        """Count each seat's positions once."""
        self.n = tuple(len(v) for v in self.seq)

    def moves(self, st: SearchState) -> list[tuple[str, float, SearchState]]:
        """Enumerate legal placements, hidden calls, skips and hand endings."""
        *idx, cur, last, last_i, skips, draws, called = st
        n = self.n
        moves: list[tuple[str, float, SearchState]] = []
        left = sum(n[k] - idx[k] for k in range(4))
        over = self.exhaustive and draws == self.most and last >= 0 and not called
        if left == 0:
            steps = 0 if self.end is None else (SEATS.index(self.end) - cur) % 4
            moves.append(("goal", SKIP_COST * steps, GOAL))
        elif (self.end is not None and SEATS[cur] == self.end and last >= 0) or over:
            # the hand is over when the end seat is to play, or the last draw's discard
            # is laid: whatever is left was laid after it (the reveal)
            moves.append(("goal", DROP_COST * left, GOAL))
        drew = draws + (0 if called else 1)  # a call turn has no draw
        if left and idx[cur] < n[cur] and not over:
            sl = self.seq[cur][idx[cur]]
            prev = self.seq[last][last_i] if last >= 0 else None
            c = self.callers[SEATS[cur]][idx[cur]]
            nxt = SEATS.index(c.seat) if c is not None else (cur + 1) % 4
            new = list(idx)
            new[cur] += 1
            step = _step_cost(prev, sl)
            if drew <= self.most:
                untaken = c is None and sl.t_removed is not None and not sl.virtual
                moves.append(
                    (
                        "place",
                        step + (UNTAKEN if untaken else 0.0),
                        (*new, nxt, cur, idx[cur], 0, drew, c is not None),
                    )
                )
                if untaken:
                    # a tile left the pond and no known call took it: a seat whose
                    # camera missed the meld called it
                    moves.extend(
                        (
                            f"hidden:{x}",
                            step + HIDDEN_CALL,
                            (*new, x, cur, idx[cur], 0, drew, True),
                        )
                        for x in ((cur + 2) % 4, (cur + 3) % 4)
                    )
            moves.append(
                ("drop", DROP_COST, (*new, cur, last, last_i, skips, draws, called))
            )
        if skips < MAX_SKIPS and left > 0 and drew <= self.most and not over:
            moves.append(
                (
                    "skip",
                    SKIP_COST,
                    (*idx, (cur + 1) % 4, last, last_i, skips + 1, drew, False),
                )
            )
        return moves

    def search(self, start: SearchState) -> tuple[list[tuple[SearchState, str]], float]:
        """Find the cheapest chronological path with stable tie ordering."""
        # Costs are non-negative: the first popped goal has the cheapest path.
        dist = {start: 0.0}
        parent: dict[SearchState, tuple[SearchState, str]] = {}
        order = itertools.count()  # ties are popped first come, first served
        heap: list[tuple[float, int, SearchState]] = [(0.0, next(order), start)]
        goal = None
        while heap:
            d, _, st = heapq.heappop(heap)
            if d > dist.get(st, float("inf")):
                continue
            if st == GOAL:
                goal = st
                break
            for kind, cost, nst in self.moves(st):
                nd = d + cost
                if nd < dist.get(nst, float("inf")):
                    dist[nst] = nd
                    parent[nst] = (st, kind)
                    heapq.heappush(heap, (nd, next(order), nst))
        if goal is None:
            raise ValueError(
                "No turn sequence satisfies the hand's wall and call constraints"
            )
        # the path, from the goal back to the start
        path = []
        st = goal
        while st != start:
            prev_st, kind = parent[st]
            path.append((prev_st, kind))
            st = prev_st
        path.reverse()
        return path, dist[goal]


def _path_turns(
    path: list[tuple[SearchState, str]],
    seq: list[list[PondSlot]],
    callers: dict[str, list[Call | None]],
    end: str | None,
) -> tuple[list[Turn], list[Finding]]:
    """Materialize chronological turns and findings from the accepted path."""
    turns: list[Turn] = []
    findings: list[Finding] = []
    pending: dict[str, Call] = {}
    open_skips: list[Skip] = []
    prev_slot: PondSlot | None = None
    for st, kind in path:
        *idx, cur, _, _, _, _, _ = st
        s = SEATS[cur]
        if kind == "goal":
            if end is not None and s != end:
                findings.append(WrongEnd(s, end))
            findings.extend(
                Dropped(SEATS[k], sl.tile, sl.t_first, after_end=True)
                for k in range(4)
                for sl in seq[k][idx[k] :]
            )
            continue
        if kind == "skip":
            skip = Skip(len(turns), s, after=turns[-1].t if turns else None)
            findings.append(skip)
            open_skips.append(skip)
            pending.pop(s, None)
            continue
        sl = seq[cur][idx[cur]]
        if kind == "drop":
            findings.append(Dropped(s, sl.tile, sl.t_first, after_end=False))
            continue
        c = callers[s][idx[cur]]
        if kind.startswith("hidden:"):
            c = _hidden_call(SEATS[int(kind.partition(":")[2])], s, sl)
            findings.append(HiddenCall(s, sl.tile, c.t_first, c.seat))
        own = pending.pop(s, None)
        turn = Turn(
            len(turns),
            s,
            "draw" if own is None else ("kan" if own.type == "kan" else "call"),
            sl,
            sl.t_first,
            c,
            riichi=sl.is_sideways,
            virtual=sl.virtual,
            own_call=own,
        )
        inv = _inversion(prev_slot, sl)
        if inv > INV_TOL:
            findings.append(Inversion(turn.i, s, sl.t_first, inv))
        for skip in open_skips:
            skip.before = turn.t
        open_skips.clear()
        prev_slot = sl
        turns.append(turn)
        if c is not None:
            pending[c.seat] = c
    return turns, findings


def _place_self_kans(turns: list[Turn], calls: list[Call]) -> list[UnmatchedCall]:
    """Place self-kans in their windows; return the external calls on no discard."""
    used = {id(t.call) for t in turns if t.call is not None}
    unmatched = []
    for c in calls:
        if c.type in ("ankan", "kakan"):
            # a self-kan happened inside its window (last calm view without it, first
            # with it): its turn is the seat's first discard after the window opened. A
            # turn already holding a call keeps it (a chi or pon turn has no draw to kan
            # with); a kan with no discard after it is left to the decoder (a tsumo on
            # the rinshan draw)
            lo, hi = c.t_window[0], max(c.t_window[1], c.t_first)
            cand = [
                t
                for t in turns
                if t.seat == c.seat
                and lo - KAN_TURN_SLACK <= t.t <= hi + KAN_TO_DISCARD
                and t.own_call is None
            ]
            if cand:
                cand[0].kind = "kan"
                cand[0].own_call = c
        elif id(c) not in used:
            unmatched.append(UnmatchedCall(c.seat, c.type, tuple(c.tiles), c.t_first))
    return unmatched


def _merge(
    logs: dict[str, list[PondSlot]],
    calls: list[Call],
    dealer: str,
    ending: HandEnding,
) -> MergeResult:
    logs = virtual_discards(logs, calls)
    seq = [list(logs.get(s, [])) for s in SEATS]
    callers = assign_calls(calls, dict(zip(SEATS, seq, strict=False)))
    most = ending.wall if ending.wall is not None else 10**6
    # indices, whose turn, last placed (seat, index), skips, draws so far, whether the
    # seat to play called
    start: SearchState = (0, 0, 0, 0, SEATS.index(dealer), -1, -1, 0, 0, False)
    path, cost = TurnSearch(seq, callers, ending.end, most, ending.exhaustive).search(
        start
    )
    turns, findings = _path_turns(path, seq, callers, ending.end)
    findings += _place_self_kans(turns, calls)
    return MergeResult(turns, calls, findings, cost)


def _hidden_call(caller: str, discarder: str, sl: PondSlot) -> Call:
    """Infer the call the turn order shows and no camera did: a pon of the tile.

    The solver chooses its tiles.
    """
    if sl.t_removed is None:
        raise ValueError("A hidden call requires a removed pond tile")
    source = rules.relative(caller, discarder)
    pos = POSITION["pon"][source]
    tiles = [rules.plain(sl.tile)] * 3
    tiles[pos] = sl.tile
    return Call(
        seat=caller,
        t_first=sl.t_removed,
        t_window=(sl.t_first, sl.t_removed),
        type="pon",
        tiles=tiles,
        called_pos=pos,
        source=source,
        called_tile=sl.tile,
        anchor="hidden",
    )
