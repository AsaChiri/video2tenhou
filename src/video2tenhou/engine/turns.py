"""Merge of the four discard logs into the turn sequence of a hand (DESIGN.md 4.8 `turns.py`).

Each pond's order is fixed; the interleaving is found as the cheapest path through the turn automaton
(E S W N in order, the dealer first; a call on a discard passes the turn to the caller, who discards next
without drawing). The cost is local: a discard placed after another costs the seconds by which it was
sighted before the previous one could have happened, capped, so one wrong time costs one inversion and
never drags the rest of the hand along. Three escape moves have fixed costs above any honest inversion:
a seat passing its turn with no visible discard (skip), a slot left out as not a discard of this hand
(drop), and the sequence ending at the wrong seat for the site's result. Every removal is a call: a removed
slot no known call took may pass the turn to a seat other than its natural successor, as a hidden call by
that seat (the camera missed the meld; the solver chooses its tiles). A discard that was called away before
any calm frame showed it never reaches the pond log; such a discard is inserted as a virtual slot from the
call itself.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..train.data import CLASS_INDEX, CLASSES
from . import rules
from .melds import POSITION, Call
from .ponds import PondSlot

SEATS = rules.SEATS
SKIP_COST = 60.0         # a seat passing its turn without a visible discard (a missed discard)
DROP_COST = 45.0         # a slot left out of the sequence (a phantom: not a discard of this hand)
INV_CAP = 40.0           # an inversion costs its seconds, at most this
INV_TOL = 20.0           # an inversion over this is reported
GAP_COST = 0.01          # per second from one discard to the next: of two equal sequences, the one whose turns follow closely
CALL_TOL = 25.0          # seconds between a pond removal and the meld appearing
TAKE_SLACK = 8.0         # s a meld may be on camera before its called tile is last seen in the pond (hand tiles first)
MAX_SKIPS = 3            # passes in a row (a fourth would return to the same seat)
HIDDEN_CALL = 8.0        # a removed slot no known call took, called by a seat out of the natural order
UNTAKEN = 12.0           # ... or passed to the natural successor as if nobody took it (a misread removal)


@dataclass
class Turn:
    """A merged table turn, with distinct calls made by this seat and on its discard."""
    i: int
    seat: str
    kind: str                       # draw | call | kan
    slot: Optional[PondSlot]
    t: float
    call: Optional[Call] = None     # the call made ON this discard (by another seat)
    riichi: bool = False
    virtual: bool = False
    own_call: Optional[Call] = None # the call this seat made just before this discard (kind call / kan)

    def to_dict(self) -> dict:
        """Serialize turn ordering together with its pond and call evidence."""
        return {"i": self.i, "seat": self.seat, "kind": self.kind, "t": self.t, "riichi": self.riichi, "virtual": self.virtual,
                "discard": self.slot.to_dict() if self.slot else None, "call": self.call.to_dict() if self.call else None,
                "own_call": self.own_call.to_dict() if self.own_call else None}


def _call_dist(c: Call, slot: PondSlot, discarder: str) -> Optional[float]:
    """Seconds between a pond removal and the call's window, None when the call cannot be on this discard."""
    if slot.t_removed is None or c.seat == discarder or c.type in ("ankan", "kakan"):
        return None
    if c.source and rules.relative(c.seat, discarder) != c.source:
        return None
    if c.called_tile and rules.plain(c.called_tile) != rules.plain(slot.tile):
        return None
    return removal_gap(c, slot)


def removal_gap(c: Call, slot: PondSlot) -> Optional[float]:
    """Seconds between a slot's removal and the call's window (the meld camera's last calm view without the meld,
    its first with it); None when the call cannot have taken it: the tile was still in its pond after the meld
    holding it was on the table, or it left too long before or after."""
    lo, hi = c.t_window[0], c.t_first
    if slot.t_removed is None or slot.t_last > hi + TAKE_SLACK:
        return None
    d = 0.0 if lo <= slot.t_removed <= hi else min(abs(lo - slot.t_removed), abs(hi - slot.t_removed))
    return d if d <= CALL_TOL else None


def assign_calls(calls: list[Call], seq: dict[str, list[PondSlot]]) -> dict[str, list[Optional[Call]]]:
    """Which call took each removed slot. A call takes one tile, so every call is assigned to its single
    nearest compatible removal and every removal to at most one call (nearest pairs first)."""
    pairs = []
    for s, sls in seq.items():
        for k, sl in enumerate(sls):
            for c in calls:
                d = _call_dist(c, sl, s)
                if d is not None:
                    pairs.append((d, s, k, c))
    pairs.sort(key=lambda x: x[0])
    out: dict[str, list[Optional[Call]]] = {s: [None] * len(sls) for s, sls in seq.items()}
    taken: set[int] = set()
    for _, s, k, c in pairs:
        if id(c) in taken or out[s][k] is not None:
            continue
        out[s][k] = c
        taken.add(id(c))
    return out


def virtual_discards(logs: dict[str, list[PondSlot]], calls: list[Call]) -> dict[str, list[PondSlot]]:
    """The logs with a slot inserted for every chi / pon / daiminkan whose called tile matches no removed pond
    slot."""
    out = {s: list(v) for s, v in logs.items()}
    next_id = max((s.id for v in logs.values() for s in v), default=-1) + 1
    for c in calls:
        if c.type in ("ankan", "kakan") or not c.called_tile or not c.source:
            continue
        src = next((s for s in SEATS if rules.relative(c.seat, s) == c.source), None)
        if src is None:
            continue
        if any(_call_dist(c, sl, src) is not None for sl in out[src]):
            continue
        p = np.zeros(len(CLASSES))
        p[CLASS_INDEX[c.called_tile]] = 1.0
        t = c.t_window[0]
        vs = PondSlot(next_id, -1, -1, p, t, (t - 5.0, t), t, 1, 0.0)
        vs.t_removed = c.t_first
        next_id += 1
        idx = sum(1 for sl in out[src] if sl.t_first <= t)        # by time among the seat's slots
        out[src].insert(idx, vs)
    return out


def merge(logs: dict[str, list[PondSlot]], calls: list[Call], dealer: str,
          end: Optional[str] = None, wall: Optional[int] = None, exhaustive: bool = False) -> tuple[list[Turn], list[str]]:
    """Turn sequence of a hand. logs: seat -> discard log (laying order). `end`: the seat whose turn it is when
    the hand ends (a tsumo winner draws the winning tile; after a ron the seat after the loser), None for an
    exhaustive draw. `wall`: the most draws the sequence may hold (the live wall less the kans, less a tsumo's
    winning draw); `exhaustive`: the hand ends with the discard of the last of them. Returns (turns, problems).

    A pon / daiminkan whose turned tile was not seen has no source; every source is tried and the one with
    the cheapest sequence wins (a source whose pond shows a matching removal first)."""
    unknown = [c for c in calls if c.type in ("pon", "kan") and not c.source]
    if not unknown:
        turns, problems, _ = _merge(logs, calls, dealer, end, wall, exhaustive)
        return turns, problems
    def set_source(c: Call, src: str) -> None:
        c.source = src
        c.called_pos = POSITION[c.type][src]
        c.called_tile = c.tiles[c.called_pos] if c.called_pos < len(c.tiles) else c.tiles[0]

    choices = []
    for c in unknown:
        shown = []
        for src in ("kamicha", "toimen", "shimocha"):
            seat = next(s for s in SEATS if rules.relative(c.seat, s) == src)
            set_source(c, src)
            if any(_call_dist(c, sl, seat) is not None for sl in logs.get(seat, [])):
                shown.append(src)
        choices.append(shown or ["kamicha", "toimen", "shimocha"])
    best = None
    for combo in itertools.product(*choices):
        for c, src in zip(unknown, combo):
            set_source(c, src)
        turns, problems, cost = _merge(logs, calls, dealer, end, wall, exhaustive)
        cost += SKIP_COST * sum(1 for t in turns if t.virtual) / 4      # a discard nobody saw is the weaker explanation
        if best is None or cost < best[0]:
            best = (cost, combo, turns, problems)
    for c, src in zip(unknown, best[1]):
        set_source(c, src)
    return best[2], best[3]


def _inversion(prev: Optional[PondSlot], sl: PondSlot) -> float:
    """Seconds by which `sl` was sighted before `prev` could have happened (the start of its window)."""
    if prev is None:
        return 0.0
    lo = prev.t_window[0] if prev.t_window else prev.t_first
    return max(0.0, lo - sl.t_first)


def _step_cost(prev: Optional[PondSlot], sl: PondSlot) -> float:
    """Placing `sl` after `prev`: the inversion (capped), and a tie-break for turns that follow each other closely."""
    gap = max(0.0, sl.t_first - prev.t_first) if prev is not None else 0.0
    return min(INV_CAP, _inversion(prev, sl)) + GAP_COST * gap


def _merge(logs: dict[str, list[PondSlot]], calls: list[Call], dealer: str, end: Optional[str],
           wall: Optional[int] = None, exhaustive: bool = False) -> tuple[list[Turn], list[str], float]:
    logs = virtual_discards(logs, calls)
    seq = [list(logs.get(s, [])) for s in SEATS]
    n = [len(v) for v in seq]
    callers = assign_calls(calls, dict(zip(SEATS, seq)))
    most = wall if wall is not None else 10 ** 6
    # indices, whose turn, last placed (seat, index), skips, draws so far, whether the seat to play called
    start = (0, 0, 0, 0, SEATS.index(dealer), -1, -1, 0, 0, False)
    # best-first search: costs are non-negative, so the first time the goal is popped its path is the cheapest
    dist = {start: 0.0}
    parent: dict[tuple, tuple] = {}
    order = itertools.count()                                   # ties are popped first come, first served
    heap = [(0.0, next(order), start)]
    goal = None
    while heap:
        d, _, st = heapq.heappop(heap)
        if d > dist.get(st, float("inf")):
            continue
        if st[0] == "goal":
            goal = st
            break
        *idx, cur, last, last_i, skips, draws, called = st
        moves = []
        left = sum(n[k] - idx[k] for k in range(4))
        over = exhaustive and draws == most and last >= 0 and not called
        if left == 0:
            steps = 0 if end is None else (SEATS.index(end) - cur) % 4
            moves.append(("goal", SKIP_COST * steps, ("goal",)))
        elif (end is not None and SEATS[cur] == end and last >= 0) or over:
            # the hand is over when the end seat is to play, or the last draw's discard is laid: whatever is left
            # was laid after it (the reveal)
            moves.append(("goal", DROP_COST * left, ("goal",)))
        drew = draws + (0 if called else 1)             # a call turn has no draw
        if left and idx[cur] < n[cur] and not over:
            sl = seq[cur][idx[cur]]
            prev = seq[last][last_i] if last >= 0 else None
            c = callers[SEATS[cur]][idx[cur]]
            nxt = SEATS.index(c.seat) if c is not None else (cur + 1) % 4
            new = list(idx)
            new[cur] += 1
            step = _step_cost(prev, sl)
            if drew <= most:
                untaken = c is None and sl.t_removed is not None and sl.row != -1
                moves.append(("place", step + (UNTAKEN if untaken else 0.0), (*new, nxt, cur, idx[cur], 0, drew, c is not None)))
                if untaken:
                    # a tile left the pond and no known call took it: a seat whose camera missed the meld called it
                    for x in ((cur + 2) % 4, (cur + 3) % 4):
                        moves.append((f"hidden:{x}", step + HIDDEN_CALL, (*new, x, cur, idx[cur], 0, drew, True)))
            moves.append(("drop", DROP_COST, (*new, cur, last, last_i, skips, draws, called)))
        if skips < MAX_SKIPS and not all(idx[k] >= n[k] for k in range(4)) and drew <= most and not over:
            moves.append(("skip", SKIP_COST, (*idx, (cur + 1) % 4, last, last_i, skips + 1, drew, False)))
        for kind, cost, nst in moves:
            nd = d + cost
            if nd < dist.get(nst, float("inf")):
                dist[nst] = nd
                parent[nst] = (st, kind)
                heapq.heappush(heap, (nd, next(order), nst))
    # the path, from the goal back to the start
    path = []
    st = goal
    while st != start:
        prev_st, kind = parent[st]
        path.append((prev_st, kind))
        st = prev_st
    path.reverse()
    turns: list[Turn] = []
    problems: list[str] = []
    pending: dict[str, Call] = {}
    prev_slot: Optional[PondSlot] = None
    for st, kind in path:
        *idx, cur, _, _, _, _, _ = st
        s = SEATS[cur]
        if kind == "goal":
            if end is not None and SEATS[cur] != end:
                problems.append(f"the sequence ends with {s} to play, but the result says {end} (the last discards are uncertain)")
            for k in range(4):
                for sl in seq[k][idx[k]:]:
                    problems.append(f"{SEATS[k]}'s {sl.tile} seen at {sl.t_first:.0f}s is not a discard of this hand "
                                    f"(dropped: laid after the hand ended)")
            continue
        if kind == "skip":
            problems.append(f"turn {len(turns)}: no discard of {s} found in its pond (skipped)")
            pending.pop(s, None)
            continue
        sl = seq[cur][idx[cur]]
        if kind == "drop":
            problems.append(f"{s}'s {sl.tile} seen at {sl.t_first:.0f}s is not a discard of this hand (dropped)")
            continue
        c = callers[s][idx[cur]]
        if kind.startswith("hidden:"):
            c = _hidden_call(SEATS[int(kind.partition(":")[2])], s, sl)
            problems.append(f"{s}'s {sl.tile} left the pond at {sl.t_removed:.0f}s and {c.seat} plays next: a call by {c.seat} "
                            f"its meld camera did not show")
        own = pending.pop(s, None)
        turn = Turn(len(turns), s, "draw" if own is None else ("kan" if own.type == "kan" else "call"), sl, sl.t_first, c,
                    riichi=sl.is_sideways, virtual=sl.row == -1, own_call=own)
        inv = _inversion(prev_slot, sl)
        if inv > INV_TOL:
            problems.append(f"turn {turn.i}: {s}'s discard at {sl.t_first:.0f}s is out of time order "
                            f"({inv:.0f} s before the previous discard could have happened)")
        prev_slot = sl
        turns.append(turn)
        if c is not None:
            pending[c.seat] = c
    used = {id(t.call) for t in turns if t.call is not None}
    for c in calls:
        if c.type in ("ankan", "kakan"):
            # a self-kan happened inside its window (last calm view without it, first with it): its turn is the
            # seat's first discard after the window opened. A turn already holding a call keeps it (a chi or pon
            # turn has no draw to kan with); a kan with no discard after it is left to the decoder (a tsumo on
            # the rinshan draw)
            lo, hi = c.t_window[0], max(c.t_window[1], c.t_first)
            cand = [t for t in turns if t.seat == c.seat and lo - 1 <= t.t <= hi + 20 and t.own_call is None]
            if cand:
                cand[0].kind = "kan"
                cand[0].own_call = c
        elif id(c) not in used:
            problems.append(f"call {c.type} {''.join(c.tiles)} by {c.seat} at {c.t_first:.0f}s matches no discard")
    return turns, problems, dist[goal]


def _hidden_call(caller: str, discarder: str, sl: PondSlot) -> Call:
    """A call the turn order shows and no camera did: a pon of the removed tile until the solver chooses its tiles."""
    source = rules.relative(caller, discarder)
    pos = POSITION["pon"][source]
    tiles = [rules.plain(sl.tile)] * 3
    tiles[pos] = sl.tile
    return Call(caller, sl.t_removed, (sl.t_first, sl.t_removed), "pon", tiles, pos, source, sl.tile, [], 0.0, anchor="hidden")
