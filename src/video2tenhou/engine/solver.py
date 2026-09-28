"""Haipai and draw reconstruction as a constraint program (OR-Tools CP-SAT).

Variables: h0[s, k] haipai counts (13 per seat, 14 for the dealer), d[s, j, k]
one draw per draw turn. The hand after turn j is a linear expression of
those. Constraints are the rules of docs/DESIGN.md section 1; evidence enters
as costs (direct draw observations, calm 13-tile states, 14-tile states, and
rows that show part of the hand: a tile seen is in it). A pond reading is
evidence too: in repair mode every discard is a choice over all kinds, costed
by its posterior, so the cheapest re-reading that makes the hand legal is found
(a discard a call took is read by the meld camera as well and stays).
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ortools.sat.python import cp_model

from ..train.data import CLASS_INDEX, CLASSES
from . import rules

TILES = rules.KINDS + ["0m", "0p", "0s"]        # 37 kinds the solver reasons about
TI = {t: i for i, t in enumerate(TILES)}
NT = len(TILES)
SCALE = 100
REPAIR_COST = 6.0             # a discard re-read as another kind in repair mode costs this × (1 − its posterior)
MATCH_SLACK = 2               # a 14-tile row names its drawn tile only if the rest differs from the 13 by at most this
MARGIN_THREADS = 4            # margin re-solves at once (CP-SAT releases the GIL; each keeps its own workers)


def posterior_to_tiles(p) -> np.ndarray:
    """39-class posterior -> 37-kind vector (X / none mass dropped, renormalised)."""
    q = np.array([p[CLASS_INDEX[t]] for t in TILES], np.float64)
    s = q.sum()
    return q / s if s > 1e-9 else q


@dataclass
class SeatTurn:
    """One seat's turn; indices include calls, while times delimit pond sightings."""
    j: int                          # turn index of this seat (0-based)
    kind: str                       # first | draw | call | kan
    discard: Optional[str]          # tile kind, None when the hand ended without a discard
    t_pre: float                    # start of the disturbed window in which the turn happened
    t_discard: float                # when the discard was first seen
    removed: list[str] = field(default_factory=list)   # tiles that left the hand into a meld this turn
    riichi: bool = False
    discard_p: Optional[np.ndarray] = None   # 37-kind posterior of the discard: the solver may take the 2nd choice at a cost
    kan: Optional[str] = None       # ankan | kakan | daiminkan when this turn contains a kan
    kan_tile: Optional[str] = None  # the kan's tile kind, None when unknown (the solver chooses it)
    two_draws: bool = False         # ankan / kakan: a normal draw, the kan, then the rinshan draw
    meld_options: list = field(default_factory=list)   # [(tiles from the hand, cost)]: the call's tiles are a choice
    taken: bool = False             # a call took this discard: the meld camera read it too, so it is never re-read
    t_draw_min: Optional[float] = None  # earliest possible draw: preceding player's last pond view without their discard


@dataclass
class HandEvidence:
    """Soft concealed-state evidence, with explicit partial-view semantics.

    ``e`` holds aggregate counts in ``TILES`` order. Partial views from
    ``hand_evidence`` also retain normalized per-box ``slots`` in that order,
    so alternative identities remain exclusive. Count-only callers can omit
    ``slots`` and retain the aggregate cost, but cannot express that exclusivity.
    """
    seat: str
    j: int                          # state after turn j (j = -1: haipai state)
    after_draw: bool                # the 14-tile state after draw j+1
    e: np.ndarray                   # expected counts per kind (37)
    weight: float
    t0: float
    t1: float
    subset: bool = False            # a row showing part of the hand: its tiles are in the hand, others may be hidden
    hidden: int = 0                 # a calm row between turns short of the hand by this many: at most these beyond it
    slots: Optional[list[np.ndarray]] = None  # per-box alternatives; partial views must not count one box twice


@dataclass
class DrawEvidence:
    """Posterior over a single draw, usually from the unsorted end of a hand row."""
    seat: str
    j: int                          # the draw of turn j
    p: np.ndarray                   # 37
    weight: float


@dataclass
class Facts:
    """Human-confirmed constraints; unlike visual evidence these cannot be traded for cost."""
    haipai: dict = field(default_factory=dict)      # seat -> list of tiles
    draws: dict = field(default_factory=dict)       # (seat, j) -> tile
    final: dict = field(default_factory=dict)       # seat -> list of tiles (concealed at the end)
    final_excl: dict = field(default_factory=dict)  # seat -> True when `final` excludes the winning draw (tsumo winner)


@dataclass
class Solution:
    """A reconstruction and its alternatives; check ``ok`` before reading tile assignments.

    Margins certify objective separation, not visual accuracy or probability.
    Candidate ``alternative_gaps`` only prioritize draw evidence acquisition;
    they must never substitute for margins in review/completion decisions.
    ``optimal`` preserves search status even when repair changes ``status``.
    """
    status: str
    objective: float
    haipai: dict                    # seat -> list of tiles
    draws: dict                     # (seat, j) -> tile or None
    hands: dict                     # (seat, j) -> list of tiles after turn j
    margins: dict = field(default_factory=dict)     # (seat, j) -> certified lower bound on the excluded-draw cost gap
    discards: dict = field(default_factory=dict)    # (seat, j) -> tile chosen when the pond identity was a choice
    draws2: dict = field(default_factory=dict)      # (seat, j) -> rinshan draw of a self-kan turn
    kans: dict = field(default_factory=dict)        # (seat, j) -> kan tile chosen when it was unknown
    kan_added: dict = field(default_factory=dict)   # (seat, j) -> the five (plain or red) a kakan of fives added
    runner_up: dict = field(default_factory=dict)   # (seat, j) -> a candidate draw with the preferred choice forbidden
    haipai_margins: dict = field(default_factory=dict)   # seat -> certified lower bound when this exact haipai is forbidden
    melds: dict = field(default_factory=dict)       # (seat, j) -> index of the meld option the solver chose
    model_fingerprint: Optional[str] = None         # exact constraints/objective for safe in-memory margin reuse
    optimal: Optional[bool] = None                  # remains meaningful when status is later changed to repaired
    alternative_gaps: dict = field(default_factory=dict)  # feasible candidate gap: acquisition only, never confidence
    discard_margins: dict = field(default_factory=dict)  # variable pond identities need certification even when unchanged
    discard_runner_up: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """A reconstruction exists (possibly after repair)."""
        return self.status in ("optimal", "feasible", "repaired")


class HandModel:
    """Mutable evidence and rules for one hand, rebuilt as CP-SAT on each solve.

    Add observations and normalized facts before solving. Counts distinguish
    red fives; meld removals and the dealer's starting fourteenth tile are
    represented explicitly rather than repaired during export.
    Concurrent ``solve`` calls on the same mutable instance are unsupported.
    """
    def __init__(self, dealer: str, turns: dict[str, list[SeatTurn]], indicators: list[str],
                 tsumo_winner: Optional[str] = None, ura: Optional[list[str]] = None):
        self.dealer = dealer
        self.turns = turns
        self.indicators = indicators
        self.ura = list(ura or [])          # ura indicators: revealed tiles too, they count against the four
        self.tsumo_winner = tsumo_winner
        self.hand_ev: list[HandEvidence] = []
        self.draw_ev: list[DrawEvidence] = []
        self.facts = Facts()
        self.repair = False                               # every discard a choice over all kinds (DESIGN.md 4.8 Repair)
        self.forbidden_hands: list[tuple[str, int, list[str]]] = []   # (seat, j, tiles): hands the next-best solve must avoid
        self.bound_hands: list[tuple[str, int, list[str]]] = []       # (seat, j, tiles): hands the site's score requires
        # the site's result as a constraint (4.8 "The result as a constraint")
        self.win: Optional[WinSpec] = None
        self.tenpai: list[tuple[str, int, int]] = []      # (seat, state j, sets needed) of the seats tenpai at a draw
        self.result_constraints = True                    # diagnose turns them off to see whether they are what fails

    # -- building ------------------------------------------------------------------
    def _draw_turns(self, seat: str) -> list[int]:
        out = [t.j for t in self.turns[seat] if t.kind in ("draw", "kan")]
        if self.tsumo_winner == seat:
            out.append(len(self.turns[seat]))          # the winning draw, no discard after it
        return out

    def build(self, forbid: Optional[tuple[str, int, str]] = None, fixed: Optional[dict] = None,
              forbid_haipai: Optional[tuple[str, list[str]]] = None):
        """Build a fresh model and return its starting-hand, draw, and state variables.

        Optional exclusions support alternative-cost measurement. Solver variable
        maps are replaced on this instance; margin workers retain their own
        variable maps and operate on independent models.
        """
        m = cp_model.CpModel()
        h0 = {s: [m.NewIntVar(0, 4, f"h0_{s}_{k}") for k in range(NT)] for s in rules.SEATS}
        d: dict[tuple[str, int], list] = {}
        d2: dict[tuple[str, int], list] = {}          # rinshan draws of self-kan turns
        y: dict[tuple[str, int], list] = {}           # kan tile choice when the kan's kind is unknown
        vr: dict[tuple[str, int], object] = {}        # kakan of fives: 1 when the added tile is the red one
        for s in rules.SEATS:
            for j in self._draw_turns(s):
                d[(s, j)] = [m.NewBoolVar(f"d_{s}_{j}_{k}") for k in range(NT)]
                m.Add(sum(d[(s, j)]) == 1)
            for t in self.turns[s]:
                if t.two_draws:
                    d2[(s, t.j)] = [m.NewBoolVar(f"r_{s}_{t.j}_{k}") for k in range(NT)]
                    m.Add(sum(d2[(s, t.j)]) == 1)
                if t.kan and t.kan_tile is None:
                    y[(s, t.j)] = [m.NewBoolVar(f"y_{s}_{t.j}_{k}") for k in range(NT)]
                    m.Add(sum(y[(s, t.j)]) == 1)
        # haipai sizes
        for s in rules.SEATS:
            m.Add(sum(h0[s]) == (14 if s == self.dealer else 13))
        # discards: fixed to the pond identity, or a choice between the top two kinds when the second is plausible
        x: dict[tuple[str, int], dict[int, object]] = {}
        disc_cost = []
        for s in rules.SEATS:
            for t in self.turns[s]:
                if t.discard is None or t.discard not in TI:
                    continue
                if self.repair and t.discard_p is not None and not t.taken:
                    # repair: the pond reading is a cost, not a constraint
                    xs = {k: m.NewBoolVar(f"x_{s}_{t.j}_{k}") for k in range(NT)}
                    m.AddExactlyOne(xs.values())
                    x[(s, t.j)] = xs
                    for k, v in xs.items():
                        if TILES[k] != t.discard:
                            disc_cost.append(int(round(SCALE * SCALE * REPAIR_COST * (1.0 - float(t.discard_p[k])))) * v)
                    continue
                if t.discard_p is not None and not t.taken:
                    order = np.argsort(-t.discard_p)
                    k1, k2 = int(order[0]), int(order[1])
                    if t.discard_p[k2] >= 0.02 and TILES[k1] == t.discard:
                        v1, v2 = m.NewBoolVar(f"x_{s}_{t.j}_a"), m.NewBoolVar(f"x_{s}_{t.j}_b")
                        m.Add(v1 + v2 == 1)
                        x[(s, t.j)] = {k1: v1, k2: v2}
                        disc_cost.append(int(round(SCALE * SCALE * 3 * (1.0 - float(t.discard_p[k2])))) * v2)
                        continue
                x[(s, t.j)] = {TI[t.discard]: 1}
        # hand after each turn as linear expressions; non-negativity
        hands: dict[tuple[str, int], list] = {}
        mo: dict[tuple[str, int], list] = {}          # meld option choices of call turns
        for s in rules.SEATS:
            expr = list(h0[s])
            hands[(s, -1)] = list(expr)
            riichi_on = False
            for t in self.turns[s]:
                xs = x.get((s, t.j), {})
                # tiles that went into a chi / pon / daiminkan left the hand before any draw of this turn
                # (a daiminkan is followed by its rinshan draw): they must have been in the hand already
                for r in t.removed:
                    if r in TI:
                        expr[TI[r]] = expr[TI[r]] - 1
                if t.meld_options:
                    # the call's tiles from the hand: one legal composition, chosen with the hand's evidence
                    bs = [m.NewBoolVar(f"m_{s}_{t.j}_{i}") for i in range(len(t.meld_options))]
                    m.AddExactlyOne(bs)
                    mo[(s, t.j)] = bs
                    for b, (tiles, cost) in zip(bs, t.meld_options):
                        for r in tiles:
                            if r in TI:
                                expr[TI[r]] = expr[TI[r]] - b
                        if cost > 0:
                            disc_cost.append(int(round(SCALE * SCALE * cost)) * b)
                if t.removed or t.meld_options:
                    for k in range(NT):
                        m.Add(expr[k] >= 0)
                if (s, t.j) in d:
                    expr = [expr[k] + d[(s, t.j)][k] for k in range(NT)]
                    if riichi_on and xs and not t.two_draws:      # after riichi every draw is discarded at once
                        for k in range(NT):
                            m.Add(d[(s, t.j)][k] == xs.get(k, 0))
                    elif riichi_on and t.two_draws:
                        # after riichi a concealed kan is only of the tile just drawn (the hand is frozen)
                        if t.kan_tile in TI:
                            red = {"5m": "0m", "5p": "0p", "5s": "0s"}.get(t.kan_tile)
                            m.Add(d[(s, t.j)][TI[t.kan_tile]] + (d[(s, t.j)][TI[red]] if red else 0) == 1)
                        elif (s, t.j) in y:
                            for k in range(NT):
                                m.Add(d[(s, t.j)][k] == y[(s, t.j)][k])
                if t.kan in ("ankan", "kakan"):
                    n_out = 4 if t.kan == "ankan" else 1
                    red = {"5m": "0m", "5p": "0p", "5s": "0s"}.get(t.kan_tile or "")
                    if t.kan_tile is not None and t.kan_tile in TI and red and t.kan == "ankan":
                        # a concealed kan of fives is all four fives: three plain and the red one
                        expr[TI[t.kan_tile]] = expr[TI[t.kan_tile]] - 3
                        expr[TI[red]] = expr[TI[red]] - 1
                    elif t.kan_tile is not None and t.kan_tile in TI and red:
                        # kakan of a five: the added tile is whichever five the player still holds
                        vr[(s, t.j)] = m.NewBoolVar(f"kr_{s}_{t.j}")
                        expr[TI[t.kan_tile]] = expr[TI[t.kan_tile]] - (1 - vr[(s, t.j)])
                        expr[TI[red]] = expr[TI[red]] - vr[(s, t.j)]
                    elif t.kan_tile is not None and t.kan_tile in TI:
                        expr[TI[t.kan_tile]] = expr[TI[t.kan_tile]] - n_out
                    elif (s, t.j) in y:
                        for k in range(NT):
                            if TILES[k] in ("5m", "5p", "5s", "0m", "0p", "0s"):
                                m.Add(y[(s, t.j)][k] == 0)     # a kan of fives is named by the meld camera, never guessed
                            expr[k] = expr[k] - n_out * y[(s, t.j)][k]
                    for k in range(NT):
                        m.Add(expr[k] >= 0)              # the kan tiles must be in the hand before the rinshan draw
                    if (s, t.j) in d2:
                        expr = [expr[k] + d2[(s, t.j)][k] for k in range(NT)]
                        if riichi_on and xs:                      # after riichi the rinshan draw is discarded at once
                            for k in range(NT):
                                m.Add(d2[(s, t.j)][k] == xs.get(k, 0))
                for k, v in xs.items():
                    expr[k] = expr[k] - v
                for k in range(NT):
                    m.Add(expr[k] >= 0)
                hands[(s, t.j)] = list(expr)
                if t.riichi:
                    riichi_on = True
            if self.tsumo_winner == s:
                jw = len(self.turns[s])
                expr = [expr[k] + d[(s, jw)][k] for k in range(NT)]
                hands[(s, jw)] = list(expr)
        if self.result_constraints:
            self._result_constraints(m, hands, x)
        # counting rule over the whole hand: hands, draws and every revealed indicator (dora and ura)
        ind = np.zeros(NT, int)
        for t in self.indicators + self.ura:
            if t in TI:
                ind[TI[t]] += 1
        for k in range(NT):
            total = sum(h0[s][k] for s in rules.SEATS) + sum(v[k] for v in d.values()) + sum(v[k] for v in d2.values()) + int(ind[k])
            m.Add(total <= rules.max_count(TILES[k]))
        # facts
        for s, tiles in self.facts.haipai.items():
            for k in range(NT):
                m.Add(h0[s][k] == sum(1 for t in tiles if t == TILES[k]))
        for (s, j), tile in self.facts.draws.items():
            if (s, j) in d and tile in TI:
                m.Add(d[(s, j)][TI[tile]] == 1)
        for s, tiles in self.facts.final.items():
            last = max((j for (ss, j) in hands if ss == s), default=None)
            if last is not None and self.facts.final_excl.get(s) and self.tsumo_winner == s:
                last -= 1                                   # the state after the last discard, before the winning draw
            if last is not None:
                for k in range(NT):
                    m.Add(hands[(s, last)][k] == sum(1 for t in tiles if t == TILES[k]))
        # Keep exclusion order unchanged when cloning draw alternatives.
        m._draw_forbid_index = len(m.proto.constraints)
        if forbid is not None:
            s, j, tile = forbid
            if (s, j) in d:
                m.Add(d[(s, j)][TI[tile]] == 0)
        if forbid_haipai is not None:
            _differs(m, h0[forbid_haipai[0]], forbid_haipai[1], f"h0_{forbid_haipai[0]}")
        for n_, (s, j, tiles) in enumerate(self.forbidden_hands):
            if (s, j) in hands:
                _differs(m, hands[(s, j)], tiles, f"nb{n_}")
        for s, j, tiles in self.bound_hands:
            if (s, j) in hands:
                for k in range(NT):
                    m.Add(hands[(s, j)][k] == sum(1 for t in tiles if t == TILES[k]))
        if fixed:
            for (s, j), tile in fixed.items():
                if (s, j) in d:
                    m.Add(d[(s, j)][TI[tile]] == 1)
        # objective
        terms = []
        for ev in self.hand_ev:
            key = (ev.seat, ev.j)
            if key not in hands:
                continue
            expr = hands[key]
            if ev.after_draw:
                nxt = (ev.seat, ev.j + 1)
                if nxt not in d:
                    continue
                expr = [expr[k] + d[nxt][k] for k in range(NT)]
            w = int(round(ev.weight * SCALE))
            if w <= 0:
                continue
            if (ev.subset or ev.hidden) and ev.slots is not None:
                terms.append(_partial_cost(m, expr, ev.slots, w * (2 if ev.hidden else 1), len(terms)))
                continue
            for k in range(NT):
                target = int(round(ev.e[k] * SCALE))
                if ev.subset and target <= 0:
                    continue
                dev = m.NewIntVar(0, 20 * SCALE, f"dev_{ev.seat}_{ev.j}_{k}_{len(terms)}")
                if not ev.subset and not ev.hidden:
                    m.Add(dev >= expr[k] * SCALE - target)       # a full row: no tile more than it shows ...
                m.Add(dev >= target - expr[k] * SCALE)           # ... and every tile it shows is in the hand
                terms.append(w * dev)
            if ev.hidden:
                # all but `hidden` tiles shown: the hand holds at most that many tiles beyond the row
                ups = []
                for k in range(NT):
                    up = m.NewIntVar(0, 20 * SCALE, f"up_{ev.seat}_{ev.j}_{k}_{len(terms)}")
                    m.Add(up >= expr[k] * SCALE - int(round(ev.e[k] * SCALE)))
                    ups.append(up)
                excess = m.NewIntVar(0, 20 * NT * SCALE, f"ex_{ev.seat}_{ev.j}_{len(terms)}")
                m.Add(excess >= sum(ups) - ev.hidden * SCALE)
                terms.append(w * excess)
        for ev in self.draw_ev:
            key = (ev.seat, ev.j)
            if key not in d:
                continue
            w = int(round(ev.weight * SCALE))
            for k in range(NT):
                c = int(round((1.0 - ev.p[k]) * SCALE))
                if c > 0:
                    terms.append(w * c * d[key][k])
        terms += disc_cost
        m.Minimize(sum(terms) if terms else 0)
        self._x, self._d2, self._y, self._vr, self._mo = x, d2, y, vr, mo
        return m, h0, d, hands

    def _result_constraints(self, m, hands: dict, x: dict) -> None:
        """The winner's final hand is a winning hand; the tenpai seats of a draw are tenpai."""
        w = self.win
        if w is not None and (w.seat, w.j) in hands:
            expr = list(hands[(w.seat, w.j)])
            if w.ron_from is not None:
                # a ron: the loser's winning discard completes the hand
                for k, v in x.get(w.ron_from, {}).items():
                    expr[k] = expr[k] + v
            _complete(m, _counts34(expr), w.sets, f"win_{w.seat}")
        for seat, j, sets in self.tenpai:
            if (seat, j) not in hands:
                continue
            wait = [m.NewBoolVar(f"wait_{seat}_{k}") for k in range(34)]
            m.AddExactlyOne(wait)
            c34 = _counts34(hands[(seat, j)])
            _complete(m, [c34[k] + wait[k] for k in range(34)], sets, f"tenpai_{seat}")

    def _discard_of(self, key) -> Optional[str]:
        s, j = key
        for t in self.turns[s]:
            if t.j == j:
                return t.discard
        return None

    # -- solving -------------------------------------------------------------------
    def solve(self, *, time_limit: float = 60.0, margins: bool = True, workers: int = 8, margin_thr: float = 0.5,
              prior: Optional["Solution"] = None) -> Solution:
        """Reconstruct the hand and optionally certify alternatives for uncertain tiles.

        ``prior`` seeds the new search with soft hints, allowing new evidence or
        facts to replace old choices. It reuses confident margins only when the complete model,
        baseline objective and selected tile or starting multiset are unchanged. New evidence must be
        re-evaluated even if it happens to preserve the same preferred tiles.
        """
        m, h0, d, hands = self.build()
        fingerprint = hashlib.sha256(str(m.proto).encode("utf-8")).hexdigest() if margins else None
        if prior is not None and prior.ok:
            # Evidence can invalidate the previous assignment. Hints preserve a
            # useful incumbent without constraining the updated reconstruction.
            for seat, variables in h0.items():
                if seat in prior.haipai:
                    counts = Counter(prior.haipai[seat])
                    for tile, variable in zip(TILES, variables):
                        m.AddHint(variable, counts[tile])
            for variables, choices in ((d, prior.draws), (self._d2, prior.draws2), (self._y, prior.kans)):
                for key, vs in variables.items():
                    if choices.get(key) in TI:
                        for tile, variable in zip(TILES, vs):
                            m.AddHint(variable, int(tile == choices[key]))
            for key, variable in self._vr.items():
                if key in prior.kan_added:
                    m.AddHint(variable, int(prior.kan_added[key] in rules.REDS))
            for key, variables in self._mo.items():
                if key in prior.melds:
                    for index, variable in enumerate(variables):
                        m.AddHint(variable, int(index == prior.melds[key]))
            for key, variables in self._x.items():
                tile = prior.discards.get(key, self._discard_of(key))
                if tile in TI:
                    for index, variable in variables.items():
                        if not isinstance(variable, int):  # fixed discards have no decision variable
                            m.AddHint(variable, int(index == TI[tile]))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = workers
        st = solver.Solve(m)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # UNKNOWN (the time limit ran out before any solution) is not a proof of infeasibility
            return Solution("infeasible" if st == cp_model.INFEASIBLE else "unsolved", float("inf"), {}, {}, {})
        obj = solver.ObjectiveValue() / SCALE / SCALE
        haipai = {s: [TILES[k] for k in range(NT) for _ in range(solver.Value(h0[s][k]))] for s in rules.SEATS}
        draws = {key: TILES[int(np.argmax([solver.Value(v) for v in vs]))] for key, vs in d.items()}
        hs = {key: [TILES[k] for k in range(NT) for _ in range(solver.Value(expr[k]))] for key, expr in hands.items()}
        sol = Solution("optimal" if st == cp_model.OPTIMAL else "feasible", obj, haipai, draws, hs)
        sol.optimal = st == cp_model.OPTIMAL
        if margins:
            sol.model_fingerprint = fingerprint
        for key, vs in self._d2.items():
            sol.draws2[key] = TILES[int(np.argmax([solver.Value(v) for v in vs]))]
        for key, vs in self._y.items():
            sol.kans[key] = TILES[int(np.argmax([solver.Value(v) for v in vs]))]
        for key, v in self._vr.items():
            tile = next(t.kan_tile for t in self.turns[key[0]] if t.j == key[1])
            sol.kan_added[key] = {"5m": "0m", "5p": "0p", "5s": "0s"}[tile] if solver.Value(v) else tile
        for key, bs in self._mo.items():
            sol.melds[key] = next(i for i, b in enumerate(bs) if solver.Value(b))
        for key, xs in self._x.items():
            if len(xs) < 2:
                continue
            for k, v in xs.items():
                if solver.Value(v) == 1 and TILES[k] != self._discard_of(key):
                    sol.discards[key] = TILES[k]
        if margins:
            hint = ({s_: [solver.Value(h0[s_][k]) for k in range(NT)] for s_ in rules.SEATS},
                    {key2: [solver.Value(v) for v in vs] for key2, vs in d.items()})
            draws_todo = []
            for key, tile in draws.items():
                if (prior is not None and prior.model_fingerprint == sol.model_fingerprint
                        and prior.objective == obj and key in prior.margins
                        and prior.draws.get(key) == tile and prior.margins[key] > margin_thr + 1e-9):
                    sol.margins[key] = prior.margins[key]
                    sol.runner_up[key] = prior.runner_up.get(key)
                    sol.alternative_gaps[key] = prior.alternative_gaps.get(key, prior.margins[key])
                elif key in self.facts.draws:
                    sol.margins[key] = float("inf")          # fixed by a fact
                    sol.alternative_gaps[key] = float("inf")
                else:
                    draws_todo.append((key, tile))
            same_model = prior is not None and prior.model_fingerprint == fingerprint and prior.objective == obj
            haipai_todo = []
            for seat in rules.SEATS:
                if seat in self.facts.haipai:
                    continue
                if (same_model and Counter(prior.haipai.get(seat, [])) == Counter(haipai[seat])
                        and prior.haipai_margins.get(seat, 0) > margin_thr + 1e-9):
                    sol.haipai_margins[seat] = prior.haipai_margins[seat]
                else:
                    haipai_todo.append(seat)
            sol.haipai_margins.update({s_: float("inf") for s_ in rules.SEATS if s_ in self.facts.haipai})
            # Capture variable maps before starting workers: haipai alternatives
            # rebuild this mutable instance's maps on their own models.
            discard_todo = []
            for key, variables in self._x.items():
                if len(variables) < 2:
                    continue
                chosen = sol.discards.get(key, self._discard_of(key))
                if (same_model and prior.discards.get(key, self._discard_of(key)) == chosen
                        and prior.discard_margins.get(key, 0) > margin_thr + 1e-9):
                    sol.discard_margins[key] = prior.discard_margins[key]
                    sol.discard_runner_up[key] = prior.discard_runner_up.get(key)
                else:
                    discard_todo.append((key, variables, variables[TI[chosen]]))
            # Exclude each choice to bound its objective separation. A returned
            # runner-up is a feasible candidate, not necessarily the closest
            # alternative when its search hits the deadline or stops on proof.
            with ThreadPoolExecutor(MARGIN_THREADS) as ex:
                fd = {key: ex.submit(self._resolve, hint, obj, workers, forbid=(key[0], key[1], tile), watch=key,
                                     baseline=(m, h0, d), margin_thr=margin_thr)
                      for key, tile in draws_todo}
                fh = {s_: ex.submit(self._resolve, hint, obj, workers, forbid_haipai=(s_, haipai[s_]), margin_thr=margin_thr) for s_ in haipai_todo}
                fx = {key: ex.submit(self._resolve, hint, obj, workers, baseline=(m, h0, d),
                                     forbid_variable=variable, watch_variables=variables, margin_thr=margin_thr)
                      for key, variables, variable in discard_todo}
                for key, f in fd.items():
                    sol.margins[key], sol.runner_up[key], sol.alternative_gaps[key] = f.result()
                for s_, f in fh.items():
                    sol.haipai_margins[s_] = f.result()[0]
                for key, f in fx.items():
                    sol.discard_margins[key], sol.discard_runner_up[key], _ = f.result()
        return sol

    def _resolve(self, hint: tuple, obj: float, workers: int, *, watch: Optional[tuple] = None,
                 baseline=None, forbid_variable=None, watch_variables=None, margin_thr: float = 0.5,
                 stop_when_certified: bool = True, **forbid) -> tuple[float, Optional[str], float]:
        """Objective increase of the model with one decision forbidden (warm-started from the solution), and the
        draw `watch` takes then. A timed-out feasible solve contributes only its
        certified lower bound: the current candidate's cost is an upper bound
        and could falsely make an ambiguous draw appear certain. Its draw is
        still useful as a possible alternative. The third value is its cost gap,
        used only to select video rereads; it never certifies confidence.
        The search can stop as soon as its lower bound certifies a gap above
        ``margin_thr``; this does not require optimizing the rejected alternative.
        A search without a candidate can still supply a certified lower bound;
        its acquisition gap remains zero because no alternative was found."""
        if forbid_variable is not None:
            m2, h0b, db = _clone_forbidden_model(*baseline, forbid_variable)
        elif baseline is not None and set(forbid) == {"forbid"}:
            m2, h0b, db = _clone_draw_model(*baseline, forbid["forbid"])
        else:
            m2, h0b, db, *_ = self.build(**forbid)
        # A cloned main model may already contain prior-assignment hints.
        # Duplicate variable hints invalidate CP-SAT's input.
        m2.ClearHints()
        h0v, dv = hint
        for s_ in rules.SEATS:
            for k in range(NT):
                m2.AddHint(h0b[s_][k], h0v[s_][k])
        skip = forbid.get("forbid")
        for key2, vs in db.items():
            if skip is not None and key2 == (skip[0], skip[1]):
                continue
            for k, v in enumerate(vs):
                m2.AddHint(v, dv[key2][k])
        s2 = cp_model.CpSolver()
        s2.parameters.max_time_in_seconds = 4.0
        s2.parameters.num_workers = workers
        proven_bound = 0.0

        def stop_after_proof(bound):
            nonlocal proven_bound
            if math.isfinite(bound):
                proven_bound = max(proven_bound, bound)
                if bound / SCALE / SCALE - obj > margin_thr + 1e-9:
                    s2.StopSearch()

        if stop_when_certified:
            # Confidence needs a threshold certificate, not the exact optimum
            # of a rejected reconstruction. Never stop on an incumbent cost.
            s2.best_bound_callback = stop_after_proof
        r = s2.Solve(m2)
        if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            alt = TILES[int(np.argmax([s2.Value(v) for v in db[watch]]))] if watch in db else None
            if watch_variables is not None:
                alt = next(TILES[k] for k, v in watch_variables.items() if s2.Value(v))
            bound = s2.ObjectiveValue() if r == cp_model.OPTIMAL else max(proven_bound, s2.BestObjectiveBound())
            return max(0.0, bound / SCALE / SCALE - obj), alt, max(0.0, s2.ObjectiveValue() / SCALE / SCALE - obj)
        if r == cp_model.UNKNOWN:
            # A proof bound does not require an incumbent. Discarding it turns
            # already-certified choices into needless review and video work.
            # All model costs are nonnegative, so the default zero bound from
            # a search stopped before initialization is conservative as well.
            bound = max(proven_bound, s2.BestObjectiveBound())
            margin = max(0.0, bound / SCALE / SCALE - obj) if math.isfinite(bound) else 0.0
            return margin, None, 0.0
        return (float("inf"), None, float("inf")) if r == cp_model.INFEASIBLE else (0.0, None, 0.0)


def _clone_draw_model(base, h0, draws, forbid):
    """Copy a draw alternative with exactly the original CP-SAT input ordering.

    Variable indices, objective terms and constraint ordering stay identical.
    Variable handles can be shared because CP-SAT uses their unchanged indices.
    Starting-hand alternatives still rebuild: those introduce extra variables.
    """
    seat, turn, tile = forbid
    variable = draws[(seat, turn)][TI[tile]] if (seat, turn) in draws else None
    return _clone_forbidden_model(base, h0, draws, variable)


def _clone_forbidden_model(base, h0, draws, variable):
    """Exclude one selected Boolean while preserving baseline ordering and maps."""
    cloned = base.Clone()
    if variable is not None:
        constraint = cloned.Add(variable == 0)
        exclusion = type(constraint.proto)()
        exclusion.copy_from(constraint.proto)
        constraints = list(base.proto.constraints)
        offset = base._draw_forbid_index
        cloned.proto.constraints.clear()
        cloned.proto.constraints.extend(constraints[:offset])
        cloned.proto.constraints.append(exclusion)
        cloned.proto.constraints.extend(constraints[offset:])
    return cloned, h0, draws


def _differs(m, expr: list, tiles: list[str], name: str) -> None:
    """The counts `expr` hold at least one kind fewer than `tiles` (the sizes are fixed, so the hand differs)."""
    fewer = []
    for k in range(NT):
        v = sum(1 for t in tiles if t == TILES[k])
        if v:
            b = m.NewBoolVar(f"fewer_{name}_{k}")
            m.Add(expr[k] <= v - 1).OnlyEnforceIf(b)
            fewer.append(b)
    m.AddBoolOr(fewer)


@dataclass
class WinSpec:
    """Where the winner's winning hand is in the model: the state after its turn j (a tsumo: after the winning
    draw), plus the loser's discard of turn j_loser for a ron; `sets` the sets the concealed part must hold."""
    seat: str
    j: int
    sets: int
    ron_from: Optional[tuple[str, int]] = None


TERMINALS = [k for k in range(34) if k >= 27 or k % 9 in (0, 8)]


def _counts34(expr: list) -> list:
    """37-kind counts -> 34 kinds (a red five is a five)."""
    out = [expr[TI[t]] for t in rules.KINDS]
    for red, plain in rules.REDS.items():
        k = rules.KINDS.index(plain)
        out[k] = out[k] + expr[TI[red]]
    return out


def _complete(m, c34: list, sets: int, name: str) -> None:
    """Constrain 34-kind counts to a complete hand with `sets` sets besides the melds and one pair — or, closed,
    seven pairs or the thirteen orphans. Integer decomposition: pons per kind, chis per start, the pair."""
    std = m.NewBoolVar(f"{name}_std")
    forms = [std]
    pon = [m.NewBoolVar(f"{name}_pon{k}") for k in range(34)]
    pair = [m.NewBoolVar(f"{name}_pair{k}") for k in range(34)]
    chi = {k: m.NewIntVar(0, 4, f"{name}_chi{k}") for k in range(27) if k % 9 <= 6}
    for k in range(34):
        covering = [chi[k - d] for d in (0, 1, 2) if (k - d) in chi and (k - d) // 9 == k // 9]
        m.Add(c34[k] == 3 * pon[k] + 2 * pair[k] + sum(covering)).OnlyEnforceIf(std)
    m.Add(sum(pair) == 1).OnlyEnforceIf(std)
    m.Add(sum(pon) + sum(chi.values()) == sets).OnlyEnforceIf(std)
    if sets == 4:
        seven = m.NewBoolVar(f"{name}_7p")
        pp = [m.NewBoolVar(f"{name}_pp{k}") for k in range(34)]
        for k in range(34):
            m.Add(c34[k] == 2 * pp[k]).OnlyEnforceIf(seven)
        m.Add(sum(pp) == 7).OnlyEnforceIf(seven)
        orphans = m.NewBoolVar(f"{name}_13o")
        for k in range(34):
            if k in TERMINALS:
                m.Add(c34[k] >= 1).OnlyEnforceIf(orphans)
            else:
                m.Add(c34[k] == 0).OnlyEnforceIf(orphans)
        forms += [seven, orphans]
    m.AddExactlyOne(forms)


# -----------------------------------------------------------------------------
# evidence mapping
# -----------------------------------------------------------------------------

def state_of(seat_turns: list[SeatTurn], t0: float, t1: float) -> Optional[int]:
    """Index j of the state a calm interval [t0, t1] of the hand band belongs to (-1 = haipai state),
    or None when the interval straddles a turn of this seat."""
    j = -1
    for t in seat_turns:
        if t.t_discard <= t0 + 0.6:
            j = t.j
        elif t.t_discard - 0.6 <= t1:
            return None           # the discard happened inside the interval
    return j


def open_turn(seat_turns: list[SeatTurn], t0: float, t1: float) -> Optional[SeatTurn]:
    """The turn whose window [t_pre, t_discard] the interval [t0, t1] enters without reaching the discard's
    first sighting: the discard may already have happened inside the interval, so a row with the resting
    count there is ambiguous (before the draw or after the discard); only the row with one tile more is a
    state, the one after the draw."""
    for t in seat_turns:
        if t.t_discard - 0.6 > t1:
            # the pond may first show a discard several seconds after it was made (an arm over the pond), but
            # not much longer: a row at rest that ends more than 8 s before the sighting was still waiting to draw
            return t if t.t_pre <= t1 and t1 >= t.t_discard - 8.0 else None
    return None


def _partial_cost(model, counts, slots, weight, tag):
    """Match each visible box to at most one concealed tile, allowing misreads.

    Summed posterior counts let a hidden extra tile satisfy both alternatives
    of one ambiguous box. Capacity-constrained matching keeps those alternatives
    exclusive. Unmatched boxes cost one; hidden tiles contribute no evidence.
    Equal posteriors share integer flows, and zero-rounded rewards need no
    variables because leaving a box unmatched has the same cost.
    """
    groups = Counter(tuple(int(round(float(p) * SCALE)) for p in slot) for slot in slots)
    used = [[] for _ in range(NT)]
    rewards = []
    for i, (posterior, n) in enumerate(groups.items()):
        choices = []
        for k, reward in enumerate(posterior):
            if reward <= 0:
                continue
            choice = model.NewIntVar(0, n, f"partial_{tag}_{i}_{k}")
            choices.append(choice)
            used[k].append(choice)
            rewards.append(reward * choice)
        model.Add(sum(choices) <= n)
    for k, choices in enumerate(used):
        if choices:
            model.Add(sum(choices) <= counts[k])
    return weight * (len(slots) * SCALE - sum(rewards))


def expected_counts(slots: list[dict]) -> np.ndarray:
    """Sum normalized tile posteriors into soft counts, excluding face-down/empty classes."""
    e = np.zeros(NT)
    for s in slots:
        e += posterior_to_tiles(s["p"])
    return e


def _tops(slots: list[dict]) -> list[str]:
    return [rules.plain(CLASSES[int(np.argmax(sl["p"]))]) for sl in slots]


def drawn_end(slots: list[dict], ref: Optional[list], prior: Optional[tuple[int, int]]) -> list[tuple[str, float]]:
    """Which end of a row with one tile more than the resting count holds the drawn tile: [(end, weight)].

    The end whose removal leaves the previous resting row `ref` (posteriors), compared as multisets (the player
    may not have sorted), names it, strongly — if the rest differs from `ref` by at most MATCH_SLACK tiles; a
    reference further off is of another state and says nothing. Undecided, both ends are candidates weighted
    by the ends this player has used (`prior` = (left, right) counts): players are consistent."""
    if ref is not None:
        want = Counter(rules.plain(CLASSES[int(np.argmax(p))]) for p in ref)
        tops = _tops(slots)
        diff = {"L": sum((want - Counter(tops[1:])).values()), "R": sum((want - Counter(tops[:-1])).values())}
        best = min(diff, key=diff.get)
        other = "R" if best == "L" else "L"
        if diff[best] <= MATCH_SLACK and diff[best] < diff[other]:
            return [(best, 2.0)]
    n_l, n_r = prior or (0, 0)
    p_l = (n_l + 1) / (n_l + n_r + 2)
    return [("L", 1.2 * p_l), ("R", 1.2 * (1 - p_l))]


def hand_evidence(seat: str, seat_turns: list[SeatTurn], observations: list[dict], melds_before: dict[int, int],
                  dealer: bool, wins_by_tsumo: bool = False) -> tuple[list[HandEvidence], list[DrawEvidence], tuple[int, int]]:
    """Map the hand observations of one seat to states (DESIGN.md 4.8, evidence 1-3). Returns the hand evidence,
    the direct draw evidence, and the seat's end habit (draws seen at the left end, at the right end).

    - a row with one tile more than the resting count, inside the turn: the state after the draw, and the drawn
      tile at an end of the row;
    - a row with the resting count between turns, calm or read-floor: the state (it shows every tile);
    - a calm row between turns one or two tiles short of it: every tile it shows is in the state, which holds at
      most that many more (a hand resting over an end of the row), at full weight;
    - anything else — a partial view, fewer tiles, a row inside a turn's window (before the draw or already after
      the discard), a view across a discard — shows part of the hand at some moment of the turn, and every such
      moment's tiles are in the hand after the turn's draw: the tiles seen are a sub-multiset of it."""
    hev: list[HandEvidence] = []
    rows14: list[tuple[int, list[dict], Optional[list], float]] = []
    prev13: dict[int, list] = {}
    kind_of = {t.j: t.kind for t in seat_turns}
    turn_of = {t.j: t for t in seat_turns}
    if wins_by_tsumo:
        kind_of[len(seat_turns)] = "draw"          # the winning draw: no discard after it
    for o in sorted(observations, key=lambda o: o["t0"]):
        if o["n_used"] == 0 or not o["slots"]:
            continue
        w = float(o["quality"]) * min(1.0, o["n_used"] / 3.0)
        j = state_of(seat_turns, o["t0"], o["t1"])
        straddle = j is None
        if straddle:
            j = max((t.j for t in seat_turns if t.t_discard <= o["t0"] + 0.6), default=-1)
        base = 14 if (j == -1 and dealer) else 13 - 3 * melds_before.get(j, 0)
        count = o["count"]
        e = expected_counts(o["slots"])
        slot_p = [posterior_to_tiles(sl["p"]) for sl in o["slots"]]
        has_draw = kind_of.get(j + 1) == "draw"
        nxt = turn_of.get(j + 1)
        before_draw = (nxt is not None and nxt.t_draw_min is not None
                       and o["t1"] < nxt.t_draw_min - 0.6)
        open_ = straddle or open_turn(seat_turns, o["t0"], o["t1"]) is not None
        if before_draw and has_draw and not straddle and count == base + 1:
            # The preceding player has not discarded yet, so this cannot be a
            # draw. Extra detections must not turn persistent hand tiles into
            # evidence for the next draw. Keep their soft lower-bound evidence
            # on the resting state; the rules expose any impossible extra tiles.
            hev.append(HandEvidence(seat, j, False, e, 0.5 * w, o["t0"], o["t1"], subset=True, slots=slot_p))
        elif count == base + 1 and has_draw and not straddle:
            hev.append(HandEvidence(seat, j, True, e, w, o["t0"], o["t1"]))
            rows14.append((j, o["slots"], prev13.get(j), w))
        elif count == base and not open_:
            # between turns the hand holds exactly the resting count: a row showing that many hides nothing, read-floor
            # (partial) or calm alike
            hev.append(HandEvidence(seat, j, False, e, w, o["t0"], o["t1"]))
            prev13[j] = [sl["p"] for sl in o["slots"]]
        elif base - 2 <= count < base and not open_ and not o.get("partial"):
            hev.append(HandEvidence(seat, j, False, e, w, o["t0"], o["t1"], hidden=base - count, slots=slot_p))
        elif count <= base + 1 and kind_of.get(j + 1) in ("draw", "call", "first", None):
            # part of the hand: a sub-multiset of the state after this turn's draw (of the state itself when the
            # turn has no draw: a call, the dealer's first turn, or after the seat's last turn)
            hev.append(HandEvidence(seat, j, has_draw, e, 0.5 * w, o["t0"], o["t1"], subset=True, slots=slot_p))
    # the drawn end: decided rows first, to learn the player's habit, which then weighs the undecided ones
    decided = [drawn_end(sl, ref, None) for _, sl, ref, _ in rows14]
    habit = (sum(1 for d in decided if len(d) == 1 and d[0][0] == "L"), sum(1 for d in decided if len(d) == 1 and d[0][0] == "R"))
    dev: list[DrawEvidence] = []
    for (j, slots, ref, w), first in zip(rows14, decided):
        for end, weight in (first if len(first) == 1 else drawn_end(slots, ref, habit)):
            dev.append(DrawEvidence(seat, j + 1, posterior_to_tiles(slots[0 if end == "L" else -1]["p"]), weight * w))
    return hev, dev, habit


@dataclass
class Culprit:
    """One decision without which an infeasible hand becomes legal: what a conflict question asks about."""
    kind: str                       # fact | result | riichi | meld | discard
    seat: Optional[str]
    j: Optional[int]                # the seat's turn (meld, discard, riichi)
    text: str
    fact: Optional[str] = None      # fact: haipai | draw | final


def diagnose(model: "HandModel", time_limit: float = 5.0) -> list[Culprit]:
    """When the model is infeasible (even after repair): the single decisions whose removal makes it legal, the
    most specific first — a reviewer's fact that contradicts the video, the result constraint (the winner's hand
    as reconstructed cannot win), a riichi turn, one call. Empty when no single decision does."""
    out: list[Culprit] = []

    def feasible() -> bool:
        m, *_ = model.build()
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = time_limit
        sv.parameters.num_workers = 8
        return sv.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    # 1. a reviewer's fact, one at a time
    full = model.facts
    singles = ([("haipai", s_, s_, v) for s_, v in full.haipai.items()] + [("draw", k[0], k, v) for k, v in full.draws.items()]
               + [("final", s_, s_, v) for s_, v in full.final.items()])
    for kind, seat, key, val in singles:
        model.facts = Facts(**{k: dict(v) for k, v in full.__dict__.items()})
        {"haipai": model.facts.haipai, "draw": model.facts.draws, "final": model.facts.final}[kind].pop(key)
        if feasible():
            shown = " ".join(val) if isinstance(val, list) else val
            out.append(Culprit("fact", seat, key[1] if kind == "draw" else None,
                               f"your {kind} fact for {seat} ({shown}) contradicts what the video shows", fact=kind))
    model.facts = full
    # 2. the site's result as a constraint
    if model.result_constraints and (model.win is not None or model.tenpai):
        model.result_constraints = False
        if feasible():
            who = model.win.seat if model.win is not None else None
            out.append(Culprit("result", who, None, "the winner's hand as reconstructed cannot win (or a tenpai seat cannot "
                                                    "be tenpai): a call or a discard of that seat is read wrong"))
        model.result_constraints = True
    # 3. a riichi turn
    for s_ in rules.SEATS:
        for t in model.turns[s_]:
            if t.riichi:
                t.riichi = False
                if feasible():
                    out.append(Culprit("riichi", s_, t.j, f"{s_}'s riichi at its turn {t.j} ({t.t_discard:.0f}s) cannot be: "
                                                          f"after it the hand would have to change"))
                t.riichi = True
    # 4. one call
    for s_ in rules.SEATS:
        for t in model.turns[s_]:
            if not (t.removed or t.meld_options):
                continue
            r, o = t.removed, t.meld_options
            t.removed, t.meld_options = [], []
            if feasible():
                shown = " ".join(r) if r else " or ".join("".join(h) for h, _ in o)
                out.append(Culprit("meld", s_, t.j, f"{s_}'s call before its discard at {t.t_discard:.0f}s cannot be: its hand "
                                                    f"never holds {shown}"))
            t.removed, t.meld_options = r, o
    return out
