# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Haipai and draw reconstruction as a constraint program (OR-Tools CP-SAT).

Variables: h0[s, k] haipai counts (13 per seat, 14 for the dealer), d[s, j, k]
one draw per draw turn, and h[s, j, k] the concealed counts after turn j (the
previous state plus the turn's acquisitions, less what left the hand).
Constraints are the rules of docs/DESIGN.md section 1; evidence enters as costs
(direct draw observations, calm 13-tile states, 14-tile states, and rows that
show part of the hand: a tile seen is in it). A pond reading is evidence too: in
repair mode every discard is a choice over all kinds, costed by its posterior,
so the cheapest re-reading that makes the hand legal is found (a discard a call
took is read by the meld camera as well and stays). Certification bounds the
cost increase of changing each decision of a solution.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from copy import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from ortools.sat.python import cp_model

from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES

from . import rules
from .confidence import FIXED, MARGIN_REVIEW, MARGIN_TOLERANCE, Certificate

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

MIN_POSTERIOR_MASS = 1e-9
MIN_ALTERNATIVE_DISCARD_PROBABILITY = 0.02
MIN_DECISION_CHOICES = 2
FIRST_HONOR_INDEX = 27
LAST_CHI_START_INDEX = 6
SETS_IN_CLOSED_HAND = 4
PAIRS_IN_SEVEN_PAIRS = 7


TILES = [*rules.KINDS, "0m", "0p", "0s"]  # 37 kinds the solver reasons about
TI = {t: i for i, t in enumerate(TILES)}
NT = len(TILES)
SCALE = 100
# a discard re-read as another kind in repair mode costs this x (1 - its posterior)
REPAIR_COST = 6.0
# a 14-tile row names its drawn tile only if the rest differs from the 13 by at most
# this
MATCH_SLACK = 2
# the inclusive review threshold on a cost increase
REVIEW_GAP = MARGIN_REVIEW + MARGIN_TOLERANCE

type TurnKey = tuple[str, int]  # (seat, j): the seat's turn j
# (field, seat, turn): a draw, a variable discard or a starting hand (turn -1)
type Decision = tuple[str, str, int]
type StartingVariables = dict[str, list[cp_model.IntVar]]
type DrawVariables = dict[TurnKey, list[cp_model.IntVar]]


@dataclass(frozen=True, kw_only=True)
class HandRole:
    """Starting-hand size and whether the seat has an un-discarded winning draw."""

    dealer: bool
    wins_by_tsumo: bool = False


def posterior_to_tiles(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """39-class posterior -> 37-kind vector (X / none mass dropped, renormalised)."""
    q = np.array([p[CLASS_INDEX[t]] for t in TILES], np.float64)
    s = q.sum()
    return q / s if s > MIN_POSTERIOR_MASS else q


@dataclass
class SeatTurn:
    """One seat's turn; indices include calls, while times delimit pond sightings."""

    j: int  # turn index of this seat (0-based)
    kind: str  # first | draw | call | kan
    discard: str | None  # tile kind, None when the hand ended without a discard
    t_pre: float  # start of the disturbed window in which the turn happened
    t_discard: float  # when the discard was first seen
    # tiles that left the hand into a meld this turn
    removed: list[str] = field(default_factory=list)
    riichi: bool = False
    # 37-kind posterior of the discard: the solver may take the 2nd choice at a cost
    discard_p: np.ndarray | None = None
    kan: str | None = None  # ankan | kakan | daiminkan when this turn contains a kan
    kan_tile: str | None = None  # the kan's tile kind, None when the solver chooses it
    two_draws: bool = False  # ankan / kakan: a normal draw, the kan, the rinshan draw
    # [(tiles from the hand, cost)]: the call's tiles are a choice
    meld_options: list = field(default_factory=list)
    # a call took this discard: the meld camera read it too, so it is never re-read
    taken: bool = False
    # earliest possible draw: the preceding player's last pond view without their
    # discard
    t_draw_min: float | None = None


@dataclass
class HandEvidence:
    """Soft concealed-state evidence, with explicit partial-view semantics.

    ``e`` holds aggregate counts in ``TILES`` order. Partial views from
    ``hand_evidence`` also retain normalized per-box ``slots`` in that order,
    so alternative identities remain exclusive. Count-only callers can omit
    ``slots`` and retain the aggregate cost, but cannot express that exclusivity.
    """

    seat: str
    j: int  # state after turn j (j = -1: haipai state)
    after_draw: bool  # the 14-tile state after draw j+1
    e: np.ndarray  # expected counts per kind (37)
    weight: float
    t0: float
    t1: float
    # a row showing part of the hand: its tiles are in the hand, others may be hidden
    subset: bool = False
    # a calm row between turns short of the hand by this many: at most these beyond it
    hidden: int = 0
    # per-box alternatives; partial views must not count one box twice
    slots: list[np.ndarray] | None = None


@dataclass
class DrawEvidence:
    """Posterior over a single draw, usually from the unsorted end of a hand row."""

    seat: str
    j: int  # the draw of turn j
    p: np.ndarray  # 37
    weight: float


@dataclass
class Facts:
    """Human-confirmed constraints that cannot be traded for lower cost."""

    haipai: dict[str, list[str]] = field(default_factory=dict)
    draws: dict[TurnKey, str] = field(default_factory=dict)
    # seat -> concealed tiles at the end
    final: dict[str, list[str]] = field(default_factory=dict)
    # seat -> True when `final` excludes the winning draw (tsumo winner)
    final_excl: dict[str, bool] = field(default_factory=dict)


@dataclass
class Solution:
    """A reconstruction; check ``ok`` before reading tiles.

    ``HandModel.certify`` fills ``certificates``: objective separation, not visual
    accuracy or probability. ``optimal`` preserves the search status even when
    repair changes ``status``.
    """

    status: str
    objective: float
    haipai: dict[str, list[str]]
    draws: dict[TurnKey, str]
    # concealed tiles after each turn (j = -1: the starting hand)
    hands: dict[TurnKey, list[str]]
    # pond identities the reconstruction reads as another kind
    discards: dict[TurnKey, str] = field(default_factory=dict)
    # rinshan draws of self-kan turns
    draws2: dict[TurnKey, str] = field(default_factory=dict)
    # kan tiles no camera named, chosen by the solver
    kans: dict[TurnKey, str] = field(default_factory=dict)
    # the five (plain or red) a kakan of fives added
    kan_added: dict[TurnKey, str] = field(default_factory=dict)
    # the index of the meld composition chosen for each call turn
    melds: dict[TurnKey, int] = field(default_factory=dict)
    # draws the rules derive after riichi from the turn's discard or ankan
    draw_sources: dict[TurnKey, str] = field(default_factory=dict)
    optimal: bool | None = None
    certificates: dict[Decision, Certificate] = field(default_factory=dict)
    certified: bool = False
    # the program this solution was found in; certification searches it
    program: Program | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        """A reconstruction exists (possibly after repair)."""
        return self.status in ("optimal", "feasible", "repaired")


@dataclass
class Program:
    """One CP-SAT model of the hand and the variables that carry its decisions."""

    model: cp_model.CpModel = field(default_factory=cp_model.CpModel)
    starting: StartingVariables = field(default_factory=dict)
    draws: DrawVariables = field(default_factory=dict)
    rinshan: DrawVariables = field(default_factory=dict)
    kan_choices: DrawVariables = field(default_factory=dict)
    red_kan_choices: dict[TurnKey, cp_model.IntVar] = field(default_factory=dict)
    # pond identity per turn: a fixed identity maps its kind to the constant 1
    discards: dict[TurnKey, dict[int, cp_model.IntVar | int]] = field(
        default_factory=dict
    )
    hands: dict[TurnKey, list[cp_model.LinearExpr]] = field(default_factory=dict)
    meld_choices: DrawVariables = field(default_factory=dict)
    # objective terms: evidence costs, then pond and meld choice costs
    terms: list[cp_model.LinearExpr | int] = field(default_factory=list)
    discard_costs: list[cp_model.LinearExpr | int] = field(default_factory=list)


@dataclass(frozen=True, eq=False)
class _OpenDecision:
    """A decision certification has yet to separate from its alternatives."""

    key: Decision
    variables: list[cp_model.IntVar]
    chosen: list[int]  # the variables' values in the solution
    differs: cp_model.LiteralT  # only a reconstruction that changes it can be true
    tiles: list[str] | None = None  # each one-hot variable's tile; None for counts

    def changed_in(self, search: cp_model.CpSolver) -> bool:
        """Whether the search's reconstruction changes this decision."""
        return any(
            search.value(v) != c
            for v, c in zip(self.variables, self.chosen, strict=True)
        )

    def value_in(self, search: cp_model.CpSolver) -> str | None:
        """Return the decision's tile in the search's reconstruction."""
        if self.tiles is None:
            return None
        return next(
            t
            for t, v in zip(self.tiles, self.variables, strict=True)
            if search.value(v)
        )


class HandModel:
    """Mutable evidence and rules for one hand, rebuilt as CP-SAT on each solve.

    Add observations and normalized facts before solving. Counts distinguish
    red fives; meld removals and the dealer's starting fourteenth tile are
    represented explicitly rather than repaired during export.
    """

    def __init__(
        self,
        dealer: str,
        turns: dict[str, list[SeatTurn]],
        indicators: list[str],
        tsumo_winner: str | None = None,
        ura: list[str] | None = None,
    ) -> None:
        """Initialize hand rules and mutable observation evidence before solving."""
        self.dealer = dealer
        self.turns = turns
        self.indicators = indicators
        # revealed ura indicators count against the four copies too
        self.ura = list(ura or [])
        self.tsumo_winner = tsumo_winner
        self.hand_ev: list[HandEvidence] = []
        self.draw_ev: list[DrawEvidence] = []
        self.facts = Facts()
        # every discard a choice over all kinds (DESIGN.md 4.8 Repair)
        self.repair = False
        # (seat, j, tiles): hands the next-best solve must avoid
        self.forbidden_hands: list[tuple[str, int, list[str]]] = []
        # (seat, j, tiles): hands the site's score requires
        self.bound_hands: list[tuple[str, int, list[str]]] = []
        # the site's result as a constraint (4.8 "The result as a constraint")
        self.win: WinSpec | None = None
        # (seat, state j, sets needed) of the seats tenpai at a draw
        self.tenpai: list[tuple[str, int, int]] = []
        # diagnose turns them off to see whether they are what fails
        self.result_constraints = True

    # -- building ------------------------------------------------------------------
    def draw_turns(self, seat: str) -> list[int]:
        """Return draw and kan turn indices, including an undiscarded winning draw."""
        out = [t.j for t in self.turns[seat] if t.kind in ("draw", "kan")]
        if self.tsumo_winner == seat:
            out.append(len(self.turns[seat]))  # the winning draw, no discard after it
        return out

    def build(self) -> Program:
        """Build a fresh CP-SAT model of the current evidence, rules and facts."""
        program = Program()
        program.starting = {
            s: [program.model.new_int_var(0, 4, f"h0_{s}_{k}") for k in range(NT)]
            for s in rules.SEATS
        }
        self._add_draw_variables(program)
        self._add_discard_variables(program)
        self._add_hand_states(program)
        self._add_inventory_limits(program)
        self._add_reviewed_facts(program)
        self._add_exclusions(program)
        self._add_hand_costs(program)
        self._add_draw_costs(program)
        program.terms += program.discard_costs
        program.model.minimize(sum(program.terms) if program.terms else 0)
        return program

    def _add_draw_variables(self, program: Program) -> None:
        """Create each draw and kan choice, then constrain starting sizes."""
        m = program.model
        h0 = program.starting
        d = program.draws
        d2 = program.rinshan
        y = program.kan_choices
        for s in rules.SEATS:
            for j in self.draw_turns(s):
                d[(s, j)] = [m.new_bool_var(f"d_{s}_{j}_{k}") for k in range(NT)]
                m.add(sum(d[(s, j)]) == 1)
            for t in self.turns[s]:
                if t.two_draws:
                    d2[(s, t.j)] = [
                        m.new_bool_var(f"r_{s}_{t.j}_{k}") for k in range(NT)
                    ]
                    m.add(sum(d2[(s, t.j)]) == 1)
                if t.kan and t.kan_tile is None:
                    y[(s, t.j)] = [
                        m.new_bool_var(f"y_{s}_{t.j}_{k}") for k in range(NT)
                    ]
                    m.add(sum(y[(s, t.j)]) == 1)
        # haipai sizes
        for s in rules.SEATS:
            m.add(sum(h0[s]) == (14 if s == self.dealer else 13))

    def _add_discard_variables(self, program: Program) -> None:
        """Fix pond identities or add evidence-weighted discard alternatives."""
        m = program.model
        x = program.discards
        disc_cost = program.discard_costs
        for s in rules.SEATS:
            for t in self.turns[s]:
                if t.discard is None or t.discard not in TI:
                    continue
                if self.repair and t.discard_p is not None and not t.taken:
                    # repair: the pond reading is a cost, not a constraint
                    xs: dict[int, cp_model.IntVar | int] = {
                        k: m.new_bool_var(f"x_{s}_{t.j}_{k}") for k in range(NT)
                    }
                    m.add_exactly_one(xs.values())
                    x[(s, t.j)] = xs
                    for k, v in xs.items():
                        if TILES[k] != t.discard:
                            disc_cost.append(
                                round(
                                    SCALE
                                    * SCALE
                                    * REPAIR_COST
                                    * (1.0 - float(t.discard_p[k]))
                                )
                                * v
                            )
                    continue
                if t.discard_p is not None and not t.taken:
                    order = np.argsort(-t.discard_p)
                    k1, k2 = int(order[0]), int(order[1])
                    if (
                        t.discard_p[k2] >= MIN_ALTERNATIVE_DISCARD_PROBABILITY
                        and TILES[k1] == t.discard
                    ):
                        v1, v2 = (
                            m.new_bool_var(f"x_{s}_{t.j}_a"),
                            m.new_bool_var(f"x_{s}_{t.j}_b"),
                        )
                        m.add(v1 + v2 == 1)
                        x[(s, t.j)] = {k1: v1, k2: v2}
                        disc_cost.append(
                            round(SCALE * SCALE * 3 * (1.0 - float(t.discard_p[k2])))
                            * v2
                        )
                        continue
                x[(s, t.j)] = {TI[t.discard]: 1}

    def _add_hand_states(self, program: Program) -> None:
        """Name each concealed state after a turn and apply the authoritative result.

        A state variable keeps every later constraint short: the alternative is
        one expression of the starting hand and every acquisition so far,
        expanded again in each evidence term. The domain 0..4 is implied by the
        inventory limits.
        """
        m = program.model
        for s in rules.SEATS:
            expr: list[cp_model.LinearExpr] = list(program.starting[s])
            program.hands[(s, -1)] = list(expr)
            riichi_on = False
            for t in self.turns[s]:
                self._remove_called_tiles(program, s, t, expr)
                expr = self._take_draw(program, s, t, expr, riichi_on=riichi_on)
                expr = self._take_kan(program, s, t, expr, riichi_on=riichi_on)
                for k, v in program.discards.get((s, t.j), {}).items():
                    expr[k] = expr[k] - v
                state = [m.new_int_var(0, 4, f"h_{s}_{t.j}_{k}") for k in range(NT)]
                for k in range(NT):
                    m.add(state[k] == expr[k])
                expr = list(state)
                program.hands[(s, t.j)] = list(expr)
                riichi_on = riichi_on or t.riichi
            if self.tsumo_winner == s:
                jw = len(self.turns[s])
                expr = [expr[k] + program.draws[(s, jw)][k] for k in range(NT)]
                program.hands[(s, jw)] = list(expr)
        if self.result_constraints:
            self._result_constraints(m, program.hands, program.discards)

    def _remove_called_tiles(
        self,
        program: Program,
        s: str,
        t: SeatTurn,
        expr: list[cp_model.LinearExpr],
    ) -> None:
        """Require the chosen call composition to exist before its draw."""
        m = program.model
        mo = program.meld_choices
        disc_cost = program.discard_costs
        # tiles that went into a chi / pon / daiminkan left the hand before any
        # draw of this turn
        # (a daiminkan is followed by its rinshan draw): they must have been in
        # the hand already
        for r in t.removed:
            if r in TI:
                expr[TI[r]] = expr[TI[r]] - 1
        if t.meld_options:
            # the call's tiles from the hand: one legal composition, chosen with
            # the hand's evidence
            bs = [
                m.new_bool_var(f"m_{s}_{t.j}_{i}") for i in range(len(t.meld_options))
            ]
            m.add_exactly_one(bs)
            mo[(s, t.j)] = bs
            for b, (tiles, cost) in zip(bs, t.meld_options, strict=False):
                for r in tiles:
                    if r in TI:
                        expr[TI[r]] = expr[TI[r]] - b
                if cost > 0:
                    disc_cost.append(round(SCALE * SCALE * cost) * b)
        if t.removed or t.meld_options:
            for k in range(NT):
                m.add(expr[k] >= 0)

    def _take_draw(
        self,
        program: Program,
        s: str,
        t: SeatTurn,
        expr: list[cp_model.LinearExpr],
        *,
        riichi_on: bool,
    ) -> list[cp_model.LinearExpr]:
        """Add a draw and enforce the frozen hand after riichi."""
        m = program.model
        d = program.draws
        y = program.kan_choices
        xs = program.discards.get((s, t.j), {})
        if (s, t.j) in d:
            expr = [expr[k] + d[(s, t.j)][k] for k in range(NT)]
            if riichi_on and xs and not t.two_draws:
                # after riichi every draw is discarded at once
                for k in range(NT):
                    m.add(d[(s, t.j)][k] == xs.get(k, 0))
            elif riichi_on and t.two_draws:
                # after riichi a concealed kan is only of the tile just drawn
                # (the hand is frozen)
                if t.kan_tile in TI:
                    red = rules.RED_OF.get(t.kan_tile)
                    m.add(
                        d[(s, t.j)][TI[t.kan_tile]]
                        + (d[(s, t.j)][TI[red]] if red else 0)
                        == 1
                    )
                elif (s, t.j) in y:
                    for k in range(NT):
                        m.add(d[(s, t.j)][k] == y[(s, t.j)][k])
        return expr

    def _remove_kan_tiles(
        self,
        program: Program,
        s: str,
        t: SeatTurn,
        expr: list[cp_model.LinearExpr],
    ) -> None:
        """Account for known or chosen kan tiles, including the red five."""
        m = program.model
        y = program.kan_choices
        vr = program.red_kan_choices
        n_out = 4 if t.kan == "ankan" else 1
        red = rules.RED_OF.get(t.kan_tile or "")
        if t.kan_tile is not None and t.kan_tile in TI and red and t.kan == "ankan":
            # a concealed kan of fives is all four fives: three plain and
            # the red one
            expr[TI[t.kan_tile]] = expr[TI[t.kan_tile]] - 3
            expr[TI[red]] = expr[TI[red]] - 1
        elif t.kan_tile is not None and t.kan_tile in TI and red:
            # kakan of a five: the added tile is whichever five the player
            # still holds
            vr[(s, t.j)] = m.new_bool_var(f"kr_{s}_{t.j}")
            expr[TI[t.kan_tile]] = expr[TI[t.kan_tile]] - (1 - vr[(s, t.j)])
            expr[TI[red]] = expr[TI[red]] - vr[(s, t.j)]
        elif t.kan_tile is not None and t.kan_tile in TI:
            expr[TI[t.kan_tile]] = expr[TI[t.kan_tile]] - n_out
        elif (s, t.j) in y:
            for k in range(NT):
                if TILES[k] in ("5m", "5p", "5s", "0m", "0p", "0s"):
                    # a kan of fives is named by the meld camera, never guessed
                    m.add(y[(s, t.j)][k] == 0)
                expr[k] = expr[k] - n_out * y[(s, t.j)][k]

    def _take_kan(
        self,
        program: Program,
        s: str,
        t: SeatTurn,
        expr: list[cp_model.LinearExpr],
        *,
        riichi_on: bool,
    ) -> list[cp_model.LinearExpr]:
        """Require kan tiles before adding the replacement draw."""
        m = program.model
        d2 = program.rinshan
        xs = program.discards.get((s, t.j), {})
        if t.kan in ("ankan", "kakan"):
            self._remove_kan_tiles(program, s, t, expr)
            # the kan tiles must be in the hand before the rinshan draw
            for k in range(NT):
                m.add(expr[k] >= 0)
            if (s, t.j) in d2:
                expr = [expr[k] + d2[(s, t.j)][k] for k in range(NT)]
                if riichi_on and xs:  # after riichi the rinshan draw is discarded
                    for k in range(NT):
                        m.add(d2[(s, t.j)][k] == xs.get(k, 0))
        return expr

    def _add_inventory_limits(self, program: Program) -> None:
        """Count every starting tile, acquisition and revealed indicator together."""
        m = program.model
        h0 = program.starting
        d = program.draws
        d2 = program.rinshan
        ind = np.zeros(NT, int)
        for t in self.indicators + self.ura:
            if t in TI:
                ind[TI[t]] += 1
        for k in range(NT):
            total = (
                sum(h0[s][k] for s in rules.SEATS)
                + sum(v[k] for v in d.values())
                + sum(v[k] for v in d2.values())
                + int(ind[k])
            )
            m.add(total <= rules.max_count(TILES[k]))

    def _add_reviewed_facts(self, program: Program) -> None:
        """Constrain confirmed starting hands, draws and final hand states."""
        m = program.model
        h0 = program.starting
        d = program.draws
        hands = program.hands
        for s, tiles in self.facts.haipai.items():
            for k in range(NT):
                m.add(h0[s][k] == sum(1 for t in tiles if t == TILES[k]))
        for (s, j), tile in self.facts.draws.items():
            if (s, j) in d and tile in TI:
                m.add(d[(s, j)][TI[tile]] == 1)
        for s, tiles in self.facts.final.items():
            last = max((j for (ss, j) in hands if ss == s), default=None)
            if (
                last is not None
                and self.facts.final_excl.get(s)
                and self.tsumo_winner == s
            ):
                last -= 1  # the state after the last discard, before the winning draw
            if last is not None:
                for k in range(NT):
                    m.add(hands[(s, last)][k] == sum(1 for t in tiles if t == TILES[k]))

    def _add_exclusions(self, program: Program) -> None:
        """Forbid the hands a next-best search avoids; bind the site-required ones."""
        m = program.model
        hands = program.hands
        for n_, (s, j, tiles) in enumerate(self.forbidden_hands):
            if (s, j) in hands:
                _differs(m, hands[(s, j)], tiles, f"nb{n_}")
        for s, j, tiles in self.bound_hands:
            if (s, j) in hands:
                for k in range(NT):
                    m.add(hands[(s, j)][k] == sum(1 for t in tiles if t == TILES[k]))

    def _add_hand_costs(self, program: Program) -> None:
        """Penalize differences between visible rows and reconstructed hand states."""
        m = program.model
        d = program.draws
        hands = program.hands
        terms = program.terms
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
            w = round(ev.weight * SCALE)
            if w <= 0:
                continue
            if (ev.subset or ev.hidden) and ev.slots is not None:
                terms.append(
                    _partial_cost(
                        m, expr, ev.slots, w * (2 if ev.hidden else 1), len(terms)
                    )
                )
                continue
            self._add_row_deviations(program, ev, expr, w)

    def _add_row_deviations(
        self,
        program: Program,
        ev: HandEvidence,
        expr: list[cp_model.LinearExpr],
        w: int,
    ) -> None:
        """Penalize visible row differences and any excess beyond hidden tiles."""
        m = program.model
        terms = program.terms
        for k in range(NT):
            target = round(ev.e[k] * SCALE)
            if ev.subset and target <= 0:
                continue
            dev = m.new_int_var(0, 20 * SCALE, f"dev_{ev.seat}_{ev.j}_{k}_{len(terms)}")
            if not ev.subset and not ev.hidden:
                # a full row: no tile more than it shows ...
                m.add(dev >= expr[k] * SCALE - target)
            # ... and every tile it shows is in the hand
            m.add(dev >= target - expr[k] * SCALE)
            terms.append(w * dev)
        if ev.hidden:
            # all but `hidden` tiles shown: the hand holds at most that many tiles
            # beyond the row
            ups = []
            for k in range(NT):
                up = m.new_int_var(
                    0, 20 * SCALE, f"up_{ev.seat}_{ev.j}_{k}_{len(terms)}"
                )
                m.add(up >= expr[k] * SCALE - round(ev.e[k] * SCALE))
                ups.append(up)
            excess = m.new_int_var(
                0, 20 * NT * SCALE, f"ex_{ev.seat}_{ev.j}_{len(terms)}"
            )
            m.add(excess >= sum(ups) - ev.hidden * SCALE)
            terms.append(w * excess)

    def _add_draw_costs(self, program: Program) -> None:
        """Add direct draw likelihood costs after the hand evidence terms."""
        d = program.draws
        terms = program.terms
        for ev in self.draw_ev:
            key = (ev.seat, ev.j)
            if key not in d:
                continue
            w = round(ev.weight * SCALE)
            for k in range(NT):
                c = round((1.0 - ev.p[k]) * SCALE)
                if c > 0:
                    terms.append(w * c * d[key][k])

    def _result_constraints(self, m: cp_model.CpModel, hands: dict, x: dict) -> None:
        """Require a winning final hand, or the tenpai states of a recorded draw."""
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
            wait = [m.new_bool_var(f"wait_{seat}_{k}") for k in range(34)]
            m.add_exactly_one(wait)
            c34 = _counts34(hands[(seat, j)])
            _complete(m, [c34[k] + wait[k] for k in range(34)], sets, f"tenpai_{seat}")

    def _discard_of(self, key: TurnKey) -> str | None:
        s, j = key
        for t in self.turns[s]:
            if t.j == j:
                return t.discard
        return None

    def _riichi_draw_sources(self) -> dict[TurnKey, str]:
        """Identify rule-linked draws without a counterfactual search.

        The declaration turn is not locked. A winning draw has no discard.
        An ankan consumes the first draw, not the subsequent rinshan draw;
        a kan of fives alone cannot distinguish a red draw from a plain five.
        """
        sources = {}
        for seat, turns in self.turns.items():
            locked = False
            for turn in turns:
                if locked and turn.kind in ("draw", "kan"):
                    if not turn.two_draws and turn.discard in TI:
                        sources[(seat, turn.j)] = "discard"
                    elif (
                        turn.two_draws
                        and turn.kan == "ankan"
                        and turn.kan_tile in TI
                        and rules.plain(turn.kan_tile) not in ("5m", "5p", "5s")
                    ):
                        sources[(seat, turn.j)] = "ankan"
                locked = locked or turn.riichi
        return sources

    # -- solving -------------------------------------------------------------------
    def solve(
        self,
        *,
        time_limit: float = 60.0,
        workers: int = 8,
        prior: Solution | None = None,
    ) -> Solution:
        """Find the cheapest legal reconstruction of the current evidence and facts.

        ``prior`` seeds the search with soft hints, never constraints, so new
        evidence or facts can replace old choices. A search that ends without a
        candidate is ``unsolved``; only a proof makes it ``infeasible``. The
        solution is uncertified until ``certify`` runs.
        """
        program = self.build()
        if prior is not None and prior.ok:
            self._add_prior_hints(program, prior)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = workers
        status = solver.solve(program.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return Solution(
                "infeasible" if status == cp_model.INFEASIBLE else "unsolved",
                math.inf,
                {},
                {},
                {},
            )
        sol = self._read_solution(solver, status, program)
        self._read_call_and_discard_choices(solver, program, sol)
        return sol

    def _add_prior_hints(self, program: Program, prior: Solution) -> None:
        """Seed a useful incumbent without constraining updated evidence."""
        m = program.model
        for seat, variables in program.starting.items():
            if seat in prior.haipai:
                counts = Counter(prior.haipai[seat])
                for tile, variable in zip(TILES, variables, strict=True):
                    m.add_hint(variable, counts[tile])
        for variables, choices in (
            (program.draws, prior.draws),
            (program.rinshan, prior.draws2),
            (program.kan_choices, prior.kans),
        ):
            for key, vs in variables.items():
                if choices.get(key) in TI:
                    for tile, variable in zip(TILES, vs, strict=True):
                        m.add_hint(variable, int(tile == choices[key]))
        self._add_prior_call_hints(program, prior)
        self._add_prior_discard_hints(program, prior)

    @staticmethod
    def _add_prior_call_hints(program: Program, prior: Solution) -> None:
        """Seed red-five kan and meld composition choices."""
        for key, variable in program.red_kan_choices.items():
            if key in prior.kan_added:
                program.model.add_hint(
                    variable, int(prior.kan_added[key] in rules.PLAIN_OF)
                )
        for key, variables in program.meld_choices.items():
            if key in prior.melds:
                for index, variable in enumerate(variables):
                    program.model.add_hint(variable, int(index == prior.melds[key]))

    def _add_prior_discard_hints(self, program: Program, prior: Solution) -> None:
        """Seed variable discards while leaving fixed pond identities untouched."""
        for key, variables in program.discards.items():
            tile = prior.discards.get(key, self._discard_of(key))
            if tile in TI:
                for index, variable in variables.items():
                    if not isinstance(variable, int):  # fixed: no decision variable
                        program.model.add_hint(variable, int(index == TI[tile]))

    def _read_solution(
        self,
        solver: cp_model.CpSolver,
        status: cp_model.CpSolverStatus,
        program: Program,
    ) -> Solution:
        """Read primary hands and acquisitions from a feasible assignment."""
        draws = {
            key: TILES[int(np.argmax([solver.value(v) for v in vs]))]
            for key, vs in program.draws.items()
        }
        return Solution(
            "optimal" if status == cp_model.OPTIMAL else "feasible",
            solver.objective_value / SCALE / SCALE,
            {
                s: [
                    TILES[k]
                    for k in range(NT)
                    for _ in range(solver.value(program.starting[s][k]))
                ]
                for s in rules.SEATS
            },
            draws,
            {
                key: [TILES[k] for k in range(NT) for _ in range(solver.value(expr[k]))]
                for key, expr in program.hands.items()
            },
            optimal=status == cp_model.OPTIMAL,
            draw_sources={
                key: source
                for key, source in self._riichi_draw_sources().items()
                if key in draws
            },
            program=program,
        )

    def _read_call_and_discard_choices(
        self, solver: cp_model.CpSolver, program: Program, sol: Solution
    ) -> None:
        """Materialize kan, meld and repaired pond choices."""
        for key, vs in program.rinshan.items():
            sol.draws2[key] = TILES[int(np.argmax([solver.value(v) for v in vs]))]
        for key, vs in program.kan_choices.items():
            sol.kans[key] = TILES[int(np.argmax([solver.value(v) for v in vs]))]
        for key, v in program.red_kan_choices.items():
            tile = next(t.kan_tile for t in self.turns[key[0]] if t.j == key[1])
            if tile is None:
                raise RuntimeError("A red-five kan choice must have a known tile kind")
            sol.kan_added[key] = rules.RED_OF[tile] if solver.value(v) else tile
        for key, bs in program.meld_choices.items():
            sol.melds[key] = next(i for i, b in enumerate(bs) if solver.value(b))
        for key, xs in program.discards.items():
            if len(xs) < MIN_DECISION_CHOICES:
                continue
            for k, v in xs.items():
                if solver.value(v) == 1 and TILES[k] != self._discard_of(key):
                    sol.discards[key] = TILES[k]

    # -- certification -------------------------------------------------------------
    def certify(
        self, sol: Solution, *, timeout: float = 60.0, workers: int = 8
    ) -> None:
        """Bound the cost increase of changing each draw, discard and starting hand.

        Each search asks for the cheapest reconstruction that changes at least one
        pending decision. A proven lower bound above the review threshold
        certifies all of them at once: the group's minimum is at most each
        decision's own. A reconstruction within the threshold is a close witness
        for every decision it changes; those are ambiguous, and the search repeats
        without them. Decisions still pending after ``timeout`` seconds are
        unresolvable with the last proven bound. Human facts are fixed, and a
        riichi-locked draw shares its discard's certificate. The searches run in
        the program ``sol`` was found in; the facts and pond readings must not
        have changed since.
        """
        program = sol.program
        if program is None:
            raise ValueError(
                "Certification needs the program the solution was found in"
            )
        base = replace(program, model=program.model.clone())
        base.model.clear_hints()
        self._add_prior_hints(base, sol)
        pending = self._open_decisions(sol, base)
        deadline = time.monotonic() + timeout
        margin = 0.0
        while pending and (seconds := deadline - time.monotonic()) > 0:
            trial = base.model.clone()
            trial.add_bool_or([d.differs for d in pending])
            search, status, margin = _closest_change(
                trial, sol.objective, seconds, workers
            )
            if margin > REVIEW_GAP:
                gap = math.inf if status == cp_model.INFEASIBLE else None
                sol.certificates.update(
                    {d.key: Certificate(margin, gap) for d in pending}
                )
                pending = []
                break
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break  # out of time before a proof or a candidate
            gap = max(0.0, search.objective_value / SCALE / SCALE - sol.objective)
            if gap > REVIEW_GAP:
                break  # out of time before a proof or a close witness
            unchanged = []
            for d in pending:
                if d.changed_in(search):
                    sol.certificates[d.key] = Certificate(
                        margin, gap, d.value_in(search)
                    )
                else:
                    unchanged.append(d)
            pending = unchanged
        sol.certificates.update({d.key: Certificate(margin) for d in pending})
        for (seat, j), source in sol.draw_sources.items():
            # Excluding a locked draw is excluding its discard: one certificate.
            sol.certificates["draw", seat, j] = (
                sol.certificates.get(("discard", seat, j), FIXED)
                if source == "discard"
                else FIXED
            )
        sol.certified = True

    def _open_decisions(self, sol: Solution, program: Program) -> list[_OpenDecision]:
        """Fix human-confirmed decisions; list the rest with what changes them."""
        pending = []
        for (seat, j), tile in sol.draws.items():
            if (seat, j) in sol.draw_sources:
                continue  # the rules derive it from its source
            if (seat, j) in self.facts.draws:
                sol.certificates["draw", seat, j] = FIXED
                continue
            variables = program.draws[seat, j]
            pending.append(
                _OpenDecision(
                    ("draw", seat, j),
                    variables,
                    [int(t == tile) for t in TILES],
                    variables[TI[tile]].Not(),
                    TILES,
                )
            )
        for (seat, j), options in program.discards.items():
            choices = {
                TILES[k]: v for k, v in options.items() if not isinstance(v, int)
            }
            chosen = sol.discards.get((seat, j), self._discard_of((seat, j)))
            if chosen is None or chosen not in choices:
                continue  # a fixed pond identity is no decision
            pending.append(
                _OpenDecision(
                    ("discard", seat, j),
                    list(choices.values()),
                    [int(t == chosen) for t in choices],
                    choices[chosen].Not(),
                    list(choices),
                )
            )
        for seat in rules.SEATS:
            if seat in self.facts.haipai:
                sol.certificates["haipai", seat, -1] = FIXED
                continue
            differs = program.model.new_bool_var(f"differs_h0_{seat}")
            starting = program.starting[seat]
            _differs(
                program.model, starting, sol.haipai[seat], f"h0_{seat}", when=differs
            )
            counts = Counter(sol.haipai[seat])
            pending.append(
                _OpenDecision(
                    ("haipai", seat, -1), starting, [counts[t] for t in TILES], differs
                )
            )
        return pending


def _closest_change(
    model: cp_model.CpModel, objective: float, seconds: float, workers: int
) -> tuple[cp_model.CpSolver, cp_model.CpSolverStatus, float]:
    """Search until the cost increase over ``objective`` is decided for review.

    Returns the search, its status and a certified lower bound on the increase.
    The search stops at a bound above the review threshold or at a reconstruction
    within it: the exact optimum of either is never needed. A candidate's cost
    never certifies anything. Model costs are nonnegative, so a zero bound is
    conservative; an infeasible model has an infinite bound.
    """
    search = cp_model.CpSolver()
    search.parameters.max_time_in_seconds = seconds
    search.parameters.num_workers = workers
    proven = 0.0

    def increase(value: float) -> float:
        return value / SCALE / SCALE - objective

    def on_bound(bound: float) -> None:
        nonlocal proven
        if math.isfinite(bound):
            proven = max(proven, increase(bound))
            if proven > REVIEW_GAP:
                search.stop_search()

    class CloseWitness(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            if increase(self.objective_value) <= REVIEW_GAP:
                self.stop_search()

    search.best_bound_callback = on_bound
    status = search.solve(model, CloseWitness())
    if status == cp_model.MODEL_INVALID:
        raise RuntimeError(f"Invalid certification model: {search.response_stats()}")
    if status == cp_model.INFEASIBLE:
        return search, status, math.inf
    if math.isfinite(search.best_objective_bound):
        proven = max(proven, increase(search.best_objective_bound))
    return search, status, proven


def _differs(
    m: cp_model.CpModel,
    expr: list,
    tiles: list[str],
    name: str,
    *,
    when: cp_model.LiteralT | None = None,
) -> None:
    """Constrain tile counts to differ from a hand of the same size.

    The counts `expr` hold at least one kind fewer than `tiles` (the sizes are
    fixed, so the hand differs), unconditionally or only where ``when`` holds.
    """
    fewer = []
    for k in range(NT):
        v = sum(1 for t in tiles if t == TILES[k])
        if v:
            b = m.new_bool_var(f"fewer_{name}_{k}")
            m.add(expr[k] <= v - 1).only_enforce_if(b)
            fewer.append(b)
    clause = m.add_bool_or(fewer)
    if when is not None:
        clause.only_enforce_if(when)


@dataclass
class WinSpec:
    """Where the winner's complete hand is in the model.

    The state after its turn j (a tsumo: after the winning draw), plus the loser's
    discard of turn j_loser for a ron; `sets` the sets the concealed part must hold.
    """

    seat: str
    j: int
    sets: int
    ron_from: tuple[str, int] | None = None


TERMINALS = [k for k in range(34) if k >= FIRST_HONOR_INDEX or k % 9 in (0, 8)]


def _counts34(expr: list) -> list:
    """37-kind counts -> 34 kinds (a red five is a five)."""
    out = [expr[TI[t]] for t in rules.KINDS]
    for red, plain in rules.PLAIN_OF.items():
        k = rules.KINDS.index(plain)
        out[k] = out[k] + expr[TI[red]]
    return out


def _complete(m: cp_model.CpModel, c34: list, sets: int, name: str) -> None:
    """Constrain 34-kind counts to a complete hand.

    `sets` sets besides the melds and one pair, or, closed, seven pairs or the
    thirteen orphans. Integer decomposition: pons per kind, chis per start, the pair.
    """
    std = m.new_bool_var(f"{name}_std")
    forms = [std]
    pon = [m.new_bool_var(f"{name}_pon{k}") for k in range(34)]
    pair = [m.new_bool_var(f"{name}_pair{k}") for k in range(34)]
    chi = {
        k: m.new_int_var(0, 4, f"{name}_chi{k}")
        for k in range(27)
        if k % 9 <= LAST_CHI_START_INDEX
    }
    for k in range(34):
        covering = [
            chi[k - d] for d in (0, 1, 2) if (k - d) in chi and (k - d) // 9 == k // 9
        ]
        m.add(c34[k] == 3 * pon[k] + 2 * pair[k] + sum(covering)).only_enforce_if(std)
    m.add(sum(pair) == 1).only_enforce_if(std)
    m.add(sum(pon) + sum(chi.values()) == sets).only_enforce_if(std)
    if sets == SETS_IN_CLOSED_HAND:
        seven = m.new_bool_var(f"{name}_7p")
        pp = [m.new_bool_var(f"{name}_pp{k}") for k in range(34)]
        for k in range(34):
            m.add(c34[k] == 2 * pp[k]).only_enforce_if(seven)
        m.add(sum(pp) == PAIRS_IN_SEVEN_PAIRS).only_enforce_if(seven)
        orphans = m.new_bool_var(f"{name}_13o")
        for k in range(34):
            if k in TERMINALS:
                m.add(c34[k] >= 1).only_enforce_if(orphans)
            else:
                m.add(c34[k] == 0).only_enforce_if(orphans)
        forms += [seven, orphans]
    m.add_exactly_one(forms)


# -----------------------------------------------------------------------------
# evidence mapping
# -----------------------------------------------------------------------------


def state_of(seat_turns: list[SeatTurn], t0: float, t1: float) -> int | None:
    """Return the state j a calm interval [t0, t1] of the hand band belongs to.

    -1 is the haipai state; None when the interval straddles a turn of this seat.
    """
    j = -1
    for t in seat_turns:
        if t.t_discard <= t0 + 0.6:
            j = t.j
        elif t.t_discard - 0.6 <= t1:
            return None  # the discard happened inside the interval
    return j


def open_turn(seat_turns: list[SeatTurn], t1: float) -> SeatTurn | None:
    """Find the turn whose window holds an interval's end t1 before its discard.

    Its window [t_pre, t_discard] contains t1 without reaching the discard's first
    sighting: the discard may already have happened inside the interval, so a row
    with the resting count there is ambiguous (before the draw or after the
    discard); only the row with one tile more is a state, the one after the draw.
    """
    for t in seat_turns:
        if t.t_discard - 0.6 > t1:
            # the pond may first show a discard several seconds after it was made (an
            # arm over the pond), but not much longer: a row at rest that ends more
            # than 8 s before the sighting was still waiting to draw
            return t if t.t_pre <= t1 and t1 >= t.t_discard - 8.0 else None
    return None


def _partial_cost(
    model: cp_model.CpModel,
    counts: list,
    slots: list[np.ndarray],
    weight: int,
    tag: int,
) -> cp_model.LinearExpr | int:
    """Match each visible box to at most one concealed tile, allowing misreads.

    Summed posterior counts let a hidden extra tile satisfy both alternatives
    of one ambiguous box. Capacity-constrained matching keeps those alternatives
    exclusive. Unmatched boxes cost one; hidden tiles contribute no evidence.
    Equal posteriors share integer flows, and zero-rounded rewards need no
    variables because leaving a box unmatched has the same cost.
    """
    groups = Counter(tuple(round(float(p) * SCALE) for p in slot) for slot in slots)
    used = [[] for _ in range(NT)]
    rewards = []
    for i, (posterior, n) in enumerate(groups.items()):
        choices = []
        for k, reward in enumerate(posterior):
            if reward <= 0:
                continue
            choice = model.new_int_var(0, n, f"partial_{tag}_{i}_{k}")
            choices.append(choice)
            used[k].append(choice)
            rewards.append(reward * choice)
        model.add(sum(choices) <= n)
    for k, choices in enumerate(used):
        if choices:
            model.add(sum(choices) <= counts[k])
    return weight * (len(slots) * SCALE - sum(rewards))


def expected_counts(slots: list[dict]) -> np.ndarray:
    """Sum normalized tile posteriors, excluding face-down and empty classes."""
    e = np.zeros(NT)
    for s in slots:
        e += posterior_to_tiles(s["p"])
    return e


def _tops(slots: list[dict]) -> list[str]:
    return [rules.plain(CLASSES[int(np.argmax(sl["p"]))]) for sl in slots]


def drawn_end(
    slots: list[dict], ref: list | None, prior: tuple[int, int] | None
) -> list[tuple[str, float]]:
    """Weigh which end of a row one tile over the resting count holds the draw.

    Returns [(end, weight)].

    The end whose removal leaves the previous resting row `ref` (posteriors), compared
    as multisets (the player may not have sorted), names it, strongly — if the rest
    differs from `ref` by at most MATCH_SLACK tiles; a reference further off is of
    another state and says nothing. Undecided, both ends are candidates weighted by the
    ends this player has used (`prior` = (left, right) counts): players are consistent.
    """
    if ref is not None:
        want = Counter(rules.plain(CLASSES[int(np.argmax(p))]) for p in ref)
        tops = _tops(slots)
        diff = {
            "L": sum((want - Counter(tops[1:])).values()),
            "R": sum((want - Counter(tops[:-1])).values()),
        }
        best = min(diff, key=diff.__getitem__)
        other = "R" if best == "L" else "L"
        if diff[best] <= MATCH_SLACK and diff[best] < diff[other]:
            return [(best, 2.0)]
    n_l, n_r = prior or (0, 0)
    p_l = (n_l + 1) / (n_l + n_r + 2)
    return [("L", 1.2 * p_l), ("R", 1.2 * (1 - p_l))]


def _draw_evidence(
    seat: str, rows14: list[tuple[int, list[dict], list | None, float]]
) -> tuple[list[DrawEvidence], tuple[int, int]]:
    """Learn the drawn-end habit from decided rows before weighting ambiguous ends."""
    # the drawn end: decided rows first, to learn the player's habit, which then weighs
    # the undecided ones
    decided = [drawn_end(sl, ref, None) for _, sl, ref, _ in rows14]
    habit = (
        sum(1 for d in decided if len(d) == 1 and d[0][0] == "L"),
        sum(1 for d in decided if len(d) == 1 and d[0][0] == "R"),
    )
    dev: list[DrawEvidence] = []
    for (j, slots, ref, w), first in zip(rows14, decided, strict=False):
        for end, weight in first if len(first) == 1 else drawn_end(slots, ref, habit):
            dev.append(
                DrawEvidence(
                    seat,
                    j + 1,
                    posterior_to_tiles(slots[0 if end == "L" else -1]["p"]),
                    weight * w,
                )
            )
    return dev, habit


def hand_evidence(
    seat: str,
    seat_turns: list[SeatTurn],
    observations: list[dict],
    melds_before: dict[int, int],
    *,
    role: HandRole,
) -> tuple[list[HandEvidence], list[DrawEvidence], tuple[int, int]]:
    """Map a seat's observed hand rows to solver states (DESIGN.md 4.8, evidence 1-3).

    Returns the hand evidence, the direct draw evidence, and the seat's end habit
    (draws seen at the left end, at the right end).

    - a row with one tile more than the resting count, inside the turn: the state
      after the draw, and the drawn tile at an end of the row;
    - a row with the resting count between turns, calm or read-floor: the state (it
      shows every tile);
    - a calm row between turns one or two tiles short of it: every tile it shows is
      in the state, which holds at most that many more (a hand resting over an end of
      the row), at full weight;
    - anything else (a partial view, fewer tiles, a row inside a turn's window before
      the draw or already after the discard, a view across a discard) shows part of
      the hand at some moment of the turn, and every such moment's tiles are in the
      hand after the turn's draw: the tiles seen are a sub-multiset of it.
    """
    dealer, wins_by_tsumo = (role.dealer, role.wins_by_tsumo)
    hev: list[HandEvidence] = []
    rows14: list[tuple[int, list[dict], list | None, float]] = []
    prev13: dict[int, list] = {}
    kind_of = {t.j: t.kind for t in seat_turns}
    turn_of = {t.j: t for t in seat_turns}
    if wins_by_tsumo:
        kind_of[len(seat_turns)] = "draw"  # the winning draw: no discard after it
    for o in sorted(observations, key=lambda o: o["t0"]):
        if o["n_used"] == 0 or not o["slots"]:
            continue
        w = float(o["quality"]) * min(1.0, o["n_used"] / 3.0)
        j = state_of(seat_turns, o["t0"], o["t1"])
        straddle = j is None
        if straddle:
            j = max(
                (t.j for t in seat_turns if t.t_discard <= o["t0"] + 0.6), default=-1
            )
        base = 14 if (j == -1 and dealer) else 13 - 3 * melds_before.get(j, 0)
        count = o["count"]
        e = expected_counts(o["slots"])
        slot_p = [posterior_to_tiles(sl["p"]) for sl in o["slots"]]
        has_draw = kind_of.get(j + 1) == "draw"
        nxt = turn_of.get(j + 1)
        before_draw = (
            nxt is not None
            and nxt.t_draw_min is not None
            and o["t1"] < nxt.t_draw_min - 0.6
        )
        open_ = straddle or open_turn(seat_turns, o["t1"]) is not None
        if before_draw and has_draw and not straddle and count == base + 1:
            # The preceding player has not discarded yet, so this cannot be a
            # draw. Extra detections must not turn persistent hand tiles into
            # evidence for the next draw. Keep their soft lower-bound evidence
            # on the resting state; the rules expose any impossible extra tiles.
            hev.append(
                HandEvidence(
                    seat,
                    j,
                    after_draw=False,
                    e=e,
                    weight=0.5 * w,
                    t0=o["t0"],
                    t1=o["t1"],
                    subset=True,
                    slots=slot_p,
                )
            )
        elif count == base + 1 and has_draw and not straddle:
            hev.append(
                HandEvidence(
                    seat, j, after_draw=True, e=e, weight=w, t0=o["t0"], t1=o["t1"]
                )
            )
            rows14.append((j, o["slots"], prev13.get(j), w))
        elif count == base and not open_:
            # between turns the hand holds exactly the resting count: a row showing that
            # many hides nothing, read-floor
            # (partial) or calm alike
            hev.append(
                HandEvidence(
                    seat, j, after_draw=False, e=e, weight=w, t0=o["t0"], t1=o["t1"]
                )
            )
            prev13[j] = [sl["p"] for sl in o["slots"]]
        elif base - 2 <= count < base and not open_ and not o.get("partial"):
            hev.append(
                HandEvidence(
                    seat,
                    j,
                    after_draw=False,
                    e=e,
                    weight=w,
                    t0=o["t0"],
                    t1=o["t1"],
                    hidden=base - count,
                    slots=slot_p,
                )
            )
        elif count <= base + 1 and kind_of.get(j + 1) in (
            "draw",
            "call",
            "first",
            None,
        ):
            # part of the hand: a sub-multiset of the state after this turn's draw (of
            # the state itself when the
            # turn has no draw: a call, the dealer's first turn, or after the seat's
            # last turn)
            hev.append(
                HandEvidence(
                    seat,
                    j,
                    after_draw=has_draw,
                    e=e,
                    weight=0.5 * w,
                    t0=o["t0"],
                    t1=o["t1"],
                    subset=True,
                    slots=slot_p,
                )
            )
    dev, habit = _draw_evidence(seat, rows14)
    return hev, dev, habit


@dataclass(frozen=True)
class Culprit:
    """One decision without which an infeasible hand becomes legal.

    A conflict question asks about it; ``text`` is the developer's diagnosis.
    """

    kind: str  # fact | result | riichi | meld
    seat: str | None
    j: int | None  # the seat's turn (meld, riichi, a draw fact)
    text: str
    fact: str | None = None  # fact: haipai | draw | final
    t: float | None = None  # the discard time of the turn (meld, riichi)


def diagnose(model: HandModel, time_limit: float = 5.0) -> list[Culprit]:
    """Find the single decisions whose removal makes an infeasible model legal.

    The most specific first: a reviewer's fact that contradicts the video, the result
    constraint (the winner's hand as reconstructed cannot win), a riichi turn, one
    call. Empty when no single decision does.
    """
    return [
        *_fact_conflicts(model, time_limit),
        *_result_conflicts(model, time_limit),
        *_riichi_conflicts(model, time_limit),
        *_meld_conflicts(model, time_limit),
    ]


def _feasible(model: HandModel, time_limit: float) -> bool:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    return solver.solve(model.build().model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def _fact_conflicts(model: HandModel, time_limit: float) -> Iterator[Culprit]:
    full = model.facts
    # (kind, seat, turn, the fact as shown, every fact but that one)
    singles: list[tuple[str, str, int | None, str, Facts]] = [
        *(
            (
                "haipai",
                seat,
                None,
                " ".join(tiles),
                replace(full, haipai=_without(full.haipai, seat)),
            )
            for seat, tiles in full.haipai.items()
        ),
        *(
            (
                "draw",
                key[0],
                key[1],
                tile,
                replace(full, draws=_without(full.draws, key)),
            )
            for key, tile in full.draws.items()
        ),
        *(
            (
                "final",
                seat,
                None,
                " ".join(tiles),
                replace(full, final=_without(full.final, seat)),
            )
            for seat, tiles in full.final.items()
        ),
    ]
    for kind, seat, j, shown, facts in singles:
        trial = copy(model)
        trial.facts = facts
        if _feasible(trial, time_limit):
            yield Culprit(
                "fact",
                seat,
                j,
                f"your {kind} fact for {seat} ({shown}) contradicts "
                "what the video shows",
                fact=kind,
            )


def _without[K, V](facts: dict[K, V], key: K) -> dict[K, V]:
    return {k: v for k, v in facts.items() if k != key}


def _result_conflicts(model: HandModel, time_limit: float) -> Iterator[Culprit]:
    if model.result_constraints and (model.win is not None or model.tenpai):
        trial = copy(model)
        trial.result_constraints = False
        if _feasible(trial, time_limit):
            yield Culprit(
                "result",
                model.win.seat if model.win is not None else None,
                None,
                "the winner's hand as reconstructed cannot win "
                "(or a tenpai seat cannot be tenpai): a call or a discard "
                "of that seat is read wrong",
            )


def _replace_trial_turn(model: HandModel, seat: str, turn: SeatTurn) -> HandModel:
    trial = copy(model)
    trial.turns = {
        **model.turns,
        seat: [
            turn if current.j == turn.j else current for current in model.turns[seat]
        ],
    }
    return trial


def _riichi_conflicts(model: HandModel, time_limit: float) -> Iterator[Culprit]:
    for seat in rules.SEATS:
        for turn in model.turns[seat]:
            if not turn.riichi:
                continue
            trial = _replace_trial_turn(model, seat, replace(turn, riichi=False))
            if _feasible(trial, time_limit):
                yield Culprit(
                    "riichi",
                    seat,
                    turn.j,
                    f"{seat}'s riichi with the discard at {turn.t_discard:.0f}s "
                    "cannot be: after it the hand would have to change",
                    t=turn.t_discard,
                )


def _meld_conflicts(model: HandModel, time_limit: float) -> Iterator[Culprit]:
    for seat in rules.SEATS:
        for turn in model.turns[seat]:
            if not (turn.removed or turn.meld_options):
                continue
            trial = _replace_trial_turn(
                model, seat, replace(turn, removed=[], meld_options=[])
            )
            if _feasible(trial, time_limit):
                shown = (
                    " ".join(turn.removed)
                    if turn.removed
                    else " or ".join("".join(tiles) for tiles, _ in turn.meld_options)
                )
                yield Culprit(
                    "meld",
                    seat,
                    turn.j,
                    f"{seat}'s call before its discard at {turn.t_discard:.0f}s "
                    f"cannot be: its hand never holds {shown}",
                    t=turn.t_discard,
                )
