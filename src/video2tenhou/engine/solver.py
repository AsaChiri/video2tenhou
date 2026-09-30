# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

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
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from ortools.sat.python import cp_model

from video2tenhou.train.data import CLASS_INDEX, CLASSES

from . import rules
from .confidence import MARGIN_REVIEW, MARGIN_TOLERANCE

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
MARGIN_THREADS = (
    4  # margin re-solves at once (CP-SAT releases the GIL; each keeps its own workers)
)

type StartingVariables = dict[str, list[cp_model.IntVar]]
type DrawVariables = dict[tuple[str, int], list[cp_model.IntVar]]


@dataclass(frozen=True, kw_only=True)
class HandRole:
    """Starting-hand size and whether the seat has an un-discarded winning draw."""

    dealer: bool
    wins_by_tsumo: bool = False


class ReconstructionModel(cp_model.CpModel):
    """Constraint model retaining the insertion point for draw alternatives."""

    draw_forbid_index: int = 0


type Baseline = tuple[ReconstructionModel, StartingVariables, DrawVariables]


@dataclass(frozen=True, kw_only=True)
class ResolveOptions:
    """Decision exclusion, watched evidence and stopping policy for one search."""

    watch: tuple[str, int] | None = None
    baseline: Baseline | None = None
    forbid_variable: cp_model.IntVar | int | None = None
    watch_variables: dict[int, cp_model.IntVar | int] | None = None
    stop_when_certified: bool = True
    deadline: float | None = None
    forbid: tuple[str, int, str] | None = None
    forbid_haipai: tuple[str, list[str]] | None = None


@dataclass(frozen=True)
class SearchProof:
    """Reference objective and the certified lower bound in CP-SAT units."""

    reference_cost: float
    lower_bound: float


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
    removed: list[str] = field(
        default_factory=list
    )  # tiles that left the hand into a meld this turn
    riichi: bool = False
    discard_p: np.ndarray | None = (
        # 37-kind posterior of the discard: the solver may take the 2nd choice at a cost
        None
    )
    kan: str | None = None  # ankan | kakan | daiminkan when this turn contains a kan
    kan_tile: str | None = (
        None  # the kan's tile kind, None when unknown (the solver chooses it)
    )
    two_draws: bool = (
        False  # ankan / kakan: a normal draw, the kan, then the rinshan draw
    )
    meld_options: list = field(
        default_factory=list
    )  # [(tiles from the hand, cost)]: the call's tiles are a choice
    # a call took this discard: the meld camera read it too, so it is never re-read
    taken: bool = False
    t_draw_min: float | None = (
        # earliest possible draw: preceding player's last pond view without their
        # discard
        None
    )


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
    slots: list[np.ndarray] | None = (
        None  # per-box alternatives; partial views must not count one box twice
    )


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

    haipai: dict = field(default_factory=dict)  # seat -> list of tiles
    draws: dict = field(default_factory=dict)  # (seat, j) -> tile
    final: dict = field(
        default_factory=dict
    )  # seat -> list of tiles (concealed at the end)
    final_excl: dict = field(
        default_factory=dict
    )  # seat -> True when `final` excludes the winning draw (tsumo winner)


@dataclass
class Solution:
    """A reconstruction and alternatives; check ``ok`` before reading tiles.

    Margins certify objective separation, not visual accuracy or probability. Candidate
    ``alternative_gaps`` establish ambiguity when close enough, but never certify a
    choice when far away. None means no candidate was found. ``optimal`` preserves
    search status even when repair changes ``status``.
    """

    status: str
    objective: float
    haipai: dict  # seat -> list of tiles
    draws: dict  # (seat, j) -> tile or None
    hands: dict  # (seat, j) -> list of tiles after turn j
    margins: dict = field(
        default_factory=dict
    )  # (seat, j) -> certified lower bound on the excluded-draw cost gap
    discards: dict = field(
        default_factory=dict
    )  # (seat, j) -> tile chosen when the pond identity was a choice
    draws2: dict = field(
        default_factory=dict
    )  # (seat, j) -> rinshan draw of a self-kan turn
    kans: dict = field(
        default_factory=dict
    )  # (seat, j) -> kan tile chosen when it was unknown
    kan_added: dict = field(
        default_factory=dict
    )  # (seat, j) -> the five (plain or red) a kakan of fives added
    runner_up: dict = field(
        default_factory=dict
    )  # (seat, j) -> a candidate draw with the preferred choice forbidden
    haipai_margins: dict = field(
        default_factory=dict
    )  # seat -> certified lower bound when this exact haipai is forbidden
    melds: dict = field(
        default_factory=dict
    )  # (seat, j) -> index of the meld option the solver chose
    model_fingerprint: str | None = (
        None  # exact constraints/objective for safe in-memory margin reuse
    )
    optimal: bool | None = (
        None  # remains meaningful when status is later changed to repaired
    )
    confidence_seconds: float = 0.0
    draw_sources: dict = field(
        default_factory=dict
    )  # draws determined by a discard or ankan
    alternative_gaps: dict = field(
        default_factory=dict
    )  # feasible candidate gap, None when no candidate was found
    haipai_alternative_gaps: dict = field(default_factory=dict)
    discard_alternative_gaps: dict = field(default_factory=dict)
    discard_margins: dict = field(
        default_factory=dict
    )  # variable pond identities need certification even when unchanged
    discard_runner_up: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """A reconstruction exists (possibly after repair)."""
        return self.status in ("optimal", "feasible", "repaired")


@dataclass
class BuildVariables:
    """Per-build variables and objective terms; never shared between solves."""

    model: ReconstructionModel = field(default_factory=ReconstructionModel)
    starting: StartingVariables = field(default_factory=dict)
    draws: DrawVariables = field(default_factory=dict)
    rinshan: DrawVariables = field(default_factory=dict)
    kan_choices: DrawVariables = field(default_factory=dict)
    red_kan_choices: dict[tuple[str, int], cp_model.IntVar] = field(
        default_factory=dict
    )
    discards: dict[tuple[str, int], dict[int, cp_model.IntVar | int]] = field(
        default_factory=dict
    )
    discard_costs: list[cp_model.LinearExpr | int] = field(default_factory=list)
    hands: dict[tuple[str, int], list[cp_model.LinearExpr]] = field(
        default_factory=dict
    )
    meld_choices: DrawVariables = field(default_factory=dict)
    terms: list[cp_model.LinearExpr | int] = field(default_factory=list)


@dataclass(frozen=True)
class ConfidenceBudget:
    """CPU allocation and total seconds shared by alternative checks."""

    workers: int
    seconds: float


class HandModel:
    """Mutable evidence and rules for one hand, rebuilt as CP-SAT on each solve.

    Add observations and normalized facts before solving. Counts distinguish
    red fives; meld removals and the dealer's starting fourteenth tile are
    represented explicitly rather than repaired during export.
    Concurrent ``solve`` calls on the same mutable instance are unsupported.
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
        self.ura = list(
            ura or []
        )  # ura indicators: revealed tiles too, they count against the four
        self.tsumo_winner = tsumo_winner
        self.hand_ev: list[HandEvidence] = []
        self.draw_ev: list[DrawEvidence] = []
        self.facts = Facts()
        self.repair = (
            False  # every discard a choice over all kinds (DESIGN.md 4.8 Repair)
        )
        self.forbidden_hands: list[
            tuple[str, int, list[str]]
        ] = []  # (seat, j, tiles): hands the next-best solve must avoid
        self.bound_hands: list[
            tuple[str, int, list[str]]
        ] = []  # (seat, j, tiles): hands the site's score requires
        # the site's result as a constraint (4.8 "The result as a constraint")
        self.win: WinSpec | None = None
        self.tenpai: list[
            tuple[str, int, int]
        ] = []  # (seat, state j, sets needed) of the seats tenpai at a draw
        self.result_constraints = (
            True  # diagnose turns them off to see whether they are what fails
        )

    # -- building ------------------------------------------------------------------
    def draw_turns(self, seat: str) -> list[int]:
        """Return draw and kan turn indices, including an undiscarded winning draw."""
        out = [t.j for t in self.turns[seat] if t.kind in ("draw", "kan")]
        if self.tsumo_winner == seat:
            out.append(len(self.turns[seat]))  # the winning draw, no discard after it
        return out

    def build(
        self,
        forbid: tuple[str, int, str] | None = None,
        forbid_haipai: tuple[str, list[str]] | None = None,
    ) -> tuple[ReconstructionModel, StartingVariables, DrawVariables, dict]:
        """Build a fresh model and return its starting-hand, draw, and state variables.

        Optional exclusions support alternative-cost measurement. Solver variable
        maps are replaced on this instance; margin workers retain their own
        variable maps and operate on independent models.
        """
        state = BuildVariables()
        state.starting = {
            s: [state.model.new_int_var(0, 4, f"h0_{s}_{k}") for k in range(NT)]
            for s in rules.SEATS
        }
        self._add_draw_variables(state)
        self._add_discard_variables(state)
        self._add_hand_states(state)
        self._add_inventory_limits(state)
        self._add_reviewed_facts(state)
        self._add_exclusions(state, forbid=forbid, forbid_haipai=forbid_haipai)
        self._add_hand_costs(state)
        self._add_draw_costs(state)
        state.terms += state.discard_costs
        state.model.minimize(sum(state.terms) if state.terms else 0)
        self._x = state.discards
        self._d2 = state.rinshan
        self._y = state.kan_choices
        self._vr = state.red_kan_choices
        self._mo = state.meld_choices
        return state.model, state.starting, state.draws, state.hands

    def _add_draw_variables(self, state: BuildVariables) -> None:
        """Create each draw and kan choice, then constrain starting sizes."""
        m = state.model
        h0 = state.starting
        d = state.draws
        d2 = state.rinshan
        y = state.kan_choices
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

    def _add_discard_variables(self, state: BuildVariables) -> None:
        """Fix pond identities or add evidence-weighted discard alternatives."""
        m = state.model
        x = state.discards
        disc_cost = state.discard_costs
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

    def _add_hand_states(self, state: BuildVariables) -> None:
        """Track chronological hand states and apply the authoritative result."""
        for s in rules.SEATS:
            expr: list[cp_model.LinearExpr] = list(state.starting[s])
            state.hands[(s, -1)] = list(expr)
            riichi_on = False
            for t in self.turns[s]:
                self._remove_called_tiles(state, s, t, expr)
                expr = self._take_draw(state, s, t, expr, riichi_on=riichi_on)
                expr = self._take_kan(state, s, t, expr, riichi_on=riichi_on)
                for k, v in state.discards.get((s, t.j), {}).items():
                    expr[k] = expr[k] - v
                for k in range(NT):
                    state.model.add(expr[k] >= 0)
                state.hands[(s, t.j)] = list(expr)
                if t.riichi:
                    riichi_on = True
            if self.tsumo_winner == s:
                jw = len(self.turns[s])
                expr = [expr[k] + state.draws[(s, jw)][k] for k in range(NT)]
                state.hands[(s, jw)] = list(expr)
        if self.result_constraints:
            self._result_constraints(state.model, state.hands, state.discards)

    def _remove_called_tiles(
        self,
        state: BuildVariables,
        s: str,
        t: SeatTurn,
        expr: list[cp_model.LinearExpr],
    ) -> None:
        """Require the chosen call composition to exist before its draw."""
        m = state.model
        mo = state.meld_choices
        disc_cost = state.discard_costs
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
        state: BuildVariables,
        s: str,
        t: SeatTurn,
        expr: list[cp_model.LinearExpr],
        *,
        riichi_on: bool,
    ) -> list[cp_model.LinearExpr]:
        """Add a draw and enforce the frozen hand after riichi."""
        m = state.model
        d = state.draws
        y = state.kan_choices
        xs = state.discards.get((s, t.j), {})
        if (s, t.j) in d:
            expr = [expr[k] + d[(s, t.j)][k] for k in range(NT)]
            if (
                riichi_on and xs and not t.two_draws
            ):  # after riichi every draw is discarded at once
                for k in range(NT):
                    m.add(d[(s, t.j)][k] == xs.get(k, 0))
            elif riichi_on and t.two_draws:
                # after riichi a concealed kan is only of the tile just drawn
                # (the hand is frozen)
                if t.kan_tile in TI:
                    red = {"5m": "0m", "5p": "0p", "5s": "0s"}.get(t.kan_tile)
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
        state: BuildVariables,
        s: str,
        t: SeatTurn,
        expr: list[cp_model.LinearExpr],
    ) -> None:
        """Account for known or chosen kan tiles, including the red five."""
        m = state.model
        y = state.kan_choices
        vr = state.red_kan_choices
        n_out = 4 if t.kan == "ankan" else 1
        red = {"5m": "0m", "5p": "0p", "5s": "0s"}.get(t.kan_tile or "")
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
                    m.add(
                        y[(s, t.j)][k] == 0
                        # a kan of fives is named by the meld camera, never
                        # guessed
                    )
                expr[k] = expr[k] - n_out * y[(s, t.j)][k]

    def _take_kan(
        self,
        state: BuildVariables,
        s: str,
        t: SeatTurn,
        expr: list[cp_model.LinearExpr],
        *,
        riichi_on: bool,
    ) -> list[cp_model.LinearExpr]:
        """Require kan tiles before adding the replacement draw."""
        m = state.model
        d2 = state.rinshan
        xs = state.discards.get((s, t.j), {})
        if t.kan in ("ankan", "kakan"):
            self._remove_kan_tiles(state, s, t, expr)
            for k in range(NT):
                m.add(
                    expr[k] >= 0
                )  # the kan tiles must be in the hand before the rinshan draw
            if (s, t.j) in d2:
                expr = [expr[k] + d2[(s, t.j)][k] for k in range(NT)]
                if (
                    riichi_on and xs
                ):  # after riichi the rinshan draw is discarded at once
                    for k in range(NT):
                        m.add(d2[(s, t.j)][k] == xs.get(k, 0))
        return expr

    def _add_inventory_limits(self, state: BuildVariables) -> None:
        """Count every starting tile, acquisition and revealed indicator together."""
        m = state.model
        h0 = state.starting
        d = state.draws
        d2 = state.rinshan
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

    def _add_reviewed_facts(self, state: BuildVariables) -> None:
        """Constrain confirmed starting hands, draws and final hand states."""
        m = state.model
        h0 = state.starting
        d = state.draws
        hands = state.hands
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

    def _add_exclusions(
        self,
        state: BuildVariables,
        *,
        forbid: tuple[str, int, str] | None,
        forbid_haipai: tuple[str, list[str]] | None,
    ) -> None:
        """Keep alternative constraints at the recorded cloning insertion point."""
        m = state.model
        h0 = state.starting
        d = state.draws
        hands = state.hands
        m.draw_forbid_index = len(m.proto.constraints)
        if forbid is not None:
            s, j, tile = forbid
            if (s, j) in d:
                m.add(d[(s, j)][TI[tile]] == 0)
        if forbid_haipai is not None:
            _differs(
                m, h0[forbid_haipai[0]], forbid_haipai[1], f"h0_{forbid_haipai[0]}"
            )
        for n_, (s, j, tiles) in enumerate(self.forbidden_hands):
            if (s, j) in hands:
                _differs(m, hands[(s, j)], tiles, f"nb{n_}")
        for s, j, tiles in self.bound_hands:
            if (s, j) in hands:
                for k in range(NT):
                    m.add(hands[(s, j)][k] == sum(1 for t in tiles if t == TILES[k]))

    def _add_hand_costs(self, state: BuildVariables) -> None:
        """Penalize differences between visible rows and reconstructed hand states."""
        m = state.model
        d = state.draws
        hands = state.hands
        terms = state.terms
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
            self._add_row_deviations(state, ev, expr, w)

    def _add_row_deviations(
        self,
        state: BuildVariables,
        ev: HandEvidence,
        expr: list[cp_model.LinearExpr],
        w: int,
    ) -> None:
        """Penalize visible row differences and any excess beyond hidden tiles."""
        m = state.model
        terms = state.terms
        for k in range(NT):
            target = round(ev.e[k] * SCALE)
            if ev.subset and target <= 0:
                continue
            dev = m.new_int_var(0, 20 * SCALE, f"dev_{ev.seat}_{ev.j}_{k}_{len(terms)}")
            if not ev.subset and not ev.hidden:
                m.add(
                    dev >= expr[k] * SCALE - target
                )  # a full row: no tile more than it shows ...
            m.add(
                dev >= target - expr[k] * SCALE
            )  # ... and every tile it shows is in the hand
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

    def _add_draw_costs(self, state: BuildVariables) -> None:
        """Add direct draw likelihood costs after the hand evidence terms."""
        d = state.draws
        terms = state.terms
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
        """Require a winning final hand or the recorded draw's tenpai states.

        The winner's final hand is a winning hand; the tenpai seats of a draw are
        tenpai.
        """
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

    def _discard_of(self, key: tuple[str, int]) -> str | None:
        s, j = key
        for t in self.turns[s]:
            if t.j == j:
                return t.discard
        return None

    def _riichi_draw_sources(self) -> dict:
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
        margins: bool = True,
        workers: int = 8,
        prior: Solution | None = None,
        confidence_timeout: float = 10.0,
    ) -> Solution:
        """Reconstruct the hand and optionally certify alternatives for uncertain tiles.

        ``prior`` seeds the new search with soft hints, allowing new evidence or facts
        to replace old choices. It reuses confident margins only when the complete
        model, baseline objective and selected tile or starting multiset are unchanged.
        New evidence must be re-evaluated even if it happens to preserve the same
        preferred tiles.
        """
        m, h0, d, hands = self.build()
        baseline = (m, h0, d)
        fingerprint = (
            hashlib.sha256(str(m.proto).encode("utf-8")).hexdigest()
            if margins
            else None
        )
        if prior is not None and prior.ok:
            self._add_prior_hints(baseline, prior)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = workers
        status = solver.solve(m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # A timeout without a candidate does not prove infeasibility.
            return Solution(
                "infeasible" if status == cp_model.INFEASIBLE else "unsolved",
                float("inf"),
                {},
                {},
                {},
            )
        sol = self._read_solution(solver, status, baseline, hands)
        if margins:
            sol.model_fingerprint = fingerprint
        self._read_call_and_discard_choices(solver, sol)
        if margins:
            self._certify_alternatives(
                baseline,
                solver,
                sol,
                prior,
                ConfidenceBudget(workers, confidence_timeout),
            )
        return sol

    def _add_prior_hints(self, baseline: Baseline, prior: Solution) -> None:
        """Seed a useful incumbent without constraining updated evidence."""
        m, h0, d = baseline
        for seat, variables in h0.items():
            if seat in prior.haipai:
                counts = Counter(prior.haipai[seat])
                for tile, variable in zip(TILES, variables, strict=False):
                    m.add_hint(variable, counts[tile])
        for variables, choices in (
            (d, prior.draws),
            (self._d2, prior.draws2),
            (self._y, prior.kans),
        ):
            for key, vs in variables.items():
                if choices.get(key) in TI:
                    for tile, variable in zip(TILES, vs, strict=False):
                        m.add_hint(variable, int(tile == choices[key]))
        self._add_prior_call_hints(m, prior)
        self._add_prior_discard_hints(m, prior)

    def _add_prior_call_hints(self, m: cp_model.CpModel, prior: Solution) -> None:
        """Seed red-five kan and meld composition choices."""
        for key, variable in self._vr.items():
            if key in prior.kan_added:
                m.add_hint(variable, int(prior.kan_added[key] in rules.REDS))
        for key, variables in self._mo.items():
            if key in prior.melds:
                for index, variable in enumerate(variables):
                    m.add_hint(variable, int(index == prior.melds[key]))

    def _add_prior_discard_hints(self, m: cp_model.CpModel, prior: Solution) -> None:
        """Seed variable discards while leaving fixed pond identities untouched."""
        for key, variables in self._x.items():
            tile = prior.discards.get(key, self._discard_of(key))
            if tile in TI:
                for index, variable in variables.items():
                    if not isinstance(
                        variable, int
                    ):  # fixed discards have no decision variable
                        m.add_hint(variable, int(index == TI[tile]))

    def _read_solution(
        self,
        solver: cp_model.CpSolver,
        st: cp_model.CpSolverStatus,
        baseline: Baseline,
        hands: dict,
    ) -> Solution:
        """Read primary hands and acquisitions from a feasible assignment."""
        _model, h0, d = baseline
        obj = solver.objective_value / SCALE / SCALE
        haipai = {
            s: [TILES[k] for k in range(NT) for _ in range(solver.value(h0[s][k]))]
            for s in rules.SEATS
        }
        draws = {
            key: TILES[int(np.argmax([solver.value(v) for v in vs]))]
            for key, vs in d.items()
        }
        hs = {
            key: [TILES[k] for k in range(NT) for _ in range(solver.value(expr[k]))]
            for key, expr in hands.items()
        }
        sol = Solution(
            "optimal" if st == cp_model.OPTIMAL else "feasible", obj, haipai, draws, hs
        )
        sol.optimal = st == cp_model.OPTIMAL
        sol.draw_sources = {
            key: source
            for key, source in self._riichi_draw_sources().items()
            if key in draws
        }
        return sol

    def _read_call_and_discard_choices(
        self, solver: cp_model.CpSolver, sol: Solution
    ) -> None:
        """Materialize kan, meld and repaired pond choices."""
        for key, vs in self._d2.items():
            sol.draws2[key] = TILES[int(np.argmax([solver.value(v) for v in vs]))]
        for key, vs in self._y.items():
            sol.kans[key] = TILES[int(np.argmax([solver.value(v) for v in vs]))]
        for key, v in self._vr.items():
            tile = next(t.kan_tile for t in self.turns[key[0]] if t.j == key[1])
            if tile is None:
                msg = "A red-five kan choice must have a known tile kind"
                raise RuntimeError(msg)
            sol.kan_added[key] = (
                {"5m": "0m", "5p": "0p", "5s": "0s"}[tile] if solver.value(v) else tile
            )
        for key, bs in self._mo.items():
            sol.melds[key] = next(i for i, b in enumerate(bs) if solver.value(b))
        for key, xs in self._x.items():
            if len(xs) < MIN_DECISION_CHOICES:
                continue
            for k, v in xs.items():
                if solver.value(v) == 1 and TILES[k] != self._discard_of(key):
                    sol.discards[key] = TILES[k]

    def _pending_draw_checks(
        self, sol: Solution, prior: Solution | None
    ) -> list[tuple[tuple[str, int], str]]:
        """Reuse valid draw certificates and select the remaining alternatives."""
        draws, obj = sol.draws, sol.objective
        draws_todo = []
        for key, tile in draws.items():
            if key in sol.draw_sources:
                continue  # use the source's confidence after checking discards
            if (
                prior is not None
                and prior.model_fingerprint == sol.model_fingerprint
                and prior.objective == obj
                and key in prior.margins
                and prior.draws.get(key) == tile
                and prior.margins[key] > MARGIN_REVIEW + MARGIN_TOLERANCE
            ):
                sol.margins[key] = prior.margins[key]
                sol.runner_up[key] = prior.runner_up.get(key)
                sol.alternative_gaps[key] = prior.alternative_gaps.get(
                    key, prior.margins[key]
                )
            elif key in self.facts.draws:
                sol.margins[key] = float("inf")  # fixed by a fact
                sol.alternative_gaps[key] = float("inf")
            else:
                draws_todo.append((key, tile))
        return draws_todo

    def _pending_starting_checks(
        self, sol: Solution, prior: Solution | None
    ) -> list[str]:
        """Reuse starting-hand certificates only for the same multiset and model."""
        fingerprint, obj, haipai = sol.model_fingerprint, sol.objective, sol.haipai
        same_model = (
            prior is not None
            and prior.model_fingerprint == fingerprint
            and prior.objective == obj
        )
        haipai_todo = []
        for seat in rules.SEATS:
            if seat in self.facts.haipai:
                continue
            if (
                same_model
                and Counter(prior.haipai.get(seat, [])) == Counter(haipai[seat])
                and prior.haipai_margins.get(seat, 0) > MARGIN_REVIEW + MARGIN_TOLERANCE
            ):
                sol.haipai_margins[seat] = prior.haipai_margins[seat]
                sol.haipai_alternative_gaps[seat] = prior.haipai_alternative_gaps.get(
                    seat
                )
            else:
                haipai_todo.append(seat)
        sol.haipai_margins.update(
            {s_: float("inf") for s_ in rules.SEATS if s_ in self.facts.haipai}
        )
        sol.haipai_alternative_gaps.update(
            {s_: float("inf") for s_ in rules.SEATS if s_ in self.facts.haipai}
        )
        return haipai_todo

    def _pending_discard_checks(
        self, sol: Solution, prior: Solution | None
    ) -> list[
        tuple[tuple[str, int], dict[int, cp_model.IntVar | int], cp_model.IntVar | int]
    ]:
        """Capture variable maps before alternative workers rebuild the model."""
        fingerprint, obj = sol.model_fingerprint, sol.objective
        same_model = (
            prior is not None
            and prior.model_fingerprint == fingerprint
            and prior.objective == obj
        )
        discard_todo = []
        for key, variables in self._x.items():
            if len(variables) < MIN_DECISION_CHOICES:
                continue
            chosen = sol.discards.get(key, self._discard_of(key))
            if (
                same_model
                and prior.discards.get(key, self._discard_of(key)) == chosen
                and prior.discard_margins.get(key, 0) > MARGIN_REVIEW + MARGIN_TOLERANCE
            ):
                sol.discard_margins[key] = prior.discard_margins[key]
                sol.discard_runner_up[key] = prior.discard_runner_up.get(key)
                sol.discard_alternative_gaps[key] = prior.discard_alternative_gaps.get(
                    key
                )
            else:
                discard_todo.append((key, variables, variables[TI[chosen]]))
        return discard_todo

    def _certify_alternatives(
        self,
        baseline: Baseline,
        solver: cp_model.CpSolver,
        sol: Solution,
        prior: Solution | None,
        budget: ConfidenceBudget,
    ) -> None:
        """Bound alternatives under one deadline and a shared CPU allocation."""
        m, h0, d = baseline
        obj, haipai = sol.objective, sol.haipai
        workers, confidence_timeout = budget.workers, budget.seconds
        hint = (
            {s_: [solver.value(h0[s_][k]) for k in range(NT)] for s_ in rules.SEATS},
            {key2: [solver.value(v) for v in vs] for key2, vs in d.items()},
        )
        draws_todo = self._pending_draw_checks(sol, prior)
        haipai_todo = self._pending_starting_checks(sol, prior)
        discard_todo = self._pending_discard_checks(sol, prior)
        # Exclude each choice to bound its objective separation. A returned
        # runner-up is a feasible candidate, not necessarily the closest
        # alternative when its search hits the deadline or stops on proof.
        # One budget for this pass, rather than a fresh full budget for
        # every tile. Share the CPU allocation across concurrent checks.
        confidence_started = time.monotonic()
        deadline = confidence_started + confidence_timeout
        check_workers = max(1, workers // MARGIN_THREADS)
        with ThreadPoolExecutor(min(MARGIN_THREADS, workers)) as ex:
            # Starting hands constrain many draws. Establish these first
            # so a tight budget still finds the most useful questions.
            fh = {
                s_: ex.submit(
                    self._resolve,
                    hint,
                    obj,
                    check_workers,
                    options=ResolveOptions(
                        forbid_haipai=(s_, haipai[s_]), deadline=deadline
                    ),
                )
                for s_ in haipai_todo
            }
            fx = {
                key: ex.submit(
                    self._resolve,
                    hint,
                    obj,
                    check_workers,
                    options=ResolveOptions(
                        baseline=(m, h0, d),
                        forbid_variable=variable,
                        watch_variables=variables,
                        deadline=deadline,
                    ),
                )
                for key, variables, variable in discard_todo
            }
            fd = {
                key: ex.submit(
                    self._resolve,
                    hint,
                    obj,
                    check_workers,
                    options=ResolveOptions(
                        forbid=(key[0], key[1], tile),
                        watch=key,
                        baseline=(m, h0, d),
                        deadline=deadline,
                    ),
                )
                for key, tile in sorted(
                    draws_todo, key=lambda row: (row[0][1], row[0][0])
                )
            }
            for key, f in fd.items():
                sol.margins[key], sol.runner_up[key], sol.alternative_gaps[key] = (
                    f.result()
                )
            for s_, f in fh.items():
                sol.haipai_margins[s_], _, sol.haipai_alternative_gaps[s_] = f.result()
            for key, f in fx.items():
                (
                    sol.discard_margins[key],
                    sol.discard_runner_up[key],
                    sol.discard_alternative_gaps[key],
                ) = f.result()
        self._link_locked_draw_confidence(sol)
        sol.confidence_seconds = time.monotonic() - confidence_started

    def _link_locked_draw_confidence(self, sol: Solution) -> None:
        """Share a riichi-locked draw certificate with its corresponding discard."""
        for key, source in sol.draw_sources.items():
            # Excluding a locked draw is exactly the same as excluding its
            # discard. Keep pond ambiguity once, without a second search.
            sol.margins[key] = (
                sol.discard_margins.get(key, float("inf"))
                if source == "discard"
                else float("inf")
            )
            sol.alternative_gaps[key] = (
                sol.discard_alternative_gaps.get(key, float("inf"))
                if source == "discard"
                else float("inf")
            )
            sol.runner_up[key] = (
                sol.discard_runner_up.get(key) if source == "discard" else None
            )

    def _alternative_model(
        self,
        baseline: Baseline | None,
        forbid_variable: cp_model.IntVar | int | None,
        forbid: tuple[str, int, str] | None,
        forbid_haipai: tuple[str, list[str]] | None,
    ) -> tuple[cp_model.CpModel, StartingVariables, DrawVariables]:
        """Clone the baseline where possible and otherwise rebuild the exclusion."""
        if forbid_variable is not None:
            if baseline is None:
                msg = "Excluding a variable requires its baseline model"
                raise ValueError(msg)
            return _clone_forbidden_model(*baseline, forbid_variable)
        if baseline is not None and forbid is not None and forbid_haipai is None:
            return _clone_draw_model(*baseline, forbid)
        m2, h0b, db, *_ = self.build(forbid=forbid, forbid_haipai=forbid_haipai)
        return m2, h0b, db

    def _resolve(
        self,
        hint: tuple,
        obj: float,
        workers: int,
        *,
        options: ResolveOptions | None = None,
    ) -> tuple[float, str | None, float | None]:
        """Measure the objective increase after forbidding one solved decision.

        Objective increase of the model with one decision forbidden (warm-started from
        the solution), and the draw `watch` takes then. A timed-out feasible solve
        contributes only its certified lower bound: the current candidate's cost is an
        upper bound and could falsely make an ambiguous draw appear certain. Its draw is
        still useful as a possible alternative. The third value is its cost gap: a close
        candidate establishes ambiguity; a distant one cannot certify confidence. It
        also controls targeted video rereads. The search can stop as soon as its lower
        bound certifies a gap above the shared review threshold; this does not require
        optimizing the rejected alternative. A search without a candidate can still
        supply a certified lower bound; its candidate gap is None because no
        alternative was found. A close
        feasible witness also ends the search: proving its exact optimal cost cannot
        change the decision to ask about it. Each check runs once; unfinished checks
        remain unresolvable with this processing budget.
        """
        options = options or ResolveOptions()
        baseline = options.baseline
        forbid_variable = options.forbid_variable
        stop_when_certified = options.stop_when_certified
        deadline = options.deadline
        forbid = options.forbid
        forbid_haipai = options.forbid_haipai
        if deadline is not None and time.monotonic() >= deadline:
            return 0.0, None, None
        m2, h0b, db = self._alternative_model(
            baseline, forbid_variable, forbid, forbid_haipai
        )
        _hint_alternative((m2, h0b, db), hint, forbid)
        s2 = cp_model.CpSolver()
        s2.parameters.max_time_in_seconds = 4.0
        s2.parameters.num_workers = workers
        proven_bound = 0.0

        def stop_after_proof(bound: float) -> None:
            nonlocal proven_bound
            if math.isfinite(bound):
                proven_bound = max(proven_bound, bound)
                if bound / SCALE / SCALE - obj > MARGIN_REVIEW + MARGIN_TOLERANCE:
                    s2.stop_search()

        if stop_when_certified:
            # Confidence needs a threshold certificate, not the exact optimum
            # of a rejected reconstruction. Never stop on an incumbent cost.
            s2.best_bound_callback = stop_after_proof

        class CloseAlternative(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self) -> None:
                if (
                    self.objective_value / SCALE / SCALE - obj
                    <= MARGIN_REVIEW + MARGIN_TOLERANCE
                ):
                    self.stop_search()

        remaining = 4.0 if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            return 0.0, None, None
        s2.parameters.max_time_in_seconds = remaining
        r = s2.solve(m2, CloseAlternative()) if stop_when_certified else s2.solve(m2)
        if r == cp_model.MODEL_INVALID:
            msg = f"Invalid alternative model: {s2.response_stats()}"
            raise RuntimeError(msg)
        return self._alternative_result(
            s2, r, db, options, SearchProof(obj, proven_bound)
        )

    @staticmethod
    def _alternative_result(
        s2: cp_model.CpSolver,
        r: cp_model.CpSolverStatus,
        db: DrawVariables,
        options: ResolveOptions,
        proof: SearchProof,
    ) -> tuple[float, str | None, float | None]:
        watch, watch_variables = options.watch, options.watch_variables
        obj, proven_bound = proof.reference_cost, proof.lower_bound
        if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            alt = (
                TILES[int(np.argmax([s2.value(v) for v in db[watch]]))]
                if watch is not None and watch in db
                else None
            )
            if watch_variables is not None:
                alt = next(TILES[k] for k, v in watch_variables.items() if s2.value(v))
            bound = (
                s2.objective_value
                if r == cp_model.OPTIMAL
                else max(proven_bound, s2.best_objective_bound)
            )
            return (
                max(0.0, bound / SCALE / SCALE - obj),
                alt,
                max(0.0, s2.objective_value / SCALE / SCALE - obj),
            )
        if r == cp_model.UNKNOWN:
            # A proof bound does not require an incumbent. Discarding it turns
            # already-certified choices into needless review and video work.
            # All model costs are nonnegative, so the default zero bound from
            # a search stopped before initialization is conservative as well.
            bound = max(proven_bound, s2.best_objective_bound)
            margin = (
                max(0.0, bound / SCALE / SCALE - obj) if math.isfinite(bound) else 0.0
            )
            return margin, None, None
        return (
            (float("inf"), None, float("inf"))
            if r == cp_model.INFEASIBLE
            else (0.0, None, None)
        )


def _hint_alternative(
    model: tuple[cp_model.CpModel, StartingVariables, DrawVariables],
    hint: tuple,
    forbid: tuple[str, int, str] | None,
) -> None:
    """Replace prior hints without hinting the explicitly forbidden draw."""
    m2, h0b, db = model
    # A cloned main model may already contain prior-assignment hints.
    # Duplicate variable hints invalidate CP-SAT's input.
    m2.clear_hints()
    h0v, dv = hint
    for s_ in rules.SEATS:
        for k in range(NT):
            m2.add_hint(h0b[s_][k], h0v[s_][k])
    for key2, vs in db.items():
        if forbid is not None and key2 == (forbid[0], forbid[1]):
            continue
        for k, v in enumerate(vs):
            m2.add_hint(v, dv[key2][k])


def _clone_draw_model(
    base: ReconstructionModel,
    h0: StartingVariables,
    draws: DrawVariables,
    forbid: tuple[str, int, str],
) -> tuple[cp_model.CpModel, StartingVariables, DrawVariables]:
    """Copy a draw alternative with exactly the original CP-SAT input ordering.

    Variable indices, objective terms and constraint ordering stay identical.
    Variable handles can be shared because CP-SAT uses their unchanged indices.
    Starting-hand alternatives still rebuild: those introduce extra variables.
    """
    seat, turn, tile = forbid
    variable = draws[(seat, turn)][TI[tile]] if (seat, turn) in draws else None
    return _clone_forbidden_model(base, h0, draws, variable)


def _clone_forbidden_model(
    base: ReconstructionModel,
    h0: StartingVariables,
    draws: DrawVariables,
    variable: cp_model.IntVar | int | None,
) -> tuple[cp_model.CpModel, StartingVariables, DrawVariables]:
    """Exclude one selected Boolean while preserving baseline ordering and maps."""
    cloned = base.clone()
    if variable is not None:
        constraint = cloned.add(variable == 0)
        exclusion = type(constraint.proto)()
        exclusion.copy_from(constraint.proto)
        constraints = list(base.proto.constraints)
        offset = base.draw_forbid_index
        cloned.proto.constraints.clear()
        cloned.proto.constraints.extend(constraints[:offset])
        cloned.proto.constraints.append(exclusion)
        cloned.proto.constraints.extend(constraints[offset:])
    return cloned, h0, draws


def _differs(m: cp_model.CpModel, expr: list, tiles: list[str], name: str) -> None:
    """Constrain tile counts to differ from a specified fixed-size hand.

    The counts `expr` hold at least one kind fewer than `tiles` (the sizes are fixed, so
    the hand differs).
    """
    fewer = []
    for k in range(NT):
        v = sum(1 for t in tiles if t == TILES[k])
        if v:
            b = m.new_bool_var(f"fewer_{name}_{k}")
            m.add(expr[k] <= v - 1).only_enforce_if(b)
            fewer.append(b)
    m.add_bool_or(fewer)


@dataclass
class WinSpec:
    """Location of the winner's complete hand within the solver state.

    Where the winner's winning hand is in the model: the state after its turn j (a
    tsumo: after the winning draw), plus the loser's discard of turn j_loser for a ron;
    `sets` the sets the concealed part must hold.
    """

    seat: str
    j: int
    sets: int
    ron_from: tuple[str, int] | None = None


TERMINALS = [k for k in range(34) if k >= FIRST_HONOR_INDEX or k % 9 in (0, 8)]


def _counts34(expr: list) -> list:
    """37-kind counts -> 34 kinds (a red five is a five)."""
    out = [expr[TI[t]] for t in rules.KINDS]
    for red, plain in rules.REDS.items():
        k = rules.KINDS.index(plain)
        out[k] = out[k] + expr[TI[red]]
    return out


def _complete(m: cp_model.CpModel, c34: list, sets: int, name: str) -> None:
    """Constrain tile counts to a complete hand under the allowed patterns.

    Constrain 34-kind counts to a complete hand with `sets` sets besides the melds and
    one pair — or, closed, seven pairs or the thirteen orphans. Integer decomposition:
    pons per kind, chis per start, the pair.
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
    """Find the hand state belonging to a calm observation interval.

    Index j of the state a calm interval [t0, t1] of the hand band belongs to (-1 =
    haipai state), or None when the interval straddles a turn of this seat.
    """
    j = -1
    for t in seat_turns:
        if t.t_discard <= t0 + 0.6:
            j = t.j
        elif t.t_discard - 0.6 <= t1:
            return None  # the discard happened inside the interval
    return j


def open_turn(seat_turns: list[SeatTurn], t1: float) -> SeatTurn | None:
    """Find the unfinished turn overlapped by an observation interval.

    The turn whose window [t_pre, t_discard] contains the interval's end t1 without
    reaching the discard's first sighting: the discard may already have happened inside
    the interval, so a row with the resting count there is ambiguous (before the draw or
    after the discard); only the row with one tile more is a state, the one after the
    draw.
    """
    for t in seat_turns:
        if t.t_discard - 0.6 > t1:
            # the pond may first show a discard several seconds after it was made (an
            # arm over the pond), but
            # not much longer: a row at rest that ends more than 8 s before the sighting
            # was still waiting to draw
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
    """Find the extra drawn tile at either end of an observed hand row.

    Which end of a row with one tile more than the resting count holds the drawn tile:
    [(end, weight)].

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
    """Associate a seat's observed hand rows with solver states.

    Map the hand observations of one seat to states (DESIGN.md 4.8, evidence 1-3).
    Returns the hand evidence, the direct draw evidence, and the seat's end habit (draws
    seen at the left end, at the right end).

    - a row with one tile more than the resting count, inside the turn: the state after
      the draw, and the drawn
      tile at an end of the row;
    - a row with the resting count between turns, calm or read-floor: the state (it
      shows every tile);
    - a calm row between turns one or two tiles short of it: every tile it shows is in
      the state, which holds at
      most that many more (a hand resting over an end of the row), at full weight;
    - anything else — a partial view, fewer tiles, a row inside a turn's window (before
      the draw or already after
      the discard), a view across a discard — shows part of the hand at some moment of
      the turn, and every such
      moment's tiles are in the hand after the turn's draw: the tiles seen are a
      sub-multiset of it.
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


@dataclass
class Culprit:
    """Decision whose removal restores a conflicting hand's feasibility.

    One decision without which an infeasible hand becomes legal: what a conflict
    question asks about.
    """

    kind: str  # fact | result | riichi | meld | discard
    seat: str | None
    j: int | None  # the seat's turn (meld, discard, riichi)
    text: str
    fact: str | None = None  # fact: haipai | draw | final


def diagnose(model: HandModel, time_limit: float = 5.0) -> list[Culprit]:
    """Find single decisions whose removal restores model feasibility.

    When the model is infeasible (even after repair): the single decisions whose removal
    makes it legal, the most specific first — a reviewer's fact that contradicts the
    video, the result constraint (the winner's hand as reconstructed cannot win), a
    riichi turn, one call. Empty when no single decision does.
    """
    return [
        *_fact_conflicts(model, time_limit),
        *_result_conflicts(model, time_limit),
        *_riichi_conflicts(model, time_limit),
        *_meld_conflicts(model, time_limit),
    ]


def _feasible(model: HandModel, time_limit: float) -> bool:
    program, *_ = model.build()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    return solver.solve(program) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def _fact_conflicts(model: HandModel, time_limit: float) -> Iterator[Culprit]:
    full = model.facts
    singles = (
        [("haipai", seat, seat, value) for seat, value in full.haipai.items()]
        + [("draw", key[0], key, value) for key, value in full.draws.items()]
        + [("final", seat, seat, value) for seat, value in full.final.items()]
    )
    for kind, seat, key, value in singles:
        trial = copy(model)
        trial.facts = Facts(
            **{name: dict(values) for name, values in full.__dict__.items()}
        )
        {
            "haipai": trial.facts.haipai,
            "draw": trial.facts.draws,
            "final": trial.facts.final,
        }[kind].pop(key)
        if _feasible(trial, time_limit):
            shown = " ".join(value) if isinstance(value, list) else value
            yield Culprit(
                "fact",
                seat,
                key[1] if kind == "draw" else None,
                f"your {kind} fact for {seat} ({shown}) contradicts "
                "what the video shows",
                fact=kind,
            )


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
                    f"{seat}'s riichi at its turn {turn.j} "
                    f"({turn.t_discard:.0f}s) cannot be: "
                    "after it the hand would have to change",
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
                )
