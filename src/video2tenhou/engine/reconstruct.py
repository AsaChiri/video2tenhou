# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 5b: haipai, draws and call tiles as one constraint program per hand.

The merged turns become the solver's model (solver.py), constrained by the hand
cameras, the reviewer's facts and the site's result. The first fit selects the draws
worth a dense hand read; a hand that is illegal as read is repaired by the cheapest
pond re-reading, or else asked as a conflict (DESIGN.md 4.8 "Repair"); a riichi on a
called tile is decided by the freeze; every kan gets its indicator. The first fit is
certified only when rereads are possible; the caller certifies the final solution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import dense, questions, rules
from .events import CORNERS, DISCARD_FACT_WINDOW, KANS
from .hand import corner_of, in_window, seat_of, site_seat_name
from .review import draws_to_reread, turn_key
from .solver import (
    TILES,
    HandEvidence,
    HandModel,
    HandRole,
    SeatTurn,
    WinSpec,
    diagnose,
    hand_evidence,
    posterior_to_tiles,
)

if TYPE_CHECKING:
    from .decode import DecodeOptions
    from .dense import DenseContext
    from .events import DeadWall, Hand, Riichi, TurnSequence
    from .melds import Call
    from .questions import Report
    from .solver import Culprit, Solution
    from .turns import Turn

MAX_PLAIN_TILE_COPIES = 4
RIICHI_CLOSE = 0.5  # the two readings of a riichi on a called tile solve this close
SOFT_HAND_WEIGHT = 3.0  # cost weight of an uncertain hand annotation
DRAW_FACT_WINDOW = 10.0  # s between a draw fact and the discard of its turn


def apply_choices(
    sol: Solution, model: HandModel, turns: list[Turn], unknown: set[int]
) -> None:
    """Write the solver's meld choices into the calls before scoring or export.

    Which legal composition a call is (its type and tiles), the tile of an ankan the
    camera never named, and which five (plain or red) a kakan of fives added.
    """
    for t in turns:
        c = t.own_call
        if c is None:
            continue
        st = next((x for x in model.turns[t.seat] if x.t_discard == t.t), None)
        if st is None:
            continue
        key = (t.seat, st.j)
        if c.options and key in sol.melds:
            o = c.options[sol.melds[key]]
            c.type, c.tiles, c.called_pos = o.type, list(o.tiles), o.called_pos
        if id(c) in unknown and key in sol.kans:
            c.tiles = [sol.kans[key]] * 4
        if c.type == "kakan" and key in sol.kan_added:
            c.tiles = [*c.tiles[:3], sol.kan_added[key]]


def seat_turns_of(
    turns: list[Turn], seat: str, dealer: str
) -> tuple[list[SeatTurn], dict[int, int]]:
    """Build a seat's turns and the number of melds laid before each state.

    melds_before[j] counts the melds laid up to state j (the hand after turn j).
    """
    out: list[SeatTurn] = []
    melds_before: dict[int, int] = {-1: 0}
    n_melds = 0
    j = 0
    for turn_index, t in enumerate(turns):
        if t.seat != seat:
            continue
        kind = t.kind
        removed: list[str] = []
        options: list = []
        if kind == "call" and t.own_call is not None:
            removed, options = _hand_tiles(t.own_call)
            n_melds += 1
        kan = kan_tile = None
        two = False
        if kind == "kan" and t.own_call is not None:
            c = t.own_call
            known = [x for x in c.tiles if x not in ("?", "X")]
            kan = c.type if c.type in ("ankan", "kakan") else "daiminkan"
            kan_tile = rules.plain(known[0]) if known else None
            if c.type in ("ankan", "kakan"):
                two = True  # normal draw, kan, rinshan draw
            else:
                removed, options = _hand_tiles(c)
            if c.type != "kakan":
                n_melds += 1
            if t.slot is None:
                # the winner's kan after its last discard: the rinshan draw is the
                # winning draw (the final draw variable), so this turn holds the
                # normal draw only (ankan / kakan) or no draw (daiminkan)
                two = False
                if kan == "daiminkan":
                    kind = "call"
        if j == 0 and seat == dealer and kind in ("draw", "kan"):
            # the dealer's 14 are one set: its first turn has no draw of its own (a
            # first-turn ankan still has its rinshan draw)
            kind = "first"
            two = kan is not None
        predecessor = turns[turn_index - 1] if turn_index else None
        draw_min = (
            predecessor.slot.t_window[0]
            if predecessor is not None
            and predecessor.slot is not None
            and not predecessor.virtual
            else None
        )
        out.append(
            SeatTurn(
                j,
                kind,
                t.slot.tile if t.slot else None,
                t.slot.t_window[0] if t.slot else t.t,
                t.t,
                removed=removed,
                riichi=t.riichi,
                discard_p=posterior_to_tiles(t.slot.p / max(t.slot.p.sum(), 1e-9))
                if t.slot is not None
                else None,
                kan=kan,
                kan_tile=kan_tile,
                two_draws=two,
                meld_options=options,
                taken=t.call is not None,
                t_draw_min=draw_min,
            )
        )
        melds_before[j] = n_melds
        j += 1
    melds_before[j] = n_melds
    return out, melds_before


def _hand_tiles(c: Call) -> tuple[list[str], list]:
    """Return the fixed tiles a call takes from the hand, or the solver's options.

    Fixed: a reviewer's meld, or a call with one composition; otherwise the legal
    compositions as [(tiles, cost)].
    """
    if c.options and not c.human and len(c.options) > 1:
        return [], [(o.hand, o.cost) for o in c.options]
    return [x for i, x in enumerate(c.tiles) if i != c.called_pos], []


def _meld_copies(c: Call) -> dict[str, int]:
    """Count revealed meld tiles without recounting the discard that was called."""
    per_kind: dict[str, int] = defaultdict(int)
    if c.type == "kakan":
        # only the added tile: the pon below it is counted already
        per_kind[rules.plain(c.tiles[-1])] += 1
    else:
        for k, tile in enumerate(c.tiles):
            if k == c.called_pos and c.type in ("chi", "pon", "kan"):
                continue  # the called tile is one of the discards
            if tile not in ("?", "X"):
                per_kind[rules.plain(tile)] += 1
    return per_kind


def over_count(
    turns: list[Turn], calls: list[Call], dora: list[str], entry: dict
) -> list[dict]:
    """Find the kinds seen more than four times among discards, melds and indicators.

    Each comes with every source, so the reviewer can mark the misread one.
    """
    src: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        if t.slot is not None:
            src[rules.plain(t.slot.tile)].append(
                {
                    "kind": "discard",
                    "seat": t.seat,
                    "t": t.t,
                    "tile": t.slot.tile,
                    "i": t.i,
                    "corner": corner_of(entry, t.seat),
                    "box": list(t.slot.xyxy) if t.slot.xyxy else None,
                    "pos": [t.slot.row, t.slot.index],
                    "call": {
                        "seat": t.call.seat,
                        "type": t.call.type,
                        "t": t.call.t_first,
                        "corner": corner_of(entry, t.call.seat),
                    }
                    if t.call
                    else None,
                    "t_pic": min(
                        t.slot.t_first + 2.0, (t.slot.t_first + t.slot.t_last) / 2.0
                    )
                    if t.slot.t_last > t.slot.t_first
                    else t.slot.t_first,
                }
            )
    for c in calls:
        for kind, n in _meld_copies(c).items():
            src[kind].append(
                {
                    "kind": "meld",
                    "seat": c.seat,
                    "t": c.t_first,
                    "tile": kind,
                    "copies": n,
                    "corner": corner_of(entry, c.seat),
                    "meld": c.type + " " + "".join(c.tiles),
                    "type": c.type,
                    "tiles": c.tiles,
                }
            )
    for d in dora:
        src[rules.plain(d)].append({"kind": "indicator", "tile": d})
    out = []
    for kind, lst in src.items():
        count = sum(x.get("copies", 1) for x in lst)
        if count > MAX_PLAIN_TILE_COPIES:
            out.append(
                {
                    "tile": kind,
                    "count": count,
                    "sources": sorted(lst, key=lambda x: x.get("t", 0)),
                }
            )
    return out


@dataclass
class Search:
    """A hand's constraint model, the turns its choices go into, and its budgets."""

    model: HandModel
    turns: list[Turn]
    unknown_kans: set[int]  # id() of each ankan whose tile the solver chooses
    options: DecodeOptions

    def solve(self, *, prior: Solution | None = None) -> Solution:
        """Solve the model; a legal solution's meld choices go into the calls."""
        sol = self.explore(prior=prior)
        if sol.ok:
            apply_choices(sol, self.model, self.turns, self.unknown_kans)
        return sol

    def explore(self, *, prior: Solution | None = None) -> Solution:
        """Solve the model as it stands, leaving the calls untouched."""
        return self.model.solve(
            time_limit=self.options.time_limit,
            workers=self.options.workers,
            prior=prior,
        )

    def certify(self, sol: Solution) -> None:
        """Certify a legal solution once, with a full pass budget of its own."""
        if sol.ok and not sol.certified:
            self.model.certify(
                sol,
                timeout=self.options.confidence_timeout,
                workers=self.options.workers,
            )


def build_model(
    hand: Hand, seq: TurnSequence, dora: list[str], report: Report
) -> tuple[HandModel, dict[str, dict[int, int]]]:
    """Model the merged turns with the hand cameras' views, the facts and the result.

    Also returns, per seat, the number of melds laid before each state.
    """
    model = HandModel(
        rules.DEALER,
        {s: [] for s in rules.SEATS},
        dora,
        tsumo_winner=hand.tsumo_winner,
        ura=hand.ura,
    )
    melds_before: dict[str, dict[int, int]] = {}
    for s in rules.SEATS:
        model.turns[s], melds_before[s] = seat_turns_of(seq.turns, s, rules.DEALER)
    for corner in CORNERS:
        seat = seat_of(hand.entry, corner)
        hev, dev, _ = hand_evidence(
            seat,
            model.turns[seat],
            in_window(hand.obs.get(f"hand:{corner}", []), hand.t0, hand.t1),
            melds_before[seat],
            role=HandRole(
                dealer=seat == rules.DEALER, wins_by_tsumo=seat == hand.tsumo_winner
            ),
        )
        model.hand_ev += hev
        model.draw_ev += dev
    apply_hand_facts(
        model,
        hand.facts,
        calls=seq.live_calls,
        window=(hand.t0, hand.t1),
        report=report,
    )
    result_constraint(model, hand, seq.turns, melds_before)
    return model, melds_before


def unnamed_kans(calls: list[Call]) -> set[int]:
    """Return id() of each ankan whose tile no camera named: the solver chooses it."""
    return {id(c) for c in calls if c.type == "ankan" and "?" in c.tiles}


def fit_evidence(
    search: Search,
    hand: Hand,
    seq: TurnSequence,
    riichi: Riichi,
    *,
    melds_before: dict[str, dict[int, int]],
    context: DenseContext,
    report: Report,
) -> tuple[Solution, dict[tuple[str, int], str]]:
    """Fit a legal hand, reread its uncertain draws, then repair or ask a conflict.

    Ambiguous draws select the dense hand reads, so the first fit is certified only
    when the context can read the video. Also returns the discards a repair re-read,
    (seat, j) -> the kind each became.
    """
    model = search.model
    sol = search.solve()
    if sol.ok and context.models is not None:
        search.certify(sol)
        low = draws_to_reread(sol, model)
        evidence_before = (len(model.hand_ev), len(model.draw_ev))
        dense.draws(low, model, seq.turns, melds_before, context=context)
        # Partial views also change the objective, even when none pins a draw.
        if (len(model.hand_ev), len(model.draw_ev)) != evidence_before:
            sol = search.solve(prior=sol)
    repaired: dict[tuple[str, int], str] = {}
    if sol.status == "infeasible":
        sol = _repair(search, hand, seq, repaired, report)
    if sol.ok and riichi.alternatives:
        sol = _riichi_on_a_called_tile(search, sol, riichi.alternatives, report)
    return sol, repaired


def apply_hand_facts(
    model: HandModel,
    facts: dict,
    *,
    calls: list[Call],
    window: tuple[float, float],
    report: Report,
) -> None:
    """Constrain the model with the reviewer's tiles; soft annotations are evidence.

    Confirmed starting, drawn, discarded and final tiles become hard facts; an
    explicitly uncertain hand annotation is a weighted observation at the window's
    start (haipai) or end (final hand). ``calls`` are the calls the hand size counts.
    """
    t0, t1 = window

    def add_soft_hand(f: dict, state: int, t: float) -> None:
        counts = Counter(f["tiles"])
        model.hand_ev.append(
            HandEvidence(
                f["seat"],
                state,
                after_draw=False,
                e=np.array([counts[tile] for tile in TILES], dtype=float),
                weight=SOFT_HAND_WEIGHT,
                t0=t,
                t1=t,
            )
        )

    _apply_discard_facts(model, facts, report)
    for f in facts.get("haipai", []):
        if f.get("soft"):
            add_soft_hand(f, -1, t0)
        else:
            model.facts.haipai[f["seat"]] = f["tiles"]
    for f in facts.get("draw", []):
        key = turn_key(model, f, DRAW_FACT_WINDOW)
        if key is None:
            report.ignore("draw", f, "no draw of this player near this time")
            continue
        model.facts.draws[key] = f["tile"]
    for f in facts.get("final_hand", []):
        # binds only a complete concealed hand at the end (13 - 3 per meld, +1 for a
        # tsumo winner)
        s = f["seat"]
        base = 13 - 3 * sum(1 for c in calls if c.seat == s and c.type != "kakan")
        tsumo_win = s == model.tsumo_winner
        want = base + (1 if tsumo_win else 0)
        # the hand without the winning draw: bound before it
        excl = tsumo_win and len(f["tiles"]) == base
        if not excl and len(f["tiles"]) != want:
            report.ignore(
                "final_hand",
                f,
                f"{len(f['tiles'])} tiles, {want} expected"
                + (f" or {base} without the winning tile" if tsumo_win else ""),
            )
        elif f.get("soft"):
            # Explicit uncertain evidence can guide a reconstruction but cannot fix
            # its tiles.
            last = len(model.turns[s]) - 1 + (1 if (tsumo_win and not excl) else 0)
            add_soft_hand(f, last, t1)
        else:
            model.facts.final[s] = f["tiles"]
            if excl:
                model.facts.final_excl[s] = True


def _apply_discard_facts(model: HandModel, facts: dict, report: Report) -> None:
    """Keep confirmed discard identities hard, even during repair searches.

    A one-hot posterior alone is still a soft cost in repair mode.
    """
    for kind in ("discard", "missing_discard"):
        for f in facts.get(kind, []):
            mine = model.turns.get(f["seat"], [])
            if mine and f.get("t") is not None:
                nearest = min(mine, key=lambda turn: abs(turn.t_discard - f["t"]))
                if abs(nearest.t_discard - f["t"]) <= DISCARD_FACT_WINDOW:
                    nearest.discard, nearest.discard_p = f["tile"], None
                    continue
            report.ignore(kind, f, "no discard of this player near this time")


def result_constraint(
    model: HandModel, hand: Hand, turns: list[Turn], melds_before: dict
) -> None:
    """Constrain the final states to the site's win or tenpai seats.

    The winner's final hand is a winning hand; at a draw the tenpai seats are tenpai
    (4.8 "The result as a constraint").
    """
    w, loser, outcome = hand.winner, hand.loser, hand.result.outcome
    if w and outcome == "tsumo":
        j = len(model.turns[w])
        model.win = WinSpec(w, j, 4 - melds_before[w].get(j, 0))
    elif w and loser and outcome == "ron":
        lt = [t for t in turns if t.seat == loser and t.slot is not None]
        stl = next(
            (x for x in model.turns[loser] if lt and x.t_discard == lt[-1].t), None
        )
        if stl is not None:
            # the winner's hand at the winning discard, by time: turns placed after it
            # that happened before it count
            jwin = sum(1 for t in turns if t.seat == w and t.t <= lt[-1].t + 1.0) - 1
            model.win = WinSpec(
                w, jwin, 4 - melds_before[w].get(jwin, 0), ron_from=(loser, stl.j)
            )
    elif outcome == "draw":
        for s in rules.SEATS:
            if site_seat_name(s, hand.entry) in hand.result.tenpai and model.turns[s]:
                j = len(model.turns[s]) - 1
                model.tenpai.append((s, j, 4 - melds_before[s].get(j, 0)))


def _repair(
    search: Search,
    hand: Hand,
    seq: TurnSequence,
    repaired: dict[tuple[str, int], str],
    report: Report,
) -> Solution:
    """Find the cheapest pond re-readings that allow a legal reconstruction.

    With every discard a choice over all kinds, costed by its posterior, the cheapest
    re-readings that make the hand legal become the discards and the hand is solved
    again. Only when no re-reading helps is the hand a conflict.
    """
    model = search.model
    model.repair = True
    found = search.explore()
    model.repair = False
    if found.ok:
        for (s, j), tile in found.discards.items():
            st = model.turns[s][j]
            report.diagnostics.append(
                f"repair: {s}'s discard at {st.t_discard:.0f}s read {st.discard} is "
                f"legal only as {tile}"
            )
            repaired[(s, j)] = tile
            st.discard, st.discard_p = tile, None
        sol = search.solve()
        if sol.ok:
            sol.optimal = sol.optimal and found.optimal
            sol.status = "repaired"
            return sol
        found = sol
    if found.status != "infeasible":
        return found
    culprits = diagnose(model)
    report.diagnostics.append(
        "no legal reconstruction: "
        + ("; ".join(c.text for c in culprits) or "no single decision explains it")
    )
    report.items.append(_conflict(culprits, model, hand, seq))
    return found


def _conflict(
    culprits: list[Culprit], model: HandModel, hand: Hand, seq: TurnSequence
) -> dict:
    """Ask the one decision a conflict hinges on, or the over-counted kinds."""
    culprit = culprits[0] if culprits else None
    call = None
    if culprit is not None and culprit.kind == "meld" and culprit.seat is not None:
        seat = culprit.seat
        call = next(
            (
                t.own_call
                for t in seq.turns
                if t.seat == seat
                and t.own_call is not None
                and (
                    st := next(
                        (x for x in model.turns[seat] if x.t_discard == t.t), None
                    )
                )
                is not None
                and st.j == culprit.j
            ),
            None,
        )
    return questions.conflict(
        culprit,
        t=seq.turns[-1].t if seq.turns else hand.t1,
        over=over_count(seq.turns, seq.calls, model.indicators, hand.entry),
        call=call,
    )


def _riichi_on_a_called_tile(
    search: Search,
    sol: Solution,
    alternatives: list[tuple[Turn, Turn]],
    report: Report,
) -> Solution:
    """Decide a declaration on a called tile by the riichi freeze.

    The declaration was the called tile or the turned one (section 1); the reading
    the hand fits better stands, and a close one is asked.
    """
    for turned, called in alternatives:
        sts = search.model.turns[turned.seat]
        a = next(x for x in sts if x.t_discard == turned.t)
        b = next(x for x in sts if x.t_discard == called.t)
        a.riichi, b.riichi = False, True
        alt = search.explore()
        gap = abs((alt.objective if alt.ok else float("inf")) - sol.objective)
        if alt.ok and alt.objective < sol.objective:
            turned.riichi, called.riichi = False, True
            sol = search.solve(prior=sol)
            chosen = called
        else:
            a.riichi, b.riichi = True, False
            chosen = turned
        report.diagnostics.append(
            f"riichi of {turned.seat}: the called discard at {called.t:.0f}s or the "
            f"turned one at {turned.t:.0f}s; the "
            f"{'called' if chosen is called else 'turned'} one fits better by {gap:.1f}"
        )
        if gap < RIICHI_CLOSE:
            report.items.append(
                questions.riichi_on_called_tile(
                    turned.seat, chosen.t, chosen.discard_slot.tile
                )
            )
    return sol


def kan_indicators(
    hand: Hand,
    seq: TurnSequence,
    wall: DeadWall,
    sol: Solution,
    t_default: float,
    report: Report,
) -> tuple[list[str], list[dict]]:
    """Give every kan its indicator: the dora and indicator lists the log carries.

    Every kan reveals an indicator (section 1): the log carries 1 + kans of them,
    always. One that no view showed (the dead wall is often outside the overhead
    crop) is a `dora` question, and the log carries the best provisional placeholder
    meanwhile: a kind with copies left. A score mismatch never changes an indicator.
    """
    kans = sorted(
        (c for c in seq.live_calls if c.type in KANS), key=lambda c: c.t_first
    )
    seen = wall.dora
    missing = 1 + len(kans) - len(seen)
    if missing <= 0:
        return seen, wall.indicators
    used = Counter(
        x
        for x in [
            *(x for h in sol.haipai.values() for x in h),
            *sol.draws.values(),
            *sol.draws2.values(),
            *seen,
            *hand.ura,
        ]
        if x in rules.KINDS or x in rules.PLAIN_OF
    )

    def rank(x: str) -> tuple:
        left = rules.max_count(x) - used[x]
        return (-left, rules.KINDS.index(x))

    lost = bool(hand.facts.get("lost_dora"))
    guesses = []
    for _ in range(missing):
        x = min((x for x in rules.KINDS if used[x] < rules.max_count(x)), key=rank)
        used[x] += 1
        guesses.append(x)
    placeholders = [
        {
            "tile": x,
            "t_first": None,
            "region": None,
            "conf": 0.0,
            "lost": True,
            "human": lost,
        }
        for x in guesses
    ]
    dora, inds = [*seen, *guesses], [*wall.indicators, *placeholders]
    if lost:
        report.diagnostics.append(
            f"dora indicator(s) {' '.join(guesses)}: no view shows them and the "
            "reviewer cannot tell; the rules' guess stands"
        )
        return dora, inds
    # the kans no seen indicator explains (paired by reconcile_kans)
    bare = [c for c in kans if id(c) not in wall.revealed]
    report.items.append(
        questions.missing_indicators(
            seen, guesses, bare[0].t_first if bare else t_default
        )
    )
    return dora, inds
