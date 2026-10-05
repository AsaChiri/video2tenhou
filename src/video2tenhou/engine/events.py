# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 5a: a hand's discards, calls, dead wall, turn sequence and riichi turns.

ponds -> discard logs; meld events anchored on the discards they took -> calls; the
dead-wall row -> dora and kans; the reviewer's facts; turns -> the sequence; the
turned tiles -> riichi (DESIGN.md 4.8). Each stage takes the hand and the results of
the stages before it and returns its own; questions, notes, answers it could not
apply and its reasoning go to the hand's report.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.perception.tiles import CLASSES

from . import dense, pond_evidence, questions, rules
from .calls import CallAnchor, meld_events
from .hand import in_window, seat_of, site_seat
from .indicators import indicator_row, reconcile_kans
from .melds import Call, meld_options
from .ponds import PondSlot, play_window, track_pond, virtual_slot
from .turns import HandEnding, Skip, Turn, assign_calls, merge

if TYPE_CHECKING:
    from pathlib import Path

    from video2tenhou.read import ReadModels
    from video2tenhou.record import HandResult

    from .dense import DenseContext
    from .questions import Report

CORNERS = ("TL", "TR", "BL", "BR")
KANS = ("kan", "ankan", "kakan")
DISCARD_FACT_WINDOW = 10  # s: a discard fact names the seat's slot nearest its time
MISSING_DISCARD_WINDOW = 15  # s: a missed-discard fact a pond slot already shows
MIN_UNANCHORED_VIEWS = 5  # views of a chi or pon whose discard no read found: asked
DUPLICATE_CALL_WINDOW = 5  # s between a meld event and the call it already is
CALL_FACT_WINDOW = 40  # s around a meld fact: the calls it replaces
MELD_TILE_COUNT = 3
TURN_FACT_WINDOW = 3  # s between a riichi-turn or kan-time fact and its discard
RON_SLACK = 1.0  # s after the winning discard a turn may still be sighted


@dataclass(frozen=True, kw_only=True)
class Hand:
    """One hand's fixed inputs: its entry, observations, site result and facts.

    ``result`` carries a reviewer's han/fu correction and ``site_han_fu`` the site's
    own values; ``t0``-``t1`` is the physical play window.
    """

    entry: dict
    obs: dict[str, list[dict]]
    result: HandResult
    site_han_fu: tuple[int | None, int | None]
    facts: dict
    t0: float
    t1: float

    @property
    def pond_obs(self) -> dict[str, list[dict]]:
        """The pond observations by region."""
        return {r: v for r, v in self.obs.items() if r.startswith("pond:")}

    @property
    def winner(self) -> str | None:
        """The winning seat of the site record."""
        return site_seat(self.result.winner, self.entry)

    @property
    def loser(self) -> str | None:
        """The seat that dealt into a ron."""
        return site_seat(self.result.loser, self.entry)

    @property
    def tsumo_winner(self) -> str | None:
        """The winner when the hand ended in a tsumo."""
        return self.winner if self.result.outcome == "tsumo" else None

    @property
    def winning_seat(self) -> str:
        """Require a named winner before evaluating winning-hand evidence."""
        if self.winner is None:
            raise ValueError(
                "Winning-hand analysis requires a winner in the site record"
            )
        return self.winner

    @property
    def ura(self) -> list[str]:
        """The ura indicators the reviewer entered."""
        return list(self.facts.get("ura") or [])


def hand_of(
    entry: dict, obs: dict[str, list[dict]], result: HandResult, facts: dict
) -> Hand:
    """Bind a hand's inputs to its play window and the reviewer's score correction.

    A confirmed `site_wrong` fact replaces the site's han/fu everywhere; the deltas
    stay the site's (4.8 scoring.py).
    """
    site = (result.han, result.fu)
    if facts.get("site_score"):
        result = dataclasses.replace(
            result, han=facts["site_score"]["han"], fu=facts["site_score"]["fu"]
        )
    ponds = {r: v for r, v in obs.items() if r.startswith("pond:")}
    t0, t1 = play_window(ponds, entry["t_start"], entry["t_end"])
    return Hand(
        entry=entry, obs=obs, result=result, site_han_fu=site, facts=facts, t0=t0, t1=t1
    )


# ---- 1. the discards ---------------------------------------------------------------


@dataclass(frozen=True)
class Ponds:
    """Each seat's discard log, with the times of the hand's first and last discard."""

    logs: dict[str, list[PondSlot]]
    first_discard: float
    last_discard: float | None


def pond_logs(hand: Hand, report: Report) -> Ponds:
    """Track each pond, then apply reviewed identities and missed discards."""
    logs = {
        seat_of(hand.entry, corner): track_pond(
            in_window(hand.obs[f"pond:{corner}"], hand.t0, hand.t1)
        )
        for corner in CORNERS
    }
    # the reviewer named the tile of a pond slot (the seat's slot nearest the time)
    for f in hand.facts.get("discard", []):
        mine = logs.get(f["seat"], [])
        if mine:
            sl = min(mine, key=lambda x: abs(x.t_first - f["t"]))
            if abs(sl.t_first - f["t"]) <= DISCARD_FACT_WINDOW:
                sl.p = np.zeros(len(CLASSES))
                sl.p[CLASSES.index(f["tile"])] = 1.0
    # a discard the reader never saw (the reviewer names seat, time and tile)
    for f in hand.facts.get("missing_discard", []):
        mine = logs[f["seat"]]
        if any(
            rules.plain(sl.tile) == rules.plain(f["tile"])
            and abs(sl.t_first - f["t"]) < MISSING_DISCARD_WINDOW
            for sl in mine
        ):
            # a later re-read found it: the fact would double it
            report.ignore("missing_discard", f, "the pond already shows this discard")
            continue
        vs = virtual_slot(logs, f["tile"], f["t"], seen=2)
        mine.insert(sum(1 for sl in mine if sl.t_first <= f["t"]), vs)
    times = [sl.t_first for sls in logs.values() for sl in sls]
    return Ponds(logs, min(times, default=hand.t0), max(times, default=None))


# ---- 2. the calls ------------------------------------------------------------------


@dataclass(frozen=True)
class AnchoredCalls:
    """The calls anchored on a discard, and clear melds whose discard no read found."""

    calls: list[Call]
    unanchored: list[Call]


def anchored_calls(hand: Hand, ponds: Ponds, context: DenseContext) -> AnchoredCalls:
    """Anchor observed meld events on the discards they took (calls.py).

    The camera says that a seat laid a meld, the pond which discard it took; with
    nothing taken, only a kan pattern is a call. Dense reads may add a called-away
    discard to the logs.
    """
    events = meld_events(hand.entry, hand.obs, hand.t0, hand.t1)
    anchor = CallAnchor(ponds.logs, hand.obs, context=context)
    calls = anchor.anchor(
        events, ponds.first_discard, ponds.last_discard, hand.tsumo_winner
    )
    # a clear chi or pon, a hand that says a meld was laid, and no discard found
    unanchored = [
        e
        for e in events
        if e.type in ("chi", "pon")
        and e.seen >= MIN_UNANCHORED_VIEWS
        and not any(
            c.seat == e.seat and abs(c.t_first - e.t_first) < DUPLICATE_CALL_WINDOW
            for c in calls
        )
        and not anchor.reread(e, calls)
    ]
    return AnchoredCalls(calls, unanchored)


def apply_meld_facts(hand: Hand, calls: list[Call], report: Report) -> list[Call]:
    """Apply the reviewer's meld facts after every inferred call (a fact always wins).

    "This meld is <type> <tiles> from <source>" replaces the call of that seat nearest
    its time and suppresses the seat's other calls within 40 s; "this meld does not
    exist" removes the nearest call (of the given type).
    """
    for f in hand.facts.get("meld", []):
        near = [
            c
            for c in calls
            if c.seat == f["seat"] and abs(c.t_first - f["t"]) < CALL_FACT_WINDOW
        ]
        typ, tiles, pos = f["type"], list(f["tiles"]), f.get("called_pos")
        if typ in KANS and len(tiles) == MELD_TILE_COUNT:
            tiles.append(rules.plain(tiles[0]))  # a kan is four tiles
        if pos is None:
            pos = 0 if typ in ("chi", "pon", "kan") else None
        nearest = min(near, key=lambda c: abs(c.t_first - f["t"])) if near else None
        source = (
            f.get("source")
            or (nearest.source if nearest else None)
            or ("kamicha" if typ == "chi" else None)
        )
        t_first, window = (
            (nearest.t_first, nearest.t_window)
            if nearest
            else (float(f["t"]), (float(f["t"]) - 8.0, float(f["t"])))
        )
        fact = Call(
            seat=f["seat"],
            t_first=t_first,
            t_window=window,
            type=typ,
            tiles=tiles,
            called_pos=pos,
            source=source,
            called_tile=tiles[pos] if pos is not None and pos < len(tiles) else None,
            conf=1.0,
            seen=9,
            human=True,
            anchor="fact",
        )
        calls = [c for c in calls if c not in near] + [fact]
        if near:
            replaced = ", ".join(f"{c.type} {''.join(c.tiles)}" for c in near)
            report.diagnostics.append(
                f"meld fact {typ} {''.join(tiles)} of {f['seat']} at {f['t']:.0f}s "
                f"replaces {replaced}"
            )
    for f in hand.facts.get("meld_remove", []):
        cands = [
            c
            for c in calls
            if c.seat == f["seat"]
            and abs(c.t_first - f["t"]) < CALL_FACT_WINDOW
            and (not f.get("type") or c.type == f["type"])
        ]
        if not cands:
            report.ignore("meld_remove", f, "no meld of this player near this time")
            continue
        gone = min(cands, key=lambda c: abs(c.t_first - f["t"]))
        calls = [c for c in calls if c is not gone]
        report.diagnostics.append(
            f"meld-remove fact: the {gone.type} {''.join(gone.tiles)} of {gone.seat} "
            f"at {gone.t_first:.0f}s is removed"
        )
    return calls


# ---- 3. the dead wall --------------------------------------------------------------


@dataclass(frozen=True)
class DeadWall:
    """The dora indicators in order, and the calls with the kans they reveal."""

    indicators: list[dict]
    calls: list[Call]
    revealed: set[int]  # id() of every kan an observed indicator explains

    @property
    def dora(self) -> list[str]:
        """The indicator tiles in order."""
        return [v["tile"] for v in self.indicators]


def dead_wall(hand: Hand, ponds: Ponds, calls: list[Call], report: Report) -> DeadWall:
    """Read the dead-wall row and reconcile its new indicators with the kans."""
    # a tsumo winner may have kanned after its last discard (a win on the rinshan
    # draw): its indicator follows the last discard; the discard after a last-turn
    # kan may be sighted before the flipped indicator is
    inds = indicator_row(
        hand.pond_obs,
        hand.t0,
        hand.t1,
        ponds.last_discard,
        t_after=15.0 if hand.tsumo_winner else 10.0,
    )
    if hand.facts.get("dora"):
        inds = _named_indicators(hand, ponds.logs, inds, report)
    else:
        for f in hand.facts.get("kan_time", []):
            report.ignore("kan_time", f, "no dora indicator without a time needs it")
    kans = reconcile_kans(
        inds,
        calls,
        logs=ponds.logs,
        obs=hand.obs,
        entry=hand.entry,
        t0=hand.t0,
        diagnostics=report.diagnostics,
        tsumo_winner=hand.tsumo_winner,
    )
    report.notes += [
        questions.kan_unplaced(v["tile"], v["t_first"]) for v in kans.unplaced
    ]
    return DeadWall(inds, kans.calls, kans.revealed)


def _named_indicators(
    hand: Hand, logs: dict[str, list[PondSlot]], inds: list[dict], report: Report
) -> list[dict]:
    """Match reviewer-named indicators to the observed ones in order.

    Each named tile takes the observed entry of that tile when there is one (its time
    and box), else the next observed entry; observed indicators beyond the list stay
    (kans).
    """
    used: set = set()
    fact_tiles = list(hand.facts["dora"])
    if len(inds) < len(fact_tiles):
        # a tile named twice in a row is one tile when the wall shows fewer entries
        # than the list (typed twice); with as many entries it is two indicators of
        # one kind, which a kan can reveal
        fact_tiles = [
            t for i, t in enumerate(fact_tiles) if i == 0 or t != fact_tiles[i - 1]
        ]
    picks: list = []
    for t in fact_tiles:
        k = next(
            (k for k, v in enumerate(inds) if v["tile"] == t and k not in used),
            None,
        )
        if k is not None:
            used.add(k)
        picks.append(k)
    for i in range(len(fact_tiles)):
        if picks[i] is None:
            picks[i] = next((k for k in range(len(inds)) if k not in used), None)
            if picks[i] is not None:
                used.add(picks[i])
    named = []
    for t, k in zip(fact_tiles, picks, strict=False):
        base = inds[k] if k is not None else {"t_first": None, "region": "fact"}
        named.append({**base, "tile": t, "human": True, "conf": 1.0})
    _place_indicator_times(hand, logs, named, report)
    # an observed indicator beyond the named ones stays, even of a kind named already
    return named + [v for k, v in enumerate(inds) if k not in used]


def _place_indicator_times(
    hand: Hand, logs: dict[str, list[PondSlot]], named: list[dict], report: Report
) -> None:
    """Apply reviewed kan times and ask about indicators that still have none.

    A kan-time fact is the discard the reviewer saw right after the kan of an
    indicator no frame shows.
    """
    unplaced = [v for i, v in enumerate(named) if i > 0 and v.get("t_first") is None]
    facts = sorted(hand.facts.get("kan_time", []), key=lambda f: f["t"])
    for f in facts[len(unplaced) :]:
        report.ignore("kan_time", f, "no dora indicator without a time needs it")
    for v, f in zip(unplaced, facts, strict=False):
        near = [
            (abs(sl.t_first - f["t"]), sl.t_first, s)
            for s, sls in logs.items()
            for sl in sls
            if abs(sl.t_first - f["t"]) <= TURN_FACT_WINDOW
            and (not f.get("seat") or s == f["seat"])
        ]
        if not near:
            report.ignore("kan_time", f, "no discard within 3 s of this time")
            continue
        _, tt, s = min(near)
        v["t_first"], v["t_before"], v["kan_maker"] = tt - 0.5, tt - 1.0, s
        report.diagnostics.append(
            f"kan-time fact: the kan of indicator {v['tile']} precedes {s}'s discard "
            f"at {tt:.0f}s"
        )
    report.items += [
        questions.kan_time(v["tile"])
        for i, v in enumerate(named)
        if i > 0 and v.get("t_first") is None
    ]


# ---- 4. the turn sequence ----------------------------------------------------------


@dataclass(frozen=True)
class TurnSequence:
    """The merged turns, every call, the calls a turn owns, and the skipped turns."""

    turns: list[Turn]
    calls: list[Call]
    live_calls: list[Call]
    skips: int

    def wall_use(self, tsumo_winner: str | None) -> tuple[int, int]:
        """Count the sequence's wall draws and kans.

        Every draw turn (the dealer's 14th tile included), every turn the merge
        skipped (a missed discard still drew) and the winning tsumo draw; each kan
        moved a live tile into the dead wall.
        """
        draws = sum(1 for t in self.turns if t.kind in ("draw", "kan"))
        draws += (1 if tsumo_winner else 0) + self.skips
        kans = sum(1 for c in self.live_calls if c.type in KANS)
        return draws, kans


def turn_sequence(
    hand: Hand,
    ponds: Ponds,
    calls: list[Call],
    unanchored: list[Call],
    context: DenseContext,
    report: Report,
) -> TurnSequence:
    """Merge the ponds and calls into turns and align the end with the result.

    A turn the merge had to skip is read again densely before it is asked; a call the
    turn order shows and no camera did takes its tiles from the solver's choice among
    the legal melds.
    """
    # where the hand ends: a tsumo winner draws the winning tile; after a ron the
    # seat after the loser would play
    end = {
        "tsumo": hand.winner,
        "ron": rules.next_seat(hand.loser) if hand.loser else None,
    }.get(hand.result.outcome)
    # the live wall less a tile per kan; a tsumo's winning draw is not in the sequence
    kans = sum(1 for c in calls if c.type in KANS)
    wall = rules.LIVE_WALL - kans - (1 if hand.result.outcome == "tsumo" else 0)
    ending = HandEnding(end=end, wall=wall, exhaustive=hand.result.outcome == "draw")
    merged = merge(ponds.logs, calls, rules.DEALER, ending=ending)
    if (
        context.models is not None
        and merged.skips
        and dense.skipped_turns(merged.skips, ponds.logs, context=context)
    ):
        merged = merge(ponds.logs, merged.calls, rules.DEALER, ending=ending)
    calls = merged.calls
    for t in merged.turns:
        c = t.call
        if c is not None and c.anchor == "hidden" and c not in calls:
            if c.called_tile is None or c.source is None:
                raise ValueError("A hidden call must identify its discard and source")
            c.options = meld_options(c.called_tile, c.source, [], four=False)
            calls.append(c)
    turns = _after_the_ron(hand, merged.turns, report)
    for finding in merged.findings:
        report.diagnostics.append(finding.text)
        if isinstance(finding, Skip):
            # a turn whose discard no read found: the log cannot be written without it
            i = finding.i
            t = turns[i].t if i < len(turns) else (turns[-1].t if turns else hand.t1)
            report.items.append(questions.missed_discard(finding.seat, t, i))
    _winners_last_kan(hand, turns, calls, report)
    owned = {id(t.own_call) for t in turns if t.own_call is not None}
    seq = TurnSequence(
        turns,
        calls,
        [c for c in calls if id(c) in owned],
        sum(1 for f in merged.findings if isinstance(f, Skip)),
    )
    _uncounted_calls(seq, unanchored, report)
    _wall(hand, seq, report)
    _unexplained_removals(ponds.logs, seq.live_calls, report)
    return seq


def _after_the_ron(hand: Hand, turns: list[Turn], report: Report) -> list[Turn]:
    """Drop the tiles laid after the hand-ending ron discard.

    A ron ends the hand on the loser's discard (section 1): nobody discards after it,
    and tiles reaching a pond later are the reveal and the clearing. Turns the merge
    placed after it that happened before it are real turns in the wrong place: they
    stay (the winner's hand is read by time).
    """
    loser = hand.loser
    if hand.result.outcome != "ron" or loser is None or loser not in rules.SEATS:
        return turns
    mine = [t for t in turns if t.seat == loser and t.slot is not None]
    if not mine:
        return turns
    win_turn = mine[-1]
    after = turns[turns.index(win_turn) + 1 :]
    late = [t for t in after if t.t > win_turn.t + RON_SLACK]
    if late:
        report.diagnostics.append(
            f"{len(late)} tile(s) seen in the ponds after {loser}'s winning discard "
            f"at {win_turn.t:.0f}s are the reveal and the clearing: dropped"
        )
        turns = [t for t in turns if t not in late]
        for i, t in enumerate(turns):
            t.i = i
    early = [t for t in after if t.t <= win_turn.t + RON_SLACK]
    if early:
        listed = ", ".join(f"{t.seat} at {t.t:.0f}s" for t in early)
        report.diagnostics.append(
            f"turns merged after {loser}'s winning discard at {win_turn.t:.0f}s "
            f"happened before it ({listed}): kept out of order"
        )
    return turns


def _winners_last_kan(
    hand: Hand, turns: list[Turn], calls: list[Call], report: Report
) -> None:
    """Add a kan followed by the winner's terminal rinshan draw as its last turn.

    A kan after the tsumo winner's last discard (the win is the rinshan draw): no
    discard follows it, so the merge left it on no turn (ankan / kakan) or pending on
    the last discard (daiminkan); it is the winner's last turn, one without a discard.
    """
    winner = hand.tsumo_winner
    if not winner:
        return
    attached = {id(t.own_call) for t in turns if t.own_call}
    my_last = max((t.t for t in turns if t.seat == winner), default=hand.t0)
    for c in calls:
        if c.seat != winner or id(c) in attached:
            continue
        if c.type in ("ankan", "kakan") and c.t_window[1] > my_last:
            t_kan = max(c.t_window[1], my_last + 0.5)
        elif c.type == "kan" and turns and turns[-1].call is c:
            t_kan = max(c.t_window[1], turns[-1].t + 0.5)
        else:
            continue
        turns.append(Turn(len(turns), winner, "kan", None, t_kan, own_call=c))
        report.diagnostics.append(
            f"{c.type} by {winner} after its last discard at {t_kan:.0f}s: the "
            "winning tile is the rinshan draw"
        )
        break


def _uncounted_calls(seq: TurnSequence, unanchored: list[Call], report: Report) -> None:
    """Ask about seen melds the log leaves out.

    A call counts only when the caller has a turn that owns it: that turn is where
    the solver takes the called tiles out of the hand. A meld no turn owns lacks the
    discard after it; a clear meld whose taken discard no read found is a call
    question.
    """
    live = {id(c) for c in seq.live_calls}
    for c in seq.calls:
        if id(c) not in live:
            i = next((t.i for t in seq.turns if t.t > c.t_first), len(seq.turns))
            report.items.append(questions.uncounted_meld(c, i))
    report.items += [questions.unanchored_call(e) for e in unanchored]


def _wall(hand: Hand, seq: TurnSequence, report: Report) -> None:
    """Note a sequence that draws more than the live wall, or less at a draw.

    Wall draws plus kans never exceed 70 (section 1), and an exhaustive draw is
    exactly 70.
    """
    draws, kans = seq.wall_use(hand.tsumo_winner)
    used = draws + kans
    if used > rules.LIVE_WALL:
        report.notes.append(questions.wall_overused(used))
    elif hand.result.outcome == "draw" and used < rules.LIVE_WALL:
        report.notes.append(questions.wall_short(used))


def _unexplained_removals(
    logs: dict[str, list[PondSlot]], live_calls: list[Call], report: Report
) -> None:
    """Note pond removals that no reconstructed call explains.

    Only a call takes a discard: the next seat played on, so either its call went
    unseen or the removal was misread. The reconstruction keeps the discard.
    """
    taken = assign_calls(live_calls, logs)
    for seat, sls in logs.items():
        for k, sl in enumerate(sls):
            if sl.t_removed is not None and taken[seat][k] is None and not sl.virtual:
                report.notes.append(
                    questions.unexplained_removal(seat, sl.tile, sl.t_removed)
                )


# ---- 5. riichi ---------------------------------------------------------------------


@dataclass(frozen=True)
class Riichi:
    """Who declared riichi, whose turn is a guess, and the turns the solver decides.

    An alternative is (turned, called): a turned tile right after the seat's own
    discard was called, so the declaration was that called tile or this one.
    """

    seats: frozenset[str]
    guessed: frozenset[str]
    alternatives: list[tuple[Turn, Turn]]


def place_riichi(
    hand: Hand, turns: list[Turn], context: DenseContext, report: Report
) -> Riichi:
    """Combine the declaring seats with the observed turned discards.

    The site record decides who declared (its deltas carry the sticks); the turned
    tile says when. A reviewer's riichi-turn fact names the declaring discard
    outright; a reviewer's riichi fact replaces the site's seats.
    """
    site = {site_seat(s, hand.entry) for s in hand.result.riichi}
    if hand.facts.get("riichi") is not None:
        answer = set(hand.facts["riichi"])
        if answer != site:
            report.diagnostics.append(
                f"riichi fact {_seats(answer)} differs from the site's "
                f"{_seats(site)}: the fact is used"
            )
        site = answer
    named = _reviewed_riichi_turns(hand, turns, report)
    turned = {t.seat for t in turns if t.riichi}
    guessed: set[str] = set()
    if turned != site:
        # a turned tile of another seat is a tile laid askew; a declaring seat with no
        # turned tile lacks its turn: its last discard is the guess and is asked
        report.diagnostics.append(
            f"riichi: turned tiles of {_seats(turned)}, declarations of {_seats(site)}"
        )
        for t in turns:
            t.riichi = t.riichi and t.seat in site
        guessed = _find_missing_riichi(sorted(site - turned), turns, context, report)
    alternatives = []
    for t in turns:
        if not t.riichi or t.seat in named or t.seat in guessed:
            continue
        prev = [
            x for x in turns if x.seat == t.seat and x.i < t.i and x.slot is not None
        ]
        if prev and prev[-1].call is not None:
            alternatives.append((t, prev[-1]))
    return Riichi(frozenset(site), frozenset(guessed), alternatives)


def _seats(seats: set[str]) -> str:
    return " ".join(sorted(seats)) or "nobody"


def _reviewed_riichi_turns(hand: Hand, turns: list[Turn], report: Report) -> set[str]:
    """Place reviewed declarations on their matching discards."""
    named = set()
    for f in hand.facts.get("riichi_turn", []):
        mine = [t for t in turns if t.seat == f["seat"] and t.slot is not None]
        near = min(mine, key=lambda t: abs(t.t - f["t"]), default=None)
        if near is None or abs(near.t - f["t"]) > TURN_FACT_WINDOW:
            report.ignore("riichi_turn", f, "no discard of this player within 3 s")
            continue
        for t in mine:
            t.riichi = t is near
        named.add(f["seat"])
    return named


def _find_missing_riichi(
    seats: list[str], turns: list[Turn], context: DenseContext, report: Report
) -> set[str]:
    """Reread missing declarations; ask about each that no read finds.

    Returns the seats whose declaration is placed on their last discard as a guess.
    """
    guessed = set()
    for s in seats:
        mine = [t for t in turns if t.seat == s and t.slot is not None]
        if not mine:
            continue
        found = dense.turned_tile(s, turns, context=context)
        if found is not None:
            found.riichi = True
            continue
        last = mine[-1]
        last.riichi = True
        guessed.add(s)
        report.items.append(questions.missing_riichi(s, last.t, last.discard_slot.tile))
    return guessed


# ---- 6. pond replacements ----------------------------------------------------------


def acquire_replacements(
    hand: Hand,
    turns: list[Turn],
    models: ReadModels | None,
    work_dir: Path | None,
    report: Report,
) -> None:
    """Read unresolved pond correspondences before the discards are costed.

    Each request is attempted once. Continuous geometry may substitute its dense
    posterior for overlapping sparse evidence, without adding turns or calls;
    ambiguous acquisition remains a specific review item. Reading failures propagate
    so broken acquisition cannot become missing evidence.
    """
    requests = pond_evidence.replacement_requests(
        turns, hand.entry, hand.facts, hand.t0, hand.t1
    )
    if models is None:
        return
    for request in requests:
        if request["acquired"] or request["window"] is None:
            continue
        slot = next(
            t.slot
            for t in turns
            if t.seat == request["seat"]
            and t.slot is not None
            and t.slot.id == request["slot_id"]
        )
        if work_dir is None:
            raise ValueError(
                "Dense evidence acquisition requires a workspace directory"
            )
        readings = pond_evidence.read_replacement(request, models, work_dir)
        if pond_evidence.consume_replacement(slot, request, readings):
            report.diagnostics.append(
                f"dense pond evidence for {request['seat']} at {request['t']:.0f}s "
                "replaced the overlapping sparse observation"
            )
