# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The site's result in the reconstruction: the winning hand's score, tenpai, ura.

The site's han and fu constrain the construction (DESIGN.md 4.8): when the best
reconstruction's winning hand scores otherwise, an unseen tsumo tile, the next-best
reconstructions and a five read plain for red (or the reverse) are tried in that
order within the review gap, and the first that scores the site's value is bound
and solved again. Observed indicators never change to make a score match. The final
solution is then scored against the site record; what no alternative explains is a
`result` question, unless the unknown ura of a riichi win can.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from video2tenhou.perception.tiles import CLASSES

from . import questions, rules
from .confidence import MARGIN_REVIEW
from .hand import corner_of, site_seat_name
from .reconstruct import MAX_PLAIN_TILE_COPIES, apply_choices
from .scoring import WinContext, is_tenpai, matches_site, payment, score_hand

if TYPE_CHECKING:
    from collections.abc import Callable

    from .events import Hand, Riichi
    from .melds import Call
    from .ponds import PondSlot
    from .questions import Report
    from .reconstruct import Search
    from .scoring import ScoreResult
    from .solver import HandModel, Solution
    from .turns import Turn

    type Scorer = Callable[[list[str], str, list[str]], ScoreResult]

NEXT_HANDS = 6  # next-best reconstructions tried for the site's score ...
NEXT_BUDGET = MARGIN_REVIEW  # ... only close ones may resolve a mismatch


def concealed_size(melds: list[dict], *, drawn_left_in: bool = False) -> int:
    """Count the winner's concealed tiles beside the melds and the winning tile.

    A hand is 13 tiles; each meld set takes three of them out (a kan's fourth tile is
    the extra tile the kan adds). A kakan is the pon it grew from: listed beside that
    pon it is not a set of its own. `drawn_left_in` is for a tsumo whose winning tile
    was never named: it is still in the list.
    """
    pons = {rules.plain(m["tiles"][0]) for m in melds if m["type"] == "pon"}
    sets = sum(
        1
        for m in melds
        if m["type"] != "kakan" or rules.plain(m["tiles"][0]) not in pons
    )
    return 13 - 3 * sets + (1 if drawn_left_in else 0)


def scoring_melds(calls: list[Call], seat: str) -> list[dict]:
    """Convert a seat's melds to the scoring library's representation.

    An ankan's face-down tiles are its kind, any kan of fives is all four fives (three
    plain and the red one) whatever the camera read, and a kakan replaces the pon it
    grew from.
    """
    mine = [c for c in calls if c.seat == seat]
    grown = {rules.plain(c.tiles[0]) for c in mine if c.type == "kakan"}
    out = []
    for c in mine:
        if c.type == "pon" and rules.plain(c.tiles[0]) in grown:
            continue
        known = [x for x in c.tiles if x not in ("X", "?")]
        if c.type in ("kan", "ankan", "kakan"):
            tiles = rules.kan_tiles(known[0]) if known else ["?"] * 4
        else:
            tiles = list(c.tiles)
        out.append({"type": c.type, "tiles": tiles})
    return out


@dataclass(frozen=True, kw_only=True)
class Win:
    """A win's evidence beyond its solution: the sequence, riichi and indicators."""

    hand: Hand
    turns: list[Turn]
    live_calls: list[Call]
    logs: dict[str, list[PondSlot]]
    riichi: Riichi
    dora: list[str]
    wall_tiles: int  # wall draws and kans of the sequence: the 70th is haitei

    @property
    def t_last(self) -> float:
        """The time of the hand's last turn."""
        return self.turns[-1].t if self.turns else self.hand.t1


@dataclass(frozen=True, kw_only=True)
class WinningHand:
    """The winner's concealed tiles, the winning tile and the state that holds them.

    ``drawn_left_in`` marks an unknown winning draw still among the concealed tiles.
    """

    concealed: list[str]
    tile: str | None
    drawn_left_in: bool
    j: int


@dataclass(frozen=True)
class _Proposal:
    """A reconstruction with its winning hand and that hand's score."""

    sol: Solution
    concealed: list[str]
    tile: str
    score: ScoreResult
    melds: list[dict]


def winning_hand(win: Win, model: HandModel, sol: Solution) -> WinningHand:
    """Extract the winner's hand from a solution.

    A tsumo holds it after the winning draw; a ron at the winner's last turn at or
    before the winning discard, by time, not by position in the merged sequence.
    """
    hand, winner = win.hand, win.hand.winning_seat
    jw = len(model.turns[winner])
    if hand.result.outcome == "tsumo":
        tiles = sol.hands.get((winner, jw), [])
        win_tile = sol.draws.get((winner, jw))
        concealed = list(tiles)
        if win_tile in concealed:
            concealed.remove(win_tile)
            return WinningHand(
                concealed=concealed, tile=win_tile, drawn_left_in=False, j=jw
            )
        return WinningHand(
            concealed=concealed, tile=win_tile, drawn_left_in=bool(tiles), j=jw
        )
    win_tile, jwin = None, jw - 1
    loser = hand.loser
    lt = [t for t in win.turns if t.seat == loser and t.slot is not None]
    if lt and loser is not None:
        # the winning tile is the loser's last discard as the reconstruction has it
        last = lt[-1]
        jwin = sum(1 for t in win.turns if t.seat == winner and t.t <= last.t + 1.0) - 1
        stl = next((x for x in model.turns[loser] if x.t_discard == last.t), None)
        win_tile = (
            sol.discards.get((loser, stl.j)) if stl else None
        ) or last.discard_slot.tile
    return WinningHand(
        concealed=list(
            sol.hands.get((winner, jwin), sol.hands.get((winner, jw - 1), []))
        ),
        tile=win_tile,
        drawn_left_in=False,
        j=jwin,
    )


def scorer(win: Win, melds: list[dict]) -> tuple[Scorer, dict]:
    """Build the win's scoring function and its situational context.

    Ippatsu, double riichi, haitei/houtei and rinshan kaihou come from the turn
    sequence; a riichi whose turn is a guess earns neither ippatsu nor double riichi.
    """
    hand, turns = win.hand, win.turns
    winner, result = hand.winning_seat, hand.result
    my_turns = [t for t in turns if t.seat == winner]
    riichi_i = next((t.i for t in my_turns if t.riichi), None)
    ippatsu = double_riichi = False
    if (
        winner in win.riichi.seats
        and riichi_i is not None
        and winner not in win.riichi.guessed
    ):
        after = [t for t in turns if t.i > riichi_i]
        ippatsu = all(t.seat != winner for t in after) and all(
            t.kind == "draw" and t.call is None for t in after
        )
        # a riichi on the seat's first discard with no call (nor kan) by anyone before
        double_riichi = riichi_i == my_turns[0].i and all(
            t.kind == "draw" and t.call is None for t in turns[:riichi_i]
        )
    # the last tile of the wall: the winning draw is the 70th (haitei), the winning
    # discard follows it (houtei)
    haitei = win.wall_tiles >= rules.LIVE_WALL
    # rinshan kaihou: the winner kanned after its last discard and drew the winning
    # tile from the dead wall
    rinshan = (
        result.outcome == "tsumo"
        and bool(my_turns)
        and my_turns[-1].slot is None
        and my_turns[-1].own_call is not None
    )
    ura = hand.ura
    context = {"ippatsu": ippatsu, "haitei": haitei, "rinshan": rinshan, "ura": ura}
    round_wind = "ESW"[hand.entry["kyoku"] // 4]

    def score_of(conc: list[str], wt: str, dora: list[str]) -> ScoreResult:
        return score_hand(
            conc,
            wt,
            melds,
            context=WinContext(
                tsumo=result.outcome == "tsumo",
                riichi=winner in win.riichi.seats,
                seat=winner,
                round_wind=round_wind,
                dora=dora,
                ura=ura,
                ippatsu=ippatsu,
                haitei=haitei,
                rinshan=rinshan,
                double_riichi=double_riichi,
            ),
        )

    return score_of, context


def fits(win: Win, r: ScoreResult) -> bool:
    """Check the site's score, allowing for ura indicators not entered yet.

    With a riichi win whose ura is unknown, any winning hand whose han do not exceed
    the site's (ura may add the rest) with the right fu (ura never change the fu).
    """
    result = win.hand.result
    if matches_site(r, result.han, result.fu):
        return True
    return (
        win.hand.winner in win.riichi.seats
        and not win.hand.ura
        and r.ok
        and r.han is not None
        and result.han is not None
        and r.han <= result.han
        and r.fu == result.fu
    )


# ---- reconciliation ----------------------------------------------------------------


def reconcile_score(
    win: Win, search: Search, sol: Solution, report: Report
) -> Solution:
    """Prefer a reconstruction whose winning hand scores the site's han and fu."""
    result, winner = win.hand.result, win.hand.winner
    if not sol.ok or not winner or result.outcome not in ("ron", "tsumo"):
        return sol
    model = search.model
    jw = len(model.turns[winner])
    wh = winning_hand(win, model, sol)
    if result.outcome == "ron" and win.turns and wh.j != jw - 1:
        last = win.turns[-1]
        report.diagnostics.append(
            f"ron by {winner} on {win.hand.loser}: the last turn is {last.seat}'s at "
            f"{last.t:.0f}s, after the winning discard; the winner's hand is read at "
            "its last turn before that discard"
        )
    melds = scoring_melds(win.live_calls, winner)
    # the solver took the same calls' tiles out of the hand, so a mismatch is the
    # decoder's own inconsistency (the score then fails and asks for the hand)
    want = concealed_size(melds, drawn_left_in=wh.drawn_left_in)
    if len(wh.concealed) != want:
        report.diagnostics.append(
            f"the winner's hand does not add up: {len(wh.concealed)} concealed tiles "
            f"beside {len(melds)} meld(s), {want} expected"
        )
    if not wh.tile:
        return sol
    score_of, _ = scorer(win, melds)
    p = _Proposal(
        sol, wh.concealed, wh.tile, score_of(wh.concealed, wh.tile, win.dora), melds
    )
    if (
        result.outcome == "tsumo"
        and not fits(win, p.score)
        and (winner, jw) not in model.facts.draws
    ):
        p = _winning_tile_from_the_site(win, search, p, jw, report)
    if not fits(win, p.score) and winner not in model.facts.final:
        p = _next_hand_from_the_site(win, search, p, wh.j, report)
    if not fits(win, p.score) and winner not in model.facts.final:
        return _red_five_from_the_site(win, search, p, wh.j, report)
    return p.sol


def _bind_winning_hand(
    win: Win, model: HandModel, concealed: list[str], win_tile: str, j_hand: int
) -> None:
    """Bind the winner's hand the site's score requires for the next solve.

    The concealed part, and for a tsumo the state with the winning draw, so the solve
    that follows (margins and all) keeps it and the draws that brought it follow.
    Not a reviewer's fact: the confidence rows do not mark it human.
    """
    winner = win.hand.winning_seat
    if win.hand.result.outcome == "tsumo":
        model.bound_hands = [
            (winner, j_hand - 1, list(concealed)),
            (winner, j_hand, [*list(concealed), win_tile]),
        ]
    else:
        model.bound_hands = [(winner, j_hand, list(concealed))]


def _next_hand_from_the_site(
    win: Win, search: Search, p: _Proposal, j_hand: int, report: Report
) -> _Proposal:
    """Search the next-best reconstructions for a hand that scores the site's value.

    The winner's hand of each is forbidden in turn (up to NEXT_HANDS, within
    NEXT_BUDGET of the best); the first whose winning hand scores the site's value is
    bound and the hand solved again (margins and all).
    """
    winner, model = win.hand.winning_seat, search.model
    best = p.sol.objective
    hand = sorted(p.sol.hands.get((winner, j_hand), []))
    for _ in range(NEXT_HANDS):
        model.forbidden_hands.append((winner, j_hand, hand))
        alt = search.explore()
        if not alt.ok or alt.objective - best > NEXT_BUDGET:
            break
        apply_choices(alt, model, search.turns, search.unknown_kans)
        wh = winning_hand(win, model, alt)
        melds = scoring_melds(win.live_calls, winner)
        score = (
            scorer(win, melds)[0](wh.concealed, wh.tile, win.dora)
            if wh.tile
            else p.score
        )
        if wh.tile and fits(win, score):
            model.forbidden_hands.clear()
            _bind_winning_hand(win, model, wh.concealed, wh.tile, j_hand)
            sol = search.solve(prior=p.sol)
            if sol.ok:
                report.diagnostics.append(
                    f"the best winning hand scores {p.score.han}/{p.score.fu}; the "
                    f"next best that scores the site's ({alt.objective - best:.1f} "
                    f"costlier) is {' '.join(sorted(wh.concealed))} + {wh.tile}"
                )
                return _Proposal(sol, wh.concealed, wh.tile, score, melds)
            model.bound_hands = []
            break
        hand = sorted(alt.hands.get((winner, j_hand), []))
    model.forbidden_hands.clear()
    apply_choices(p.sol, model, search.turns, search.unknown_kans)
    return p


def _reveal_views(win: Win) -> list[dict]:
    """Return the winner's hand views after the last discard: the reveal."""
    corner = corner_of(win.hand.entry, win.hand.winning_seat)
    return [
        o
        for o in win.hand.obs.get(f"hand:{corner}", [])
        if o["t0"] >= win.t_last and o["n_used"] > 0
    ]


def _reveal_mass(win: Win, x: str) -> float:
    """How strongly the reveal reads tile x."""
    return sum(
        sl["p"][CLASSES.index(x)] for o in _reveal_views(win) for sl in o["slots"]
    )


def _reveal_extra(win: Win, x: str, concealed: list[str]) -> float:
    """Measure the reveal's support for a tile beyond the concealed hand.

    Over the views of the whole hand with the winning tile (one tile more than the
    concealed part), the reading of x less the copies the hand already holds: the
    winning tile, not a tile of its kind inside the hand. The plain reading without
    such a view.
    """
    k = CLASSES.index(x)
    full = [o for o in _reveal_views(win) if o["count"] == len(concealed) + 1]
    if not full:
        return _reveal_mass(win, x)
    held = sum(1 for t in concealed if t == x)
    return sum(max(0.0, sum(sl["p"][k] for sl in o["slots"]) - held) for o in full)


def _five_swaps(
    win: Win, concealed: list[str], win_tiles: list[str], melds: list[dict]
) -> list[tuple]:
    """Find red-five substitutions that score the site's value.

    The winner's fives read plain for red or the reverse, with the winning tiles that
    then fit: [(five, swapped, concealed after, winning tile)]. One red of a suit
    exists: a red seen anywhere else (a meld, a pond, an indicator, the winning tile)
    cannot be in the hand.
    """
    score_of, _ = scorer(win, melds)
    seen = (
        [x for m in melds for x in m["tiles"]]
        + win.dora
        + win.hand.ura
        + [sl.tile for sls in win.logs.values() for sl in sls]
    )
    out = []
    for i, x in enumerate(concealed):
        alt = rules.OTHER_FIVE.get(x)
        if alt is None or x in concealed[:i]:
            continue
        conc2 = [*concealed[:i], alt, *concealed[i + 1 :]]
        if alt in rules.PLAIN_OF and (alt in seen or conc2.count(alt) > 1):
            continue
        out += [
            (x, alt, conc2, w)
            for w in win_tiles
            if not (alt in rules.PLAIN_OF and w == alt)
            and fits(win, score_of(conc2, w, win.dora))
        ]
    return out


def _winning_tile_from_the_site(
    win: Win, search: Search, p: _Proposal, jw: int, report: Report
) -> _Proposal:
    """Choose an unseen tsumo tile with the site's score.

    The tile that makes the hand win with the site's score; among several, the one
    the reveal frames show most. A five read plain for red (or the reverse) changes
    the han by one: with no tile fitting as read, each swap is tried with each tile.
    """
    score_of, _ = scorer(win, p.melds)
    winner, model = win.hand.winning_seat, search.model
    concealed = p.concealed
    used = Counter(
        rules.plain(x)
        for x in concealed
        + [x for m in p.melds for x in m["tiles"]]
        + win.dora
        + win.hand.ura
    )
    for sls in win.logs.values():
        for sl in sls:
            used[rules.plain(sl.tile)] += 1
    tiles = [
        x
        for x in CLASSES
        if x not in ("X", "none") and used[rules.plain(x)] < MAX_PLAIN_TILE_COPIES
    ]
    cands = [x for x in tiles if fits(win, score_of(concealed, x, win.dora))]
    if not cands:
        swaps = _five_swaps(win, concealed, tiles, p.melds)
        if swaps:
            # several fives could be the red one: the reveal frames rank them
            opts = sorted(
                sorted({(a, b) for a, b, _, _ in swaps}),
                key=lambda ab: -(_reveal_mass(win, ab[1]) - _reveal_mass(win, ab[0])),
            )
            x, alt = opts[0]
            concealed = next(c2 for a, b, c2, _ in swaps if (a, b) == (x, alt))
            cands = sorted({x2 for a, b, _, x2 in swaps if (a, b) == (x, alt)})
    extra = {x: _reveal_extra(win, x, concealed) for x in cands}
    cands.sort(key=lambda x: -extra[x])
    if not cands:
        return p
    old_bounds = list(model.bound_hands)
    _bind_winning_hand(win, model, concealed, cands[0], jw)
    sol = search.solve(prior=p.sol)
    if not sol.ok or sol.objective - p.sol.objective > NEXT_BUDGET:
        model.bound_hands = old_bounds
        apply_choices(p.sol, model, search.turns, search.unknown_kans)
        return p
    win_tile = cands[0]
    concealed = list(sol.hands.get((winner, jw), []))
    if win_tile in concealed:
        concealed.remove(win_tile)
    report.diagnostics.append(
        f"unseen winning tile: {win_tile} wins with the site's score"
        + (f" (also {', '.join(cands[1:])})" if cands[1:] else "")
    )
    if len(cands) > 1 and extra[cands[0]] < 2 * extra[cands[1]]:
        report.items.append(
            questions.uncertain_winning_tile(
                winner, win.t_last, concealed, cands, win_tile
            )
        )
    return _Proposal(
        sol, concealed, win_tile, score_of(concealed, win_tile, win.dora), p.melds
    )


def _red_five_from_the_site(
    win: Win, search: Search, p: _Proposal, j_hand: int, report: Report
) -> Solution:
    """Resolve a plain/red five with the site's han.

    The swap that fits, ranked by the reveal frames, is bound as the winner's hand
    and the hand solved again, so the draws that brought the five follow. A swap the
    draws cannot deliver is left as read.
    """
    swaps = _five_swaps(win, p.concealed, [p.tile], p.melds)
    if not swaps:
        return p.sol
    model = search.model
    x, alt, conc2, _ = max(
        swaps, key=lambda sw: _reveal_mass(win, sw[1]) - _reveal_mass(win, sw[0])
    )
    _bind_winning_hand(win, model, conc2, p.tile, j_hand)
    sol = search.solve(prior=p.sol)
    if not sol.ok or sol.objective - p.sol.objective > NEXT_BUDGET:
        model.bound_hands = []
        apply_choices(p.sol, model, search.turns, search.unknown_kans)
        why = (
            "no draw can bring it"
            if not sol.ok
            else f"the video reads it otherwise ({sol.objective - p.sol.objective:.1f} "
            "costlier)"
        )
        report.diagnostics.append(
            f"the winner's {x} as {alt} would score the site's value, but {why}: "
            "kept as read"
        )
        return p.sol
    others = sorted({b for _, b, _, _ in swaps if b != alt})
    report.diagnostics.append(
        f"the winner's {x} is {alt}: the site's han need it"
        + (f" (also possible: {', '.join(others)})" if others else "")
    )
    return sol


# ---- the final check ---------------------------------------------------------------


def check_score(win: Win, search: Search, sol: Solution, report: Report) -> dict | None:
    """Score the final reconstruction against the site record.

    The accepted solution is scored, including any meld choices it changed: a
    reconciliation proposal's predicted score is never the final record.
    """
    result, winner = win.hand.result, win.hand.winner
    if not sol.ok or not winner or result.outcome not in ("ron", "tsumo"):
        return None
    wh = winning_hand(win, search.model, sol)
    if not wh.tile:
        return None
    melds = scoring_melds(win.live_calls, winner)
    score_of, context = scorer(win, melds)
    sc = score_of(wh.concealed, wh.tile, win.dora)
    _score_review(win, search.model, sol, sc, wh, melds, report)
    return {
        "ok": sc.ok,
        "han": sc.han,
        "fu": sc.fu,
        "yaku": [y.to_dict() for y in sc.yaku],
        "error": sc.error,
        "site": [result.han, result.fu],
        "match": matches_site(sc, result.han, result.fu),
        "concealed": wh.concealed,
        "win_tile": wh.tile,
        "melds": melds,
        "context": context,
    }


def _score_review(
    win: Win,
    model: HandModel,
    sol: Solution,
    sc: ScoreResult,
    wh: WinningHand,
    melds: list[dict],
    report: Report,
) -> None:
    """Ask about a score the site disputes, or about an uncertain winning tile."""
    hand = win.hand
    result, winner = hand.result, hand.winning_seat
    jw = len(model.turns[winner])
    ura_explains = (
        winner in win.riichi.seats
        and not hand.ura
        and sc.ok
        and sc.han is not None
        and result.han is not None
        and sc.han < result.han
    )
    if not matches_site(sc, result.han, result.fu) and not ura_explains:
        item = {
            "kind": "result",
            "seat": winner,
            "t": win.t_last,
            "tiles": wh.concealed,
            "win_tile": wh.tile,
            "site": [result.han, result.fu],
            "same_payment": False,
        }
        same: bool | None = None
        if sc.ok and sc.han and sc.fu is not None and result.han:
            # the site's han/fu can be a scorer's slip, the costless one most of all:
            # a pair that pays the same
            how = {"dealer": winner == rules.DEALER, "tsumo": result.outcome == "tsumo"}
            same = payment(sc.han, sc.fu, **how) == payment(
                result.han, result.fu or 0, **how
            )
            item.update(han=sc.han, fu=sc.fu, same_payment=same)
        item["text"] = questions.score_mismatch(sc, same_payment=same)
        report.items.append(item)
    elif (
        result.outcome == "tsumo"
        and (winner, jw) not in model.facts.draws
        and (certificate := sol.certificates.get(("draw", winner, jw))) is not None
        and certificate.state == "ambiguous"
    ):
        # the tile the log ends on, read almost as well as another that scores the same
        alt = certificate.runner_up
        if (
            alt
            and alt != wh.tile
            and wh.tile
            and fits(win, scorer(win, melds)[0](wh.concealed, alt, win.dora))
        ):
            report.items.append(
                questions.uncertain_winning_tile(
                    winner, win.t_last, wh.concealed, [alt], wh.tile
                )
            )


def check_tenpai_and_ura(
    hand: Hand,
    model: HandModel,
    sol: Solution,
    turns: list[Turn],
    riichi: Riichi,
    report: Report,
) -> None:
    """Ask about tenpai the site disputes, and for the ura of a riichi win.

    At an exhaustive draw the site's tenpai seats must be tenpai with their final
    states; a riichi win reveals ura indicators, without which the han cannot be
    checked.
    """
    result = hand.result
    t_last = turns[-1].t if turns else hand.t1
    if sol.ok and result.outcome == "draw":
        for s in rules.SEATS:
            tiles = sol.hands.get((s, len(model.turns[s]) - 1), sol.haipai.get(s, []))
            tp = is_tenpai(list(tiles))
            if tp != (site_seat_name(s, hand.entry) in result.tenpai):
                report.items.append(questions.tenpai(s, t_last, list(tiles), tenpai=tp))
    winner = hand.winner
    if (
        result.outcome in ("ron", "tsumo")
        and winner in riichi.seats
        and winner is not None
        and not hand.facts.get("ura")
    ):
        report.items.append(questions.ura(winner, t_last))
