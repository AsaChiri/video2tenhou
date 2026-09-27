"""Stage 5: observations of one hand -> events, solver margins, review items (DESIGN.md 4.8 `decode.py`).

ponds -> discard logs; meld events anchored on the discards they took -> calls; the dead-wall row -> dora and
kans; the reviewer's facts; turns -> the sequence; solver -> haipai, draws and the calls' tiles, with the site's
result as a constraint; scoring -> the han / fu check against the site record; review -> the few questions
that settle the hand, and the confidence of every decision. Each step is one method of `HandDecoder`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np

from ..record import Game, HandResult
from ..paths import DATA_DIR
from ..train.data import CLASSES
from . import dense, rules
from .hand import corner_of, in_window, seat_of, site_seat, site_seat_name
from .calls import CallAnchor, meld_events
from .indicators import KAN_MATCH, indicator_row, reconcile_kans
from .melds import Call, meld_options
from .ponds import PondSlot, play_window, track_pond
from . import pond_evidence
from .review import MARGIN_REVIEW, changed_discard, confidence_rows, draws_to_reread, facts_for_hand, load_facts, low_margin, ranked, turn_key, unseen_draw, uncertain_tiles, uncertain_discards
from .scoring import is_tenpai, matches_site, payment, score_hand, score_text
from .solver import (Culprit, HandEvidence, HandModel, SeatTurn, Solution, TI, TILES, WinSpec, diagnose, hand_evidence,
                     posterior_to_tiles)
from .turns import Turn, assign_calls, merge

RIICHI_CLOSE = 0.5          # the two readings of a riichi on a called tile solve this close: the reviewer is asked
SWAP_FIVE = {"5m": "0m", "5p": "0p", "5s": "0s", "0m": "5m", "0p": "5p", "0s": "5s"}   # a five read as the other one
NEXT_HANDS = 6              # next-best reconstructions tried for the site's score ...
NEXT_BUDGET = 20.0          # ... at most this much costlier than the best (a re-reading beyond it is not the evidence's)
DECODER_VERSION = 11        # bump when reconstruction semantics change; old cached logs must be rebuilt


# ---------------------------------------------------------------------------------------------------------
# pieces used by the decoder (and by the tests)
# ---------------------------------------------------------------------------------------------------------

def apply_choices(sol: Solution, model: HandModel, turns: list[Turn], unknown: set[int]) -> None:
    """The solver's choices about the calls go into them before anything scores or writes them: which legal
    composition a call is (its type and tiles), the tile of an ankan the camera never named, and which five
    (plain or red) a kakan of fives added."""
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
            c.tiles = c.tiles[:3] + [sol.kan_added[key]]


def concealed_size(melds: list[dict], drawn_left_in: bool = False) -> int:
    """How many tiles the winner holds beside their melds and beside the winning tile (section 1).

    A hand is 13 tiles; each meld set takes three of them out (a kan's fourth tile is the extra tile the
    kan adds, not a fourteenth hand tile). A kakan is the pon it grew from: listed beside that pon it is not a
    set of its own. `drawn_left_in` is for a tsumo whose winning tile was never named: it is still in the list.
    """
    pons = {rules.plain(m["tiles"][0]) for m in melds if m["type"] == "pon"}
    sets = sum(1 for m in melds if m["type"] != "kakan" or rules.plain(m["tiles"][0]) not in pons)
    return 13 - 3 * sets + (1 if drawn_left_in else 0)


def scoring_melds(calls: list[Call], seat: str) -> list[dict]:
    """A seat's melds for the scoring library: an ankan's face-down tiles are its kind, any kan of fives is all
    four fives (three plain and the red one) whatever the camera read of them, and a kakan replaces the pon it
    grew from (it is that pon with its fourth tile, not a meld beside it)."""
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


def over_count(turns: list[Turn], calls: list[Call], dora: list[str], entry: dict) -> list[dict]:
    """Kinds that appear more than four times among discards, meld tiles and indicators, with every source."""
    src: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        if t.slot is not None:
            src[rules.plain(t.slot.tile)].append({"kind": "discard", "seat": t.seat, "t": t.t, "tile": t.slot.tile, "i": t.i,
                                                  "corner": corner_of(entry, t.seat), "box": list(t.slot.xyxy) if t.slot.xyxy else None,
                                                  "pos": [t.slot.row, t.slot.index],
                                                  "call": {"seat": t.call.seat, "type": t.call.type, "t": t.call.t_first,
                                                           "corner": corner_of(entry, t.call.seat)} if t.call else None,
                                                  "t_pic": min(t.slot.t_first + 2.0, (t.slot.t_first + t.slot.t_last) / 2.0)
                                                  if t.slot.t_last > t.slot.t_first else t.slot.t_first})
    for c in calls:
        per_kind: dict[str, int] = defaultdict(int)
        if c.type == "kakan":
            per_kind[rules.plain(c.tiles[-1])] += 1      # only the added tile: the pon below it is counted already
        else:
            for k, tile in enumerate(c.tiles):
                if k == c.called_pos and c.type in ("chi", "pon", "kan"):
                    continue                             # the called tile is one of the discards above
                if tile not in ("?", "X"):
                    per_kind[rules.plain(tile)] += 1
        for kind, n in per_kind.items():
            src[kind].append({"kind": "meld", "seat": c.seat, "t": c.t_first, "tile": kind, "copies": n, "corner": corner_of(entry, c.seat),
                              "meld": c.type + " " + "".join(c.tiles), "type": c.type, "tiles": c.tiles})
    for d in dora:
        src[rules.plain(d)].append({"kind": "indicator", "tile": d})
    out = []
    for kind, lst in src.items():
        count = sum(x.get("copies", 1) for x in lst)
        if count > 4:
            out.append({"tile": kind, "count": count, "sources": sorted(lst, key=lambda x: x.get("t", 0))})
    return out


def seat_turns_of(turns: list[Turn], seat: str, dealer: str) -> tuple[list[SeatTurn], dict[int, int]]:
    """SeatTurns of one seat in order, plus melds_before[j] = melds laid up to state j (the hand after turn j)."""
    out: list[SeatTurn] = []
    melds_before: dict[int, int] = {-1: 0}
    n_melds = 0
    j = 0
    for turn_index, t in enumerate(turns):
        if t.seat != seat:
            continue
        kind = t.kind
        removed: list[str] = []
        rinshan = False
        options: list = []
        if kind == "call" and t.own_call is not None:
            c = t.own_call
            removed, options = _hand_tiles(c)
            n_melds += 1
        kan = kan_tile = None
        two = False
        if kind == "kan" and t.own_call is not None:
            c = t.own_call
            known = [x for x in c.tiles if x not in ("?", "X")]
            kan = c.type if c.type in ("ankan", "kakan") else "daiminkan"
            kan_tile = rules.plain(known[0]) if known else None
            if c.type in ("ankan", "kakan"):
                two = True                           # normal draw, kan, rinshan draw
            else:
                removed, options = _hand_tiles(c)
            if c.type != "kakan":
                n_melds += 1
            rinshan = True
            if t.slot is None:
                # the winner's kan after its last discard: the rinshan draw is the winning draw (the final draw
                # variable), so this turn holds the normal draw only (ankan / kakan) or no draw (daiminkan)
                two = False
                if kan == "daiminkan":
                    kind = "call"
        if j == 0 and seat == dealer and kind in ("draw", "kan"):
            # the dealer's 14 are one set: its first turn has no draw of its own (a first-turn ankan still has
            # its rinshan draw)
            kind = "first"
            two = kan is not None
        predecessor = turns[turn_index - 1] if turn_index else None
        draw_min = (predecessor.slot.t_window[0]
                    if predecessor is not None and predecessor.slot is not None and not predecessor.virtual else None)
        out.append(SeatTurn(j, kind, t.slot.tile if t.slot else None, t.slot.t_window[0] if t.slot else t.t, t.t,
                            removed=removed, riichi=t.riichi, rinshan=rinshan,
                            discard_p=posterior_to_tiles(t.slot.p / max(t.slot.p.sum(), 1e-9)) if t.slot is not None else None,
                            kan=kan, kan_tile=kan_tile, two_draws=two, meld_options=options, taken=t.call is not None,
                            t_draw_min=draw_min))
        melds_before[j] = n_melds
        j += 1
    melds_before[j] = n_melds
    return out, melds_before


def _hand_tiles(c: Call) -> tuple[list[str], list]:
    """The tiles a call takes from the hand: fixed (a reviewer's meld, or a call with one composition), or the
    solver's choice among the legal compositions [(tiles, cost)]."""
    if c.options and not c.human and len(c.options) > 1:
        return [], [(o.hand, o.cost) for o in c.options]
    return [x for i, x in enumerate(c.tiles) if i != c.called_pos], []


def sanitize(o):
    """Replace NaN / inf floats (not valid JSON for browsers) by None / a large number."""
    if isinstance(o, float):
        if o != o:
            return None
        if o in (float("inf"), float("-inf")):
            return 1e9 if o > 0 else -1e9
        return o
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


# ---------------------------------------------------------------------------------------------------------
# the decoder
# ---------------------------------------------------------------------------------------------------------


class HandDecoder:
    """One hand, step by step. `models` = (detector, classifier, video path, calibration) enables the dense
    reads (dense.py); without them the decode uses the calm observations only."""

    def __init__(self, entry: dict, obs: dict[str, list[dict]], result: HandResult, facts: dict, *,
                 time_limit: float, models, work_dir: Optional[Path]):
        self.site_han_fu = (result.han, result.fu)
        if facts.get("site_score"):
            # the reviewer confirmed the site's han/fu wrong: theirs stand for it everywhere, the deltas stay (4.8 scoring.py)
            result = dataclasses.replace(result, han=facts["site_score"]["han"], fu=facts["site_score"]["fu"])
        self.entry, self.obs, self.result, self.facts = entry, obs, result, facts
        self.time_limit, self.models, self.work_dir = time_limit, models, work_dir
        self.dealer = rules.DEALER
        self.winner = site_seat(result.winner, entry)
        self.loser = site_seat(result.loser, entry)
        self.tsumo_winner = self.winner if result.outcome == "tsumo" else None
        self.ura = list(facts.get("ura") or [])
        self.pond_obs = {r: v for r, v in obs.items() if r.startswith("pond:")}
        self.problems: list[str] = []
        self.items: list[dict] = []
        self.guessed_riichi: set[str] = set()
        self.riichi_alternatives: list[tuple[Turn, Turn]] = []
        self.repaired: dict[tuple[str, int], str] = {}        # (seat, j) -> the kind a discard was re-read as
        if facts.get("site_score"):
            self.problems.append(f"the site's {self.site_han_fu[0]}/{self.site_han_fu[1]} is wrong (the reviewer): the hand is "
                                 f"checked against, and the log written with, {result.han}/{result.fu}")

    # ---- 1. the window and the discards ------------------------------------------------------------------

    def window(self) -> None:
        """Restrict observations to the table clearing and play interval of this hand."""
        e = self.entry
        self.t0, self.t1 = play_window(self.pond_obs, e["t_start"], e["t_end"],
                                       anchor=tuple(e["t_overlay"]) if e.get("t_overlay") else None)

    def discards(self) -> None:
        """Track each pond, then apply reviewed identities and missing-discard facts."""
        self.logs: dict[str, list[PondSlot]] = {}
        for corner in ("TL", "TR", "BL", "BR"):
            self.logs[seat_of(self.entry, corner)] = track_pond(in_window(self.pond_obs[f"pond:{corner}"], self.t0, self.t1))
        # the reviewer named the tile of a pond slot (the seat's slot nearest the given time)
        for f in self.facts.get("discard", []):
            mine = self.logs.get(f["seat"], [])
            if mine:
                sl = min(mine, key=lambda x: abs(x.t_first - f["t"]))
                if abs(sl.t_first - f["t"]) <= 10:
                    sl.p = np.zeros(len(CLASSES))
                    sl.p[CLASSES.index(f["tile"])] = 1.0
        # a discard the reader never saw (the reviewer names seat, time and tile)
        for f in self.facts.get("missing_discard", []):
            mine = self.logs[f["seat"]]
            if any(rules.plain(sl.tile) == rules.plain(f["tile"]) and abs(sl.t_first - f["t"]) < 15 for sl in mine):
                # a later re-read found it: the fact would double it
                self.problems.append(f"missing-discard fact {f['tile']} of {f['seat']} at {f['t']:.0f}s: the pond now shows it, fact not applied")
                continue
            pp = np.zeros(len(CLASSES))
            pp[CLASSES.index(f["tile"])] = 1.0
            vs = PondSlot(9000 + len(mine), -1, -1, pp, f["t"], (f["t"] - 5.0, f["t"]), f["t"], 2, 0.0)
            mine.insert(sum(1 for sl in mine if sl.t_first <= f["t"]), vs)
        times = [sl.t_first for sls in self.logs.values() for sl in sls]
        self.first_discard = min(times, default=self.t0)
        self.last_discard = max(times, default=None)

    # ---- 2. the calls ---------------------------------------------------------------------------------

    def calls(self) -> None:
        """Meld events anchored on the discards they took (calls.py): the camera says that a seat laid a meld,
        the pond which discard it took; with nothing taken, only a kan pattern is a call."""
        events = meld_events(self.entry, self.obs, self.t0, self.t1)
        anchor = CallAnchor(self.logs, self.entry, self.models, self.work_dir, self.t0, self.problems, self.obs)
        self.calls = anchor.anchor(events, self.first_discard, self.last_discard, self.tsumo_winner)
        # meld events with strong camera evidence, a hand that says a meld was laid, and no discard found: a pond
        # read may have missed it
        self.unanchored = [e for e in events if e.type in ("chi", "pon") and e.seen >= 5
                           and not any(c.seat == e.seat and abs(c.t_first - e.t_first) < 5 for c in self.calls)
                           and not anchor.reread(e, self.calls)]

    def meld_facts(self) -> None:
        """The reviewer's meld facts, after everything the program inferred (a fact always wins): "this meld is
        <type> <tiles> from <source>" replaces the call of that seat nearest its time and suppresses the seat's
        other calls within 40 s; "this meld does not exist" removes the nearest call (of the given type)."""
        for f in self.facts.get("meld", []):
            near = [c for c in self.calls if c.seat == f["seat"] and abs(c.t_first - f["t"]) < 40]
            typ, tiles, pos = f["type"], list(f["tiles"]), f.get("called_pos")
            if typ in ("kan", "ankan", "kakan") and len(tiles) == 3:
                tiles.append(rules.plain(tiles[0]))            # a kan is four tiles
            if pos is None:
                pos = 0 if typ in ("chi", "pon", "kan") else None
            nearest = min(near, key=lambda c: abs(c.t_first - f["t"])) if near else None
            source = f.get("source") or (nearest.source if nearest else None) or ("kamicha" if typ == "chi" else None)
            t_first, window = (nearest.t_first, nearest.t_window) if nearest else (float(f["t"]), (float(f["t"]) - 8.0, float(f["t"])))
            fact = Call(f["seat"], t_first, window, typ, tiles, pos, source,
                        tiles[pos] if pos is not None and pos < len(tiles) else None, [], 1.0, 0, 9, human=True, anchor="fact")
            self.calls = [c for c in self.calls if c not in near] + [fact]
            if near:
                self.problems.append(f"your meld fact for {f['seat']} at {f['t']:.0f}s ({typ} {''.join(tiles)}) replaces "
                                     + ", ".join(f"{c.type} {''.join(c.tiles)}" for c in near))
        for f in self.facts.get("meld_remove", []):
            cands = [c for c in self.calls if c.seat == f["seat"] and abs(c.t_first - f["t"]) < 40 and (not f.get("type") or c.type == f["type"])]
            if not cands:
                self.problems.append(f"meld-remove fact for {f['seat']} at {f['t']:.0f}s matched no call")
                continue
            gone = min(cands, key=lambda c: abs(c.t_first - f["t"]))
            self.calls = [c for c in self.calls if c is not gone]
            self.problems.append(f"the {gone.type} {''.join(gone.tiles)} of {gone.seat} at {gone.t_first:.0f}s was removed by the reviewer")

    # ---- 3. the dead wall ---------------------------------------------------------------------------------

    def indicators(self) -> None:
        """Read the dead-wall timeline and reconcile its new indicators with kans."""
        # a tsumo winner may have kanned after its last discard (a win on the rinshan draw): its indicator follows
        # the last discard; the discard after a last-turn kan may be sighted before the flipped indicator is
        inds = indicator_row(self.pond_obs, self.t0, self.t1, self.last_discard, t_after=15.0 if self.tsumo_winner else 10.0)
        if self.facts.get("dora"):
            inds = self._named_indicators(inds)
        self.inds = inds
        self.calls = reconcile_kans(inds, self.logs, self.calls, self.obs, self.entry, self.t0, self.problems,
                                    tsumo_winner=self.tsumo_winner)
        self.dora = [v["tile"] for v in inds]

    def _named_indicators(self, inds: list[dict]) -> list[dict]:
        """The reviewer named the indicators in order: each takes the observed entry of that tile when there is one
        (its time and box), else the next observed entry; observed indicators beyond the list stay (kans)."""
        used: set = set()
        fact_tiles = list(self.facts["dora"])
        if len(inds) < len(fact_tiles):
            # a tile named twice in a row is one tile when the wall shows fewer entries than the list (typed twice);
            # with as many entries it is two indicators of one kind, which a kan can reveal
            fact_tiles = [t for i, t in enumerate(fact_tiles) if i == 0 or t != fact_tiles[i - 1]]
        picks: list = []
        for t in fact_tiles:
            k = next((k for k, v in enumerate(inds) if v["tile"] == t and k not in used), None)
            if k is not None:
                used.add(k)
            picks.append(k)
        for i in range(len(fact_tiles)):
            if picks[i] is None:
                picks[i] = next((k for k in range(len(inds)) if k not in used), None)
                if picks[i] is not None:
                    used.add(picks[i])
        named = []
        for t, k in zip(fact_tiles, picks):
            base = inds[k] if k is not None else {"t_first": None, "region": "fact"}
            named.append({**base, "tile": t, "human": True, "conf": 1.0})
        # kan-time facts: the reviewer clicked the discard that followed the kan of an indicator no frame shows
        unplaced = [v for i, v in enumerate(named) if i > 0 and v.get("t_first") is None]
        for v, f in zip(unplaced, sorted(self.facts.get("kan_time", []), key=lambda f: f["t"])):
            near = [(abs(sl.t_first - f["t"]), sl.t_first, s) for s, sls in self.logs.items() for sl in sls
                    if abs(sl.t_first - f["t"]) <= 3 and (not f.get("seat") or s == f["seat"])]
            if not near:
                self.problems.append(f"kan-time fact at {f['t']:.0f}s matches no discard: indicator {v['tile']} stays unplaced")
                continue
            _, tt, s = min(near)
            v["t_first"], v["t_before"], v["kan_maker"] = tt - 0.5, tt - 1.0, s
            self.problems.append(f"indicator {v['tile']}: the reviewer placed its kan before {s}'s discard at {tt:.0f}s")
        for i, v in enumerate(named):
            if i > 0 and v.get("t_first") is None:
                self.problems.append(f"indicator {v['tile']} named by the reviewer was never observed: the time of its kan is unknown")
                self.items.append({"kind": "kan", "text": f"The reviewer named a later dora indicator {v['tile']}, but no frame shows it, so "
                                                          f"the kan that revealed it has no time. Which discard followed that kan?"})
        # an observed indicator beyond the named ones stays, even of a kind named already: the wall shows it
        return named + [v for k, v in enumerate(inds) if k not in used]

    # ---- 5. the turn sequence -----------------------------------------------------------------------------

    def turn_sequence(self) -> None:
        """Merge ponds and calls, consulting dense reads for unexplained skipped turns."""
        # where the hand ends: a tsumo winner draws the winning tile; after a ron the seat after the loser would play
        end = {"tsumo": self.winner, "ron": rules.next_seat(self.loser) if self.loser else None}.get(self.result.outcome)
        # the live wall less a tile per kan; a tsumo's winning draw is not in the sequence
        kans = sum(1 for c in self.calls if c.type in ("kan", "ankan", "kakan"))
        wall = rules.LIVE_WALL - kans - (1 if self.result.outcome == "tsumo" else 0)
        exhaustive = self.result.outcome == "draw"
        turns, order_problems = merge(self.logs, self.calls, self.dealer, end, wall, exhaustive)
        if self.models is not None and any("skipped" in p for p in order_problems):
            if dense.skipped_turns(order_problems, turns, self.logs, self.entry, self.models, self.work_dir, self.t0, self.t1, self.problems):
                turns, order_problems = merge(self.logs, self.calls, self.dealer, end, wall, exhaustive)
        self.turns, self.order_problems = turns, order_problems
        # a call the turn order shows and no camera did: its tiles are the solver's choice among the legal melds
        for t in turns:
            c = t.call
            if c is not None and c.anchor == "hidden" and c not in self.calls:
                c.options = meld_options(c.called_tile, c.source, [], four=False)
                self.calls.append(c)

    def end_of_hand(self) -> None:
        """Align the final turn with the site result and enforce the live-wall budget."""
        self._after_the_ron()
        for p in self.order_problems:
            self.problems.append(p)
            m = re.match(r"turn (\d+): no discard of ([ESWN])", p)
            if m:
                # a turn whose discard no read found: the log cannot be written without it
                i = int(m.group(1))
                t = self.turns[i].t if i < len(self.turns) else (self.turns[-1].t if self.turns else self.t1)
                self.items.append({"kind": "order", "seat": m.group(2), "t": t, "i": i,
                                   "text": f"{m.group(2)} must have discarded around {t:.0f} s (its turn came), but no read of its pond "
                                           f"found the tile, not even a dense one. Which tile did {m.group(2)} discard?"})
        self._winners_last_kan()
        self._live_calls()
        self._wall()
        self._unexplained_removals()

    def _after_the_ron(self) -> None:
        """A ron ends the hand on the loser's discard (section 1): nobody discards after it, and tiles reaching a
        pond later are the reveal and the clearing. Turns the merge placed after it that happened before it are
        real turns in the wrong place: they stay (the winner's hand is read by time) and the order is asked."""
        if self.result.outcome != "ron" or self.loser not in rules.SEATS:
            return
        mine = [t for t in self.turns if t.seat == self.loser and t.slot is not None]
        if not mine:
            return
        win_turn = mine[-1]
        after = self.turns[self.turns.index(win_turn) + 1:]
        late = [t for t in after if t.t > win_turn.t + 1.0]
        if late:
            self.problems.append(f"{len(late)} tile(s) seen in the ponds after {self.loser}'s winning discard at {win_turn.t:.0f}s "
                                 f"are not turns (the reveal and the clearing): dropped")
            self.turns = [t for t in self.turns if t not in late]
            for i, t in enumerate(self.turns):
                t.i = i
        early = [t for t in after if t.t <= win_turn.t + 1.0]
        if early:
            listed = ", ".join(f"{t.seat}'s discard at {t.t:.0f} s" for t in early)
            self.problems.append(f"{len(early)} turn(s) the merge placed after {self.loser}'s winning discard at {win_turn.t:.0f}s "
                                 f"happened before it ({listed}): the order is wrong but the turns are kept")

    def _winners_last_kan(self) -> None:
        """A kan after the tsumo winner's last discard (the win is the rinshan draw): no discard follows it, so the
        merge left it on no turn (ankan / kakan) or pending on the last discard (daiminkan); it is the winner's
        last turn, one without a discard."""
        if not self.tsumo_winner:
            return
        attached = {id(t.own_call) for t in self.turns if t.own_call}
        my_last = max((t.t for t in self.turns if t.seat == self.tsumo_winner), default=self.t0)
        for c in self.calls:
            if c.seat != self.tsumo_winner or id(c) in attached:
                continue
            if c.type in ("ankan", "kakan") and c.t_window[1] > my_last:
                t_kan = max(c.t_window[1], my_last + 0.5)
            elif c.type == "kan" and self.turns and self.turns[-1].call is c:
                t_kan = max(c.t_window[1], self.turns[-1].t + 0.5)
            else:
                continue
            self.turns.append(Turn(len(self.turns), self.tsumo_winner, "kan", None, t_kan, own_call=c))
            self.problems.append(f"{c.type} by {self.tsumo_winner} after its last discard at {t_kan:.0f}s: the winning tile is the rinshan draw")
            break

    def _live_calls(self) -> None:
        """A call counts only when the caller has a turn that owns it: that turn is where the solver takes the
        called tiles out of their hand (otherwise the winner comes out with 13 concealed tiles and a meld)."""
        owned = {id(t.own_call) for t in self.turns if t.own_call is not None}
        self.live_calls = [c for c in self.calls if id(c) in owned]
        for c in self.calls:
            if id(c) not in owned:
                self.problems.append(f"the {c.type} {''.join(c.tiles)} of {c.seat} at {c.t_first:.0f}s is on no turn of {c.seat} "
                                     f"(the discard that follows a call was not found): not counted for the hand size or the scoring")
        for e in self.unanchored:
            # the camera shows a meld clearly, and no read found the discard it took
            self.items.append({"kind": "call", "seat": e.seat, "t": e.t_first, "tile": e.tiles[e.called_pos or 0],
                               "type": e.type, "tiles": e.tiles, "source": e.source,
                               "text": f"{e.seat}'s meld camera shows a {e.type} of {' '.join(e.tiles)} from {e.t_first:.0f} s, but no "
                                       f"pond shows the discard it took, not even in a dense read. What is this meld, and whose "
                                       f"discard did it take?"})

    def _wall_draws(self) -> tuple[int, int]:
        """(wall draws, kans) of the sequence: every draw turn (the dealer's 14th tile included), every turn the
        merge skipped (a missed discard still drew), the winning tsumo draw; each kan moved a live tile into the
        dead wall."""
        draws = sum(1 for t in self.turns if t.kind in ("draw", "kan")) + (1 if self.tsumo_winner else 0)
        draws += sum(1 for p in self.order_problems if "skipped" in p)
        kans = sum(1 for c in self.live_calls if c.type in ("kan", "ankan", "kakan"))
        return draws, kans

    def _wall(self) -> None:
        """Wall draws plus kans never exceed 70 (section 1), and an exhaustive draw is exactly 70."""
        draws, kans = self._wall_draws()
        used = draws + kans
        if used > rules.LIVE_WALL:
            text = (f"The turn sequence needs {draws} wall draws and {kans} kan(s), {used} tiles, but the live wall holds "
                    f"{rules.LIVE_WALL}: the sequence holds turns that did not happen (tiles of another hand, or misread ponds).")
            self.problems.append(text)
        elif self.result.outcome == "draw" and used < rules.LIVE_WALL:
            self.problems.append(f"The hand ended in an exhaustive draw, which takes all {rules.LIVE_WALL} tiles of the live wall, "
                                 f"but the sequence has {draws} draws and {kans} kan(s): {rules.LIVE_WALL - used} turn(s) are missing.")

    def _unexplained_removals(self) -> None:
        """A tile left a pond (only a call takes a discard) but no call explains it: the next seat played on, so
        either its call went unseen or the removal was misread. A note; the reconstruction keeps the discard."""
        taken = assign_calls(self.live_calls, self.logs)
        for s2, sls in self.logs.items():
            for k, sl in enumerate(sls):
                if sl.t_removed is not None and taken[s2][k] is None and sl.row != -1:
                    self.problems.append(f"{s2}'s {sl.tile} left the pond at {sl.t_removed:.0f}s and no call explains it")

    # ---- 6. riichi --------------------------------------------------------------------------------------------

    def riichi(self) -> None:
        """The site record decides who declared (its deltas carry the sticks); the turned tile says when. A
        reviewer's riichi-turn fact names the declaring discard outright."""
        site = {site_seat(s, self.entry) for s in self.result.riichi}
        if self.facts.get("riichi") is not None:
            if set(self.facts["riichi"]) != site:
                self.problems.append(f"your riichi fact {sorted(self.facts['riichi']) or 'nobody'} contradicts the site record "
                                     f"{sorted(site) or 'nobody'}: the fact is used")
            site = set(self.facts["riichi"])
        self.site_riichi = site
        named = set()
        for f in self.facts.get("riichi_turn", []):
            mine = [t for t in self.turns if t.seat == f["seat"] and t.slot is not None]
            near = min(mine, key=lambda t: abs(t.t - f["t"]), default=None)
            if near is None or abs(near.t - f["t"]) > 3:
                self.problems.append(f"riichi-turn fact for {f['seat']} at {f['t']:.0f}s matches no discard of that seat")
                continue
            for t in mine:
                t.riichi = t is near
            named.add(f["seat"])
        turned = {t.seat for t in self.turns if t.riichi}
        if turned != site:
            # a turned tile of another seat is a tile laid askew, noted but not asked; a site riichi with no turned
            # tile lacks its turn: the last discard of that seat is the guess and the reviewer is asked
            self.problems.append(f"riichi seats from ponds {sorted(turned)} vs site {sorted(site)}")
            for t in self.turns:
                t.riichi = t.riichi and t.seat in site
            for s in site - turned:
                mine = [t for t in self.turns if t.seat == s and t.slot is not None]
                if not mine:
                    continue
                found = dense.turned_tile(s, self.turns, self.entry, self.models, self.work_dir, self.t0, self.problems)
                if found is not None:
                    found.riichi = True
                    continue
                mine[-1].riichi = True
                self.guessed_riichi.add(s)
                src = "Your riichi fact says" if self.facts.get("riichi") is not None else "The site record says"
                self.items.append({"kind": "riichi", "seat": s, "t": mine[-1].t,
                                   "text": f"{src} {s} declared riichi but no turned tile was seen in its pond, not even in a dense "
                                           f"read of the moment each of its tiles was laid; the last discard ({mine[-1].slot.tile} at "
                                           f"{mine[-1].t:.0f}s) is assumed. Which discard was the riichi?"})
        # a turned tile right after the seat's own discard was called: the declaration was that called tile or this one
        for t in self.turns:
            if not t.riichi or t.seat in named or t.seat in self.guessed_riichi:
                continue
            prev = [x for x in self.turns if x.seat == t.seat and x.i < t.i and x.slot is not None]
            if prev and prev[-1].call is not None:
                self.riichi_alternatives.append((t, prev[-1]))

    # ---- 7. haipai and draws ----------------------------------------------------------------------------------

    def pond_replacements(self) -> None:
        """Acquire unresolved pond correspondences before building discard costs.

        Each request is attempted once. Continuous geometry may substitute its
        dense posterior for overlapping sparse evidence, without adding turns or
        calls; failed or ambiguous acquisition remains a specific review item.
        """
        requests = pond_evidence.replacement_requests(self.turns, self.entry, self.facts, self.t0, self.t1)
        if self.models is None:
            return
        for request in requests:
            if request["acquired"] or request["window"] is None:
                continue
            slot = next(t.slot for t in self.turns if t.seat == request["seat"]
                        and t.slot is not None and t.slot.id == request["slot_id"])
            try:
                readings = pond_evidence.read_replacement(request, self.models, self.work_dir)
                consumed = pond_evidence.consume_replacement(slot, request, readings)
            except Exception as ex:  # noqa: BLE001
                self.problems.append(f"dense pond correspondence for {request['seat']} at {request['t']:.0f}s failed: {ex}")
                continue
            if consumed:
                self.problems.append(f"continuous dense pond evidence for {request['seat']} at {request['t']:.0f}s "
                                     "replaced its overlapping sparse observation")

    def solve(self) -> None:
        """Fit legal hands to camera evidence, reread uncertain draws, then attempt repair."""
        model = HandModel(self.dealer, {s: [] for s in rules.SEATS}, self.dora, tsumo_winner=self.tsumo_winner, ura=self.ura)
        self.unknown_kans = {id(c) for c in self.calls if c.type == "ankan" and "?" in c.tiles}
        self.melds_before: dict[str, dict[int, int]] = {}
        for s in rules.SEATS:
            model.turns[s], self.melds_before[s] = seat_turns_of(self.turns, s, self.dealer)
        for corner in ("TL", "TR", "BL", "BR"):
            seat = seat_of(self.entry, corner)
            hev, dev, habit = hand_evidence(seat, model.turns[seat], in_window(self.obs.get(f"hand:{corner}", []), self.t0, self.t1),
                                            self.melds_before[seat], dealer=(seat == self.dealer), wins_by_tsumo=(seat == self.tsumo_winner))
            model.hand_ev += hev
            model.draw_ev += dev
            model.end_prior[seat] = habit
        self._apply_hand_facts(model)
        self._result_constraint(model)
        self.model = model
        sol = self._solve()
        if sol.ok and self.models is not None:
            low = draws_to_reread(sol, model)
            evidence_before = (len(model.hand_ev), len(model.draw_ev))
            dense.draws(low, model, self.turns, self.melds_before, self.entry, self.obs, self.models, self.work_dir, self.t0, self.problems)
            # Partial views also change the objective, even when none pins a draw.
            if (len(model.hand_ev), len(model.draw_ev)) != evidence_before:
                sol = self._solve(prior=sol)
        if not sol.ok:
            sol = self._repair()
        if sol.ok and self.riichi_alternatives:
            sol = self._riichi_on_a_called_tile(sol)
        self.sol = sol

    def kan_indicators(self) -> None:
        """Every kan reveals an indicator (section 1): the log carries 1 + kans of them, always. One that no view
        showed (the dead wall is often outside the overhead crop) is a `dora` question, and the log carries the best
        guess meanwhile: a kind with copies left whose dora keeps the winner's han at the site's, the one with the
        most copies left among those."""
        kans = sorted((c for c in self.live_calls if c.type in ("kan", "ankan", "kakan")), key=lambda c: c.t_first)
        missing = 1 + len(kans) - len(self.dora)
        if missing <= 0:
            return
        sol, seen = self.sol, list(self.dora)
        used = Counter(x for x in [*(x for h in sol.haipai.values() for x in h), *sol.draws.values(), *sol.draws2.values(),
                                   *self.dora, *self.ura] if x in rules.KINDS or x in rules.REDS)
        score_of = None
        if sol.ok and self.winner and self.result.outcome in ("ron", "tsumo"):
            concealed, win_tile, _, _ = self._winning_hand(sol)
            if win_tile:
                scorer = self._scorer(scoring_melds(self.live_calls, self.winner))
                score_of = lambda dora: scorer(concealed, win_tile, dora)       # noqa: E731
        base = score_of(self.dora) if score_of else None

        def rank(x: str) -> tuple:
            left = rules.max_count(x) - used[x]
            if score_of is None:
                return (-left, rules.KINDS.index(x))
            sc = score_of(self.dora + [x])
            return (not self._fits(sc), sc.han != base.han, -left, rules.KINDS.index(x))

        lost = bool(self.facts.get("lost_dora"))
        guesses = []
        for _ in range(missing):
            x = min((x for x in rules.KINDS if used[x] < rules.max_count(x)), key=rank)
            used[x] += 1
            self.dora.append(x)
            self.inds.append({"tile": x, "t_first": None, "region": None, "conf": 0.0, "lost": True, "human": lost})
            guesses.append(x)
        # the kans no seen indicator explains (the same pairing as reconcile_kans)
        explained: set = set()
        for ind in self.inds[1:len(seen)]:
            t = ind.get("t_first")
            k = next((c for c in kans if t is not None and id(c) not in explained
                      and c.t_window[0] - KAN_MATCH <= t <= c.t_first + KAN_MATCH), None)
            if k is not None:
                explained.add(id(k))
        bare = [c for c in kans if id(c) not in explained]
        if lost:
            self.problems.append(f"dora indicator(s) {' '.join(guesses)}: no view shows them and the reviewer cannot tell; "
                                 f"written as the rules' guess")
            return
        what = "; ".join(f"the {c.type} of {rules.plain(next((x for x in c.tiles if x not in ('X', '?')), '?'))} by {c.seat} "
                         f"at {c.t_first:.0f}s" for c in bare) or "the kans of this hand"
        head = "No dora indicator was seen, and " if not seen else ""
        self.items.append({"kind": "dora", "t": bare[0].t_first if bare else (self.last_discard or self.t1),
                           "tiles": seen, "guess": guesses,
                           "text": f"{head}{what} revealed a dora indicator that no view shows (the dead wall is outside "
                                   f"the overhead crop). The log carries {' '.join(guesses)} as a guess. Enter all the dora "
                                   f"indicators in order: {' '.join(seen) + ' (seen), then ' if seen else ''}the kan's."})

    def unseen_tiles(self) -> None:
        """Every tile the rules chose and nothing showed is a question (section 6, `lost`), with the choice as the
        guess: a draw no frame covers, the caller's tiles of a call the turn order implies and no camera read, the
        tile of an ankan no camera named. A reviewer's `lost` fact (Can't tell) closes a draw's question."""
        sol, model = self.sol, self.model
        if not sol.ok:
            return                                  # the conflict question comes first
        lost_keys = {k for k in (turn_key(model, f, 3.0) for f in self.facts.get("lost", [])) if k is not None}
        for (s, j), tile in sorted(sol.draws.items()):
            if tile is None or (s, j) in lost_keys or (s, j) in model.facts.draws or not unseen_draw(model, sol, s, j):
                continue
            st = model.turns[s][j] if j < len(model.turns[s]) else None
            t = st.t_discard if st else (self.turns[-1].t if self.turns else self.t1)
            self.items.append({"kind": "draw", "seat": s, "j": j, "t": t, "tile": tile, "runner_up": sol.runner_up.get((s, j)),
                               "margin": sol.margins.get((s, j), 0.0),
                               "text": f"No frame shows the tile {s} drew in the turn ending at {t:.0f}s, and nothing later "
                                       f"pins it: the log carries {tile}, the rules' guess."})
        for c in self.live_calls:
            if c.human:
                continue
            if c.anchor == "hidden":
                why = (f"no camera shows this meld: the turn order says {c.seat} called {c.called_tile} from its "
                       f"{c.source} at {c.t_first:.0f}s, and the log carries {c.type} {' '.join(c.tiles)}, the rules' guess")
            elif id(c) in self.unknown_kans:
                why = (f"no camera names the tile of {c.seat}'s ankan at {c.t_first:.0f}s: the log carries "
                       f"{' '.join(c.tiles)}, the rules' guess")
            else:
                continue
            self.items.append({"kind": "call", "seat": c.seat, "t": c.t_first, "tile": c.called_tile or c.tiles[0],
                               "type": c.type, "tiles": list(c.tiles), "source": c.source,
                               "text": f"{why[0].upper()}{why[1:]}. What was the meld?"})

    def _solve(self, **kw) -> Solution:
        sol = self.model.solve(time_limit=self.time_limit, margin_thr=MARGIN_REVIEW, **kw)
        if sol.ok:
            apply_choices(sol, self.model, self.turns, self.unknown_kans)     # the calls take the solver's choices before scoring
        return sol

    def _result_constraint(self, model: HandModel) -> None:
        """The site record says who won: the winner's final hand is a winning hand; at a draw the tenpai seats are
        tenpai (4.8 "The result as a constraint")."""
        w = self.winner
        if w and self.result.outcome == "tsumo":
            j = len(model.turns[w])
            model.win = WinSpec(w, j, 4 - self.melds_before[w].get(j, 0))
        elif w and self.loser and self.result.outcome == "ron":
            lt = [t for t in self.turns if t.seat == self.loser and t.slot is not None]
            stl = next((x for x in model.turns[self.loser] if lt and x.t_discard == lt[-1].t), None)
            if stl is not None:
                # the winner's hand at the winning discard, by time: turns placed after it that happened before it count
                jwin = sum(1 for t in self.turns if t.seat == w and t.t <= lt[-1].t + 1.0) - 1
                model.win = WinSpec(w, jwin, 4 - self.melds_before[w].get(jwin, 0), ron_from=(self.loser, stl.j))
        elif self.result.outcome == "draw":
            for s in rules.SEATS:
                if site_seat_name(s, self.entry) in self.result.tenpai and model.turns[s]:
                    j = len(model.turns[s]) - 1
                    model.tenpai.append((s, j, 4 - self.melds_before[s].get(j, 0)))

    def _apply_hand_facts(self, model: HandModel) -> None:
        for f in self.facts.get("discard", []) + self.facts.get("missing_discard", []):
            mine = model.turns.get(f["seat"], [])
            if mine and f.get("t") is not None:
                nearest = min(mine, key=lambda turn: abs(turn.t_discard - f["t"]))
                if abs(nearest.t_discard - f["t"]) <= 10.0:
                    # A one-hot posterior alone is still a soft cost in repair
                    # mode. A human identity must remain a hard constraint.
                    nearest.discard, nearest.discard_p = f["tile"], None
        for f in self.facts.get("haipai", []):
            model.facts.haipai[f["seat"]] = f["tiles"]
        for f in self.facts.get("draw", []):
            key = turn_key(model, f, 10.0)
            if key is None:
                self.problems.append(f"draw fact {f['tile']} of {f['seat']} matches no draw turn: ignored")
                continue
            model.facts.draws[key] = f["tile"]
        for f in self.facts.get("final_hand", []):
            # binds only when it is a complete concealed hand at the end (13 - 3 per meld, +1 for a tsumo winner)
            s = f["seat"]
            base = 13 - 3 * sum(1 for c in self.live_calls if c.seat == s and c.type != "kakan")
            tsumo_win = s == self.tsumo_winner
            want = base + (1 if tsumo_win else 0)
            excl = tsumo_win and len(f["tiles"]) == base        # the hand without the winning draw: bound before it
            if not excl and len(f["tiles"]) != want:
                self.problems.append(f"final-hand fact for {s} has {len(f['tiles'])} tiles, expected {want}"
                                     + (f" or {base} without the winning tile" if tsumo_win else "") + ": ignored")
            elif f.get("soft"):
                # a legacy annotation: strong evidence on the final state, not a hard constraint
                last = len(model.turns[s]) - 1 + (1 if (tsumo_win and not excl) else 0)
                e = np.zeros(len(TILES))
                for t in f["tiles"]:
                    if t in TI:
                        e[TI[t]] += 1
                model.hand_ev.append(HandEvidence(s, last, False, e, 3.0, self.t1, self.t1))
            else:
                model.facts.final[s] = f["tiles"]
                if excl:
                    model.facts.final_excl[s] = True

    def _repair(self) -> Solution:
        """No legal reconstruction with every discard as read: find the cheapest re-readings of the ponds that make
        the hand legal (every discard a choice over all kinds, costed by its posterior), take them as the
        discards, and solve again. Only when no re-reading helps is the hand a conflict."""
        self.model.repair = True
        found = self.model.solve(time_limit=self.time_limit, margins=False)
        self.model.repair = False
        if found.ok:
            for (s, j), tile in found.discards.items():
                st = self.model.turns[s][j]
                self.problems.append(f"{s}'s discard at {st.t_discard:.0f}s read {st.discard}: the hand is legal only as {tile}, which is used")
                self.repaired[(s, j)] = tile
                st.discard, st.discard_p = tile, None
            sol = self._solve()
            if sol.ok:
                sol.optimal = sol.optimal and found.optimal
                sol.status = "repaired"
                return sol
        culprits = diagnose(self.model)
        self.problems.append("solver: no legal reconstruction: " + ("; ".join(c.text for c in culprits) or "no single decision explains it"))
        self.items.append(self._conflict_question(culprits))
        return found

    def _conflict_question(self, culprits: list[Culprit]) -> dict:
        """The one decision a conflict hinges on, asked with the control that answers it; the over-counted kinds
        with their sources when no single decision explains it."""
        item = {"kind": "conflict", "t": self.turns[-1].t if self.turns else self.t1,
                "over": over_count(self.turns, self.calls, self.dora, self.entry)}
        if not culprits:
            return {**item, "culprit": None,
                    "text": "No reconstruction of this hand is legal, and no single discard, call or fact explains why. The kinds "
                            "below appear more than four times; mark the reading that is wrong."}
        c = culprits[0]
        item.update(culprit=c.kind, seat=c.seat, fact_seat=c.seat if c.kind == "fact" else None, fact_kind=c.fact)
        if c.kind == "meld":
            call = next((t.own_call for t in self.turns if t.seat == c.seat and t.own_call is not None
                         and (st := next((x for x in self.model.turns[c.seat] if x.t_discard == t.t), None)) is not None and st.j == c.j), None)
            if call is not None:
                item.update(t=call.t_first, type=call.type, tiles=call.tiles, source=call.source, tile=call.called_tile,
                            text=f"{c.seat}'s {call.type} of {' '.join(call.tiles)} (from its {call.source or 'own hand'}) at "
                                 f"{call.t_first:.0f} s cannot be: {c.text.split(': ', 1)[-1]}. What is this meld?")
                return item
        if c.kind == "result":
            return {**item, "text": f"{c.seat}'s winning hand cannot be completed from what was read. Enter the hand {c.seat} "
                                    f"revealed (a call or a discard of that seat is read wrong)."}
        if c.kind == "riichi":
            st = self.model.turns[c.seat][c.j]
            return {**item, "t": st.t_discard, "text": f"{c.text}. Which discard was {c.seat}'s riichi?"}
        return {**item, "text": c.text[0].upper() + c.text[1:] + ". Check that fact."}

    def _riichi_on_a_called_tile(self, sol: Solution) -> Solution:
        """The declaration was the called tile or the turned one (section 1): the freeze decides."""
        for turned, called in self.riichi_alternatives:
            sts = self.model.turns[turned.seat]
            a = next(x for x in sts if x.t_discard == turned.t)
            b = next(x for x in sts if x.t_discard == called.t)
            a.riichi, b.riichi = False, True
            alt = self.model.solve(time_limit=self.time_limit, margins=False)
            gap = abs((alt.objective if alt.ok else float("inf")) - sol.objective)
            if alt.ok and alt.objective < sol.objective:
                turned.riichi, called.riichi = False, True
                sol = self._solve(prior=sol)
                chosen = called
            else:
                a.riichi, b.riichi = True, False
                chosen = turned
            if gap < RIICHI_CLOSE:
                self.items.append({"kind": "riichi", "seat": turned.seat, "t": chosen.t,
                                   "text": f"{turned.seat}'s turned tile at {turned.t:.0f}s follows its discard at {called.t:.0f}s that was "
                                           f"called: the declaration was one of the two, and the hand fits both about as well. "
                                           f"{chosen.slot.tile} at {chosen.t:.0f}s is assumed. Which discard was the riichi?"})
            self.problems.append(f"riichi of {turned.seat}: the called discard at {called.t:.0f}s or the turned one at "
                                 f"{turned.t:.0f}s; {'the called one' if chosen is called else 'the turned one'} fits the hand better")
        return sol

    # ---- 8. the result -------------------------------------------------------------------------------------------

    def check_score(self) -> None:
        """Reconcile the winning reconstruction with the site score and review facts."""
        self.score = None
        sol, result, winner = self.sol, self.result, self.winner
        if not sol.ok or not winner or result.outcome not in ("ron", "tsumo"):
            return
        jw = len(self.model.turns[winner])
        concealed, win_tile, drawn_left_in, j_hand = self._winning_hand(sol)
        if result.outcome == "ron" and self.turns and j_hand != jw - 1:
            self.problems.append(f"ron by {winner} on {self.loser}: the hand's last turn is {self.turns[-1].seat}'s at "
                                 f"{self.turns[-1].t:.0f}s, after the winning discard; the winner's hand is read at their turn "
                                 f"{j_hand}, the last one at or before that discard")
        melds = scoring_melds(self.live_calls, winner)
        # the winner's concealed tiles beside the melds and the winning tile must be a hand of 13 (section 1); the
        # solver took the same calls' tiles out of the hand, so a mismatch is the decoder's own inconsistency: a
        # note (the score then fails and asks for the revealed hand)
        want = concealed_size(melds, drawn_left_in)
        if len(concealed) != want:
            self.problems.append(f"the winner's hand does not add up: {len(concealed)} concealed tiles beside {len(melds)} meld(s), "
                                 f"{want} expected. The melds and the turn sequence disagree.")
        if not win_tile:
            return
        score_of = self._scorer(melds)
        sc = score_of(concealed, win_tile, self.dora)
        if result.outcome == "tsumo" and not self._fits(sc) and (winner, jw) not in self.model.facts.draws:
            concealed, win_tile, sc = self._winning_tile_from_the_site(concealed, win_tile, sc, melds, score_of, jw)
        if not self._fits(sc) and winner not in self.model.facts.final:
            concealed, win_tile, sc, melds = self._next_hand_from_the_site(concealed, win_tile, sc, melds, j_hand)
        if not self._fits(sc) and winner not in self.model.facts.final:
            concealed, sc = self._red_five_from_the_site(concealed, win_tile, sc, melds, self._scorer(melds), j_hand)
        if not matches_site(sc, result.han, result.fu) and self.inds and not self.facts.get("dora") and self.inds[0].get("p") \
                and not (winner in self.site_riichi and not self.ura):
            sc = self._dora_from_the_site(concealed, win_tile, sc, score_of)
        self.score = {"ok": sc.ok, "han": sc.han, "fu": sc.fu, "yaku": sc.yaku, "error": sc.error, "site": [result.han, result.fu],
                      "match": matches_site(sc, result.han, result.fu), "concealed": concealed, "win_tile": win_tile, "melds": melds,
                      "context": self.context}
        ura_explains = winner in self.site_riichi and not self.ura and sc.ok and sc.han is not None and sc.han < result.han
        if not self.score["match"] and not ura_explains:
            found = ", ".join(sc.yaku) if sc.yaku else (sc.error or "no yaku")
            item = {"kind": "result", "seat": winner, "t": self.turns[-1].t if self.turns else self.t1,
                    "tiles": concealed, "win_tile": win_tile, "site": [result.han, result.fu], "same_payment": False}
            if sc.ok and sc.han and result.han:
                # the site's han/fu can be a scorer's slip, the costless one most of all: a pair that pays the same
                how = dict(dealer=winner == self.dealer, tsumo=result.outcome == "tsumo")
                mine, site = score_text(sc.han, sc.fu, **how), score_text(result.han, result.fu or 0, **how)
                item.update(han=sc.han, fu=sc.fu, same_payment=payment(sc.han, sc.fu, **how) == payment(result.han, result.fu or 0, **how))
                if item["same_payment"]:
                    item["text"] = (f"The reconstructed winning hand scores {sc.han}/{sc.fu} [{found}], which pays what the site's "
                                    f"{result.han}/{result.fu} pay ({site}): the site's han/fu may be the scorer's slip. Confirm "
                                    f"that the site is wrong, or enter the hand {winner} revealed.")
                else:
                    item["text"] = (f"The reconstructed winning hand scores {sc.han}/{sc.fu} [{found}], {mine}; the site says "
                                    f"{result.han}/{result.fu}, {site}. Enter the hand {winner} revealed, or confirm that "
                                    f"the site is wrong.")
            else:
                item["text"] = (f"The reconstructed winning hand scores {sc.han}/{sc.fu} [{found}]; the site says "
                                f"{result.han}/{result.fu}. Enter the hand {winner} revealed.")
            self.items.append(item)
        elif result.outcome == "tsumo" and (winner, jw) not in self.model.facts.draws \
                and low_margin(self.sol.margins.get((winner, jw))):
            # the tile the log ends on, read almost as well as another that scores the same: the one question
            alt = self.sol.runner_up.get((winner, jw))
            if alt and alt != win_tile and self._fits(self._scorer(melds)(concealed, alt, self.dora)):
                self.items.append({"kind": "result", "seat": winner, "t": self.turns[-1].t if self.turns else self.t1,
                                   "text": f"{winner}'s winning tile is uncertain: {win_tile} or {alt}, which both complete the hand "
                                           f"with the site's {result.han}/{result.fu}, and the reveal reads neither clearly. Which "
                                           f"tile did {winner} win on?",
                                   "tiles": concealed, "win_tile": win_tile, "candidates": [alt]})

    def _winning_hand(self, sol: Solution) -> tuple[list[str], Optional[str], bool, int]:
        """The winner's hand in a solution: its concealed tiles (the winning tile out), the winning tile, whether an
        unknown winning draw is still among them, and the state j that holds the hand (a tsumo: after the winning
        draw; a ron: the winner's last turn at or before the winning discard, by time, not by position in the merged
        sequence)."""
        winner = self.winner
        jw = len(self.model.turns[winner])
        if self.result.outcome == "tsumo":
            hand = sol.hands.get((winner, jw), [])
            win_tile = sol.draws.get((winner, jw))
            concealed = list(hand)
            if win_tile in concealed:
                concealed.remove(win_tile)
                return concealed, win_tile, False, jw
            return concealed, win_tile, bool(hand), jw
        win_tile, jwin = None, jw - 1
        lt = [t for t in self.turns if t.seat == self.loser and t.slot is not None]
        if lt:
            # the winning tile is the loser's last discard as the reconstruction has it
            last = lt[-1]
            jwin = sum(1 for t in self.turns if t.seat == winner and t.t <= last.t + 1.0) - 1
            stl = next((x for x in self.model.turns[self.loser] if x.t_discard == last.t), None)
            win_tile = (sol.discards.get((self.loser, stl.j)) if stl else None) or last.slot.tile
        return list(sol.hands.get((winner, jwin), sol.hands.get((winner, jw - 1), []))), win_tile, False, jwin

    def _bind_winning_hand(self, concealed: list[str], win_tile: str, j_hand: int) -> None:
        """Bind the winner's hand the site's score requires (the concealed part, and for a tsumo the state with the
        winning draw), so the solve that follows, margins and all, keeps it and the draws that brought it follow.
        Not a reviewer's fact: the confidence rows do not mark it human."""
        winner = self.winner
        if self.result.outcome == "tsumo":
            self.model.bound_hands = [(winner, j_hand - 1, list(concealed)), (winner, j_hand, list(concealed) + [win_tile])]
        else:
            self.model.bound_hands = [(winner, j_hand, list(concealed))]

    def _next_hand_from_the_site(self, concealed, win_tile, sc, melds, j_hand):
        """The next-best reconstructions: the winner's hand of each is forbidden in turn (up to NEXT_HANDS, within
        NEXT_BUDGET of the best), and the first whose winning hand scores the site's value is the hand: bound, and
        the hand solved again (margins and all)."""
        winner, result, model = self.winner, self.result, self.model
        best = self.sol.objective
        hand = sorted(self.sol.hands.get((winner, j_hand), []))
        for _ in range(NEXT_HANDS):
            model.forbidden_hands.append((winner, j_hand, hand))
            alt = model.solve(time_limit=self.time_limit, margins=False)
            if not alt.ok or alt.objective - best > NEXT_BUDGET:
                break
            apply_choices(alt, model, self.turns, self.unknown_kans)
            conc2, win2, _, _ = self._winning_hand(alt)
            melds2 = scoring_melds(self.live_calls, winner)
            sc2 = self._scorer(melds2)(conc2, win2, self.dora) if win2 else sc
            if win2 and self._fits(sc2):
                model.forbidden_hands.clear()
                self._bind_winning_hand(conc2, win2, j_hand)
                sol = self._solve(prior=self.sol)
                if sol.ok:
                    self.sol = sol
                    self.problems.append(f"the best reconstruction's winning hand scores {sc.han}/{sc.fu}, the site {result.han}/"
                                         f"{result.fu}: the next best that scores it ({alt.objective - best:.1f} costlier) is "
                                         f"{' '.join(sorted(conc2))} + {win2}")
                    return conc2, win2, sc2, melds2
                model.bound_hands = []
                break
            hand = sorted(alt.hands.get((winner, j_hand), []))
        model.forbidden_hands.clear()
        apply_choices(self.sol, model, self.turns, self.unknown_kans)
        return concealed, win_tile, sc, melds

    def _scorer(self, melds: list[dict]):
        """The scoring function of this win, with its situational yaku from the turn sequence."""
        winner, result = self.winner, self.result
        my_turns = [t for t in self.turns if t.seat == winner]
        riichi_i = next((t.i for t in my_turns if t.riichi), None)
        ippatsu = double_riichi = False
        if winner in self.site_riichi and riichi_i is not None and winner not in self.guessed_riichi:
            after = [t for t in self.turns if t.i > riichi_i]
            ippatsu = all(t.seat != winner for t in after) and all(t.kind == "draw" and t.call is None for t in after)
            # a riichi on the seat's first discard with no call (nor kan) by anyone before it
            double_riichi = riichi_i == my_turns[0].i and all(t.kind == "draw" and t.call is None for t in self.turns[:riichi_i])
        # the last tile of the wall: the winning draw is the 70th (haitei), the winning discard follows it (houtei)
        haitei = sum(self._wall_draws()) >= rules.LIVE_WALL
        # rinshan kaihou: the winner kanned after its last discard and drew the winning tile from the dead wall
        rinshan = result.outcome == "tsumo" and bool(my_turns) and my_turns[-1].slot is None and my_turns[-1].own_call is not None
        self.context = {"ippatsu": ippatsu, "haitei": haitei, "rinshan": rinshan, "ura": self.ura}
        round_wind = "ESW"[self.entry["kyoku"] // 4]

        def score_of(conc, wt, dora):
            return score_hand(conc, wt, melds, tsumo=result.outcome == "tsumo", riichi=winner in self.site_riichi, seat=winner,
                              round_wind=round_wind, dora=dora, ura=self.ura, ippatsu=ippatsu, haitei=haitei, rinshan=rinshan,
                              double_riichi=double_riichi)
        return score_of

    def _fits(self, r) -> bool:
        """The site's score; with a riichi win whose ura is not known yet, any winning hand whose han do not exceed
        the site's (ura may add the rest) with the right fu (ura never change the fu)."""
        result = self.result
        if matches_site(r, result.han, result.fu):
            return True
        return (self.winner in self.site_riichi and not self.ura and r.ok and r.han is not None and r.han <= result.han
                and r.fu == result.fu)

    def _reveal_views(self) -> list[dict]:
        """The winner's hand views after the last discard: the reveal."""
        t_last = self.turns[-1].t if self.turns else self.t1
        return [o for o in self.obs.get(f"hand:{corner_of(self.entry, self.winner)}", []) if o["t0"] >= t_last and o["n_used"] > 0]

    def _reveal_mass(self, x: str) -> float:
        """How strongly the reveal reads tile x."""
        return sum(sl["p"][CLASSES.index(x)] for o in self._reveal_views() for sl in o["slots"])

    def _reveal_extra(self, x: str, concealed: list[str]) -> float:
        """How strongly the reveal shows tile x beyond the concealed hand: over the views of the whole hand with the
        winning tile (one tile more than the concealed part), the reading of x less the copies the hand already
        holds — the winning tile, not a tile of its kind inside the hand. The plain reading without such a view."""
        k = CLASSES.index(x)
        full = [o for o in self._reveal_views() if o["count"] == len(concealed) + 1]
        if not full:
            return self._reveal_mass(x)
        held = sum(1 for t in concealed if t == x)
        return sum(max(0.0, sum(sl["p"][k] for sl in o["slots"]) - held) for o in full)

    def _five_swaps(self, concealed: list[str], win_tiles: list[str], melds: list[dict], score_of) -> list[tuple]:
        """The winner's fives read plain for red or the reverse, with the winning tiles that then score the site's
        value: [(five, swapped, concealed after, winning tile)]. One red of a suit exists: a red seen anywhere else
        (a meld, a pond, an indicator, the winning tile) cannot be in the hand."""
        seen = [x for m in melds for x in m["tiles"]] + self.dora + self.ura + [sl.tile for sls in self.logs.values() for sl in sls]
        out = []
        for i, x in enumerate(concealed):
            alt = SWAP_FIVE.get(x)
            if alt is None or x in concealed[:i]:
                continue
            conc2 = concealed[:i] + [alt] + concealed[i + 1:]
            if alt in rules.REDS and (alt in seen or conc2.count(alt) > 1):
                continue
            out += [(x, alt, conc2, w) for w in win_tiles if not (alt in rules.REDS and w == alt)
                    and self._fits(score_of(conc2, w, self.dora))]
        return out

    def _winning_tile_from_the_site(self, concealed, win_tile, sc, melds, score_of, jw):
        """The site record as a constraint: an unseen tsumo tile is the tile that makes the hand win with the site's
        score; among several, the one the reveal frames show most. A five read plain for red (or the reverse)
        changes the han by one: with no tile fitting as read, each swap is tried with each tile."""
        winner, result = self.winner, self.result
        used = Counter(rules.plain(x) for x in concealed + [x for m in melds for x in m["tiles"]] + self.dora + self.ura)
        for sls in self.logs.values():
            for sl in sls:
                used[rules.plain(sl.tile)] += 1
        t_last = self.turns[-1].t if self.turns else self.t1
        mass = self._reveal_mass
        tiles = [x for x in CLASSES if x not in ("X", "none") and used[rules.plain(x)] < 4]
        cands = [x for x in tiles if self._fits(score_of(concealed, x, self.dora))]
        if not cands:
            swaps = self._five_swaps(concealed, tiles, melds, score_of)
            if swaps:
                # several fives could be the red one: the reveal frames rank them
                opts = sorted({(a, b) for a, b, _, _ in swaps}, key=lambda ab: -(mass(ab[1]) - mass(ab[0])))
                x, alt = opts[0]
                concealed = next(c2 for a, b, c2, _ in swaps if (a, b) == (x, alt))
                self.problems.append(f"the winner's {x} must be {alt} for the hand to score the site's {result.han}/{result.fu}: swapped"
                                     + (f" (also possible: {', '.join(b for _, b in opts[1:])})" if len(opts) > 1 else ""))
                if winner in self.model.facts.final and x in self.model.facts.final[winner]:
                    ft = list(self.model.facts.final[winner])
                    ft[ft.index(x)] = alt
                    self.model.facts.final[winner] = ft
                cands = sorted({x2 for a, b, _, x2 in swaps if (a, b) == (x, alt)})
        extra = {x: self._reveal_extra(x, concealed) for x in cands}
        cands.sort(key=lambda x: -extra[x])
        if not cands:
            return concealed, win_tile, sc
        self.model.facts.draws[(winner, jw)] = cands[0]
        sol2 = self._solve(prior=self.sol)
        if not sol2.ok:
            del self.model.facts.draws[(winner, jw)]
            return concealed, win_tile, sc
        self.sol = sol2
        win_tile = cands[0]
        concealed = list(sol2.hands.get((winner, jw), []))
        if win_tile in concealed:
            concealed.remove(win_tile)
        others = ", ".join(cands[1:])
        self.problems.append(f"the winning tile was not seen: {cands[0]} makes the hand win with the site's score"
                             + (f" (also possible: {others})" if others else ""))
        if len(cands) > 1 and extra[cands[0]] < 2 * extra[cands[1]]:
            self.items.append({"kind": "result", "seat": winner, "t": t_last,
                               "text": f"The winning tile was not seen. These tiles make the hand win with the site's score: {', '.join(cands)}; "
                                       f"{cands[0]} is used. Confirm the final hand and the winning tile.",
                               "tiles": concealed, "win_tile": win_tile, "candidates": cands})
        return concealed, win_tile, score_of(concealed, win_tile, self.dora)

    def _red_five_from_the_site(self, concealed, win_tile, sc, melds, score_of, j_hand):
        """The site's han says a five of the winner's hand is the other one (red for plain or the reverse): the swap
        that fits, ranked by the reveal frames, is bound as the winner's hand and the hand solved again, so the
        draws that brought the five follow. A swap the draws cannot deliver is left as read."""
        result = self.result
        swaps = self._five_swaps(concealed, [win_tile], melds, score_of)
        if not swaps:
            return concealed, sc
        mass = self._reveal_mass
        x, alt, conc2, _ = max(swaps, key=lambda sw: mass(sw[1]) - mass(sw[0]))
        self._bind_winning_hand(conc2, win_tile, j_hand)
        sol2 = self._solve(prior=self.sol)
        if not sol2.ok or sol2.objective - self.sol.objective > NEXT_BUDGET:
            self.model.bound_hands = []
            apply_choices(self.sol, self.model, self.turns, self.unknown_kans)
            self.problems.append(f"the winner's {x} as {alt} would score the site's {result.han}/{result.fu}, but "
                                 + ("no draw can bring it" if not sol2.ok else f"the video reads it otherwise ({sol2.objective - self.sol.objective:.0f} costlier)")
                                 + ": kept as read")
            return concealed, sc
        self.sol = sol2
        others = sorted({b for _, b, _, _ in swaps if b != alt})
        self.problems.append(f"the winner's {x} is {alt}: the site's {result.han}/{result.fu} needs it (a red five is one han)"
                             + (f"; also possible: {', '.join(others)}" if others else ""))
        return conc2, score_of(conc2, win_tile, self.dora)

    def _dora_from_the_site(self, concealed, win_tile, sc, score_of):
        """A dora indicator contradicted by the site's han: the next reading of the same tile that fits the score."""
        result = self.result
        pv = np.asarray(self.inds[0]["p"])
        alt = []
        for k in np.argsort(pv)[::-1][:8]:
            cnd = CLASSES[int(k)]
            if cnd in ("X", "none") or cnd == self.dora[0] or pv[k] <= 0:
                continue
            if matches_site(score_of(concealed, win_tile, [cnd] + self.dora[1:]), result.han, result.fu):
                alt.append(cnd)
        if not alt:
            return sc
        old = self.dora[0]
        self.dora[0] = alt[0]
        self.inds[0]["tile"], self.inds[0]["corrected_from"] = alt[0], old
        self.problems.append(f"dora indicator read as {old} contradicts the site score; {alt[0]}, the next reading, fits it: used")
        return score_of(concealed, win_tile, self.dora)

    def check_draw(self) -> None:
        """At an exhaustive draw the site's tenpai seats must be tenpai with their final states; a riichi win
        reveals ura indicators, without which the han cannot be checked."""
        sol, result = self.sol, self.result
        if sol.ok and result.outcome == "draw":
            for s in rules.SEATS:
                hand = sol.hands.get((s, len(self.model.turns[s]) - 1), sol.haipai.get(s, []))
                tp = is_tenpai(list(hand), scoring_melds(self.live_calls, s))
                want = site_seat_name(s, self.entry) in result.tenpai
                if tp != want:
                    self.items.append({"kind": "result", "seat": s, "t": self.turns[-1].t if self.turns else self.t1,
                                       "text": f"{s} is {'tenpai' if tp else 'noten'} in the reconstruction, site says {'tenpai' if want else 'noten'}",
                                       "tiles": list(hand)})
        if result.outcome in ("ron", "tsumo") and self.winner in self.site_riichi and not self.facts.get("ura"):
            self.items.append({"kind": "ura", "seat": self.winner, "t": self.turns[-1].t if self.turns else self.t1,
                               "text": "riichi win: ura indicator(s) not known; add an ura fact"})

    # ---- 9. review and output ----------------------------------------------------------------------------------

    def output(self) -> dict:
        """Build the decode artifact, preserving evidence, alternatives, and unresolved questions."""
        sol, model = self.sol, self.model
        lost_keys = {k for k in (turn_key(model, f, 3.0) for f in self.facts.get("lost", [])) if k is not None}
        out_turns = []
        output_items = list(self.items)
        replacements = {(r["seat"], r["slot_id"]): r for r in pond_evidence.replacement_requests(
            self.turns, self.entry, self.facts, self.t0, self.t1)}
        for t in self.turns:
            st = next((x for x in model.turns[t.seat] if x.t_discard == t.t), None)
            j = st.j if st else None
            draw = sol.draws.get((t.seat, j)) if j is not None else None
            chosen = (sol.discards.get((t.seat, j)) or self.repaired.get((t.seat, j))) if j is not None else None
            draw2 = sol.draws2.get((t.seat, j)) if j is not None else None
            replacement = replacements.get((t.seat, t.slot.id)) if t.slot is not None else None
            receipt = t.slot.replacement_acquisition if t.slot is not None else None
            discard_margin = sol.discard_margins.get((t.seat, j))
            replacement_certified = bool(replacement and replacement["acquired"] and receipt
                and (chosen or t.slot.tile) in receipt["supported_tiles"]
                and sol.ok and discard_margin is not None and discard_margin == discard_margin
                and not low_margin(discard_margin))
            if t.own_call is not None and id(t.own_call) in self.unknown_kans and (t.seat, j) in sol.kans:
                self.problems.append(f"the ankan of {t.seat} at {t.t:.0f}s: tile {sol.kans[(t.seat, j)]} chosen by the rules, not seen")
            if chosen is not None and t.slot is not None:
                self.problems.append(f"{t.seat}'s discard at {t.t:.0f}s read {t.slot.tile}: the rules need {chosen}")
                question = changed_discard(t.seat, j, t.t, t.slot.tile, chosen, self.facts.get("discard", []))
                if question and not replacement_certified and not any(i.get("kind") == "discard" and i.get("seat") == t.seat and i.get("j") == j for i in output_items):
                    output_items.append(question)
            if replacement and not replacement_certified and not any(
                    i.get("kind") == "discard" and i.get("seat") == t.seat and i.get("j") == j for i in output_items):
                question = pond_evidence.replacement_question(replacement, chosen=chosen)
                question["j"] = j
                output_items.append(question)
            margin = sol.margins.get((t.seat, j)) if j is not None else None
            out_turns.append({"i": t.i, "seat": t.seat, "j": j, "kind": t.kind, "t": t.t, "t_prev": out_turns[-1]["t"] if out_turns else self.t0,
                              "riichi": t.riichi, "hand_before": sol.hands.get((t.seat, j - 1)) if j is not None else None,
                              "hand_after": sol.hands.get((t.seat, j)) if j is not None else None, "draw2": draw2,
                              "discard_box": [round(float(v), 1) for v in t.slot.xyxy] if (t.slot and t.slot.xyxy) else None,
                              "discard_pos": [t.slot.row, t.slot.index] if t.slot else None,
                              "discard": chosen or (t.slot.tile if t.slot else None), "discard_conf": round(t.slot.conf, 3) if t.slot else None,
                              "discard_id": t.slot.id if t.slot else None, "virtual": t.virtual, "draw": draw,
                              "pond_tracking": ({"pending_replacement": t.slot.to_dict().get("pending_replacement"),
                                                 "acquisition": receipt} if replacement else None),
                              "margin": margin if margin is not None and margin == margin else None,
                              "tsumogiri": (draw2 or draw) is not None and t.slot is not None and (draw2 or draw) == (chosen or t.slot.tile),
                              "call": t.call.to_dict() if t.call else None, "own_call": t.own_call.to_dict() if t.own_call else None})
        confidence = confidence_rows(model, sol, self.turns, self.live_calls, self.inds, self.entry, self.facts, lost_keys, self.t0)
        output_items.extend(uncertain_discards(confidence, output_items))
        pending = uncertain_tiles(confidence, output_items)
        return {"decoder_version": DECODER_VERSION,
                "hand": self.entry["hand"], "game": self.entry["game"], "kyoku": self.entry["kyoku"], "honba": self.entry["honba"],
                "play_window": [self.t0, self.t1], "t_last": self.turns[-1].t if self.turns else self.t1, "dealer": self.dealer,
                "turns": out_turns, "haipai": sol.haipai, "haipai_margin": sol.haipai_margins,
                "draws": {f"{s}:{j}": v for (s, j), v in sol.draws.items()},
                "dora": self.dora, "ura": self.ura, "indicators": self.inds, "riichi": sorted(self.site_riichi),
                "result": {"outcome": self.result.outcome, "winner": self.winner, "loser": self.loser, "han": self.result.han,
                           "fu": self.result.fu, "site": list(self.site_han_fu), "site_wrong": bool(self.facts.get("site_score")), "deltas": self.result.deltas, "tenpai": [site_seat(s, self.entry) for s in self.result.tenpai]},
                "score": self.score,
                "solver": {"status": sol.status, "objective": round(sol.objective, 3),
                           "optimal": sol.optimal,
                           "low_margin": sum(1 for mg in sol.margins.values() if low_margin(mg))},
                "calls": [c.to_dict() for c in self.calls], "problems": self.problems, "items": ranked(output_items + ([pending] if pending else [])),
                "confidence": confidence,
                "stats": {"discards": sum(len(v) for v in self.logs.values()), "turns": len(self.turns), "calls": len(self.calls),
                          "hand_evidence": len(model.hand_ev), "draw_evidence": len(model.draw_ev)}}

    def run(self) -> dict:
        """Execute the stateful decoder once; stages depend on the preceding stages' results."""
        self.window()
        self.discards()
        self.calls()
        self.indicators()
        self.meld_facts()
        self.turn_sequence()
        self.end_of_hand()
        self.riichi()
        self.pond_replacements()
        self.solve()
        self.kan_indicators()
        self.check_score()
        self.check_draw()
        self.unseen_tiles()
        if self.sol.ok and self.sol.optimal is False:
            self.items.append({"kind": "solver_incomplete", "t": self.turns[-1].t if self.turns else self.t1,
                               "text": "A legal reconstruction was found, but the search ended before proving it was the best "
                                       "fit to the evidence. The log remains provisional; rebuild this hand to retry the search."})
        return self.output()


def decode_hand(entry: dict, obs: dict[str, list[dict]], result: HandResult, facts: Optional[dict] = None,
                *, time_limit: float = 60.0, log=print, models=None, work_dir: Optional[Path] = None) -> dict:
    """Reconstruct one hand from observations and its mandatory site result.

    Facts are the normalized output of ``facts_for_hand``. Supplying ``models``
    as (detector, classifier, video path, calibration) enables targeted rereads;
    otherwise the returned artifact exposes uncertainty from existing evidence.
    This function does not write artifacts or alter human facts.
    """
    return HandDecoder(entry, obs, result, facts or {}, time_limit=time_limit, models=models, work_dir=work_dir).run()


def _decode_input_binding(work: Path, hand: int, *, evidence_policy=None) -> dict | None:
    from ..perception.evidence_policy import load_policy, resolve_policy
    policy = load_policy() if evidence_policy is None else resolve_policy(evidence_policy)
    observation = work / "obs" / f"{hand:02d}.json"
    provenance = work / "obs" / "provenance" / f"{hand:02d}.json"
    if not observation.exists() or not provenance.exists():
        return None
    return {"observation_sha256": hashlib.sha256(observation.read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
            "dense_policy": policy.fingerprint_for('dense')}


def _publish_decode(path: Path, decoded: dict) -> None:
    pending = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".decode-", suffix=".tmp",
                                delete=False) as stream:
            pending = Path(stream.name)
            json.dump(sanitize(decoded), stream, ensure_ascii=False, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def run_decode(work: Path, hands: list[dict], games: list[Game], *, force: bool = False, only: Optional[set] = None,
               log=print, video_path: Optional[Path] = None, cal=None, evidence_policy=None) -> list[dict]:
    """Decode selected observed hands and persist each result under ``work/decode``.

    Current-version results with matching observation/provenance content hashes
    are reused unless ``force`` is true. Results are published atomically. Models are loaded only
    when a hand actually needs decoding and a video is available. Human labels
    and the default video resolve under VIDEO2TENHOU_HOME (the working directory
    by default), never inside an installed package.
    When dense rereads are enabled, sparse-read manifests must match the current
    source, models/runtime and calibration before any decoded hand is replaced.
    `evidence_policy` is an explicit policy or is read from detector metadata
    without loading weights. Sparse-policy provenance must match; the dense policy
    also binds decoded caches so dense-only changes cannot reuse an old result.
    """
    from ..observe import load_obs
    from ..perception.evidence_policy import DEFAULT_POLICY, load_policy, resolve_policy
    # Resolve before optional model loading: invalid policy is not an offline fallback.
    policy = load_policy() if evidence_policy is None else resolve_policy(evidence_policy)
    ddir = work / "decode"
    ddir.mkdir(parents=True, exist_ok=True)
    all_facts = load_facts(DATA_DIR / "labels" / work.name)
    models = None
    selected = [h for h in hands if only is None or h["hand"] in only]
    bindings = {h["hand"]: _decode_input_binding(work, h["hand"], evidence_policy=policy) for h in selected}
    cached = {}
    if not force:
        for h in selected:
            p = ddir / f"{h['hand']:02d}.json"
            if p.exists():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if (isinstance(d, dict) and d.get("decoder_version") == DECODER_VERSION
                        and bindings[h["hand"]] is not None
                        and d.get("decode_inputs") == bindings[h["hand"]]):
                    cached[h["hand"]] = d
    pending = [h for h in selected if h["hand"] not in cached
               and (work / "obs" / f"{h['hand']:02d}.json").exists()]
    if video_path is None:
        cand = DATA_DIR / "samples" / f"{work.name}.mp4"
        video_path = cand if cand.exists() else None
    if pending and video_path is not None:
        # Keep metadata failures outside the optional-model fallback, including
        # when the caller resolved an explicit policy before this stage began.
        if load_policy() != policy:
            raise ValueError("Recognition evidence policy changed before rebuilding. "
                             "Choose Analyze recording to refresh evidence.")
        try:
            from ..layout import Calibration
            from ..perception.classifier import Classifier
            from ..perception.detector import Detector
            models = (Detector(), Classifier(), video_path, cal or Calibration.load("pml", video_path))
        except Exception as ex:  # noqa: BLE001
            log(f"  dense reads disabled: {ex}")
    if models is not None:
        from ..read import validate_read_cache
        from ..observe import validate_observation_cache
        actual_policy = getattr(models[0], 'evidence_policy', DEFAULT_POLICY)
        if actual_policy != policy:
            raise ValueError('Recognition evidence policy changed before rebuilding. Choose Analyze recording to refresh evidence.')
        # Do not turn stale evidence into the optional-model fallback above:
        # every selected hand must pass before any existing result is replaced.
        validate_read_cache(video_path, models[3], work, pending, models[0], models[1])
        validate_observation_cache(work, pending, policy=policy)
    out = []
    for h in selected:
        p = ddir / f"{h['hand']:02d}.json"
        if h["hand"] in cached:
            out.append(cached[h["hand"]])
            continue
        if not (work / "obs" / f"{h['hand']:02d}.json").exists():
            continue
        obs = load_obs(work, h["hand"])
        result = games[h["game"]].hands[h["site_index"]]
        d = decode_hand(h, obs, result, facts_for_hand(all_facts, h), log=log, models=models, work_dir=work)
        d["decode_inputs"] = bindings[h["hand"]]
        _publish_decode(p, d)
        out.append(d)
        sc = d["score"]
        log(f"  hand {h['hand']:2d}: {d['stats']['turns']} turns, {d['stats']['calls']} calls, dora {d['dora']}, "
            f"solver {d['solver']['status']} obj {d['solver']['objective']}, low-margin draws {d['solver']['low_margin']}, "
            f"score {'ok' if sc and sc['match'] else (sc['han'] if sc else '-')}/{sc['fu'] if sc else '-'} vs site {d['result']['han']}/{d['result']['fu']}, "
            f"problems {len(d['problems'])}")
    return out
