# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 5: reconstruct hand events, solver margins and review questions.

Stage 5: observations of one hand -> events, solver margins, review items (DESIGN.md 4.8
`decode.py`).

ponds -> discard logs; meld events anchored on the discards they took -> calls; the
dead-wall row -> dora and kans; the reviewer's facts; turns -> the sequence; solver ->
haipai, draws and the calls' tiles, with the site's result as a constraint; scoring ->
the han / fu check against the site record; review -> the few questions that settle the
hand, and the confidence of every decision. Each step is one method of `HandDecoder`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.engine.dense import DenseContext
from video2tenhou.engine.indicators import KanEvidence
from video2tenhou.engine.review import ReviewContext
from video2tenhou.engine.solver import HandRole
from video2tenhou.engine.turns import HandEnding
from video2tenhou.files import atomic_write_json, sanitize, sha256_file
from video2tenhou.layout import Calibration
from video2tenhou.observe import load_obs, validate_observation_cache
from video2tenhou.paths import DATA_DIR
from video2tenhou.perception import detector
from video2tenhou.perception import evidence_policy as retention
from video2tenhou.read import ReadContext, validate_read_cache
from video2tenhou.train.data import CLASSES

from . import dense, pond_evidence, rules
from .calls import CallAnchor, meld_events
from .confidence import MARGIN_REVIEW, low_margin
from .hand import corner_of, in_window, seat_of, site_seat, site_seat_name
from .indicators import KAN_MATCH, indicator_row, reconcile_kans
from .melds import Call, meld_options
from .ponds import PondSlot, play_window, track_pond
from .review import (
    changed_discard,
    confidence_rows,
    draws_to_reread,
    facts_for_hand,
    load_facts,
    ranked,
    turn_key,
    uncertain_discards,
    uncertain_tiles,
    unseen_draw,
)
from .scoring import (
    WinContext,
    is_tenpai,
    matches_site,
    payment,
    score_hand,
    score_text,
)
from .solver import (
    TILES,
    Culprit,
    HandEvidence,
    HandModel,
    SeatTurn,
    Solution,
    WinSpec,
    diagnose,
    hand_evidence,
    posterior_to_tiles,
)
from .turns import Turn, assign_calls, merge
from .validation import review_artifact

if TYPE_CHECKING:
    from video2tenhou.engine.scoring import ScoreResult


if TYPE_CHECKING:
    from collections.abc import Callable

    from video2tenhou.perception.evidence_policy import EvidencePolicy
    from video2tenhou.read import ReadModels


if TYPE_CHECKING:
    from pathlib import Path

    from video2tenhou.record import Game, HandResult

MAX_PLAIN_TILE_COPIES = 4
DISCARD_FACT_WINDOW = 10
MISSING_DISCARD_WINDOW = 15
MIN_UNANCHORED_KAN_VIEWS = 5
DUPLICATE_CALL_WINDOW = 5
CALL_FACT_WINDOW = 40
MELD_TILE_COUNT = 3
RIICHI_FACT_WINDOW = 3


# the two readings of a riichi on a called tile solve this close: the reviewer is asked
RIICHI_CLOSE = 0.5
SWAP_FIVE = {
    "5m": "0m",
    "5p": "0p",
    "5s": "0s",
    "0m": "5m",
    "0p": "5p",
    "0s": "5s",
}  # a five read as the other one
NEXT_HANDS = 6  # next-best reconstructions tried for the site's score ...
# only close hand alternatives may resolve a score mismatch automatically
NEXT_BUDGET = MARGIN_REVIEW
DECODER_VERSION = (
    17  # bump when reconstruction semantics change; old cached logs must be rebuilt
)


# ------------------------------------------------------------------------------
# pieces used by the decoder (and by the tests)
# ------------------------------------------------------------------------------


LOGGER = logging.getLogger("video2tenhou.engine.decode")


@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeOptions:
    """Solver budgets and optional targeted-reading inputs for one hand."""

    time_limit: float = 60.0
    confidence_timeout: float = 10.0
    models: ReadModels | None = None
    work_dir: Path | None = None


def apply_choices(
    sol: Solution, model: HandModel, turns: list[Turn], unknown: set[int]
) -> None:
    """Apply solver-selected meld compositions before scoring or export.

    The solver's choices about the calls go into them before anything scores or writes
    them: which legal composition a call is (its type and tiles), the tile of an ankan
    the camera never named, and which five (plain or red) a kakan of fives added.
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


def concealed_size(melds: list[dict], *, drawn_left_in: bool = False) -> int:
    """Count the winner's concealed tiles excluding the winning tile.

    How many tiles the winner holds beside their melds and beside the winning tile
    (section 1).

    A hand is 13 tiles; each meld set takes three of them out (a kan's fourth tile is
    the extra tile the kan adds, not a fourteenth hand tile). A kakan is the pon it grew
    from: listed beside that pon it is not a set of its own. `drawn_left_in` is for a
    tsumo whose winning tile was never named: it is still in the list.
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

    A seat's melds for the scoring library: an ankan's face-down tiles are its kind, any
    kan of fives is all four fives (three plain and the red one) whatever the camera
    read of them, and a kakan replaces the pon it grew from (it is that pon with its
    fourth tile, not a meld beside it).
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


def _meld_copies(c: Call) -> dict[str, int]:
    """Count revealed meld tiles without recounting the discard that was called."""
    per_kind: dict[str, int] = defaultdict(int)
    if c.type == "kakan":
        per_kind[rules.plain(c.tiles[-1])] += (
            1  # only the added tile: the pon below it is counted already
        )
    else:
        for k, tile in enumerate(c.tiles):
            if k == c.called_pos and c.type in ("chi", "pon", "kan"):
                continue  # the called tile is one of the discards above
            if tile not in ("?", "X"):
                per_kind[rules.plain(tile)] += 1
    return per_kind


def over_count(
    turns: list[Turn], calls: list[Call], dora: list[str], entry: dict
) -> list[dict]:
    """Find tile kinds with more than four revealed copies.

    Kinds that appear more than four times among discards, meld tiles and indicators,
    with every source.
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
        per_kind = _meld_copies(c)
        for kind, n in per_kind.items():
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


def seat_turns_of(
    turns: list[Turn], seat: str, dealer: str
) -> tuple[list[SeatTurn], dict[int, int]]:
    """Build a seat's turns and counts of melds laid before each state.

    SeatTurns of one seat in order, plus melds_before[j] = melds laid up to state j (the
    hand after turn j).
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
                two = True  # normal draw, kan, rinshan draw
            else:
                removed, options = _hand_tiles(c)
            if c.type != "kakan":
                n_melds += 1
            if t.slot is None:
                # the winner's kan after its last discard: the rinshan draw is the
                # winning draw (the final draw
                # variable), so this turn holds the normal draw only (ankan / kakan) or
                # no draw (daiminkan)
                two = False
                if kan == "daiminkan":
                    kind = "call"
        if j == 0 and seat == dealer and kind in ("draw", "kan"):
            # the dealer's 14 are one set: its first turn has no draw of its own (a
            # first-turn ankan still has
            # its rinshan draw)
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
    """Return the fixed or solver-selected tiles a call takes from the hand.

    The tiles a call takes from the hand: fixed (a reviewer's meld, or a call with one
    composition), or the solver's choice among the legal compositions [(tiles, cost)].
    """
    if c.options and not c.human and len(c.options) > 1:
        return [], [(o.hand, o.cost) for o in c.options]
    return [x for i, x in enumerate(c.tiles) if i != c.called_pos], []


# ------------------------------------------------------------------------------
# the decoder
# ------------------------------------------------------------------------------


class HandDecoder:
    """Reconstruct one hand in stages, with optional dense video evidence.

    One hand, step by step. `models` = (detector, classifier, video path, calibration)
    enables the dense reads (dense.py); without them the decode uses the calm
    observations only.
    """

    @property
    def dense_work(self) -> Path:
        """Require an evidence cache directory before acquiring dense readings."""
        if self.work_dir is None:
            msg = "Dense evidence acquisition requires a workspace directory"
            raise ValueError(msg)
        return self.work_dir

    @property
    def winning_seat(self) -> str:
        """Require a named winner before evaluating winning-hand evidence."""
        if self.winner is None:
            msg = "Winning-hand analysis requires a winner in the site record"
            raise ValueError(msg)
        return self.winner

    def __init__(
        self,
        entry: dict,
        obs: dict[str, list[dict]],
        result: HandResult,
        facts: dict,
        *,
        options: DecodeOptions,
    ) -> None:
        """Bind authoritative results, observations and facts for one hand."""
        time_limit, models, work_dir, confidence_timeout = (
            options.time_limit,
            options.models,
            options.work_dir,
            options.confidence_timeout,
        )
        self.site_han_fu = (result.han, result.fu)
        if facts.get("site_score"):
            # the reviewer confirmed the site's han/fu wrong: theirs stand for it
            # everywhere, the deltas stay (4.8 scoring.py)
            result = dataclasses.replace(
                result, han=facts["site_score"]["han"], fu=facts["site_score"]["fu"]
            )
        self.entry, self.obs, self.result, self.facts = entry, obs, result, facts
        self.time_limit, self.models, self.work_dir = time_limit, models, work_dir
        self.confidence_remaining = max(0.0, confidence_timeout)
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
        self.repaired: dict[
            tuple[str, int], str
        ] = {}  # (seat, j) -> the kind a discard was re-read as
        if facts.get("site_score"):
            self.problems.append(
                f"the site's {self.site_han_fu[0]}/{self.site_han_fu[1]} is "
                "wrong (the reviewer): the hand is checked against, and the log"
                f" written with, {result.han}/{result.fu}"
            )

    # ---- 1. the window and the discards
    # ------------------------------------------------------------------

    def window(self) -> None:
        """Restrict observations to this hand's clearing and play interval."""
        e = self.entry
        self.t0, self.t1 = play_window(
            self.pond_obs,
            e["t_start"],
            e["t_end"],
        )

    def discards(self) -> None:
        """Track each pond, then apply reviewed identities and missing-discard facts."""
        self.logs: dict[str, list[PondSlot]] = {}
        for corner in ("TL", "TR", "BL", "BR"):
            self.logs[seat_of(self.entry, corner)] = track_pond(
                in_window(self.pond_obs[f"pond:{corner}"], self.t0, self.t1)
            )
        # the reviewer named the tile of a pond slot (the seat's slot nearest the given
        # time)
        for f in self.facts.get("discard", []):
            mine = self.logs.get(f["seat"], [])
            if mine:
                sl = min(mine, key=lambda x: abs(x.t_first - f["t"]))
                if abs(sl.t_first - f["t"]) <= DISCARD_FACT_WINDOW:
                    sl.p = np.zeros(len(CLASSES))
                    sl.p[CLASSES.index(f["tile"])] = 1.0
        # a discard the reader never saw (the reviewer names seat, time and tile)
        for f in self.facts.get("missing_discard", []):
            mine = self.logs[f["seat"]]
            if any(
                rules.plain(sl.tile) == rules.plain(f["tile"])
                and abs(sl.t_first - f["t"]) < MISSING_DISCARD_WINDOW
                for sl in mine
            ):
                # a later re-read found it: the fact would double it
                self.problems.append(
                    f"missing-discard fact {f['tile']} of {f['seat']} at "
                    f"{f['t']:.0f}s: the pond now shows it, fact not applied"
                )
                continue
            pp = np.zeros(len(CLASSES))
            pp[CLASSES.index(f["tile"])] = 1.0
            vs = PondSlot(
                9000 + len(mine),
                -1,
                -1,
                pp,
                f["t"],
                (f["t"] - 5.0, f["t"]),
                f["t"],
                2,
                0.0,
            )
            mine.insert(sum(1 for sl in mine if sl.t_first <= f["t"]), vs)
        times = [sl.t_first for sls in self.logs.values() for sl in sls]
        self.first_discard = min(times, default=self.t0)
        self.last_discard = max(times, default=None)

    # ---- 2. the calls
    # ---------------------------------------------------------------------------------

    def anchor_calls(self) -> None:
        """Anchor observed meld events to the discards they took.

        Meld events anchored on the discards they took (calls.py): the camera says that
        a seat laid a meld, the pond which discard it took; with nothing taken, only a
        kan pattern is a call.
        """
        events = meld_events(self.entry, self.obs, self.t0, self.t1)
        anchor = CallAnchor(
            self.logs,
            self.obs,
            context=DenseContext(
                entry=self.entry,
                models=self.models,
                work_dir=self.work_dir,
                t0=self.t0,
                problems=self.problems,
            ),
        )
        self.calls = anchor.anchor(
            events, self.first_discard, self.last_discard, self.tsumo_winner
        )
        # meld events with strong camera evidence, a hand that says a meld was laid, and
        # no discard found: a pond
        # read may have missed it
        self.unanchored = [
            e
            for e in events
            if e.type in ("chi", "pon")
            and e.seen >= MIN_UNANCHORED_KAN_VIEWS
            and not any(
                c.seat == e.seat and abs(c.t_first - e.t_first) < DUPLICATE_CALL_WINDOW
                for c in self.calls
            )
            and not anchor.reread(e, self.calls)
        ]

    def meld_facts(self) -> None:
        """Apply human meld facts after inferred calls.

        The reviewer's meld facts, after everything the program inferred (a fact always
        wins): "this meld is <type> <tiles> from <source>" replaces the call of that
        seat nearest its time and suppresses the seat's other calls within 40 s; "this
        meld does not exist" removes the nearest call (of the given type).
        """
        for f in self.facts.get("meld", []):
            near = [
                c
                for c in self.calls
                if c.seat == f["seat"] and abs(c.t_first - f["t"]) < CALL_FACT_WINDOW
            ]
            typ, tiles, pos = f["type"], list(f["tiles"]), f.get("called_pos")
            if typ in ("kan", "ankan", "kakan") and len(tiles) == MELD_TILE_COUNT:
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
                f["seat"],
                t_first,
                window,
                typ,
                tiles,
                pos,
                source,
                tiles[pos] if pos is not None and pos < len(tiles) else None,
                [],
                1.0,
                0,
                9,
                human=True,
                anchor="fact",
            )
            self.calls = [c for c in self.calls if c not in near] + [fact]
            if near:
                self.problems.append(
                    (
                        f"your meld fact for {f['seat']} at {f['t']:.0f}s ({typ} "
                        f"{''.join(tiles)}) replaces "
                    )
                    + ", ".join(f"{c.type} {''.join(c.tiles)}" for c in near)
                )
        for f in self.facts.get("meld_remove", []):
            cands = [
                c
                for c in self.calls
                if c.seat == f["seat"]
                and abs(c.t_first - f["t"]) < CALL_FACT_WINDOW
                and (not f.get("type") or c.type == f["type"])
            ]
            if not cands:
                self.problems.append(
                    f"meld-remove fact for {f['seat']} at {f['t']:.0f}s matched no call"
                )
                continue
            gone = min(cands, key=lambda c: abs(c.t_first - f["t"]))
            self.calls = [c for c in self.calls if c is not gone]
            self.problems.append(
                f"the {gone.type} {''.join(gone.tiles)} of {gone.seat} at "
                f"{gone.t_first:.0f}s was removed by the reviewer"
            )

    # ---- 3. the dead wall
    # ---------------------------------------------------------------------------------

    def indicators(self) -> None:
        """Read the dead-wall timeline and reconcile its new indicators with kans."""
        # a tsumo winner may have kanned after its last discard (a win on the rinshan
        # draw): its indicator follows
        # the last discard; the discard after a last-turn kan may be sighted before the
        # flipped indicator is
        inds = indicator_row(
            self.pond_obs,
            self.t0,
            self.t1,
            self.last_discard,
            t_after=15.0 if self.tsumo_winner else 10.0,
        )
        if self.facts.get("dora"):
            inds = self._named_indicators(inds)
        self.inds = inds
        self.calls = reconcile_kans(
            inds,
            self.calls,
            context=KanEvidence(
                logs=self.logs,
                obs=self.obs,
                entry=self.entry,
                t0=self.t0,
                problems=self.problems,
                tsumo_winner=self.tsumo_winner,
            ),
        )
        self.dora = [v["tile"] for v in inds]

    def _place_indicator_times(self, named: list[dict]) -> None:
        """Apply reviewed kan times and question still-unplaced indicators."""
        # kan-time facts: the reviewer clicked the discard that followed the kan of an
        # indicator no frame shows
        unplaced = [
            v for i, v in enumerate(named) if i > 0 and v.get("t_first") is None
        ]
        for v, f in zip(
            unplaced,
            sorted(self.facts.get("kan_time", []), key=lambda f: f["t"]),
            strict=False,
        ):
            near = [
                (abs(sl.t_first - f["t"]), sl.t_first, s)
                for s, sls in self.logs.items()
                for sl in sls
                if abs(sl.t_first - f["t"]) <= RIICHI_FACT_WINDOW
                and (not f.get("seat") or s == f["seat"])
            ]
            if not near:
                self.problems.append(
                    f"kan-time fact at {f['t']:.0f}s matches no discard: "
                    f"indicator {v['tile']} stays unplaced"
                )
                continue
            _, tt, s = min(near)
            v["t_first"], v["t_before"], v["kan_maker"] = tt - 0.5, tt - 1.0, s
            self.problems.append(
                f"indicator {v['tile']}: the reviewer placed its kan before {s}"
                f"'s discard at {tt:.0f}s"
            )
        for i, v in enumerate(named):
            if i > 0 and v.get("t_first") is None:
                self.problems.append(
                    f"indicator {v['tile']} named by the reviewer was never "
                    "observed: the time of its kan is unknown"
                )
                self.items.append(
                    {
                        "kind": "kan",
                        "text": (
                            f"The reviewer named a later dora indicator {v['tile']}"
                            ", but no frame shows it, so the kan that revealed it "
                            "has no time. Which discard followed that kan?"
                        ),
                    }
                )

    def _named_indicators(self, inds: list[dict]) -> list[dict]:
        """Match reviewer-named indicators to observations in order.

        The reviewer named the indicators in order: each takes the observed entry of
        that tile when there is one (its time and box), else the next observed entry;
        observed indicators beyond the list stay (kans).
        """
        used: set = set()
        fact_tiles = list(self.facts["dora"])
        if len(inds) < len(fact_tiles):
            # a tile named twice in a row is one tile when the wall shows fewer entries
            # than the list (typed twice);
            # with as many entries it is two indicators of one kind, which a kan can
            # reveal
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
        self._place_indicator_times(named)
        # an observed indicator beyond the named ones stays, even of a kind named
        # already: the wall shows it
        return named + [v for k, v in enumerate(inds) if k not in used]

    # ---- 5. the turn sequence
    # -----------------------------------------------------------------------------

    def turn_sequence(self) -> None:
        """Merge ponds and calls; reread unexplained skipped turns."""
        # where the hand ends: a tsumo winner draws the winning tile; after a ron the
        # seat after the loser would play
        end = {
            "tsumo": self.winner,
            "ron": rules.next_seat(self.loser) if self.loser else None,
        }.get(self.result.outcome)
        # the live wall less a tile per kan; a tsumo's winning draw is not in the
        # sequence
        kans = sum(1 for c in self.calls if c.type in ("kan", "ankan", "kakan"))
        wall = rules.LIVE_WALL - kans - (1 if self.result.outcome == "tsumo" else 0)
        exhaustive = self.result.outcome == "draw"
        turns, order_problems = merge(
            self.logs,
            self.calls,
            self.dealer,
            ending=HandEnding(end=end, wall=wall, exhaustive=exhaustive),
        )
        if (
            self.models is not None
            and any("skipped" in p for p in order_problems)
            and dense.skipped_turns(
                order_problems,
                turns,
                self.logs,
                context=DenseContext(
                    entry=self.entry,
                    models=self.models,
                    work_dir=self.dense_work,
                    t0=self.t0,
                    t1=self.t1,
                    problems=self.problems,
                ),
            )
        ):
            turns, order_problems = merge(
                self.logs,
                self.calls,
                self.dealer,
                ending=HandEnding(end=end, wall=wall, exhaustive=exhaustive),
            )
        self.turns, self.order_problems = turns, order_problems
        # a call the turn order shows and no camera did: its tiles are the solver's
        # choice among the legal melds
        for t in turns:
            c = t.call
            if c is not None and c.anchor == "hidden" and c not in self.calls:
                if c.called_tile is None or c.source is None:
                    msg = "A hidden call must identify its discard and source"
                    raise ValueError(msg)
                c.options = meld_options(c.called_tile, c.source, [], four=False)
                self.calls.append(c)

    def end_of_hand(self) -> None:
        """Align the last turn with the result and enforce the live-wall budget."""
        self._after_the_ron()
        for p in self.order_problems:
            self.problems.append(p)
            m = re.match(r"turn (\d+): no discard of ([ESWN])", p)
            if m:
                # a turn whose discard no read found: the log cannot be written without
                # it
                i = int(m.group(1))
                t = (
                    self.turns[i].t
                    if i < len(self.turns)
                    else (self.turns[-1].t if self.turns else self.t1)
                )
                self.items.append(
                    {
                        "kind": "order",
                        "seat": m.group(2),
                        "t": t,
                        "i": i,
                        "text": (
                            f"{m.group(2)} must have discarded around {t:.0f} s "
                            "(its turn came), but no read of its pond found the "
                            "tile, not even a dense one. Which tile did "
                            f"{m.group(2)} discard?"
                        ),
                    }
                )
        self._winners_last_kan()
        self._live_calls()
        self._wall()
        self._unexplained_removals()

    def _after_the_ron(self) -> None:
        """Remove events that occur after the hand-ending ron discard.

        A ron ends the hand on the loser's discard (section 1): nobody discards after
        it, and tiles reaching a pond later are the reveal and the clearing. Turns the
        merge placed after it that happened before it are real turns in the wrong place:
        they stay (the winner's hand is read by time) and the order is asked.
        """
        if (
            self.result.outcome != "ron"
            or self.loser is None
            or self.loser not in rules.SEATS
        ):
            return
        mine = [t for t in self.turns if t.seat == self.loser and t.slot is not None]
        if not mine:
            return
        win_turn = mine[-1]
        after = self.turns[self.turns.index(win_turn) + 1 :]
        late = [t for t in after if t.t > win_turn.t + 1.0]
        if late:
            self.problems.append(
                f"{len(late)} tile(s) seen in the ponds after {self.loser}'s "
                f"winning discard at {win_turn.t:.0f}s are not turns (the "
                "reveal and the clearing): dropped"
            )
            self.turns = [t for t in self.turns if t not in late]
            for i, t in enumerate(self.turns):
                t.i = i
        early = [t for t in after if t.t <= win_turn.t + 1.0]
        if early:
            listed = ", ".join(f"{t.seat}'s discard at {t.t:.0f} s" for t in early)
            self.problems.append(
                f"{len(early)} turn(s) the merge placed after {self.loser}'s "
                f"winning discard at {win_turn.t:.0f}s happened before it ("
                f"{listed}): the order is wrong but the turns are kept"
            )

    def _winners_last_kan(self) -> None:
        """Recover a kan followed by the winner's terminal rinshan draw.

        A kan after the tsumo winner's last discard (the win is the rinshan draw): no
        discard follows it, so the merge left it on no turn (ankan / kakan) or pending
        on the last discard (daiminkan); it is the winner's last turn, one without a
        discard.
        """
        if not self.tsumo_winner:
            return
        attached = {id(t.own_call) for t in self.turns if t.own_call}
        my_last = max(
            (t.t for t in self.turns if t.seat == self.tsumo_winner), default=self.t0
        )
        for c in self.calls:
            if c.seat != self.tsumo_winner or id(c) in attached:
                continue
            if c.type in ("ankan", "kakan") and c.t_window[1] > my_last:
                t_kan = max(c.t_window[1], my_last + 0.5)
            elif c.type == "kan" and self.turns and self.turns[-1].call is c:
                t_kan = max(c.t_window[1], self.turns[-1].t + 0.5)
            else:
                continue
            self.turns.append(
                Turn(len(self.turns), self.tsumo_winner, "kan", None, t_kan, own_call=c)
            )
            self.problems.append(
                f"{c.type} by {self.tsumo_winner} after its last discard at "
                f"{t_kan:.0f}s: the winning tile is the rinshan draw"
            )
            break

    def _live_calls(self) -> None:
        """Keep calls owned by a turn in the reconstructed sequence.

        A call counts only when the caller has a turn that owns it: that turn is where
        the solver takes the called tiles out of their hand (otherwise the winner comes
        out with 13 concealed tiles and a meld).
        """
        owned = {id(t.own_call) for t in self.turns if t.own_call is not None}
        self.live_calls = [c for c in self.calls if id(c) in owned]
        for c in self.calls:
            if id(c) not in owned:
                self.problems.append(
                    f"the {c.type} {''.join(c.tiles)} of {c.seat} at "
                    f"{c.t_first:.0f}s is on no turn of {c.seat} (the discard "
                    "that follows a call was not found): not counted for the "
                    "hand size or the scoring"
                )
        for e in self.unanchored:
            # the camera shows a meld clearly, and no read found the discard it took
            self.items.append(
                {
                    "kind": "call",
                    "seat": e.seat,
                    "t": e.t_first,
                    "tile": e.tiles[e.called_pos or 0],
                    "type": e.type,
                    "tiles": e.tiles,
                    "source": e.source,
                    "text": (
                        f"{e.seat}'s meld camera shows a {e.type} of "
                        f"{' '.join(e.tiles)} from {e.t_first:.0f} s, but no pond "
                        "shows the discard it took, not even in a dense read. What "
                        "is this meld, and whose discard did it take?"
                    ),
                }
            )

    def _wall_draws(self) -> tuple[int, int]:
        """Count wall draws and kans in a turn sequence.

        (wall draws, kans) of the sequence: every draw turn (the dealer's 14th tile
        included), every turn the merge skipped (a missed discard still drew), the
        winning tsumo draw; each kan moved a live tile into the dead wall.
        """
        draws = sum(1 for t in self.turns if t.kind in ("draw", "kan")) + (
            1 if self.tsumo_winner else 0
        )
        draws += sum(1 for p in self.order_problems if "skipped" in p)
        kans = sum(1 for c in self.live_calls if c.type in ("kan", "ankan", "kakan"))
        return draws, kans

    def _wall(self) -> None:
        """Enforce the 70-tile wall limit, including kan replacement draws.

        Wall draws plus kans never exceed 70 (section 1), and an exhaustive draw is
        exactly 70.
        """
        draws, kans = self._wall_draws()
        used = draws + kans
        if used > rules.LIVE_WALL:
            text = (
                f"The turn sequence needs {draws} wall draws and {kans} kan(s),"
                f" {used} tiles, but the live wall holds {rules.LIVE_WALL}: the"
                " sequence holds turns that did not happen (tiles of another "
                "hand, or misread ponds)."
            )
            self.problems.append(text)
        elif self.result.outcome == "draw" and used < rules.LIVE_WALL:
            self.problems.append(
                "The hand ended in an exhaustive draw, which takes all "
                f"{rules.LIVE_WALL} tiles of the live wall, but the sequence "
                f"has {draws} draws and {kans} kan(s): {rules.LIVE_WALL - used}"
                " turn(s) are missing."
            )

    def _unexplained_removals(self) -> None:
        """Find pond removals that no reconstructed call explains.

        A tile left a pond (only a call takes a discard) but no call explains it: the
        next seat played on, so either its call went unseen or the removal was misread.
        A note; the reconstruction keeps the discard.
        """
        taken = assign_calls(self.live_calls, self.logs)
        for s2, sls in self.logs.items():
            for k, sl in enumerate(sls):
                if sl.t_removed is not None and taken[s2][k] is None and sl.row != -1:
                    self.problems.append(
                        f"{s2}'s {sl.tile} left the pond at {sl.t_removed:.0f}s"
                        " and no call explains it"
                    )

    # ---- 6. riichi
    # --------------------------------------------------------------------------

    def _find_missing_riichi(self, seats: set[str]) -> None:
        """Reread missing declarations and retain a question when none is observed."""
        for s in seats:
            mine = [t for t in self.turns if t.seat == s and t.slot is not None]
            if not mine:
                continue
            found = dense.turned_tile(
                s,
                self.turns,
                context=DenseContext(
                    entry=self.entry,
                    models=self.models,
                    work_dir=self.work_dir,
                    t0=self.t0,
                    problems=self.problems,
                ),
            )
            if found is not None:
                found.riichi = True
                continue
            mine[-1].riichi = True
            self.guessed_riichi.add(s)
            src = (
                "Your riichi fact says"
                if self.facts.get("riichi") is not None
                else "The site record says"
            )
            self.items.append(
                {
                    "kind": "riichi",
                    "seat": s,
                    "t": mine[-1].t,
                    "text": (
                        f"{src} {s} declared riichi but no turned tile was seen"
                        " in its pond, not even in a dense read of the moment "
                        "each of its tiles was laid; the last discard ("
                        f"{mine[-1].discard_slot.tile} at {mine[-1].t:.0f}s) is"
                        " assumed. Which discard was the riichi?"
                    ),
                }
            )

    def _reviewed_riichi_turns(self) -> set[str]:
        """Place reviewed declarations on their matching discards."""
        named = set()
        for f in self.facts.get("riichi_turn", []):
            mine = [t for t in self.turns if t.seat == f["seat"] and t.slot is not None]
            near = min(mine, key=lambda t: abs(t.t - f["t"]), default=None)
            if near is None or abs(near.t - f["t"]) > RIICHI_FACT_WINDOW:
                self.problems.append(
                    f"riichi-turn fact for {f['seat']} at {f['t']:.0f}s matches"
                    " no discard of that seat"
                )
                continue
            for t in mine:
                t.riichi = t is near
            named.add(f["seat"])
        return named

    def riichi(self) -> None:
        """Combine authoritative declarations with observed turned discards.

        The site record decides who declared (its deltas carry the sticks); the turned
        tile says when. A reviewer's riichi-turn fact names the declaring discard
        outright.
        """
        site = {site_seat(s, self.entry) for s in self.result.riichi}
        if self.facts.get("riichi") is not None:
            if set(self.facts["riichi"]) != site:
                self.problems.append(
                    "your riichi fact "
                    f"{sorted(self.facts['riichi']) or 'nobody'} contradicts "
                    f"the site record {sorted(site) or 'nobody'}: the fact is "
                    "used"
                )
            site = set(self.facts["riichi"])
        self.site_riichi = site
        named = self._reviewed_riichi_turns()
        turned = {t.seat for t in self.turns if t.riichi}
        if turned != site:
            # a turned tile of another seat is a tile laid askew, noted but not asked; a
            # site riichi with no turned
            # tile lacks its turn: the last discard of that seat is the guess and the
            # reviewer is asked
            self.problems.append(
                f"riichi seats from ponds {sorted(turned)} vs site {sorted(site)}"
            )
            for t in self.turns:
                t.riichi = t.riichi and t.seat in site
            self._find_missing_riichi(site - turned)
        # a turned tile right after the seat's own discard was called: the declaration
        # was that called tile or this one
        for t in self.turns:
            if not t.riichi or t.seat in named or t.seat in self.guessed_riichi:
                continue
            prev = [
                x
                for x in self.turns
                if x.seat == t.seat and x.i < t.i and x.slot is not None
            ]
            if prev and prev[-1].call is not None:
                self.riichi_alternatives.append((t, prev[-1]))

    # ---- 7. haipai and draws
    # ----------------------------------------------------------------------------------

    def pond_replacements(self) -> None:
        """Acquire unresolved pond correspondences before building discard costs.

        Each request is attempted once. Continuous geometry may substitute its
        dense posterior for overlapping sparse evidence, without adding turns or
        calls; ambiguous acquisition remains a specific review item. Reading
        failures propagate so broken acquisition cannot become missing evidence.
        """
        requests = pond_evidence.replacement_requests(
            self.turns, self.entry, self.facts, self.t0, self.t1
        )
        if self.models is None:
            return
        for request in requests:
            if request["acquired"] or request["window"] is None:
                continue
            slot = next(
                t.slot
                for t in self.turns
                if t.seat == request["seat"]
                and t.slot is not None
                and t.slot.id == request["slot_id"]
            )
            readings = pond_evidence.read_replacement(
                request, self.models, self.dense_work
            )
            consumed = pond_evidence.consume_replacement(slot, request, readings)
            if consumed:
                self.problems.append(
                    f"continuous dense pond evidence for {request['seat']} at "
                    f"{request['t']:.0f}s replaced its overlapping sparse "
                    "observation"
                )

    def solve(self) -> None:
        """Construct one hand with indicators and site score in the same solve phase.

        Reconciliation proposes constraints; only a complete legal solution may
        replace the current candidate. Nothing changes tiles after this phase.
        """
        self._fit_evidence()
        indicators = tuple(self.dora)
        self.kan_indicators()
        if self.sol.ok and tuple(self.dora) != indicators:
            self.model.indicators = list(self.dora)
            self.sol = self._solve(prior=self.sol)
        self.check_score()

    def _fit_evidence(self) -> None:
        """Fit legal hands, reread uncertain draws, then attempt repair."""
        model = HandModel(
            self.dealer,
            {s: [] for s in rules.SEATS},
            self.dora,
            tsumo_winner=self.tsumo_winner,
            ura=self.ura,
        )
        self.unknown_kans = {
            id(c) for c in self.calls if c.type == "ankan" and "?" in c.tiles
        }
        self.melds_before: dict[str, dict[int, int]] = {}
        for s in rules.SEATS:
            model.turns[s], self.melds_before[s] = seat_turns_of(
                self.turns, s, self.dealer
            )
        for corner in ("TL", "TR", "BL", "BR"):
            seat = seat_of(self.entry, corner)
            hev, dev, _ = hand_evidence(
                seat,
                model.turns[seat],
                in_window(self.obs.get(f"hand:{corner}", []), self.t0, self.t1),
                self.melds_before[seat],
                role=HandRole(
                    dealer=seat == self.dealer, wins_by_tsumo=seat == self.tsumo_winner
                ),
            )
            model.hand_ev += hev
            model.draw_ev += dev
        self._apply_hand_facts(model)
        self._result_constraint(model)
        self.model = model
        sol = self._solve()
        if sol.ok and self.models is not None:
            low = draws_to_reread(sol, model)
            evidence_before = (len(model.hand_ev), len(model.draw_ev))
            dense.draws(
                low,
                model,
                self.turns,
                self.melds_before,
                context=DenseContext(
                    entry=self.entry,
                    models=self.models,
                    work_dir=self.dense_work,
                    t0=self.t0,
                    problems=self.problems,
                ),
            )
            # Partial views also change the objective, even when none pins a draw.
            if (len(model.hand_ev), len(model.draw_ev)) != evidence_before:
                sol = self._solve(prior=sol)
        if sol.status == "infeasible":
            sol = self._repair()
        if sol.ok and self.riichi_alternatives:
            sol = self._riichi_on_a_called_tile(sol)
        self.sol = sol

    def kan_indicators(self) -> None:
        """Require one indicator per kan in addition to the initial indicator.

        Every kan reveals an indicator (section 1): the log carries 1 + kans of them,
        always. One that no view showed (the dead wall is often outside the overhead
        crop) is a `dora` question, and the log carries the best provisional placeholder
        meanwhile: a kind with copies left. A score mismatch must never change an
        indicator to manufacture a matching score.
        """
        kans = sorted(
            (c for c in self.live_calls if c.type in ("kan", "ankan", "kakan")),
            key=lambda c: c.t_first,
        )
        missing = 1 + len(kans) - len(self.dora)
        if missing <= 0:
            return
        sol, seen = self.sol, list(self.dora)
        used = Counter(
            x
            for x in [
                *(x for h in sol.haipai.values() for x in h),
                *sol.draws.values(),
                *sol.draws2.values(),
                *self.dora,
                *self.ura,
            ]
            if x in rules.KINDS or x in rules.REDS
        )

        def rank(x: str) -> tuple:
            left = rules.max_count(x) - used[x]
            return (-left, rules.KINDS.index(x))

        lost = bool(self.facts.get("lost_dora"))
        guesses = []
        for _ in range(missing):
            x = min((x for x in rules.KINDS if used[x] < rules.max_count(x)), key=rank)
            used[x] += 1
            self.dora.append(x)
            self.inds.append(
                {
                    "tile": x,
                    "t_first": None,
                    "region": None,
                    "conf": 0.0,
                    "lost": True,
                    "human": lost,
                }
            )
            guesses.append(x)
        # the kans no seen indicator explains (the same pairing as reconcile_kans)
        explained: set = set()
        for ind in self.inds[1 : len(seen)]:
            t = ind.get("t_first")
            k = next(
                (
                    c
                    for c in kans
                    if t is not None
                    and id(c) not in explained
                    and c.t_window[0] - KAN_MATCH <= t <= c.t_first + KAN_MATCH
                ),
                None,
            )
            if k is not None:
                explained.add(id(k))
        bare = [c for c in kans if id(c) not in explained]
        if lost:
            self.problems.append(
                f"dora indicator(s) {' '.join(guesses)}: no view shows them and"
                " the reviewer cannot tell; written as the rules' guess"
            )
            return

        def describe_kan(call: Call) -> str:
            tile = next((x for x in call.tiles if x not in ("X", "?")), "?")
            return (
                f"the {call.type} of {rules.plain(tile)} "
                f"by {call.seat} at {call.t_first:.0f}s"
            )

        what = "; ".join(describe_kan(c) for c in bare) or "the kans of this hand"
        head = "No dora indicator was seen, and " if not seen else ""
        self.items.append(
            {
                "kind": "dora",
                "t": bare[0].t_first if bare else (self.last_discard or self.t1),
                "tiles": seen,
                "guess": guesses,
                "text": (
                    f"{head}{what} revealed a dora indicator that no view shows "
                    "(the dead wall is outside the overhead crop). The log carries "
                    f"{' '.join(guesses)} as a guess. Enter all the dora indicators"
                    " in order: "
                    f"{(' '.join(seen) + ' (seen), then ' if seen else '')}the "
                    "kan's."
                ),
            }
        )

    def unseen_tiles(self) -> None:
        """Expose rule-selected tiles without supporting views for review.

        Every tile the rules chose and nothing showed is a question (section 6, `lost`),
        with the choice as the guess: a draw no frame covers, the caller's tiles of a
        call the turn order implies and no camera read, the tile of an ankan no camera
        named. A reviewer's `lost` fact (Can't tell) closes a draw's question.
        """
        sol, model = self.sol, self.model
        if not sol.ok:
            return  # the conflict question comes first
        lost_keys = {
            k
            for k in (turn_key(model, f, 3.0) for f in self.facts.get("lost", []))
            if k is not None
        }
        for (s, j), tile in sorted(sol.draws.items()):
            if (
                tile is None
                or (s, j) in lost_keys
                or (s, j) in model.facts.draws
                or not unseen_draw(model, sol, s, j)
            ):
                continue
            st = model.turns[s][j] if j < len(model.turns[s]) else None
            t = st.t_discard if st else (self.turns[-1].t if self.turns else self.t1)
            self.items.append(
                {
                    "kind": "draw",
                    "seat": s,
                    "j": j,
                    "t": t,
                    "tile": tile,
                    "runner_up": sol.runner_up.get((s, j)),
                    "margin": sol.margins.get((s, j), 0.0),
                    "text": (
                        f"No frame shows the tile {s} drew in the turn ending at "
                        f"{t:.0f}s, and nothing later pins it: the log carries "
                        f"{tile}, the rules' guess."
                    ),
                }
            )
        for c in self.live_calls:
            if c.human:
                continue
            if c.anchor == "hidden":
                why = (
                    f"no camera shows this meld: the turn order says {c.seat} "
                    f"called {c.called_tile} from its {c.source} at "
                    f"{c.t_first:.0f}s, and the log carries {c.type} "
                    f"{' '.join(c.tiles)}, the rules' guess"
                )
            elif id(c) in self.unknown_kans:
                why = (
                    f"no camera names the tile of {c.seat}'s ankan at "
                    f"{c.t_first:.0f}s: the log carries {' '.join(c.tiles)}, "
                    "the rules' guess"
                )
            else:
                continue
            self.items.append(
                {
                    "kind": "call",
                    "seat": c.seat,
                    "t": c.t_first,
                    "tile": c.called_tile or c.tiles[0],
                    "type": c.type,
                    "tiles": list(c.tiles),
                    "source": c.source,
                    "text": f"{why[0].upper()}{why[1:]}. What was the meld?",
                }
            )

    def _solve(self, *, prior: Solution | None = None) -> Solution:
        sol = self.model.solve(
            time_limit=self.time_limit,
            confidence_timeout=self.confidence_remaining,
            prior=prior,
        )
        self.confidence_remaining = max(
            0.0, self.confidence_remaining - sol.confidence_seconds
        )
        if sol.ok:
            apply_choices(
                sol, self.model, self.turns, self.unknown_kans
            )  # the calls take the solver's choices before scoring
        return sol

    def _result_constraint(self, model: HandModel) -> None:
        """Constrain final states to the site's win or tenpai result.

        The site record says who won: the winner's final hand is a winning hand; at a
        draw the tenpai seats are tenpai (4.8 "The result as a constraint").
        """
        w = self.winner
        if w and self.result.outcome == "tsumo":
            j = len(model.turns[w])
            model.win = WinSpec(w, j, 4 - self.melds_before[w].get(j, 0))
        elif w and self.loser and self.result.outcome == "ron":
            lt = [t for t in self.turns if t.seat == self.loser and t.slot is not None]
            stl = next(
                (x for x in model.turns[self.loser] if lt and x.t_discard == lt[-1].t),
                None,
            )
            if stl is not None:
                # the winner's hand at the winning discard, by time: turns placed after
                # it that happened before it count
                jwin = (
                    sum(1 for t in self.turns if t.seat == w and t.t <= lt[-1].t + 1.0)
                    - 1
                )
                model.win = WinSpec(
                    w,
                    jwin,
                    4 - self.melds_before[w].get(jwin, 0),
                    ron_from=(self.loser, stl.j),
                )
        elif self.result.outcome == "draw":
            for s in rules.SEATS:
                if (
                    site_seat_name(s, self.entry) in self.result.tenpai
                    and model.turns[s]
                ):
                    j = len(model.turns[s]) - 1
                    model.tenpai.append((s, j, 4 - self.melds_before[s].get(j, 0)))

    def _apply_discard_facts(self, model: HandModel) -> None:
        """Keep confirmed discard identities hard even during repair searches."""
        for f in self.facts.get("discard", []) + self.facts.get("missing_discard", []):
            mine = model.turns.get(f["seat"], [])
            if mine and f.get("t") is not None:
                nearest = min(mine, key=lambda turn: abs(turn.t_discard - f["t"]))
                if abs(nearest.t_discard - f["t"]) <= DISCARD_FACT_WINDOW:
                    # A one-hot posterior alone is still a soft cost in repair
                    # mode. A human identity must remain a hard constraint.
                    nearest.discard, nearest.discard_p = f["tile"], None

    def _apply_hand_facts(self, model: HandModel) -> None:
        def add_soft_hand(f: dict, state: int, t: float) -> None:
            counts = Counter(f["tiles"])
            model.hand_ev.append(
                HandEvidence(
                    f["seat"],
                    state,
                    after_draw=False,
                    e=np.array([counts[tile] for tile in TILES], dtype=float),
                    weight=3.0,
                    t0=t,
                    t1=t,
                )
            )

        self._apply_discard_facts(model)
        for f in self.facts.get("haipai", []):
            if f.get("soft"):
                add_soft_hand(f, -1, self.t0)
            else:
                model.facts.haipai[f["seat"]] = f["tiles"]
        for f in self.facts.get("draw", []):
            key = turn_key(model, f, 10.0)
            if key is None:
                self.problems.append(
                    f"draw fact {f['tile']} of {f['seat']} matches no draw "
                    "turn: ignored"
                )
                continue
            model.facts.draws[key] = f["tile"]
        for f in self.facts.get("final_hand", []):
            # binds only when it is a complete concealed hand at the end (13 - 3 per
            # meld, +1 for a tsumo winner)
            s = f["seat"]
            base = 13 - 3 * sum(
                1 for c in self.live_calls if c.seat == s and c.type != "kakan"
            )
            tsumo_win = s == self.tsumo_winner
            want = base + (1 if tsumo_win else 0)
            excl = (
                tsumo_win and len(f["tiles"]) == base
            )  # the hand without the winning draw: bound before it
            if not excl and len(f["tiles"]) != want:
                self.problems.append(
                    (
                        f"final-hand fact for {s} has {len(f['tiles'])} tiles, "
                        f"expected {want}"
                    )
                    + (f" or {base} without the winning tile" if tsumo_win else "")
                    + ": ignored"
                )
            elif f.get("soft"):
                # Explicit uncertain evidence can guide a reconstruction but cannot fix
                # its tiles.
                last = len(model.turns[s]) - 1 + (1 if (tsumo_win and not excl) else 0)
                add_soft_hand(f, last, self.t1)
            else:
                model.facts.final[s] = f["tiles"]
                if excl:
                    model.facts.final_excl[s] = True

    def _repair(self) -> Solution:
        """Find the cheapest pond rereadings that allow a legal reconstruction.

        No legal reconstruction with every discard as read: find the cheapest
        re-readings of the ponds that make the hand legal (every discard a choice over
        all kinds, costed by its posterior), take them as the discards, and solve again.
        Only when no re-reading helps is the hand a conflict.
        """
        self.model.repair = True
        found = self.model.solve(time_limit=self.time_limit, margins=False)
        self.model.repair = False
        if found.ok:
            for (s, j), tile in found.discards.items():
                st = self.model.turns[s][j]
                self.problems.append(
                    f"{s}'s discard at {st.t_discard:.0f}s read {st.discard}: "
                    f"the hand is legal only as {tile}, which is used"
                )
                self.repaired[(s, j)] = tile
                st.discard, st.discard_p = tile, None
            sol = self._solve()
            if sol.ok:
                sol.optimal = sol.optimal and found.optimal
                sol.status = "repaired"
                return sol
            found = sol
        if found.status != "infeasible":
            return found
        culprits = diagnose(self.model)
        self.problems.append(
            "solver: no legal reconstruction: "
            + ("; ".join(c.text for c in culprits) or "no single decision explains it")
        )
        self.items.append(self._conflict_question(culprits))
        return found

    def _conflict_question(self, culprits: list[Culprit]) -> dict:
        """Ask the review question that can settle a reconstruction conflict.

        The one decision a conflict hinges on, asked with the control that answers it;
        the over-counted kinds with their sources when no single decision explains it.
        """
        item = {
            "kind": "conflict",
            "t": self.turns[-1].t if self.turns else self.t1,
            "over": over_count(self.turns, self.calls, self.dora, self.entry),
        }
        if not culprits:
            return {
                **item,
                "culprit": None,
                "text": (
                    "No reconstruction of this hand is legal, and no single "
                    "discard, call or fact explains why. The kinds below appear "
                    "more than four times; mark the reading that is wrong."
                ),
            }
        c = culprits[0]
        item.update(
            culprit=c.kind,
            seat=c.seat,
            fact_seat=c.seat if c.kind == "fact" else None,
            fact_kind=c.fact,
        )
        if c.kind == "meld" and c.seat is not None:
            call = next(
                (
                    t.own_call
                    for t in self.turns
                    if t.seat == c.seat
                    and t.own_call is not None
                    and (
                        st := next(
                            (x for x in self.model.turns[c.seat] if x.t_discard == t.t),
                            None,
                        )
                    )
                    is not None
                    and st.j == c.j
                ),
                None,
            )
            if call is not None:
                item.update(
                    t=call.t_first,
                    type=call.type,
                    tiles=call.tiles,
                    source=call.source,
                    tile=call.called_tile,
                    text=(
                        f"{c.seat}'s {call.type} of {' '.join(call.tiles)} (from "
                        f"its {call.source or 'own hand'}) at {call.t_first:.0f} s "
                        f"cannot be: {c.text.split(': ', 1)[-1]}. What is this "
                        "meld?"
                    ),
                )
                return item
        if c.kind == "result":
            return {
                **item,
                "text": (
                    f"{c.seat}'s winning hand cannot be completed from what was "
                    f"read. Enter the hand {c.seat} revealed (a call or a discard "
                    "of that seat is read wrong)."
                ),
            }
        if c.kind == "riichi" and c.seat is not None and c.j is not None:
            st = self.model.turns[c.seat][c.j]
            return {
                **item,
                "t": st.t_discard,
                "text": f"{c.text}. Which discard was {c.seat}'s riichi?",
            }
        return {**item, "text": c.text[0].upper() + c.text[1:] + ". Check that fact."}

    def _riichi_on_a_called_tile(self, sol: Solution) -> Solution:
        """Resolve the declaration discard using the riichi freeze.

        The declaration was the called tile or the turned one (section 1): the freeze
        decides.
        """
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
                self.items.append(
                    {
                        "kind": "riichi",
                        "seat": turned.seat,
                        "t": chosen.t,
                        "text": (
                            f"{turned.seat}'s turned tile at {turned.t:.0f}s "
                            f"follows its discard at {called.t:.0f}s that was "
                            "called: the declaration was one of the two, and the "
                            "hand fits both about as well. "
                            f"{chosen.discard_slot.tile} at {chosen.t:.0f}s is "
                            "assumed. Which discard was the riichi?"
                        ),
                    }
                )
            self.problems.append(
                f"riichi of {turned.seat}: the called discard at {called.t:.0f}"
                f"s or the turned one at {turned.t:.0f}s; "
                f"{('the called one' if chosen is called else 'the turned one')}"
                " fits the hand better"
            )
        return sol

    # ---- 8. the result
    # --------------------------------------------------------------------------

    def check_score(self) -> None:
        """Reconcile the winning reconstruction with the site score and review facts."""
        self.score = None
        sol, result, winner = self.sol, self.result, self.winner
        if not sol.ok or not winner or result.outcome not in ("ron", "tsumo"):
            return
        jw = len(self.model.turns[winner])
        concealed, win_tile, drawn_left_in, j_hand = self._winning_hand(sol)
        if result.outcome == "ron" and self.turns and j_hand != jw - 1:
            self.problems.append(
                f"ron by {winner} on {self.loser}: the hand's last turn is "
                f"{self.turns[-1].seat}'s at {self.turns[-1].t:.0f}s, after the"
                " winning discard; the winner's hand is read at their turn "
                f"{j_hand}, the last one at or before that discard"
            )
        melds = scoring_melds(self.live_calls, winner)
        # the winner's concealed tiles beside the melds and the winning tile must be a
        # hand of 13 (section 1); the
        # solver took the same calls' tiles out of the hand, so a mismatch is the
        # decoder's own inconsistency: a
        # note (the score then fails and asks for the revealed hand)
        want = concealed_size(melds, drawn_left_in=drawn_left_in)
        if len(concealed) != want:
            self.problems.append(
                f"the winner's hand does not add up: {len(concealed)} concealed"
                f" tiles beside {len(melds)} meld(s), {want} expected. The "
                "melds and the turn sequence disagree."
            )
        if not win_tile:
            return
        score_of = self._scorer(melds)
        sc = score_of(concealed, win_tile, self.dora)
        if (
            result.outcome == "tsumo"
            and not self._fits(sc)
            and (winner, jw) not in self.model.facts.draws
        ):
            concealed, win_tile, sc = self._winning_tile_from_the_site(
                concealed, win_tile, sc, melds, jw
            )
        if not self._fits(sc) and winner not in self.model.facts.final:
            concealed, win_tile, sc, melds = self._next_hand_from_the_site(
                concealed, win_tile, sc, melds, j_hand
            )
        if not self._fits(sc) and winner not in self.model.facts.final:
            concealed, sc = self._red_five_from_the_site(
                concealed, win_tile, sc, melds, j_hand
            )
        # Score the actual accepted solution, including any meld choices it
        # changed. A proposal's predicted score is never the final record.
        concealed, win_tile, _, _ = self._winning_hand(self.sol)
        if win_tile is None:
            message = "Accepted winning solution has no winning tile"
            raise ValueError(message)
        melds = scoring_melds(self.live_calls, winner)
        sc = self._scorer(melds)(concealed, win_tile, self.dora)
        self.score = {
            "ok": sc.ok,
            "han": sc.han,
            "fu": sc.fu,
            "yaku": sc.yaku,
            "error": sc.error,
            "site": [result.han, result.fu],
            "match": matches_site(sc, result.han, result.fu),
            "concealed": concealed,
            "win_tile": win_tile,
            "melds": melds,
            "context": self.context,
        }
        self._score_review(sc, concealed, win_tile, melds, jw)

    def _score_review(
        self,
        sc: ScoreResult,
        concealed: list[str],
        win_tile: str,
        melds: list[dict],
        jw: int,
    ) -> None:
        """Preserve score disagreements and unresolved winning-tile alternatives."""
        result, winner = self.result, self.winning_seat
        ura_explains = (
            winner in self.site_riichi
            and not self.ura
            and sc.ok
            and sc.han is not None
            and result.han is not None
            and sc.han < result.han
        )
        if not matches_site(sc, result.han, result.fu) and not ura_explains:
            found = ", ".join(sc.yaku) if sc.yaku else (sc.error or "no yaku")
            item = {
                "kind": "result",
                "seat": winner,
                "t": self.turns[-1].t if self.turns else self.t1,
                "tiles": concealed,
                "win_tile": win_tile,
                "site": [result.han, result.fu],
                "same_payment": False,
            }
            if sc.ok and sc.han and sc.fu is not None and result.han:
                # the site's han/fu can be a scorer's slip, the costless one most of
                # all: a pair that pays the same
                how = {
                    "dealer": winner == self.dealer,
                    "tsumo": result.outcome == "tsumo",
                }
                mine, site = (
                    score_text(sc.han, sc.fu, **how),
                    score_text(result.han, result.fu or 0, **how),
                )
                item.update(
                    han=sc.han,
                    fu=sc.fu,
                    same_payment=payment(sc.han, sc.fu, **how)
                    == payment(result.han, result.fu or 0, **how),
                )
                if item["same_payment"]:
                    item["text"] = (
                        f"The reconstructed winning hand scores {sc.han}/"
                        f"{sc.fu} [{found}], which pays what the site's "
                        f"{result.han}/{result.fu} pay ({site}): the site's "
                        "han/fu may be the scorer's slip. Confirm that the site"
                        f" is wrong, or enter the hand {winner} revealed."
                    )
                else:
                    item["text"] = (
                        f"The reconstructed winning hand scores {sc.han}/"
                        f"{sc.fu} [{found}], {mine}; the site says {result.han}"
                        f"/{result.fu}, {site}. Enter the hand {winner} "
                        "revealed, or confirm that the site is wrong."
                    )
            else:
                item["text"] = (
                    f"The reconstructed winning hand scores {sc.han}/{sc.fu} ["
                    f"{found}]; the site says {result.han}/{result.fu}. Enter "
                    f"the hand {winner} revealed."
                )
            self.items.append(item)
        elif (
            result.outcome == "tsumo"
            and (winner, jw) not in self.model.facts.draws
            and low_margin(self.sol.margins.get((winner, jw)))
        ):
            # the tile the log ends on, read almost as well as another that scores the
            # same: the one question
            alt = self.sol.runner_up.get((winner, jw))
            if (
                alt
                and alt != win_tile
                and self._fits(self._scorer(melds)(concealed, alt, self.dora))
            ):
                self.items.append(
                    {
                        "kind": "result",
                        "seat": winner,
                        "t": self.turns[-1].t if self.turns else self.t1,
                        "text": (
                            f"{winner}'s winning tile is uncertain: {win_tile} or "
                            f"{alt}, which both complete the hand with the site's "
                            f"{result.han}/{result.fu}, and the reveal reads "
                            f"neither clearly. Which tile did {winner} win on?"
                        ),
                        "tiles": concealed,
                        "win_tile": win_tile,
                        "candidates": [alt],
                    }
                )

    def _winning_hand(self, sol: Solution) -> tuple[list[str], str | None, bool, int]:
        """Extract the concealed hand and winning tile from a solution.

        The winner's hand in a solution: its concealed tiles (the winning tile out), the
        winning tile, whether an unknown winning draw is still among them, and the state
        j that holds the hand (a tsumo: after the winning draw; a ron: the winner's last
        turn at or before the winning discard, by time, not by position in the merged
        sequence).
        """
        winner = self.winning_seat
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
        if lt and self.loser is not None:
            # the winning tile is the loser's last discard as the reconstruction has it
            last = lt[-1]
            jwin = (
                sum(1 for t in self.turns if t.seat == winner and t.t <= last.t + 1.0)
                - 1
            )
            stl = next(
                (x for x in self.model.turns[self.loser] if x.t_discard == last.t), None
            )
            win_tile = (
                sol.discards.get((self.loser, stl.j)) if stl else None
            ) or last.discard_slot.tile
        return (
            list(sol.hands.get((winner, jwin), sol.hands.get((winner, jw - 1), []))),
            win_tile,
            False,
            jwin,
        )

    def _bind_winning_hand(
        self, concealed: list[str], win_tile: str, j_hand: int
    ) -> None:
        """Bind the winner's concealed and winning states to the site score.

        Bind the winner's hand the site's score requires (the concealed part, and for a
        tsumo the state with the winning draw), so the solve that follows, margins and
        all, keeps it and the draws that brought it follow. Not a reviewer's fact: the
        confidence rows do not mark it human.
        """
        winner = self.winning_seat
        if self.result.outcome == "tsumo":
            self.model.bound_hands = [
                (winner, j_hand - 1, list(concealed)),
                (winner, j_hand, [*list(concealed), win_tile]),
            ]
        else:
            self.model.bound_hands = [(winner, j_hand, list(concealed))]

    def _next_hand_from_the_site(
        self,
        concealed: list[str],
        win_tile: str,
        sc: ScoreResult,
        melds: list[dict],
        j_hand: int,
    ) -> tuple[list[str], str, ScoreResult, list[dict]]:
        """Search successive reconstructions for a hand matching the score.

        The next-best reconstructions: the winner's hand of each is forbidden in turn
        (up to NEXT_HANDS, within NEXT_BUDGET of the best), and the first whose winning
        hand scores the site's value is the hand: bound, and the hand solved again
        (margins and all).
        """
        winner, result, model = self.winning_seat, self.result, self.model
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
                    self.problems.append(
                        "the best reconstruction's winning hand scores "
                        f"{sc.han}/{sc.fu}, the site {result.han}/{result.fu}: "
                        "the next best that scores it ("
                        f"{alt.objective - best:.1f} costlier) is "
                        f"{' '.join(sorted(conc2))} + {win2}"
                    )
                    return conc2, win2, sc2, melds2
                model.bound_hands = []
                break
            hand = sorted(alt.hands.get((winner, j_hand), []))
        model.forbidden_hands.clear()
        apply_choices(self.sol, model, self.turns, self.unknown_kans)
        return concealed, win_tile, sc, melds

    def _scorer(
        self, melds: list[dict]
    ) -> Callable[[list[str], str, list[str]], ScoreResult]:
        """Build the winning-hand scorer with turn-dependent situational yaku.

        The scoring function of this win, with its situational yaku from the turn
        sequence.
        """
        winner, result = self.winning_seat, self.result
        my_turns = [t for t in self.turns if t.seat == winner]
        riichi_i = next((t.i for t in my_turns if t.riichi), None)
        ippatsu = double_riichi = False
        if (
            winner in self.site_riichi
            and riichi_i is not None
            and winner not in self.guessed_riichi
        ):
            after = [t for t in self.turns if t.i > riichi_i]
            ippatsu = all(t.seat != winner for t in after) and all(
                t.kind == "draw" and t.call is None for t in after
            )
            # a riichi on the seat's first discard with no call (nor kan) by anyone
            # before it
            double_riichi = riichi_i == my_turns[0].i and all(
                t.kind == "draw" and t.call is None for t in self.turns[:riichi_i]
            )
        # the last tile of the wall: the winning draw is the 70th (haitei), the winning
        # discard follows it (houtei)
        haitei = sum(self._wall_draws()) >= rules.LIVE_WALL
        # rinshan kaihou: the winner kanned after its last discard and drew the winning
        # tile from the dead wall
        rinshan = (
            result.outcome == "tsumo"
            and bool(my_turns)
            and my_turns[-1].slot is None
            and my_turns[-1].own_call is not None
        )
        self.context = {
            "ippatsu": ippatsu,
            "haitei": haitei,
            "rinshan": rinshan,
            "ura": self.ura,
        }
        round_wind = "ESW"[self.entry["kyoku"] // 4]

        def score_of(conc: list[str], wt: str, dora: list[str]) -> ScoreResult:
            return score_hand(
                conc,
                wt,
                melds,
                context=WinContext(
                    tsumo=result.outcome == "tsumo",
                    riichi=winner in self.site_riichi,
                    seat=winner,
                    round_wind=round_wind,
                    dora=dora,
                    ura=self.ura,
                    ippatsu=ippatsu,
                    haitei=haitei,
                    rinshan=rinshan,
                    double_riichi=double_riichi,
                ),
            )

        return score_of

    def _fits(self, r: ScoreResult) -> bool:
        """Check the site score while allowing for still-unseen ura indicators.

        The site's score; with a riichi win whose ura is not known yet, any winning hand
        whose han do not exceed the site's (ura may add the rest) with the right fu (ura
        never change the fu).
        """
        result = self.result
        if matches_site(r, result.han, result.fu):
            return True
        return (
            self.winner in self.site_riichi
            and not self.ura
            and r.ok
            and r.han is not None
            and result.han is not None
            and r.han <= result.han
            and r.fu == result.fu
        )

    def _reveal_views(self) -> list[dict]:
        """Collect winner hand views after the last discard.

        The winner's hand views after the last discard: the reveal.
        """
        t_last = self.turns[-1].t if self.turns else self.t1
        return [
            o
            for o in self.obs.get(
                f"hand:{corner_of(self.entry, self.winning_seat)}", []
            )
            if o["t0"] >= t_last and o["n_used"] > 0
        ]

    def _reveal_mass(self, x: str) -> float:
        """How strongly the reveal reads tile x."""
        return sum(
            sl["p"][CLASSES.index(x)] for o in self._reveal_views() for sl in o["slots"]
        )

    def _reveal_extra(self, x: str, concealed: list[str]) -> float:
        """Measure reveal support for a tile beyond the concealed hand.

        How strongly the reveal shows tile x beyond the concealed hand: over the views
        of the whole hand with the winning tile (one tile more than the concealed part),
        the reading of x less the copies the hand already holds — the winning tile, not
        a tile of its kind inside the hand. The plain reading without such a view.
        """
        k = CLASSES.index(x)
        full = [o for o in self._reveal_views() if o["count"] == len(concealed) + 1]
        if not full:
            return self._reveal_mass(x)
        held = sum(1 for t in concealed if t == x)
        return sum(max(0.0, sum(sl["p"][k] for sl in o["slots"]) - held) for o in full)

    def _five_swaps(
        self,
        concealed: list[str],
        win_tiles: list[str],
        melds: list[dict],
        score_of: Callable[[list[str], str, list[str]], ScoreResult],
    ) -> list[tuple]:
        """Find red-five substitutions compatible with the site's score.

        The winner's fives read plain for red or the reverse, with the winning tiles
        that then score the site's value: [(five, swapped, concealed after, winning
        tile)]. One red of a suit exists: a red seen anywhere else (a meld, a pond, an
        indicator, the winning tile) cannot be in the hand.
        """
        seen = (
            [x for m in melds for x in m["tiles"]]
            + self.dora
            + self.ura
            + [sl.tile for sls in self.logs.values() for sl in sls]
        )
        out = []
        for i, x in enumerate(concealed):
            alt = SWAP_FIVE.get(x)
            if alt is None or x in concealed[:i]:
                continue
            conc2 = [*concealed[:i], alt, *concealed[i + 1 :]]
            if alt in rules.REDS and (alt in seen or conc2.count(alt) > 1):
                continue
            out += [
                (x, alt, conc2, w)
                for w in win_tiles
                if not (alt in rules.REDS and w == alt)
                and self._fits(score_of(conc2, w, self.dora))
            ]
        return out

    def _winning_tile_from_the_site(
        self,
        concealed: list[str],
        win_tile: str,
        sc: ScoreResult,
        melds: list[dict],
        jw: int,
    ) -> tuple[list[str], str, ScoreResult]:
        """Constrain an unseen tsumo tile using the authoritative score.

        The site record as a constraint: an unseen tsumo tile is the tile that makes the
        hand win with the site's score; among several, the one the reveal frames show
        most. A five read plain for red (or the reverse) changes the han by one: with no
        tile fitting as read, each swap is tried with each tile.
        """
        score_of = self._scorer(melds)
        winner = self.winner
        original_concealed = list(concealed)
        used = Counter(
            rules.plain(x)
            for x in concealed
            + [x for m in melds for x in m["tiles"]]
            + self.dora
            + self.ura
        )
        for sls in self.logs.values():
            for sl in sls:
                used[rules.plain(sl.tile)] += 1
        t_last = self.turns[-1].t if self.turns else self.t1
        mass = self._reveal_mass
        tiles = [
            x
            for x in CLASSES
            if x not in ("X", "none") and used[rules.plain(x)] < MAX_PLAIN_TILE_COPIES
        ]
        cands = [x for x in tiles if self._fits(score_of(concealed, x, self.dora))]
        if not cands:
            swaps = self._five_swaps(concealed, tiles, melds, score_of)
            if swaps:
                # several fives could be the red one: the reveal frames rank them
                opts = sorted(
                    {(a, b) for a, b, _, _ in swaps},
                    key=lambda ab: -(mass(ab[1]) - mass(ab[0])),
                )
                x, alt = opts[0]
                concealed = next(c2 for a, b, c2, _ in swaps if (a, b) == (x, alt))
                cands = sorted({x2 for a, b, _, x2 in swaps if (a, b) == (x, alt)})
        extra = {x: self._reveal_extra(x, concealed) for x in cands}
        cands.sort(key=lambda x: -extra[x])
        if not cands:
            return original_concealed, win_tile, sc
        old_bounds = list(self.model.bound_hands)
        self._bind_winning_hand(concealed, cands[0], jw)
        sol2 = self._solve(prior=self.sol)
        if not sol2.ok or sol2.objective - self.sol.objective > NEXT_BUDGET:
            self.model.bound_hands = old_bounds
            apply_choices(self.sol, self.model, self.turns, self.unknown_kans)
            return original_concealed, win_tile, sc
        self.sol = sol2
        win_tile = cands[0]
        concealed = list(sol2.hands.get((winner, jw), []))
        if win_tile in concealed:
            concealed.remove(win_tile)
        others = ", ".join(cands[1:])
        self.problems.append(
            (
                f"the winning tile was not seen: {cands[0]} makes the hand win with"
                " the site's score"
            )
            + (f" (also possible: {others})" if others else "")
        )
        if len(cands) > 1 and extra[cands[0]] < 2 * extra[cands[1]]:
            self.items.append(
                {
                    "kind": "result",
                    "seat": winner,
                    "t": t_last,
                    "text": (
                        "The winning tile was not seen. These tiles make the hand "
                        f"win with the site's score: {', '.join(cands)}; {cands[0]}"
                        " is used. Confirm the final hand and the winning tile."
                    ),
                    "tiles": concealed,
                    "win_tile": win_tile,
                    "candidates": cands,
                }
            )
        return concealed, win_tile, score_of(concealed, win_tile, self.dora)

    def _red_five_from_the_site(
        self,
        concealed: list[str],
        win_tile: str,
        sc: ScoreResult,
        melds: list[dict],
        j_hand: int,
    ) -> tuple[list[str], ScoreResult]:
        """Resolve a plain/red five mismatch using the authoritative han.

        The site's han says a five of the winner's hand is the other one (red for plain
        or the reverse): the swap that fits, ranked by the reveal frames, is bound as
        the winner's hand and the hand solved again, so the draws that brought the five
        follow. A swap the draws cannot deliver is left as read.
        """
        score_of = self._scorer(melds)
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
            self.problems.append(
                (
                    f"the winner's {x} as {alt} would score the site's {result.han}"
                    f"/{result.fu}, but "
                )
                + (
                    "no draw can bring it"
                    if not sol2.ok
                    else (
                        "the video reads it otherwise ("
                        f"{sol2.objective - self.sol.objective:.0f} costlier)"
                    )
                )
                + ": kept as read"
            )
            return concealed, sc
        self.sol = sol2
        others = sorted({b for _, b, _, _ in swaps if b != alt})
        self.problems.append(
            (
                f"the winner's {x} is {alt}: the site's {result.han}/{result.fu} "
                "needs it (a red five is one han)"
            )
            + (f"; also possible: {', '.join(others)}" if others else "")
        )
        return conc2, score_of(conc2, win_tile, self.dora)

    def check_draw(self) -> None:
        """Validate the site's tenpai seats against final reconstructed hands.

        At an exhaustive draw the site's tenpai seats must be tenpai with their final
        states; a riichi win reveals ura indicators, without which the han cannot be
        checked.
        """
        sol, result = self.sol, self.result
        if sol.ok and result.outcome == "draw":
            for s in rules.SEATS:
                hand = sol.hands.get(
                    (s, len(self.model.turns[s]) - 1), sol.haipai.get(s, [])
                )
                tp = is_tenpai(list(hand))
                want = site_seat_name(s, self.entry) in result.tenpai
                if tp != want:
                    self.items.append(
                        {
                            "kind": "result",
                            "seat": s,
                            "t": self.turns[-1].t if self.turns else self.t1,
                            "text": (
                                f"{s} is {('tenpai' if tp else 'noten')} in the "
                                "reconstruction, site says "
                                f"{('tenpai' if want else 'noten')}"
                            ),
                            "tiles": list(hand),
                        }
                    )
        if (
            result.outcome in ("ron", "tsumo")
            and self.winner in self.site_riichi
            and not self.facts.get("ura")
        ):
            self.items.append(
                {
                    "kind": "ura",
                    "seat": self.winner,
                    "t": self.turns[-1].t if self.turns else self.t1,
                    "text": "riichi win: ura indicator(s) not known; add an ura fact",
                }
            )

    # ---- 9. review and output
    # ----------------------------------------------------------------------------------

    def output(self) -> dict:
        """Build the decode artifact with evidence and unresolved alternatives.

        Build the decode artifact, preserving evidence, alternatives, and unresolved
        questions.
        """
        sol, model = self.sol, self.model
        lost_keys = {
            k
            for k in (turn_key(model, f, 3.0) for f in self.facts.get("lost", []))
            if k is not None
        }
        out_turns = []
        output_items = list(self.items)
        replacements = {
            (r["seat"], r["slot_id"]): r
            for r in pond_evidence.replacement_requests(
                self.turns, self.entry, self.facts, self.t0, self.t1
            )
        }
        for t in self.turns:
            st = next((x for x in model.turns[t.seat] if x.t_discard == t.t), None)
            j = st.j if st else None
            draw = sol.draws.get((t.seat, j)) if j is not None else None
            chosen = (
                (sol.discards.get((t.seat, j)) or self.repaired.get((t.seat, j)))
                if j is not None
                else None
            )
            draw2 = sol.draws2.get((t.seat, j)) if j is not None else None
            replacement = (
                replacements.get((t.seat, t.slot.id)) if t.slot is not None else None
            )
            receipt = t.slot.replacement_acquisition if t.slot is not None else None
            discard_margin = sol.discard_margins.get((t.seat, j))
            replacement_certified = bool(
                replacement
                and replacement["acquired"]
                and receipt
                and t.slot is not None
                and (chosen or t.slot.tile) in receipt["supported_tiles"]
                and sol.ok
                and discard_margin is not None
                and not math.isnan(discard_margin)
                and not low_margin(discard_margin)
            )
            if (
                t.own_call is not None
                and id(t.own_call) in self.unknown_kans
                and (t.seat, j) in sol.kans
            ):
                self.problems.append(
                    f"the ankan of {t.seat} at {t.t:.0f}s: tile "
                    f"{sol.kans[t.seat, j]} chosen by the rules, not seen"
                )
            if chosen is not None and t.slot is not None and j is not None:
                self.problems.append(
                    f"{t.seat}'s discard at {t.t:.0f}s read {t.slot.tile}: the "
                    f"rules need {chosen}"
                )
                question = changed_discard(
                    (t.seat, j), t.t, t.slot.tile, chosen, self.facts.get("discard", [])
                )
                if (
                    question
                    and not replacement_certified
                    and not any(
                        i.get("kind") == "discard"
                        and i.get("seat") == t.seat
                        and i.get("j") == j
                        for i in output_items
                    )
                ):
                    output_items.append(question)
            if (
                replacement
                and not replacement_certified
                and not any(
                    i.get("kind") == "discard"
                    and i.get("seat") == t.seat
                    and i.get("j") == j
                    for i in output_items
                )
            ):
                question = pond_evidence.replacement_question(
                    replacement, chosen=chosen
                )
                question["j"] = j
                output_items.append(question)
            margin = sol.margins.get((t.seat, j)) if j is not None else None
            out_turns.append(
                {
                    "i": t.i,
                    "seat": t.seat,
                    "j": j,
                    "kind": t.kind,
                    "t": t.t,
                    "t_prev": out_turns[-1]["t"] if out_turns else self.t0,
                    "riichi": t.riichi,
                    "hand_before": sol.hands.get((t.seat, j - 1))
                    if j is not None
                    else None,
                    "hand_after": sol.hands.get((t.seat, j)) if j is not None else None,
                    "draw2": draw2,
                    "discard_box": [round(float(v), 1) for v in t.slot.xyxy]
                    if (t.slot and t.slot.xyxy)
                    else None,
                    "discard_pos": [t.slot.row, t.slot.index] if t.slot else None,
                    "discard": chosen or (t.slot.tile if t.slot else None),
                    "discard_conf": round(t.slot.conf, 3) if t.slot else None,
                    "discard_id": t.slot.id if t.slot else None,
                    "virtual": t.virtual,
                    "draw": draw,
                    "pond_tracking": (
                        {
                            "pending_replacement": t.slot.to_dict().get(
                                "pending_replacement"
                            ),
                            "acquisition": receipt,
                        }
                        if replacement and t.slot is not None
                        else None
                    ),
                    "margin": margin
                    if margin is not None and not math.isnan(margin)
                    else None,
                    "tsumogiri": (draw2 or draw) is not None
                    and t.slot is not None
                    and (draw2 or draw) == (chosen or t.slot.tile),
                    "call": t.call.to_dict() if t.call else None,
                    "own_call": t.own_call.to_dict() if t.own_call else None,
                }
            )
        confidence = confidence_rows(
            model,
            sol,
            context=ReviewContext(
                turns=self.turns,
                calls=self.live_calls,
                inds=self.inds,
                entry=self.entry,
                facts=self.facts,
                lost_keys=lost_keys,
                t0=self.t0,
            ),
        )
        output_items.extend(uncertain_discards(confidence, output_items))
        pending = uncertain_tiles(confidence, output_items)
        return {
            "decoder_version": DECODER_VERSION,
            "hand": self.entry["hand"],
            "game": self.entry["game"],
            "kyoku": self.entry["kyoku"],
            "honba": self.entry["honba"],
            "play_window": [self.t0, self.t1],
            "t_last": self.turns[-1].t if self.turns else self.t1,
            "dealer": self.dealer,
            "turns": out_turns,
            "haipai": sol.haipai,
            "haipai_margin": sol.haipai_margins,
            "draws": {f"{s}:{j}": v for (s, j), v in sol.draws.items()},
            "dora": self.dora,
            "ura": self.ura,
            "indicators": self.inds,
            "riichi": sorted(self.site_riichi),
            "result": {
                "outcome": self.result.outcome,
                "winner": self.winner,
                "loser": self.loser,
                "han": self.result.han,
                "fu": self.result.fu,
                "site": list(self.site_han_fu),
                "site_wrong": bool(self.facts.get("site_score")),
                "deltas": self.result.deltas,
                "riichi": self.result.riichi,
                "tenpai": [site_seat(s, self.entry) for s in self.result.tenpai],
            },
            "score": self.score,
            "solver": {
                "status": sol.status,
                "objective": round(sol.objective, 3),
                "optimal": sol.optimal,
                "low_margin": sum(1 for mg in sol.margins.values() if low_margin(mg)),
            },
            "calls": [c.to_dict() for c in self.calls],
            "problems": self.problems,
            "notes": ["No legal hand was found within the processing time limit."]
            if sol.status == "unsolved"
            else ["Some automatic checks reached their time limit."]
            if any(row["state"] == "unresolvable" for row in confidence)
            else [],
            "items": ranked(output_items + ([pending] if pending else [])),
            "confidence": confidence,
            "stats": {
                "discards": sum(len(v) for v in self.logs.values()),
                "turns": len(self.turns),
                "calls": len(self.calls),
                "hand_evidence": len(model.hand_ev),
                "draw_evidence": len(model.draw_ev),
            },
        }

    def run(self) -> dict:
        """Execute each decoder stage once in dependency order.

        Execute the stateful decoder once; stages depend on the preceding stages'
        results.
        """
        self.window()
        self.discards()
        self.anchor_calls()
        self.indicators()
        self.meld_facts()
        self.turn_sequence()
        self.end_of_hand()
        self.riichi()
        self.pond_replacements()
        self.solve()
        self.check_draw()
        self.unseen_tiles()
        return self.output()


def decode_hand(
    entry: dict,
    obs: dict[str, list[dict]],
    result: HandResult,
    facts: dict | None = None,
    *,
    options: DecodeOptions | None = None,
) -> dict:
    """Reconstruct one hand from observations and its mandatory site result.

    Facts are the normalized output of ``facts_for_hand``. Supplying ``options.models``
    as (detector, classifier, video path, calibration) enables targeted rereads;
    otherwise the returned artifact exposes uncertainty from existing evidence.
    This function does not write artifacts or alter human facts.
    ``options.confidence_timeout`` bounds alternative checking across all solves of
    this hand. A larger value allows a longer single attempt, never retries.
    """
    decoded = HandDecoder(
        entry, obs, result, facts or {}, options=options or DecodeOptions()
    ).run()
    return review_artifact(decoded, entry)


def _decode_input_binding(
    work: Path, hand: int, *, evidence_policy: EvidencePolicy | dict | None = None
) -> dict | None:

    policy = (
        retention.load_policy()
        if evidence_policy is None
        else retention.resolve_policy(evidence_policy)
    )
    observation = work / "obs" / f"{hand:02d}.json"
    provenance = work / "obs" / "provenance" / f"{hand:02d}.json"
    if not observation.exists() or not provenance.exists():
        return None
    return {
        "observation_sha256": sha256_file(observation),
        "provenance_sha256": sha256_file(provenance),
        "dense_policy": policy.fingerprint_for("dense"),
    }


def _decode_context(entry: dict, result: HandResult, all_facts: list[dict]) -> dict:
    """Bind a reconstruction to its seat map, authoritative result and human inputs."""
    return {
        "entry": entry,
        "result": dataclasses.asdict(result),
        "facts": [
            fact
            for fact in all_facts
            if all(fact.get(key) == entry[key] for key in ("game", "kyoku", "honba"))
        ],
    }


def _cached_decodes(
    ddir: Path,
    selected: list[dict],
    games: list[Game],
    all_facts: list[dict],
    bindings: dict,
) -> dict[int, dict]:
    """Reuse only complete results bound to current observations, results and facts."""
    cached = {}
    for h in selected:
        p = ddir / f"{h['hand']:02d}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if (
                isinstance(d, dict)
                and d.get("decoder_version") == DECODER_VERSION
                and bindings[h["hand"]] is not None
                and d.get("decode_inputs") == bindings[h["hand"]]
                and d.get("decode_context")
                == _decode_context(
                    h, games[h["game"]].hands[h["site_index"]], all_facts
                )
            ):
                cached[h["hand"]] = d
    return cached


@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeRunOptions:
    """Hand selection, recording and evidence policy for a workspace decode."""

    force: bool = False
    only: set[int] | None = None
    video_path: Path | None = None
    cal: Calibration | None = None
    evidence_policy: EvidencePolicy | dict | None = None


def run_decode(
    work: Path,
    hands: list[dict],
    games: list[Game],
    *,
    log: Callable[[str], None] = LOGGER.info,
    options: DecodeRunOptions | None = None,
) -> list[dict]:
    """Decode selected observed hands and persist each result under ``work/decode``.

    Current-version results with matching observation/provenance content hashes
    and matching hand metadata, site result and facts are reused unless ``force``
    is true. Results are published atomically. Models are loaded only
    when a hand actually needs decoding and a video is available. Human labels
    and the default video resolve under VIDEO2TENHOU_HOME (the working directory
    by default), never inside an installed package.
    When dense rereads are enabled, sparse-read manifests must match the current
    source, models/runtime and calibration before any decoded hand is replaced.
    `evidence_policy` is an explicit policy or is read from detector metadata
    without loading weights. Sparse-policy provenance must match; the dense policy
    also binds decoded caches so dense-only changes cannot reuse an old result.
    An explicitly supplied video must exist. Model, reading and filesystem
    failures propagate; only reconstruction without a video skips dense reads.
    """
    options = options or DecodeRunOptions()
    force, only, video_path, cal, evidence_policy = (
        options.force,
        options.only,
        options.video_path,
        options.cal,
        options.evidence_policy,
    )
    if video_path is not None and not video_path.is_file():
        raise FileNotFoundError(video_path)
    policy = (
        retention.load_policy()
        if evidence_policy is None
        else retention.resolve_policy(evidence_policy)
    )
    ddir = work / "decode"
    ddir.mkdir(parents=True, exist_ok=True)
    all_facts = load_facts(DATA_DIR / "labels" / work.name)
    models = None
    selected = [h for h in hands if only is None or h["hand"] in only]
    bindings = {
        h["hand"]: _decode_input_binding(work, h["hand"], evidence_policy=policy)
        for h in selected
    }
    cached = (
        {} if force else _cached_decodes(ddir, selected, games, all_facts, bindings)
    )
    pending = [
        h
        for h in selected
        if h["hand"] not in cached and (work / "obs" / f"{h['hand']:02d}.json").exists()
    ]
    if video_path is None:
        cand = DATA_DIR / "samples" / f"{work.name}.mp4"
        video_path = cand if cand.exists() else None
    if pending and video_path is not None:
        if retention.load_policy() != policy:
            msg = (
                "Recognition evidence policy changed before rebuilding. "
                "Choose Analyze recording to refresh evidence."
            )
            raise ValueError(msg)
        from video2tenhou.perception.classifier import Classifier  # noqa: PLC0415

        models = (
            detector.Detector(),
            Classifier(),
            video_path,
            cal or Calibration.load("pml", video_path),
        )
    if models is not None:
        actual_policy = models[0].evidence_policy
        if actual_policy != policy:
            msg = (
                "Recognition evidence policy changed before rebuilding. Choose "
                "Analyze recording to refresh evidence."
            )
            raise ValueError(msg)
        # Every selected hand must pass before any existing result is replaced.
        validate_read_cache(
            ReadContext(models[2], models[3], work, models[0], models[1]), pending
        )
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
        d = decode_hand(
            h,
            obs,
            result,
            facts_for_hand(all_facts, h),
            options=DecodeOptions(models=models, work_dir=work),
        )
        d["decode_inputs"] = bindings[h["hand"]]
        d["decode_context"] = _decode_context(h, result, all_facts)
        atomic_write_json(p, sanitize(d), indent=1)
        out.append(d)
        sc = d["score"]
        log(
            f"  hand {h['hand']:2d}: {d['stats']['turns']} turns, "
            f"{d['stats']['calls']} calls, dora {d['dora']}, solver "
            f"{d['solver']['status']} obj {d['solver']['objective']}, "
            f"low-margin draws {d['solver']['low_margin']}, score "
            f"{('ok' if sc and sc['match'] else sc['han'] if sc else '-')}/"
            f"{(sc['fu'] if sc else '-')} vs site {d['result']['han']}/"
            f"{d['result']['fu']}, problems {len(d['problems'])}"
        )
    return out
