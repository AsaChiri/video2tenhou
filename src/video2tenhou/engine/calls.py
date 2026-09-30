# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Calls anchored on the discards they took (DESIGN.md 4.8 "call anchor").

The meld camera says *that* a seat laid a meld, and roughly what it looks like; the pond
says which discard it took; the hand says which of its tiles went with it. So a meld
event becomes a call only when

- a discard was taken — a removal in a calm pond log, or a tile a dense read sees laid
  and gone again in the
  seconds before the meld appeared: a chi, pon or daiminkan of that tile from that seat,
  whose tiles from the
  hand are a choice among the legal melds on the called tile (`melds.meld_options`),
  decided by the solver
  with the caller's hand;
- the caller's hand confirms it: a chi, pon or daiminkan leaves the resting hand three
  tiles shorter, so a
  hand that still holds as many tiles as the seat's known calls allow says the camera
  re-read a meld already
  counted (the insets regroup their tiles from view to view), and nothing is taken;
- nothing was taken but the camera shows a kan pattern — a pair of identical face-up
  tiles (an ankan's middle)
  or a pon grown by a tile of its kind: an ankan or a kakan. An ankan holds all four
  tiles of its kind, so a
  kind that a call or a pond shows elsewhere cannot be one: a pair seen only as a
  fragment is then no kan, and
  a kan the camera saw whole has its kind misread (the solver names it).

Anything else the camera shows is not a call.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.engine.dense import DenseContext
from video2tenhou.train.data import CLASS_INDEX, CLASSES

from . import dense, rules
from .hand import MELD_LOOK, in_window, melds_shown, seat_of
from .melds import Call, fragments, meld_options, track_melds
from .turns import removal_gap

if TYPE_CHECKING:
    from .ponds import PondSlot

# Seconds before a meld appears to scan densely for the discard it took.
DENSE_BEFORE = 25.0
PLAY_SLACK = 20.0  # s: the first discard may have been called away before any calm view


@dataclass
class Taken:
    """The discard a call took."""

    seat: str  # the discarder
    slot: PondSlot
    tile: str  # its pond reading: the called tile
    t: float  # when it left the pond


def meld_events(entry: dict, obs: dict, t0: float, t1: float) -> list[Call]:
    """Collect newly observed melds and persistent two-tile fragments.

    Every new group of every seat's meld camera: legal melds and fragments (two tiles of
    a meld with no legal third). None of them is a call yet.
    """
    out: list[Call] = []
    for corner in ("TL", "TR", "BL", "BR"):
        seat = seat_of(entry, corner)
        views = in_window(obs.get(f"meld:{corner}", []), t0, t1)
        seen = track_melds(seat, views, include_contradicted=True)
        out += seen
        for f in fragments(seat, views, seen):
            before = f.t_before if f.t_before is not None else f.t_first - 8.0
            out.append(
                Call(
                    seat,
                    f.t_first,
                    (before, f.t_first),
                    "fragment",
                    f.tiles,
                    None,
                    None,
                    None,
                    f.ps,
                    float(np.mean([p.max() for p in f.ps])) if f.ps else 0.0,
                    0,
                    f.seen,
                )
            )
    return out


class CallAnchor:
    """Turns meld events into calls, claiming each discard once."""

    def __init__(
        self,
        logs: dict[str, list[PondSlot]],
        obs: dict | None = None,
        *,
        context: DenseContext,
    ) -> None:
        """Bind pond logs and optional dense readers for single-use call anchoring."""
        entry, models, work_dir, t0, problems = (
            context.entry,
            context.models,
            context.work_dir,
            context.t0,
            context.problems,
        )
        self.logs, self.entry, self.models, self.work_dir, self.t0 = (
            logs,
            entry,
            models,
            work_dir,
            t0,
        )
        self.problems = problems
        self.obs = obs or {}
        self.claimed: set[int] = set()

    def anchor(
        self,
        events: list[Call],
        first_discard: float,
        last_discard: float | None,
        tsumo_winner: str | None,
    ) -> list[Call]:
        """Validate camera hypotheses against pond removals and the play interval.

        Targeted dense reads may add a called-away discard to the shared logs.
        Diagnostics explain rejected hypotheses; only established calls are returned.
        """
        calls: list[Call] = []
        kans: list[Call] = []
        last = last_discard if last_discard is not None else float("inf")
        for ev in sorted(events, key=lambda c: c.t_first):
            late = ev.t_window[0] > last + 2.0 and not (
                ev.seat == tsumo_winner and ev.type in ("kan", "ankan", "kakan")
            )
            if ev.t_first < first_discard - PLAY_SLACK or late:
                self.problems.append(
                    f"{ev.seat}'s meld camera shows {' '.join(ev.tiles)} at "
                    f"{ev.t_first:.0f}s, "
                    f"{('before the first' if not late else 'after the last')} "
                    "discard: not a call"
                )
                continue
            if ev.type == "kakan":
                ev.anchor = "kan"  # a pon grown by a tile of its kind: nothing taken
                calls.append(ev)
                continue
            if ev.type in ("chi", "pon", "kan", "fragment") and self.reread(ev, calls):
                continue
            taken = self._calm(ev) or self._dense(ev)
            if taken is not None:
                self.claimed.add(id(taken.slot))
                calls.append(self._on_discard(ev, taken))
            elif not ev.contradicted and self._kan_pattern(ev):
                kans.append(ev)
            else:
                self.problems.append(
                    f"{ev.seat}'s meld camera shows {' '.join(ev.tiles)} at "
                    f"{ev.t_first:.0f}s, but no discard was taken and it is no "
                    "kan: not a call"
                )
        # the kans once every call is known: the pons they grow and the tiles an ankan
        # cannot share
        for ev in kans:
            kan = self._self_kan(ev, calls)
            if kan is not None:
                calls.append(kan)
        return sorted(calls, key=lambda c: c.t_first)

    def reread(self, ev: Call, calls: list[Call]) -> bool:
        """Check whether the hand size identifies a repeated view of a known meld.

        The caller's hand holds as many tiles as its known calls allow: the camera
        re-read a meld already counted (a note).
        """
        shown = melds_shown(
            self.obs, self.entry, ev.seat, ev.t_first, ev.t_first + MELD_LOOK
        )
        known = sum(
            1
            for c in calls
            if c.seat == ev.seat and c.type != "kakan" and c.t_first <= ev.t_first + 2
        )
        if shown is None or shown > known:
            return False
        self.problems.append(
            f"{ev.seat}'s meld camera shows a {ev.type} of {' '.join(ev.tiles)}"
            f" at {ev.t_first:.0f}s, but the hand still holds the tiles of "
            f"{shown} meld(s), which the known calls explain: a re-read, not a "
            "call"
        )
        return True

    # -- was a discard taken?
    # --------------------------------------------------------------------------

    @staticmethod
    def _support(ev: Call, tile: str) -> float:
        """How strongly the event's boxes read this tile (its best box)."""
        ids = [CLASS_INDEX[rules.plain(tile)]] + (
            [CLASS_INDEX[tile]] if tile in rules.REDS else []
        )
        return max(
            (float(sum(np.asarray(p)[i] for i in ids)) for p in ev.p), default=0.0
        )

    @staticmethod
    def _suit_agrees(ev: Call, tile: str) -> bool:
        """Check the taken tile's suit against the observed meld.

        A chi or pon is of one suit, and the meld insets confuse numbers, not suits: the
        taken tile's suit is the one most of the event's boxes read (any, when it has
        none).
        """
        suits = Counter(
            rules.suit(CLASSES[int(np.argmax(p))])
            for p in ev.p
            if CLASSES[int(np.argmax(p))] not in ("X", "none")
        )
        return not suits or suits.most_common(1)[0][0] == rules.suit(tile)

    @staticmethod
    def _compatible_removal(ev: Call, seat: str, tile: str) -> bool:
        """Require positive composition/direction evidence for a contradicted event.

        Normal camera hypotheses retain their existing alternative-composition
        behavior. A later-disputed hypothesis cannot rescue itself by claiming
        an unrelated removal of merely the same suit.
        """
        if not ev.contradicted:
            return True
        source = rules.relative(ev.seat, seat)
        if ev.source is not None and source != ev.source:
            return False
        if ev.type == "chi" and source != "kamicha":
            return False
        return ev.called_tile is not None and rules.plain(tile) == rules.plain(
            ev.called_tile
        )

    def _calm(self, ev: Call) -> Taken | None:
        """Find an unused calm pond removal matching the meld's window and suit.

        A removal of another seat's calm pond log in the event's window, of the suit the
        camera reads, that no call has taken yet.
        """
        best = None
        for s, sls in self.logs.items():
            if s == ev.seat:
                continue
            for sl in sls:
                if (
                    sl.row == -1
                    or id(sl) in self.claimed
                    or not self._suit_agrees(ev, sl.tile)
                    or not self._compatible_removal(ev, s, sl.tile)
                ):
                    continue
                d = removal_gap(ev, sl)
                if d is None or sl.t_removed is None:
                    continue
                key = (d, -self._support(ev, sl.tile))
                if best is None or key < best[0]:
                    best = (key, Taken(s, sl, sl.tile, sl.t_removed))
        return best[1] if best else None

    def _dense(self, ev: Call) -> Taken | None:
        """Find a recently placed and removed tile through dense pond readings.

        A tile laid in another pond and gone again in the seconds before the meld
        appeared (5 fps reads).
        """
        if self.models is None:
            return None
        if self.work_dir is None:
            msg = "Dense call evidence requires a workspace directory"
            raise ValueError(msg)
        lo, hi = max(self.t0, ev.t_window[0] - DENSE_BEFORE), ev.t_first + 1.0
        runs = [
            (s, r)
            for s, r in dense.taken_discards(
                ev.seat,
                lo,
                hi,
                self.logs,
                context=DenseContext(
                    entry=self.entry, models=self.models, work_dir=self.work_dir
                ),
            )
            if (
                r["t_last"] <= ev.t_first + 1.0
                and self._suit_agrees(ev, r["tile"])
                and self._compatible_removal(ev, s, r["tile"])
            )
        ]
        if not runs:
            return None
        s, run = min(
            runs,
            key=lambda x: (
                abs(x[1]["t_last"] - ev.t_window[0])
                if x[1]["t_last"] < ev.t_window[0]
                else 0.0,
                -self._support(ev, x[1]["tile"]),
            ),
        )
        sl = dense.place_taken(s, run, self.logs)
        if sl.t_removed is None:
            msg = "A dense taken-discard reading must include its removal time"
            raise ValueError(msg)
        self.problems.append(
            f"{ev.seat}'s meld at {ev.t_first:.0f}s took {s}'s {run['tile']}, "
            f"seen in its pond from {run['t_first']:.1f}s to "
            f"{run['t_last']:.1f}s in a dense read"
        )
        return Taken(s, sl, sl.tile, sl.t_removed)

    # -- the call
    # --------------------------------------------------------------------------

    def _on_discard(self, ev: Call, taken: Taken) -> Call:
        """Build call candidates taking the observed discard.

        A chi, pon or daiminkan of the taken tile: its tiles from the hand are a choice
        (the solver's).
        """
        source = rules.relative(ev.seat, taken.seat)
        options = meld_options(
            taken.tile, source, [np.asarray(p) for p in ev.p], four=ev.type == "kan"
        )
        best = options[0]
        window = (min(ev.t_window[0], taken.t), ev.t_first)
        return Call(
            ev.seat,
            ev.t_first,
            window,
            best.type,
            best.tiles,
            best.called_pos,
            source,
            taken.tile,
            ev.p,
            ev.conf,
            ev.group,
            ev.seen,
            partial_only=ev.partial_only,
            absent=ev.absent,
            anchor="discard",
            options=options,
            contradicted=ev.contradicted,
        )

    @staticmethod
    def _kan_pattern(ev: Call) -> bool:
        if ev.type in ("ankan", "kan"):
            return True
        return ev.type == "fragment" and rules.plain(ev.tiles[0]) == rules.plain(
            ev.tiles[1]
        )

    def _self_kan(self, ev: Call, calls: list[Call]) -> Call | None:
        """Resolve a shown kan with no taken discard as kakan or ankan.

        Nothing taken and a kan shown: a kakan on the seat's pon of that kind, else an
        ankan — if its kind can still be all four.
        """
        known = [t for t in ev.tiles if t not in ("X", "?")]
        if not known:
            return Call(
                ev.seat,
                ev.t_first,
                ev.t_window,
                "ankan",
                ["?", "?", "X", "X"],
                None,
                None,
                None,
                ev.p,
                ev.conf,
                ev.group,
                ev.seen,
                anchor="kan",
            )
        kind = rules.plain(known[0])
        if any(
            c.seat == ev.seat
            and c.type in ("ankan", "kakan")
            and rules.plain(c.tiles[0]) == kind
            for c in calls
        ):
            return None  # the same kan, seen again
        pon = next(
            (
                c
                for c in calls
                if c.seat == ev.seat
                and c.type == "pon"
                and c.t_first < ev.t_first
                and rules.plain(c.tiles[0]) == kind
            ),
            None,
        )
        if pon is not None:
            self.problems.append(
                f"{ev.seat} added a {kind} to its pon at {ev.t_first:.0f}s "
                "(nothing taken from a pond): kakan"
            )
            return Call(
                ev.seat,
                ev.t_first,
                ev.t_window,
                "kakan",
                [*pon.tiles, kind],
                pon.called_pos,
                pon.source,
                pon.called_tile,
                ev.p,
                ev.conf,
                ev.group,
                ev.seen,
                anchor="kan",
            )
        elsewhere = self._shown_elsewhere(kind, calls)
        if elsewhere and ev.type == "fragment":
            self.problems.append(
                f"{ev.seat}'s meld camera shows {' '.join(ev.tiles)} at "
                f"{ev.t_first:.0f}s and no discard was taken, but {elsewhere} "
                f"holds a {kind}: an ankan needs all four, so it is no kan"
            )
            return None
        if elsewhere:
            self.problems.append(
                f"{ev.seat}'s meld camera shows an ankan at {ev.t_first:.0f}s "
                f"read {kind}, but {elsewhere} holds a {kind}: the kan's kind "
                "is misread and left to the solver"
            )
            kind = "?"
        else:
            self.problems.append(
                f"{ev.seat}'s meld camera shows {' '.join(ev.tiles)} at "
                f"{ev.t_first:.0f}s and no discard was taken: an ankan of "
                f"{kind}"
            )
        return Call(
            ev.seat,
            ev.t_first,
            ev.t_window,
            "ankan",
            [kind, kind, "X", "X"],
            None,
            None,
            None,
            ev.p,
            ev.conf,
            ev.group,
            ev.seen,
            anchor="kan",
        )

    def _shown_elsewhere(self, kind: str, calls: list[Call]) -> str | None:
        """Find visible copies that rule out a proposed concealed kan.

        Where a tile of `kind` is seen outside a would-be ankan: a call's called tile or
        a pon/kan of it, or a discard at rest in a pond.
        """
        for c in calls:
            if c.called_tile is not None and rules.plain(c.called_tile) == kind:
                return f"{c.seat}'s {c.type} at {c.t_first:.0f}s"
            if c.type in ("pon", "kan", "kakan", "ankan") and any(
                rules.plain(x) == kind for x in c.tiles
            ):
                return f"{c.seat}'s {c.type} at {c.t_first:.0f}s"
        for s, sls in self.logs.items():
            for sl in sls:
                if sl.confirmed and rules.plain(sl.tile) == kind:
                    return f"{s}'s discard at {sl.t_first:.0f}s"
        return None
