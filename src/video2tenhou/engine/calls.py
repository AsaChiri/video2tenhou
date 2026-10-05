# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Calls anchored on the discards they took (DESIGN.md 4.8 "call anchor").

The meld camera says *that* a seat laid a meld, and roughly what it looks like; the pond
says which discard it took; the hand says which of its tiles went with it. So a meld
event becomes a call only when

- a discard was taken — a removal in a calm pond log, or a tile a dense read sees laid
  and gone again in the seconds before the meld appeared: a chi, pon or daiminkan of
  that tile from that seat, whose tiles from the hand are a choice among the legal
  melds on the called tile (`melds.meld_options`), decided by the solver with the
  caller's hand;
- the caller's hand confirms it: a chi, pon or daiminkan leaves the resting hand three
  tiles shorter, so a hand that still holds as many tiles as the seat's known calls
  allow says the camera re-read a meld already counted (the insets regroup their tiles
  from view to view), and nothing is taken;
- nothing was taken but the camera shows a kan pattern — a pair of identical face-up
  tiles (an ankan's middle) or a pon grown by a tile of its kind: an ankan or a kakan.
  An ankan holds all four tiles of its kind, so a kind that a call or a pond shows
  elsewhere cannot be one: a pair seen only as a fragment is then no kan, and a kan the
  camera saw whole has its kind misread (the solver names it).

Anything else the camera shows is not a call.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.engine.dense import DenseContext
from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES

from . import dense, rules
from .hand import MELD_LOOK, in_window, melds_shown, seat_of
from .melds import Call, fragments, meld_options, read_views, track_melds
from .turns import CALL_TOL, removal_gap

if TYPE_CHECKING:
    from .dense import DenseContext
    from .ponds import PondSlot

# s before a meld appears to scan densely for the discard it took: as far back as a
# calm removal is matched
DENSE_BEFORE = CALL_TOL
DENSE_AFTER = 1.0  # s after the meld first appears that the dense scan still covers
PLAY_SLACK = 20.0  # s: the first discard may have been called away before any calm view
LATE_SLACK = 2.0  # s after the last discard that a call's window may still open
# s before a fragment's first sighting when the camera's previous full view is unknown
FRAGMENT_BEFORE = 8.0
# s after an event that a call of the same seat counts as already known
KNOWN_CALL_SLACK = 2.0


@dataclass
class Taken:
    """The discard a call took."""

    seat: str  # the discarder
    slot: PondSlot
    tile: str  # its pond reading: the called tile
    t: float  # when it left the pond


def meld_events(entry: dict, obs: dict, t0: float, t1: float) -> list[Call]:
    """Collect every new group of every seat's meld camera.

    Legal melds and fragments (two tiles of a meld with no legal third). None of them is
    a call yet.
    """
    out: list[Call] = []
    for corner in ("TL", "TR", "BL", "BR"):
        seat = seat_of(entry, corner)
        views = read_views(in_window(obs.get(f"meld:{corner}", []), t0, t1))
        seen = track_melds(seat, views)
        out += seen
        for f in fragments(seat, views, seen):
            before = (
                f.t_before if f.t_before is not None else f.t_first - FRAGMENT_BEFORE
            )
            out.append(
                Call(
                    seat=seat,
                    t_first=f.t_first,
                    t_window=(before, f.t_first),
                    type="fragment",
                    tiles=f.tiles,
                    p=f.ps,
                    conf=float(np.mean([p.max() for p in f.ps])) if f.ps else 0.0,
                    seen=f.seen,
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
        """Bind pond logs and the hand's dense-reading context."""
        self.logs = logs
        self.obs = obs or {}
        self.context = context
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
        diagnostics = self.context.diagnostics
        calls: list[Call] = []
        kans: list[Call] = []
        last = last_discard if last_discard is not None else float("inf")
        for ev in sorted(events, key=lambda c: c.t_first):
            late = ev.t_window[0] > last + LATE_SLACK and not (
                ev.seat == tsumo_winner and ev.type in ("kan", "ankan", "kakan")
            )
            if ev.t_first < first_discard - PLAY_SLACK or late:
                diagnostics.append(
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
            if ev.type in ("chi", "pon", "kan", "fragment"):
                shown = self._reread_melds(ev, calls)
                if shown is not None:
                    diagnostics.append(
                        f"{ev.seat}'s meld camera shows a {ev.type} of "
                        f"{' '.join(ev.tiles)} at {ev.t_first:.0f}s, but the hand "
                        f"still holds the tiles of {shown} meld(s), which the known "
                        "calls explain: a re-read, not a call"
                    )
                    continue
            taken = self._calm(ev) or self._dense(ev)
            if taken is not None:
                self.claimed.add(id(taken.slot))
                calls.append(self._on_discard(ev, taken))
            elif not ev.contradicted and self._kan_pattern(ev):
                kans.append(ev)
            else:
                diagnostics.append(
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
        """Check whether the hand says the camera re-read a meld the calls explain."""
        return self._reread_melds(ev, calls) is not None

    def _reread_melds(self, ev: Call, calls: list[Call]) -> int | None:
        """Count the melds the hand shows when its known calls explain them all."""
        shown = melds_shown(
            self.obs, self.context.entry, ev.seat, ev.t_first, ev.t_first + MELD_LOOK
        )
        known = sum(
            1
            for c in calls
            if c.seat == ev.seat
            and c.type != "kakan"
            and c.t_first <= ev.t_first + KNOWN_CALL_SLACK
        )
        return shown if shown is not None and shown <= known else None

    # -- was a discard taken?
    # --------------------------------------------------------------------------

    @staticmethod
    def _support(ev: Call, tile: str) -> float:
        """How strongly the event's boxes read this tile (its best box)."""
        ids = [CLASS_INDEX[rules.plain(tile)]] + (
            [CLASS_INDEX[tile]] if tile in rules.PLAIN_OF else []
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
        """Find an unclaimed removal of another seat's calm pond log in the window.

        It must be of the suit the camera reads.
        """
        best = None
        for s, sls in self.logs.items():
            if s == ev.seat:
                continue
            for sl in sls:
                if (
                    sl.virtual
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
        """Find a tile laid in another pond and gone again before the meld appeared.

        The ponds are read at 5 fps.
        """
        if self.context.models is None:
            return None
        lo = max(self.context.t0, ev.t_window[0] - DENSE_BEFORE)
        hi = ev.t_first + DENSE_AFTER
        runs = [
            (s, r)
            for s, r in dense.taken_discards(
                ev.seat, lo, hi, self.logs, context=self.context
            )
            if (
                r["t_last"] <= hi
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
            raise ValueError(
                "A dense taken-discard reading must include its removal time"
            )
        self.context.diagnostics.append(
            f"{ev.seat}'s meld at {ev.t_first:.0f}s took {s}'s {run['tile']}, "
            f"seen in its pond from {run['t_first']:.1f}s to "
            f"{run['t_last']:.1f}s in a dense read"
        )
        return Taken(s, sl, sl.tile, sl.t_removed)

    # -- the call
    # --------------------------------------------------------------------------

    @staticmethod
    def _on_discard(ev: Call, taken: Taken) -> Call:
        """Build the call on the taken tile; the solver picks its hand tiles."""
        source = rules.relative(ev.seat, taken.seat)
        options = meld_options(
            taken.tile, source, [np.asarray(p) for p in ev.p], four=ev.type == "kan"
        )
        best = options[0]
        return replace(
            ev,
            t_window=(min(ev.t_window[0], taken.t), ev.t_first),
            type=best.type,
            tiles=best.tiles,
            called_pos=best.called_pos,
            source=source,
            called_tile=taken.tile,
            anchor="discard",
            options=options,
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

        A kakan on the seat's pon of that kind, else an ankan — if its kind can still
        be all four.
        """
        # a kan is a new call: none of the event's camera bookkeeping carries over
        kan = replace(
            ev,
            type="ankan",
            tiles=["?", "?", "X", "X"],
            called_pos=None,
            source=None,
            called_tile=None,
            partial_only=False,
            unseen=0,
            absent=0,
            anchor="kan",
            contradicted=False,
        )
        known = [t for t in ev.tiles if t not in ("X", "?")]
        if not known:
            return kan
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
        diagnostics = self.context.diagnostics
        if pon is not None:
            diagnostics.append(
                f"{ev.seat} added a {kind} to its pon at {ev.t_first:.0f}s "
                "(nothing taken from a pond): kakan"
            )
            return replace(
                kan,
                type="kakan",
                tiles=[*pon.tiles, kind],
                called_pos=pon.called_pos,
                source=pon.source,
                called_tile=pon.called_tile,
            )
        elsewhere = self._shown_elsewhere(kind, calls)
        if elsewhere and ev.type == "fragment":
            diagnostics.append(
                f"{ev.seat}'s meld camera shows {' '.join(ev.tiles)} at "
                f"{ev.t_first:.0f}s and no discard was taken, but {elsewhere} "
                f"holds a {kind}: an ankan needs all four, so it is no kan"
            )
            return None
        if elsewhere:
            diagnostics.append(
                f"{ev.seat}'s meld camera shows an ankan at {ev.t_first:.0f}s "
                f"read {kind}, but {elsewhere} holds a {kind}: the kan's kind "
                "is misread and left to the solver"
            )
            kind = "?"
        else:
            diagnostics.append(
                f"{ev.seat}'s meld camera shows {' '.join(ev.tiles)} at "
                f"{ev.t_first:.0f}s and no discard was taken: an ankan of "
                f"{kind}"
            )
        return replace(kan, tiles=[kind, kind, "X", "X"])

    def _shown_elsewhere(self, kind: str, calls: list[Call]) -> str | None:
        """Find where a tile of `kind` is seen outside a would-be ankan.

        A call's called tile or a pon/kan of it, or a discard at rest in a pond.
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
