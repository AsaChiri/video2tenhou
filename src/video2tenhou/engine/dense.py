# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Targeted dense video readings to resolve specific evidence gaps.

When unclear, look again more closely (DESIGN.md 4.8): targeted dense reads at 5 fps of
the place and the moment that decide an open question, before a human is asked.

- the discard a call took: the three other ponds over the seconds before the meld
  appeared;
- a turn the merge had to skip: that player's pond over the gap;
- an ambiguous draw (certification found a close competing reconstruction): its
  hand row from its previous discard to the next player's discard, with still runs
  providing views before the draw, just after it, or after the discard;
- an unconfirmed open-kan replacement without direct draw evidence: the same
  bounded hand window, even when earlier hand counts imply a confident guess;
- a riichi no turned tile was read for: the pond at the moment each of the seat's tiles
  was laid.

Video-reading entrypoints take a `DenseContext` built once per hand and cache reads
under work/<video>/dense/ (read.dense_reads).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.engine.solver import HandRole
from video2tenhou.read import ReadContext, dense_pond_reads, dense_reads

from . import rules
from .hand import corner_of
from .ponds import PondSlot, insert_slot, next_slot_id, tail_runs
from .solver import hand_evidence

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from video2tenhou.engine.solver import HandModel
    from video2tenhou.read import ReadModels

    from .turns import Skip, Turn

MAX_MISSING_TURN_WINDOW = 90
# s: a hand-model turn and a merged turn this close are the same discard
SAME_DISCARD = 0.6
MAX_RIICHI_READ_WINDOW = 30.0
SIDEWAYS_ASPECT = 1.15  # width over height of a pond box lying across the row
MIN_RIICHI_RUN = 3
MIN_SIDEWAYS_SHARE = 0.6  # of the frames a riichi tile is seen in, at least this wide
# a seat's declaring discard is this much wider than its median discard box
WIDE_FACTOR = 1.25
RIICHI_BEFORE = 2.0  # s before a discard's window that its pond is read for a riichi
RIICHI_AFTER = 4.0  # s after the discard's first sighting
DENSE_WINDOW = 1.0  # s before a dense read's first sighting that a tile can be laid
FRAME = 0.2  # s between 5 fps frames: a tile gone after its last frame left by then
GAP_MARGIN = 1.0  # s read on either side of a skipped turn's gap
READ_MARGIN = 0.5  # s read on either side of an uncertain draw's window
AFTER_OWN_DISCARD = 1.0  # s after the seat's previous discard its hand row is read
STILL_FRAMES = 3  # a still run: this many 5 fps frames (0.6 s) ...
# ... each reading like the run's first, but for a single frame off in this many boxes
STILL_FLICKER = 1
GAP = 0.5  # s: frames further apart than this are not consecutive
# s of hand row read before and after the discard of an uncertain draw
BEFORE_MAX, AFTER_MAX = 60.0, 20.0


@dataclass(frozen=True, kw_only=True)
class DenseContext:
    """Recording and hand boundaries shared by targeted evidence acquisitions."""

    entry: dict
    models: ReadModels | None = None
    work_dir: Path | None = None
    t0: float = 0.0
    t1: float = float("inf")
    diagnostics: list[str] = field(default_factory=list)

    def reader(self) -> ReadContext:
        """Require a complete recording context before acquiring new evidence."""
        if self.models is None or self.work_dir is None:
            raise ValueError(
                "Dense evidence acquisition requires models and a workspace directory"
            )
        det, clf, video, cal = self.models
        return ReadContext(video, cal, self.work_dir, det, clf)


def _slot(logs: dict[str, list[PondSlot]], b: dict) -> PondSlot:
    """Build a slot for a tile a dense read found (not yet in any log)."""
    return PondSlot(
        next_slot_id(logs),
        b["row"],
        b["col"],
        np.asarray(b["p"]),
        b["t_first"],
        (b["t_first"] - DENSE_WINDOW, b["t_first"]),
        b["t_last"],
        b["n"],
        0.0,
        xyxy=tuple(b["xyxy"]),
    )


def taken_discards(
    caller: str,
    lo: float,
    hi: float,
    logs: dict[str, list[PondSlot]],
    *,
    context: DenseContext,
) -> list[tuple[str, dict]]:
    """Find tiles laid in the three other ponds over [lo, hi] that vanished before hi.

    These are the candidates for the discard a call of `caller` took (a called tile
    rests in the pond for a second or two, often while the calm reads see nothing).
    Returns [(discarder, run)] with run = {t_first, t_last, n, row, col, xyxy, p, tile}.
    """
    others = [s for s in rules.SEATS if s != caller]
    corners = [corner_of(context.entry, s) for s in others]
    reads = dense_pond_reads(context.reader(), lo, hi, corners)
    out = []
    for s, corner in zip(others, corners, strict=False):
        out += [
            (s, run)
            for run in tail_runs(logs[s], reads[corner])
            if run["gone"] and run["tile"] not in ("X", "none")
        ]
    return out


def place_taken(seat: str, run: dict, logs: dict[str, list[PondSlot]]) -> PondSlot:
    """Insert a called tile a dense read found into its pond's log, by its time."""
    sl = _slot(logs, run)
    sl.t_removed = run["t_last"] + FRAME
    insert_slot(logs[seat], sl)
    return sl


def skipped_turns(
    skips: list[Skip], logs: dict[str, list[PondSlot]], *, context: DenseContext
) -> int:
    """Read each skipped seat's pond densely over the gap; return the discards added.

    A seat that skipped a turn probably discarded while no calm frame showed its pond:
    the first new tile a dense read finds over the gap is that discard.
    """
    reader = context.reader()
    added = 0
    for skip in skips:
        t_a = skip.after if skip.after is not None else context.t0
        t_b = skip.before if skip.before is not None else context.t1
        if t_b - t_a > MAX_MISSING_TURN_WINDOW or t_b <= t_a:
            continue
        region = f"pond:{corner_of(context.entry, skip.seat)}"
        reads = dense_reads(reader, t_a - GAP_MARGIN, t_b + GAP_MARGIN, [region])
        b = next(
            (
                b
                for b in tail_runs(logs[skip.seat], reads[region])
                if b["tile"] not in ("X", "none")
            ),
            None,
        )
        if b is None:
            continue
        sl = _slot(logs, b)
        if b["gone"]:
            sl.t_removed = b["t_last"] + FRAME
        insert_slot(logs[skip.seat], sl)
        context.diagnostics.append(
            f"{skip.seat}'s discard {b['tile']} at {b['t_first']:.1f}s found in a "
            f"dense read (turn {skip.i} was skipped)"
        )
        added += 1
    return added


def _row(frame: dict) -> list[dict]:
    """Return the hand row of one reading, left to right (no melds or slivers)."""
    return sorted(
        (b for b in frame["boxes"] if b["role"] == "tile"),
        key=lambda b: (b["xyxy"][0] + b["xyxy"][2]) / 2,
    )


def still_runs(frames: list[dict], region: str) -> list[dict]:
    """Find still views of a hand row by its reading, not by the motion test.

    Runs of at least STILL_FRAMES consecutive frames with the same count and the same
    tile in every position (a single frame may be off in STILL_FLICKER boxes when the
    next one reads as the run again). Each is an observation in the calm format (mean
    posterior per position), so hand_evidence takes it as a calm view.
    """
    rows = [_row(f) for f in frames]
    tops = [[int(np.argmax(b["p"])) for b in r] for r in rows]
    out: list[dict] = []

    def diff(a: int, b: int) -> int:
        return (
            99
            if len(rows[a]) != len(rows[b])
            else sum(1 for x, y in zip(tops[a], tops[b], strict=False) if x != y)
        )

    i = 0
    while i < len(frames):
        k = i + 1
        # a frame holds the run when it reads like its first, or differs in a box or so
        # that the next frame restores (a flicker); a change that stays is a change, and
        # a gap in the frames ends the run
        while (
            k < len(frames)
            and frames[k]["t"] - frames[k - 1]["t"] <= GAP
            and (
                diff(k, i) == 0
                or (
                    diff(k, i) <= STILL_FLICKER
                    and (k + 1 == len(frames) or diff(k + 1, i) == 0)
                )
            )
        ):
            k += 1
        if k - i >= STILL_FRAMES and rows[i]:
            n = len(rows[i])
            ps = [
                np.mean(
                    [np.asarray(rows[q][x]["p"], np.float64) for q in range(i, k)],
                    axis=0,
                )
                for x in range(n)
            ]
            out.append(
                {
                    "region": region,
                    "t0": frames[i]["t"],
                    "t1": frames[k - 1]["t"],
                    "n_readings": k - i,
                    "n_used": k - i,
                    "count": n,
                    "quality": float(np.mean([p.max() for p in ps])),
                    "partial": False,
                    "dense": True,
                    "slots": [
                        {"key": [0, x], "p": p.tolist()} for x, p in enumerate(ps)
                    ],
                }
            )
        i = k
    return out


def draws(
    low: list[tuple[str, int]],
    model: HandModel,
    turns: list[Turn],
    melds_before: dict[str, dict[int, int]],
    *,
    context: DenseContext,
) -> None:
    """Read each listed draw's hand row densely and add its still views to the model.

    The row is read at 5 fps from the seat's previous discard to the next player's
    discard, and every still run of it is a view of the hand (DESIGN.md 4.8 "Uncertain
    draws"): the hand before the draw, the moment after it, the hand after the discard.
    The draws of one seat whose stretches overlap are read in one window. Callers
    select acquisition candidates by feasible alternative cost gaps; certified
    confidence bounds independently determine what needs review.
    """
    reader = context.reader()
    spans: dict[str, list[tuple[float, float, int]]] = {}
    for s, j in low:
        st = next((x for x in model.turns[s] if x.j == j), None)
        if st is None or st.kind not in ("draw", "kan"):
            continue
        turn = next(
            (
                t
                for t in turns
                if t.seat == s and abs(t.t - st.t_discard) < SAME_DISCARD
            ),
            None,
        )
        if turn is None:
            continue
        own_prev = max(
            (t.t for t in turns if t.seat == s and t.t < turn.t - SAME_DISCARD),
            default=context.t0,
        )
        nxt = (
            turns[turn.i + 1].t if turn.i + 1 < len(turns) else st.t_discard + AFTER_MAX
        )
        spans.setdefault(s, []).append(
            (
                max(
                    own_prev + AFTER_OWN_DISCARD,
                    st.t_discard - BEFORE_MAX,
                    context.t0,
                ),
                min(nxt, st.t_discard + AFTER_MAX),
                j,
            )
        )
    n_new = 0
    for s, sp in spans.items():
        region = f"hand:{corner_of(context.entry, s)}"
        windows: list[list] = []
        for lo, hi, _ in sorted(sp):
            if windows and lo <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], hi)
            else:
                windows.append([lo, hi])
        runs: list[dict] = []
        for lo, hi in windows:
            reads = dense_reads(reader, lo - READ_MARGIN, hi + READ_MARGIN, [region])
            runs += still_runs(reads[region], region)
        hev, dev, _ = hand_evidence(
            s,
            model.turns[s],
            runs,
            melds_before[s],
            role=HandRole(
                dealer=s == model.dealer, wins_by_tsumo=s == model.tsumo_winner
            ),
        )
        model.hand_ev += hev
        model.draw_ev += dev
        n_new += sum(
            1
            for _, _, j in sp
            if any(d.j == j for d in dev)
            or any(
                e.j in (j - 1, j) and not e.after_draw and not e.subset and not e.hidden
                for e in hev
            )
        )
    if low:
        context.diagnostics.append(
            f"dense hand reads for {len(low)} uncertain draws: still views pin "
            f"{n_new} of them"
        )


def _sideways_run(reads: list[dict], box: Sequence[float]) -> tuple[int, int] | None:
    """Require a persistent sideways tile near the original pond position."""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    tw = box[2] - box[0]
    # a tile at rest does not turn: the box at this position must be wide in most
    # frames it is there, over at least three of them; a single wide box is a tile in
    # the player's fingers, not a declaration
    run = side = 0
    for r in reads:
        near = [
            b
            for b in r["boxes"]
            if b.get("role") == "tile"
            and abs((b["xyxy"][0] + b["xyxy"][2]) / 2 - cx) < tw
            and abs((b["xyxy"][1] + b["xyxy"][3]) / 2 - cy) < tw
        ]
        if not near:
            run = side = 0
            continue
        b = min(near, key=lambda b: abs((b["xyxy"][0] + b["xyxy"][2]) / 2 - cx))
        run += 1
        side += 1 if _aspect(b["xyxy"]) > SIDEWAYS_ASPECT else 0
        if run >= MIN_RIICHI_RUN and side >= MIN_SIDEWAYS_SHARE * run:
            return side, run
    return None


def turned_tile(seat: str, turns: list[Turn], *, context: DenseContext) -> Turn | None:
    """Find the declaring discard of a seat whose turned tile no calm reading flagged.

    A riichi tile lies across the row, so it is far wider than its neighbours: first the
    boxes already read, then, when the tile was called away before any calm frame (or
    the box was cut), a dense read of the pond around each discard.
    """
    mine = [(t, t.slot) for t in turns if t.seat == seat and t.slot is not None]
    if not mine:
        return None
    boxes = [(t, _aspect(slot.xyxy)) for t, slot in mine if slot.xyxy]
    if boxes:
        med = float(np.median([a for _, a in boxes]))
        wide = [(t, a) for t, a in boxes if a > WIDE_FACTOR * med]
        if len(wide) == 1:
            t, a = wide[0]
            context.diagnostics.append(
                f"riichi of {seat}: the discard at {t.t:.0f}s lies across the "
                f"row (aspect {a:.2f} against {med:.2f}): it is the declaration"
            )
            return t
    if context.models is None:
        return None
    reader = context.reader()
    corner = corner_of(context.entry, seat)
    for t, slot in mine:
        lo = max(context.t0, slot.t_window[0] - RIICHI_BEFORE)
        hi = t.t + RIICHI_AFTER
        if hi - lo > MAX_RIICHI_READ_WINDOW or not slot.xyxy:
            continue
        reads = dense_pond_reads(reader, lo, hi, [corner])[corner]
        match = _sideways_run(reads, slot.xyxy)
        if match is not None:
            side, run = match
            context.diagnostics.append(
                f"riichi of {seat}: its discard at {t.t:.0f}s lies across "
                f"the row in {side} of {run} dense frames: it is the "
                "declaration"
            )
            return t
    return None


def _aspect(box: Sequence[float]) -> float:
    """Width over height of a pond box: an upright tile about 0.78, a turned one 1.3."""
    x0, y0, x1, y1 = box
    return (x1 - x0) / max(1.0, y1 - y0)
