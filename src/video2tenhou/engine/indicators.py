# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Track the dead-wall indicators and the kans they imply (DESIGN.md 4.8 "Kans").

The face-up indicators lie side by side in one row of the dead wall. The row can be
pushed as a whole, but a face-up tile never turns back: the count only grows, by one per
kan, and the order along the row does not change. So every view of a pond region is
aligned with the row it has shown so far — by order and identity, never by position —
and a view showing as many tiles as the row holds is a re-reading of each, in order,
whatever it reads. Readings that disagree about one tile are a misrecognition to
resolve, not a new tile.

Every indicator after the first was revealed by a kan. A kan the call anchor established
and a new indicator explain each other; an indicator no kan explains is a kan the
cameras missed, whose maker is the player who discards next. A kan with no indicator
stands: the dead wall is often cut off by the edge of the overhead.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.perception.tiles import CLASSES

from . import rules
from .hand import in_window
from .melds import KAN_TYPES, POSITION, Call
from .ponds import MISSING_COST, NOISE_COST, align_order, cosine, face_posterior
from .turns import call_distance

if TYPE_CHECKING:
    from .ponds import PondSlot

MIN_SHORT_ROW_VIEWS = 2
MAX_SHORT_INDICATOR_ROW = 4
ANKAN_FACE_COUNT = 2
KAN_DISCARD_MATCH_WINDOW = 25
MIN_VIEWS = 3  # a real indicator is seen in at least this many full views ...
# ... and in at least this share of the full views of its region after its first
# sighting
PERSIST = 0.5
# s: a camera kan and a new indicator this close in time explain each other
KAN_MATCH = 30.0
# alignment cost of a reading a tile width or more from where its tile was last seen
MOVED_COST = 0.3
# s: an indicator seen this soon after play starts is the dora indicator's row read
# again, not a kan's
START_GRACE = 3.0
KAN_LEAD = 8.0  # s before its indicator appears that a kan can have been declared
# s before an indicator appears that the search for its maker's discard starts, when
# the region's last view without it is unknown
UNKNOWN_BEFORE = 12.0
# s after the maker's previous discard before its next kan can happen
AFTER_OWN_DISCARD = 4.0
PAIR_WINDOW = 40.0  # s around a kan to look for an ankan's face-up pair
INFERRED_KAN_CONF = 0.5  # confidence of a kan inferred from its indicator
UNNAMED_KAN_CONF = 0.3  # ... whose tile no camera names


@dataclass(frozen=True)
class KanReconciliation:
    """The calls with the kans the cameras missed, and which kans an indicator shows.

    ``unplaced`` holds the new indicators no discard follows: their kans have no turn.
    """

    calls: list[Call]
    revealed: set[int]  # id() of every kan an observed indicator revealed
    unplaced: list[dict]


@dataclass
class _Track:
    """One face-up tile of the dead-wall row."""

    p: np.ndarray
    t_first: float
    t_last: float
    t_before: float | None  # the region's last full view before the tile was first seen
    box: list | None
    conf: float
    seen: int = 0  # full views showing it
    tops: Counter = field(default_factory=Counter)
    x: float | None = None  # centre of its latest reading along the row

    def add(self, s: dict, o: dict) -> None:
        self.p = self.p + np.asarray(s["p"]) * max(1, s.get("seen", 1))
        self.t_first, self.t_last = (
            min(self.t_first, o["t0"]),
            max(self.t_last, o["t1"]),
        )
        self.conf = max(self.conf, s["conf"])
        self.tops[s["tile"]] += 1
        box = s.get("xyxy")
        if box:
            self.x = (box[0] + box[2]) / 2
            if not self.box or box[3] - box[1] > self.box[3] - self.box[1]:
                self.box = list(box)  # the tallest reading shows most of the tile
        if not o.get("partial"):
            self.seen += 1


def _distance(tr: _Track, s: dict) -> float:
    """Cost of a reading far from where the tile was last seen (MOVED_COST at most)."""
    box = s.get("xyxy")
    if tr.x is None or not box:
        return 0.0
    return MOVED_COST * min(
        1.0, abs((box[0] + box[2]) / 2 - tr.x) / max(1.0, box[2] - box[0])
    )


def _align(row: list[_Track], seen: list[dict]) -> list[int | None]:
    """For each reading of a view (left to right), its tile's index in the row.

    None marks a new tile. A view showing as many tiles as the row has established
    (seen twice or more) re-reads each of them in order; otherwise an order-preserving
    alignment by identity (a pair costs 2·(1 - cosine)). Where identity cannot tell (a
    new tile reading like its neighbour), nearness to where each tile was last seen
    does: the row is seldom pushed between two views.
    """
    est = [k for k, t in enumerate(row) if t.seen >= MIN_SHORT_ROW_VIEWS]
    if est and len(seen) == len(est):
        return list(est)
    pair = np.zeros((len(row), len(seen)))
    if row:
        tracks = np.array([t.p for t in row], dtype=np.float64)
        tiles = np.array([s["p"] for s in seen], dtype=np.float64)
        moved = np.array([[_distance(t, s) for s in seen] for t in row])
        pair = 2.0 * (1.0 - cosine(tracks, tiles)) + moved
    where: list[int | None] = [None] * len(seen)
    # a new tile of the row costs what a noise reading costs in a pond, anywhere
    pairs = align_order(pair, missing=MISSING_COST, extra=NOISE_COST, tail=NOISE_COST)
    for i, j in pairs:
        if j is not None:
            where[j] = i
    return where


def _region_row(obs: list[dict]) -> tuple[list[_Track], list[float]]:
    """Return the dead-wall row a region shows and the starts of its full views."""
    row: list[_Track] = []
    views: list[float] = []
    prev_full: float | None = None
    for o in sorted(obs, key=lambda o: o["t0"]):
        if not o.get("partial"):
            views.append(o["t0"])
        seen = sorted(
            (s for s in o["indicators"] if s.get("p")),
            key=lambda s: (s.get("xyxy") or [0])[0],
        )
        if seen:
            where = _align(row, seen)
            # new tiles go into the row at the place the alignment gives them (after the
            # matched tile before them)
            for j, s in enumerate(seen):
                position = where[j]
                if position is None:
                    before = max((w for w in where[:j] if w is not None), default=-1)
                    row.insert(
                        before + 1,
                        _Track(
                            np.zeros(len(CLASSES)),
                            o["t0"],
                            o["t1"],
                            prev_full,
                            None,
                            0.0,
                        ),
                    )
                    where = [
                        w + 1 if (w is not None and w > before) else w for w in where
                    ]
                    position = before + 1
                    where[j] = position
                row[position].add(s, o)
        if not o.get("partial"):
            prev_full = o["t1"]
    return row, views


def indicator_row(
    pond_obs: dict[str, list[dict]],
    t0: float,
    t1: float,
    last_discard: float | None = None,
    t_after: float = 0.0,
) -> list[dict]:
    """Track the dora indicators of a hand in the order they were revealed.

    The first is the dora indicator, every later one was revealed by a kan and carries
    its time. A tile is real when it persists (MIN_VIEWS full views, and PERSIST of the
    full views of its region after its first sighting). The dead wall lies in one pond
    region: the one whose row is seen best. A tile first seen after the last discard was
    not revealed by a kan (nothing follows it), except within `t_after` s of it: a tsumo
    winner may have kanned after its last discard and won on the rinshan draw.
    """
    rows = {}
    for region, obs in pond_obs.items():
        row, views = _region_row(in_window(obs, t0, t1))
        real = []
        for tr in row:
            after = [t for t in views if t >= tr.t_first]
            persistence = tr.seen / len(after) if after else 0.0
            if (
                after
                and persistence >= PERSIST
                and (
                    tr.seen >= MIN_VIEWS
                    or (
                        tr.seen >= MIN_SHORT_ROW_VIEWS
                        and len(after) <= MAX_SHORT_INDICATOR_ROW
                    )
                )
            ):
                real.append((tr, round(persistence, 2)))
        if real:
            rows[region] = real
    if not rows:
        return []
    region = max(rows, key=lambda r: max(tr.seen for tr, _ in rows[r]))
    out: list[dict] = []
    for tr, persistence in sorted(rows[region], key=lambda x: x[0].t_first):
        p = face_posterior(tr.p)
        tile = CLASSES[int(np.argmax(p))]
        p = p / max(p.sum(), 1e-9)
        # the readings disagree about this tile: a misrecognition, with the other
        # reading as the alternative
        others = [
            (t, n)
            for t, n in tr.tops.most_common()
            if t != tile and t not in ("X", "none")
        ]
        out.append(
            {
                "tile": tile,
                "seen": tr.seen,
                "t_first": tr.t_first,
                "t_last": tr.t_last,
                "t_before": tr.t_before,
                "conf": tr.conf,
                "region": region,
                "box": tr.box,
                "p": p.tolist(),
                "persistence": persistence,
                "readings": dict(tr.tops),
                "alternative": others[0][0] if others else None,
            }
        )
    if last_discard is not None:
        out = [out[0]] + [v for v in out[1:] if v["t_first"] <= last_discard + t_after]
    return out


def pair_groups(meld_obs: list[dict], t: float) -> list[list[str]]:
    """Find identical face-up pairs alone in a meld group near t, most frequent first.

    An ankan shows its two middle tiles face up (the reader does not box the face-down
    ones).
    """
    kinds: Counter = Counter()
    tokens: dict[str, list[str]] = {}
    for o in meld_obs:
        if o["t1"] < t - PAIR_WINDOW or o["t0"] > t + PAIR_WINDOW:
            continue
        groups: dict[int, list[dict]] = defaultdict(list)
        for sl in o["slots"]:
            groups[sl["key"][0]].append(sl)
        for g in groups.values():
            if (
                len(g) == ANKAN_FACE_COUNT
                and rules.plain(g[0]["tile"]) == rules.plain(g[1]["tile"])
                and g[0]["tile"] not in ("X", "none")
            ):
                k = rules.plain(g[0]["tile"])
                kinds[k] += 1
                tokens.setdefault(k, [g[0]["tile"], g[1]["tile"]])
    return [tokens[k] for k, _ in kinds.most_common()]


def _claimed(sl: PondSlot, seat: str, calls: list[Call]) -> bool:
    return any(call_distance(c, sl, seat) is not None for c in calls)


def reconcile_kans(
    inds: list[dict],
    calls: list[Call],
    *,
    logs: dict[str, list[PondSlot]],
    obs: dict,
    entry: dict,
    t0: float,
    diagnostics: list[str],
    tsumo_winner: str | None = None,
) -> KanReconciliation:
    """Match indicator revelations to anchored or previously unseen kans.

    Every indicator after the first was revealed by a kan. A kan the call anchor
    established and an indicator within KAN_MATCH s explain each other (the indicator
    then times the kan more tightly); an indicator no kan explains is a kan the cameras
    missed, which is added from the pond logs and meld observations. A kan with no
    indicator stands: the dead wall is often cut off at the edge of the overhead.
    """
    kans = sorted(
        (c for c in calls if c.type in KAN_TYPES), key=lambda c: c.t_window[0]
    )
    revealed: set[int] = set()
    inferred: list[Call] = []
    unplaced: list[dict] = []
    for ind in inds[1:]:
        t = ind["t_first"]
        if t is None:
            continue  # named by the reviewer, never observed: no time to place a kan
        if t <= t0 + START_GRACE:
            diagnostics.append(
                f"indicator {ind['tile']} visible from the start of play: not a"
                " kan indicator"
            )
            continue
        kan = next(
            (
                c
                for c in kans
                if id(c) not in revealed
                and c.t_window[0] - KAN_MATCH <= t <= c.t_first + KAN_MATCH
            ),
            None,
        )
        if kan is not None:
            revealed.add(id(kan))
            if kan.t_window[0] <= t:
                hi = min(kan.t_first, t)
                kan.t_window = (max(kan.t_window[0], min(t - KAN_LEAD, hi)), hi)
            continue
        c = _kan_the_cameras_missed(
            ind,
            calls + inferred,
            logs=logs,
            meld_obs={
                seat: obs.get(f"meld:{corner}", [])
                for corner, seat in entry["corner_wind"].items()
            },
            t0=t0,
            diagnostics=diagnostics,
            tsumo_winner=tsumo_winner,
        )
        if c is None:
            unplaced.append(ind)
        else:
            inferred.append(c)
            revealed.add(id(c))
    return KanReconciliation(calls + inferred, revealed, unplaced)


def _kan_the_cameras_missed(
    ind: dict,
    calls: list[Call],
    *,
    logs: dict[str, list[PondSlot]],
    meld_obs: dict[str, list[dict]],
    t0: float,
    diagnostics: list[str],
    tsumo_winner: str | None,
) -> Call | None:
    """Infer the kan a new indicator reveals when no camera read it.

    The maker discards next: the first discard sighted after the indicator's region was
    last seen without it (the rinshan draw is kept or discarded, then the maker
    discards); a reviewer's kan-time fact names the maker outright. None when no
    discard follows the indicator.
    """
    t = ind["t_first"]
    maker = ind.get("kan_maker")
    after = ind.get("t_before")
    if after is None:
        after = t - UNKNOWN_BEFORE
    nxt = min(
        (
            (sl.t_first, seat)
            for seat, sls in logs.items()
            for sl in sls
            if sl.t_first > after and (maker is None or seat == maker)
        ),
        default=None,
    )
    if nxt is not None:
        t_next, seat = nxt
    elif tsumo_winner:
        # no discard follows: the winner drew the rinshan tile and won
        t_next, seat = None, tsumo_winner
    else:
        return None
    # the kan happened after the maker's previous discard (a full rotation earlier) and
    # before the indicator appeared
    t_prev = max(
        (
            sl.t_first
            for sl in logs[seat]
            if sl.t_first < (t_next if t_next is not None else t)
        ),
        default=t0,
    )
    window = (min(max(t - KAN_LEAD, t_prev + AFTER_OWN_DISCARD), t), t)
    # an ankan whose tile the solver chooses, unless the evidence below names the kan
    unnamed = Call(
        seat=seat,
        t_first=t,
        t_window=window,
        type="ankan",
        tiles=["?", "?", "X", "X"],
        conf=UNNAMED_KAN_CONF,
        seen=MIN_VIEWS,
        anchor="indicator",
    )
    # a daiminkan takes a removed discard of another seat that no call has taken already
    removed = [
        (sl, s2)
        for s2, sls in logs.items()
        for sl in sls
        if s2 != seat
        and sl.t_removed is not None
        and abs(sl.t_removed - t) < KAN_DISCARD_MATCH_WINDOW
        and not _claimed(sl, s2, calls)
    ]
    if removed:
        sl, s2 = min(removed, key=lambda x: abs(x[0].t_removed - t))
        tile = sl.tile
        src = rules.relative(seat, s2)
        diagnostics.append(
            f"kan by {seat} at {t:.0f}s inferred from the new indicator "
            f"{ind['tile']} (daiminkan on {s2}'s {tile})"
        )
        return replace(
            unnamed,
            type="kan",
            tiles=rules.kan_tiles(tile),
            called_pos=POSITION["kan"][src],
            source=src,
            called_tile=tile,
            conf=INFERRED_KAN_CONF,
        )
    pairs = pair_groups(meld_obs.get(seat, []), t)
    if pairs:
        diagnostics.append(
            f"ankan of {rules.plain(pairs[0][0])} by {seat} at {t:.0f}s "
            f"inferred from the new indicator {ind['tile']} and the pair in its"
            " meld camera"
        )
        return replace(unnamed, tiles=[*pairs[0], "X", "X"], conf=INFERRED_KAN_CONF)
    prior_pon = [
        c for c in calls if c.seat == seat and c.type == "pon" and c.t_first < t
    ]
    if prior_pon:
        pon = prior_pon[-1]
        tile = rules.plain(pon.tiles[0])
        diagnostics.append(
            f"kakan on the pon of {tile} by {seat} at {t:.0f}s inferred from "
            f"the new indicator {ind['tile']}"
        )
        return replace(
            unnamed,
            type="kakan",
            tiles=[*pon.tiles, tile],
            called_pos=pon.called_pos,
            source=pon.source,
            called_tile=pon.called_tile,
            conf=INFERRED_KAN_CONF,
        )
    diagnostics.append(
        f"a kan by {seat} at {t:.0f}s inferred from the new indicator "
        f"{ind['tile']}; its tile is left to the solver"
    )
    return unnamed
