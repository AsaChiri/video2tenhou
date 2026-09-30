"""The dead-wall row: the dora and kan indicators, and the kans they imply (DESIGN.md 4.8 "Kans").

The face-up indicators lie side by side in one row of the dead wall. The row can be pushed as a whole, but
a face-up tile never turns back: the count only grows, by one per kan, and the order along the row does not
change. So every view of a pond region is aligned with the row it has shown so far — by order and identity,
never by position — and a view showing as many tiles as the row holds is a re-reading of each, in order,
whatever it reads. Readings that disagree about one tile are a misrecognition to resolve, not a new tile.

Every indicator after the first was revealed by a kan. A kan the call anchor established and a new
indicator explain each other; an indicator no kan explains is a kan the cameras missed, whose maker is the
player who discards next. A kan with no indicator stands: the dead wall is often cut off by the edge of the
overhead.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..train.data import CLASSES
from . import rules
from .hand import corner_of, in_window
from .melds import POSITION, Call
from .ponds import PondSlot

MIN_VIEWS = 3  # a real indicator is seen in at least this many full views ...
PERSIST = 0.5  # ... and in at least this share of the full views of its region after its first sighting
KAN_MATCH = (
    30.0  # s: a camera kan and a new indicator this close in time explain each other
)


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

    def tile(self) -> str:
        p = self.p.copy()
        for k in ("X", "none"):
            p[CLASSES.index(k)] = 0.0
        return CLASSES[int(np.argmax(p))]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def _distance(tr: _Track, s: dict) -> float:
    """A small cost for a reading far from where the tile was last seen (0.3 at a tile width or more)."""
    box = s.get("xyxy")
    if tr.x is None or not box:
        return 0.0
    return 0.3 * min(1.0, abs((box[0] + box[2]) / 2 - tr.x) / max(1.0, box[2] - box[0]))


def _align(row: list[_Track], seen: list[dict]) -> list[int | None]:
    """For each reading of a view (left to right), the index of its tile in the row, or None for a new tile.

    A view showing as many tiles as the row has established (seen twice or more) re-reads each of them in
    order; otherwise an order-preserving alignment by identity (a pair costs 2·(1 − cosine), a tile of the
    row not shown 1.0, a new tile 1.2). Where identity cannot tell (a new tile reading like its neighbour),
    nearness to where each tile was last seen does: the row is seldom pushed between two views.
    """
    est = [k for k, t in enumerate(row) if t.seen >= 2]
    if est and len(seen) == len(est):
        return est
    n, m = len(row), len(seen)
    dp = np.full((n + 1, m + 1), 1e9)
    back = np.zeros((n + 1, m + 1), np.int8)
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i, j] >= 1e9:
                continue
            if i < n and j < m:
                c = (
                    dp[i, j]
                    + 2.0 * (1.0 - _cos(row[i].p, np.asarray(seen[j]["p"])))
                    + _distance(row[i], seen[j])
                )
                if c < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1], back[i + 1, j + 1] = c, 1
            if i < n and dp[i, j] + 1.0 < dp[i + 1, j]:
                dp[i + 1, j], back[i + 1, j] = dp[i, j] + 1.0, 2
            if j < m and dp[i, j] + 1.2 < dp[i, j + 1]:
                dp[i, j + 1], back[i, j + 1] = dp[i, j] + 1.2, 3
    out: list[int | None] = [None] * m
    i, j = n, m
    while i > 0 or j > 0:
        b = back[i, j]
        if b == 1:
            out[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        else:
            j -= 1
    return out


def _region_row(obs: list[dict]) -> tuple[list[_Track], list[float]]:
    """The dead-wall row one pond region shows over the hand, and the start times of its full views."""
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
            # new tiles go into the row at the place the alignment gives them (after the matched tile before them)
            for j, s in enumerate(seen):
                if where[j] is None:
                    before = max(
                        (where[q] for q in range(j) if where[q] is not None), default=-1
                    )
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
                    where[j] = before + 1
                row[where[j]].add(s, o)
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
    """The dora indicators of a hand in the order they were revealed: the first is the dora indicator, every
    later one was revealed by a kan and carries its time. A tile is real when it persists (MIN_VIEWS full
    views, and PERSIST of the full views of its region after its first sighting). The dead wall lies in one
    pond region: the one whose row is seen best. A tile first seen after the last discard was not revealed by
    a kan (nothing follows it), except within `t_after` s of it: a tsumo winner may have kanned after its last
    discard and won on the rinshan draw.
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
                and (tr.seen >= MIN_VIEWS or (tr.seen >= 2 and len(after) <= 4))
            ):
                real.append((tr, round(persistence, 2)))
        if real:
            rows[region] = real
    if not rows:
        return []
    region = max(rows, key=lambda r: max(tr.seen for tr, _ in rows[r]))
    out = []
    for tr, persistence in sorted(rows[region], key=lambda x: x[0].t_first):
        tile = tr.tile()
        p = tr.p.copy()
        for k in ("X", "none"):
            p[CLASSES.index(k)] = 0.0
        p = p / max(p.sum(), 1e-9)
        # the readings disagree about this tile: a misrecognition, with the other reading as the alternative
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


def pair_groups(meld_obs: list[dict], t: float, tol: float = 40.0) -> list[list[str]]:
    """Two identical face-up tiles alone in a group of a seat's meld camera near t: an ankan shows its two
    middle tiles face up (the reader does not box the face-down ones). Most frequent first.
    """
    kinds: Counter = Counter()
    tokens: dict[str, list[str]] = {}
    for o in meld_obs:
        if o["t1"] < t - tol or o["t0"] > t + tol:
            continue
        groups: dict[int, list[dict]] = defaultdict(list)
        for sl in o["slots"]:
            groups[sl["key"][0]].append(sl)
        for g in groups.values():
            if (
                len(g) == 2
                and rules.plain(g[0]["tile"]) == rules.plain(g[1]["tile"])
                and g[0]["tile"] not in ("X", "none")
            ):
                k = rules.plain(g[0]["tile"])
                kinds[k] += 1
                tokens.setdefault(k, [g[0]["tile"], g[1]["tile"]])
    return [tokens[k] for k, _ in kinds.most_common()]


def _claimed(sl: PondSlot, seat: str, calls: list[Call]) -> bool:
    from .turns import _call_dist

    return any(_call_dist(c, sl, seat) is not None for c in calls)


def reconcile_kans(
    inds: list[dict],
    logs: dict[str, list[PondSlot]],
    calls: list[Call],
    obs: dict,
    entry: dict,
    t0: float,
    problems: list[str],
    tsumo_winner: str | None = None,
) -> list[Call]:
    """Every indicator after the first was revealed by a kan. A kan the call anchor established and an indicator
    within KAN_MATCH s explain each other (the indicator then times the kan more tightly); an indicator no kan
    explains is a kan the cameras missed, which is added. A kan with no indicator stands: the dead wall is often
    cut off at the edge of the overhead. Returns the calls with the missed kans added.
    """
    kans = sorted(
        (c for c in calls if c.type in ("kan", "ankan", "kakan")),
        key=lambda c: c.t_window[0],
    )
    explained: set[int] = set()
    inferred: list[Call] = []
    for ind in inds[1:]:
        t = ind["t_first"]
        if t is None:
            continue  # named by the reviewer, never observed: no time to place a kan
        if t <= t0 + 3.0:
            # visible from the first moment of play: a second reading of the wall, not revealed during the hand
            problems.append(
                f"indicator {ind['tile']} visible from the start of play: not a kan indicator"
            )
            continue
        kan = next(
            (
                c
                for c in kans
                if id(c) not in explained
                and c.t_window[0] - KAN_MATCH <= t <= c.t_first + KAN_MATCH
            ),
            None,
        )
        if kan is not None:
            explained.add(id(kan))
            if kan.t_window[0] <= t:
                hi = min(kan.t_first, t)
                kan.t_window = (max(kan.t_window[0], min(t - 8.0, hi)), hi)
            continue
        c = _kan_the_cameras_missed(
            ind, logs, calls + inferred, obs, entry, t0, problems, tsumo_winner
        )
        if c is not None:
            c.anchor = "indicator"
            inferred.append(c)
    return calls + inferred


def _kan_the_cameras_missed(
    ind: dict,
    logs: dict[str, list[PondSlot]],
    calls: list[Call],
    obs: dict,
    entry: dict,
    t0: float,
    problems: list[str],
    tsumo_winner: str | None,
) -> Call | None:
    """The kan a new indicator reveals when no camera read it. The maker discards next: the first discard
    sighted after the indicator's region was last seen without it (the rinshan draw is kept or discarded,
    then the maker discards); a reviewer's kan-time fact names the maker outright.
    """
    t = ind["t_first"]
    maker = ind.get("kan_maker")
    after = ind.get("t_before") if ind.get("t_before") is not None else t - 12.0
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
        t_next, seat = (
            None,
            tsumo_winner,
        )  # no discard follows: the winner drew the rinshan tile and won
    else:
        problems.append(
            f"indicator {ind['tile']} appeared at {t:.0f}s but no discard follows: kan not placed"
        )
        return None
    # the kan happened after the maker's previous discard (a full rotation earlier) and before the indicator appeared
    t_prev = max(
        (
            sl.t_first
            for sl in logs[seat]
            if sl.t_first < (t_next if t_next is not None else t)
        ),
        default=t0,
    )
    window = (min(max(t - 8.0, t_prev + 4.0), t), t)
    # a daiminkan takes a removed discard of another seat that no call has taken already
    removed = [
        (sl, s2)
        for s2, sls in logs.items()
        for sl in sls
        if s2 != seat
        and sl.t_removed is not None
        and abs(sl.t_removed - t) < 25
        and not _claimed(sl, s2, calls)
    ]
    if removed:
        sl, s2 = min(removed, key=lambda x: abs(x[0].t_removed - t))
        tile = sl.tile
        src = rules.relative(seat, s2)
        problems.append(
            f"kan by {seat} at {t:.0f}s inferred from the new indicator {ind['tile']} (daiminkan on {s2}'s {tile})"
        )
        pos = POSITION["kan"][src]
        return Call(
            seat, t, window, "kan", rules.kan_tiles(tile), pos, src, tile, [], 0.5, 0, 3
        )
    pairs = pair_groups(obs.get(f"meld:{corner_of(entry, seat)}", []), t)
    if pairs:
        problems.append(
            f"ankan of {rules.plain(pairs[0][0])} by {seat} at {t:.0f}s inferred from the new indicator "
            f"{ind['tile']} and the pair in its meld camera"
        )
        return Call(
            seat,
            t,
            window,
            "ankan",
            pairs[0] + ["X", "X"],
            None,
            None,
            None,
            [],
            0.5,
            0,
            3,
        )
    prior_pon = [
        c for c in calls if c.seat == seat and c.type == "pon" and c.t_first < t
    ]
    if prior_pon:
        pon = prior_pon[-1]
        tile = rules.plain(pon.tiles[0])
        problems.append(
            f"kakan on the pon of {tile} by {seat} at {t:.0f}s inferred from the new indicator {ind['tile']}"
        )
        return Call(
            seat,
            t,
            window,
            "kakan",
            pon.tiles + [tile],
            pon.called_pos,
            pon.source,
            pon.called_tile,
            [],
            0.5,
            0,
            3,
        )
    problems.append(
        f"a kan by {seat} at {t:.0f}s inferred from the new indicator {ind['tile']}; its tile is left to the solver"
    )
    return Call(
        seat, t, window, "ankan", ["?", "?", "X", "X"], None, None, None, [], 0.3, 0, 3
    )
