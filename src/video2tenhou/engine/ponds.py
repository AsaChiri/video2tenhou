"""Pond tracking: observations of one pond -> the discard log of its owner (DESIGN.md 4.8 `ponds.py`).

A pond is a stack read in reading order: a discard is laid at the end, and the only tile that can leave
is the last one, taken by a call. Tiles get nudged (a row straightened, a block pushed by an elbow), so
their positions move and the reader's rows move with them; their order does not. Each observation is
therefore flattened into one reading-order sequence and aligned with the stack as a whole:

- a matched tile is a new reading of its slot (a disagreeing reading is a misrecognition, not a tile);
- tiles after the last slot are new discards; a new slot needs a view at rest (a full view of at least
  two readings, or a second view), otherwise it stays tentative and the next full view without it drops it;
- an unmatched tile anywhere else is noise: nothing is inserted in the middle of a pond;
- only the last slot can leave, when two consecutive full views lack it while the slot before it is seen;
- a full view that lacks at least three slots and more than half of the stack is the table being cleared.

A slot's (row, index) is its position in the stack when it was laid, six per row: a called-away tile's
successor takes its position (the refill rule of section 1). The same observations give the hand's
physical window (`play_window`): the clearings between hands are where a pond's count drops by more than
a call can explain.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ..train.data import CLASSES

ROW = 6  # tiles per pond row
MIN_READINGS = (
    2  # a view at rest: a full view of this many readings confirms a new slot at once
)
REMOVE_AFTER = (
    2  # consecutive full views without the last slot before it counts as called away
)
CLEAR_MISSING = (
    3  # a full view lacking this many slots, and over half the stack, is the clearing
)
TWO_TURNS = 10.0  # s: a seat cannot discard twice in less (the turn has to come round)
RESET_DROP = 3  # within a hand a pond loses at most one tile at a time (a call)
CLEAR_JOIN = 40.0  # s: resets of several ponds this close together are one clearing
CONFIRM_SPAN = 30.0  # s: how far after a reset a calm view can still deny it


@dataclass
class PondSlot:
    """One persistent discard, including sightings before a call removes it from the pond."""

    id: int
    row: int
    index: int  # position in the row: the stack's position when the slot was laid
    p: np.ndarray  # accumulated posterior (sum over observations, weighted by readings)
    t_first: float  # start of the first observation showing it
    t_window: tuple[
        float, float
    ]  # (end of the previous full view, t_first): when the discard happened
    t_last: float  # end of the last observation showing it
    seen: int = 0  # observations showing it
    sideways: float = 0.0  # accumulated sideways fraction
    missing: int = 0  # consecutive full views not showing it
    t_removed: float | None = None  # called away: the first full view without it
    disagree: int = 0  # views whose reading disagreed with the slot's identity
    xyxy: tuple | None = None  # box in the upright pond region at first sighting
    confirmed: bool = True  # seen at rest; a tentative slot is not (yet) a discard
    t_absent: float | None = (
        None  # the first full view of the current run of views without it
    )
    rest_views: int = (
        0  # views that count towards confirming it (not single-frame partial ones)
    )
    pending_replacement: dict | None = (
        None  # conflicting terminal sighting, not an extra discard
    )
    replacement_acquisition: dict | None = (
        None  # receipt for dense evidence substituted once
    )

    @property
    def tile(self) -> str:
        """Best tile kind (never X / none: a pond slot is a face-up tile)."""
        p = self.p.copy()
        p[CLASSES.index("X")] = 0
        p[CLASSES.index("none")] = 0
        return CLASSES[int(np.argmax(p))]

    @property
    def conf(self) -> float:
        """Largest normalized posterior mass accumulated across this slot's sightings."""
        p = self.p / max(self.p.sum(), 1e-9)
        return float(p.max())

    @property
    def is_sideways(self) -> bool:
        """Whether most sightings show the tile turned, as evidence of riichi declaration."""
        return self.seen > 0 and self.sideways / self.seen >= 0.5

    def to_dict(self) -> dict:
        """Serialize normalized evidence and physical timing for decode/review artifacts."""
        p = self.p / max(self.p.sum(), 1e-9)
        result = {
            "id": self.id,
            "row": self.row,
            "index": self.index,
            "tile": self.tile,
            "conf": round(self.conf, 4),
            "t_first": self.t_first,
            "t_window": list(self.t_window),
            "t_last": self.t_last,
            "seen": self.seen,
            "sideways": self.is_sideways,
            "t_removed": self.t_removed,
            "disagree": self.disagree,
            "xyxy": [round(float(v), 1) for v in self.xyxy] if self.xyxy else None,
            "p": [round(float(v), 4) for v in p],
        }
        if self.pending_replacement is not None:
            result["pending_replacement"] = deepcopy(self.pending_replacement)
        if self.replacement_acquisition is not None:
            result["replacement_acquisition"] = deepcopy(self.replacement_acquisition)
        return result


def flatten(obs: dict) -> list[dict]:
    """The observation's tiles in reading order: row by row, left to right."""
    return sorted(obs["slots"], key=lambda s: (s["key"][0], s["key"][1]))


def _similarity(slot: PondSlot, seen: dict) -> float:
    pl = slot.p / max(slot.p.sum(), 1e-9)
    ps = np.asarray(seen["p"])
    return float(np.dot(pl, ps) / max(np.linalg.norm(pl) * np.linalg.norm(ps), 1e-9))


def align(
    stack: list[PondSlot], seen: list[dict]
) -> list[tuple[int | None, int | None]]:
    """Order-preserving alignment of the stack's slots with an observation's tiles.

    Costs: a pair 2·(1 − cosine of the posteriors), so two different identities (about 2) cost more than
    a missing slot plus an appended tile (1.1); a missing slot 1.0; an unmatched tile 0.1 after every
    slot (a new discard) and 1.2 anywhere else (noise). Returns (slot index | None, tile index | None).
    """
    n, m = len(stack), len(seen)
    inf = 1e9
    dp = np.full((n + 1, m + 1), inf)
    back = np.zeros((n + 1, m + 1), np.int8)
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i, j] >= inf:
                continue
            if i < n and j < m:
                c = dp[i, j] + 2.0 * (1.0 - _similarity(stack[i], seen[j]))
                if c < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = c
                    back[i + 1, j + 1] = 1
            if i < n and dp[i, j] + 1.0 < dp[i + 1, j]:
                dp[i + 1, j] = dp[i, j] + 1.0
                back[i + 1, j] = 2
            if j < m:
                c = dp[i, j] + (0.1 if i == n else 1.2)
                if c < dp[i, j + 1]:
                    dp[i, j + 1] = c
                    back[i, j + 1] = 3
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        b = back[i, j]
        if b == 1:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif b == 2:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def assign_positions(log: list[PondSlot]) -> None:
    """(row, index) of every slot of a log in laying order: its position in the stack when it was laid.
    A slot called away before a later one was laid no longer holds its position.
    """
    for k, s in enumerate(log):
        pos = sum(1 for x in log[:k] if x.t_removed is None or x.t_removed > s.t_first)
        s.row, s.index = divmod(pos, ROW)


class PondTracker:
    """Incrementally align one pond while retaining the history of called-away discards."""

    def __init__(self, region: str):
        self.region = region
        self.slots: list[PondSlot] = []  # in laying order, tentative ones included
        self.cleared_at: float | None = None
        self._next = 0
        self._prev_end: float | None = None
        self._last_views: dict[int, dict] = {}

    def visible(self) -> list[PondSlot]:
        """Return the current pond stack; removed tiles remain only in the historical log."""
        return [s for s in self.slots if s.t_removed is None]

    def update(self, obs: dict) -> None:
        """Consume the next chronological observation, confirming additions and removals."""
        if self.cleared_at is not None or obs["n_used"] == 0:
            return
        partial = bool(obs.get("partial"))
        at_rest = not partial and obs["n_used"] >= MIN_READINGS
        # how much this view counts towards confirming a slot: a full view at rest confirms it, a full view of
        # one reading or a partial run of still frames counts half, a single partial frame (the stillest
        # frame of a blind span, the one taken while tiles are being pushed) not at all
        rest = (
            2 if at_rest else (1 if not partial or obs["n_used"] >= MIN_READINGS else 0)
        )
        stack = self.visible()
        seen = flatten(obs)
        pairs = align(stack, seen)
        matched = {i: j for i, j in pairs if i is not None and j is not None}
        # tiles the alignment places after every slot are new discards; unmatched ones before that are noise
        appended, consumed = [], 0
        for i, j in pairs:
            if i is not None:
                consumed = i + 1
            elif consumed == len(stack):
                appended.append(j)
        absent = [s for k, s in enumerate(stack) if k not in matched]
        # the clearing is judged on the slots seen at rest: tentative ones (a push seen in passing) are noise
        confirmed = [s for s in stack if s.confirmed]
        gone = [s for s in absent if s.confirmed]
        if (
            at_rest
            and len(gone) >= CLEAR_MISSING
            and 2 * (len(confirmed) - len(gone)) < len(confirmed)
        ):
            self.cleared_at = obs["t0"]
            return
        last = self._last_confirmed(
            stack
        )  # before this view's sightings confirm anything new
        last_absent = last is not None and last in absent
        if (
            last_absent
            and last is stack[-1]
            and len(appended) >= 2
            and self._prev_end is not None
            and obs["t0"] - self._prev_end < TWO_TURNS
        ):
            # the last tile gone and two new ones since the previous view: a call and two more discards of
            # this seat cannot fit in that time, so the first "new" tile is the last one misread
            matched[len(stack) - 1] = appended.pop(0)
            absent.remove(last)
            last_absent = False
        for k, j in matched.items():
            self._see(stack[k], seen[j], obs, rest)
        # a new tile in the very view that first lacks the last slot is either its refill (a call) or the last
        # slot misread: it waits for a second view, which decides. After a view without the last slot the
        # new tile is simply the next discard.
        swap = last_absent and last.missing == 0
        for j in appended:
            candidate_id = self._next
            self._lay(
                seen[j], obs, refill=last_absent, rest=min(rest, 1) if swap else rest
            )
            if (
                swap
                and len(appended) == 1
                and last is stack[-1]
                and self._next != candidate_id
            ):
                last.pending_replacement = {
                    "candidate_id": candidate_id,
                    "stable_end": last.t_last,
                    "stable_view": deepcopy(self._last_views[last.id]),
                    "stack_size": len(stack),
                    "t0": obs["t0"],
                    "t1": obs["t1"],
                    "prefix_complete": all(k in matched for k in range(len(stack) - 1)),
                    "views": [self._replacement_view(seen[j], obs)],
                }
        for old in self.slots:
            pending = old.pending_replacement
            if pending is None:
                continue
            candidate_id = pending["candidate_id"]
            candidate_match = next(
                (j for k, j in matched.items() if stack[k].id == candidate_id), None
            )
            if candidate_match is not None:
                pending["views"].append(
                    self._replacement_view(seen[candidate_match], obs)
                )
                pending["t1"] = obs["t1"]
            elif not partial and any(stack[k] is old for k in matched):
                # The established tile returned and its tentative alternative
                # did not; this is the existing transient-misread resolution.
                old.pending_replacement = None
        if partial:
            return  # an absent tile may be hidden; a partial view narrows no window
        for s in absent:
            if not s.confirmed:
                self.slots.remove(
                    s
                )  # tentative, and a full view does not show it: it was noise
                continue
            if s.missing == 0:
                s.t_absent = obs["t0"]
            s.missing += 1
        if (
            last_absent
            and last.missing >= REMOVE_AFTER
            and self._removable(last, stack, matched)
        ):
            last.t_removed = last.t_absent
        self._prev_end = obs["t1"]

    @staticmethod
    def _replacement_view(tile: dict, obs: dict) -> dict:
        return {
            "t0": obs["t0"],
            "t1": obs["t1"],
            "n_used": obs["n_used"],
            "partial": bool(obs.get("partial")),
            "tile": deepcopy(tile),
        }

    @staticmethod
    def _last_confirmed(stack: list[PondSlot]) -> PondSlot | None:
        return next((s for s in reversed(stack) if s.confirmed), None)

    def _removable(
        self, s: PondSlot, stack: list[PondSlot], matched: dict[int, int]
    ) -> bool:
        """Only the last discard can be called: no later slot was called away since it was laid, and the
        confirmed slot before it is seen (so the view is not simply hiding the end of the pond).
        """
        later = self.slots[self.slots.index(s) + 1 :]
        if any(x.t_removed is not None for x in later):
            return False
        before = [
            k for k, x in enumerate(stack) if x.confirmed and x is not s and x.id < s.id
        ]
        return not before or before[-1] in matched

    def _see(self, s: PondSlot, o: dict, obs: dict, rest: int) -> None:
        p = np.asarray(o["p"]) * max(1, o["seen"])
        if CLASSES[int(np.argmax(p))] != s.tile:
            s.disagree += 1
        s.p = s.p + p
        s.seen += 1
        s.sideways += o["sideways"]
        s.t_last = obs["t1"]
        s.missing = 0
        s.t_absent = None
        s.rest_views += rest
        s.confirmed = s.confirmed or s.rest_views >= 2
        self._last_views[s.id] = self._replacement_view(o, obs)

    def _lay(self, o: dict, obs: dict, *, refill: bool, rest: int) -> None:
        """A new discard at the end of the stack. Its row must be the one the stack puts it in (six per
        row): a tile seen in a later row before the earlier ones are full is not a discard of this pond.
        """
        pos = len(self.visible())
        rows = {pos // ROW} | ({(pos - 1) // ROW} if refill else set())
        if o["key"][0] not in rows:
            return
        row, index = divmod(pos, ROW)
        self.slots.append(
            PondSlot(
                self._next,
                row,
                index,
                np.asarray(o["p"]) * max(1, o["seen"]),
                obs["t0"],
                (
                    self._prev_end if self._prev_end is not None else obs["t0"],
                    obs["t0"],
                ),
                obs["t1"],
                1,
                o["sideways"],
                xyxy=tuple(o.get("xyxy") or ()) or None,
                confirmed=rest >= 2,
                rest_views=rest,
            )
        )
        self._next += 1
        self._last_views[self.slots[-1].id] = self._replacement_view(o, obs)

    def log(self) -> list[PondSlot]:
        """The discard log in laying order (confirmed slots only), with stack positions.

        A slot "called away" whose successor was laid in the very view it vanished from, and turned out to
        be the same tile after readings that disagreed, was never called: it was misread for a while, and
        the two slots are one tile.
        """
        out = [s for s in self.slots if s.confirmed]
        k = 0
        while k + 1 < len(out):
            r, s = out[k], out[k + 1]
            if (
                r.t_removed is not None
                and s.t_first == r.t_removed
                and s.disagree
                and s.tile == r.tile
            ):
                r.p, r.seen, r.sideways = (
                    r.p + s.p,
                    r.seen + s.seen,
                    r.sideways + s.sideways,
                )
                r.t_last, r.disagree, r.t_removed = (
                    max(r.t_last, s.t_last),
                    r.disagree + s.disagree,
                    s.t_removed,
                )
                r.pending_replacement = None
                out.pop(k + 1)
                continue
            k += 1
        assign_positions(out)
        return out


def track_pond(observations: list[dict]) -> list[PondSlot]:
    """Track a hand's observations chronologically and return confirmed discard history."""
    tr = PondTracker(observations[0]["region"] if observations else "")
    for o in sorted(observations, key=lambda o: o["t0"]):
        tr.update(o)
    return tr.log()


def insert_slot(log: list[PondSlot], sl: PondSlot) -> None:
    """Place a slot found afterwards (a dense read) into a discard log at its time; its position is the one the
    stack gives it.
    """
    k = sum(1 for x in log if x.t_first <= sl.t_first)
    log.insert(k, sl)
    assign_positions(log)


TAIL_MIN = 2  # dense readings a tile beyond the stack must be seen in


def tail_runs(log: list[PondSlot], readings: list[dict]) -> list[dict]:
    """Tiles a dense read (5 fps) sees beyond the end of a pond's stack: laid after every slot the calm reads
    know. Each reading is aligned with the stack as it stood then, in reading order, the way the tracker aligns
    a calm view (never by grid position: tiles are not flush). A slot counts from the start of its window, so
    a tile the calm reads saw later is matched, not new. Returns [{t_first, t_last, n, row, col, xyxy, p, tile,
    gone}] in time order; `gone`: it vanished while the rest of the pond stayed in view (a called tile).
    """
    found: list[dict] = []
    cur: dict | None = None

    def close(gone: bool) -> None:
        if cur is not None and cur["n"] >= TAIL_MIN:
            found.append({**cur, "gone": gone})

    for r in readings:
        t = r["t"]
        stack = [
            x
            for x in log
            if x.t_window[0] <= t and (x.t_removed is None or x.t_removed > t)
        ]
        seen = sorted(
            (b for b in r["boxes"] if b["role"] == "tile" and "row" in b),
            key=lambda b: (b["row"], b["col"]),
        )
        pairs = align(stack, seen)
        matched = sum(1 for i, j in pairs if i is not None and j is not None)
        last = max((k for k, (i, _) in enumerate(pairs) if i is not None), default=-1)
        tail = [seen[j] for i, j in pairs[last + 1 :] if j is not None]
        if not tail:
            if matched >= len(stack) - 1:  # the pond in view without it: it is gone
                close(True)
                cur = None
            continue  # an occluded view says nothing
        b = tail[0]
        if (
            cur is not None
            and float(np.dot(cur["p"] / cur["n"], np.asarray(b["p"]))) > 0.3
        ):
            cur["t_last"], cur["n"], cur["p"] = (
                t,
                cur["n"] + 1,
                cur["p"] + np.asarray(b["p"]),
            )
            continue
        close(True)  # another tile at the end: the previous one was taken
        cur = {
            "t_first": t,
            "t_last": t,
            "n": 1,
            "row": b["row"],
            "col": b["col"],
            "xyxy": b["xyxy"],
            "p": np.asarray(b["p"], float),
        }
    close(False)
    for run in found:
        run["p"] = (run["p"] / max(run["p"].sum(), 1e-9)).tolist()
        run["tile"] = CLASSES[int(np.argmax(run["p"]))]
    return found


# ---------------------------------------------------------------------------------------------------------
# the hand's physical window
# ---------------------------------------------------------------------------------------------------------


def _full_views(obs: list[dict], t_start: float, t_end: float) -> list[dict]:
    return sorted(
        (
            o
            for o in obs
            if not o.get("partial")
            and o.get("n_used", 1)
            and t_start <= o["t0"] <= t_end
        ),
        key=lambda o: o["t0"],
    )


def resets(
    obs: list[dict], t_start: float, t_end: float
) -> list[tuple[float, float, int]]:
    """Where one pond's count falls by more than a call explains: (end of the last calm view with the old
    tiles, start of the first without them, the drop). A drop to zero counts whatever its size (a short
    hand leaves small ponds). A dip that the next calm view undoes was an occlusion or a misread.
    """
    full = _full_views(obs, t_start, t_end)
    out = []
    for k in range(1, len(full)):
        a, b = full[k - 1], full[k]
        drop = a["count"] - b["count"]
        if drop < RESET_DROP and not (a["count"] > 0 and b["count"] == 0):
            continue
        nxt = full[k + 1] if k + 1 < len(full) else None
        if (
            nxt is not None
            and nxt["t0"] - b["t0"] <= CONFIRM_SPAN
            and nxt["count"] >= a["count"] - 1
        ):
            continue
        out.append((a["t1"], b["t0"], drop))
    return out


def _denies(obs: list[dict], a: float, b: float) -> bool:
    """Does this pond, seen at rest just before and just after [a, b], show that nothing was cleared?"""
    before = [
        o
        for o in obs
        if not o.get("partial")
        and o.get("n_used", 1)
        and a - CLEAR_JOIN <= o["t1"] <= b
    ]
    after = [
        o
        for o in obs
        if not o.get("partial")
        and o.get("n_used", 1)
        and a <= o["t0"] <= b + CLEAR_JOIN
    ]
    if not before or not after:
        return False
    x = max(before, key=lambda o: o["t1"])
    y = min(after, key=lambda o: o["t0"])
    return x["count"] >= RESET_DROP and y["count"] >= x["count"] - 1


def clearings(
    pond_obs: dict[str, list[dict]], t_start: float, t_end: float
) -> list[float]:
    """The table clearings in [t_start, t_end], each as the end of the last calm view that still held the
    old hand's tiles. Resets of two or more ponds within CLEAR_JOIN s are one clearing; a single pond's
    reset of RESET_DROP or more stands unless another pond, seen at rest on both sides, denies it.
    """
    events = sorted(
        (a, b, drop, region)
        for region, obs in pond_obs.items()
        for a, b, drop in resets(obs, t_start, t_end)
    )
    groups: list[list[tuple]] = []
    for ev in events:
        if groups and ev[0] <= max(e[1] for e in groups[-1]) + CLEAR_JOIN:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    out = []
    for g in groups:
        ponds = {e[3] for e in g}
        a, b = min(e[0] for e in g), max(e[1] for e in g)
        if len(ponds) < 2:
            if g[0][2] < RESET_DROP or any(
                _denies(obs, a, b) for r, obs in pond_obs.items() if r not in ponds
            ):
                continue
        out.append(max(e[0] for e in g))
    return out


def play_window(
    pond_obs: dict[str, list[dict]],
    t_start: float,
    t_end: float,
) -> tuple[float, float]:
    """Refine a hand window to the longest stretch between observed table clearings.

    Full tile readings can tighten boundaries estimated from the initial counts.
    Partial views cannot establish that a table has cleared.
    """
    cuts = clearings(pond_obs, t_start, t_end)
    bounds = [t_start] + cuts + [t_end]
    segments = list(zip(bounds, bounds[1:], strict=False))
    return max(segments, key=lambda s: s[1] - s[0])
